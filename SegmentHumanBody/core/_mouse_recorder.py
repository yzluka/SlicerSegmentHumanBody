"""Passive timestamped input event recorder.

_InputRecordFilter  — application-level Qt event filter; always returns False.
InputEventRecord    — namedtuple: (timestamp, xy_global, ras, event_type, payload).
MouseEventRecorder  — lifecycle wrapper: start / stop / clear / export / load.

Event types and their payload dicts
------------------------------------
'move'              payload: None
'press'             payload: {'segment_id', 'tool', 'view_name', 'axis',
                              'slice_idx', 'brush_radius_mm'}
'release'           payload: None
'action'            payload: {'name': str}
'session_start'     payload: {'volume': dict, 'segmentation': str,
                               'sample_rate_hz': int}
'session_stop'      payload: {'total_records': int}
'segment_created'   payload: {'segment_id': str, 'seg_name': str}
'segment_changed'   payload: {'from_id': str|None, 'to_id': str, 'seg_name': str}
'volume_changed'    payload: {'volume': str|None}

Volume metadata dict (stored in session_start['volume'])
---------------------------------------------------------
{'name', 'dimensions': [I,J,K], 'spacing': [dx,dy,dz],
 'origin': [ox,oy,oz], 'ijk_to_ras': [16 floats row-major]}

Replay eligibility
------------------
A loaded record can be replayed against a volume only when
``recorder.matches_volume(volume_node)`` returns (True, '').
The check compares dimensions (exact) and spacing (0.1% tolerance).

Singleton
---------
Use ``get_recorder()`` — the process-scoped instance survives module switches.
The widget sets ``recorder.context_fn`` in setup() and clears it in cleanup()
without stopping the recording.
"""

import collections
import datetime
import json
import logging

import numpy as np
import qt
import vtk
import slicer

log = logging.getLogger(__name__)

InputEventRecord = collections.namedtuple(
    'InputEventRecord',
    ['timestamp', 'xy_global', 'ras', 'event_type', 'payload'],
)

# ---------------------------------------------------------------------------
# Event-type constants
# ---------------------------------------------------------------------------
MOVE                   = 'move'
PRESS                  = 'press'
RELEASE                = 'release'
ACTION                 = 'action'
SESSION_START          = 'session_start'
SESSION_STOP           = 'session_stop'
SEGMENT_CREATED        = 'segment_created'
SEGMENT_CHANGED        = 'segment_changed'
SEGMENT_REMOVED        = 'segment_removed'
VOLUME_CHANGED         = 'volume_changed'
MODEL_FAMILY_CHANGED   = 'model_family_changed'
MODEL_VARIANT_CHANGED  = 'model_variant_changed'
MODEL_CONFIRMED        = 'model_confirmed'
WINDOW_LEVEL_APPLIED   = 'window_level_applied'
BRUSH_DIAMETER_CHANGED = 'brush_diameter_changed'
BRUSH_SPHERE_CHANGED   = 'brush_sphere_changed'
POINT_PLACED           = 'point_placed'


# ---------------------------------------------------------------------------
# Volume metadata helpers
# ---------------------------------------------------------------------------

def _volume_metadata(volume_node) -> dict | None:
    """Extract dimensions, spacing, origin and IJK→RAS matrix from *volume_node*."""
    if volume_node is None:
        return None
    dims = list(volume_node.GetImageData().GetDimensions())  # [I, J, K]
    spacing = list(volume_node.GetSpacing())                  # [dx, dy, dz]
    origin  = list(volume_node.GetOrigin())                   # [ox, oy, oz]
    mat = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(mat)
    ijk_to_ras = [mat.GetElement(r, c) for r in range(4) for c in range(4)]
    return {
        'name':       volume_node.GetName(),
        'dimensions': dims,
        'spacing':    spacing,
        'origin':     origin,
        'ijk_to_ras': ijk_to_ras,
    }


# ---------------------------------------------------------------------------
# Qt event filter
# ---------------------------------------------------------------------------

class _InputRecordFilter(qt.QObject):
    """Application-level Qt event filter — always returns False."""

    def __init__(self, on_mouse):
        super().__init__()
        self._on_mouse = on_mouse

    def eventFilter(self, obj, event):
        t = event.type()
        try:
            if t == qt.QEvent.MouseMove:
                pos = event.globalPos()
                self._on_mouse((pos.x(), pos.y()), MOVE)
            elif t == qt.QEvent.MouseButtonPress and event.button() == qt.Qt.LeftButton:
                pos = event.globalPos()
                self._on_mouse((pos.x(), pos.y()), PRESS)
            elif t == qt.QEvent.MouseButtonRelease and event.button() == qt.Qt.LeftButton:
                pos = event.globalPos()
                self._on_mouse((pos.x(), pos.y()), RELEASE)
        except Exception as exc:
            log.error('[InputRecordFilter] %s', exc)
        return False


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class MouseEventRecorder:
    """Records timestamped mouse events and named widget-action calls.

    Obtain via :func:`get_recorder` for the process-scoped singleton.

    Move events are sampled at *sample_rate_hz* (default 32 Hz ≈ 31 ms).
    All other events are recorded immediately.

    The widget sets ``recorder.context_fn = self._recorder_context`` in
    ``setup()`` and clears it to ``None`` in ``cleanup()``.  At press time
    the recorder calls ``context_fn()`` to capture the current segment,
    tool, view, and slice — if ``context_fn`` is None the press payload
    is an empty dict (module not loaded during that event).
    """

    MOVE                   = MOVE
    PRESS                  = PRESS
    RELEASE                = RELEASE
    ACTION                 = ACTION
    SESSION_START          = SESSION_START
    SESSION_STOP           = SESSION_STOP
    SEGMENT_CREATED        = SEGMENT_CREATED
    SEGMENT_CHANGED        = SEGMENT_CHANGED
    SEGMENT_REMOVED        = SEGMENT_REMOVED
    VOLUME_CHANGED         = VOLUME_CHANGED
    MODEL_FAMILY_CHANGED   = MODEL_FAMILY_CHANGED
    MODEL_VARIANT_CHANGED  = MODEL_VARIANT_CHANGED
    MODEL_CONFIRMED        = MODEL_CONFIRMED
    WINDOW_LEVEL_APPLIED   = WINDOW_LEVEL_APPLIED
    BRUSH_DIAMETER_CHANGED = BRUSH_DIAMETER_CHANGED
    BRUSH_SPHERE_CHANGED   = BRUSH_SPHERE_CHANGED
    POINT_PLACED           = POINT_PLACED

    def __init__(self, sample_rate_hz: int = 32):
        if sample_rate_hz <= 0:
            raise ValueError(f'sample_rate_hz must be positive, got {sample_rate_hz}')
        self._records: list                          = []
        self._filter                                 = None
        self._move_interval_ms: float                = 1000.0 / sample_rate_hz
        self._last_move_ts: datetime.datetime | None = None
        self._last_xy: tuple                         = (0, 0)
        self.context_fn                              = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    @property
    def is_active(self) -> bool:
        return self._filter is not None

    def start(self, volume_node=None, segmentation_name: str | None = None):
        """Install the event filter and record a session_start event.

        Parameters
        ----------
        volume_node : vtkMRMLScalarVolumeNode | None
            Current source volume; its metadata is stored for replay
            eligibility checks.
        segmentation_name : str | None
            Name of the active segmentation node (informational).
        """
        if self._filter is not None:
            return
        self._filter = _InputRecordFilter(on_mouse=self._on_mouse)
        slicer.app.installEventFilter(self._filter)
        hz = round(1000.0 / self._move_interval_ms)
        self._append(SESSION_START, (0, 0), None, {
            'volume':         _volume_metadata(volume_node),
            'segmentation':   segmentation_name,
            'sample_rate_hz': hz,
        })
        log.debug('[MouseEventRecorder] started — %d Hz move sampling', hz)

    def stop(self):
        """Record session_stop, remove the event filter.  Records are kept."""
        if self._filter is not None:
            slicer.app.removeEventFilter(self._filter)
            self._filter = None
        self._append(SESSION_STOP, self._last_xy, _cursor_ras(),
                     {'total_records': len(self._records)})
        log.debug('[MouseEventRecorder] stopped — %d records', len(self._records))

    def clear(self):
        """Discard all accumulated records without stopping."""
        self._records.clear()
        self._last_move_ts = None

    # ------------------------------------------------------------------ #
    # Named-event recording                                                #
    # ------------------------------------------------------------------ #

    def record_action(self, name: str):
        self._append(ACTION, self._last_xy, _cursor_ras(), {'name': name})

    def record_segment_created(self, segment_id: str, seg_name: str):
        self._append(SEGMENT_CREATED, self._last_xy, _cursor_ras(),
                     {'segment_id': segment_id, 'seg_name': seg_name})

    def record_segment_changed(self, from_id: str | None, to_id: str, seg_name: str):
        self._append(SEGMENT_CHANGED, self._last_xy, _cursor_ras(),
                     {'from_id': from_id, 'to_id': to_id, 'seg_name': seg_name})

    def record_volume_changed(self, volume_name: str | None):
        self._append(VOLUME_CHANGED, self._last_xy, _cursor_ras(),
                     {'volume': volume_name})

    def record_model_family_changed(self, family: str):
        self._append(MODEL_FAMILY_CHANGED, self._last_xy, _cursor_ras(),
                     {'family': family})

    def record_model_variant_changed(self, variant: str):
        self._append(MODEL_VARIANT_CHANGED, self._last_xy, _cursor_ras(),
                     {'variant': variant})

    def record_model_confirmed(self, family: str, variant: str):
        self._append(MODEL_CONFIRMED, self._last_xy, _cursor_ras(),
                     {'family': family, 'variant': variant})

    def record_window_level_applied(self, window: int, level: int):
        self._append(WINDOW_LEVEL_APPLIED, self._last_xy, _cursor_ras(),
                     {'window': window, 'level': level})

    def record_brush_diameter_changed(self, diameter_mm: float):
        self._append(BRUSH_DIAMETER_CHANGED, self._last_xy, _cursor_ras(),
                     {'diameter_mm': diameter_mm})

    def record_brush_sphere_changed(self, sphere: bool):
        self._append(BRUSH_SPHERE_CHANGED, self._last_xy, _cursor_ras(),
                     {'sphere': sphere})

    def record_segment_removed(self, segment_id: str, seg_name: str):
        self._append(SEGMENT_REMOVED, self._last_xy, _cursor_ras(),
                     {'segment_id': segment_id, 'seg_name': seg_name})

    def record_point_placed(self, segment_id: str, ras: list, is_negative: bool):
        self._append(POINT_PLACED, self._last_xy, _cursor_ras(),
                     {'segment_id': segment_id, 'ras': list(ras),
                      'is_negative': is_negative})

    # ------------------------------------------------------------------ #
    # File I/O                                                             #
    # ------------------------------------------------------------------ #

    def save_to_file(self, path: str):
        """Serialise all records to a JSON file at *path*."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.export_data(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> 'MouseEventRecorder':
        """Deserialise a JSON file produced by :meth:`save_to_file`.

        Returns a new (stopped) ``MouseEventRecorder`` whose ``_records``
        contain the loaded events.
        """
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                'Not a recording file — expected a JSON array at the top level. '
                '(Annotation log files have a different format; use Import Annotation Log instead.)'
            )
        recorder = cls()
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f'Record {i} is not a JSON object (got {type(item).__name__!r}). '
                    'File may be corrupt or saved by a different tool.'
                )
            ts      = datetime.datetime.fromisoformat(item['timestamp'])
            xy      = tuple(item['xy_global'])
            ras     = item.get('ras')
            evt     = item['event']
            payload = item.get('payload')
            recorder._records.append(InputEventRecord(ts, xy, ras, evt, payload))
        return recorder

    # ------------------------------------------------------------------ #
    # Replay eligibility                                                   #
    # ------------------------------------------------------------------ #

    def matches_volume(self, volume_node) -> tuple:
        """Check whether the session-start metadata matches *volume_node*.

        Returns
        -------
        (True, '')           — volume is compatible; replay is allowed.
        (False, reason: str) — incompatible; reason explains the mismatch.
        """
        starts = [r for r in self._records if r.event_type == SESSION_START]
        if not starts:
            return False, 'No session_start record found'
        meta = (starts[0].payload or {}).get('volume')
        if not meta:
            return False, 'No volume metadata in record'
        if volume_node is None:
            return False, 'No volume selected in scene'

        cur_dims = list(volume_node.GetImageData().GetDimensions())
        if cur_dims != meta.get('dimensions'):
            return False, (f"Dimension mismatch: "
                           f"recorded {meta['dimensions']}, current {cur_dims}")

        rec_sp  = meta.get('spacing', [])
        cur_sp  = list(volume_node.GetSpacing())
        for i, (rs, cs) in enumerate(zip(rec_sp, cur_sp)):
            tol = 0.001 * max(abs(rs), abs(cs), 1e-9)
            if abs(rs - cs) > tol:
                return False, (f"Spacing mismatch at axis {i}: "
                               f"recorded {rs:.4f} mm, current {cs:.4f} mm")
        return True, ''

    # ------------------------------------------------------------------ #
    # Record access                                                        #
    # ------------------------------------------------------------------ #

    @property
    def records(self) -> list:
        return list(self._records)

    def filter_types(self, *event_types) -> list:
        keep = frozenset(event_types)
        return [r for r in self._records if r.event_type in keep]

    def export_data(self) -> list:
        # Build RAS→IJK from the session_start volume metadata so every
        # record with a RAS position gets an IJK field for easy analysis.
        ras_to_ijk = None
        for r in self._records:
            if r.event_type == SESSION_START:
                meta = (r.payload or {}).get('volume') or {}
                raw  = meta.get('ijk_to_ras')
                if raw:
                    try:
                        ras_to_ijk = np.linalg.inv(
                            np.array(raw, dtype=np.float64).reshape(4, 4)
                        )
                    except np.linalg.LinAlgError:
                        pass
                break

        result = []
        for r in self._records:
            entry = {
                'timestamp': r.timestamp.isoformat(timespec='milliseconds'),
                'xy_global': list(r.xy_global),
                'ras':       r.ras,
                'event':     r.event_type,
                'payload':   r.payload,
            }
            if r.ras is not None and ras_to_ijk is not None:
                ras_h = np.array([r.ras[0], r.ras[1], r.ras[2], 1.0],
                                 dtype=np.float64)
                ijk_h = ras_to_ijk @ ras_h
                entry['ijk'] = [int(round(float(ijk_h[i]))) for i in range(3)]
            result.append(entry)
        return result

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _on_mouse(self, xy_global: tuple, event_type: str):
        now = datetime.datetime.now()
        self._last_xy = xy_global

        if event_type == MOVE:
            if (self._last_move_ts is not None and
                    (now - self._last_move_ts).total_seconds() * 1000
                    < self._move_interval_ms):
                return
            self._last_move_ts = now
            self._records.append(
                InputEventRecord(now, xy_global, _cursor_ras(), MOVE, None))
            return

        ras     = _cursor_ras()
        payload = (self.context_fn() if callable(self.context_fn) else {}) \
                  if event_type == PRESS else None
        self._records.append(InputEventRecord(now, xy_global, ras, event_type, payload))

    def _append(self, event_type, xy_global, ras, payload):
        self._records.append(
            InputEventRecord(datetime.datetime.now(), xy_global, ras, event_type, payload))

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        state = 'active' if self.is_active else 'stopped'
        hz    = round(1000.0 / self._move_interval_ms)
        return (f'MouseEventRecorder({state}, {len(self._records)} records, '
                f'{hz} Hz move sampling)')


# ---------------------------------------------------------------------------
# Process-scoped singleton
# ---------------------------------------------------------------------------

_recorder: MouseEventRecorder | None = None


def get_recorder(sample_rate_hz: int = 32) -> MouseEventRecorder:
    """Return the process-scoped MouseEventRecorder (created on first call)."""
    global _recorder
    if _recorder is None:
        _recorder = MouseEventRecorder(sample_rate_hz=sample_rate_hz)
    return _recorder


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cursor_ras():
    try:
        ch  = slicer.util.getNode('Crosshair')
        ras = [0.0, 0.0, 0.0]
        ch.GetCursorPositionRAS(ras)
        return ras
    except Exception:
        return None
