"""Mouse-centered timestamped input recorder.

Mouse events are recorded only when they occur inside a Slicer slice view and
the event RAS position maps inside the active volume.  UI-panel movement is
therefore ignored.  The recording is not a general UI macro; it captures the
mouse trajectory in 3D relative to the volume, the active handler/tool and its
parameters, mouse-triggered semantic actions, and segmentation-changing hotkeys.
"""

import collections
import datetime
import json
import logging

import qt
import vtk
import slicer

log = logging.getLogger(__name__)

InputEventRecord = collections.namedtuple(
    'InputEventRecord',
    ['event_id', 'timestamp', 'ras', 'event_type', 'payload'],
)

METADATA               = 'metadata'
MOVE                   = 'move'
PRESS                  = 'press'
RELEASE                = 'release'
KEY_PRESS              = 'key_press'
KEY_RELEASE            = 'key_release'
ACTION                 = 'action'
SESSION_START          = 'session_start'
SESSION_STOP           = 'session_stop'
SEGMENT_CREATED        = 'segment_created'
SEGMENT_REMOVED        = 'segment_removed'
SEGMENT_RENAMED        = 'segment_renamed'
VOLUME_CHANGED         = 'volume_changed'
MODEL_FAMILY_CHANGED   = 'model_family_changed'
MODEL_VARIANT_CHANGED  = 'model_variant_changed'
MODEL_CONFIRMED        = 'model_confirmed'
BRUSH_DIAMETER_CHANGED = 'brush_diameter_changed'
BRUSH_SPHERE_CHANGED   = 'brush_sphere_changed'
POINT_PLACED           = 'point_placed'
POINT_REMOVED          = 'point_removed'
VIEW_CHANGED           = 'view_changed'
POINT_DRAG_START       = 'point_drag_start'
POINT_DRAG_MOVE        = 'point_drag_move'
POINT_DRAG_END         = 'point_drag_end'

EXPORT_TYPE            = 'SegmentHumanBody.annotation_process'
VTK_OBSERVER_PRIORITY  = 1000.0


def _volume_metadata(volume_node) -> dict | None:
    if volume_node is None:
        return None
    dims = list(volume_node.GetImageData().GetDimensions())
    spacing = list(volume_node.GetSpacing())
    origin = list(volume_node.GetOrigin())
    mat = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(mat)
    ijk_to_ras = [mat.GetElement(r, c) for r in range(4) for c in range(4)]
    return {
        'name': volume_node.GetName(),
        'dimensions': dims,
        'spacing': spacing,
        'origin': origin,
        'ijk_to_ras': ijk_to_ras,
    }


_QtObject = getattr(qt, 'QObject', object)


class _SliceRecordFilter(_QtObject):
    """Per-slice-view Qt event filter; never consumes events."""

    def __init__(self, view_name, on_mouse, capture_moves=True):
        super().__init__()
        self._view_name = view_name
        self._on_mouse = on_mouse
        self._capture_moves = bool(capture_moves)

    def eventFilter(self, obj, event):
        t = event.type()
        try:
            if t == qt.QEvent.MouseMove:
                if not self._capture_moves:
                    return False
                pos = event.pos()
                buttons = event.buttons()
                self._on_mouse(self._view_name, (pos.x(), pos.y()), MOVE,
                               {'mouse_status': 'move',
                                'left_button_down': bool(buttons & qt.Qt.LeftButton),
                                'analysis_event_type': 'trajectory_event'})
            elif t == qt.QEvent.MouseButtonPress and event.button() == qt.Qt.LeftButton:
                pos = event.pos()
                self._on_mouse(self._view_name, (pos.x(), pos.y()), PRESS,
                               {'mouse_status': 'press',
                                'left_button_down': True,
                                'analysis_event_type': 'boundary_event'})
            elif t == qt.QEvent.MouseButtonRelease and event.button() == qt.Qt.LeftButton:
                pos = event.pos()
                self._on_mouse(self._view_name, (pos.x(), pos.y()), RELEASE,
                               {'mouse_status': 'release',
                                'left_button_down': False,
                                'analysis_event_type': 'boundary_event'})
            elif t == qt.QEvent.Wheel:
                pos = event.pos()
                delta = event.angleDelta()
                self._on_mouse(self._view_name, (pos.x(), pos.y()), VIEW_CHANGED,
                               {'mouse_status': 'view',
                                'view_event': 'wheel',
                                'wheel_delta': [delta.x(), delta.y()],
                                'analysis_event_type': 'trajectory_event'})
        except Exception as exc:
            log.error('[SliceRecordFilter] %s', exc)
        return False


class _SliceRecordInteractorObserver:
    """Per-slice-view VTK interactor observer for Segment Editor effects."""

    def __init__(self, view_name, slice_view, on_mouse):
        self._view_name = view_name
        self._slice_view = slice_view
        self._on_mouse = on_mouse
        self._interactor = None
        self._tags = []
        self._left_down = False

    def install(self):
        self._interactor = _slice_interactor(self._slice_view)
        if self._interactor is None:
            return False
        self._left_down = _left_button_is_down()
        for event_name in (
            'MouseMoveEvent',
            'LeftButtonPressEvent',
            'LeftButtonReleaseEvent',
            'MouseWheelForwardEvent',
            'MouseWheelBackwardEvent',
        ):
            try:
                tag = self._interactor.AddObserver(
                    event_name, self._on_event, VTK_OBSERVER_PRIORITY)
                self._tags.append(tag)
            except TypeError:
                tag = self._interactor.AddObserver(event_name, self._on_event)
                self._tags.append(tag)
            except Exception as exc:
                log.debug('[SliceRecordInteractorObserver] %s failed: %s',
                          event_name, exc)
        return bool(self._tags)

    def remove(self):
        if self._interactor is not None:
            for tag in self._tags:
                try:
                    self._interactor.RemoveObserver(tag)
                except Exception:
                    pass
        self._interactor = None
        self._tags = []
        self._left_down = False

    def _on_event(self, caller, event_name):
        try:
            x, y = caller.GetEventPosition()
            xy = _vtk_event_xy_to_slice_xy(self._slice_view, x, y)
            if event_name == 'MouseMoveEvent':
                self._on_mouse(self._view_name, xy, MOVE, {
                    'mouse_status': 'move',
                    'left_button_down': self._left_down or _left_button_is_down(),
                    'input_source': 'vtk_interactor',
                    'analysis_event_type': 'trajectory_event',
                })
            elif event_name == 'LeftButtonPressEvent':
                self._left_down = True
                self._on_mouse(self._view_name, xy, PRESS, {
                    'mouse_status': 'press',
                    'left_button_down': True,
                    'input_source': 'vtk_interactor',
                    'analysis_event_type': 'boundary_event',
                })
            elif event_name == 'LeftButtonReleaseEvent':
                self._on_mouse(self._view_name, xy, RELEASE, {
                    'mouse_status': 'release',
                    'left_button_down': False,
                    'input_source': 'vtk_interactor',
                    'analysis_event_type': 'boundary_event',
                })
                self._left_down = False
            elif event_name in ('MouseWheelForwardEvent', 'MouseWheelBackwardEvent'):
                delta = 120 if event_name == 'MouseWheelForwardEvent' else -120
                self._on_mouse(self._view_name, xy, VIEW_CHANGED, {
                    'mouse_status': 'view',
                    'view_event': 'wheel',
                    'wheel_delta': [0, delta],
                    'input_source': 'vtk_interactor',
                    'analysis_event_type': 'trajectory_event',
                })
        except Exception as exc:
            log.error('[SliceRecordInteractorObserver] %s', exc)


class MouseEventRecorder:
    """Records timestamped in-volume slice-view events and named actions."""

    MOVE                   = MOVE
    METADATA               = METADATA
    PRESS                  = PRESS
    RELEASE                = RELEASE
    KEY_PRESS              = KEY_PRESS
    KEY_RELEASE            = KEY_RELEASE
    ACTION                 = ACTION
    SESSION_START          = SESSION_START
    SESSION_STOP           = SESSION_STOP
    SEGMENT_CREATED        = SEGMENT_CREATED
    SEGMENT_REMOVED        = SEGMENT_REMOVED
    SEGMENT_RENAMED        = SEGMENT_RENAMED
    VOLUME_CHANGED         = VOLUME_CHANGED
    MODEL_FAMILY_CHANGED   = MODEL_FAMILY_CHANGED
    MODEL_VARIANT_CHANGED  = MODEL_VARIANT_CHANGED
    MODEL_CONFIRMED        = MODEL_CONFIRMED
    BRUSH_DIAMETER_CHANGED = BRUSH_DIAMETER_CHANGED
    BRUSH_SPHERE_CHANGED   = BRUSH_SPHERE_CHANGED
    POINT_PLACED           = POINT_PLACED
    POINT_REMOVED          = POINT_REMOVED
    VIEW_CHANGED           = VIEW_CHANGED
    POINT_DRAG_START       = POINT_DRAG_START
    POINT_DRAG_MOVE        = POINT_DRAG_MOVE
    POINT_DRAG_END         = POINT_DRAG_END

    def __init__(self, sample_rate_hz: int = 30):
        if sample_rate_hz <= 0:
            raise ValueError(f'sample_rate_hz must be positive, got {sample_rate_hz}')
        self._records: list = []
        self._filters: list = []
        self._active = False
        self._move_interval_ms = 1000.0 / sample_rate_hz
        self._move_timer = None
        self._pending_move_sample = None
        self._last_sampled_move_key = None
        self._last_boundary_key = None
        self._active_mouse_press = False
        self._last_point_drag_ts: datetime.datetime | None = None
        self._volume_node = None
        self.record_non_annotative_movement = False
        self.context_fn = None
        self.on_record_appended = None
        self._next_event_id = 1

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, volume_node=None, segmentation_name: str | None = None,
              record_non_annotative_movement: bool = False):
        if self._active:
            return
        self._active = True
        self._volume_node = volume_node
        self.record_non_annotative_movement = bool(record_non_annotative_movement)
        self._active_mouse_press = _left_button_is_down()
        for view_name in ('Red', 'Green', 'Yellow'):
            view = _slice_view(view_name)
            if view is None:
                continue
            observer = _SliceRecordInteractorObserver(view_name, view, self._on_mouse)
            vtk_capture = observer.install()
            filt = _SliceRecordFilter(
                view_name, self._on_mouse, capture_moves=not vtk_capture)
            view.installEventFilter(filt)
            self._filters.append((view, filt, observer))
        if self._filters:
            self._start_move_timer()
        hz = round(1000.0 / self._move_interval_ms)
        self._append(METADATA, None, {
            'volume': _volume_metadata(volume_node),
            'segmentation': segmentation_name,
            'sample_rate_hz': hz,
            'capture_scope': 'slice-view-in-volume',
            'recorder_style': 'event_listener',
            'record_non_annotative_movement': self.record_non_annotative_movement,
            'started_with_left_button_down': self._active_mouse_press,
            'initial_visual_state': _all_slice_visual_state(),
        })
        log.debug('[MouseEventRecorder] started with %d slice filters', len(self._filters))

    def stop(self):
        self._sample_pending_move()
        self._stop_move_timer()
        for view, filt, observer in self._filters:
            try:
                view.removeEventFilter(filt)
            except Exception:
                pass
            observer.remove()
        self._filters = []
        self._active = False
        log.debug('[MouseEventRecorder] stopped with %d records', len(self._records))

    def clear(self):
        self._records.clear()
        self._pending_move_sample = None
        self._last_sampled_move_key = None
        self._last_boundary_key = None
        self._active_mouse_press = False
        self._last_point_drag_ts = None
        self._next_event_id = 1

    def record_action(self, name: str):
        self._append(ACTION, None, {'name': name})

    def record_segment_created(self, segment_id: str, seg_name: str):
        # Segment creation is intentionally not part of the current process log.
        return

    def record_volume_changed(self, volume_name: str | None):
        self._append(VOLUME_CHANGED, None, {'volume': volume_name})

    def set_volume_node(self, volume_node):
        self._volume_node = volume_node

    def record_model_family_changed(self, family: str):
        self._append(MODEL_FAMILY_CHANGED, None, {'family': family})

    def record_model_variant_changed(self, variant: str):
        self._append(MODEL_VARIANT_CHANGED, None, {'variant': variant})

    def record_model_confirmed(self, family: str, variant: str):
        self._append(MODEL_CONFIRMED, None, {
            'family': family, 'variant': variant})

    def record_brush_diameter_changed(self, diameter_mm: float):
        self._append(BRUSH_DIAMETER_CHANGED, None, {'diameter_mm': diameter_mm})

    def record_brush_sphere_changed(self, sphere: bool):
        self._append(BRUSH_SPHERE_CHANGED, None, {'sphere': sphere})

    def record_segment_removed(self, segment_id: str, seg_name: str):
        self._append(SEGMENT_REMOVED, None, {
            'segment_id': segment_id, 'seg_name': seg_name})

    def record_segment_renamed(self, segment_id: str, old_name: str, new_name: str):
        self._append(SEGMENT_RENAMED, None, {
            'segment_id': segment_id,
            'old_name': old_name,
            'new_name': new_name,
        })

    def record_point_placed(self, segment_id: str, ras: list, is_negative: bool,
                            view_name: str | None = None, point_index=None,
                            point_id: str | None = None,
                            point_name: str | None = None,
                            point_action: str = 'place'):
        payload = {
            'segment_id': segment_id,
            'ras': list(ras),
            'is_negative': bool(is_negative),
        }
        if point_index is not None:
            payload['point_index'] = int(point_index)
        if point_id:
            payload['point_id'] = point_id
        if point_name:
            payload['point_name'] = point_name
        if view_name:
            payload['view_name'] = view_name
            payload['handler'] = 'point'
            payload['handler_params'] = {}
        payload['analysis_event_type'] = 'boundary_event'
        payload['mouse_status'] = 'release'
        payload['point_action'] = point_action
        payload['trajectory_kind'] = None
        payload['trajectory_role'] = None
        self._append(POINT_PLACED, list(ras), payload)

    def record_point_removed(self, segment_id: str, ras: list | None,
                             is_negative: bool, view_name: str | None = None,
                             point_index=None, point_id: str | None = None,
                             point_name: str | None = None):
        payload = {
            'segment_id': segment_id,
            'is_negative': bool(is_negative),
            'handler': 'point',
            'handler_params': {},
            'analysis_event_type': 'boundary_event',
            'mouse_status': 'release',
            'point_action': 'remove',
            'trajectory_kind': None,
            'trajectory_role': None,
        }
        if ras is not None:
            payload['ras'] = list(ras)
        if point_index is not None:
            payload['point_index'] = int(point_index)
        if point_id:
            payload['point_id'] = point_id
        if point_name:
            payload['point_name'] = point_name
        if view_name:
            payload['view_name'] = view_name
        self._append(POINT_REMOVED, list(ras) if ras is not None else None, payload)

    def record_point_drag(self, phase: str, segment_id: str, ras: list,
                          is_negative: bool, view_name: str | None = None,
                          point_index=None, point_id: str | None = None,
                          point_name: str | None = None):
        event_type = {
            'start': POINT_DRAG_START,
            'move': POINT_DRAG_MOVE,
            'end': POINT_PLACED,
        }.get(phase)
        if event_type is None:
            raise ValueError(f'Unknown point drag phase: {phase}')
        now = datetime.datetime.now()
        if event_type == POINT_DRAG_MOVE:
            if (self._last_point_drag_ts is not None and
                    (now - self._last_point_drag_ts).total_seconds() * 1000
                    < self._move_interval_ms):
                return
            self._last_point_drag_ts = now
        elif event_type == POINT_DRAG_START:
            self._last_point_drag_ts = now
        else:
            self._last_point_drag_ts = None
        payload = {
            'segment_id': segment_id,
            'ras': list(ras),
            'is_negative': bool(is_negative),
            'handler': 'point',
            'handler_params': {},
            'analysis_event_type': (
                'trajectory_event' if event_type == POINT_DRAG_MOVE
                else 'boundary_event'
            ),
            'trajectory_kind': (
                'non_annotation_move' if event_type == POINT_DRAG_MOVE
                else None
            ),
            'trajectory_role': (
                'visualization_trajectory' if event_type == POINT_DRAG_MOVE
                else None
            ),
            'point_drag_phase': phase,
            'point_action': {
                POINT_DRAG_START: 'grab',
                POINT_DRAG_MOVE: 'move',
                POINT_PLACED: 'replace',
            }[event_type],
        }
        if point_index is not None:
            payload['point_index'] = int(point_index)
        if point_id:
            payload['point_id'] = point_id
        if point_name:
            payload['point_name'] = point_name
        if view_name:
            payload['view_name'] = view_name
        self._records.append(self._new_record(
            now, list(ras), event_type, payload))
        self._notify_record_appended()

    def should_sample_point_drag(self, phase: str) -> bool:
        if phase != 'move':
            return True
        if self._last_point_drag_ts is None:
            return True
        elapsed_ms = (
            datetime.datetime.now() - self._last_point_drag_ts
        ).total_seconds() * 1000
        return elapsed_ms >= self._move_interval_ms

    def save_to_file(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.export_data(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> 'MouseEventRecorder':
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return cls._load_export_v2(data)
        if not isinstance(data, list):
            raise ValueError(
                'Not a recording file - expected a JSON object or array at the top level.'
            )
        recorder = cls()
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f'Record {i} is not a JSON object')
            recorder._records.append(InputEventRecord(
                int(item['event_id']),
                datetime.datetime.fromisoformat(item['timestamp']),
                item.get('ras'),
                item['event'],
                item.get('payload'),
            ))
        recorder._next_event_id = (
            max((r.event_id for r in recorder._records), default=0) + 1
        )
        return recorder

    @classmethod
    def _load_export_v2(cls, data: dict) -> 'MouseEventRecorder':
        if data.get('type') != EXPORT_TYPE:
            raise ValueError('Not a SegmentHumanBody annotation-process recording')
        recorder = cls()
        metadata = data.get('metadata') or {}
        start_raw = metadata.get('start_time')
        start_ts = (
            datetime.datetime.fromisoformat(start_raw)
            if start_raw else datetime.datetime.now()
        )
        recorder._records.append(InputEventRecord(0, start_ts, None, METADATA, metadata))
        for item in data.get('events') or []:
            ts = start_ts + datetime.timedelta(
                milliseconds=float(item.get('t_ms', 0)))
            payload = cls._payload_from_compact_event(item)
            event_id = int(item.get('id', recorder._next_event_id))
            recorder._records.append(InputEventRecord(
                event_id,
                ts,
                item.get('ras'),
                item.get('event'),
                payload,
            ))
            recorder._next_event_id = max(recorder._next_event_id, event_id + 1)
        return recorder

    @staticmethod
    def _payload_from_compact_event(item: dict) -> dict:
        payload = {}
        key_map = {
            'view': 'view_name',
            'mouse': 'mouse_status',
            'kind': 'trajectory_kind',
            'role': 'trajectory_role',
            'tool': 'handler',
            'segment': 'segment_id',
            'brush_mm': 'brush_radius_mm',
            'point': 'point_id',
            'point_name': 'point_name',
            'point_index': 'point_index',
            'point_action': 'point_action',
            'negative': 'is_negative',
            'analysis': 'analysis_event_type',
            'view_event': 'view_event',
            'wheel_delta': 'wheel_delta',
            'visual_state': 'visual_state',
            'name': 'name',
            'old_name': 'old_name',
            'new_name': 'new_name',
            'seg_name': 'seg_name',
            'boundary_source': 'boundary_source',
        }
        for src, dst in key_map.items():
            if src in item:
                payload[dst] = item[src]
        if 'pressed' in item:
            is_pressed = bool(item['pressed'])
            payload['mouse_button_state'] = 'pressed' if is_pressed else 'released'
            payload['left_button_down'] = is_pressed
        if 'ras' in item:
            payload.setdefault('ras', item['ras'])
        return payload

    def matches_volume(self, volume_node) -> tuple:
        starts = [
            r for r in self._records
            if r.event_type in (METADATA, SESSION_START)
        ]
        if not starts:
            return False, 'No session_start record found'
        meta = (starts[0].payload or {}).get('volume')
        if not meta:
            return False, 'No volume metadata in record'
        if volume_node is None:
            return False, 'No volume selected in scene'
        cur_dims = list(volume_node.GetImageData().GetDimensions())
        if cur_dims != meta.get('dimensions'):
            return False, f"Dimension mismatch: recorded {meta['dimensions']}, current {cur_dims}"
        rec_sp = meta.get('spacing', [])
        cur_sp = list(volume_node.GetSpacing())
        for i, (rs, cs) in enumerate(zip(rec_sp, cur_sp)):
            tol = 0.001 * max(abs(rs), abs(cs), 1e-9)
            if abs(rs - cs) > tol:
                return False, f"Spacing mismatch at axis {i}: recorded {rs:.4f} mm, current {cs:.4f} mm"
        return True, ''

    @property
    def records(self) -> list:
        return list(self._records)

    def filter_types(self, *event_types) -> list:
        keep = frozenset(event_types)
        return [r for r in self._records if r.event_type in keep]

    def export_data(self) -> dict:
        metadata_record = next(
            (r for r in self._records if r.event_type == METADATA), None)
        start_ts = (
            metadata_record.timestamp if metadata_record is not None
            else (self._records[0].timestamp if self._records
                  else datetime.datetime.now())
        )
        metadata = dict(metadata_record.payload or {}) if metadata_record else {}
        metadata['start_time'] = start_ts.isoformat(timespec='milliseconds')
        metadata['coordinate_system'] = 'RAS'
        metadata['event_time_base'] = 't_ms_from_metadata_start_time'
        return {
            'type': EXPORT_TYPE,
            'metadata': metadata,
            'events': [
                _compact_event(r, start_ts, export_id)
                for export_id, r in enumerate(
                    _records_for_compact_export(self._records),
                    start=1,
                )
            ],
        }

    def _on_mouse(self, view_name: str, xy_local: tuple, event_type: str, extra=None):
        timestamp = datetime.datetime.now()
        if event_type == MOVE:
            self._capture_pending_move(timestamp, view_name, xy_local, extra)
            return
        self._sample_pending_move()
        self._append_mouse_record(timestamp, view_name, xy_local, event_type, extra)

    def _capture_pending_move(self, timestamp, view_name, xy_local, extra=None):
        sample = self._mouse_sample(timestamp, view_name, xy_local, MOVE, extra)
        if sample is not None:
            self._pending_move_sample = sample

    def _sample_pending_move(self):
        if not self._active or self._pending_move_sample is None:
            return
        timestamp, view_name, ras, payload = self._pending_move_sample
        self._pending_move_sample = None
        self._append_prepared_mouse_record(
            timestamp, view_name, ras, MOVE, payload, dedupe=True)

    def _append_mouse_record(self, timestamp, view_name, xy_local, event_type,
                             extra=None, dedupe=False):
        sample = self._mouse_sample(timestamp, view_name, xy_local, event_type, extra)
        if sample is None:
            return
        timestamp, view_name, ras, payload = sample
        self._append_prepared_mouse_record(
            timestamp, view_name, ras, event_type, payload, dedupe=dedupe)

    def _mouse_sample(self, timestamp, view_name, xy_local, event_type, extra=None):
        ras = _slice_xy_to_ras(view_name, xy_local)
        if ras is None or not _ras_inside_volume(self._volume_node, ras):
            if event_type == MOVE:
                self._last_sampled_move_key = None
            return None
        payload = self._event_payload(view_name)
        if extra:
            payload.update(extra)
        if event_type == VIEW_CHANGED:
            payload['visual_state'] = _visual_state(view_name)
        if event_type == MOVE:
            payload['mouse_button_state'] = (
                'pressed'
                if payload.get('left_button_down') or self._active_mouse_press
                else 'released'
            )
        payload['trajectory_kind'] = _trajectory_kind(payload)
        payload['trajectory_role'] = _trajectory_role(payload)
        return timestamp, view_name, ras, payload

    def _append_prepared_mouse_record(self, timestamp, view_name, ras, event_type,
                                      payload, dedupe=False):
        if self._is_duplicate_boundary(timestamp, view_name, ras, event_type, payload):
            return
        self._maybe_append_missing_press(timestamp, ras, payload, event_type)
        if dedupe:
            key = _move_sample_key(view_name, ras, payload)
            if key == self._last_sampled_move_key:
                return
            self._last_sampled_move_key = key
        self._records.append(self._new_record(
            timestamp, ras, event_type, payload))
        if event_type == PRESS:
            self._active_mouse_press = True
            self._remember_boundary(timestamp, view_name, ras, event_type, payload)
        elif event_type == RELEASE:
            self._active_mouse_press = False
            self._remember_boundary(timestamp, view_name, ras, event_type, payload)
        self._notify_record_appended()

    def _should_skip_non_annotative_move(self, event_type, payload):
        if event_type != MOVE:
            return False
        if payload.get('analysis_event_type') != 'trajectory_event':
            return False
        if payload.get('view_event'):
            return False
        if payload.get('trajectory_kind') == 'annotation_move':
            return False
        if (payload.get('handler') == 'point' and
                payload.get('mouse_button_state') == 'pressed'):
            return False
        if self.record_non_annotative_movement:
            return False
        return True

    @staticmethod
    def _should_skip_raw_point_boundary(event_type, payload):
        if event_type not in (PRESS, RELEASE):
            return False
        return payload.get('handler') == 'point'

    def _maybe_append_missing_press(self, timestamp, ras, payload, event_type):
        if event_type not in (MOVE, RELEASE):
            return
        if self._active_mouse_press:
            return
        if event_type == MOVE and not payload.get('left_button_down'):
            return
        if payload.get('handler') not in ('brush', 'erase'):
            return
        press_payload = dict(payload)
        press_payload['mouse_status'] = 'press'
        press_payload['analysis_event_type'] = 'boundary_event'
        press_payload['trajectory_kind'] = None
        press_payload['trajectory_role'] = None
        press_payload['boundary_source'] = (
            'inferred_from_release'
            if event_type == RELEASE
            else 'inferred_from_first_drag_sample'
        )
        self._records.append(self._new_record(
            timestamp, ras, PRESS, press_payload))
        self._active_mouse_press = True
        self._remember_boundary(timestamp, press_payload.get('view_name'), ras,
                                PRESS, press_payload)
        self._notify_record_appended()

    def _is_duplicate_boundary(self, timestamp, view_name, ras, event_type, payload):
        if event_type not in (PRESS, RELEASE):
            return False
        last = self._last_boundary_key
        if last is None:
            return False
        key = _boundary_key(view_name, ras, event_type, payload)
        last_key, last_ts = last
        if key != last_key:
            return False
        return (timestamp - last_ts).total_seconds() * 1000.0 <= 100.0

    def _remember_boundary(self, timestamp, view_name, ras, event_type, payload):
        self._last_boundary_key = (
            _boundary_key(view_name, ras, event_type, payload),
            timestamp,
        )

    def _start_move_timer(self):
        if self._move_timer is not None:
            return
        try:
            timer = qt.QTimer()
            timer.connect('timeout()', self._sample_pending_move)
            timer.start(int(round(self._move_interval_ms)))
            self._move_timer = timer
        except Exception as exc:
            log.debug('[MouseEventRecorder] move timer unavailable: %s', exc)

    def _stop_move_timer(self):
        timer = self._move_timer
        self._move_timer = None
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        self._pending_move_sample = None
        self._last_sampled_move_key = None
        self._last_boundary_key = None

    def _event_payload(self, view_name):
        payload = {}
        if callable(self.context_fn):
            try:
                payload = self.context_fn(view_name=view_name) or {}
            except TypeError:
                payload = self.context_fn() or {}
        payload.setdefault('view_name', view_name)
        tool = payload.get('tool')
        payload['handler'] = tool
        payload['handler_params'] = _handler_params(payload)
        payload.setdefault('analysis_event_type', None)
        return payload

    def _append(self, event_type, ras, payload):
        self._records.append(self._new_record(
            datetime.datetime.now(), ras, event_type, payload))
        self._notify_record_appended()

    def _new_record(self, timestamp, ras, event_type, payload):
        record = InputEventRecord(
            self._next_event_id, timestamp, ras, event_type, payload)
        self._next_event_id += 1
        return record

    def _notify_record_appended(self):
        if callable(self.on_record_appended):
            try:
                self.on_record_appended()
            except Exception as exc:
                log.debug('[MouseEventRecorder] on_record_appended failed: %s', exc)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        state = 'active' if self.is_active else 'stopped'
        hz = round(1000.0 / self._move_interval_ms)
        return f'MouseEventRecorder({state}, {len(self._records)} records, {hz} Hz)'


_recorder: MouseEventRecorder | None = None


def get_recorder(sample_rate_hz: int = 30) -> MouseEventRecorder:
    global _recorder
    if _recorder is None:
        _recorder = MouseEventRecorder(sample_rate_hz=sample_rate_hz)
    return _recorder


def _records_for_compact_export(records) -> list:
    result = []
    point_drag_active = False
    for record in records:
        if record.event_type == METADATA:
            continue
        payload = record.payload or {}
        if record.event_type == POINT_DRAG_START:
            _drop_immediate_raw_point_press(result, record)
            point_drag_active = True
            result.append(record)
            continue
        if point_drag_active and _is_raw_mouse_event(record):
            continue
        result.append(record)
        if (point_drag_active and record.event_type == POINT_PLACED and
                payload.get('point_action') == 'replace'):
            point_drag_active = False
        elif point_drag_active and record.event_type == POINT_REMOVED:
            point_drag_active = False
    return result


def _drop_immediate_raw_point_press(result: list, point_drag_start):
    if not result:
        return
    previous = result[-1]
    if previous.event_type != PRESS or not _is_raw_mouse_event(previous):
        return
    elapsed_ms = (
        point_drag_start.timestamp - previous.timestamp
    ).total_seconds() * 1000.0
    if 0 <= elapsed_ms <= 250.0:
        result.pop()


def _is_raw_mouse_event(record) -> bool:
    if record.event_type not in (PRESS, MOVE, RELEASE):
        return False
    payload = record.payload or {}
    return not payload.get('point_action') and not payload.get('view_event')


def _compact_event(record, start_ts, export_id=None) -> dict:
    payload = record.payload or {}
    event = {
        'id': int(export_id if export_id is not None else record.event_id),
        't_ms': int(round((record.timestamp - start_ts).total_seconds() * 1000)),
        'event': record.event_type,
    }
    if record.ras is not None:
        event['ras'] = [float(v) for v in record.ras]
    _copy_compact(payload, event, 'view_name', 'view')
    _copy_compact(payload, event, 'mouse_status', 'mouse')
    pressed = _compact_pressed_state(record.event_type, payload)
    if pressed is not None:
        event['pressed'] = pressed
    _copy_compact(payload, event, 'analysis_event_type', 'analysis')
    _copy_compact(payload, event, 'trajectory_kind', 'kind')
    _copy_compact(payload, event, 'trajectory_role', 'role')
    _copy_compact(payload, event, 'handler', 'tool')
    _copy_compact(payload, event, 'segment_id', 'segment')
    _copy_compact(payload, event, 'point_id', 'point')
    _copy_compact(payload, event, 'point_name', 'point_name')
    _copy_compact(payload, event, 'point_index', 'point_index')
    _copy_compact(payload, event, 'point_action', 'point_action')
    _copy_compact(payload, event, 'is_negative', 'negative')
    _copy_compact(payload, event, 'boundary_source', 'boundary_source')
    _copy_compact(payload, event, 'view_event', 'view_event')
    _copy_compact(payload, event, 'wheel_delta', 'wheel_delta')
    _copy_compact(payload, event, 'visual_state', 'visual_state')
    _copy_compact(payload, event, 'name', 'name')
    _copy_compact(payload, event, 'family', 'family')
    _copy_compact(payload, event, 'variant', 'variant')
    _copy_compact(payload, event, 'volume', 'volume')
    _copy_compact(payload, event, 'diameter_mm', 'diameter_mm')
    _copy_compact(payload, event, 'sphere', 'sphere')
    _copy_compact(payload, event, 'seg_name', 'seg_name')
    _copy_compact(payload, event, 'old_name', 'old_name')
    _copy_compact(payload, event, 'new_name', 'new_name')
    brush_mm = payload.get('brush_radius_mm')
    if brush_mm is None:
        brush_mm = (payload.get('handler_params') or {}).get('brush_radius_mm')
    if brush_mm is not None:
        event['brush_mm'] = brush_mm
    return event


def _copy_compact(source: dict, target: dict, source_key: str, target_key: str):
    value = source.get(source_key)
    if value is not None:
        target[target_key] = value


def _compact_pressed_state(event_type, payload):
    if event_type == PRESS:
        return 1
    if event_type == RELEASE:
        return 0
    state = payload.get('mouse_button_state')
    if state == 'pressed':
        return 1
    if state == 'released':
        return 0
    if 'left_button_down' in payload:
        return 1 if payload.get('left_button_down') else 0
    return None


def _slice_view(view_name):
    try:
        sw = slicer.app.layoutManager().sliceWidget(view_name)
        return sw.sliceView() if sw else None
    except Exception:
        return None


def _left_button_is_down():
    try:
        return bool(qt.QApplication.mouseButtons() & qt.Qt.LeftButton)
    except Exception:
        return False


def _slice_interactor(slice_view):
    if slice_view is None:
        return None
    try:
        interactor = slice_view.interactor()
        if interactor is not None:
            return interactor
    except Exception:
        pass
    try:
        return slice_view.renderWindow().GetInteractor()
    except Exception:
        return None


def _vtk_event_xy_to_slice_xy(slice_view, x, y):
    height = _slice_view_render_height(slice_view)
    if height <= 0:
        height = _slice_view_qt_height(slice_view)
    if height > 0:
        y = height - int(y) - 1
    return int(x), int(y)


def _slice_view_render_height(slice_view):
    try:
        return int(slice_view.renderWindow().GetSize()[1])
    except Exception:
        return 0


def _slice_view_qt_height(slice_view):
    try:
        height = slice_view.height
        return int(height() if callable(height) else height)
    except Exception:
        try:
            size = slice_view.size()
            height = size.height
            return int(height() if callable(height) else height)
        except Exception:
            return 0


def _slice_xy_to_ras(view_name, xy):
    try:
        sw = slicer.app.layoutManager().sliceWidget(view_name)
        mat = sw.mrmlSliceNode().GetXYToRAS()
        x, y = xy
        return [
            mat.GetElement(r, 0) * x
            + mat.GetElement(r, 1) * y
            + mat.GetElement(r, 3)
            for r in range(3)
        ]
    except Exception:
        return None


def _ras_inside_volume(volume_node, ras):
    if volume_node is None or ras is None:
        return False
    try:
        mat = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(mat)
        ijk = [
            sum(mat.GetElement(r, c) * (ras[c] if c < 3 else 1.0)
                for c in range(4))
            for r in range(3)
        ]
        dims = volume_node.GetImageData().GetDimensions()
        return all(-0.5 <= ijk[i] <= dims[i] - 0.5 for i in range(3))
    except Exception:
        return False


def _visual_state(view_name):
    try:
        node = slicer.app.layoutManager().sliceWidget(view_name).mrmlSliceNode()
        return {
            'slice_offset': float(node.GetSliceOffset()),
            'field_of_view': [float(v) for v in node.GetFieldOfView()],
        }
    except Exception:
        return {}


def _all_slice_visual_state():
    return {
        view_name: _visual_state(view_name)
        for view_name in ('Red', 'Green', 'Yellow')
    }


def _handler_params(payload):
    tool = payload.get('tool')
    if tool in ('brush', 'erase'):
        return {
            'brush_radius_mm': payload.get('brush_radius_mm'),
            'axis': payload.get('axis'),
            'slice_idx': payload.get('slice_idx'),
        }
    if tool == 'point':
        return {
            'axis': payload.get('axis'),
            'slice_idx': payload.get('slice_idx'),
        }
    return {}


def _trajectory_kind(payload):
    if payload.get('analysis_event_type') != 'trajectory_event':
        return None
    if payload.get('view_event'):
        return 'view_change'
    if payload.get('point_drag_phase') == 'move':
        return 'non_annotation_move'
    if payload.get('tool') in ('brush', 'erase'):
        if payload.get('mouse_button_state') == 'pressed' or payload.get('left_button_down'):
            return 'annotation_move'
        return 'non_annotation_move'
    return 'non_annotation_move'


def _trajectory_role(payload):
    if payload.get('analysis_event_type') != 'trajectory_event':
        return None
    if payload.get('view_event'):
        return 'visualization_trajectory'
    if payload.get('trajectory_kind') == 'annotation_move':
        return 'annotation_trajectory'
    return 'visualization_trajectory'


def _move_sample_key(view_name, ras, payload):
    return (
        view_name,
        tuple(round(float(v), 3) for v in ras),
        payload.get('handler'),
        payload.get('mouse_status'),
        payload.get('trajectory_role'),
        payload.get('trajectory_kind'),
    )


def _boundary_key(view_name, ras, event_type, payload):
    return (
        view_name,
        event_type,
        tuple(round(float(v), 3) for v in ras),
        payload.get('handler'),
        payload.get('segment_id'),
    )
