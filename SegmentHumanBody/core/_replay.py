"""Replay engine for MouseEventRecorder sessions.

ReplayEngine
    Animates a loaded recording onto a new segmentation node.
    Brush / erase strokes are painted directly with numpy — no SegmentEditor
    mouse simulation required.

Usage::

    engine = ReplayEngine()
    engine.start(recorder, widget, on_done=callback)
    # later, if needed:
    engine.stop()
"""

import logging

import numpy as np
import qt
import vtk
import slicer

from core.utils import AXIS_TO_XY_COLS, ras_to_ijk_3d
from core._mouse_recorder import (
    MOVE, PRESS, RELEASE, ACTION, SESSION_START, SESSION_STOP,
    SEGMENT_CREATED, SEGMENT_CHANGED, VOLUME_CHANGED,
)

log = logging.getLogger(__name__)


class ReplayEngine:
    """Step-by-step replay of a MouseEventRecorder session.

    Creates a new vtkMRMLSegmentationNode, maps original segment IDs to fresh
    IDs, and replays every event at approximately its original inter-event
    timing via QTimer.
    """

    _MIN_STEP_MS = 16  # UI breathes between steps

    def __init__(self):
        self._ops:    list  = []
        self._op_idx: int   = 0
        self._timer         = None
        self._seg_id_map: dict = {}
        self._widget        = None
        self._replay_seg    = None   # vtkMRMLSegmentationNode
        self._vol_node      = None
        self._ras_to_ijk    = None   # numpy (4,4)
        self._spacing       = None   # [sx_I, sy_J, sz_K] in mm
        self._on_done       = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def is_running(self) -> bool:
        return self._timer is not None and self._timer.isActive()

    def start(self, recorder, widget, on_done=None):
        """Begin replaying *recorder* onto a new segmentation.

        Parameters
        ----------
        recorder : MouseEventRecorder   (loaded, stopped)
        widget   : SegmentHumanBodyWidget
        on_done  : callable | None      fired on completion or stop
        """
        if self.is_running:
            self.stop()

        self._widget  = widget
        self._on_done = on_done

        pn = getattr(widget, '_parameterNode', None)
        if pn is None:
            slicer.util.errorDisplay('Replay: no parameter node — load a volume first.')
            return

        vol_node, _ = widget.logic.getVolumeAndSegmentation(pn)
        if vol_node is None:
            slicer.util.errorDisplay('Replay: no volume selected.')
            return

        ok, reason = recorder.matches_volume(vol_node)
        if not ok:
            slicer.util.errorDisplay(f'Replay: volume mismatch\n{reason}')
            return

        self._vol_node = vol_node
        self._spacing  = list(vol_node.GetSpacing())   # [I_sp, J_sp, K_sp]

        m = vtk.vtkMatrix4x4()
        vol_node.GetRASToIJKMatrix(m)
        self._ras_to_ijk = np.array(
            [[m.GetElement(r, c) for c in range(4)] for r in range(4)]
        )

        self._replay_seg = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLSegmentationNode', 'Replay'
        )
        self._replay_seg.CreateDefaultDisplayNodes()
        self._replay_seg.SetReferenceImageGeometryParameterFromVolumeNode(vol_node)
        self._replay_seg.CreateClosedSurfaceRepresentation()
        self._seg_id_map = {}

        self._ops    = self._preprocess(recorder.records)
        self._op_idx = 0

        self._timer = qt.QTimer()
        self._timer.setSingleShot(True)
        self._timer.connect('timeout()', self._step)
        self._timer.start(0)
        log.debug('[ReplayEngine] started — %d ops', len(self._ops))

    def stop(self):
        """Abort replay and fire on_done."""
        if self._timer:
            self._timer.stop()
            self._timer = None
        self._notify_done()
        log.debug('[ReplayEngine] stopped')

    # ------------------------------------------------------------------ #
    # Pre-processing                                                        #
    # ------------------------------------------------------------------ #

    def _preprocess(self, records):
        """Convert flat record list to typed (kind, ts, ...) ops.

        Groups PRESS → MOVEs → RELEASE into a single 'stroke' op so the
        mask is updated in one numpy call rather than per-move.
        """
        ops = []
        i   = 0
        n   = len(records)
        while i < n:
            r  = records[i]
            et = r.event_type

            if et == PRESS:
                payload  = r.payload or {}
                ras_path = [r.ras] if r.ras else []
                i += 1
                while i < n and records[i].event_type != RELEASE:
                    if records[i].event_type == MOVE and records[i].ras:
                        ras_path.append(records[i].ras)
                    i += 1
                if i < n:          # consume the RELEASE
                    i += 1
                ops.append(('stroke', r.timestamp, payload, ras_path))

            elif et == MOVE:
                if r.ras:
                    ops.append(('move', r.timestamp, r.ras))
                i += 1

            elif et == SEGMENT_CREATED:
                p = r.payload or {}
                ops.append(('segment_created', r.timestamp,
                             p.get('segment_id', ''), p.get('seg_name', '')))
                i += 1

            elif et == SEGMENT_CHANGED:
                p = r.payload or {}
                ops.append(('segment_changed', r.timestamp,
                             p.get('to_id', ''), p.get('seg_name', '')))
                i += 1

            elif et == ACTION:
                p = r.payload or {}
                ops.append(('action', r.timestamp, p.get('name', '')))
                i += 1

            else:
                i += 1  # skip SESSION_START / SESSION_STOP / VOLUME_CHANGED

        # Sentinel to trigger _finish
        last_ts = records[-1].timestamp if records else __import__('datetime').datetime.now()
        ops.append(('done', last_ts))
        return ops

    # ------------------------------------------------------------------ #
    # Step loop                                                             #
    # ------------------------------------------------------------------ #

    def _step(self):
        if self._op_idx >= len(self._ops):
            self._finish()
            return

        op   = self._ops[self._op_idx]
        kind = op[0]
        ts   = op[1]
        self._op_idx += 1

        try:
            if kind == 'move':
                self._do_move(op[2])
            elif kind == 'stroke':
                self._do_stroke(op[2], op[3])
            elif kind == 'segment_created':
                self._do_segment_created(op[2], op[3])
            elif kind == 'segment_changed':
                self._do_segment_changed(op[2])
            elif kind == 'action':
                self._do_action(op[2])
            elif kind == 'done':
                self._finish()
                return
        except Exception as exc:
            log.error('[ReplayEngine] op %s failed: %s', kind, exc)

        # Schedule the next step at the original inter-event delay
        if self._op_idx < len(self._ops):
            next_ts  = self._ops[self._op_idx][1]
            delay_ms = max(self._MIN_STEP_MS,
                           int((next_ts - ts).total_seconds() * 1000))
            self._timer.start(delay_ms)
        else:
            self._finish()

    def _finish(self):
        if self._timer:
            self._timer.stop()
            self._timer = None
        log.debug('[ReplayEngine] finished')
        self._notify_done()

    def _notify_done(self):
        cb, self._on_done = self._on_done, None
        if callable(cb):
            cb()

    # ------------------------------------------------------------------ #
    # Op handlers                                                           #
    # ------------------------------------------------------------------ #

    def _do_move(self, ras):
        try:
            ch = slicer.util.getNode('Crosshair')
            ch.SetCursorPositionRAS(ras)
            slicer.modules.markups.logic().JumpSlicesToLocation(
                ras[0], ras[1], ras[2], True
            )
        except Exception:
            pass

    def _do_stroke(self, payload, ras_path):
        """Paint / erase a brush stroke using direct numpy mask editing."""
        if not ras_path:
            return

        seg_id_orig  = payload.get('segment_id')
        tool         = payload.get('tool')
        axis         = payload.get('axis')
        slice_idx    = payload.get('slice_idx')
        brush_radius = payload.get('brush_radius_mm')

        if tool not in ('brush', 'erase'):
            return
        if axis is None or slice_idx is None:
            return
        if not seg_id_orig or seg_id_orig not in self._seg_id_map:
            return
        if not brush_radius or brush_radius <= 0:
            return

        seg_id   = self._seg_id_map[seg_id_orig]
        seg_node = self._replay_seg
        vol_node = self._vol_node

        # Ensure the replay seg node is still in the scene
        if not slicer.mrmlScene.IsNodePresent(seg_node):
            log.warning('[ReplayEngine] replay seg node removed — stopping')
            self.stop()
            return

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            seg_node, seg_id, vol_node
        ).copy()

        xc, yc = AXIS_TO_XY_COLS[axis]
        sp = self._spacing
        rx = brush_radius / max(sp[xc], 1e-9)
        ry = brush_radius / max(sp[yc], 1e-9)

        # Build a 2D view of the target slice (mutable via assignment to mask)
        idx = [slice(None), slice(None), slice(None)]
        idx[axis] = slice_idx
        sl = mask[tuple(idx)]
        H, W = sl.shape
        # Jump slice views to the stroke position before painting
        start_ras = next((r for r in ras_path if r is not None), None)
        if start_ras is not None:
            self._do_move(start_ras)

        ys, xs = np.ogrid[0:H, 0:W]

        for ras in ras_path:
            if ras is None:
                continue
            ijk = ras_to_ijk_3d(self._ras_to_ijk, ras)
            px, py = ijk[xc], ijk[yc]
            if not (0 <= px < W and 0 <= py < H):
                continue
            disk = ((xs - px) / max(rx, 0.5)) ** 2 + ((ys - py) / max(ry, 0.5)) ** 2 <= 1
            if tool == 'brush':
                sl[disk] = 1
            else:
                sl[disk] = 0

        slicer.util.updateSegmentBinaryLabelmapFromArray(
            mask, seg_node, seg_id, vol_node
        )

    def _do_segment_created(self, original_id: str, seg_name: str):
        if not original_id:
            return
        name   = seg_name or original_id
        new_id = self._replay_seg.GetSegmentation().AddEmptySegment(name)
        self._seg_id_map[original_id] = new_id
        log.debug('[ReplayEngine] segment_created %s → %s', original_id, new_id)

    def _do_segment_changed(self, original_id: str):
        if original_id in self._seg_id_map:
            log.debug('[ReplayEngine] segment_changed → %s',
                      self._seg_id_map[original_id])

    _ACTION_MAP = {
        'onUndo':   'onUndo',
        'onRedo':   'onRedo',
        'onExpand': '_onExpand',
    }

    def _do_action(self, name: str):
        mapped = self._ACTION_MAP.get(name)
        if not mapped:
            return
        method = getattr(self._widget, mapped, None)
        if callable(method):
            try:
                method()
            except Exception as exc:
                log.error('[ReplayEngine] action %s failed: %s', name, exc)
