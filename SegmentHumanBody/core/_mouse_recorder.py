"""Mouse-centered timestamped input recorder.

Mouse events are recorded only when they occur inside a Slicer slice view and
the event VTK device XY maps inside the active volume through the same
XY-to-IJK route used by Slicer DataProbe. UI-panel movement is therefore
ignored. The recording is not a general UI macro; raw export captures original
mouse/source facts only. Semantic reconstruction belongs to TimeLogInterpreter.
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
SEGMENT_SELECTED       = 'segment_selected'
VOLUME_CHANGED         = 'volume_changed'
MODEL_FAMILY_CHANGED   = 'model_family_changed'
MODEL_VARIANT_CHANGED  = 'model_variant_changed'
MODEL_CONFIRMED        = 'model_confirmed'
BRUSH_DIAMETER_CHANGED = 'brush_diameter_changed'
BRUSH_SPHERE_CHANGED   = 'brush_sphere_changed'
BRUSH_PARAMETERS_CHANGED = 'brush_parameters_changed'
BRUSH_CLICK            = 'brush_click'
POINT_PLACED           = 'point_placed'
POINT_REPLACED         = 'point_replaced'
POINT_REMOVED          = 'point_removed'
VIEW_CHANGED           = 'view_changed'
POINT_DRAG_START       = 'point_drag_start'
POINT_DRAG_MOVE        = 'point_drag_move'
POINT_DRAG_END         = 'point_drag_end'
TOOL_SELECTED          = 'tool_selected'
SPX_BRUSH_FILL         = 'spx_brush_fill'
SPX_ERASE_FILL         = 'spx_erase_fill'
FILL_HOLE              = 'fill_hole'
SPX_BOUNDARY_TOGGLED   = 'spx_boundary_toggled'
OVERWRITE_MODE_CHANGED = 'overwrite_mode_changed'

EXPORT_TYPE            = 'SegmentHumanBody.annotation_process'
VTK_OBSERVER_PRIORITY  = 1000.0

MOUSE_EVENT            = 'mouse'
MOUSE_PRESS            = 'press'
MOUSE_HOLD             = 'hold'
MOUSE_RELEASE          = 'release'
MOUSE_IDLE             = 'idle'
BOUNDARY_EVENT         = 'boundary_event'
TRAJECTORY_EVENT       = 'trajectory_event'

MOVE_ANNOTATION_TARGET_IJK = 0.5
MOVE_HOVER_TARGET_IJK      = 2.0
MOVE_ANNOTATION_PIXEL_CLAMP = (1, 4)
MOVE_HOVER_PIXEL_CLAMP      = (2, 12)
MOVE_ANNOTATION_MAX_INTERVAL_MS = 100.0
MOVE_HOVER_MAX_INTERVAL_MS      = 250.0


def _volume_metadata(volume_node) -> dict | None:
    if volume_node is None:
        return None
    dims = list(volume_node.GetImageData().GetDimensions())
    spacing = list(volume_node.GetSpacing())
    mat = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(mat)
    ijk_to_ras = [mat.GetElement(r, c) for r in range(4) for c in range(4)]
    inv_mat = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(mat, inv_mat)
    ras_to_ijk = [inv_mat.GetElement(r, c) for r in range(4) for c in range(4)]
    return {
        'name': volume_node.GetName(),
        'dimensions': dims,
        'spacing': spacing,
        'ijk_to_ras': ijk_to_ras,
        'ras_to_ijk': ras_to_ijk,
    }


_QObjectBase = getattr(qt, 'QObject', object)


class _SliceResizeFilter(_QObjectBase):
    """Qt event filter that marks a slice view dirty when it is resized."""

    def __init__(self, view_name, on_modified):
        super().__init__()
        self._view_name = view_name
        self._on_modified = on_modified

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.Resize:
            self._on_modified(self._view_name)
        return False


class _SliceRecordInteractorObserver:
    """Per-slice-view VTK interactor observer using DataProbe device XY."""

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
        return bool(self._tags)

    def remove(self):
        if self._interactor is not None:
            for tag in self._tags:
                self._interactor.RemoveObserver(tag)
        self._interactor = None
        self._tags = []
        self._left_down = False

    def _on_event(self, caller, event_name):
        try:
            x, y = caller.GetEventPosition()
            xy = [int(x), int(y)]
            if event_name == 'MouseMoveEvent':
                self._on_mouse(self._view_name, xy, MOVE, {
                    'mouse_status': 'move',
                    'left_button_down': self._left_down or _left_button_is_down(),
                    'input_source': 'vtk_interactor',
                    'xy_source': 'vtk_device',
                    'analysis_event_type': 'trajectory_event',
                })
            elif event_name == 'LeftButtonPressEvent':
                self._left_down = True
                self._on_mouse(self._view_name, xy, PRESS, {
                    'mouse_status': 'press',
                    'left_button_down': True,
                    'input_source': 'vtk_interactor',
                    'xy_source': 'vtk_device',
                    'analysis_event_type': 'boundary_event',
                })
            elif event_name == 'LeftButtonReleaseEvent':
                self._on_mouse(self._view_name, xy, RELEASE, {
                    'mouse_status': 'release',
                    'left_button_down': False,
                    'input_source': 'vtk_interactor',
                    'xy_source': 'vtk_device',
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
                    'xy_source': 'vtk_device',
                    'analysis_event_type': 'boundary_event',
                })
        except Exception as exc:
            log.error('[SliceRecordInteractorObserver] %s', exc)


class _SliceRecordListener:
    """Owns the single DataProbe-compatible input backend for a slice view."""

    def __init__(self, view_name, slice_view, on_mouse):
        self._view_name = view_name
        self._slice_view = slice_view
        self._on_mouse = on_mouse
        self._backend = None
        self._observer = None

    @property
    def backend(self):
        return self._backend

    def install(self):
        observer = _SliceRecordInteractorObserver(
            self._view_name, self._slice_view, self._on_mouse)
        if observer.install():
            self._observer = observer
            self._backend = 'vtk_interactor'
            return True
        return False

    def remove(self):
        if self._observer is not None:
            self._observer.remove()
            self._observer = None
        self._backend = None


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
    SEGMENT_SELECTED       = SEGMENT_SELECTED
    VOLUME_CHANGED         = VOLUME_CHANGED
    MODEL_FAMILY_CHANGED   = MODEL_FAMILY_CHANGED
    MODEL_VARIANT_CHANGED  = MODEL_VARIANT_CHANGED
    MODEL_CONFIRMED        = MODEL_CONFIRMED
    BRUSH_DIAMETER_CHANGED = BRUSH_DIAMETER_CHANGED
    BRUSH_SPHERE_CHANGED   = BRUSH_SPHERE_CHANGED
    BRUSH_PARAMETERS_CHANGED = BRUSH_PARAMETERS_CHANGED
    BRUSH_CLICK            = BRUSH_CLICK
    POINT_PLACED           = POINT_PLACED
    POINT_REPLACED         = POINT_REPLACED
    POINT_REMOVED          = POINT_REMOVED
    VIEW_CHANGED           = VIEW_CHANGED
    POINT_DRAG_START       = POINT_DRAG_START
    POINT_DRAG_MOVE        = POINT_DRAG_MOVE
    POINT_DRAG_END         = POINT_DRAG_END

    def __init__(self, sample_rate_hz: int = 60):
        if sample_rate_hz <= 0:
            raise ValueError(f'sample_rate_hz must be positive, got {sample_rate_hz}')
        self._records: list = []
        self._listeners: list = []
        self._active = False
        self._paused = False
        self._move_interval_ms = 1000.0 / sample_rate_hz
        self._move_timer = None
        self._pending_move_sample = None
        self._last_recorded_move_state = None
        self._last_sampled_move_key = None
        self._last_boundary_key = None
        self._last_mouse_boundary_by_view = {}
        self._active_mouse_press = False
        self._last_point_drag_ts: datetime.datetime | None = None
        self._active_brush_stroke_params = None
        self._pending_brush_params_record = None
        self._volume_node = None
        self._active_region_gate = None
        self._bound_tool_context = {'tool': None}
        self.context_fn = None
        self.on_record_appended = None
        self._next_event_id = 1
        self._slice_node_tags: dict = {}
        self._resize_filters: dict = {}
        self._dirty_views: set = set()
        self._refresh_scheduled: bool = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_paused(self) -> bool:
        return self._active and self._paused

    def pause(self) -> None:
        if self._active:
            self._paused = True

    def resume(self) -> None:
        self._paused = False

    def start(self, volume_node=None, segmentation_name: str | None = None,
              volume_sequences: list | None = None,
              initial_overwrite_mode: dict | None = None):
        if self._active:
            return
        self._active = True
        self._volume_node = volume_node
        self._active_mouse_press = _left_button_is_down()
        self._last_recorded_move_state = None
        for view_name in ('Red', 'Green', 'Yellow'):
            view = _slice_view(view_name)
            if view is None:
                continue
            listener = _SliceRecordListener(view_name, view, self._on_mouse)
            if listener.install():
                self._listeners.append(listener)
        for view_name in ('Red', 'Green', 'Yellow'):
            sw = _slice_widget(view_name)
            if sw is None:
                continue
            node = sw.mrmlSliceNode()
            if node is None:
                continue
            cb = lambda c, e, vn=view_name: self._on_slice_node_modified(vn)
            tag = node.AddObserver(vtk.vtkCommand.ModifiedEvent, cb)
            self._slice_node_tags[view_name] = (node, tag, cb)
        for view_name in ('Red', 'Green', 'Yellow'):
            sw = _slice_widget(view_name)
            if sw is None:
                continue
            flt = _SliceResizeFilter(view_name, self._on_slice_node_modified)
            sw.installEventFilter(flt)
            self._resize_filters[view_name] = (sw, flt)
        if self._listeners:
            self._start_move_timer()
        hz = round(1000.0 / self._move_interval_ms)
        initial_visual_state = _all_slice_visual_state(volume_node)
        initial_handler_context = self._all_handler_context()
        self._bound_tool_context = _initial_bound_tool_context(
            initial_handler_context)
        metadata = {
            'volume': _volume_metadata(volume_node),
            'volume_sequences': list(volume_sequences or []),
            'segmentation': segmentation_name,
            'sample_rate_hz': hz,
            'move_thinning': _move_thinning_metadata(),
            'capture_scope': 'slice-view-in-volume',
            'recorder_style': 'event_listener',
            'started_with_left_button_down': self._active_mouse_press,
            'initial_visual_state': initial_visual_state,
            'initial_handler_context': initial_handler_context,
            'mouse_xy_system': 'vtk_device_dataprobe',
            'interpreted_coordinate_system': 'IJK',
            'mouse_state_terms': [MOUSE_PRESS, MOUSE_HOLD,
                                  MOUSE_RELEASE, MOUSE_IDLE],
            'coordinate_sources': {
                'cursor': (
                    'VTK interactor GetEventPosition device XY interpreted '
                    'with the per-view xy_to_ijk matrices stored in '
                    'metadata.initial_visual_state and updated via '
                    'view_changed events with visual_state during recording.'
                ),
            },
            'initial_overwrite_mode': (
                initial_overwrite_mode
                or {'mode': 'OverwriteNone', 'mode_label': ''}
            ),
        }
        self._active_region_gate = _ActiveRegionGate(metadata)
        self._append(METADATA, None, metadata)
        log.debug('[MouseEventRecorder] started with %d slice listeners',
                  len(self._listeners))

    def stop(self):
        self._sample_pending_move()
        self._stop_move_timer()
        self._flush_dirty_views()
        for view_name, (node, tag, _cb) in self._slice_node_tags.items():
            node.RemoveObserver(tag)
        self._slice_node_tags.clear()
        for view_name, (sw, flt) in self._resize_filters.items():
            sw.removeEventFilter(flt)
        self._resize_filters.clear()
        for listener in self._listeners:
            listener.remove()
        self._listeners = []
        self._active = False
        log.debug('[MouseEventRecorder] stopped with %d records', len(self._records))

    def clear(self):
        self._records.clear()
        self._pending_move_sample = None
        self._last_recorded_move_state = None
        self._last_sampled_move_key = None
        self._last_boundary_key = None
        self._last_mouse_boundary_by_view = {}
        self._active_mouse_press = False
        self._last_point_drag_ts = None
        self._active_brush_stroke_params = None
        self._pending_brush_params_record = None
        self._active_region_gate = None
        self._bound_tool_context = {'tool': None}
        self._next_event_id = 1
        self._dirty_views.clear()
        self._refresh_scheduled = False

    def record_action(self, name: str):
        self._append(ACTION, None, {'name': name})

    def record_tool_selected(self, tool: str | None,
                             segment_id: str | None = None):
        if not self._active:
            return
        self._bound_tool_context = {'tool': tool}
        if segment_id is not None:
            self._bound_tool_context['segment_id'] = segment_id
        payload = {'tool': tool, 'analysis_event_type': BOUNDARY_EVENT}
        if segment_id is not None:
            payload['segment_id'] = segment_id
        self._append(TOOL_SELECTED, None, payload)

    def record_segment_created(self, segment_id: str, seg_name: str):
        payload = self._source_context_payload()
        payload.update({
            'segment_id': segment_id,
            'seg_name': seg_name,
            'analysis_event_type': BOUNDARY_EVENT,
        })
        self._append(SEGMENT_CREATED, None, payload)

    def record_volume_changed(self, volume_name: str | None,
                              volume_id: str | None = None,
                              sequence_index: int | None = None):
        payload = self._source_context_payload()
        payload.update({
            'volume': volume_name,
            'analysis_event_type': BOUNDARY_EVENT,
        })
        if volume_id is not None:
            payload['volume_id'] = volume_id
        if sequence_index is not None:
            payload['sequence_index'] = int(sequence_index)
        self._append(VOLUME_CHANGED, None, payload)

    def set_volume_node(self, volume_node):
        self._volume_node = volume_node
        self._update_metadata_volume(volume_node)
        self._active_region_gate = _ActiveRegionGate({
            'volume': _volume_metadata(volume_node),
            'initial_visual_state': _all_slice_visual_state(volume_node),
        })

    def refresh_visual_state(self):
        """Re-sample xy_to_ijk matrices after slice views finish updating."""
        visual_state = _all_slice_visual_state(self._volume_node)
        for idx, record in enumerate(self._records):
            if record.event_type != METADATA:
                continue
            payload = dict(record.payload or {})
            payload['initial_visual_state'] = visual_state
            self._records[idx] = record._replace(payload=payload)
            self._active_region_gate = _ActiveRegionGate({
                'volume': payload.get('volume'),
                'initial_visual_state': visual_state,
            })
            return

    def _update_metadata_volume(self, volume_node):
        volume = _volume_metadata(volume_node)
        visual_state = _all_slice_visual_state(volume_node)
        for idx, record in enumerate(self._records):
            if record.event_type != METADATA:
                continue
            payload = dict(record.payload or {})
            payload['volume'] = volume
            payload['initial_visual_state'] = visual_state
            self._records[idx] = record._replace(payload=payload)
            return

    def record_model_family_changed(self, family: str):
        self._append(MODEL_FAMILY_CHANGED, None, {'family': family})

    def record_model_variant_changed(self, variant: str):
        self._append(MODEL_VARIANT_CHANGED, None, {'variant': variant})

    def record_model_confirmed(self, family: str, variant: str):
        self._append(MODEL_CONFIRMED, None, {
            'family': family, 'variant': variant})

    def record_overwrite_mode_changed(self, mode_label: str, mode_key: str):
        self._append(OVERWRITE_MODE_CHANGED, None,
                     {'mode': mode_key, 'mode_label': mode_label})

    def record_spx_fill(self, *, additive, label_id, view, axis, slice_idx,
                        delta_pixels, model_key, params, segment_id,
                        segmentation_id, volume_id):
        event_type = SPX_BRUSH_FILL if additive else SPX_ERASE_FILL
        payload = {
            'operation': 'spx_expand',
            'additive': bool(additive),
            'label_id': int(label_id),
            'view': view,
            'axis': axis,
            'slice_idx': int(slice_idx),
            'delta_pixels': int(delta_pixels) if delta_pixels is not None else None,
            'model_key': model_key,
            'params': dict(params or {}),
            'segment_id': segment_id,
            'segmentation_id': segmentation_id,
            'volume_id': volume_id,
            'analysis_event_type': BOUNDARY_EVENT,
        }
        self._append(event_type, None, payload)

    def record_fill_hole(self, *, view, axis, slice_idx, delta_pixels,
                         segment_id, segmentation_id, volume_id):
        payload = {
            'view': view,
            'axis': axis,
            'slice_idx': int(slice_idx),
            'delta_pixels': int(delta_pixels) if delta_pixels is not None else None,
            'segment_id': segment_id,
            'segmentation_id': segmentation_id,
            'volume_id': volume_id,
            'analysis_event_type': BOUNDARY_EVENT,
        }
        self._append(FILL_HOLE, None, payload)

    def record_spx_boundary_toggled(self, *, visible, view, slice_idx, model_key):
        payload = {
            'visible': bool(visible),
            'view': view,
            'slice_idx': int(slice_idx) if slice_idx is not None else None,
            'model_key': model_key,
            'analysis_event_type': BOUNDARY_EVENT,
        }
        self._append(SPX_BOUNDARY_TOGGLED, None, payload)

    def record_brush_diameter_changed(self, diameter_mm: float):
        payload = self._current_brush_param_payload()
        payload['diameter_mm'] = float(diameter_mm)
        payload['brush_radius_mm'] = float(diameter_mm) / 2.0
        self._record_or_buffer_brush_params(payload)

    def record_brush_sphere_changed(self, sphere: bool):
        payload = self._current_brush_param_payload()
        payload['sphere'] = bool(sphere)
        self._record_or_buffer_brush_params(payload)

    def record_segment_removed(self, segment_id: str, seg_name: str):
        payload = self._source_context_payload()
        payload.update({
            'segment_id': segment_id,
            'seg_name': seg_name,
            'analysis_event_type': BOUNDARY_EVENT,
        })
        self._append(SEGMENT_REMOVED, None, payload)

    def record_segment_renamed(self, segment_id: str, old_name: str, new_name: str):
        payload = self._source_context_payload()
        payload.update({
            'segment_id': segment_id,
            'old_name': old_name,
            'new_name': new_name,
            'analysis_event_type': BOUNDARY_EVENT,
        })
        self._append(SEGMENT_RENAMED, None, payload)

    def record_segment_selected(self, segment_id: str | None, seg_name: str | None = None):
        self._bound_tool_context['segment_id'] = segment_id
        payload = self._source_context_payload()
        payload.update({
            'segment_id': segment_id,
            'analysis_event_type': BOUNDARY_EVENT,
        })
        if seg_name:
            payload['seg_name'] = seg_name
        self._append(SEGMENT_SELECTED, None, payload)

    def record_point_placed(self, segment_id: str, ras: list, is_negative: bool,
                            view_name: str | None = None, point_index=None,
                            point_id: str | None = None,
                            point_name: str | None = None):
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
            self._attach_last_mouse_boundary(payload, view_name, RELEASE)
        payload['analysis_event_type'] = 'boundary_event'
        payload['mouse_status'] = 'release'
        payload['point_action'] = 'place'
        payload['ras_source'] = 'markup_world'
        payload['trajectory_kind'] = None
        payload['trajectory_role'] = None
        self._append(POINT_PLACED, list(ras), payload)

    def record_point_removed(self, segment_id: str, ras: list | None,
                             is_negative: bool, view_name: str | None = None,
                             point_index=None, point_id: str | None = None,
                             point_name: str | None = None,
                             mouse_pressed: bool = False):
        boundary = PRESS if mouse_pressed else RELEASE
        payload = {
            'segment_id': segment_id,
            'is_negative': bool(is_negative),
            'handler': 'point',
            'handler_params': {},
            'analysis_event_type': 'boundary_event',
            'mouse_status': 'press' if mouse_pressed else 'release',
            'mouse_pressed': bool(mouse_pressed),
            'point_action': 'remove',
            'ras_source': 'markup_world',
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
            self._attach_last_mouse_boundary(payload, view_name, boundary)
        self._append(POINT_REMOVED, list(ras) if ras is not None else None, payload)

    def record_point_drag(self, phase: str, segment_id: str, ras: list,
                          is_negative: bool, view_name: str | None = None,
                          point_index=None, point_id: str | None = None,
                          point_name: str | None = None):
        event_type = {
            'start': POINT_DRAG_START,
            'move': POINT_DRAG_MOVE,
            'end': POINT_REPLACED,
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
            'ras_source': 'markup_world',
            'point_action': {
                POINT_DRAG_START: 'grab',
                POINT_DRAG_MOVE: 'move',
                POINT_REPLACED: 'replace',
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
            boundary = RELEASE if event_type == POINT_REPLACED else PRESS
            self._attach_last_mouse_boundary(payload, view_name, boundary)
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
        base = path[:-5] if path.endswith('.json') else path
        raw_data = self.export_raw_data()
        from core.TimeLogInterpreter import TimeLogInterpreter
        interpreted_data = TimeLogInterpreter(raw_data).export()
        from core.TimeLogSummarizer import TimeLogSummarizer
        summary_text = TimeLogSummarizer(interpreted_data).export_text()
        with open(base + '.json', 'w', encoding='utf-8') as f:
            json.dump(interpreted_data, f, indent=2)
        with open(base + '_raw.json', 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2)
        with open(base + '_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary_text)
            if summary_text:
                f.write('\n')

    def _export_metadata_block(self):
        """Return (metadata_record, metadata_dict) with start_time already set."""
        metadata_record = next(
            (r for r in self._records if r.event_type == METADATA), None)
        start_ts = (
            metadata_record.timestamp if metadata_record is not None
            else (self._records[0].timestamp if self._records
                  else datetime.datetime.now())
        )
        metadata = dict(metadata_record.payload or {}) if metadata_record else {}
        metadata['start_time'] = start_ts.isoformat(timespec='milliseconds')
        return metadata_record, metadata

    def export_interpreted_data(self) -> dict:
        """Semantic process log derived only from the exported raw log."""
        from core.TimeLogInterpreter import TimeLogInterpreter
        return TimeLogInterpreter(self.export_raw_data()).export()

    def export_raw_data(self) -> dict:
        """Raw source log.

        Mouse entries keep only original device XY, view, slice, button state,
        and sparse context deltas. Source entries keep tool/segment/point facts.
        RAS/IJK and semantic classification are intentionally absent here.
        """
        _, metadata = self._export_metadata_block()
        active_region = _ActiveRegionGate(metadata)
        context = _RawExportContext(metadata)
        events = []
        pending_trajectory = None
        for record in self._records:
            if record.event_type == METADATA:
                continue
            if not active_region.accepts(record):
                continue
            ev = _raw_event(record, context)
            if ev is not None:
                pending_trajectory = _append_raw_export_event(
                    events, pending_trajectory, ev)
        if pending_trajectory is not None:
            events.append(pending_trajectory)
        return {
            'type': 'SegmentHumanBody.raw_input',
            'metadata': metadata,
            'events': events,
        }

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
            if 'timestamp' in item:
                ts = datetime.datetime.fromisoformat(item['timestamp'])
            else:
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
            'ras_source': 'ras_source',
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
        metadata_record, metadata = self._export_metadata_block()
        metadata['coordinate_system'] = 'RAS'
        ras_to_ijk = _ras_to_ijk_matrix(metadata)
        events = [
            _compact_event(r, None, export_id)
            for export_id, r in enumerate(
                _records_for_compact_export(self._records),
                start=1,
            )
        ]
        if ras_to_ijk is not None:
            for ev in events:
                if 'ras' in ev:
                    ev['ijk'] = _ras_to_ijk_coords(ras_to_ijk, ev['ras'])
        return {
            'type': EXPORT_TYPE,
            'metadata': metadata,
            'events': events,
        }

    def _flush_dirty_views(self) -> None:
        self._refresh_scheduled = False
        if not self._active:
            self._dirty_views.clear()
            return
        for view_name in list(self._dirty_views):
            self._dirty_views.discard(view_name)
            new_state = _visual_state(view_name, self._volume_node)
            new_mat = new_state.get('xy_to_ijk')
            old_mat = (self._active_region_gate.xy_to_ijk_by_view.get(view_name)
                       if self._active_region_gate is not None else None)
            if new_mat is None or new_mat == old_mat:
                continue
            if self._active_region_gate is not None:
                self._active_region_gate._update_visual_state({
                    'view_name': view_name,
                    'visual_state': new_state,
                })
            if not self._paused:
                payload = {
                    'view_name': view_name,
                    'visual_state': new_state,
                    'analysis_event_type': BOUNDARY_EVENT,
                }
                if new_mat is not None and len(new_mat) == 16:
                    payload['slice_idx'] = int(round(new_mat[11]))
                payload.update(self._bound_tool_payload())
                self._append(VIEW_CHANGED, None, payload)

    def _on_slice_node_modified(self, view_name: str) -> None:
        if not self._active:
            return
        self._dirty_views.add(view_name)
        if not self._refresh_scheduled:
            self._refresh_scheduled = True
            qt.QTimer.singleShot(0, self._flush_dirty_views)

    def _on_mouse(self, view_name: str, xy_local: tuple, event_type: str, extra=None):
        if self._paused:
            self._note_mouse_button_state(event_type)
            return
        timestamp = datetime.datetime.now()
        if not self._mouse_xy_is_active(view_name, xy_local, event_type):
            self._note_mouse_button_state(event_type)
            if event_type == MOVE:
                self._pending_move_sample = None
            return
        if event_type == MOVE:
            self._capture_pending_move(timestamp, view_name, xy_local, extra)
            return
        self._sample_pending_move()
        self._append_mouse_record(timestamp, view_name, xy_local, event_type, extra)

    def _mouse_xy_is_active(self, view_name, xy_local, event_type):
        gate = self._active_region_gate
        if gate is None:
            return True
        return gate.accepts_xy(view_name, xy_local)

    def _note_mouse_button_state(self, event_type):
        if event_type == PRESS:
            self._active_mouse_press = True
        elif event_type == RELEASE:
            self._active_mouse_press = False
            self._active_brush_stroke_params = None
            self._pending_brush_params_record = None

    def _capture_pending_move(self, timestamp, view_name, xy_local, extra=None):
        payload = self._raw_mouse_payload(
            view_name, xy_local, MOVE, extra)
        self._pending_move_sample = (timestamp, view_name, None, payload)

    def _sample_pending_move(self):
        if not self._active or self._pending_move_sample is None:
            return
        timestamp, view_name, ras, payload = self._pending_move_sample
        self._pending_move_sample = None
        self._append_prepared_mouse_record(
            timestamp, view_name, ras, MOVE, payload, dedupe=True)

    def _append_mouse_record(self, timestamp, view_name, xy_local, event_type,
                             extra=None, dedupe=False):
        payload = self._raw_mouse_payload(
            view_name, xy_local, event_type, extra)
        ras = None
        self._append_prepared_mouse_record(
            timestamp, view_name, ras, event_type, payload, dedupe=dedupe)

    def _raw_mouse_payload(self, view_name, xy_local, event_type, extra=None):
        payload = {'view_name': view_name}
        payload.update(self._bound_tool_payload())
        if event_type in (PRESS, RELEASE, VIEW_CHANGED):
            payload.update(self._view_change_payload(view_name))
        if extra:
            payload.update(extra)
        payload['view_name'] = view_name
        payload['xy'] = [int(round(float(xy_local[0]))),
                         int(round(float(xy_local[1])))]
        payload['mouse_state'] = _mouse_state_from_payload(
            event_type, payload, self._active_mouse_press)
        payload['analysis_event_type'] = (
            BOUNDARY_EVENT
            if (
                event_type == VIEW_CHANGED or
                payload['mouse_state'] in (MOUSE_PRESS, MOUSE_RELEASE)
            )
            else TRAJECTORY_EVENT
        )
        if event_type == MOVE:
            payload['mouse_button_state'] = (
                'pressed'
                if payload.get('left_button_down') or self._active_mouse_press
                else 'released'
            )
        return payload

    def _view_change_payload(self, view_name):
        return self._event_payload(view_name)

    def _append_prepared_mouse_record(self, timestamp, view_name, ras, event_type,
                                      payload, dedupe=False):
        if self._is_duplicate_boundary(timestamp, view_name, ras, event_type, payload):
            return False
        if event_type == MOVE and not self._move_sample_adds_information(
                timestamp, view_name, payload):
            return False
        if dedupe:
            key = _move_sample_key(view_name, ras, payload)
            if key == self._last_sampled_move_key:
                return False
            self._last_sampled_move_key = key
        if event_type == RELEASE:
            self._flush_brush_params_before_release(timestamp, payload)
        self._records.append(self._new_record(
            timestamp, ras, event_type, payload))
        if event_type == MOVE:
            self._remember_recorded_move(timestamp, view_name, payload)
        if event_type == PRESS:
            self._active_mouse_press = True
            self._active_brush_stroke_params = _brush_param_state(payload)
            self._remember_mouse_boundary(timestamp, view_name, payload)
            self._remember_boundary(timestamp, view_name, ras, event_type, payload)
        elif event_type == RELEASE:
            self._active_mouse_press = False
            self._active_brush_stroke_params = None
            self._pending_brush_params_record = None
            self._remember_mouse_boundary(timestamp, view_name, payload)
            self._remember_boundary(timestamp, view_name, ras, event_type, payload)
        self._notify_record_appended()
        return True

    def _move_sample_adds_information(self, timestamp, view_name, payload):
        previous = self._last_recorded_move_state
        if previous is None:
            return True
        xy = _move_xy_tuple(payload)
        if xy is None:
            return True
        pressed = _move_pressed(payload)
        context = _move_context_signature(view_name, payload)
        if previous['view_name'] != view_name:
            return True
        if previous['pressed'] != pressed:
            return True
        if previous['context'] != context:
            return True
        dx = xy[0] - previous['xy'][0]
        dy = xy[1] - previous['xy'][1]
        threshold = self._move_pixel_threshold(view_name, pressed)
        if (dx * dx + dy * dy) >= threshold * threshold:
            return True
        elapsed_ms = (timestamp - previous['timestamp']).total_seconds() * 1000.0
        max_interval_ms = _move_max_interval_ms(pressed)
        return elapsed_ms >= max_interval_ms

    def _move_pixel_threshold(self, view_name, pressed):
        gate = self._active_region_gate
        if gate is not None:
            return gate.move_pixel_threshold(view_name, pressed)
        return _move_pixel_threshold_from_scale(None, pressed)

    def _remember_recorded_move(self, timestamp, view_name, payload):
        xy = _move_xy_tuple(payload)
        if xy is None:
            return
        self._last_recorded_move_state = {
            'timestamp': timestamp,
            'view_name': view_name,
            'xy': xy,
            'pressed': _move_pressed(payload),
            'context': _move_context_signature(view_name, payload),
        }

    def _is_duplicate_boundary(self, timestamp, view_name, ras, event_type, payload):
        return False

    def _remember_boundary(self, timestamp, view_name, ras, event_type, payload):
        self._last_boundary_key = (
            _boundary_key(view_name, ras, event_type, payload),
            timestamp,
        )

    def _remember_mouse_boundary(self, timestamp, view_name, payload):
        if not view_name or payload.get('xy') is None:
            return
        entry = {
            'timestamp': timestamp.isoformat(timespec='milliseconds'),
            'mouse_state': payload.get('mouse_state'),
            'xy': list(payload.get('xy')),
        }
        self._last_mouse_boundary_by_view[view_name] = entry

    def _attach_last_mouse_boundary(self, payload, view_name, expected_state):
        boundary = self._last_mouse_boundary_by_view.get(view_name)
        if not boundary or boundary.get('mouse_state') != expected_state:
            return
        payload['mouse_boundary'] = dict(boundary)

    def _start_move_timer(self):
        if self._move_timer is not None:
            return
        timer = qt.QTimer()
        timer.connect('timeout()', self._sample_pending_move)
        timer.start(int(round(self._move_interval_ms)))
        self._move_timer = timer

    def _stop_move_timer(self):
        timer = self._move_timer
        self._move_timer = None
        if timer is not None:
            timer.stop()
        self._pending_move_sample = None
        self._last_sampled_move_key = None
        self._last_boundary_key = None

    def _event_payload(self, view_name):
        payload = (self.context_fn(view_name=view_name) or {}) if callable(self.context_fn) else {}
        payload.setdefault('view_name', view_name)
        tool = payload.get('tool')
        payload['handler'] = tool
        payload['handler_params'] = _handler_params(payload)
        payload.setdefault('analysis_event_type', None)
        return payload

    def _bound_tool_payload(self):
        payload = dict(self._bound_tool_context)
        if 'tool' in payload:
            payload['handler'] = payload.get('tool')
        return payload

    def _source_context_payload(self):
        if not callable(self.context_fn):
            return {}
        return self._event_payload(None)

    def _all_handler_context(self):
        if not callable(self.context_fn):
            return {}
        return {
            view_name: self._event_payload(view_name)
            for view_name in ('Red', 'Green', 'Yellow')
        }

    def _current_brush_param_payload(self):
        payload = self._event_payload(None) if callable(self.context_fn) else {}
        payload['event_source'] = 'brush_parameters'
        payload['analysis_event_type'] = 'boundary_event'
        return payload

    def _record_or_buffer_brush_params(self, payload):
        timestamp = datetime.datetime.now()
        payload = _brush_param_event_payload(payload)
        if self._active_mouse_press:
            self._pending_brush_params_record = (timestamp, payload)
            return
        self._records.append(self._new_record(
            timestamp, None, BRUSH_PARAMETERS_CHANGED, payload))
        self._notify_record_appended()

    def _flush_brush_params_before_release(self, timestamp, release_payload):
        pending = self._pending_brush_params_record
        final_payload = _brush_param_event_payload(release_payload)
        start = self._active_brush_stroke_params
        final_state = _brush_param_state(final_payload)
        should_emit_final = (
            final_state is not None and start is not None and
            final_state != start
        )
        if pending is not None:
            pending_timestamp, pending_payload = pending
            payload = dict(pending_payload or {})
            if final_state is not None:
                payload.update(final_payload)
            self._records.append(self._new_record(
                pending_timestamp, None, BRUSH_PARAMETERS_CHANGED, payload))
            self._notify_record_appended()
            return
        if should_emit_final:
            self._records.append(self._new_record(
                timestamp, None, BRUSH_PARAMETERS_CHANGED, final_payload))
            self._notify_record_appended()

    def _append(self, event_type, ras, payload):
        if self._paused:
            return
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
            self.on_record_appended()

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        state = 'active' if self.is_active else 'stopped'
        hz = round(1000.0 / self._move_interval_ms)
        return f'MouseEventRecorder({state}, {len(self._records)} records, {hz} Hz)'


_recorder: MouseEventRecorder | None = None


def get_recorder(sample_rate_hz: int = 60) -> MouseEventRecorder:
    global _recorder
    if _recorder is None:
        _recorder = MouseEventRecorder(sample_rate_hz=sample_rate_hz)
    return _recorder


def _ras_to_ijk_matrix(metadata):
    vol = (metadata or {}).get('volume') or {}
    m = vol.get('ras_to_ijk')
    return m if (m is not None and len(m) == 16) else None


def _initial_bound_tool_context(initial_handler_context):
    for view_name in ('Red', 'Green', 'Yellow'):
        payload = (initial_handler_context or {}).get(view_name) or {}
        tool = payload.get('handler') or payload.get('tool')
        if tool is None:
            continue
        context = {'tool': tool}
        if payload.get('segment_id') is not None:
            context['segment_id'] = payload.get('segment_id')
        return context
    return {'tool': None}


def _volume_dims_from_metadata(metadata):
    dims = ((metadata or {}).get('volume') or {}).get('dimensions')
    return dims if dims is not None and len(dims) >= 3 else None


def _volume_spacing_from_metadata(metadata):
    spacing = ((metadata or {}).get('volume') or {}).get('spacing')
    return spacing if spacing is not None and len(spacing) >= 3 else None


def _ras_to_ijk_coords(ras_to_ijk, ras):
    return _ijk_export_coords(_ras_to_ijk_float(ras_to_ijk, ras))


def _ijk_export_coords(ijk):
    return [int(round(float(v))) for v in ijk[:3]]


def _ras_to_ijk_float(ras_to_ijk, ras):
    m = ras_to_ijk
    rx, ry, rz = float(ras[0]), float(ras[1]), float(ras[2])
    i = m[0]*rx + m[1]*ry + m[2]*rz  + m[3]
    j = m[4]*rx + m[5]*ry + m[6]*rz  + m[7]
    k = m[8]*rx + m[9]*ry + m[10]*rz + m[11]
    return [i, j, k]


def _xy_to_ijk_from_matrix(xy, mat16):
    if xy is None or mat16 is None or len(mat16) != 16:
        return None
    x, y = float(xy[0]), float(xy[1])
    return [
        mat16[0] * x + mat16[1] * y + mat16[3],
        mat16[4] * x + mat16[5] * y + mat16[7],
        mat16[8] * x + mat16[9] * y + mat16[11],
    ]


def _mouse_state_from_payload(event_type, payload, active_mouse_press):
    if event_type == PRESS:
        return MOUSE_PRESS
    if event_type == RELEASE:
        return MOUSE_RELEASE
    if event_type == MOVE:
        pressed = (
            payload.get('left_button_down') or
            payload.get('mouse_button_state') == 'pressed' or
            active_mouse_press
        )
        return MOUSE_HOLD if pressed else MOUSE_IDLE
    return MOUSE_HOLD if (
        payload.get('left_button_down') or active_mouse_press
    ) else MOUSE_IDLE


def _ras_inside_volume_matrix(ras, ras_to_ijk, dims):
    if ras is None:
        return False
    if ras_to_ijk is None or dims is None:
        return True
    return _ijk_inside_volume(_ras_to_ijk_float(ras_to_ijk, ras), dims)


def _ijk_inside_volume(ijk, dims):
    if ijk is None or dims is None:
        return False
    return all(0.0 <= float(ijk[i]) < int(dims[i]) for i in range(3))


def _record_inside_active_volume(record, ras_to_ijk, dims):
    payload = record.payload or {}
    if payload.get('ijk') is not None:
        return _ijk_inside_volume(payload.get('ijk'), dims)
    if record.ras is None:
        return True
    return _ras_inside_volume_matrix(record.ras, ras_to_ijk, dims)


class _ActiveRegionGate:
    """Cached XY-to-IJK active-volume gate for hot-path and raw export checks."""

    def __init__(self, metadata):
        self.ras_to_ijk = _ras_to_ijk_matrix(metadata)
        self.dims = _volume_dims_from_metadata(metadata)
        self.xy_to_ijk_by_view = {}
        initial = (metadata or {}).get('initial_visual_state') or {}
        for view_name, state in initial.items():
            self._update_visual_state({
                'view_name': view_name,
                'visual_state': state or {},
            })

    def accepts(self, record):
        payload = record.payload or {}
        if record.event_type in (VIEW_CHANGED, PRESS):
            self._update_visual_state(payload)
        if record.event_type not in (PRESS, RELEASE, MOVE, VIEW_CHANGED):
            return True
        if self.ras_to_ijk is None or self.dims is None:
            return True
        ras = record.ras
        if ras is None:
            return self.accepts_xy(payload.get('view_name'), payload.get('xy'))
        return _ras_inside_volume_matrix(ras, self.ras_to_ijk, self.dims)

    def accepts_xy(self, view_name, xy, visual_state=None):
        if visual_state is not None:
            self._update_visual_state({
                'view_name': view_name,
                'visual_state': visual_state,
            })
        if self.ras_to_ijk is None or self.dims is None:
            return True
        ijk = _xy_to_ijk_from_matrix(
            xy, self.xy_to_ijk_by_view.get(view_name))
        if ijk is None:
            return True
        return _ijk_inside_volume(ijk, self.dims)

    def move_pixel_threshold(self, view_name, pressed):
        return _move_pixel_threshold_from_scale(
            _ijk_per_xy_pixel(self.xy_to_ijk_by_view.get(view_name)),
            pressed,
        )

    def _update_visual_state(self, payload):
        view_name = payload.get('view_name')
        state = payload.get('visual_state') or {}
        xy_to_ijk = state.get('xy_to_ijk')
        if view_name and xy_to_ijk is not None:
            self.xy_to_ijk_by_view[view_name] = xy_to_ijk


class _RawExportContext:
    """Tracks already-exported raw context so events only emit deltas."""

    def __init__(self, metadata=None):
        self.slice_by_view = {}
        self.emitted_context_by_view = collections.defaultdict(dict)
        self.ras_to_ijk = _ras_to_ijk_matrix(metadata)
        initial = (metadata or {}).get('initial_handler_context') or {}
        for view_name, payload in initial.items():
            self._prime_context(view_name, payload or {})

    def _prime_context(self, view_name, payload):
        if payload.get('slice_idx') is not None:
            self.slice_by_view[view_name] = payload.get('slice_idx')
        context = self.emitted_context_by_view[view_name]
        tool = payload.get('handler') or payload.get('tool')
        if tool is not None:
            context['tool'] = tool
        if payload.get('segment_id') is not None:
            context['segment'] = payload.get('segment_id')
        diameter_mm = _brush_diameter_mm(payload)
        if diameter_mm is not None:
            context['diameter_mm'] = diameter_mm

    def note_slice(self, payload):
        view_name = payload.get('view_name')
        if view_name and payload.get('slice_idx') is not None:
            self.slice_by_view[view_name] = payload.get('slice_idx')

    def slice_for(self, payload):
        if payload.get('slice_idx') is not None:
            return payload.get('slice_idx')
        view_name = payload.get('view_name')
        return self.slice_by_view.get(view_name)

    def context_delta(self, payload):
        view_name = payload.get('view_name')
        if not view_name:
            return {}
        current = self.emitted_context_by_view[view_name]
        candidates = {}
        tool = payload.get('handler') or payload.get('tool')
        if tool is not None:
            candidates['tool'] = tool
        if payload.get('segment_id') is not None:
            candidates['segment'] = payload.get('segment_id')
        diameter_mm = _brush_diameter_mm(payload)
        if diameter_mm is not None:
            candidates['diameter_mm'] = diameter_mm
        delta = {}
        for key, value in candidates.items():
            if current.get(key) != value:
                delta[key] = value
                current[key] = value
        return delta


class _ExportContext:
    """Interprets raw XY records with cached matrices and sparse tool context."""

    def __init__(self, metadata):
        self.ras_to_ijk = _ras_to_ijk_matrix(metadata)
        self.dims = _volume_dims_from_metadata(metadata)
        self.xy_to_ijk_by_view = {}
        self.context_by_view = collections.defaultdict(dict)
        self.active_brush_strokes = {}
        initial = (metadata or {}).get('initial_visual_state') or {}
        for view_name, state in initial.items():
            mat = (state or {}).get('xy_to_ijk')
            if mat is not None and len(mat) == 16:
                self.xy_to_ijk_by_view[view_name] = mat
        initial_context = (metadata or {}).get('initial_handler_context') or {}
        for view_name, payload in initial_context.items():
            self._update_handler_context(payload or {})

    def prepare(self, record):
        payload = dict(record.payload or {})
        if record.event_type == VIEW_CHANGED:
            self._update_visual_state(payload)
            return record._replace(payload=payload)
        if record.event_type == PRESS:
            self._update_visual_state(payload)
            self._update_handler_context(payload)
            enriched = self._enrich_cursor_record(record, payload)
            self._start_brush_stroke(enriched)
            return enriched
        if record.event_type == BRUSH_PARAMETERS_CHANGED:
            self._update_handler_context(payload)
            return record._replace(payload=payload)
        if record.event_type in (
                POINT_PLACED, POINT_REPLACED, POINT_REMOVED,
                POINT_DRAG_START, POINT_DRAG_MOVE):
            return self._enrich_cursor_record(record, payload)
        if record.event_type == MOVE:
            enriched = self._enrich_cursor_record(record, payload)
            self._note_brush_move(enriched)
            return enriched
        if record.event_type == RELEASE:
            enriched = self._enrich_cursor_record(record, payload)
            return self._finish_brush_stroke(enriched)
        return record

    def _update_visual_state(self, payload):
        view_name = payload.get('view_name')
        state = payload.get('visual_state') or {}
        mat = state.get('xy_to_ijk')
        if view_name and mat is not None and len(mat) == 16:
            self.xy_to_ijk_by_view[view_name] = mat

    def _update_handler_context(self, payload):
        view_name = payload.get('view_name')
        if not view_name:
            return
        context = self.context_by_view[view_name]
        for key in ('handler', 'tool', 'segment_id', 'brush_radius_mm',
                    'axis', 'slice_idx', 'handler_params'):
            if payload.get(key) is not None:
                context[key] = payload.get(key)

    def _enrich_cursor_record(self, record, payload):
        view_name = payload.get('view_name')
        context = self.context_by_view.get(view_name, {})
        for key in ('handler', 'tool', 'segment_id', 'brush_radius_mm',
                    'axis', 'slice_idx', 'handler_params'):
            if payload.get(key) is None and context.get(key) is not None:
                payload[key] = context[key]
        if payload.get('handler') is None and payload.get('tool') is not None:
            payload['handler'] = payload.get('tool')
        if payload.get('tool') is None and payload.get('handler') is not None:
            payload['tool'] = payload.get('handler')
        if payload.get('handler_params') is None:
            payload['handler_params'] = _handler_params(payload)
        if record.event_type == MOVE:
            payload.setdefault('analysis_event_type', 'trajectory_event')
            if payload.get('trajectory_kind') is None:
                payload['trajectory_kind'] = _trajectory_kind(payload)
            if payload.get('trajectory_role') is None:
                payload['trajectory_role'] = _trajectory_role(payload)
            payload['ras_source'] = 'cursor'
        ras = record.ras
        ijk = payload.get('ijk')
        if ijk is None:
            ijk = _xy_to_ijk_from_matrix(
                payload.get('xy'), self.xy_to_ijk_by_view.get(view_name))
            if ijk is not None:
                payload['ijk'] = ijk
        if ijk is not None and not _ijk_inside_volume(ijk, self.dims):
            ras = None
            payload.pop('ijk', None)
        elif ras is not None and not _ras_inside_volume_matrix(
                ras, self.ras_to_ijk, self.dims):
            ras = None
        return record._replace(ras=ras, payload=payload)

    def _start_brush_stroke(self, record):
        payload = record.payload or {}
        view_name = payload.get('view_name')
        if view_name and _record_has_position(record) and _is_brush_tool(payload):
            self.active_brush_strokes[view_name] = {'had_annotation_move': False}

    def _note_brush_move(self, record):
        payload = record.payload or {}
        view_name = payload.get('view_name')
        stroke = self.active_brush_strokes.get(view_name)
        if (
            stroke is not None and _record_has_position(record) and
            _is_brush_tool(payload) and
            payload.get('trajectory_kind') == 'annotation_move'
        ):
            stroke['had_annotation_move'] = True

    def _finish_brush_stroke(self, record):
        payload = dict(record.payload or {})
        view_name = payload.get('view_name')
        stroke = self.active_brush_strokes.pop(view_name, None)
        if (
            stroke is not None and not stroke.get('had_annotation_move') and
            _record_has_position(record) and _is_brush_tool(payload)
        ):
            payload['analysis_event_type'] = 'boundary_event'
            payload['brush_action'] = 'click'
            return record._replace(event_type=BRUSH_CLICK, payload=payload)
        return record._replace(payload=payload)


_RAW_MOUSE_TYPES = (
    PRESS, RELEASE, MOVE, VIEW_CHANGED,
)

_RAW_SOURCE_TYPES = (
    TOOL_SELECTED,
    BRUSH_PARAMETERS_CHANGED,
    SEGMENT_CREATED,
    SEGMENT_REMOVED,
    SEGMENT_RENAMED,
    SEGMENT_SELECTED,
    VOLUME_CHANGED,
    MODEL_FAMILY_CHANGED,
    MODEL_VARIANT_CHANGED,
    MODEL_CONFIRMED,
    POINT_PLACED,
    POINT_REPLACED,
    POINT_REMOVED,
    POINT_DRAG_START,
    POINT_DRAG_MOVE,
    SPX_BRUSH_FILL,
    SPX_ERASE_FILL,
    FILL_HOLE,
    SPX_BOUNDARY_TOGGLED,
    OVERWRITE_MODE_CHANGED,
)

_INTERPRETED_TYPE_MAP = {
    POINT_PLACED:    'point_placement',
    POINT_REPLACED:  'point_move',
    POINT_REMOVED:   'point_removed',
    BRUSH_PARAMETERS_CHANGED: 'brush_parameters',
    BRUSH_CLICK:     'brush_click',
}

_POINT_VERDICT_TYPES = (POINT_PLACED, POINT_REPLACED, POINT_REMOVED)


def _record_has_position(record):
    payload = record.payload or {}
    return record.ras is not None or payload.get('ijk') is not None


def _get_brush_radius_mm(payload):
    mm = payload.get('brush_radius_mm')
    if mm is None:
        mm = (payload.get('handler_params') or {}).get('brush_radius_mm')
    return mm


def _is_brush_tool(payload):
    tool = (payload or {}).get('handler') or (payload or {}).get('tool')
    return tool in ('brush', 'erase')


def _brush_param_state(payload):
    if not payload:
        return None
    tool = payload.get('handler') or payload.get('tool')
    if tool not in ('brush', 'erase'):
        return None
    state = {'tool': tool}
    diameter_mm = _brush_diameter_mm(payload)
    if diameter_mm is not None:
        state['diameter_mm'] = diameter_mm
    if payload.get('sphere') is not None:
        state['sphere'] = bool(payload['sphere'])
    return state if len(state) > 1 else None


def _brush_param_event_payload(payload):
    payload = dict(payload or {})
    tool = payload.get('handler') or payload.get('tool')
    if tool in ('brush', 'erase'):
        payload['handler'] = tool
        payload['tool'] = tool
    if payload.get('diameter_mm') is None:
        radius_mm = _get_brush_radius_mm(payload)
        if radius_mm is not None:
            payload['diameter_mm'] = float(radius_mm) * 2.0
    payload['analysis_event_type'] = 'boundary_event'
    payload['event_source'] = 'brush_parameters'
    return payload


def _brush_param_export_fields(payload, *, spacing=None, raw=False):
    fields = {}
    tool = payload.get('handler') or payload.get('tool')
    if tool:
        fields['tool'] = tool
    if payload.get('view_name'):
        fields['view'] = payload['view_name']
    if payload.get('segment_id'):
        fields['segment'] = payload['segment_id']
    diameter_mm = _brush_diameter_mm(payload)
    if raw:
        if diameter_mm is not None:
            fields['diameter_mm'] = diameter_mm
    else:
        diameter_ijk = _brush_diameter_ijk(diameter_mm, spacing)
        if diameter_ijk is not None:
            fields['diameter_ijk'] = diameter_ijk
    if payload.get('sphere') is not None:
        fields['sphere'] = bool(payload['sphere'])
    return fields


def _brush_diameter_mm(payload):
    if (payload or {}).get('diameter_mm') is not None:
        return float(payload['diameter_mm'])
    radius_mm = _get_brush_radius_mm(payload or {})
    return None if radius_mm is None else float(radius_mm) * 2.0


def _brush_diameter_ijk(diameter_mm, spacing):
    if diameter_mm is None or spacing is None:
        return None
    result = []
    for value in spacing[:3]:
        value = float(value)
        if value == 0.0:
            return None
        result.append(float(diameter_mm) / value)
    return result


def _event_add_ras(event_dict, record):
    if record.ras is not None:
        event_dict['ras'] = [float(v) for v in record.ras]


def _event_add_ijk(event_dict, record, ras_to_ijk):
    payload = record.payload or {}
    if payload.get('ijk') is not None:
        event_dict['ijk'] = _ijk_export_coords(payload.get('ijk'))
        return
    if record.ras is not None and ras_to_ijk is not None:
        event_dict['ijk'] = _ras_to_ijk_coords(ras_to_ijk, record.ras)


def _raw_event(record, context=None) -> dict | None:
    """Raw input/source signal in the current no-back-compat format."""
    if record.event_type not in _RAW_MOUSE_TYPES + _RAW_SOURCE_TYPES:
        return None
    payload = record.payload or {}
    event_name = (
        MOUSE_EVENT if record.event_type in (PRESS, RELEASE, MOVE)
        else record.event_type
    )
    ev = {
        'timestamp': record.timestamp.isoformat(timespec='milliseconds'),
        'event': event_name,
    }
    analysis_event_type = payload.get('analysis_event_type')
    if analysis_event_type:
        ev['event_type'] = analysis_event_type
    view = payload.get('view_name')
    if view:
        ev['view'] = view
    slice_idx = (
        context.slice_for(payload)
        if context is not None else payload.get('slice_idx')
    )
    if slice_idx is not None:
        ev['slice'] = int(slice_idx)
        ev['z'] = int(slice_idx)
    if context is not None:
        context.note_slice(payload)
    if payload.get('xy') is not None:
        ev['xy'] = [int(v) for v in payload['xy']]
    if record.event_type in (PRESS, RELEASE, MOVE):
        ev['mouse_state'] = payload.get('mouse_state') or _mouse_state_from_payload(
            record.event_type, payload, False)
        _copy_mouse_binding_fields(ev, payload)
        return ev
    if record.event_type in (
            TOOL_SELECTED,
            SEGMENT_CREATED, SEGMENT_REMOVED, SEGMENT_RENAMED,
            SEGMENT_SELECTED, VOLUME_CHANGED,
            MODEL_FAMILY_CHANGED, MODEL_VARIANT_CHANGED, MODEL_CONFIRMED,
            POINT_PLACED, POINT_REPLACED, POINT_REMOVED,
            POINT_DRAG_START, POINT_DRAG_MOVE,
            SPX_BRUSH_FILL, SPX_ERASE_FILL, FILL_HOLE, SPX_BOUNDARY_TOGGLED,
            OVERWRITE_MODE_CHANGED):
        ev.setdefault('event_type', BOUNDARY_EVENT)
        _copy_raw_payload_fields(ev, payload)
        if record.event_type in (
                POINT_PLACED, POINT_REPLACED, POINT_REMOVED,
                POINT_DRAG_START, POINT_DRAG_MOVE):
            _event_add_ijk(
                ev, record,
                context.ras_to_ijk if context is not None else None)
        return ev
    if record.event_type == VIEW_CHANGED:
        ev.setdefault('event_type', TRAJECTORY_EVENT)
        ev['mouse_state'] = payload.get('mouse_state') or (
            MOUSE_HOLD if payload.get('left_button_down') else MOUSE_IDLE)
        _copy_mouse_binding_fields(ev, payload)
        if payload.get('wheel_delta') is not None:
            ev['wheel_delta'] = payload['wheel_delta']
        if payload.get('visual_state') is not None:
            ev['visual_state'] = payload['visual_state']
    if record.event_type == BRUSH_PARAMETERS_CHANGED:
        ev.setdefault('event_type', BOUNDARY_EVENT)
        fields = _brush_param_export_fields(payload, raw=True)
        if context is not None:
            context.note_slice(payload)
            context.context_delta(payload)
        ev.update(fields)
        return ev
    if context is None:
        delta = {}
        tool = payload.get('handler') or payload.get('tool')
        if tool is not None:
            delta['tool'] = tool
        if payload.get('segment_id') is not None:
            delta['segment'] = payload.get('segment_id')
        diameter_mm = _brush_diameter_mm(payload)
        if diameter_mm is not None:
            delta['diameter_mm'] = diameter_mm
    else:
        delta = context.context_delta(payload)
        context.note_slice(payload)
    ev.update(delta)
    return ev


def _append_raw_export_event(events, pending_trajectory, ev):
    if not _is_mergeable_mouse_trajectory(ev):
        if pending_trajectory is not None:
            events.append(pending_trajectory)
        events.append(ev)
        return None
    if (
        pending_trajectory is None or
        _trajectory_merge_key(pending_trajectory) != _trajectory_merge_key(ev)
    ):
        if pending_trajectory is not None:
            events.append(pending_trajectory)
        return _new_raw_trajectory_event(ev)
    _append_raw_trajectory_sample(pending_trajectory, ev)
    return pending_trajectory


def _is_mergeable_mouse_trajectory(ev):
    return (
        ev.get('event') == MOUSE_EVENT and
        ev.get('event_type') == TRAJECTORY_EVENT and
        ev.get('mouse_state') in (MOUSE_HOLD, MOUSE_IDLE) and
        ev.get('xy') is not None
    )


def _new_raw_trajectory_event(ev):
    merged = {
        key: value
        for key, value in ev.items()
        if key not in ('timestamp', 'xy')
    }
    merged['timestamps'] = [ev['timestamp']]
    merged['xy'] = [list(ev['xy'])]
    return merged


def _append_raw_trajectory_sample(merged, ev):
    merged['timestamps'].append(ev['timestamp'])
    merged['xy'].append(list(ev['xy']))


def _trajectory_merge_key(ev):
    return (
        ev.get('event'),
        ev.get('event_type'),
        ev.get('mouse_state'),
        ev.get('view'),
        ev.get('slice'),
        ev.get('tool'),
        ev.get('segment'),
        ev.get('diameter_mm'),
    )


def _copy_mouse_binding_fields(ev, payload):
    if 'tool' in payload:
        ev['tool'] = payload.get('tool')
    elif 'handler' in payload:
        ev['tool'] = payload.get('handler')
    else:
        ev['tool'] = None
    if payload.get('segment_id') is not None:
        ev['segment'] = payload.get('segment_id')
    diameter_mm = _brush_diameter_mm(payload)
    if diameter_mm is not None:
        ev['diameter_mm'] = diameter_mm


def _copy_raw_payload_fields(ev, payload):
    field_map = (
        ('segment_id', 'segment'),
        ('seg_name', 'seg_name'),
        ('volume_id', 'volume_id'),
        ('volume_name', 'volume_name'),
        ('segmentation_id', 'segmentation_id'),
        ('segmentation_name', 'segmentation_name'),
        ('old_name', 'old_name'),
        ('new_name', 'new_name'),
        ('volume', 'volume'),
        ('sequence_index', 'sequence_index'),
        ('family', 'family'),
        ('variant', 'variant'),
        ('point_id', 'point'),
        ('point_name', 'point_name'),
        ('point_index', 'point_index'),
        ('is_negative', 'negative'),
        ('point_action', 'point_action'),
        ('point_drag_phase', 'point_drag_phase'),
        ('mouse_pressed', 'mouse_pressed'),
        ('sphere', 'sphere'),
        ('mode', 'mode'),
        ('mode_label', 'mode_label'),
    )
    if 'tool' in payload:
        ev['tool'] = payload.get('tool')
    elif 'handler' in payload:
        ev['tool'] = payload.get('handler')
    if payload.get('mouse_boundary') is not None:
        ev['mouse_boundary'] = payload.get('mouse_boundary')
    for src, dst in field_map:
        if src in payload and payload.get(src) is not None:
            ev[dst] = payload.get(src)
    # SPX / fill-hole specific fields
    for key in ('operation', 'additive', 'label_id', 'delta_pixels',
                'model_key', 'params', 'axis', 'slice_idx', 'visible'):
        if key in payload and payload.get(key) is not None:
            ev[key] = payload[key]
    # SPX events store view under 'view' (not 'view_name') — copy it directly.
    if 'view' in payload and 'view' not in ev:
        ev['view'] = payload['view']
    diameter_mm = _brush_diameter_mm(payload)
    if diameter_mm is not None:
        ev['diameter_mm'] = diameter_mm


def _interpreted_event(record, ras_to_ijk, export_id, spacing=None) -> dict | None:
    """Semantic event: free_move, brush_move, point_placement, point_move etc."""
    payload = record.payload or {}
    if record.event_type == MOVE:
        if not _record_has_position(record):
            return None
        kind = payload.get('trajectory_kind')
        if kind == 'annotation_move':
            interp_type = 'brush_move'
        else:
            return None
    elif record.event_type in _INTERPRETED_TYPE_MAP:
        interp_type = _INTERPRETED_TYPE_MAP[record.event_type]
    else:
        return None
    ev = {
        'id': export_id,
        'timestamp': record.timestamp.isoformat(timespec='milliseconds'),
        'event': interp_type,
    }
    _event_add_ijk(ev, record, ras_to_ijk)
    view = payload.get('view_name')
    if view:
        ev['view'] = view
    if payload.get('slice_idx') is not None:
        ev['z'] = int(payload.get('slice_idx'))
    seg = payload.get('segment_id')
    if seg:
        ev['segment'] = seg
    tool = payload.get('handler')
    if interp_type in ('brush_move', 'brush_click', 'brush_parameters') and tool:
        ev['tool'] = tool
    if interp_type in ('brush_move', 'brush_click', 'brush_parameters'):
        brush_mm = _get_brush_radius_mm(payload)
        if brush_mm is not None:
            ev['brush_mm'] = brush_mm
    if interp_type == 'brush_parameters':
        ev.update(_brush_param_export_fields(payload, spacing=spacing))
    if record.event_type in _POINT_VERDICT_TYPES:
        for src, dst in (('point_id', 'point'), ('point_name', 'point_name'),
                         ('point_index', 'point_index'), ('is_negative', 'negative')):
            val = payload.get(src)
            if val is not None:
                ev[dst] = val
    if interp_type == 'view_changed':
        if payload.get('wheel_delta') is not None:
            ev['wheel_delta'] = payload['wheel_delta']
        if payload.get('visual_state') is not None:
            ev['visual_state'] = payload['visual_state']
    if interp_type == 'segment_removed' and payload.get('seg_name'):
        ev['seg_name'] = payload['seg_name']
    if interp_type == 'segment_renamed':
        for k in ('old_name', 'new_name'):
            if payload.get(k):
                ev[k] = payload[k]
    if interp_type == 'tool_selected':
        ev['tool'] = payload.get('tool')  # None is meaningful: tool was deselected
    return ev


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
        if point_drag_active and record.event_type == POINT_REPLACED:
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
        'timestamp': record.timestamp.isoformat(timespec='milliseconds'),
        'event': record.event_type,
    }
    if record.ras is not None:
        event['ras'] = [float(v) for v in record.ras]
    _copy_compact(payload, event, 'view_name', 'view')
    _copy_compact(payload, event, 'xy', 'xy')
    if payload.get('slice_idx') is not None:
        event['z'] = int(payload.get('slice_idx'))
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
    _copy_compact(payload, event, 'ras_source', 'ras_source')
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
    brush_mm = _get_brush_radius_mm(payload)
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


def _move_thinning_metadata():
    return {
        'mode': 'xy_to_ijk_scaled',
        'ann_ijk': MOVE_ANNOTATION_TARGET_IJK,
        'hover_ijk': MOVE_HOVER_TARGET_IJK,
        'ann_px': list(MOVE_ANNOTATION_PIXEL_CLAMP),
        'hover_px': list(MOVE_HOVER_PIXEL_CLAMP),
        'ann_ms': int(MOVE_ANNOTATION_MAX_INTERVAL_MS),
        'hover_ms': int(MOVE_HOVER_MAX_INTERVAL_MS),
    }


def _move_xy_tuple(payload):
    xy = (payload or {}).get('xy')
    if xy is None or len(xy) < 2:
        return None
    return float(xy[0]), float(xy[1])


def _move_pressed(payload):
    payload = payload or {}
    state = payload.get('mouse_button_state')
    if state == 'pressed':
        return True
    if state == 'released':
        return False
    return bool(payload.get('left_button_down'))


def _move_context_signature(view_name, payload):
    payload = payload or {}
    return (
        view_name,
        payload.get('handler') or payload.get('tool'),
        payload.get('segment_id'),
        payload.get('brush_radius_mm'),
        payload.get('axis'),
        payload.get('slice_idx'),
        payload.get('view_event'),
    )


def _move_max_interval_ms(pressed):
    return (
        MOVE_ANNOTATION_MAX_INTERVAL_MS
        if pressed else MOVE_HOVER_MAX_INTERVAL_MS
    )


def _move_pixel_threshold_from_scale(ijk_per_pixel, pressed):
    if ijk_per_pixel is None or ijk_per_pixel <= 0.0:
        return float(MOVE_ANNOTATION_PIXEL_CLAMP[0] if pressed else 3)
    target = (
        MOVE_ANNOTATION_TARGET_IJK if pressed
        else MOVE_HOVER_TARGET_IJK
    )
    lo, hi = (
        MOVE_ANNOTATION_PIXEL_CLAMP if pressed
        else MOVE_HOVER_PIXEL_CLAMP
    )
    value = int(round(float(target) / float(ijk_per_pixel)))
    return float(max(lo, min(hi, value)))


def _ijk_per_xy_pixel(mat16):
    if mat16 is None or len(mat16) != 16:
        return None
    x_scale = (
        float(mat16[0]) * float(mat16[0]) +
        float(mat16[4]) * float(mat16[4]) +
        float(mat16[8]) * float(mat16[8])
    ) ** 0.5
    y_scale = (
        float(mat16[1]) * float(mat16[1]) +
        float(mat16[5]) * float(mat16[5]) +
        float(mat16[9]) * float(mat16[9])
    ) ** 0.5
    scale = max(x_scale, y_scale)
    return scale if scale > 0.0 else None


def _slice_widget(view_name):
    if not hasattr(slicer, 'app'):
        return None
    return slicer.app.layoutManager().sliceWidget(view_name)


def _slice_view(view_name):
    sw = _slice_widget(view_name)
    return sw.sliceView() if sw else None


def _left_button_is_down():
    if not hasattr(qt, 'QApplication') or not hasattr(qt, 'Qt'):
        return False
    return bool(qt.QApplication.mouseButtons() & qt.Qt.LeftButton)


def _slice_interactor(slice_view):
    if slice_view is None:
        return None
    return slice_view.interactor()


def _point_xy(point):
    x = point.x() if callable(point.x) else point.x
    y = point.y() if callable(point.y) else point.y
    return [int(round(float(x))), int(round(float(y)))]


def _slice_device_xy_to_ijk_matrix(view_name, volume_node=None):
    sw = _slice_widget(view_name)
    if sw is None or not hasattr(sw, 'sliceView') or not hasattr(sw, 'sliceLogic'):
        return None
    try:
        slice_view = sw.sliceView()
        layer = sw.sliceLogic().GetBackgroundLayer()
        if layer is None:
            return None
        if volume_node is not None and layer.GetVolumeNode() != volume_node:
            return None
        xyz_to_ijk = layer.GetXYToIJKTransform()
        if xyz_to_ijk is None:
            return None

        def transform(x, y):
            xyz = slice_view.convertDeviceToXYZ((float(x), float(y)))
            return [float(v) for v in xyz_to_ijk.TransformDoublePoint(xyz)]

        p0 = transform(0.0, 0.0)
        px = transform(1.0, 0.0)
        py = transform(0.0, 1.0)
        return [
            px[0] - p0[0], py[0] - p0[0], 0.0, p0[0],
            px[1] - p0[1], py[1] - p0[1], 0.0, p0[1],
            px[2] - p0[2], py[2] - p0[2], 0.0, p0[2],
            0.0, 0.0, 0.0, 1.0,
        ]
    except Exception:
        return None


def _ras_inside_volume(volume_node, ras):
    if volume_node is None or ras is None:
        return False
    mat = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(mat)
    ijk = [
        sum(mat.GetElement(r, c) * (ras[c] if c < 3 else 1.0)
            for c in range(4))
        for r in range(3)
    ]
    dims = volume_node.GetImageData().GetDimensions()
    return _ijk_inside_volume(ijk, dims)


def _visual_state(view_name, volume_node=None):
    node = _slice_widget(view_name).mrmlSliceNode()
    mat = node.GetXYToRAS()
    xy_to_ras = [mat.GetElement(r, c) for r in range(4) for c in range(4)]
    state = {
        'slice_offset': float(node.GetSliceOffset()),
        'field_of_view': [float(v) for v in node.GetFieldOfView()],
        'xy_to_ras': xy_to_ras,
    }
    xy_to_ijk = _slice_device_xy_to_ijk_matrix(view_name, volume_node)
    if xy_to_ijk is not None:
        state['xy_to_ijk'] = xy_to_ijk
        state['xy_coordinate_system'] = 'vtk_device'
    return state


def _all_slice_visual_state(volume_node=None):
    return {
        view_name: _visual_state(view_name, volume_node)
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
    pos = (
        tuple(round(float(v), 3) for v in ras)
        if ras is not None
        else tuple(payload.get('xy') or ())
    )
    return (
        view_name,
        pos,
        payload.get('handler'),
        payload.get('mouse_status'),
        payload.get('trajectory_role'),
        payload.get('trajectory_kind'),
        payload.get('mouse_button_state'),
    )


def _boundary_key(view_name, ras, event_type, payload):
    pos = (
        tuple(round(float(v), 3) for v in ras)
        if ras is not None
        else tuple(payload.get('xy') or ())
    )
    return (
        view_name,
        event_type,
        pos,
        payload.get('handler'),
        payload.get('segment_id'),
    )
