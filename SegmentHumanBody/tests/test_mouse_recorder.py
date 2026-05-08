from core._mouse_recorder import (
    MOVE, PRESS, RELEASE, VIEW_CHANGED, ACTION, POINT_DRAG_START, POINT_DRAG_MOVE,
    POINT_DRAG_END, POINT_PLACED, POINT_REPLACED, POINT_REMOVED, METADATA,
    BRUSH_PARAMETERS_CHANGED, MouseEventRecorder,
    _SliceRecordInteractorObserver,
    _SliceRecordListener,
)
import datetime
import core._mouse_recorder as recorder_mod


def test_default_sample_rate_is_60_hz():
    recorder = MouseEventRecorder()
    assert round(1000.0 / recorder._move_interval_ms) == 60


def test_metadata_caches_initial_slice_visual_state(monkeypatch):
    monkeypatch.setattr(
        recorder_mod,
        '_all_slice_visual_state',
        lambda volume_node=None: {'Red': {'view_name': 'Red', 'slice_offset': 12.0}},
    )
    recorder = MouseEventRecorder()

    recorder.start(volume_node=None, segmentation_name=None)

    start = recorder.records[0]
    assert start.event_type == METADATA
    assert start.payload['initial_visual_state'] == {
        'Red': {'view_name': 'Red', 'slice_offset': 12.0},
    }


def test_metadata_records_compact_move_thinning_policy(monkeypatch):
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda volume_node=None: {})
    recorder = MouseEventRecorder()

    recorder.start(volume_node=None, segmentation_name=None)

    policy = recorder.records[0].payload['move_thinning']
    assert policy == {
        'mode': 'xy_to_ijk_scaled',
        'ann_ijk': 0.5,
        'hover_ijk': 2.0,
        'ann_px': [1, 4],
        'hover_px': [2, 12],
        'ann_ms': 100,
        'hover_ms': 250,
    }


def test_stop_does_not_append_second_metadata_or_session_stop(monkeypatch):
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda volume_node=None: {})
    recorder = MouseEventRecorder()

    recorder.start(volume_node=None, segmentation_name=None)
    recorder.stop()

    assert [r.event_type for r in recorder.records] == [METADATA]


def test_recorder_callback_updates_after_append():
    recorder = MouseEventRecorder()
    counts = []
    recorder.on_record_appended = lambda: counts.append(len(recorder))

    recorder.record_action('onUndo')

    assert counts == [1]


def test_event_ids_are_sequential_and_exported():
    recorder = MouseEventRecorder()
    recorder.record_action('first')
    recorder.record_action('second')

    assert [r.event_id for r in recorder.records] == [1, 2]
    exported = recorder.export_data()
    assert exported['type'] == 'SegmentHumanBody.annotation_process'
    assert [item['id'] for item in exported['events']] == [1, 2]
    assert all('t_ms' not in item for item in exported['events'])


def test_export_ids_start_at_one_after_metadata(monkeypatch):
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda volume_node=None: {})
    recorder = MouseEventRecorder()

    recorder.start(volume_node=None, segmentation_name=None)
    recorder.record_action('first-process-event')

    exported = recorder.export_data()

    assert [r.event_type for r in recorder.records] == [METADATA, 'action']
    assert exported['events'][0]['id'] == 1


def test_visual_state_does_not_repeat_slice_view_dimensions(monkeypatch):
    class _Matrix:
        def GetElement(self, r, c): return 1.0 if r == c else 0.0

    class _SliceNode:
        def GetSliceOffset(self): return 12.0
        def GetFieldOfView(self): return [100.0, 100.0, 1.0]
        def GetDimensions(self): return [512, 512, 1]
        def GetXYToRAS(self): return _Matrix()

    class _SliceWidget:
        def mrmlSliceNode(self): return _SliceNode()

    class _Layout:
        def sliceWidget(self, view_name): return _SliceWidget()

    class _App:
        def layoutManager(self): return _Layout()

    monkeypatch.setattr(recorder_mod.slicer, 'app', _App(), raising=False)

    state = recorder_mod._visual_state('Red')

    _identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert state == {
        'slice_offset': 12.0,
        'field_of_view': [100.0, 100.0, 1.0],
        'xy_to_ras': _identity,
    }


def test_slice_device_xy_to_ijk_matrix_matches_dataprobe_path(monkeypatch):
    class _Transform:
        def TransformDoublePoint(self, xyz):
            return [2.0 * xyz[0], 3.0 * xyz[1], xyz[2] + 5.0]

    class _Layer:
        def GetVolumeNode(self):
            return volume

        def GetXYToIJKTransform(self):
            return _Transform()

    class _SliceLogic:
        def GetBackgroundLayer(self):
            return _Layer()

    class _SliceView:
        def convertDeviceToXYZ(self, xy):
            return [float(xy[0]) + 10.0, float(xy[1]) + 20.0, 7.0]

    class _SliceWidget:
        def sliceView(self):
            return _SliceView()

        def sliceLogic(self):
            return _SliceLogic()

    class _Layout:
        def sliceWidget(self, view_name):
            return _SliceWidget()

    class _App:
        def layoutManager(self):
            return _Layout()

    volume = object()
    monkeypatch.setattr(recorder_mod.slicer, 'app', _App(), raising=False)

    mat = recorder_mod._slice_device_xy_to_ijk_matrix('Red', volume)
    ijk = recorder_mod._xy_to_ijk_from_matrix([4, 5], mat)

    assert ijk == [28.0, 75.0, 12.0]


def test_slice_device_xy_to_ijk_matrix_ignores_nonmatching_background_volume(monkeypatch):
    class _Layer:
        def GetVolumeNode(self):
            return object()

    class _SliceLogic:
        def GetBackgroundLayer(self):
            return _Layer()

    class _SliceWidget:
        def sliceView(self):
            return object()

        def sliceLogic(self):
            return _SliceLogic()

    class _Layout:
        def sliceWidget(self, view_name):
            return _SliceWidget()

    class _App:
        def layoutManager(self):
            return _Layout()

    monkeypatch.setattr(recorder_mod.slicer, 'app', _App(), raising=False)

    assert recorder_mod._slice_device_xy_to_ijk_matrix('Red', object()) is None


def test_export_annotates_events_with_ijk_when_ras_to_ijk_present():
    # identity ras_to_ijk: RAS [x,y,z] -> IJK [x,y,z]
    identity = [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]
    recorder = MouseEventRecorder()
    recorder._append(METADATA, None, {'volume': {'ras_to_ijk': identity}})
    recorder._append(
        MOVE, [1.0, 2.0, 3.0],
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._append(
        MOVE, [7.0, 8.0, 9.0],
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )

    exported = recorder.export_data()['events']

    assert exported[0]['ijk'] == [1, 2, 3]
    assert exported[1]['ijk'] == [7, 8, 9]


def test_export_omits_ijk_when_no_ras_to_ijk():
    recorder = MouseEventRecorder()
    recorder._append(
        MOVE, [1.0, 2.0, 3.0],
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )

    event = recorder.export_data()['events'][0]

    assert 'ijk' not in event


def test_export_omits_ijk_on_events_without_ras():
    identity = [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]
    recorder = MouseEventRecorder()
    recorder._append(METADATA, None, {'volume': {'ras_to_ijk': identity}})
    recorder._append(ACTION, None, {'name': 'undo'})

    events = recorder.export_data()['events']
    action_event = next(e for e in events if e['event'] == 'action')

    assert 'ijk' not in action_event


def test_clear_resets_event_ids():
    recorder = MouseEventRecorder()
    recorder.record_action('first')
    recorder.clear()
    recorder.record_action('second')

    assert recorder.records[0].event_id == 1


def test_export_preserves_compact_wheel_metadata():
    recorder = MouseEventRecorder()
    recorder._append(
        MOVE,
        [1.0, 2.0, 3.0],
        {
            'mouse_status': 'wheel',
            'analysis_event_type': 'trajectory_event',
            'wheel_delta': [0, 120],
        },
    )

    exported = recorder.export_data()
    event = exported['events'][0]

    assert event['event'] == MOVE
    assert event['mouse'] == 'wheel'
    assert event['analysis'] == 'trajectory_event'
    assert event['wheel_delta'] == [0, 120]
    assert 'payload' not in event


def test_export_uses_numeric_pressed_state():
    recorder = MouseEventRecorder()
    recorder._append(
        PRESS,
        [1.0, 2.0, 3.0],
        {'mouse_status': 'press', 'analysis_event_type': 'boundary_event'},
    )
    recorder._append(
        MOVE,
        [2.0, 3.0, 4.0],
        {
            'mouse_status': 'move',
            'mouse_button_state': 'pressed',
            'analysis_event_type': 'trajectory_event',
        },
    )
    recorder._append(
        RELEASE,
        [3.0, 4.0, 5.0],
        {'mouse_status': 'release', 'analysis_event_type': 'boundary_event'},
    )

    exported = recorder.export_data()['events']

    assert [event['pressed'] for event in exported] == [1, 1, 0]
    assert all('button' not in event for event in exported)


def test_export_preserves_point_name_for_placed_verdict_events():
    recorder = MouseEventRecorder()

    recorder.record_point_placed(
        'seg-a', [1.0, 2.0, 3.0], False,
        point_index=0, point_id='cp-0', point_name='Positive 1')

    event = recorder.export_data()['events'][0]

    assert event['event'] == 'point_placed'
    assert event['point_action'] == 'place'
    assert event['ras_source'] == 'markup_world'
    assert event['point'] == 'cp-0'
    assert event['point_name'] == 'Positive 1'


def test_export_preserves_point_name_for_replaced_verdict_events():
    recorder = MouseEventRecorder()
    import datetime
    recorder.record_point_drag(
        'end', 'seg-a', [3.0, 4.0, 5.0], False,
        view_name='Red', point_index=1, point_id='cp-1',
        point_name='Positive 2')

    event = recorder.export_data()['events'][0]

    assert event['event'] == 'point_replaced'
    assert event['point_action'] == 'replace'
    assert event['ras_source'] == 'markup_world'
    assert event['point'] == 'cp-1'
    assert event['point_name'] == 'Positive 2'


def test_export_suppresses_raw_mouse_companions_during_point_drag():
    recorder = MouseEventRecorder()
    t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)
    sequence = [
        (PRESS, [40.0, 233.0, -1.0],
         {'mouse_status': 'press', 'analysis_event_type': 'boundary_event'}),
        (POINT_DRAG_START, [39.0, 93.0, -1.0],
         {
             'view_name': 'Red',
             'segment_id': 'seg-a',
             'handler': 'point',
             'point_id': 'cp-0',
             'point_name': 'Positive 1',
             'point_action': 'grab',
             'analysis_event_type': 'boundary_event',
         }),
        (MOVE, [33.0, 231.0, -1.0],
         {
             'mouse_status': 'move',
             'mouse_button_state': 'pressed',
             'analysis_event_type': 'trajectory_event',
         }),
        (POINT_DRAG_MOVE, [22.0, 92.0, -1.0],
         {
             'view_name': 'Red',
             'segment_id': 'seg-a',
             'handler': 'point',
             'point_id': 'cp-0',
             'point_name': 'Positive 1',
             'point_action': 'move',
             'analysis_event_type': 'trajectory_event',
             'trajectory_kind': 'non_annotation_move',
             'trajectory_role': 'visualization_trajectory',
         }),
        (RELEASE, [4.0, 252.0, -1.0],
         {'mouse_status': 'release', 'analysis_event_type': 'boundary_event'}),
        (POINT_REPLACED, [3.0, 74.0, -1.0],
         {
             'view_name': 'Red',
             'segment_id': 'seg-a',
             'handler': 'point',
             'point_id': 'cp-0',
             'point_name': 'Positive 1',
             'point_action': 'replace',
             'ras_source': 'markup_world',
             'analysis_event_type': 'boundary_event',
         }),
    ]
    for offset_ms, (event_type, ras, payload) in enumerate(sequence):
        recorder._records.append(recorder._new_record(
            t0 + datetime.timedelta(milliseconds=offset_ms * 20),
            ras,
            event_type,
            payload,
        ))

    exported = recorder.export_data()['events']

    assert [event['id'] for event in exported] == [1, 2, 3]
    assert [event['event'] for event in exported] == [
        POINT_DRAG_START, POINT_DRAG_MOVE, POINT_REPLACED,
    ]
    assert [event['point_action'] for event in exported] == [
        'grab', 'move', 'replace',
    ]
    assert exported[-1]['ras'] == [3.0, 74.0, -1.0]


def test_start_uses_only_vtk_listener_when_interactor_is_available(monkeypatch):
    class _Timer:
        def connect(self, *args):
            pass

        def start(self, *args):
            pass

        def stop(self):
            pass

    class _Interactor:
        def __init__(self):
            self.callbacks = {}
            self.removed = []

        def AddObserver(self, event_name, callback, priority=None):
            self.callbacks[event_name] = callback
            return event_name

        def RemoveObserver(self, tag):
            self.removed.append(tag)

    class _SliceView:
        def __init__(self):
            self._interactor = _Interactor()
            self.filters = []
            self.removed_filters = []

        def interactor(self):
            return self._interactor

        def installEventFilter(self, filt):
            self.filters.append(filt)

        def removeEventFilter(self, filt):
            self.removed_filters.append(filt)

    view = _SliceView()
    monkeypatch.setattr(
        recorder_mod, '_slice_view',
        lambda view_name: view if view_name == 'Red' else None)
    monkeypatch.setattr(recorder_mod, '_left_button_is_down', lambda: False)
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda volume_node=None: {})
    monkeypatch.setattr(recorder_mod.qt, 'QTimer', _Timer, raising=False)

    recorder = MouseEventRecorder()
    recorder.start(volume_node=None)

    assert len(recorder._listeners) == 1
    assert recorder._listeners[0].backend == 'vtk_interactor'
    assert view.filters == []
    assert 'LeftButtonPressEvent' in view._interactor.callbacks

    recorder.stop()

    assert view.removed_filters == []
    assert 'LeftButtonPressEvent' in view._interactor.removed


def test_start_skips_slice_view_when_vtk_interactor_is_unavailable(monkeypatch):
    class _Timer:
        def connect(self, *args):
            pass

        def start(self, *args):
            pass

        def stop(self):
            pass

    class _SliceView:
        def __init__(self):
            self.filters = []
            self.removed_filters = []

        def interactor(self):
            return None

        def installEventFilter(self, filt):
            self.filters.append(filt)

        def removeEventFilter(self, filt):
            self.removed_filters.append(filt)

    view = _SliceView()
    monkeypatch.setattr(
        recorder_mod, '_slice_view',
        lambda view_name: view if view_name == 'Red' else None)
    monkeypatch.setattr(recorder_mod, '_left_button_is_down', lambda: False)
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda volume_node=None: {})
    monkeypatch.setattr(recorder_mod.qt, 'QTimer', _Timer, raising=False)

    recorder = MouseEventRecorder()
    recorder.start(volume_node=None)

    assert recorder._listeners == []
    assert view.filters == []

    recorder.stop()

    assert view.removed_filters == []


def test_slice_record_listener_uses_one_backend_for_all_mouse_events(monkeypatch):
    class _Interactor:
        def __init__(self):
            self.callbacks = {}
            self.position = (0, 0)

        def AddObserver(self, event_name, callback, priority=None):
            self.callbacks[event_name] = callback
            return event_name

        def RemoveObserver(self, tag):
            pass

        def GetEventPosition(self):
            return self.position

    class _SliceView:
        height = 200

        def __init__(self):
            self._interactor = _Interactor()
            self.filters = []

        def interactor(self):
            return self._interactor

        def installEventFilter(self, filt):
            self.filters.append(filt)

    calls = []
    monkeypatch.setattr(recorder_mod, '_left_button_is_down', lambda: False)
    view = _SliceView()
    listener = _SliceRecordListener('Red', view, lambda *args: calls.append(args))

    assert listener.install() is True
    assert listener.backend == 'vtk_interactor'
    assert view.filters == []

    view._interactor.position = (10, 20)
    view._interactor.callbacks['LeftButtonPressEvent'](
        view._interactor, 'LeftButtonPressEvent')
    view._interactor.position = (11, 21)
    view._interactor.callbacks['MouseMoveEvent'](
        view._interactor, 'MouseMoveEvent')
    view._interactor.position = (12, 22)
    view._interactor.callbacks['LeftButtonReleaseEvent'](
        view._interactor, 'LeftButtonReleaseEvent')

    assert [call[2] for call in calls] == [PRESS, MOVE, RELEASE]
    assert {call[3]['input_source'] for call in calls} == {'vtk_interactor'}


def test_vtk_interactor_observer_records_brush_path_events():
    class _Interactor:
        def __init__(self):
            self.callbacks = {}
            self.priorities = {}
            self.position = (0, 0)

        def AddObserver(self, event_name, callback, priority=None):
            self.callbacks[event_name] = callback
            self.priorities[event_name] = priority
            return event_name

        def RemoveObserver(self, tag):
            self.callbacks.pop(tag, None)

        def GetEventPosition(self):
            return self.position

    class _SliceView:
        def __init__(self):
            self._interactor = _Interactor()

        def interactor(self):
            return self._interactor

    calls = []
    view = _SliceView()
    observer = _SliceRecordInteractorObserver(
        'Red', view, lambda *args: calls.append(args))
    assert observer.install() is True

    view._interactor.position = (10, 20)
    view._interactor.callbacks['LeftButtonPressEvent'](
        view._interactor, 'LeftButtonPressEvent')
    view._interactor.position = (11, 21)
    view._interactor.callbacks['MouseMoveEvent'](
        view._interactor, 'MouseMoveEvent')
    view._interactor.position = (12, 22)
    view._interactor.callbacks['LeftButtonReleaseEvent'](
        view._interactor, 'LeftButtonReleaseEvent')

    assert [call[2] for call in calls] == [PRESS, MOVE, RELEASE]
    assert calls[1][3]['left_button_down'] is True
    assert calls[0][3]['input_source'] == 'vtk_interactor'
    assert calls[2][3]['mouse_status'] == 'release'
    assert all(priority == recorder_mod.VTK_OBSERVER_PRIORITY
               for priority in view._interactor.priorities.values())


def test_vtk_interactor_uses_dataprobe_device_xy(monkeypatch):
    class _Point:
        def __init__(self, x, y):
            self._x = x
            self._y = y

        def x(self): return self._x
        def y(self): return self._y

    class _Interactor:
        def __init__(self):
            self.callbacks = {}
            self.position = (10, 20)

        def AddObserver(self, event_name, callback, priority=None):
            self.callbacks[event_name] = callback
            return event_name

        def RemoveObserver(self, tag):
            pass

        def GetEventPosition(self):
            return self.position

    class _SliceView:
        height = 200

        def __init__(self):
            self._interactor = _Interactor()

        def interactor(self):
            return self._interactor

        def mapFromGlobal(self, point):
            return _Point(point.x() - 1000, point.y() - 2000)

    class _Cursor:
        @staticmethod
        def pos():
            return _Point(1368, 2131)

    class _QPoint:
        def __init__(self, x, y):
            self._x = x
            self._y = y

        def x(self): return self._x
        def y(self): return self._y

    calls = []
    view = _SliceView()
    monkeypatch.setattr(recorder_mod.qt, 'QCursor', _Cursor, raising=False)
    monkeypatch.setattr(recorder_mod.qt, 'QPoint', _QPoint, raising=False)
    observer = _SliceRecordInteractorObserver(
        'Red', view, lambda *args: calls.append(args))
    observer.install()

    view._interactor.callbacks['MouseMoveEvent'](
        view._interactor, 'MouseMoveEvent')

    assert calls[0][1] == [10, 20]
    assert calls[0][3]['xy_source'] == 'vtk_device'
    assert 'xy_global' not in calls[0][3]


def test_event_payload_exposes_handler_and_params():
    recorder = MouseEventRecorder()
    recorder.context_fn = lambda view_name=None: {
        'tool': 'brush',
        'axis': 0,
        'slice_idx': 3,
        'brush_radius_mm': 2.5,
    }

    payload = recorder._event_payload('Red')

    assert payload['handler'] == 'brush'
    assert payload['handler_params'] == {
        'brush_radius_mm': 2.5,
        'axis': 0,
        'slice_idx': 3,
    }



def test_move_sampling_uses_latest_position_without_duplicates(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.context_fn = lambda view_name=None: {'tool': None}
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view, volume_node=None: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), MOVE,
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._on_mouse(
        'Red', (30, 40), MOVE,
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._sample_pending_move()
    recorder._sample_pending_move()

    assert [r.event_type for r in recorder.records] == [MOVE]
    assert recorder.records[0].ras is None
    assert recorder.records[0].payload['xy'] == [30, 40]


def test_latest_raw_move_is_kept_without_volume_conversion(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.context_fn = lambda view_name=None: {'tool': None}
    inside_ts = datetime.datetime(2026, 1, 1, 12, 0, 0)
    outside_ts = inside_ts + datetime.timedelta(milliseconds=10)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view, volume_node=None: {'view_name': view})

    recorder._capture_pending_move(
        inside_ts, 'Red', (10, 20),
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._capture_pending_move(
        outside_ts, 'Red', (999, 999),
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._sample_pending_move()

    assert [r.event_type for r in recorder.records] == [MOVE]
    assert recorder.records[0].ras is None
    assert recorder.records[0].payload['xy'] == [999, 999]
    assert recorder.records[0].timestamp == outside_ts


def test_hot_path_drops_inactive_xy_before_raw_record_accumulates():
    identity = [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._active_region_gate = recorder_mod._ActiveRegionGate({
        'volume': {
            'dimensions': [5, 5, 5],
            'ras_to_ijk': identity,
        },
        'initial_visual_state': {
            'Red': {'xy_to_ras': identity, 'xy_to_ijk': identity},
        },
    })

    recorder._on_mouse(
        'Red', (-1, 1), MOVE,
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._sample_pending_move()
    recorder._on_mouse(
        'Red', (1, 1), MOVE,
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._sample_pending_move()

    assert [r.payload['xy'] for r in recorder.records] == [[1, 1]]


def test_active_region_gate_prefers_dataprobe_xy_to_ijk_for_bounds():
    identity = [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]
    outside_xy_to_ras = [100.0, 0.0, 0.0, 1000.0,
                         0.0, 100.0, 0.0, 1000.0,
                         0.0, 0.0, 1.0, 1000.0,
                         0.0, 0.0, 0.0, 1.0]
    inside_xy_to_ijk = [1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 2.0,
                        0.0, 0.0, 0.0, 1.0]
    gate = recorder_mod._ActiveRegionGate({
        'volume': {
            'dimensions': [20, 20, 5],
            'ras_to_ijk': identity,
        },
        'initial_visual_state': {
            'Red': {
                'xy_to_ras': outside_xy_to_ras,
                'xy_to_ijk': inside_xy_to_ijk,
            },
        },
    })

    assert gate.accepts_xy('Red', (10, 10)) is True
    assert gate.accepts_xy('Red', (25, 10)) is False


def test_hot_path_drops_inactive_boundary_but_updates_button_state(monkeypatch):
    identity = [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._active_region_gate = recorder_mod._ActiveRegionGate({
        'volume': {
            'dimensions': [5, 5, 5],
            'ras_to_ijk': identity,
        },
        'initial_visual_state': {
            'Red': {'xy_to_ras': identity, 'xy_to_ijk': identity},
        },
    })
    monkeypatch.setattr(
        recorder_mod, '_visual_state',
        lambda view, volume_node=None: {'xy_to_ras': identity, 'xy_to_ijk': identity})

    recorder._on_mouse(
        'Red', (-1, 1), PRESS,
        {'mouse_status': 'press', 'analysis_event_type': 'boundary_event'},
    )
    assert recorder.records == []
    assert recorder._active_mouse_press is True

    recorder._on_mouse(
        'Red', (-1, 1), RELEASE,
        {'mouse_status': 'release', 'analysis_event_type': 'boundary_event'},
    )
    assert recorder.records == []
    assert recorder._active_mouse_press is False


def test_released_hover_moves_are_thinned_by_pixel_and_time_threshold():
    recorder = MouseEventRecorder(sample_rate_hz=30)
    recorder._active = True
    t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)

    recorder._capture_pending_move(
        t0, 'Red', (10, 10),
        {'mouse_status': 'move', 'left_button_down': False},
    )
    recorder._sample_pending_move()
    recorder._capture_pending_move(
        t0 + datetime.timedelta(milliseconds=30), 'Red', (11, 10),
        {'mouse_status': 'move', 'left_button_down': False},
    )
    recorder._sample_pending_move()
    recorder._capture_pending_move(
        t0 + datetime.timedelta(milliseconds=60), 'Red', (13, 10),
        {'mouse_status': 'move', 'left_button_down': False},
    )
    recorder._sample_pending_move()
    recorder._capture_pending_move(
        t0 + datetime.timedelta(milliseconds=320), 'Red', (14, 10),
        {'mouse_status': 'move', 'left_button_down': False},
    )
    recorder._sample_pending_move()

    assert [r.payload['xy'] for r in recorder.records] == [
        [10, 10], [13, 10], [14, 10],
    ]


def test_pressed_annotation_moves_keep_one_pixel_changes():
    recorder = MouseEventRecorder(sample_rate_hz=30)
    recorder._active = True
    t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)

    recorder._capture_pending_move(
        t0, 'Red', (10, 10),
        {'mouse_status': 'move', 'left_button_down': True},
    )
    recorder._sample_pending_move()
    recorder._capture_pending_move(
        t0 + datetime.timedelta(milliseconds=30), 'Red', (11, 10),
        {'mouse_status': 'move', 'left_button_down': True},
    )
    recorder._sample_pending_move()

    assert [r.payload['xy'] for r in recorder.records] == [[10, 10], [11, 10]]
    assert [r.payload['mouse_button_state'] for r in recorder.records] == [
        'pressed', 'pressed',
    ]


def test_move_pressed_state_change_is_kept_even_without_xy_change():
    recorder = MouseEventRecorder(sample_rate_hz=30)
    recorder._active = True
    t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)

    recorder._capture_pending_move(
        t0, 'Red', (10, 10),
        {'mouse_status': 'move', 'left_button_down': False},
    )
    recorder._sample_pending_move()
    recorder._capture_pending_move(
        t0 + datetime.timedelta(milliseconds=30), 'Red', (10, 10),
        {'mouse_status': 'move', 'left_button_down': True},
    )
    recorder._sample_pending_move()

    assert [r.payload['xy'] for r in recorder.records] == [[10, 10], [10, 10]]
    assert [r.payload['mouse_button_state'] for r in recorder.records] == [
        'released', 'pressed',
    ]


def test_hover_move_threshold_scales_with_xy_to_ijk():
    xy_to_ijk = [0.25, 0.0, 0.0, 0.0,
                 0.0, 0.25, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0]
    recorder = MouseEventRecorder(sample_rate_hz=60)
    recorder._active = True
    recorder._active_region_gate = recorder_mod._ActiveRegionGate({
        'volume': {
            'dimensions': [100, 100, 100],
            'ras_to_ijk': [1.0, 0.0, 0.0, 0.0,
                           0.0, 1.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 1.0],
        },
        'initial_visual_state': {
            'Red': {'xy_to_ijk': xy_to_ijk},
        },
    })
    t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)

    for dt, xy in ((0, (10, 10)), (30, (16, 10)), (60, (18, 10))):
        recorder._capture_pending_move(
            t0 + datetime.timedelta(milliseconds=dt), 'Red', xy,
            {'mouse_status': 'move', 'left_button_down': False},
        )
        recorder._sample_pending_move()

    assert [r.payload['xy'] for r in recorder.records] == [[10, 10], [18, 10]]


def test_pressed_move_threshold_scales_with_xy_to_ijk():
    xy_to_ijk = [0.25, 0.0, 0.0, 0.0,
                 0.0, 0.25, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0]
    recorder = MouseEventRecorder(sample_rate_hz=60)
    recorder._active = True
    recorder._active_region_gate = recorder_mod._ActiveRegionGate({
        'volume': {
            'dimensions': [100, 100, 100],
            'ras_to_ijk': [1.0, 0.0, 0.0, 0.0,
                           0.0, 1.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 1.0],
        },
        'initial_visual_state': {
            'Red': {'xy_to_ijk': xy_to_ijk},
        },
    })
    t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)

    for dt, xy in ((0, (10, 10)), (30, (11, 10)), (60, (12, 10))):
        recorder._capture_pending_move(
            t0 + datetime.timedelta(milliseconds=dt), 'Red', xy,
            {'mouse_status': 'move', 'left_button_down': True},
        )
        recorder._sample_pending_move()

    assert [r.payload['xy'] for r in recorder.records] == [[10, 10], [12, 10]]


def test_pending_move_flushes_before_boundary(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.context_fn = lambda view_name=None: {'tool': 'brush'}
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view, volume_node=None: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), MOVE,
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._on_mouse(
        'Red', (11, 21), PRESS,
        {'mouse_status': 'press', 'analysis_event_type': 'boundary_event'},
    )

    assert [r.event_type for r in recorder.records] == [MOVE, PRESS]
    assert recorder.records[0].ras is None
    assert recorder.records[0].payload['xy'] == [10, 20]
    assert recorder.records[1].ras is None
    assert recorder.records[1].payload['xy'] == [11, 21]


def test_brush_released_move_is_recorded(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.context_fn = lambda view_name=None: {
        'tool': 'brush',
        'axis': 0,
        'slice_idx': 3,
        'brush_radius_mm': 2.5,
    }
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view, volume_node=None: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), MOVE,
        {
            'mouse_status': 'move',
            'left_button_down': False,
            'analysis_event_type': 'trajectory_event',
        },
    )
    recorder._sample_pending_move()

    assert [r.event_type for r in recorder.records] == [MOVE]
    assert recorder.records[0].payload['mouse_button_state'] == 'released'
    assert recorder.records[0].ras is None
    assert recorder.records[0].payload['xy'] == [10, 20]
    assert 'trajectory_kind' not in recorder.records[0].payload


def test_brush_drag_move_stays_raw_when_press_was_not_seen(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.context_fn = lambda view_name=None: {
        'tool': 'brush',
        'axis': 0,
        'slice_idx': 3,
        'brush_radius_mm': 2.5,
    }
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view, volume_node=None: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), MOVE,
        {
            'mouse_status': 'move',
            'left_button_down': True,
            'analysis_event_type': 'trajectory_event',
        },
    )
    recorder._sample_pending_move()

    assert [r.event_type for r in recorder.records] == [MOVE]
    assert recorder.records[0].payload['left_button_down'] is True
    assert recorder.records[0].payload['mouse_button_state'] == 'pressed'
    assert recorder.records[0].ras is None



def test_view_changed_records_boundary_z_without_repeating_visual_state(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._volume_node = object()
    recorder.context_fn = lambda view_name=None: {
        'tool': 'brush',
        'view_name': view_name or 'Red',
        'slice_idx': 42,
    }
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view, volume_node=None: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), VIEW_CHANGED,
        {
            'mouse_status': 'view',
            'view_event': 'wheel',
            'analysis_event_type': 'trajectory_event',
        },
    )

    assert 'visual_state' not in recorder.records[0].payload
    assert recorder.records[0].payload['analysis_event_type'] == 'boundary_event'
    assert recorder.records[0].payload['slice_idx'] == 42
    assert 'trajectory_role' not in recorder.records[0].payload

    event = recorder.export_raw_data()['events'][0]
    assert event['event'] == VIEW_CHANGED
    assert event['event_type'] == 'boundary_event'
    assert event['z'] == 42

    recorder._append(MOVE, None, {
        'view_name': 'Red',
        'xy': [11, 21],
        'mouse_button_state': 'released',
    })
    events = recorder.export_raw_data()['events']
    assert events[1]['z'] == 42


def test_point_drag_records_boundary_and_non_annotation_trajectory(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view, volume_node=None: {'view_name': view})

    recorder.record_point_drag(
        'start', 'seg-a', [1.0, 2.0, 3.0], False,
        view_name='Red', point_index=2, point_id='cp-2',
        point_name='Point 2')
    recorder._last_point_drag_ts -= datetime.timedelta(milliseconds=100)
    recorder.record_point_drag(
        'move', 'seg-a', [2.0, 3.0, 4.0], False,
        view_name='Red', point_index=2, point_id='cp-2',
        point_name='Point 2')
    recorder.record_point_drag(
        'end', 'seg-a', [3.0, 4.0, 5.0], False,
        view_name='Red', point_index=2, point_id='cp-2',
        point_name='Point 2')

    assert [r.event_type for r in recorder.records] == [
        POINT_DRAG_START, POINT_DRAG_MOVE, POINT_REPLACED,
    ]
    assert recorder.records[0].payload['analysis_event_type'] == 'boundary_event'
    assert recorder.records[0].payload['point_action'] == 'grab'
    assert recorder.records[1].payload['analysis_event_type'] == 'trajectory_event'
    assert recorder.records[1].payload['trajectory_kind'] == 'non_annotation_move'
    assert recorder.records[1].payload['trajectory_role'] == 'visualization_trajectory'
    assert recorder.records[2].payload['analysis_event_type'] == 'boundary_event'
    assert recorder.records[2].payload['point_action'] == 'replace'
    assert recorder.records[1].payload['point_id'] == 'cp-2'
    assert recorder.records[2].payload['point_name'] == 'Point 2'


def test_point_removed_is_boundary_event():
    recorder = MouseEventRecorder(sample_rate_hz=12)

    recorder.record_point_removed(
        'seg-a', [1.0, 2.0, 3.0], True,
        view_name='Red', point_index=2, point_id='cp-2',
        point_name='Point 2')

    assert [r.event_type for r in recorder.records] == [POINT_REMOVED]
    payload = recorder.records[0].payload
    assert payload['analysis_event_type'] == 'boundary_event'
    assert payload['point_action'] == 'remove'
    assert payload['point_id'] == 'cp-2'
    assert payload['point_name'] == 'Point 2'
    assert payload['is_negative'] is True


def test_point_placement_is_single_release_boundary_event(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view, volume_node=None: {'view_name': view})

    recorder.record_point_placed(
        'seg-a', [1.0, 2.0, 3.0], False,
        view_name='Red', point_index=2, point_id='cp-2',
        point_name='Point 2')

    assert [r.event_type for r in recorder.records] == ['point_placed']
    payload = recorder.records[0].payload
    assert payload['analysis_event_type'] == 'boundary_event'
    assert payload['mouse_status'] == 'release'
    assert payload['point_action'] == 'place'
    assert payload['point_name'] == 'Point 2'
    assert payload['trajectory_kind'] is None
    assert payload['trajectory_role'] is None



def test_mouse_listener_records_xy_without_coordinate_conversion(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder.context_fn = lambda view_name=None: {'tool': 'brush'}
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view, volume_node=None: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), MOVE,
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._sample_pending_move()
    recorder._on_mouse(
        'Red', (11, 21), PRESS,
        {'mouse_status': 'press', 'analysis_event_type': 'boundary_event'},
    )

    assert [r.ras for r in recorder.records] == [None, None]
    assert [r.payload['xy'] for r in recorder.records] == [[10, 20], [11, 21]]


def test_point_drag_sampling_can_be_checked_before_node_work():
    recorder = MouseEventRecorder(sample_rate_hz=12)

    assert recorder.should_sample_point_drag('move') is True

    recorder._last_point_drag_ts = datetime.datetime.now()

    assert recorder.should_sample_point_drag('move') is False
    assert recorder.should_sample_point_drag('start') is True
    assert recorder.should_sample_point_drag('end') is True


# ---------------------------------------------------------------------------
# export_interpreted_data / export_raw_data
# ---------------------------------------------------------------------------

def test_interpreted_omits_non_annotation_move_from_cleaned_record():
    recorder = MouseEventRecorder()
    recorder._append(
        MOVE, [1.0, 2.0, 3.0],
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event',
         'trajectory_kind': 'non_annotation_move', 'handler': 'brush',
         'segment_id': 'seg-1'},
    )
    events = recorder.export_interpreted_data()['events']
    assert events == []


def test_interpreted_prefers_dataprobe_xy_to_ijk_over_xy_to_ras():
    identity = [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]
    xy_to_ijk = [10.0, 0.0, 0.0, 0.0,
                 0.0, 10.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 5.0,
                 0.0, 0.0, 0.0, 1.0]
    recorder = MouseEventRecorder()
    recorder._append(METADATA, None, {
        'volume': {
            'dimensions': [100, 100, 10],
            'ras_to_ijk': identity,
        },
        'initial_visual_state': {
            'Red': {
                'xy_to_ras': identity,
                'xy_to_ijk': xy_to_ijk,
                'xy_coordinate_system': 'vtk_device',
            },
        },
        'initial_handler_context': {
            'Red': {
                'view_name': 'Red',
                'handler': 'brush',
                'tool': 'brush',
                'segment_id': 'seg-a',
            },
        },
    })
    recorder._append(
        MOVE, None,
        {
            'view_name': 'Red',
            'xy': [3, 4],
            'mouse_status': 'move',
            'left_button_down': True,
            'mouse_button_state': 'pressed',
            'analysis_event_type': 'trajectory_event',
        },
    )

    events = recorder.export_interpreted_data()['events']

    assert events[0]['ijk'] == [[30, 40, 5]]


def test_interpreted_skips_raw_xy_outside_volume():
    identity = [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]
    recorder = MouseEventRecorder()
    recorder._append(METADATA, None, {
        'volume': {'dimensions': [5, 5, 5], 'ras_to_ijk': identity},
        'initial_visual_state': {'Red': {'xy_to_ras': identity, 'xy_to_ijk': identity}},
    })
    recorder._append(
        MOVE, None,
        {
            'view_name': 'Red',
            'xy': [100, 100],
            'mouse_status': 'move',
            'mouse_button_state': 'released',
            'analysis_event_type': 'trajectory_event',
        },
    )

    assert recorder.export_interpreted_data()['events'] == []


def test_interpreted_uses_strict_ijk_bounds_for_active_region():
    identity = [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]
    recorder = MouseEventRecorder()
    recorder._append(METADATA, None, {
        'volume': {
            'dimensions': [5, 5, 5],
            'ras_to_ijk': identity,
        },
        'initial_visual_state': {
            'Red': {'xy_to_ras': identity, 'xy_to_ijk': identity},
        },
        'initial_handler_context': {
            'Red': {
                'view_name': 'Red',
                'handler': 'brush',
                'tool': 'brush',
                'segment_id': 'seg-a',
            },
        },
    })
    recorder._append(
        MOVE, None,
        {
            'view_name': 'Red',
            'xy': [-0.25, 1],
            'mouse_status': 'move',
            'left_button_down': True,
            'mouse_button_state': 'pressed',
            'analysis_event_type': 'trajectory_event',
        },
    )
    recorder._append(
        MOVE, None,
        {
            'view_name': 'Red',
            'xy': [1, 1],
            'mouse_status': 'move',
            'left_button_down': True,
            'mouse_button_state': 'pressed',
            'analysis_event_type': 'trajectory_event',
        },
    )

    events = recorder.export_interpreted_data()['events']

    assert events[0]['ijk'] == [[1, 1, 0]]


def test_raw_export_filters_mouse_events_outside_ijk_bounds():
    identity = [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]
    recorder = MouseEventRecorder()
    recorder._append(METADATA, None, {
        'volume': {
            'dimensions': [5, 5, 5],
            'ras_to_ijk': identity,
        },
        'initial_visual_state': {
            'Red': {'xy_to_ras': identity, 'xy_to_ijk': identity},
        },
    })
    recorder._append(
        MOVE, None,
        {
            'view_name': 'Red',
            'xy': [-0.25, 1],
            'xy_global': [100, 200],
            'mouse_button_state': 'released',
        },
    )
    recorder._append(
        MOVE, None,
        {
            'view_name': 'Red',
            'xy': [1, 1],
            'xy_global': [101, 201],
            'mouse_button_state': 'released',
        },
    )

    events = recorder.export_raw_data()['events']

    assert [event['xy'] for event in events] == [[1, 1]]
    assert 'ijk' not in events[0]


def test_interpreted_suppresses_press_and_release():
    recorder = MouseEventRecorder()
    recorder._append(PRESS, [1.0, 2.0, 3.0],
                     {'mouse_status': 'press', 'analysis_event_type': 'boundary_event'})
    recorder._append(RELEASE, [1.0, 2.0, 3.0],
                     {'mouse_status': 'release', 'analysis_event_type': 'boundary_event'})
    events = recorder.export_interpreted_data()['events']
    assert events == []


def test_interpreted_uses_absolute_timestamps():
    identity = [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]
    recorder = MouseEventRecorder()
    recorder._append(METADATA, None, {
        'volume': {'dimensions': [10, 10, 10], 'ras_to_ijk': identity},
        'initial_visual_state': {'Red': {'xy_to_ijk': identity}},
        'initial_handler_context': {'Red': {'view_name': 'Red', 'handler': 'brush'}},
    })
    recorder._append(
        MOVE, None,
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event',
         'mouse_button_state': 'pressed', 'view_name': 'Red', 'xy': [1, 2]},
    )
    events = recorder.export_interpreted_data()['events']
    ts = events[0]['timestamp']
    assert all('T' in t for t in ts)  # ISO-8601 format with date and time
    assert 't_ms' not in events[0]


def test_raw_includes_press_release_move():
    recorder = MouseEventRecorder()
    recorder._append(PRESS, [1.0, 2.0, 3.0],
                     {'mouse_status': 'press', 'analysis_event_type': 'boundary_event',
                      'view_name': 'Red', 'slice_idx': 7,
                      'xy': [100, 200],
                      'xy_global': [100, 200]})
    recorder._append(
        MOVE, [2.0, 3.0, 4.0],
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event',
         'view_name': 'Red', 'mouse_button_state': 'pressed',
         'trajectory_kind': 'annotation_move', 'slice_idx': 7,
         'xy': [101, 201],
         'xy_global': [101, 201], 'handler': 'brush',
         'segment_id': 'seg-a', 'brush_radius_mm': 2.5},
    )
    recorder._append(RELEASE, [3.0, 4.0, 5.0],
                     {'mouse_status': 'release', 'analysis_event_type': 'boundary_event',
                      'view_name': 'Red', 'slice_idx': 7,
                      'xy': [102, 202],
                      'xy_global': [102, 202]})
    events = recorder.export_raw_data()['events']
    assert [e['event'] for e in events] == ['mouse', 'mouse', 'mouse']
    assert [e['mouse_state'] for e in events] == ['press', 'hold', 'release']
    assert events[1]['view'] == 'Red'
    assert events[1]['slice'] == 7
    assert all('xy_global' not in e for e in events)
    assert events[1]['xy'] == [[101, 201]]
    assert all('ras' not in e for e in events)
    assert all('ijk' not in e for e in events)
    assert events[1]['tool'] == 'brush'
    assert events[1]['segment'] == 'seg-a'
    assert events[1]['diameter_mm'] == 5.0


def test_raw_context_is_exported_as_deltas_not_repeated():
    recorder = MouseEventRecorder()
    recorder._append(METADATA, None, {
        'initial_handler_context': {
            'Red': {
                'view_name': 'Red',
                'handler': 'brush',
                'segment_id': 'seg-a',
                'brush_radius_mm': 2.5,
                'slice_idx': 7,
            },
        },
    })
    recorder._append(
        MOVE, None,
        {
            'view_name': 'Red',
            'xy': [10, 20],
            'xy_global': [100, 200],
            'mouse_button_state': 'pressed',
            'handler': 'brush',
            'segment_id': 'seg-a',
            'brush_radius_mm': 2.5,
        },
    )
    recorder._append(
        PRESS, None,
        {
            'view_name': 'Red',
            'xy': [11, 21],
            'xy_global': [101, 201],
            'handler': 'brush',
            'segment_id': 'seg-b',
            'brush_radius_mm': 2.5,
        },
    )

    events = recorder.export_raw_data()['events']

    assert events[0] == {
        'timestamp': recorder.records[1].timestamp.isoformat(timespec='milliseconds'),
        'event': 'mouse',
        'view': 'Red',
        'slice': 7,
        'z': 7,
        'mouse_state': 'hold',
        'tool': 'brush',
        'segment': 'seg-a',
        'diameter_mm': 5.0,
        'xy': [10, 20],
    }
    assert all('xy_global' not in e for e in events)
    assert events[1]['segment'] == 'seg-b'
    assert events[1]['tool'] == 'brush'
    assert events[1]['diameter_mm'] == 5.0


def test_raw_includes_markup_source_events_for_offline_interpretation():
    recorder = MouseEventRecorder()
    recorder.record_point_placed(
        'seg-a', [1.0, 2.0, 3.0], False,
        point_index=0, point_id='cp-0', point_name='Pos-1')
    events = recorder.export_raw_data()['events']
    assert events[0]['event'] == POINT_PLACED
    assert 'markup_ras' not in events[0]
    assert 'ras' not in events[0]
    assert 'ijk' not in events[0]
    assert events[0]['segment'] == 'seg-a'
    assert events[0]['point'] == 'cp-0'


def test_raw_uses_absolute_timestamps():
    recorder = MouseEventRecorder()
    recorder._append(
        PRESS, [1.0, 2.0, 3.0],
        {'mouse_status': 'press', 'analysis_event_type': 'boundary_event'},
    )
    events = recorder.export_raw_data()['events']
    ts = events[0]['timestamp']
    assert 'T' in ts
    assert 't_ms' not in events[0]


def test_raw_type_is_raw_input():
    recorder = MouseEventRecorder()
    result = recorder.export_raw_data()
    assert result['type'] == 'SegmentHumanBody.raw_input'


def test_interpreted_type_is_annotation_process():
    recorder = MouseEventRecorder()
    result = recorder.export_interpreted_data()
    assert result['type'] == 'SegmentHumanBody.annotation_process'
