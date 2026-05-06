from core._mouse_recorder import (
    MOVE, PRESS, RELEASE, VIEW_CHANGED, POINT_DRAG_START, POINT_DRAG_MOVE,
    POINT_DRAG_END, POINT_PLACED, POINT_REMOVED, METADATA, MouseEventRecorder,
    _SliceRecordFilter,
    _SliceRecordInteractorObserver,
)
import datetime
import core._mouse_recorder as recorder_mod


def test_default_sample_rate_is_30_hz():
    recorder = MouseEventRecorder()
    assert round(1000.0 / recorder._move_interval_ms) == 30


def test_metadata_caches_initial_slice_visual_state(monkeypatch):
    monkeypatch.setattr(
        recorder_mod,
        '_all_slice_visual_state',
        lambda: {'Red': {'view_name': 'Red', 'slice_offset': 12.0}},
    )
    recorder = MouseEventRecorder()

    recorder.start(volume_node=None, segmentation_name=None)

    start = recorder.records[0]
    assert start.event_type == METADATA
    assert start.payload['initial_visual_state'] == {
        'Red': {'view_name': 'Red', 'slice_offset': 12.0},
    }


def test_metadata_records_non_annotative_movement_option(monkeypatch):
    monkeypatch.setattr(
        recorder_mod,
        '_all_slice_visual_state',
        lambda: {},
    )
    recorder = MouseEventRecorder()

    recorder.start(
        volume_node=None, segmentation_name=None,
        record_non_annotative_movement=True)

    assert recorder.records[0].payload['record_non_annotative_movement'] is True


def test_stop_does_not_append_second_metadata_or_session_stop(monkeypatch):
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda: {})
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


def test_export_ids_start_at_one_after_metadata(monkeypatch):
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda: {})
    recorder = MouseEventRecorder()

    recorder.start(volume_node=None, segmentation_name=None)
    recorder.record_action('first-process-event')

    exported = recorder.export_data()

    assert [r.event_type for r in recorder.records] == [METADATA, 'action']
    assert exported['events'][0]['id'] == 1


def test_visual_state_does_not_repeat_slice_view_dimensions(monkeypatch):
    class _SliceNode:
        def GetSliceOffset(self): return 12.0
        def GetFieldOfView(self): return [100.0, 100.0, 1.0]
        def GetDimensions(self): return [512, 512, 1]

    class _SliceWidget:
        def mrmlSliceNode(self): return _SliceNode()

    class _Layout:
        def sliceWidget(self, view_name): return _SliceWidget()

    class _App:
        def layoutManager(self): return _Layout()

    monkeypatch.setattr(recorder_mod.slicer, 'app', _App(), raising=False)

    state = recorder_mod._visual_state('Red')

    assert state == {
        'slice_offset': 12.0,
        'field_of_view': [100.0, 100.0, 1.0],
    }


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


def test_export_preserves_point_name_for_verdict_events():
    recorder = MouseEventRecorder()

    recorder.record_point_placed(
        'seg-a', [1.0, 2.0, 3.0], False,
        point_index=0, point_id='cp-0', point_name='Positive 1',
        point_action='replace')

    event = recorder.export_data()['events'][0]

    assert event['event'] == 'point_placed'
    assert event['point_action'] == 'replace'
    assert event['point'] == 'cp-0'
    assert event['point_name'] == 'Positive 1'


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
        (POINT_PLACED, [3.0, 74.0, -1.0],
         {
             'view_name': 'Red',
             'segment_id': 'seg-a',
             'handler': 'point',
             'point_id': 'cp-0',
             'point_name': 'Positive 1',
             'point_action': 'replace',
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
        POINT_DRAG_START, POINT_DRAG_MOVE, POINT_PLACED,
    ]
    assert [event['point_action'] for event in exported] == [
        'grab', 'move', 'replace',
    ]
    assert exported[-1]['ras'] == [3.0, 74.0, -1.0]


def test_wheel_events_record_view_changed_by_slice_filter(monkeypatch):
    class _QEvent:
        MouseMove = object()
        MouseButtonPress = object()
        MouseButtonRelease = object()
        Wheel = object()

    class _Pos:
        def x(self): return 10
        def y(self): return 20

    class _Delta:
        def x(self): return 0
        def y(self): return 120

    class _Event:
        def type(self):
            return recorder_mod.qt.QEvent.Wheel

        def pos(self):
            return _Pos()

        def angleDelta(self):
            return _Delta()

    calls = []
    monkeypatch.setattr(recorder_mod.qt, 'QEvent', _QEvent, raising=False)
    filt = _SliceRecordFilter('Red', lambda *args: calls.append(args))

    filt.eventFilter(None, _Event())

    assert calls == [
        ('Red', (10, 20), VIEW_CHANGED, {
            'mouse_status': 'view',
            'view_event': 'wheel',
            'wheel_delta': [0, 120],
            'analysis_event_type': 'trajectory_event',
        }),
    ]


def test_slice_filter_can_skip_mouse_move_when_vtk_captures_moves(monkeypatch):
    class _QEvent:
        MouseMove = object()

    class _Pos:
        def x(self): return 10
        def y(self): return 20

    class _Event:
        def type(self):
            return recorder_mod.qt.QEvent.MouseMove

        def pos(self):
            return _Pos()

        def buttons(self):
            return 0

    calls = []
    monkeypatch.setattr(recorder_mod.qt, 'QEvent', _QEvent, raising=False)
    filt = _SliceRecordFilter(
        'Red', lambda *args: calls.append(args), capture_moves=False)

    assert filt.eventFilter(None, _Event()) is False
    assert calls == []


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


def test_vtk_interactor_coordinates_are_flipped_to_slice_xy():
    class _SliceView:
        height = 200

    assert recorder_mod._vtk_event_xy_to_slice_xy(_SliceView(), 10, 20) == (10, 179)


def test_vtk_interactor_coordinates_prefer_render_window_height():
    class _RenderWindow:
        def GetSize(self): return (400, 400)

    class _SliceView:
        height = 200
        def renderWindow(self): return _RenderWindow()

    assert recorder_mod._vtk_event_xy_to_slice_xy(_SliceView(), 10, 20) == (10, 379)


def test_vtk_interactor_coordinates_support_callable_qt_height():
    class _SliceView:
        def height(self): return 240

    assert recorder_mod._vtk_event_xy_to_slice_xy(_SliceView(), 10, 20) == (10, 219)


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


def test_press_and_release_are_recorded_even_inside_move_sample_window(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.record_non_annotative_movement = True
    recorder.context_fn = lambda view_name=None: {
        'tool': 'brush',
        'axis': 0,
        'slice_idx': 3,
        'brush_radius_mm': 2.5,
    }
    monkeypatch.setattr(recorder_mod, '_slice_xy_to_ras', lambda view, xy: [1.0, 2.0, 3.0])
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), MOVE,
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._sample_pending_move()
    recorder._on_mouse(
        'Red', (11, 21), PRESS,
        {'mouse_status': 'press', 'analysis_event_type': 'boundary_event'},
    )
    recorder._on_mouse(
        'Red', (12, 22), RELEASE,
        {'mouse_status': 'release', 'analysis_event_type': 'boundary_event'},
    )

    assert [r.event_type for r in recorder.records] == [MOVE, PRESS, RELEASE]
    assert recorder.records[0].payload['analysis_event_type'] == 'trajectory_event'
    assert recorder.records[0].payload['trajectory_kind'] == 'non_annotation_move'
    assert recorder.records[0].payload['trajectory_role'] == 'visualization_trajectory'
    assert recorder.records[1].payload['mouse_status'] == 'press'
    assert recorder.records[1].payload['analysis_event_type'] == 'boundary_event'
    assert recorder.records[1].payload['handler'] == 'brush'
    assert recorder.records[2].payload['mouse_status'] == 'release'
    assert recorder.records[2].payload['analysis_event_type'] == 'boundary_event'


def test_move_sampling_uses_latest_position_without_duplicates(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.record_non_annotative_movement = True
    recorder.context_fn = lambda view_name=None: {'tool': None}
    ras_by_xy = {
        (10, 20): [1.0, 2.0, 3.0],
        (30, 40): [3.0, 4.0, 5.0],
    }
    monkeypatch.setattr(
        recorder_mod, '_slice_xy_to_ras',
        lambda view, xy: ras_by_xy[xy],
    )
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

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
    assert recorder.records[0].ras == [3.0, 4.0, 5.0]


def test_in_volume_move_is_kept_if_cursor_leaves_before_timer(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.record_non_annotative_movement = True
    recorder.context_fn = lambda view_name=None: {'tool': None}
    inside_ts = datetime.datetime(2026, 1, 1, 12, 0, 0)
    outside_ts = inside_ts + datetime.timedelta(milliseconds=10)
    ras_by_xy = {
        (10, 20): [1.0, 2.0, 3.0],
        (999, 999): [999.0, 999.0, 999.0],
    }
    monkeypatch.setattr(
        recorder_mod, '_slice_xy_to_ras',
        lambda view, xy: ras_by_xy[xy],
    )
    monkeypatch.setattr(
        recorder_mod, '_ras_inside_volume',
        lambda volume, ras: ras != [999.0, 999.0, 999.0],
    )
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

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
    assert recorder.records[0].ras == [1.0, 2.0, 3.0]
    assert recorder.records[0].timestamp == inside_ts


def test_pending_move_flushes_before_boundary(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.record_non_annotative_movement = True
    recorder.context_fn = lambda view_name=None: {'tool': 'brush'}
    ras_by_xy = {
        (10, 20): [1.0, 2.0, 3.0],
        (11, 21): [2.0, 3.0, 4.0],
    }
    monkeypatch.setattr(
        recorder_mod, '_slice_xy_to_ras',
        lambda view, xy: ras_by_xy[xy],
    )
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), MOVE,
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._on_mouse(
        'Red', (11, 21), PRESS,
        {'mouse_status': 'press', 'analysis_event_type': 'boundary_event'},
    )

    assert [r.event_type for r in recorder.records] == [MOVE, PRESS]
    assert recorder.records[0].ras == [1.0, 2.0, 3.0]
    assert recorder.records[1].ras == [2.0, 3.0, 4.0]


def test_non_annotative_move_is_recorded_by_listener_when_option_is_off(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.record_non_annotative_movement = False
    recorder.context_fn = lambda view_name=None: {'tool': None}
    monkeypatch.setattr(recorder_mod, '_slice_xy_to_ras', lambda view, xy: [1.0, 2.0, 3.0])
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), MOVE,
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._sample_pending_move()

    assert [r.event_type for r in recorder.records] == [MOVE]
    assert recorder.records[0].payload['trajectory_kind'] == 'non_annotation_move'


def test_non_annotative_move_is_recorded_when_option_is_on(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.record_non_annotative_movement = True
    recorder.context_fn = lambda view_name=None: {'tool': None}
    monkeypatch.setattr(recorder_mod, '_slice_xy_to_ras', lambda view, xy: [1.0, 2.0, 3.0])
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), MOVE,
        {'mouse_status': 'move', 'analysis_event_type': 'trajectory_event'},
    )
    recorder._sample_pending_move()

    assert [r.event_type for r in recorder.records] == [MOVE]
    assert recorder.records[0].payload['trajectory_kind'] == 'non_annotation_move'


def test_brush_drag_move_records_when_non_annotative_option_is_off(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.record_non_annotative_movement = False
    recorder.context_fn = lambda view_name=None: {
        'tool': 'brush',
        'axis': 0,
        'slice_idx': 3,
        'brush_radius_mm': 2.5,
    }
    monkeypatch.setattr(recorder_mod, '_slice_xy_to_ras', lambda view, xy: [1.0, 2.0, 3.0])
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), MOVE,
        {
            'mouse_status': 'move',
            'left_button_down': True,
            'analysis_event_type': 'trajectory_event',
        },
    )
    recorder._sample_pending_move()

    assert [r.event_type for r in recorder.records] == [PRESS, MOVE]
    assert recorder.records[1].payload['mouse_button_state'] == 'pressed'
    assert recorder.records[1].payload['trajectory_kind'] == 'annotation_move'


def test_brush_released_move_is_recorded_by_listener_when_non_annotative_option_is_off(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.record_non_annotative_movement = False
    recorder.context_fn = lambda view_name=None: {
        'tool': 'brush',
        'axis': 0,
        'slice_idx': 3,
        'brush_radius_mm': 2.5,
    }
    monkeypatch.setattr(recorder_mod, '_slice_xy_to_ras', lambda view, xy: [1.0, 2.0, 3.0])
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

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
    assert recorder.records[0].payload['trajectory_kind'] == 'non_annotation_move'


def test_brush_released_move_is_recorded_when_non_annotative_option_is_on(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.record_non_annotative_movement = True
    recorder.context_fn = lambda view_name=None: {
        'tool': 'brush',
        'axis': 0,
        'slice_idx': 3,
        'brush_radius_mm': 2.5,
    }
    monkeypatch.setattr(recorder_mod, '_slice_xy_to_ras', lambda view, xy: [1.0, 2.0, 3.0])
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

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
    assert recorder.records[0].payload['trajectory_kind'] == 'non_annotation_move'
    assert recorder.records[0].payload['trajectory_role'] == 'visualization_trajectory'


def test_brush_drag_move_infers_missing_press_boundary(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.context_fn = lambda view_name=None: {
        'tool': 'brush',
        'axis': 0,
        'slice_idx': 3,
        'brush_radius_mm': 2.5,
    }
    monkeypatch.setattr(recorder_mod, '_slice_xy_to_ras', lambda view, xy: [1.0, 2.0, 3.0])
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), MOVE,
        {
            'mouse_status': 'move',
            'left_button_down': True,
            'analysis_event_type': 'trajectory_event',
        },
    )
    recorder._sample_pending_move()

    assert [r.event_type for r in recorder.records] == [PRESS, MOVE]
    assert recorder.records[0].payload['mouse_status'] == 'press'
    assert recorder.records[0].payload['analysis_event_type'] == 'boundary_event'
    assert recorder.records[0].payload['boundary_source'] == 'inferred_from_first_drag_sample'
    assert recorder.records[1].payload['trajectory_role'] == 'annotation_trajectory'


def test_brush_release_infers_missing_press_boundary(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.context_fn = lambda view_name=None: {
        'tool': 'brush',
        'axis': 0,
        'slice_idx': 3,
        'brush_radius_mm': 2.5,
    }
    monkeypatch.setattr(recorder_mod, '_slice_xy_to_ras', lambda view, xy: [1.0, 2.0, 3.0])
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), RELEASE,
        {'mouse_status': 'release', 'analysis_event_type': 'boundary_event'},
    )

    assert [r.event_type for r in recorder.records] == [PRESS, RELEASE]
    assert recorder.records[0].payload['boundary_source'] == 'inferred_from_release'
    assert recorder.records[1].payload['mouse_status'] == 'release'


def test_visualization_trajectory_role_for_view_events(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._volume_node = object()
    monkeypatch.setattr(recorder_mod, '_slice_xy_to_ras', lambda view, xy: [1.0, 2.0, 3.0])
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), VIEW_CHANGED,
        {
            'mouse_status': 'view',
            'view_event': 'wheel',
            'analysis_event_type': 'trajectory_event',
        },
    )

    assert recorder.records[0].payload['trajectory_role'] == 'visualization_trajectory'


def test_point_drag_records_boundary_and_non_annotation_trajectory(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

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
        POINT_DRAG_START, POINT_DRAG_MOVE, 'point_placed',
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
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

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


def test_raw_point_press_release_are_recorded_as_listener_events(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.context_fn = lambda view_name=None: {'tool': 'point'}
    monkeypatch.setattr(recorder_mod, '_slice_xy_to_ras', lambda view, xy: [1.0, 2.0, 3.0])
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

    recorder._on_mouse(
        'Red', (10, 20), PRESS,
        {'mouse_status': 'press', 'analysis_event_type': 'boundary_event'},
    )
    recorder._on_mouse(
        'Red', (10, 20), RELEASE,
        {'mouse_status': 'release', 'analysis_event_type': 'boundary_event'},
    )

    assert [r.event_type for r in recorder.records] == [PRESS, RELEASE]
    assert recorder.records[0].payload['handler'] == 'point'
    assert recorder.records[1].payload['handler'] == 'point'


def test_point_held_move_records_non_annotation_when_non_annotative_option_is_off(monkeypatch):
    recorder = MouseEventRecorder(sample_rate_hz=12)
    recorder._active = True
    recorder._volume_node = object()
    recorder.record_non_annotative_movement = False
    recorder.context_fn = lambda view_name=None: {'tool': 'point'}
    monkeypatch.setattr(recorder_mod, '_slice_xy_to_ras', lambda view, xy: [1.0, 2.0, 3.0])
    monkeypatch.setattr(recorder_mod, '_ras_inside_volume', lambda volume, ras: True)
    monkeypatch.setattr(recorder_mod, '_visual_state', lambda view: {'view_name': view})

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
    assert recorder.records[0].payload['mouse_button_state'] == 'pressed'
    assert recorder.records[0].payload['trajectory_kind'] == 'non_annotation_move'
    assert recorder.records[0].payload['trajectory_role'] == 'visualization_trajectory'


def test_point_drag_sampling_can_be_checked_before_node_work():
    recorder = MouseEventRecorder(sample_rate_hz=12)

    assert recorder.should_sample_point_drag('move') is True

    recorder._last_point_drag_ts = datetime.datetime.now()

    assert recorder.should_sample_point_drag('move') is False
    assert recorder.should_sample_point_drag('start') is True
    assert recorder.should_sample_point_drag('end') is True
