import SegmentHumanBody as segment_module
from SegmentHumanBody import SegmentHumanBodyWidget
from core._input import PointHandler as _PointHandlerBase


class _FakeSegment:
    def __init__(self, name):
        self._name = name

    def GetName(self):
        return self._name


class _FakeSegmentation:
    def __init__(self):
        self._segments = {}

    def GetNumberOfSegments(self):
        return len(self._segments)

    def GetNthSegmentID(self, index):
        return list(self._segments)[index]

    def GetSegment(self, segment_id):
        return self._segments.get(segment_id)

    def add(self, segment_id, name):
        self._segments[segment_id] = _FakeSegment(name)

    def remove(self, segment_id):
        self._segments.pop(segment_id)


class _FakeSegNode:
    def __init__(self):
        self._segmentation = _FakeSegmentation()

    def GetSegmentation(self):
        return self._segmentation


class _FakeLogic:
    def __init__(self):
        self.prompt_nodes_created = []
        self.synced_prompt_names = []

    def create_segment_prompt_nodes(self, seg_node, segment_id):
        self.prompt_nodes_created.append((seg_node, segment_id))

    def sync_prompt_node_names(self, seg_node, segment_id):
        self.synced_prompt_names.append((seg_node, segment_id))


class _FakeRecorder:
    is_active = True

    def __init__(self):
        self.created = []
        self.removed = []
        self.renamed = []

    def record_segment_created(self, segment_id, seg_name):
        self.created.append((segment_id, seg_name))

    def record_segment_removed(self, segment_id, seg_name):
        self.removed.append((segment_id, seg_name))

    def record_segment_renamed(self, segment_id, old_name, new_name):
        self.renamed.append((segment_id, old_name, new_name))


class _RestartRecorder:
    def __init__(self, *, active=False, count=0):
        self.is_active = active
        self.count = count
        self.started = 0
        self.stopped = 0
        self.cleared = 0
        self.saved_paths = []

    def __len__(self):
        return self.count

    def start(self, **kwargs):
        self.started += 1
        self.start_kwargs = kwargs
        self.is_active = True
        self.count = 1

    def stop(self):
        self.stopped += 1
        self.is_active = False

    def clear(self):
        self.cleared += 1
        self.count = 0

    def save_to_file(self, path):
        self.saved_paths.append(path)


class _TextWidget:
    def __init__(self):
        self.text = ''
        self.visible = None
        self.enabled = None

    def setText(self, text):
        self.text = text

    def setVisible(self, visible):
        self.visible = visible

    def setEnabled(self, enabled):
        self.enabled = enabled


class _ComboBoxStub:
    currentIndex = 0
    def itemData(self, idx): return -1
    def setEnabled(self, v): pass
    def clear(self): pass
    def addItem(self, text, data=None): pass


class _CheckBoxStub:
    def __init__(self, checked=True):
        self._checked = checked
        self.enabled = True

    def isChecked(self):
        return self._checked

    def setChecked(self, v):
        self._checked = v

    def setEnabled(self, v):
        self.enabled = v

    def blockSignals(self, v):
        pass


class _RecordUI:
    def __init__(self):
        self.recordToggleButton = _TextWidget()
        self.recordMouseKeyCheckBox = _CheckBoxStub(checked=True)
        self.recordAudioCheckBox = _CheckBoxStub(checked=True)
        self.exportRecordButton = _TextWidget()
        self.recordStatusLabel = _TextWidget()
        self.audioDeviceComboBox = _ComboBoxStub()


def _widget(seg_node, recorder=None):
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget.logic = _FakeLogic()
    widget._recorder = recorder or _FakeRecorder()
    widget._observed_segmentation = seg_node
    widget._observed_segment_ids = SegmentHumanBodyWidget._segment_ids(seg_node)
    widget._observed_segment_names = SegmentHumanBodyWidget._segment_names(seg_node)
    return widget


def _record_widget(recorder):
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = recorder
    widget._recording_saved = False
    widget._parameterNode = None
    widget._audio_recorder = None
    widget._audio_only_mode = False
    widget.ui = _RecordUI()
    widget._prompt_place_states = lambda: [False, False]
    widget._set_prompt_place_states = lambda states: None
    return widget


def test_segment_added_event_creates_prompt_nodes_and_records_creation():
    seg_node = _FakeSegNode()
    recorder = _FakeRecorder()
    widget = _widget(seg_node, recorder)

    seg_node.GetSegmentation().add('seg-a', 'Femur')
    widget._onSegmentAdded(seg_node.GetSegmentation(), None, 'seg-a')

    assert recorder.created == [('seg-a', 'Femur')]
    assert widget.logic.prompt_nodes_created == [(seg_node, 'seg-a')]
    assert widget._observed_segment_ids == {'seg-a'}
    assert widget._observed_segment_names == {'seg-a': 'Femur'}


def test_start_recording_cancels_restart_when_unsaved_record_is_kept():
    recorder = _RestartRecorder(active=False, count=3)
    widget = _record_widget(recorder)
    widget._prompt_unsaved_recording = lambda: 'cancel'

    widget.onRecordToggle()

    assert recorder.stopped == 0
    assert recorder.cleared == 0
    assert recorder.started == 0
    assert widget._recording_saved is False


def test_start_recording_discards_unsaved_record_and_restarts():
    recorder = _RestartRecorder(active=False, count=3)
    widget = _record_widget(recorder)
    widget._prompt_unsaved_recording = lambda: 'discard'

    widget.onRecordToggle()

    assert recorder.stopped == 0
    assert recorder.cleared == 1
    assert recorder.started == 1
    assert widget._recording_saved is False
    assert widget.ui.recordToggleButton.text == 'Stop Recording'


def test_start_recording_saves_unsaved_record_before_restart():
    recorder = _RestartRecorder(active=False, count=3)
    widget = _record_widget(recorder)
    widget._prompt_unsaved_recording = lambda: 'save'
    saved = []

    def _save():
        saved.append(True)
        widget._recording_saved = True
        return True

    widget._save_recording_to_user_path = _save

    widget.onRecordToggle()

    assert saved == [True]
    assert recorder.cleared == 1
    assert recorder.started == 1
    assert widget._recording_saved is False


def test_start_recording_does_not_restart_when_save_is_cancelled():
    recorder = _RestartRecorder(active=False, count=3)
    widget = _record_widget(recorder)
    widget._prompt_unsaved_recording = lambda: 'save'
    widget._save_recording_to_user_path = lambda: False

    widget.onRecordToggle()

    assert recorder.cleared == 0
    assert recorder.started == 0
    assert widget._recording_saved is False


def test_record_ui_marks_unsaved_recording():
    recorder = _RestartRecorder(active=False, count=4)
    widget = _record_widget(recorder)

    widget._update_record_ui()

    assert widget.ui.recordToggleButton.text == 'Start Recording'
    assert widget.ui.exportRecordButton.enabled is True
    assert widget.ui.recordStatusLabel.text == 'Recorded: 4 events (unsaved)'


def test_segment_removed_event_records_previous_name():
    seg_node = _FakeSegNode()
    seg_node.GetSegmentation().add('seg-a', 'Femur')
    recorder = _FakeRecorder()
    widget = _widget(seg_node, recorder)

    seg_node.GetSegmentation().remove('seg-a')
    widget._onSegmentRemoved(seg_node.GetSegmentation(), None, 'seg-a')

    assert recorder.removed == [('seg-a', 'Femur')]
    assert widget._observed_segment_ids == set()
    assert widget._observed_segment_names == {}


def test_segment_lifecycle_events_do_not_record_when_recorder_inactive():
    class _InactiveRecorder(_FakeRecorder):
        is_active = False

    seg_node = _FakeSegNode()
    recorder = _InactiveRecorder()
    widget = _widget(seg_node, recorder)

    seg_node.GetSegmentation().add('seg-a', 'Femur')
    widget._onSegmentAdded(seg_node.GetSegmentation(), None, 'seg-a')
    seg_node.GetSegmentation().remove('seg-a')
    widget._onSegmentRemoved(seg_node.GetSegmentation(), None, 'seg-a')

    assert recorder.created == []
    assert recorder.removed == []


def test_segment_modified_skips_prompt_sync_when_name_did_not_change():
    seg_node = _FakeSegNode()
    seg_node.GetSegmentation().add('seg-a', 'Femur')
    widget = _widget(seg_node)

    widget._onSegmentModified(seg_node.GetSegmentation(), None, 'seg-a')

    assert widget.logic.synced_prompt_names == []


def test_segment_modified_syncs_prompt_names_only_when_name_changes():
    seg_node = _FakeSegNode()
    seg_node.GetSegmentation().add('seg-a', 'Femur')
    widget = _widget(seg_node)
    seg_node.GetSegmentation().add('seg-a', 'Femur renamed')

    widget._onSegmentModified(seg_node.GetSegmentation(), None, 'seg-a')

    assert widget._recorder.renamed == [
        ('seg-a', 'Femur', 'Femur renamed'),
    ]
    assert widget.logic.synced_prompt_names == [(seg_node, 'seg-a')]


class _FakeSegmentSelector:
    def currentSegmentID(self):
        return 'seg-a'


class _FakeUI:
    segmentSelector = _FakeSegmentSelector()
    segmentationNodeSelector = type('_Selector', (), {'currentNode': lambda self: None})()


class _FakePromptNode:
    def __init__(self, status):
        self._status = status
        self._removed = False

    def GetID(self):
        return 'node-a'

    def GetNumberOfControlPoints(self):
        return 0 if self._removed else 1

    def GetNthControlPointPositionStatus(self, index):
        return self._status

    def GetNthControlPointPositionWorld(self, index, ras):
        ras[:] = [1.0, 2.0, 3.0]

    def GetNthControlPointID(self, index):
        if not self._removed and index == 0:
            return 'cp-0'
        return None

    def GetNthControlPointLabel(self, index):
        if not self._removed and index == 0:
            return 'P-0'
        return None

    def simulate_removal(self):
        """Put node in post-removal state, matching real Slicer when PointRemovedEvent fires."""
        self._removed = True


class _MovingPromptNode(_FakePromptNode):
    def __init__(self, positions):
        super().__init__(status=2)
        self.positions = [list(pos) for pos in positions]

    def GetNthControlPointPositionWorld(self, index, ras):
        ras[:] = self.positions.pop(0) if self.positions else [4.0, 5.0, 6.0]


class _TwoPointPromptNode(_FakePromptNode):
    def GetNumberOfControlPoints(self):
        return 2

    def GetNthControlPointPositionWorld(self, index, ras):
        ras[:] = [float(index), float(index + 1), float(index + 2)]

    def GetNthControlPointID(self, index):
        return f'cp-{index}'

    def GetNthControlPointLabel(self, index):
        return f'P-{index}'


class _FakePointDragRecorder:
    is_active = True
    _active_mouse_press = False  # required by _onPromptPointRemovedForRecording

    def __init__(self):
        self.drags = []
        self.points = []
        self.removed_points = []

    def record_point_drag(self, *args, **kwargs):
        self.drags.append((args, kwargs))

    def record_point_placed(self, *args, **kwargs):
        self.points.append((args, kwargs))

    def record_point_removed(self, *args, **kwargs):
        self.removed_points.append((args, kwargs))

    def should_sample_point_drag(self, phase):
        return True


def test_prompt_node_observers_request_integer_calldata():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorded_prompt_node_ids = set()
    observers = []
    widget.addObserver = lambda node, event, method: observers.append((event, method))
    node = _FakePromptNode(status=2)

    SegmentHumanBodyWidget._observe_prompt_node_for_recording(widget, node, False)

    assert observers
    assert all(
        getattr(method, 'CallDataType', None) == segment_module.vtk.VTK_INT
        for _, method in observers
    )


def test_point_defined_uses_calldata_index_not_last_defined_point():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget._pending_drag_removals = {}
    widget._recently_placed = {}
    widget._recorded_prompt_point_cache = {}
    widget._segment_id_for_prompt_node = lambda node: 'seg-a'
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _TwoPointPromptNode(status=2)

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, False, 0)

    assert len(widget._recorder.points) == 1
    args, kwargs = widget._recorder.points[0]
    assert args[:3] == ('seg-a', [0.0, 1.0, 2.0], False)
    assert kwargs['point_index'] == 0
    assert kwargs['point_id'] == 'cp-0'


def test_point_drag_recording_ignores_not_yet_defined_points():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=0)

    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'start', '0')

    assert widget._recorder.drags == []
    assert widget._active_point_drags == {}


def test_point_drag_move_throttle_happens_before_node_lookup():
    class _Recorder(_FakePointDragRecorder):
        def should_sample_point_drag(self, phase):
            return False

    class _ExplodingNode:
        def GetID(self):
            raise AssertionError('node should not be touched when throttled')

    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _Recorder()
    widget._active_point_drags = {('node-a', False): 0}
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'

    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, _ExplodingNode(), False, 'move', '0')

    assert widget._recorder.drags == []


def test_point_drag_move_requires_accepted_drag_start():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)

    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'move', '0')

    assert widget._recorder.drags == []
    assert widget._active_point_drags == {}


def test_point_drag_recording_accepts_previously_defined_points():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget._recorded_prompt_point_cache = {}
    widget._segment_id_for_prompt_node = lambda node: 'seg-a'
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)

    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'start', '0')

    assert len(widget._recorder.drags) == 1
    args, kwargs = widget._recorder.drags[0]
    assert args[:4] == ('start', 'seg-a', [1.0, 2.0, 3.0], False)
    assert kwargs['point_index'] == 0
    assert kwargs['point_id'] == 'cp-0'


def test_point_placement_records_release_verdict_after_point_is_defined():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget._pending_drag_removals = {}
    widget._recently_placed = {}
    widget._recorded_prompt_point_cache = {}
    widget._segment_id_for_prompt_node = lambda node: 'seg-a'
    widget._schedule_pending_point_confirmation = lambda *_: None
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)
    widget._recorder._active_mouse_press = True

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, False)

    assert widget._recorder.points == []

    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'end', '0')

    assert len(widget._recorder.points) == 1
    args, kwargs = widget._recorder.points[0]
    assert args[:3] == ('seg-a', [1.0, 2.0, 3.0], False)
    assert kwargs['point_index'] == 0
    assert kwargs['point_id'] == 'cp-0'
    assert kwargs['point_name'] == 'P-0'


def test_point_placement_confirms_on_release_when_end_event_is_missing():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget._pending_drag_removals = {}
    widget._recently_placed = {}
    widget._recorded_prompt_point_cache = {}
    widget._segment_id_for_prompt_node = lambda node: 'seg-a'
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)
    widget._recorder._active_mouse_press = True

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, False)
    assert widget._recorder.points == []

    widget._recorder._active_mouse_press = False
    SegmentHumanBodyWidget._confirm_all_pending_points(widget)

    assert len(widget._recorder.points) == 1
    assert widget._pending_point_confirmations == {}


def test_pending_point_start_does_not_record_relocation_drag():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget._pending_drag_removals = {}
    widget._recently_placed = {}
    widget._recorded_prompt_point_cache = {}
    widget._segment_id_for_prompt_node = lambda node: 'seg-a'
    widget._schedule_pending_point_confirmation = lambda *_: None
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)
    widget._recorder._active_mouse_press = True

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, False)
    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'start', '0')

    assert widget._recorder.points == []
    assert widget._recorder.drags == []
    assert widget._active_point_drags == {}

    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'end', '0')

    assert len(widget._recorder.points) == 1
    assert widget._recorder.drags == []


def test_pending_point_remove_is_treated_as_unconfirmed_placement_cancel():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget._pending_drag_removals = {}
    widget._recently_placed = {}
    widget._recorded_prompt_point_cache = {}
    widget._segment_id_for_prompt_node = lambda node: 'seg-a'
    widget._schedule_pending_point_confirmation = lambda *_: None
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)
    widget._recorder._active_mouse_press = True

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, False)
    node.simulate_removal()
    SegmentHumanBodyWidget._onPromptPointRemovedForRecording(
        widget, node, False, '0')

    assert widget._recorder.points == []
    assert widget._recorder.removed_points == []
    assert widget._pending_point_confirmations == {}


def test_point_remove_during_active_drag_is_not_semantic_deletion():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget._pending_drag_removals = {}
    widget._recently_placed = {}
    widget._recorded_prompt_point_cache = {}
    widget._segment_id_for_prompt_node = lambda node: 'seg-a'
    widget._schedule_pending_point_confirmation = lambda *_: None
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, False)
    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'end', '0')
    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'start', '0')
    node.simulate_removal()
    SegmentHumanBodyWidget._onPromptPointRemovedForRecording(
        widget, node, False, '0')
    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'end', '0')

    assert widget._recorder.removed_points == []
    assert [entry[0][0] for entry in widget._recorder.drags] == ['start']


def test_point_drag_end_without_displacement_is_not_replacement():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget._recorded_prompt_point_cache = {}
    widget._segment_id_for_prompt_node = lambda node: 'seg-a'
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)

    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'start', '0')
    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'end', '0')

    assert [entry[0][0] for entry in widget._recorder.drags] == ['start']
    assert widget._active_point_drags == {}


def test_point_drag_end_with_displacement_records_replacement():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget._recorded_prompt_point_cache = {}
    widget._segment_id_for_prompt_node = lambda node: 'seg-a'
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _MovingPromptNode([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'start', '0')
    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'end', '0')

    assert [entry[0][0] for entry in widget._recorder.drags] == ['start', 'end']
    assert widget._recorder.drags[-1][0][2] == [4.0, 5.0, 6.0]
    assert widget._active_point_drags == {}


def test_point_removal_records_cached_control_point():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget._pending_drag_removals = {}
    widget._recently_placed = {}
    widget._recorded_prompt_point_cache = {}
    widget._segment_id_for_prompt_node = lambda node: 'seg-a'
    widget._schedule_pending_point_confirmation = lambda *_: None
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, True)
    SegmentHumanBodyWidget._confirm_pending_point(widget, node, 0)
    # Arm the newly placed point for real deletion without relying on elapsed
    # time. This test verifies record_point_removed, not internal Slicer
    # remove/recreate suppression.
    SegmentHumanBodyWidget._arm_recorded_prompt_points_for_deletion(widget)
    node.simulate_removal()
    SegmentHumanBodyWidget._onPromptPointRemovedForRecording(
        widget, node, True, '0')

    assert len(widget._recorder.removed_points) == 1
    args, kwargs = widget._recorder.removed_points[0]
    assert args[:3] == ('seg-a', [1.0, 2.0, 3.0], True)
    assert kwargs['point_id'] == 'cp-0'
    assert kwargs['point_name'] == 'P-0'
    assert 'point_index' not in kwargs  # dropped: unreliable after removal+renumber


def test_fresh_point_internal_remove_is_suppressed_until_next_press():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget._pending_drag_removals = {}
    widget._recently_placed = {}
    widget._recorded_prompt_point_cache = {}
    widget._segment_id_for_prompt_node = lambda node: 'seg-a'
    widget._schedule_pending_point_confirmation = lambda *_: None
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, False)
    SegmentHumanBodyWidget._confirm_pending_point(widget, node, 0)
    node.simulate_removal()
    SegmentHumanBodyWidget._onPromptPointRemovedForRecording(
        widget, node, False, '0')

    assert widget._recorder.removed_points == []
    assert (node.GetID(), 'cp-0') in widget._recorded_prompt_point_cache

    SegmentHumanBodyWidget._arm_recorded_prompt_points_for_deletion(widget)
    SegmentHumanBodyWidget._onPromptPointRemovedForRecording(
        widget, node, False, '0')

    assert len(widget._recorder.removed_points) == 1


# ─────────────────────────────────────────────────────────────────────
# Segment-delete guard regression tests
# Covers three bugs fixed together:
#   • F-1 spurious fiducial: _configureUnlimitedPlacement called on None node
#   • no-tool auto-switch: _set_prompt_nodes didn't force place states off
#   • Qt error: setPlaceModeEnabled(True) called after last segment deleted
# ─────────────────────────────────────────────────────────────────────

class _NopPointHandler(_PointHandlerBase):
    """PointHandler with Qt-free _on_detach for pure-Python unit tests."""
    def _on_detach(self, widget):
        pass


class _ExtendedFakeRecorder(_FakeRecorder):
    """_FakeRecorder that also handles record_tool_selected (called by detach)."""
    def record_tool_selected(self, tool, segment_id=None):
        pass

    def record_action(self, name):
        pass


class _DeleteLogic:
    def delete_segment_prompt_nodes(self, seg_node, segment_id):
        pass

    def remove_segment(self, seg_node, segment_id):
        seg_node.GetSegmentation().remove(segment_id)


def _onremove_widget(seg_node, handler=None, prompt_widget=None):
    """Minimal SegmentHumanBodyWidget stub for _onRemoveSegment testing."""
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _ExtendedFakeRecorder()
    widget._active_handler = handler
    widget._attaching_handler = None
    widget._active_prompt_widget = prompt_widget
    widget._suppressing_place_mode = False
    widget.logic = _DeleteLogic()
    widget._set_prompt_nodes = lambda pos, neg: None
    widget._ensure_current_prompt_nodes = lambda: None
    widget._record_action = lambda name: None

    class _SegSelector:
        def currentSegmentID(self_):
            seg = seg_node.GetSegmentation()
            return seg.GetNthSegmentID(0) if seg.GetNumberOfSegments() else ''
        def setCurrentSegmentID(self_, id_):
            pass

    class _FakeRemoveUI:
        segmentSelector = _SegSelector()
        segmentationNodeSelector = type(
            '_N', (), {'currentNode': lambda self_: seg_node})()

    widget.ui = _FakeRemoveUI()
    return widget


def test_configure_unlimited_placement_not_called_when_nodes_are_none(monkeypatch):
    """F-1 regression: _configureUnlimitedPlacement must not fire when nodes are None.

    Calling it with a None-backed widget causes Slicer to auto-create a default
    "F-1" fiducial node and silently activate placement mode.
    """
    configure_calls = []
    monkeypatch.setattr(
        SegmentHumanBodyWidget,
        '_configureUnlimitedPlacement',
        staticmethod(lambda mw: configure_calls.append(mw)),
    )

    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._suppressing_place_mode = False

    class _FakeMarkupWidget:
        def setCurrentNode(self, node):
            pass
        def findChildren(self, t):
            return []

    class _FakeUI:
        positivePrompts = _FakeMarkupWidget()
        negativePrompts = _FakeMarkupWidget()

    widget.ui = _FakeUI()
    widget._prompt_place_states = lambda: [False, False]
    widget._set_prompt_place_states = lambda s: None
    widget._observe_prompt_node_for_recording = lambda n, is_negative: None

    widget._set_prompt_nodes_preserving_place_mode(None, None)

    assert configure_calls == [], (
        'F-1 regression: _configureUnlimitedPlacement was called when both nodes '
        'are None — this creates a spurious Slicer "F-1" fiducial node'
    )


def test_set_prompt_nodes_forces_place_states_off_when_no_handler_active(monkeypatch):
    """No-tool regression: segment delete must not auto-switch cursor to point mode.

    When _active_handler is not PointHandler, _set_prompt_nodes must pass
    force_states=[False, False] so programmatic node rewiring cannot re-activate
    placement mode through a deferred Qt signal.
    """
    calls = []

    def _spy(pos_node, neg_node, force_states=None):
        calls.append(force_states)

    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._active_handler = None
    widget._set_prompt_nodes_preserving_place_mode = _spy

    widget._set_prompt_nodes(None, None)

    assert calls == [[False, False]], (
        'no-tool regression: _set_prompt_nodes did not pass force_states=[False, False] '
        'when no handler is active — segment deletion auto-activates point mode'
    )


def test_remove_last_segment_with_point_handler_does_not_reactivate_place_mode():
    """Qt-error regression: setPlaceModeEnabled(True) must not be called when no segment remains.

    Deleting the last segment while PointHandler is active used to call
    setPlaceModeEnabled(True) unconditionally in the finally block, which raised:
      "[Qt] void qSlicerMarkupsPlaceWidget::setPlaceModeEnabled(bool) activate failed:
       Markups module logic, scene, or interaction node is invalid"
    """
    activate_calls = []

    class _TrackingPlaceWidget:
        def setPlaceModeEnabled(self, active):
            if active:
                activate_calls.append(True)

    class _TrackingPromptWidget:
        def findChildren(self, t):
            return [_TrackingPlaceWidget()]

    seg_node = _FakeSegNode()
    seg_node.GetSegmentation().add('seg-a', 'Femur')

    handler = _NopPointHandler()
    widget = _onremove_widget(
        seg_node, handler=handler, prompt_widget=_TrackingPromptWidget())

    SegmentHumanBodyWidget._onRemoveSegment(widget)

    assert activate_calls == [], (
        'Qt error regression: setPlaceModeEnabled(True) called after deleting '
        'the last segment with PointHandler active'
    )
