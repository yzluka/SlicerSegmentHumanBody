import SegmentHumanBody as segment_module
from SegmentHumanBody import SegmentHumanBodyWidget


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


class _RecordUI:
    def __init__(self):
        self.recordButton = _TextWidget()
        self.stopRecordButton = _TextWidget()
        self.exportRecordButton = _TextWidget()
        self.recordStatusLabel = _TextWidget()


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
    widget.ui = _RecordUI()
    widget._prompt_place_states = lambda: [False, False]
    widget._set_prompt_place_states = lambda states: None
    return widget


def test_segment_added_event_creates_prompt_nodes_without_recording_creation():
    seg_node = _FakeSegNode()
    recorder = _FakeRecorder()
    widget = _widget(seg_node, recorder)

    seg_node.GetSegmentation().add('seg-a', 'Femur')
    widget._onSegmentAdded(seg_node.GetSegmentation(), None, 'seg-a')

    assert recorder.created == []
    assert widget.logic.prompt_nodes_created == [(seg_node, 'seg-a')]
    assert widget._observed_segment_ids == {'seg-a'}
    assert widget._observed_segment_names == {'seg-a': 'Femur'}


def test_start_recording_cancels_restart_when_unsaved_record_is_kept():
    recorder = _RestartRecorder(active=True, count=3)
    widget = _record_widget(recorder)
    widget._prompt_unsaved_recording = lambda: 'cancel'

    widget.onRecord()

    assert recorder.stopped == 0
    assert recorder.cleared == 0
    assert recorder.started == 0
    assert widget._recording_saved is False


def test_start_recording_discards_unsaved_record_and_restarts():
    recorder = _RestartRecorder(active=True, count=3)
    widget = _record_widget(recorder)
    widget._prompt_unsaved_recording = lambda: 'discard'

    widget.onRecord()

    assert recorder.stopped == 1
    assert recorder.cleared == 1
    assert recorder.started == 1
    assert widget._recording_saved is False
    assert widget.ui.recordButton.text == 'Restart Record'


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

    widget.onRecord()

    assert saved == [True]
    assert recorder.cleared == 1
    assert recorder.started == 1
    assert widget._recording_saved is False


def test_start_recording_does_not_restart_when_save_is_cancelled():
    recorder = _RestartRecorder(active=False, count=3)
    widget = _record_widget(recorder)
    widget._prompt_unsaved_recording = lambda: 'save'
    widget._save_recording_to_user_path = lambda: False

    widget.onRecord()

    assert recorder.cleared == 0
    assert recorder.started == 0
    assert widget._recording_saved is False


def test_record_ui_marks_unsaved_recording():
    recorder = _RestartRecorder(active=False, count=4)
    widget = _record_widget(recorder)

    widget._update_record_ui()

    assert widget.ui.recordButton.visible is True
    assert widget.ui.recordButton.text == 'Restart Record'
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

    def GetID(self):
        return 'node-a'

    def GetNumberOfControlPoints(self):
        return 1

    def GetNthControlPointPositionStatus(self, index):
        return self._status

    def GetNthControlPointPositionWorld(self, index, ras):
        ras[:] = [1.0, 2.0, 3.0]

    def GetNthControlPointID(self, index):
        return 'cp-0'

    def GetNthControlPointLabel(self, index):
        return 'P-0'


class _MovingPromptNode(_FakePromptNode):
    def __init__(self, positions):
        super().__init__(status=2)
        self.positions = [list(pos) for pos in positions]

    def GetNthControlPointPositionWorld(self, index, ras):
        ras[:] = self.positions.pop(0) if self.positions else [4.0, 5.0, 6.0]


class _FakePointDragRecorder:
    is_active = True

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
    widget._recorded_prompt_point_cache = {}
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)

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


def test_point_placement_fallback_confirms_when_end_event_is_missing(monkeypatch):
    callbacks = []

    class _Timer:
        @staticmethod
        def singleShot(delay_ms, callback):
            callbacks.append((delay_ms, callback))

    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget._recorded_prompt_point_cache = {}
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)
    monkeypatch.setattr(segment_module.qt, 'QTimer', _Timer, raising=False)

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, False)
    assert widget._recorder.points == []
    assert callbacks[0][0] == 500

    callbacks[0][1]()

    assert len(widget._recorder.points) == 1
    assert widget._pending_point_confirmations == {}


def test_pending_point_start_does_not_record_relocation_drag():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)

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
    widget._recorded_prompt_point_cache = {}
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, False)
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
    widget._recorded_prompt_point_cache = {}
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, False)
    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'end', '0')
    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'start', '0')
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
    widget._recorded_prompt_point_cache = {}
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, True)
    SegmentHumanBodyWidget._confirm_pending_point(widget, node, True, 0)
    SegmentHumanBodyWidget._onPromptPointRemovedForRecording(
        widget, node, True, '0')

    assert len(widget._recorder.removed_points) == 1
    args, kwargs = widget._recorder.removed_points[0]
    assert args[:3] == ('seg-a', [1.0, 2.0, 3.0], True)
    assert kwargs['point_index'] == 0
    assert kwargs['point_id'] == 'cp-0'
    assert kwargs['point_name'] == 'P-0'
