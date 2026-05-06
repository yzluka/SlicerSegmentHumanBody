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


def _widget(seg_node, recorder=None):
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget.logic = _FakeLogic()
    widget._recorder = recorder or _FakeRecorder()
    widget._observed_segmentation = seg_node
    widget._observed_segment_ids = SegmentHumanBodyWidget._segment_ids(seg_node)
    widget._observed_segment_names = SegmentHumanBodyWidget._segment_names(seg_node)
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


def test_point_placement_records_verdict_when_point_is_defined():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget._recorder = _FakePointDragRecorder()
    widget._active_point_drags = {}
    widget._pending_point_confirmations = {}
    widget.ui = _FakeUI()
    widget.currentViewName = 'Red'
    node = _FakePromptNode(status=2)

    SegmentHumanBodyWidget._onPromptPointDefinedForRecording(
        widget, node, False)

    assert len(widget._recorder.points) == 1
    args, kwargs = widget._recorder.points[0]
    assert args[:3] == ('seg-a', [1.0, 2.0, 3.0], False)
    assert kwargs['point_index'] == 0
    assert kwargs['point_id'] == 'cp-0'
    assert kwargs['point_name'] == 'P-0'

    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'end', '0')

    assert len(widget._recorder.points) == 1


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

    assert len(widget._recorder.points) == 1
    assert widget._recorder.drags == []
    assert widget._active_point_drags == {}

    SegmentHumanBodyWidget._onPromptPointDragForRecording(
        widget, node, False, 'end', '0')

    assert len(widget._recorder.points) == 1
    assert widget._recorder.drags == []


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
