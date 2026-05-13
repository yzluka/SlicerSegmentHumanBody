from unittest.mock import MagicMock

from SegmentHumanBody import SegmentHumanBodyLogic, SegmentHumanBodyWidget


class _Volume:
    def __init__(self, name):
        self._name = name
        self._id = f'vtkMRMLScalarVolumeNode_{name}'

    def GetStorageNode(self):
        return None

    def GetName(self):
        return self._name

    def GetID(self):
        return self._id


class _Segmentation:
    def __init__(self, ids):
        self._ids = list(ids)

    def GetNumberOfSegments(self):
        return len(self._ids)

    def GetNthSegmentID(self, index):
        return self._ids[index]


class _SegNode:
    def __init__(self, ids):
        self._segmentation = _Segmentation(ids)

    def GetSegmentation(self):
        return self._segmentation


def test_select_relative_segment_wraps_forward_and_backward():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget.logic = SegmentHumanBodyLogic.__new__(SegmentHumanBodyLogic)
    widget.logic.segment_ids = lambda seg: ['A', 'B', 'C']
    widget.ui = MagicMock()
    widget.ui.segmentationNodeSelector.currentNode.return_value = _SegNode(['A', 'B', 'C'])
    widget.ui.segmentSelector.currentSegmentID.return_value = 'C'

    widget._select_relative_segment(1)
    widget.ui.segmentSelector.setCurrentSegmentID.assert_called_with('A')

    widget.ui.segmentSelector.setCurrentSegmentID.reset_mock()
    widget.ui.segmentSelector.currentSegmentID.return_value = 'A'
    widget._select_relative_segment(-1)
    widget.ui.segmentSelector.setCurrentSegmentID.assert_called_with('C')


def test_select_relative_volume_wraps_without_touching_segmentation():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    first, second = _Volume('first'), _Volume('second')
    widget.logic = MagicMock()
    widget.logic.scalar_volume_nodes.return_value = [first, second]
    widget.ui = MagicMock()
    widget.ui.sourceVolumeSelector.currentNode.return_value = first
    widget._sync_selected_nodes_to_views = MagicMock()

    widget._select_relative_volume(1)

    widget.ui.sourceVolumeSelector.setCurrentNode.assert_called_with(second)
    widget.ui.segmentationNodeSelector.setCurrentNode.assert_not_called()


def test_q_and_s_shortcut_targets_toggle_existing_checkboxes():
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget.ui = MagicMock()

    widget._toggle_current_segment_visibility()
    widget._toggle_saved_segments_visibility()

    widget.ui.showCurrentSegmentCheckBox.toggle.assert_called_once()
    widget.ui.showSegmentsCheckBox.toggle.assert_called_once()
