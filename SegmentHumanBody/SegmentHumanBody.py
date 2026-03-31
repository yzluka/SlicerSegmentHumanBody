import numpy as np
import qt
import vtk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin
from behavior import InteractiveBehavior, SPXBehavior, AutoBehavior
# =========================
# Module
# =========================

class SegmentHumanBody(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "SegmentHumanBody (Modular Stub)"
        self.parent.categories = ["Segmentation"]


# =========================
# Interaction State
# =========================

class InteractionState:
    def __init__(self):
        self.segment = None
        self.mask = None
        self.mode = None

    def update(self, segment, mask, mode):
        self.segment = segment
        self.mask = mask
        self.mode = mode
        print(f"[State] seg={segment}, mask={mask}, mode={mode}")


# =========================
# Prompt Collector
# =========================

class PromptCollector:
    def __init__(self, paramNode):
        self.paramNode = paramNode

    def get_positive_points(self):
        node = self.paramNode.GetNodeReference("positivePromptPointsNode")
        n = node.GetNumberOfControlPoints()
        print(f"[Points] {n}")
        return node

    def get_rois(self):
        rois = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        print(f"[ROIs] {len(rois)}")
        return rois


# =========================
# Renderer (loop)
# =========================

class SegmentationRenderer:
    def __init__(self, widget):
        self.widget = widget

    def start(self):
        self.widget.currentlySegmenting = True
        self.update()

    def stop(self):
        self.widget.currentlySegmenting = False

    def update(self):
        if self.widget.currentlySegmenting:
            print("[Renderer] updating...")

            self.widget.prompts.get_positive_points()
            self.widget.prompts.get_rois()

        qt.QTimer.singleShot(100, self.update)


# =========================
# Widget
# =========================

class SegmentHumanBodyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        super().__init__(parent)
        VTKObservationMixin.__init__(self)

        self.logic = SegmentHumanBodyLogic()
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        self.currentlySegmenting = False

        # modular components
        self.state = InteractionState()
        self.prompts = None
        self.renderer = None

    # -------------------------
    # Setup
    # -------------------------
    def setup(self):
        super().setup()
        self.behavior = SegmentHumanBodyBehavior(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SegmentHumanBody.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.initializeParameterNode()

        # modules
        self.prompts = PromptCollector(self._parameterNode)
        self.renderer = SegmentationRenderer(self)

        self.connectSignals()

        print("[Setup] complete")

    # -------------------------
    # Connections
    # -------------------------
    def connectSignals(self):
        self.ui.segmentButton.connect("clicked(bool)", self.onStart)
        self.ui.stopSegmentButton.connect("clicked(bool)", self.onStop)

        self.ui.modelDropDown.connect("currentIndexChanged(int)", self.onUIChanged)
        self.ui.maskDropDown.connect("currentIndexChanged(int)", self.onUIChanged)
        self.ui.segmentationDropDown.connect("currentIndexChanged(int)", self.onUIChanged)

        self.ui.positivePrompts.connect("markupsNodeChanged()", self.onUIChanged)
        self.ui.negativePrompts.connect("markupsNodeChanged()", self.onUIChanged)

    # -------------------------
    # Parameter Node
    # -------------------------
    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())

        # Input volume
        if not self._parameterNode.GetNodeReference("InputVolume"):
            volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if volumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", volumeNode.GetID())

        # Prompt nodes
        if not self._parameterNode.GetNodeReferenceID("positivePromptPointsNode"):
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "positive")
            node.GetDisplayNode().SetSelectedColor(0, 1, 0)
            self._parameterNode.SetNodeReferenceID("positivePromptPointsNode", node.GetID())

        if not self._parameterNode.GetNodeReferenceID("negativePromptPointsNode"):
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "negative")
            node.GetDisplayNode().SetSelectedColor(1, 0, 0)
            self._parameterNode.SetNodeReferenceID("negativePromptPointsNode", node.GetID())

        self.ui.positivePrompts.setCurrentNode(
            self._parameterNode.GetNodeReference("positivePromptPointsNode")
        )
        self.ui.negativePrompts.setCurrentNode(
            self._parameterNode.GetNodeReference("negativePromptPointsNode")
        )

        # Segmentation
        if not self._parameterNode.GetNodeReferenceID("SegmentationNode"):
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segNode.CreateDefaultDisplayNodes()
            segId = segNode.GetSegmentation().AddEmptySegment("Segment_1")

            self._parameterNode.SetNodeReferenceID("SegmentationNode", segNode.GetID())
            self._parameterNode.SetParameter("CurrentSegment", segId)

        # Dropdowns
        self.ui.segmentationDropDown.clear()
        segNode = self._parameterNode.GetNodeReference("SegmentationNode")
        if segNode:
            seg = segNode.GetSegmentation()
            for i in range(seg.GetNumberOfSegments()):
                self.ui.segmentationDropDown.addItem(seg.GetNthSegment(i).GetName())

        self.ui.maskDropDown.clear()
        self.ui.maskDropDown.addItems(["Mask-1", "Mask-2"])

        self.ui.modelDropDown.clear()
        self.ui.modelDropDown.addItems([
            "SPX-Assisted Annotation",
            "Interactive",
            "Auto"
        ])
        if hasattr(self.ui, "ctSegmentationModelDropdown"):
            self.ui.ctSegmentationModelDropdown.clear()
            self.ui.ctSegmentationModelDropdown.addItems([
                "Custom",
                "2D",
                "3D",
                "Both"
            ])

    def setParameterNode(self, node):
        self._parameterNode = node

    # -------------------------
    # UI → State
    # -------------------------
    def onUIChanged(self, *args):
        segNode = self._parameterNode.GetNodeReference("SegmentationNode")

        segId = None
        if segNode:
            segId = segNode.GetSegmentation().GetSegmentIdBySegmentName(
                self.ui.segmentationDropDown.currentText
            )

        self.state.update(
            segId,
            self.ui.maskDropDown.currentText,
            self.ui.modelDropDown.currentText
        )

        print("[UI Changed]")

    # -------------------------
    # Buttons
    # -------------------------
    def onStart(self):
        print("[Start]")
        self.renderer.start()

    def onStop(self):
        print("[Stop]")
        self.renderer.stop()


# =========================
# Logic
# =========================

class SegmentHumanBodyLogic(ScriptedLoadableModuleLogic):
    def setDefaultParameters(self, parameterNode):
        pass