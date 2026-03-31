import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin

from shb_models import BaseModel, InteractiveModel, SPXModel, AutoModel
from utils import call_if_exists


#
# Module
#

class SegmentHumanBody(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "SegmentHumanBody (Final Template)"
        self.parent.categories = ["Segmentation"]


#
# Renderer (kept here)
#

class SegmentationRenderer:
    def __init__(self, widget):
        self.widget = widget
        self.running = False

    def start(self):
        print("[Renderer] start")
        self.running = True
        self.update()

    def stop(self):
        print("[Renderer] stop")
        self.running = False

    def update(self):
        if self.running:
            print("[Renderer] updating...")

            # optional: model-specific render hook
            call_if_exists(self.widget.model, "on_render")

        qt.QTimer.singleShot(100, self.update)


#
# Widget
#

class SegmentHumanBodyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        super().__init__(parent)
        VTKObservationMixin.__init__(self)

        self.logic = SegmentHumanBodyLogic()
        self._parameterNode = None

        # model system
        self.model = None
        self.model_cache = {}

    # -------------------------
    # Setup
    # -------------------------
    def setup(self):
        super().setup()

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SegmentHumanBody.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # --- model registry ---
        self.model_classes = {
            "Interactive": InteractiveModel,
            "SPX-Assisted Annotation": SPXModel,
            "Auto": AutoModel,
        }

        # --- renderer ---
        self.renderer = SegmentationRenderer(self)

        self.initializeUI()
        self.connectSignals()
        print("[Setup complete]")

    # -------------------------
    # Signals
    # -------------------------
    def connectSignals(self):

        connections = [
            ("segmentButton", "on_start"),
            ("stopSegmentButton", "on_stop"),
            ("assignLabel2D", "on_assign_2d"),
            ("assignLabel3D", "on_assign_3d"),
            ("goToSegmentEditorButton", "on_go_to_editor"),
            ("goToMarkupsButton", "on_go_to_markups"),
        ]

        for ui_name, method in connections:
            widget = getattr(self.ui, ui_name)

            widget.connect(
                "clicked(bool)",
                lambda _, m=method: call_if_exists(self.model, m)
            )

        # dropdown handled separately
        self.ui.modelDropDown.connect(
            "currentIndexChanged(int)", self.onModeChanged
        )

    def updateUIVisibility(self):
        mapping = [
            ("assignLabel2D", "on_assign_2d"),
            ("assignLabel3D", "on_assign_3d"),
            ("segmentButton", "on_start"),
            ("stopSegmentButton", "on_stop"),
        ]

        for ui_name, method in mapping:
            widget = getattr(self.ui, ui_name)
            widget.setVisible(hasattr(self.model, method))
    
    def initializeUI(self):
        """Populate all dropdowns"""

        # -------------------------
        # Block signals (prevent early triggers)
        # -------------------------
        dropdowns = [
            "modelDropDown",
            "maskDropDown",
            "segmentationDropDown",
            "ctSegmentationModelDropdown",
        ]

        for name in dropdowns:
            if hasattr(self.ui, name):
                getattr(self.ui, name).blockSignals(True)

        # -------------------------
        # Model dropdown (auto from registry)
        # -------------------------
        if hasattr(self.ui, "modelDropDown"):
            self.ui.modelDropDown.clear()
            self.ui.modelDropDown.addItems(list(self.model_classes.keys()))
            self.ui.modelDropDown.setCurrentIndex(0)

        # -------------------------
        # Mask dropdown
        # -------------------------
        if hasattr(self.ui, "maskDropDown"):
            self.ui.maskDropDown.clear()
            self.ui.maskDropDown.addItems(["Mask-1", "Mask-2"])
            self.ui.maskDropDown.setCurrentIndex(0)

        # -------------------------
        # Segmentation dropdown
        # -------------------------
        if hasattr(self.ui, "segmentationDropDown"):
            self.ui.segmentationDropDown.clear()

            segNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")

            if segNode:
                seg = segNode.GetSegmentation()
                for i in range(seg.GetNumberOfSegments()):
                    self.ui.segmentationDropDown.addItem(seg.GetNthSegment(i).GetName())

            # fallback if none exists
            if self.ui.segmentationDropDown.count == 0:
                segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                segNode.CreateDefaultDisplayNodes()
                segNode.GetSegmentation().AddEmptySegment("Segment_1")

                self.ui.segmentationDropDown.addItem("Segment_1")

            self.ui.segmentationDropDown.setCurrentIndex(0)

        # -------------------------
        # Optional CT dropdown
        # -------------------------
        if hasattr(self.ui, "ctSegmentationModelDropdown"):
            self.ui.ctSegmentationModelDropdown.clear()
            self.ui.ctSegmentationModelDropdown.addItems([
                "Custom", "2D", "3D", "Both"
            ])
            self.ui.ctSegmentationModelDropdown.setCurrentIndex(0)

        # -------------------------
        # Re-enable signals
        # -------------------------
        for name in dropdowns:
            if hasattr(self.ui, name):
                getattr(self.ui, name).blockSignals(False)

        print("[UI Initialized]")

    # -------------------------
    # Mode switching (lazy init)
    # -------------------------
    def onModeChanged(self, *args):
        mode = self.ui.modelDropDown.currentText
        print(f"[Mode Changed] {mode}")

        if mode not in self.model_cache:
            ModelClass = self.model_classes.get(mode, BaseModel)
            self.model_cache[mode] = ModelClass(self)

        self.model = self.model_cache[mode]

        print("[Model Ready]")


#
# Logic
#

class SegmentHumanBodyLogic(ScriptedLoadableModuleLogic):
    def setDefaultParameters(self, parameterNode):
        pass