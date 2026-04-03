import qt, vtk, slicer
import logging
import numpy as np
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin

from core.modelFamilies import BaseModelFamily, InteractiveModelFamily, SPXModelFamily, AutoModelFamily
from core.utils import call_if_exists, get_slice_from_volume, write_slice_to_volume

log = logging.getLogger(__name__)

POS_NODE = 'positivePromptPointsNode'
NEG_NODE = 'negativePromptPointsNode'
INPUT_VOLUME = "InputVolume"
SEGMENTATION = "Segmentation"

#
# Module
#

class SegmentHumanBody(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = 'SegmentHumanBody (Optimized)'
        self.parent.categories = ['Segmentation']

#
# Renderer
#

class SegmentationRenderer:
    def __init__(self, widget):
        self.widget = widget
        self.timer = qt.QTimer()
        self.timer.timeout.connect(self.update)

    def start(self):
        log.debug('[Renderer] start')
        self.timer.start(100)

    def stop(self):
        log.debug('[Renderer] stop')
        self.timer.stop()

    def update(self):
        self.widget.logic.onRender(self.widget.modelFamily)


#
# Widget
#

class SegmentHumanBodyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        super().__init__(parent)
        VTKObservationMixin.__init__(self)

        self.logic = SegmentHumanBodyLogic()
        self._parameterNode = None

        self.modelFamily = None

        self._updatingGUI = False
        self.currentViewName = None  # default


    # -------------------------
    # Setup
    # -------------------------
    def setup(self):
        super().setup()

        self.currentViewName = "Red"
        

        uiWidget = slicer.util.loadUI(self.resourcePath('UI/SegmentHumanBody.ui'))
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.ui.sourceVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.segmentSelector.setMRMLScene(slicer.mrmlScene)

        self.model_classes = {
            'None': BaseModelFamily,
            'Interactive': InteractiveModelFamily,
            'SPX-Assisted Annotation': SPXModelFamily,
            'Auto': AutoModelFamily,
        }

        # self.renderer = SegmentationRenderer(self)
        self.renderer = None

        self.initializeUI()
        self.connectSignals()

        # Lock selectors
        self.ui.positivePrompts.setNodeSelectorVisible(False)
        self.ui.negativePrompts.setNodeSelectorVisible(False)

        qt.QTimer.singleShot(0, self._initializeAfterSetup)

        log.debug('[Setup complete]')

    def _initializeAfterSetup(self):
        if not slicer.mrmlScene:
            return

        nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLMarkupsFiducialNode')

        log.debug('[Existing markups nodes]:')
        for i in range(nodes.GetNumberOfItems()):
            node = nodes.GetItemAsObject(i)
            log.debug(f' - {node.GetName()}')

        self.initializeParameterNode()
        self.setParameterNode(self._parameterNode)
        self.onModelFamilyChanged()

    # -------------------------
    # Signals
    # -------------------------
    def connectSignals(self):
        ui = self.ui

        model_button_connections = [
            ('enterInteractiveModeButton', 'on_enter_interactive'),
            ('stopInteractiveModeButton', 'on_stop_interactive'),
            ('assignLabel2D', 'on_assign_2d'),
            ('assignLabel3D', 'on_assign_3d'),
            ('propagateSelectedLabelButton', 'on_propagate'),
            ('runAutomaticSegmentation', 'on_automatic_segmentation'),
        ]

        for ui_name, method_name in model_button_connections:
            getattr(ui, ui_name).connect(
                'clicked(bool)',
                self.bind(method_name, target="logic")
            )

        widget_button_connections = [
            ('goToSegmentEditorButton', self.on_go_to_editor),
            ('goToMarkupsButton', self.on_go_to_markups),
            ('confirmModelSelection', self.onConfirmClicked),
        ]

        for ui_name, method in widget_button_connections:
            widget_btn = getattr(ui, ui_name)
            widget_btn.connect('clicked(bool)', lambda *args, m=method: m())

        ui.modelFamilyDropdown.connect('currentIndexChanged(int)', self.onModelFamilyChanged)
        ui.modelVariantDropdown.connect('currentIndexChanged(int)', self.onVariantChanged)
        ui.sliceViewDropdown.connect('currentTextChanged(QString)', self.onSliceViewChanged)
        ui.sourceVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)


    # -------------------------
    # Observers
    # -------------------------
    def _observeMarkupsNodes(self):
        self.removeObservers()

        posNode, negNode = self.logic.getPromptNodes(self._parameterNode)

        for node in [posNode, negNode]:
            if node:
                self.addObserver(
                    node,
                    vtk.vtkCommand.ModifiedEvent,
                    self._onMarkupsModified
                )

    def _onMarkupsModified(self, caller=None, event=None):
        if not self._parameterNode:
            return

        self.logic.updateParameterNodeFromMarkups(
            self._parameterNode,
            self.ui.positivePrompts.currentNode(),
            self.ui.negativePrompts.currentNode(),
        )
    
    def onSliceViewChanged(self, viewName):
        self.currentViewName = viewName
        print(f"[UI] Selected view: {viewName}")
    

    # -------------------------
    # UI
    # -------------------------
    def updateUIVisibility(self):
        mapping = [
            ('assignLabel2D', 'on_assign_2d'),
            ('assignLabel3D', 'on_assign_3d'),
            ('enterInteractiveModeButton', 'on_enter_interactive'),
            ('stopInteractiveModeButton', 'on_stop_interactive'),
            ('propagateSelectedLabelButton', 'on_propagate'),
            ('runAutomaticSegmentation', 'on_automatic_segmentation'),
            ('goToMarkupsButton', 'on_go_to_markups'),
            ('samMaskDropdown','get_requested_mask')
        ]

        ui = self.ui

        for ui_name, method in mapping:
            widget = getattr(ui, ui_name)
            widget.setVisible(hasattr(self.modelFamily, method))

    def initializeUI(self):
        dropdowns = [
            'modelFamilyDropdown',
            'samMaskDropdown',
            'modelVariantDropdown'
        ]

        for name in dropdowns:
            if hasattr(self.ui, name):
                getattr(self.ui, name).blockSignals(True)

        self.ui.modelFamilyDropdown.clear()
        self.ui.modelFamilyDropdown.addItems(list(self.model_classes.keys()))

        self.ui.samMaskDropdown.clear()
        self.ui.samMaskDropdown.addItems(['Mask-1', 'Mask-2', 'Mask-3'])
        self.ui.sliceViewDropdown.clear()
        self.ui.sliceViewDropdown.addItems(["Red", "Green", "Yellow"])
        self.ui.sliceViewDropdown.setCurrentText("Red")

        for name in dropdowns:
            if hasattr(self.ui, name):
                getattr(self.ui, name).blockSignals(False)

    # -------------------------
    # Parameter Node
    # -------------------------
    def initializeParameterNode(self):
        self._parameterNode = self.logic.getParameterNode()

        if not self._parameterNode:
            self._parameterNode = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLScriptedModuleNode'
            )

        self.logic.setDefaultParameters(self._parameterNode)
        self.logic.ensurePromptNodesExist(self._parameterNode)

    def setParameterNode(self, inputParameterNode):
        if self._parameterNode:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode
            )

        self._parameterNode = inputParameterNode

        if self._parameterNode:
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode
            )
            self._observeMarkupsNodes()

        qt.QTimer.singleShot(0, self.updateGUIFromParameterNode)

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if not self._parameterNode or self._updatingGUI:
            return

        self._updatingGUI = True
        try:
            posNode, negNode = self.logic.getPromptNodes(self._parameterNode)
            volumeNode, segNode = self.logic.getVolumeAndSegmentation(self._parameterNode)

            self.ui.positivePrompts.setCurrentNode(posNode)
            self.ui.negativePrompts.setCurrentNode(negNode)

            self.ui.sourceVolumeSelector.setCurrentNode(volumeNode)

        finally:
            self._updatingGUI = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        if not self._parameterNode:
            return

        self.logic.updateParameterNodeFromMarkups(
            self._parameterNode,
            self.ui.positivePrompts.currentNode(),
            self.ui.negativePrompts.currentNode(),
        )

        self.logic.setVolumeAndSegmentation(
            self._parameterNode,
            self.ui.sourceVolumeSelector.currentNode(),
            self.ui.segmentSelector.currentNode(),
        )
    # -------------------------
    # Model Switching
    # -------------------------
    def onModelFamilyChanged(self, *args):
        self.setConfirmState(False)

        modelFamilyName = self.ui.modelFamilyDropdown.currentText
        ModelClass = self.model_classes.get(modelFamilyName, BaseModelFamily)

        self.modelFamily = ModelClass()

        self.updateModelVariants()

        if hasattr(self.modelFamily, "VARIANTS") and self.modelFamily.VARIANTS:
            self.modelFamily.variant = self.modelFamily.VARIANTS[0]

        self.updateUIVisibility()

    def updateModelVariants(self):
        dropdown = self.ui.modelVariantDropdown

        dropdown.blockSignals(True)
        dropdown.clear()

        if self.modelFamily and hasattr(self.modelFamily, "VARIANTS"):
            variants = self.modelFamily.VARIANTS
        else:
            variants = ["None"]

        dropdown.addItems(variants)

        if variants:
            dropdown.setCurrentIndex(0)

        dropdown.blockSignals(False)

    def onVariantChanged(self, *args):
        self.setConfirmState(False)

        if not self.modelFamily:
            return

        variant = self.ui.modelVariantDropdown.currentText
        self.modelFamily.variant = variant

    def onConfirmClicked(self, *args):
        if not self.modelFamily:
            return

        self.logic.on_confirm_model(self)
        self.setConfirmState(True)


    def setConfirmState(self, confirmed: bool):
        button = self.ui.confirmModelSelection

        if confirmed:
            button.setEnabled(False)
            button.setText("Model Confirmed")
        else:
            button.setEnabled(True)
            button.setText("Confirm Model Selection")

    def on_go_to_editor(self, *args):
        slicer.util.selectModule('SegmentEditor')

    def on_go_to_markups(self, *args):
        slicer.util.selectModule('Markups')
    
    def bind(self, method_name, target="logic"):
        if target == "logic":
            return lambda _=None: getattr(self.logic, method_name)(self)

        elif target == "model":
            return lambda _=None: call_if_exists(self.modelFamily, method_name)

        else:  # widget
            return getattr(self, method_name)


#
# Logic
#

class SegmentHumanBodyLogic(ScriptedLoadableModuleLogic):

    def setDefaultParameters(self, parameterNode):
        pass

    # -------------------------
    # Prompt Nodes
    # -------------------------
    def setVolumeAndSegmentation(self, parameterNode, volumeNode, segmentationNode):
        if volumeNode:
            parameterNode.SetNodeReferenceID(INPUT_VOLUME, volumeNode.GetID())
        if segmentationNode:
            parameterNode.SetNodeReferenceID(SEGMENTATION, segmentationNode.GetID())

    def getVolumeAndSegmentation(self, parameterNode):
        return (
            parameterNode.GetNodeReference(INPUT_VOLUME),
            parameterNode.GetNodeReference(SEGMENTATION),
        )
    
    def ensurePromptNodesExist(self, parameterNode):
        configs = {
            POS_NODE: ([0, 1, 0], 'positive'),
            NEG_NODE: ([1, 0, 0], 'negative'),
        }

        for ref_name, (color, label) in configs.items():
            if not parameterNode.GetNodeReference(ref_name):
                node = slicer.mrmlScene.AddNewNodeByClass(
                    'vtkMRMLMarkupsFiducialNode', label
                )

                node.CreateDefaultDisplayNodes()
                displayNode = node.GetDisplayNode()

                displayNode.SetSelectedColor(*color)
                displayNode.SetColor(*color)
                displayNode.SetActiveColor(*color)

                node.SetHideFromEditors(True)

                parameterNode.SetNodeReferenceID(
                    ref_name, node.GetID()
                )

    def setPromptNodes(self, parameterNode, posNode, negNode):
        parameterNode.SetNodeReferenceID(
            POS_NODE, posNode.GetID() if posNode else None
        )
        parameterNode.SetNodeReferenceID(
            NEG_NODE, negNode.GetID() if negNode else None
        )

    def getPromptNodes(self, parameterNode):
        return (
            parameterNode.GetNodeReference(POS_NODE),
            parameterNode.GetNodeReference(NEG_NODE),
        )

    def updateParameterNodeFromMarkups(self, parameterNode, posNode, negNode):
        self.setPromptNodes(parameterNode, posNode, negNode)

    # -------------------------
    # Model Interaction
    # -------------------------
    def onRender(self, modelFamily):
        call_if_exists(modelFamily, 'on_render')

    def onModelConfirmed(self, modelFamily):
        call_if_exists(modelFamily, 'on_confirm_model_selection')

    def on_confirm_model(self, widget):
        if not widget.modelFamily:
            return

        widget.modelFamily.confirm_model()
        
    def on_propagate(self, widget):
        print("[SPX] Propagate clicked")

        modelFamily = widget.modelFamily

        if not modelFamily:
            slicer.util.warningDisplay("Please select a model first.")
            return

        if not getattr(modelFamily, "model", None):
            slicer.util.warningDisplay("Please click 'Confirm Model Selection' before running.")
            return

        volumeNode = widget.ui.sourceVolumeSelector.currentNode()
        segNode = widget.ui.segmentSelector.currentNode()
        segmentID = widget.ui.segmentSelector.currentSegmentID()

        if not volumeNode:
            slicer.util.warningDisplay("Please select a source volume.")
            return

        if not segNode or not segmentID:
            slicer.util.warningDisplay("Please select a segmentation and segment.")
            return

        if not segmentID:
            slicer.util.warningDisplay("No segment selected.")
            return

        volumeArray = slicer.util.arrayFromVolume(volumeNode)

        axis, sliceIndex = self.getAxisAndSlice(widget)
        img = get_slice_from_volume(volumeArray, axis, sliceIndex)

        print("[SPX] Running model...")

        labels = modelFamily.model.forward(img=img)

        self.expandSegWithSPX(segNode, segmentID, volumeNode, labels, axis, sliceIndex)
    
    
    def on_enter_interactive(self, widget):
        print("[Logic] start interactive")

        if not widget.renderer:
            widget.renderer = SegmentationRenderer(widget)

        widget.renderer.start()


    def on_stop_interactive(self, widget):
        print("[Logic] stop interactive")

        if widget.renderer:
            widget.renderer.stop()

    def on_assign_2d(self, widget):
        call_if_exists(widget.modelFamily, 'on_assign_2d')

    def on_assign_3d(self, widget):
        call_if_exists(widget.modelFamily, 'on_assign_3d')
    
    def on_automatic_segmentation(self, widget):
        call_if_exists(widget.modelFamily, 'run')


    def _ras_to_ijk(self, volumeNode, scrib):
        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)

        def convert(points):
            ijk_pts = []
            for p in points:
                ras = list(p) + [1]
                ijk = rasToIjk.MultiplyPoint(ras)
                ijk_pts.append([int(ijk[0]), int(ijk[1])])
            return ijk_pts

        return {
            "positive": convert(scrib["positive"]),
            "negative": convert(scrib["negative"])
        }

    def updateSegmentationFromArray(self, mask, volumeNode, sliceIndex):
        import numpy as np

        print("[SPX] Rendering segmentation...")

        # --- create segmentation node ---
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segNode.CreateDefaultDisplayNodes()
        segNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)

        # --- create segment ---
        segmentID = segNode.GetSegmentation().AddEmptySegment("SPX")

        # --- make full 3D mask ---
        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        fullMask = np.zeros_like(volumeArray, dtype=np.uint8)

        axis = self.getSliceAccessorDimension(volumeNode)

        write_slice_to_volume(fullMask, mask, axis, sliceIndex)

        # --- push to slicer ---
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            fullMask, segNode, segmentID, volumeNode
        )



    def getAxisAndSlice(self, widget):
        viewName = widget.currentViewName

        lm = slicer.app.layoutManager()
        sliceWidget = lm.sliceWidget(viewName)
        logic = sliceWidget.sliceLogic()

        offset = logic.GetSliceOffset()
        sliceIndex = logic.GetSliceIndexFromOffset(offset) - 1

        if viewName == "Red":
            axis = 0   
        elif viewName == "Green":
            axis = 1   
        else:
            axis = 2   

        print(f"[SPX] View={viewName}, axis={axis}, slice={sliceIndex}")

        return axis, sliceIndex
    
    def getCurrentSegment(self, widget):
        segNode = widget.ui.segmentSelector.currentNode()
        segmentID = widget.ui.segmentSelector.currentSegmentID()

        if not segmentID and segNode:
            segmentation = segNode.GetSegmentation()
            if segmentation.GetNumberOfSegments() > 0:
                segmentID = segmentation.GetNthSegmentID(0)

        return segNode, segmentID


    def expandSegWithSPX(self, segNode, segmentID, volumeNode, labels, axis, sliceIndex):

        mask3d = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentID, volumeNode)

        sliceMask = get_slice_from_volume(mask3d, axis, sliceIndex)

        selected_labels = np.unique(labels[sliceMask > 0])
        expanded = np.isin(labels, selected_labels).astype(np.uint8)

        fullMask = np.zeros_like(mask3d, dtype=np.uint8)
        write_slice_to_volume(fullMask, expanded, axis, sliceIndex)

        slicer.util.updateSegmentBinaryLabelmapFromArray(
            fullMask, segNode, segmentID, volumeNode
        )
    
