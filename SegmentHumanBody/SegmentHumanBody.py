import qt, vtk, slicer
import logging

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin

from core.modelFamilies import BaseModelFamily, InteractiveModelFamily, SPXModelFamily, AutoModelFamily
from core.utils import call_if_exists, make_model_callback, make_widget_callback

log = logging.getLogger(__name__)

POS_NODE = 'positivePromptPointsNode'
NEG_NODE = 'negativePromptPointsNode'


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

    # -------------------------
    # Setup
    # -------------------------
    def setup(self):
        super().setup()

        uiWidget = slicer.util.loadUI(self.resourcePath('UI/SegmentHumanBody.ui'))
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        self.model_classes = {
            'None': BaseModelFamily,
            'Interactive': InteractiveModelFamily,
            'SPX-Assisted Annotation': SPXModelFamily,
            'Auto': AutoModelFamily,
        }

        self.renderer = SegmentationRenderer(self)

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
            widget_btn = getattr(ui, ui_name)

            widget_btn.connect(
                'clicked(bool)',
                make_model_callback(self, method_name)
            )

        widget_button_connections = [
            ('goToSegmentEditorButton', self.on_go_to_editor),
            ('goToMarkupsButton', self.on_go_to_markups),
            ('confirmModelSelection', self.onConfirmClicked),
        ]

        for ui_name, method in widget_button_connections:
            widget_btn = getattr(ui, ui_name)

            widget_btn.connect(
                'clicked(bool)',
                make_widget_callback(self, method)
            )

        ui.modelFamilyDropdown.connect('currentIndexChanged(int)', self.onModelFamilyChanged)
        ui.modelVariantDropdown.connect('currentIndexChanged(int)', self.onVariantChanged)

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

            self.ui.positivePrompts.setCurrentNode(posNode)
            self.ui.negativePrompts.setCurrentNode(negNode)
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
    # -------------------------
    # Model Switching
    # -------------------------
    def onModelFamilyChanged(self, *args):
        self.setConfirmState(False)
        modelFamilyName = self.ui.modelFamilyDropdown.currentText

        ModelClass = self.model_classes.get(modelFamilyName, BaseModelFamily)
        self.modelFamily = ModelClass(self)

        self.updateModelVariants()
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

        if not variant:
            return

        self.modelFamily.variant = variant

    def onConfirmClicked(self, *args):
        if not self.modelFamily:
            return

        self.logic.onModelConfirmed(self.modelFamily)
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


#
# Logic
#

class SegmentHumanBodyLogic(ScriptedLoadableModuleLogic):

    def setDefaultParameters(self, parameterNode):
        pass

    # -------------------------
    # Prompt Nodes
    # -------------------------
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