import qt, vtk, slicer
import logging, ast
import numpy as np
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin

from core.modelFamilies import BaseModelFamily, SAMFamily, SPXModelFamily, AutoModelFamily
from core.utils import (
    call_if_exists,
    get_slice_from_volume,
    write_slice_to_volume,
    apply_window_level,
    spx_boundary_mask,
    labels_at_points,
    collect_confirmed_points,
    collect_preview_points,
    POSITION_UNDEFINED,
    POSITION_PREVIEW,
    POSITION_DEFINED,
)
from core.modelRegistry import ModelRegistry
from core.undoStack import UndoStack


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
        if self.widget._pauseRender or self.widget._isRendering:
            #print("[Renderer] Skipping frame")
            return

        #print("[Renderer] START update")

        self.widget._isRendering = True
        try:
            #print("[Renderer] Calling logic.onRender")
            self.widget.logic.onRender(self.widget.modelFamily, self.widget)
            #print("[Renderer] logic.onRender finished OK")

        except Exception as e:
            #print("[Renderer] EXCEPTION CAUGHT")
            #print(e)

            log.error(f"[Renderer Error] {e}")

            self.stop()
            #print("[Renderer] STOP called")

            slicer.util.errorDisplay(f"Rendering stopped:\n{str(e)}")
            self.widget.setInteractiveState(False)

        finally:
            self.widget._isRendering = False
            #print("[Renderer] END update")
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
        self._isInteractive = False   # True while the render loop is running
        self.currentViewName = None  # default
        self._isRendering = False
        self._pauseRender = False

        # Undo tracking
        self._interactive_point_stack = []  # (node, controlPointID) — interactive mode
        self._undo_shortcut = None

        # IDs of control points that are still in PositionPreview state (the
        # hover-cursor Slicer creates when the user clicks "+" in the widget but
        # has not yet clicked on a slice to confirm placement).  These points
        # must not drive a real render until they are confirmed.  They ARE used
        # for the hover preview when the Preview checkbox is on.
        self._preview_cp_ids: set = set()

        # SPX boundary overlay
        self._spx_boundary_node    = None   # vtkMRMLLabelMapVolumeNode
        self._spx_boundary_visible = False
        self._spx_boundary_view    = None   # view name the label is currently set on
        self._spx_boundary_shortcut = None


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
        self.ui.segmentationNodeSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.segmentSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.segmentSelector.segmentationNodeSelectorVisible = False
        self.ui.docLinkLabel.setOpenExternalLinks(True)
        self.setInteractiveState(False)

        self.model_classes = {
            'None': BaseModelFamily,
            'SAM-Style': SAMFamily,
            'SPX-Assisted Annotation': SPXModelFamily,
            'Auto': AutoModelFamily,
        }

        self.renderer = None

        self.initializeUI()
        self.connectSignals()

        # Lock selectors
        self.ui.positivePrompts.setNodeSelectorVisible(False)
        self.ui.negativePrompts.setNodeSelectorVisible(False)

        # Ctrl+Z undo shortcut — parented to the module widget so it is
        # active only while this panel is visible.
        self._undo_shortcut = qt.QShortcut(qt.QKeySequence("Ctrl+Z"), uiWidget)
        self._undo_shortcut.connect('activated()', self.onUndo)

        # Ctrl+Q — toggle SPX superpixel boundary overlay
        self._spx_boundary_shortcut = qt.QShortcut(qt.QKeySequence("Ctrl+Q"), uiWidget)
        self._spx_boundary_shortcut.connect('activated()', self.onToggleSPXBoundary)

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

        qt.QTimer.singleShot(0, self.updateGUIFromParameterNode)

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
            ('addSegmentButton', self.onAddSegment),
            ('removeSegmentButton', self.onRemoveSegment),
            ('applyWindowLevelButton', self.onApplyWindowLevel),
        ]

        for ui_name, method in widget_button_connections:
            getattr(ui, ui_name).connect('clicked(bool)', method)

        ui.modelFamilyDropdown.connect('currentIndexChanged(int)', self.onModelFamilyChanged)
        ui.modelVariantDropdown.connect('currentIndexChanged(int)', self.onVariantChanged)
        ui.sliceViewDropdown.connect('currentTextChanged(QString)', self.onSliceViewChanged)
        ui.sourceVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        ui.segmentationNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        ui.segmentSelector.connect("currentSegmentChanged(QString)", self.onSegmentChanged)

        # Window/Level slider ↔ spinbox sync + live display preview
        ui.windowSlider.connect('valueChanged(int)', self._onWindowSliderChanged)
        ui.windowSpinBox.connect('valueChanged(int)', self._onWindowSpinBoxChanged)
        ui.levelSlider.connect('valueChanged(int)', self._onLevelSliderChanged)
        ui.levelSpinBox.connect('valueChanged(int)', self._onLevelSpinBoxChanged)

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
                self.addObserver(
                    node,
                    slicer.vtkMRMLMarkupsNode.PointAddedEvent,
                    self._onPointAdded
                )
                self.addObserver(
                    node,
                    slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                    self._onPointConfirmed
                )

    def _onMarkupsModified(self, caller=None, event=None):
        if not self._parameterNode:
            return

        self.logic.updateParameterNodeFromMarkups(
            self._parameterNode,
            self.ui.positivePrompts.currentNode(),
            self.ui.negativePrompts.currentNode(),
        )

    def _onPointAdded(self, caller=None, event=None):
        """Record each newly added prompt point so Ctrl+Z can remove it.

        If the new point is in PositionPreview state (Slicer's placement cursor
        before the user clicks on a slice), mark it as unconfirmed so the render
        loop ignores it until PointPositionDefinedEvent fires.
        """
        if caller is None:
            return
        n = caller.GetNumberOfControlPoints()
        if n > 0:
            cp_id = caller.GetNthControlPointID(n - 1)
            status = caller.GetNthControlPointPositionStatus(n - 1)
            if status == slicer.vtkMRMLMarkupsNode.PositionPreview:
                self._preview_cp_ids.add(cp_id)
            self._interactive_point_stack.append((caller, cp_id))

    def _onPointConfirmed(self, caller=None, event=None):
        """PointPositionDefinedEvent — a placement was just confirmed by the user.

        Evict ALL point IDs from this node from _preview_cp_ids.  We do NOT
        filter by PositionDefined status because in some Slicer builds confirmed
        points retain PositionPreview status indefinitely — the event itself is
        the authoritative confirmation signal, not the resulting status value.

        Any new placement cursor created afterward (multi-place mode) will be
        re-added to _preview_cp_ids when its own PointAddedEvent fires, which
        always follows PointPositionDefinedEvent in Slicer's event ordering.
        """
        if caller is None:
            return
        for i in range(caller.GetNumberOfControlPoints()):
            self._preview_cp_ids.discard(caller.GetNthControlPointID(i))

    def onSliceViewChanged(self, viewName):
        self.currentViewName = viewName

    # -------------------------
    # Window / Level
    # -------------------------

    def _syncWLSlidersFromVolume(self, volumeNode):
        """Populate W/L sliders from the volume's current display node.
        Called when a new volume is selected so the sliders reflect the
        volume's existing display settings rather than the UI defaults.
        Also resets the Apply button to its pending (unlocked) state.
        """
        if not volumeNode:
            return
        displayNode = volumeNode.GetScalarVolumeDisplayNode()
        if not displayNode:
            return
        # GetWindow/GetLevel returns the effective value whether auto or manual.
        w = max(1,     min(4000, int(displayNode.GetWindow())))
        l = max(-1000, min(3000, int(displayNode.GetLevel())))
        for widget, value in [
            (self.ui.windowSlider,  w),
            (self.ui.windowSpinBox, w),
            (self.ui.levelSlider,   l),
            (self.ui.levelSpinBox,  l),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        # New volume → any previously confirmed W/L is stale.
        self._resetWLButton()
        self.logic.set_window_level(None, None)

    def _resetWLButton(self):
        """Restore the Apply button to its unlocked (pending) state."""
        btn = self.ui.applyWindowLevelButton
        btn.setText("Apply Window / Level")
        btn.setEnabled(True)

    def _updateDisplayNodeWL(self):
        """Push current slider values to the volume display node (live preview)."""
        volumeNode = self.ui.sourceVolumeSelector.currentNode()
        if not volumeNode:
            return
        displayNode = volumeNode.GetScalarVolumeDisplayNode()
        if not displayNode:
            return
        # Disable auto W/L — without this Slicer silently overrides any
        # manually set values with its own computed range.
        displayNode.AutoWindowLevelOff()
        displayNode.SetWindow(self.ui.windowSlider.value)
        displayNode.SetLevel(self.ui.levelSlider.value)

    def _onWLControlChanged(self, _=None):
        """Called whenever any W/L slider or spinbox changes.
        Updates the live display preview and, if W/L was previously confirmed,
        un-confirms it — the user must click Apply again to lock in the new values.
        """
        self._updateDisplayNodeWL()
        btn = self.ui.applyWindowLevelButton
        if not btn.isEnabled():
            self.logic.set_window_level(None, None)
            self._resetWLButton()

    def _onWindowSliderChanged(self, value):
        self.ui.windowSpinBox.blockSignals(True)
        self.ui.windowSpinBox.setValue(value)
        self.ui.windowSpinBox.blockSignals(False)
        self._onWLControlChanged()

    def _onWindowSpinBoxChanged(self, value):
        self.ui.windowSlider.blockSignals(True)
        self.ui.windowSlider.setValue(value)
        self.ui.windowSlider.blockSignals(False)
        self._onWLControlChanged()

    def _onLevelSliderChanged(self, value):
        self.ui.levelSpinBox.blockSignals(True)
        self.ui.levelSpinBox.setValue(value)
        self.ui.levelSpinBox.blockSignals(False)
        self._onWLControlChanged()

    def _onLevelSpinBoxChanged(self, value):
        self.ui.levelSlider.blockSignals(True)
        self.ui.levelSlider.setValue(value)
        self.ui.levelSlider.blockSignals(False)
        self._onWLControlChanged()

    def onApplyWindowLevel(self, _=None):
        """Confirm the current W/L values for model inference and lock the button.
        The volume's scalar data is never modified — only the display node and
        the per-slice normalization applied before data reaches models.
        Moving any slider after this will unlock the button automatically.
        """
        self.logic.set_window_level(
            self.ui.windowSpinBox.value,
            self.ui.levelSpinBox.value,
        )
        btn = self.ui.applyWindowLevelButton
        btn.setText("W/L Applied")
        btn.setEnabled(False)

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

        if not self._parameterNode:
            return

        self._updatingGUI = True
        try:
            posNode, negNode = self.logic.getPromptNodes(self._parameterNode)
            volumeNode, segNode = self.logic.getVolumeAndSegmentation(self._parameterNode)

            # Never re-set the markups widgets while the render loop is active.
            # The prompt nodes are managed exclusively by clearPrompts during a
            # session; calling setCurrentNode here would reset the active
            # placement cursor (e.g. "Positive 1" → "Negative 1") because the
            # negative widget enters placement mode on the empty fresh node.
            if not self._isInteractive:
                self.ui.positivePrompts.setCurrentNode(posNode)
                self.ui.negativePrompts.setCurrentNode(negNode)

            self.ui.sourceVolumeSelector.setCurrentNode(volumeNode)
            self.ui.segmentationNodeSelector.setCurrentNode(segNode)
            self.ui.segmentSelector.setCurrentNode(segNode)
            self.ui.addSegmentButton.setEnabled(segNode is not None)

        finally:
            self._updatingGUI = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):

        if not self._parameterNode:
            return

        volumeNode = self.ui.sourceVolumeSelector.currentNode()
        self._syncWLSlidersFromVolume(volumeNode)
        segNode = self.ui.segmentationNodeSelector.currentNode()

        # Auto-create a segmentation node when a volume is selected but no
        # segmentation exists yet.  This is intentional: selecting a volume
        # is the natural trigger for creating the paired segmentation container.
        if volumeNode and not segNode:
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segNode.CreateDefaultDisplayNodes()
            segNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)

        self.logic.updateParameterNodeFromMarkups(
            self._parameterNode,
            self.ui.positivePrompts.currentNode(),
            self.ui.negativePrompts.currentNode(),
        )

        self.logic.setVolumeAndSegmentation(self._parameterNode, volumeNode, segNode)

        self._parameterNode.Modified()
        self.updateGUIFromParameterNode()
    
    def getUserParameters(self):
        text = self.ui.paramTextEdit.toPlainText()

        if not text.strip():
            return {}

        try:
            items = []
            for part in text.split(","):
                if not part.strip():
                    continue
                key, value = part.split("=", 1)
                items.append(f"'{key.strip()}': {value.strip()}")

            dict_str = "{" + ", ".join(items) + "}"

            return ast.literal_eval(dict_str)

        except Exception as e:
            raise ValueError(f"Invalid parameter format:\n{str(e)}")
    
    def updateParamPlaceholder(self):
        if not self.modelFamily or not self.modelFamily.variant:
            self.ui.paramTextEdit.setPlaceholderText("Select a model variant.")
            return

        try:
            if hasattr(self.modelFamily, "_get_model_key"):
                key = self.modelFamily._get_model_key()
            else:
                key = self.modelFamily.variant

            placeholder = ModelRegistry.get_param_hint(key)

        except Exception:
            placeholder = "Model not available. Please select a different model."

        self.ui.paramTextEdit.setPlaceholderText(placeholder)
    
    def updateDocLink(self):
        model = getattr(self.modelFamily, "model", None)

        if not model:
            self.ui.docLinkLabel.setText("")
            return

        url = getattr(model, "DOC_URL", None)

        if url:
            self.ui.docLinkLabel.setText(
                f'<a href="{url}">View documentation</a>'
            )
        else:
            self.ui.docLinkLabel.setText("")

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
        self.updateParamPlaceholder()

        # Hide SPX boundary overlay when leaving the SPX family.
        if not isinstance(self.modelFamily, SPXModelFamily):
            self._hideSPXBoundary()

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
        self.updateParamPlaceholder()

    def onConfirmClicked(self, *args):
        if not self.modelFamily:
            return

        self.logic.on_confirm_model(self)
        self.setConfirmState(True)
        self.updateDocLink()


    def setConfirmState(self, confirmed: bool):
        button = self.ui.confirmModelSelection

        if confirmed:
            button.setEnabled(False)
            button.setText("Model Confirmed")
        else:
            button.setEnabled(True)
            button.setText("Confirm Model Selection")
    
    def setInteractiveState(self, is_running: bool):
        self._isInteractive = is_running
        self.ui.enterInteractiveModeButton.setEnabled(not is_running)
        self.ui.stopInteractiveModeButton.setEnabled(is_running)

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
    
    def onSegmentChanged(self, segmentID):
        if not segmentID:
            return

        # The working mask and render key are segment-specific — switching
        # segments must invalidate both so the next render reads fresh data.
        self.logic.reset_render_state()

        # clearPrompts clears _interactive_point_stack and _preview_cp_ids,
        # recreates fresh prompt nodes, and re-wires the markups widgets.
        self._pauseRender = True
        try:
            self.clearPrompts()
        finally:
            self._pauseRender = False
    
    def clearPrompts(self):
        # Clear tracking state FIRST so any PointAddedEvent that fires when
        # the new nodes are wired to the markups widgets (the widget may
        # auto-create a placement cursor on an empty node) is recorded in
        # _preview_cp_ids and excluded from the render loop.
        self._interactive_point_stack.clear()
        self._preview_cp_ids.clear()

        # Recreate fresh markup nodes (counter=0).  This is the structural
        # fix for "starts at Positive 2": fresh nodes have never had a point
        # added, so the widget's first auto-cursor is always "Positive 1".
        # Reusing and clearing old nodes leaves the counter at N, causing the
        # next auto-cursor to be labeled "Positive N+1".
        self.logic.recreatePromptNodes(self._parameterNode)

        # Re-attach VTK observers (_onPointAdded, _onPointConfirmed, …) to
        # the newly created nodes; the old node references are now gone.
        self._observeMarkupsNodes()

        posNode, negNode = self.logic.getPromptNodes(self._parameterNode)

        # Deactivate all placement before wiring nodes.
        # setCurrentNode on an empty node inherits the widget's current
        # placement state — if the negative widget was in placement mode from
        # a previous session, calling setCurrentNode(negNode) would create a
        # "Negative 1" cursor regardless of call order.  Switching to
        # ViewTransform first ensures neither widget can auto-create a cursor.
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SwitchToViewTransformMode()

        # Wire both widgets while placement is inactive — no auto-cursor fires.
        self.ui.negativePrompts.setCurrentNode(negNode)
        self.ui.positivePrompts.setCurrentNode(posNode)

        # Explicitly start persistent placement on the POSITIVE node only.
        # This creates the "Positive 1" tracking cursor without touching the
        # negative widget's placement state.
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetActivePlaceNodeID(posNode.GetID())
        selectionNode.SetActivePlaceNodeClassName(posNode.GetClassName())
        interactionNode.SwitchToPersistentPlaceMode()

    def onUndo(self):
        """Ctrl+Z handler.

        Interactive mode: remove the most recently added prompt point.
        Non-interactive mode: restore the previous 2-D slice state from the
        per-segment undo stack.
        """
        in_interactive = bool(self.renderer and self.renderer.timer.isActive())

        if in_interactive:
            # Pop and remove the last added prompt point.
            while self._interactive_point_stack:
                node, cp_id = self._interactive_point_stack.pop()
                idx = node.GetControlPointIndexByID(cp_id)
                if idx >= 0:
                    node.RemoveNthControlPoint(idx)
                    return   # removed one point — done for this undo step
            # Nothing left to undo
        else:
            self.logic.undo(self)

    # -------------------------
    # SPX Boundary Overlay  (Ctrl+Q)
    # -------------------------

    def _hideSPXBoundary(self):
        """Remove the SPX boundary label from the slice view it was shown on."""
        if not self._spx_boundary_visible:
            return
        if self._spx_boundary_view:
            lm = slicer.app.layoutManager()
            sw = lm.sliceWidget(self._spx_boundary_view)
            if sw:
                sw.sliceLogic().GetSliceCompositeNode().SetLabelVolumeID("")
        self._spx_boundary_visible = False
        self._spx_boundary_view    = None

    def onToggleSPXBoundary(self):
        """Ctrl+Q handler — show or hide the SPX superpixel boundary overlay.

        If the boundary has not been computed for the current slice it is
        generated on the fly (reusing the SPX label-map cache when available).
        """
        if not isinstance(self.modelFamily, SPXModelFamily):
            return

        # Second press: hide.
        if self._spx_boundary_visible:
            self._hideSPXBoundary()
            return

        # First press (or after being hidden): compute and show.
        boundary_2d, axis, sliceIndex, err = self.logic.compute_spx_boundary(self)
        if boundary_2d is None:
            slicer.util.warningDisplay(f"SPX boundary: {err}")
            return

        volumeNode = self.ui.sourceVolumeSelector.currentNode()

        # Create the label-map node once; reuse across toggles.
        if (self._spx_boundary_node is None
                or not slicer.mrmlScene.IsNodePresent(self._spx_boundary_node)):
            self._spx_boundary_node = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLLabelMapVolumeNode', 'SPX Boundaries'
            )
            self._spx_boundary_node.CreateDefaultDisplayNodes()
            # Allocate image data with the same geometry as the source volume.
            sourceArray = slicer.util.arrayFromVolume(volumeNode)
            zeroArray = np.zeros(sourceArray.shape, dtype=np.uint8)
            slicer.util.updateVolumeFromArray(self._spx_boundary_node, zeroArray)
            ijkToRAS = vtk.vtkMatrix4x4()
            volumeNode.GetIJKToRASMatrix(ijkToRAS)
            self._spx_boundary_node.SetIJKToRASMatrix(ijkToRAS)
            # Yellow colour for boundary pixels (label index 1).
            # The default "Labels" color table is read-only; create a small
            # User-type table so we can freely set label 1 to yellow.
            colorNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLColorTableNode')
            colorNode.SetTypeToUser()
            colorNode.SetNumberOfColors(2)
            colorNode.SetColor(0, 0.0, 0.0, 0.0, 0.0)  # 0 → transparent
            colorNode.SetColor(1, 1.0, 1.0, 0.0, 1.0)  # 1 → yellow, opaque
            self._spx_boundary_node.GetDisplayNode().SetAndObserveColorNodeID(
                colorNode.GetID()
            )

        # Fill the boundary for the current slice; clear every other slice.
        boundaryArray = slicer.util.arrayFromVolume(self._spx_boundary_node)
        boundaryArray[:] = 0
        write_slice_to_volume(boundaryArray, boundary_2d, axis, sliceIndex)
        self._spx_boundary_node.GetImageData().Modified()
        self._spx_boundary_node.Modified()

        # Set as the label layer in the current slice view.
        viewName = self.currentViewName
        lm = slicer.app.layoutManager()
        composite = lm.sliceWidget(viewName).sliceLogic().GetSliceCompositeNode()
        composite.SetLabelVolumeID(self._spx_boundary_node.GetID())
        composite.SetLabelOpacity(0.8)

        self._spx_boundary_visible = True
        self._spx_boundary_view    = viewName

    def onAddSegment(self, *args):
        self._pauseRender = True
        try:
            segNode = self.getOrCreateSegmentationNode()

            if not segNode:
                slicer.util.warningDisplay("Please select a volume first.")
                return

            segmentation = segNode.GetSegmentation()
            existing = {
                segmentation.GetNthSegment(i).GetName()
                for i in range(segmentation.GetNumberOfSegments())
            }

            index = 1
            while f"Segment_{index}" in existing:
                index += 1

            name = f"Segment_{index}"

            segmentID = segmentation.AddEmptySegment(name)

            self.ui.segmentSelector.setCurrentSegmentID(segmentID)

        finally:
            self._pauseRender = False
    
    def onRemoveSegment(self, *args):
        self._pauseRender = True
        try:
            segNode = self.ui.segmentationNodeSelector.currentNode()
            segmentID = self.ui.segmentSelector.currentSegmentID()

            if not segNode or not segmentID:
                slicer.util.warningDisplay("No segment selected.")
                return

            segNode.GetSegmentation().RemoveSegment(segmentID)

        finally:
            self._pauseRender = False
    
    def getOrCreateSegmentationNode(self):
        volumeNode = self.ui.sourceVolumeSelector.currentNode()

        segNode = self.ui.segmentationNodeSelector.currentNode()

        # fallback to parameter node
        if not segNode and self._parameterNode:
            _, segNode = self.logic.getVolumeAndSegmentation(self._parameterNode)

        # create if needed
        if not segNode and volumeNode:
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segNode.CreateDefaultDisplayNodes()
            segNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)

            self.logic.setVolumeAndSegmentation(self._parameterNode, volumeNode, segNode)
            self._parameterNode.Modified()

        return segNode


#
# Logic
#

class SegmentHumanBodyLogic(ScriptedLoadableModuleLogic):

    def __init__(self, parent=None):
        super().__init__(parent)
        # --- Render-skip optimisation ---
        # Tracks the last (points, axis, slice, params) tuple that produced a
        # result.  If it matches the current frame we skip applyResult entirely,
        # avoiding a full 3D labelmap read/write on every idle tick.
        self._last_render_key = None

        # --- Working-mask optimisation ---
        # We keep a single pre-allocated numpy array for the current segment's
        # 3D labelmap.  Only the 2D slice changes each frame, so we update it
        # in-place and push back to Slicer, eliminating the per-frame
        # arrayFromSegmentBinaryLabelmap() call and its .copy().
        self._working_mask = None
        self._working_mask_segment = None   # (segNodeID, segmentID) sentinel

        # --- Non-interactive undo ---
        # Stores 2D slice snapshots keyed by (segNodeID, segmentID) so mask
        # changes from propagate/expand operations can be reversed.
        self._undo = UndoStack()

        # --- Interactive base mask ---
        # Full 3D snapshot of the segment taken when entering interactive mode.
        # Each render computes: result = (base_slice | pos_region) & ~neg_region
        # so that neg prompts erase existing painted data and removing a pos
        # prompt reverts its region to the base state rather than to zero.
        self._interactive_base_mask = None

        # --- Window / Level ---
        # Set by set_window_level() when the user clicks "Apply Window/Level".
        # None = raw values are passed to models (no normalization).
        # When set, each 2D slice is clipped to [level-window/2, level+window/2]
        # and scaled to [0, 255] before being passed to any model.
        # The underlying vtkMRMLScalarVolumeNode data is never written to.
        self._wl_window = None
        self._wl_level  = None

    def setDefaultParameters(self, parameterNode):
        pass

    def reset_render_state(self):
        """Clear per-segment caches.  Call whenever the active segment or
        volume changes so the next render reads fresh data from Slicer."""
        self._last_render_key = None
        self._working_mask = None
        self._working_mask_segment = None
        self._interactive_base_mask = None
        # W/L is intentionally NOT reset here — it is a user preference that
        # should persist across segment/volume changes until explicitly changed.

    def set_window_level(self, window, level):
        """Confirm W/L values for model inference.
        Subsequent calls to onRender / on_propagate will normalize each slice
        to [0, 255] using these values before passing it to the model.
        Call with (None, None) to revert to raw values.
        """
        self._wl_window = window
        self._wl_level  = level

    def _apply_wl_to_slice(self, img):
        """Delegate to ``apply_window_level`` using the confirmed W/L values.
        Returns the original array unchanged when no W/L has been confirmed.
        The source volume data is never modified.
        """
        return apply_window_level(img, self._wl_window, self._wl_level)

    def _ensure_interactive_base_mask(self, widget, volumeNode):
        """Lazily reload _interactive_base_mask when it was cleared by a segment
        switch while the renderer was still running.

        onSegmentChanged → reset_render_state() sets _interactive_base_mask to
        None.  Without this reload the render loop falls back to classic mode
        (no base mask) and negative prompts stop erasing existing painted data.
        """
        if self._interactive_base_mask is not None:
            return
        segNode = widget.ui.segmentSelector.currentNode()
        segmentID = widget.ui.segmentSelector.currentSegmentID()
        if segNode and segmentID and volumeNode:
            raw = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentID, volumeNode)
            self._interactive_base_mask = raw.copy()

    def undo(self, widget):
        """Restore the previous 2-D slice state for the active segment.

        Called by the Widget in non-interactive mode when Ctrl+Z is pressed.
        """
        segNode, segmentID = self.getCurrentSegment(widget)
        if not segNode or not segmentID:
            return

        entry = self._undo.pop(segNode.GetID(), segmentID)
        if entry is None:
            return

        axis, sliceIndex, slice_2d = entry

        volumeNode = widget.ui.sourceVolumeSelector.currentNode()
        if not volumeNode:
            return

        # Ensure the working mask is loaded for this segment.
        segment_key = (segNode.GetID(), segmentID)
        if self._working_mask is None or self._working_mask_segment != segment_key:
            raw = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentID, volumeNode)
            self._working_mask = raw.copy()
            self._working_mask_segment = segment_key

        write_slice_to_volume(self._working_mask, slice_2d, axis, sliceIndex)
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            self._working_mask, segNode, segmentID, volumeNode
        )

        # Invalidate the render key so the next interactive render re-applies
        # the current prompt points on top of the restored slice.
        self._last_render_key = None

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

    def recreatePromptNodes(self, parameterNode):
        """Replace any existing prompt markup nodes with brand-new ones.

        Fresh nodes have an internal label counter of 0, so the placement
        cursor the markups widget auto-creates on an empty node is always
        labeled 'Positive 1' / 'Negative 1' — regardless of how many points
        were placed and removed in previous interactive sessions.
        """
        configs = {
            POS_NODE: ([0, 1, 0], 'positive'),
            NEG_NODE: ([1, 0, 0], 'negative'),
        }
        for ref_name, (color, label) in configs.items():
            old = parameterNode.GetNodeReference(ref_name)
            if old:
                slicer.mrmlScene.RemoveNode(old)
            node = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLMarkupsFiducialNode', label
            )
            node.CreateDefaultDisplayNodes()
            dn = node.GetDisplayNode()
            dn.SetSelectedColor(*color)
            dn.SetColor(*color)
            dn.SetActiveColor(*color)
            node.SetHideFromEditors(True)
            parameterNode.SetNodeReferenceID(ref_name, node.GetID())

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
    def onRender(self, modelFamily, widget):
        if not modelFamily or not modelFamily.model:
            return

        parameterNode = widget._parameterNode
        posNode, negNode = self.getPromptNodes(parameterNode)

        # --- Volume and slice info first — needed for the hover slice check ---
        volumeNode = widget.ui.sourceVolumeSelector.currentNode()
        if not volumeNode:
            return

        axis, sliceIndex = self.getAxisAndSlice(widget, volumeNode)

        # --- Extract placed prompt points ---
        # Uses collect_confirmed_points / collect_preview_points from core.utils.
        # A point is excluded from "confirmed" when BOTH:
        #   1. Its status is PositionPreview (unconfirmed cursor)
        #   2. Its ID is still in _preview_cp_ids (not yet confirmed by click)
        # This dual condition means a point whose ID was removed from _preview_cp_ids
        # (by _onPointConfirmed) is included even if its status stayed at Preview.
        preview_ids = widget._preview_cp_ids

        def _node_records(node):
            if not node:
                return []
            return [
                (node.GetNthControlPointPositionStatus(i),
                 node.GetNthControlPointID(i),
                 node.GetNthControlPointPosition(i))
                for i in range(node.GetNumberOfControlPoints())
            ]

        pos_points = collect_confirmed_points(_node_records(posNode), preview_ids)
        neg_points = collect_confirmed_points(_node_records(negNode), preview_ids)

        # --- Hover preview ---
        # When the Preview checkbox is on, re-include the unconfirmed
        # PositionPreview control points that collect_confirmed_points filtered out.
        # Slicer continuously updates each PositionPreview point's position to
        # track the cursor — so it IS the hover point.  A PositionPreview point
        # only exists after the user has explicitly clicked "+" in the markups
        # widget, so this never fires without a deliberate user action.
        # Once the user clicks on the slice the point is confirmed and its ID is
        # removed from _preview_cp_ids, moving it to pos/neg_points instead.
        if widget.ui.previewCheckBox.isChecked():
            pos_points = list(pos_points) + collect_preview_points(_node_records(posNode), preview_ids)
            neg_points = list(neg_points) + collect_preview_points(_node_records(negNode), preview_ids)

        # --- Build render key ---
        params = widget.getUserParameters()
        if params is None:
            return

        render_key = (
            tuple(tuple(p) for p in pos_points),
            tuple(tuple(p) for p in neg_points),
            axis,
            sliceIndex,
            tuple(sorted(params.items())) if params else (),
        )

        if render_key == self._last_render_key:
            return

        # --- Defensive segment check ---
        # When any prompt cursor reaches the slice (confirmed OR still in
        # PositionPreview hover state), guarantee a segment holder exists
        # BEFORE running the model.  This is the proactive check: create the
        # segment at the moment the user interacts, not at mode-entry time.
        # The cursor stays in its current PositionPreview state (unassigned)
        # — the user still has to click to confirm it.  The model runs with it
        # as a preview so the user sees what would be selected before clicking.
        if pos_points or neg_points:
            seg = widget.ui.segmentSelector.currentNode()
            seg_id = widget.ui.segmentSelector.currentSegmentID()
            if not seg or not seg_id:
                self._ensure_seg_and_segment(widget, volumeNode)
                # Snapshot the just-created (all-zeros) segment as the base so
                # subsequent renders have a clean baseline to build on.
                seg = widget.ui.segmentSelector.currentNode()
                seg_id = widget.ui.segmentSelector.currentSegmentID()
                if seg and seg_id and self._interactive_base_mask is None:
                    raw = slicer.util.arrayFromSegmentBinaryLabelmap(seg, seg_id, volumeNode)
                    self._interactive_base_mask = raw.copy()

        # Reload base mask if it was cleared by a segment switch mid-session.
        self._ensure_interactive_base_mask(widget, volumeNode)
        has_base = self._interactive_base_mask is not None
        if not pos_points and not neg_points and not has_base:
            self._last_render_key = render_key
            return

        # --- Compute result ---
        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        img = get_slice_from_volume(volumeArray, axis, sliceIndex)
        img = self._apply_wl_to_slice(img)

        scribbles_ijk = self._ras_to_ijk(volumeNode, {
            "positive": pos_points,
            "negative": neg_points,
        }, axis)

        # Pass the base slice so the family can:
        #   - add pos regions on top of pre-existing painted data
        #   - erase neg regions from pre-existing painted data
        #   - restore base when all pos prompts are removed
        base_slice = (
            get_slice_from_volume(self._interactive_base_mask, axis, sliceIndex)
            if has_base else None
        )

        result = call_if_exists(
            modelFamily,
            "onRender",
            img=img,
            pos_points=scribbles_ijk["positive"],
            neg_points=scribbles_ijk["negative"],
            base_mask=base_slice,
            **params
        )

        # Always record the key — even when result is None — so the same
        # computation is not repeated on the next tick.
        self._last_render_key = render_key
        if result is not None:
            self.applyResult(widget, result, axis, sliceIndex)
    
    def _ensure_seg_and_segment(self, widget, volumeNode):
        """Guarantee a segmentation node and at least one segment exist.

        Creates them if absent and updates the UI selectors (with signals
        blocked so no downstream cascades fire).  Returns (segNode, segmentID).
        Safe to call multiple times — a no-op when everything already exists.
        """
        segNode = widget.ui.segmentSelector.currentNode()
        segmentID = widget.ui.segmentSelector.currentSegmentID()

        if not segNode:
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segNode.CreateDefaultDisplayNodes()
            segNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
            self.setVolumeAndSegmentation(widget._parameterNode, volumeNode, segNode)
            widget.ui.segmentationNodeSelector.blockSignals(True)
            widget.ui.segmentationNodeSelector.setCurrentNode(segNode)
            widget.ui.segmentationNodeSelector.blockSignals(False)
            widget.ui.segmentSelector.blockSignals(True)
            widget.ui.segmentSelector.setCurrentNode(segNode)
            widget.ui.segmentSelector.blockSignals(False)

        if not segmentID:
            segmentID = segNode.GetSegmentation().AddEmptySegment("Segment_1")
            widget.ui.segmentSelector.blockSignals(True)
            widget.ui.segmentSelector.setCurrentSegmentID(segmentID)
            widget.ui.segmentSelector.blockSignals(False)
            widget.ui.addSegmentButton.setEnabled(True)

        return segNode, segmentID

    def applyResult(self, widget, mask2d, axis, sliceIndex):
        volumeNode = widget.ui.sourceVolumeSelector.currentNode()
        if not volumeNode:
            return

        segNode = widget.ui.segmentSelector.currentNode()
        segmentID = widget.ui.segmentSelector.currentSegmentID()
        if not segNode or not segmentID:
            # Should not reach here — onRender creates the segment before
            # calling applyResult.  Guard defensively rather than auto-create,
            # which would re-trigger updateGUIFromParameterNode mid-render.
            return

        # Maintain a persistent 3D working mask so we only read the full
        # labelmap from Slicer once per segment, not every frame.
        # We update just the 2D slice in-place each render.
        segment_key = (segNode.GetID(), segmentID)
        if self._working_mask is None or self._working_mask_segment != segment_key:
            raw = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentID, volumeNode)
            self._working_mask = raw.copy()
            self._working_mask_segment = segment_key

        write_slice_to_volume(self._working_mask, mask2d, axis, sliceIndex)

        slicer.util.updateSegmentBinaryLabelmapFromArray(
            self._working_mask, segNode, segmentID, volumeNode
        )

    def compute_spx_boundary(self, widget):
        """Compute SPX superpixel boundary pixels for the current slice.

        Reuses the SPX label-map cache when available (no extra forward pass
        if the user is already in interactive mode or has propagated).  Falls
        back to running the model if the cache is empty.

        Returns (boundary_uint8_2d, axis, sliceIndex, error_msg).
        On success error_msg is None; on failure boundary/axis/sliceIndex are None.
        """
        modelFamily = widget.modelFamily
        if not isinstance(modelFamily, SPXModelFamily):
            return None, None, None, "Please select an SPX model family first."
        if not modelFamily.model:
            return None, None, None, "Please confirm a model first (click 'Confirm Model')."

        volumeNode = widget.ui.sourceVolumeSelector.currentNode()
        if not volumeNode:
            return None, None, None, "Please select an image volume first."

        axis, sliceIndex = self.getAxisAndSlice(widget, volumeNode)

        # Always go through on_propagate so its cache key (which includes
        # img.shape) is validated against the current axis/slice.  Bypassing
        # it with a raw _cache_labels check causes a shape mismatch when the
        # user switches slice planes (e.g. Red → Green) after the model ran.
        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        img = get_slice_from_volume(volumeArray, axis, sliceIndex)
        img = self._apply_wl_to_slice(img)
        params = widget.getUserParameters()
        if params is None:
            return None, None, None, "Invalid model parameters."
        labels = modelFamily.on_propagate(img=img, **params)

        if labels is None:
            return None, None, None, "SPX model returned no labels for this slice."

        return spx_boundary_mask(labels), axis, sliceIndex, None

    def on_confirm_model(self, widget):
        if not widget.modelFamily:
            return

        widget.modelFamily.confirm_model()
        
    def on_propagate(self, widget):

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

        params = widget.getUserParameters()

        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        axis, sliceIndex = self.getAxisAndSlice(widget, volumeNode)
        img = get_slice_from_volume(volumeArray, axis, sliceIndex)
        img = self._apply_wl_to_slice(img)

        # Collect confirmed negative prompt points using the same helper
        # used by onRender, so the filtering logic is not duplicated.
        _, negNode = self.getPromptNodes(widget._parameterNode)
        preview_ids = widget._preview_cp_ids

        def _node_records(node):
            if not node:
                return []
            return [
                (node.GetNthControlPointPositionStatus(i),
                 node.GetNthControlPointID(i),
                 node.GetNthControlPointPosition(i))
                for i in range(node.GetNumberOfControlPoints())
            ]

        neg_points_ras = collect_confirmed_points(_node_records(negNode), preview_ids)
        neg_ijk = self._ras_to_ijk(
            volumeNode, {"positive": [], "negative": neg_points_ras}, axis
        )["negative"]

        # Delegate to the family so the correct algorithm and user params are
        # used, and the SPX label cache is consulted before recomputing.
        labels = call_if_exists(modelFamily, 'on_propagate', img=img, **params)

        if labels is None:
            slicer.util.warningDisplay("This model does not support propagation.")
            return

        self.expandSegWithSPX(segNode, segmentID, volumeNode, labels, axis, sliceIndex,
                              neg_points=neg_ijk)
    
    
    def on_enter_interactive(self, widget):
        if not widget.renderer:
            widget.renderer = SegmentationRenderer(widget)

        # Mark interactive BEFORE clearPrompts so that any parameterNode
        # ModifiedEvent fired during node recreation (recreatePromptNodes calls
        # SetNodeReferenceID which fires ModifiedEvent → updateGUIFromParameterNode)
        # sees _isInteractive=True and skips the setCurrentNode calls that would
        # otherwise activate the negative widget's placement mode.
        widget.setInteractiveState(True)

        # Create fresh markup nodes (counter=0) so the widget's first
        # placement cursor is always "Positive 1".  Re-attaches observers
        # and wires the new nodes to the markups widgets.
        widget.clearPrompts()

        # Start from a clean render state.
        self.reset_render_state()

        # If a segment already exists, snapshot it now as the interactive
        # base so the render loop can restore original data when prompts are
        # removed.  If no segment exists yet, the snapshot is taken inside
        # onRender the first time a prompt cursor reaches the slice — ensuring
        # the segment is created only when the user actually interacts, and
        # keeping the cursor in its PositionPreview (unassigned) state.
        volumeNode = widget.ui.sourceVolumeSelector.currentNode()
        segNode = widget.ui.segmentSelector.currentNode()
        segmentID = widget.ui.segmentSelector.currentSegmentID()
        if volumeNode and segNode and segmentID:
            raw = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentID, volumeNode)
            self._interactive_base_mask = raw.copy()

        widget.ui.previewCheckBox.setChecked(True)
        widget.renderer.start()

    def on_stop_interactive(self, widget):
        if widget.renderer:
            widget.renderer.stop()
            widget.setInteractiveState(False)

        widget.ui.previewCheckBox.setChecked(False)
        self.reset_render_state()

    def on_assign_2d(self, widget):
        call_if_exists(widget.modelFamily, 'on_assign_2d')

    def on_assign_3d(self, widget):
        call_if_exists(widget.modelFamily, 'on_assign_3d')
    
    def on_automatic_segmentation(self, widget):
        call_if_exists(widget.modelFamily, 'on_automatic_segmentation')


    def _ras_to_ijk(self, volumeNode, scrib, axis):
        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)

        # Extract the 4×4 VTK matrix into numpy once, then batch-multiply all
        # points at once instead of calling MultiplyPoint in a Python loop.
        mat = np.array([[rasToIjk.GetElement(r, c) for c in range(4)]
                        for r in range(4)])  # (4, 4)

        # axis → (x_col, y_col) in the IJK triple (I=0, J=1, K=2)
        # axis=0 Red/axial:      slice is array[k,:,:], 2D pt = [I, J]
        # axis=1 Green/coronal:  slice is array[:,j,:], 2D pt = [I, K]
        # axis=2 Yellow/sagittal:slice is array[:,:,i], 2D pt = [J, K]
        axis_to_xy_cols = {0: (0, 1), 1: (0, 2), 2: (1, 2)}
        xc, yc = axis_to_xy_cols[axis]

        def convert(points):
            if not points:
                return []
            pts = np.array(points, dtype=np.float64)          # (N, 3)
            pts_h = np.hstack([pts, np.ones((len(pts), 1))])  # (N, 4)
            ijk = (mat @ pts_h.T).T[:, :3].astype(int)        # (N, 3)
            return ijk[:, [xc, yc]].tolist()

        return {
            "positive": convert(scrib["positive"]),
            "negative": convert(scrib["negative"])
        }

    def getAxisAndSlice(self, widget, volumeNode=None):
        viewName = widget.currentViewName

        lm = slicer.app.layoutManager()
        sliceWidget = lm.sliceWidget(viewName)

        if viewName == "Red":
            axis = 0
        elif viewName == "Green":
            axis = 1
        else:
            axis = 2

        if volumeNode is not None:
            # Convert the slice plane's RAS origin to the volume's IJK space so
            # that the index is correct regardless of the volume's spacing/origin.
            sliceNode = sliceWidget.mrmlSliceNode()
            sliceToRAS = sliceNode.GetSliceToRAS()
            ras = [sliceToRAS.GetElement(r, 3) for r in range(3)]
            rasToIjk = vtk.vtkMatrix4x4()
            volumeNode.GetRASToIJKMatrix(rasToIjk)
            ijk = rasToIjk.MultiplyPoint(ras + [1])
            # axis=0 (Red/axial)    → K = ijk[2]
            # axis=1 (Green/coronal) → J = ijk[1]
            # axis=2 (Yellow/sagittal) → I = ijk[0]
            component = {0: 2, 1: 1, 2: 0}[axis]
            sliceIndex = int(round(ijk[component]))
        else:
            logic = sliceWidget.sliceLogic()
            sliceIndex = logic.GetSliceIndexFromOffset(logic.GetSliceOffset()) - 1

        return axis, sliceIndex
    
    def getCurrentSegment(self, widget):
        segNode = widget.ui.segmentSelector.currentNode()
        segmentID = widget.ui.segmentSelector.currentSegmentID()

        if not segmentID and segNode:
            segmentation = segNode.GetSegmentation()
            if segmentation.GetNumberOfSegments() > 0:
                segmentID = segmentation.GetNthSegmentID(0)

        return segNode, segmentID


    def expandSegWithSPX(self, segNode, segmentID, volumeNode, labels, axis, sliceIndex,
                         neg_points=None):
        mask3d = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentID, volumeNode)

        sliceMask = get_slice_from_volume(mask3d, axis, sliceIndex)

        # Snapshot the current slice before modifying it so Ctrl+Z can restore.
        self._undo.push(segNode.GetID(), segmentID, axis, sliceIndex, sliceMask)

        selected_labels = set(np.unique(labels[sliceMask > 0]).tolist())

        # Remove regions touched by negative prompt points.
        if neg_points:
            selected_labels -= labels_at_points(neg_points, labels)

        expanded = np.isin(labels, list(selected_labels)).astype(np.uint8)

        # Preserve all other annotated slices — only replace the current one.
        fullMask = mask3d.copy()
        write_slice_to_volume(fullMask, expanded, axis, sliceIndex)

        slicer.util.updateSegmentBinaryLabelmapFromArray(
            fullMask, segNode, segmentID, volumeNode
        )

        # Keep the working mask in sync so the next interactive render does
        # not overwrite the propagation result with stale data.
        segment_key = (segNode.GetID(), segmentID)
        if self._working_mask_segment == segment_key:
            write_slice_to_volume(self._working_mask, expanded, axis, sliceIndex)


#
# SegmentHumanBodyTest
#

class SegmentHumanBodyTest(ScriptedLoadableModuleTest):
    """Entry point for the 3D Slicer "Reload and Test" button.

    The actual test cases live in Testing/Python/SegmentHumanBodyTest.py as a
    standard unittest.TestCase so they can also be run via
    slicer_add_python_unittest in CMake.  This class discovers and delegates
    to them so that "Reload and Test" remains a meaningful action.
    """

    def runTest(self):
        import importlib
        import os
        import sys
        import unittest

        test_dir = os.path.join(os.path.dirname(__file__), 'Testing', 'Python')
        if test_dir not in sys.path:
            sys.path.insert(0, test_dir)

        import SegmentHumanBodyTest as ext
        importlib.reload(ext)

        suite = unittest.TestLoader().loadTestsFromTestCase(ext.SegmentHumanBodyLogicTest)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise Exception(
                f'{len(result.failures) + len(result.errors)} test(s) failed — '
                'see the Python console for details'
            )
