import qt, vtk, slicer
import logging
import numpy as np
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin

from core.modelFamilies import BaseModelFamily, SPXModelFamily, FAMILY_REGISTRY
from core.utils import (
    call_if_exists,
    write_slice_to_volume,
    next_segment_name,
    parse_user_parameters,
)
from core.modelRegistry import ModelRegistry
from core._logic import SegmentHumanBodyLogic
from core._state import WidgetState
from core._input import StrokeHandler, BrushHandler, EraseHandler, PointHandler

log = logging.getLogger(__name__)

#
# Module
#

class SegmentHumanBody(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = 'SegmentHumanBodyV2'
        self.parent.categories = ['Segmentation']
        self.parent.contributors = [
            'Yixin Zhang (Duke University)',
            'Zafer Yildiz (Duke University)',
        ]
        self.parent.acknowledgementText = (
            'Developed at Duke University.'
        )


#
# Widget
#

class SegmentHumanBodyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        super().__init__(parent)
        VTKObservationMixin.__init__(self)

        self.logic = SegmentHumanBodyLogic()
        self.ctrl = WidgetState(self)
        self._parameterNode = None
        self.modelFamily = None
        self.currentViewName = None  # default

        # Single unified action history for Ctrl+Z.  Each entry is a list:
        #   ['brush',  change]                  — Paint stroke
        #   ['erase',  change]                  — Erase stroke
        #   ['expand', change]                  — expand operation
        #   ['point',  change, node, cp_id]     — confirmed prompt control point
        # ``change`` is a MaskChange (or None when the action produced no net
        # mask change).  Undo pops the last entry and calls reverse_delta if
        # a change is present; for 'point' entries the control point is also
        # removed from the markup node.  Lists (not tuples) are used so that
        # the 'point' path can fill ``change`` after the async render completes.
        self._history = []
        self._undo_shortcut = None

        # The currently active input handler (BrushHandler, EraseHandler, or
        # PointHandler), or None before setup completes.  Mutual exclusion is
        # enforced by InputHandler._detach_current_tool_if_exists: every
        # attach() call deactivates the previous handler first.
        self._active_handler = None

        # SPX boundary overlay
        self._spx_boundary_node    = None   # vtkMRMLLabelMapVolumeNode
        self._spx_boundary_visible = False
        self._spx_boundary_view    = None   # view name the label is currently set on
        self._spx_boundary_shortcut = None
        self._expand_shortcut    = None

    # -------------------------
    # Setup / Cleanup
    # -------------------------
    def cleanup(self):
        """Called by Slicer when the module is unloaded."""
        if self._active_handler:
            self._active_handler.detach(self)
        super().cleanup()

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

        self.model_classes = FAMILY_REGISTRY

        self.initializeUI()
        self.connectSignals()

        # Lock selectors
        self.ui.positivePrompts.setNodeSelectorVisible(False)
        self.ui.negativePrompts.setNodeSelectorVisible(False)


        # Ctrl+Z undo shortcut — parented to the module widget so it is
        # active only while this panel is visible.
        self._undo_shortcut = qt.QShortcut(qt.QKeySequence("Ctrl+Z"), uiWidget)
        self._undo_shortcut.connect('activated()', self.onUndo)

        # Q — toggle SPX superpixel boundary overlay
        self._spx_boundary_shortcut = qt.QShortcut(qt.QKeySequence("Q"), uiWidget)
        self._spx_boundary_shortcut.connect('activated()', self.onToggleSPXBoundary)

        # E — expand selected label
        # When the brush or erase tool is active, temporarily deactivate it
        # before running expand so no active stroke is held open, then
        # restore the tool afterward so the user can keep painting.
        self._expand_shortcut = qt.QShortcut(qt.QKeySequence("E"), uiWidget)
        self._expand_shortcut.connect('activated()', self._onExpandShortcut)

        qt.QTimer.singleShot(0, self._initializeAfterSetup)

        log.debug('[Setup complete]')


    def _add_history(self, entry):
        """Append *entry* to the undo history (called by input handlers)."""
        self._history.append(entry)

    def _resolveActiveView(self):
        """Update currentViewName to whichever slice view the cursor is over.

        Called lazily at the moments that need an accurate view (expand, point
        confirmed) rather than on every mouse-move event.
        """
        lm = slicer.app.layoutManager()
        for viewName in ("Red", "Green", "Yellow"):
            sw = lm.sliceWidget(viewName)
            if sw and sw.sliceView().underMouse():
                self.currentViewName = viewName
                return

    def _onExpand(self):
        """Run expand and record the result in history.

        Shared by the E shortcut and the Expand button.  Returns immediately
        when pre-conditions fail (handled inside ``on_expand``).
        """
        self._resolveActiveView()
        change = self.logic.on_expand(self)
        if change is not None:
            self._history.append(['expand', change])

    def _onExpandShortcut(self):
        """Handle the E hotkey: flush any active stroke → expand → restore tool."""
        active = self._active_handler
        prior_stroke_class = type(active) if isinstance(active, StrokeHandler) else None
        if prior_stroke_class:
            active.detach(self)
        self._onExpand()
        if prior_stroke_class:
            prior_stroke_class().attach(self)


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
        # Pre-initialize the Segment Editor widget so the brush button works without
        # a visible module switch later.  Both selectModule calls happen before the
        # event loop repaints, so the user sees no flash.
        qt.QTimer.singleShot(0, self._preloadSegmentEditor)
        # Slice observers are wired by _observeMarkupsNodes (called above via
        # setParameterNode).  No direct call needed here.

    def _connectSliceObservers(self):
        """Add VTK observer for interaction mode changes.

        Registered here so it survives the removeObservers() call inside
        _observeMarkupsNodes.
        """
        # Deactivate the brush when the user switches to point-placement mode.
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if interactionNode:
            self.addObserver(interactionNode,
                             vtk.vtkCommand.ModifiedEvent,
                             self._onInteractionModeChanged)

    def _onPlaceModeChanged(self, active: bool):
        """Qt slot: fires when the user clicks either markup place-widget button.

        Activates PointHandler (flushing and detaching any active stroke handler)
        when placement mode is turned on.  No-op when turned off — the point
        handler stays registered as the active handler in the background.
        """
        if not active or self.ctrl.is_paused:
            return
        if not isinstance(self._active_handler, PointHandler):
            PointHandler().attach(self)

    def _onInteractionModeChanged(self, caller=None, event=None):
        """Activate point mode when Slicer enters placement mode for our nodes."""
        if self.ctrl.is_paused:
            return
        interactionNode = caller
        if interactionNode.GetCurrentInteractionMode() != interactionNode.Place:
            return
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        posNode, negNode = self.logic.getPromptNodes(self._parameterNode)
        activePlaceID = selectionNode.GetActivePlaceNodeID()
        if activePlaceID not in (
            posNode.GetID() if posNode else None,
            negNode.GetID() if negNode else None,
        ):
            return
        if not isinstance(self._active_handler, PointHandler):
            PointHandler().attach(self)

    def _preloadSegmentEditor(self):
        """Silently initialize the Segment Editor module widget if not done yet.

        Triggers creation by switching to it and immediately back — both happen
        in the same call stack so the user sees no flash.  activeEffectChanged
        is wired per-handler in StrokeHandler.attach; no global connection here.
        """
        if slicer.modules.segmenteditor.widgetRepresentation() is None:
            slicer.util.selectModule('SegmentEditor')
            slicer.util.selectModule(self.moduleName)


    # -------------------------
    # Signals
    # -------------------------
    def connectSignals(self):
        ui = self.ui

        # Buttons that call a method directly on modelFamily (no logic guards needed)
        for ui_name, method_name in [
            ('assignLabel2D',          'on_assign_2d'),
            ('assignLabel3D',          'on_assign_3d'),
            ('runAutomaticSegmentation', 'on_automatic_segmentation'),
        ]:
            getattr(ui, ui_name).connect('clicked(bool)', self.bind(method_name, target="model"))

        # on_expand has guards, neg-point collection, etc. — handled by Logic
        ui.expandSelectedLabelButton.connect('clicked(bool)', lambda _=None: self._onExpand())

        widget_button_connections = [
            ('goToMarkupsButton', self.on_go_to_markups),
            ('confirmModelSelection', self.onConfirmClicked),
            ('addSegmentButton', self.onAddSegment),
            ('removeSegmentButton', self.onRemoveSegment),
            ('applyWindowLevelButton', self.onApplyWindowLevel),
        ]

        for ui_name, method in widget_button_connections:
            getattr(ui, ui_name).connect('clicked(bool)', method)

        # Checkboxes
        ui.showSPXBoundaryCheckBox.connect('toggled(bool)', self.onToggleSPXBoundary)

        ui.modelFamilyDropdown.connect('currentIndexChanged(int)', self.onModelFamilyChanged)
        ui.modelVariantDropdown.connect('currentIndexChanged(int)', self.onVariantChanged)
        ui.sourceVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        ui.segmentationNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        ui.segmentSelector.connect("currentSegmentChanged(QString)", self.onSegmentChanged)

        # Window/Level slider ↔ spinbox sync + live display preview
        ui.windowSlider.connect('valueChanged(int)', self._onWindowSliderChanged)
        ui.windowSpinBox.connect('valueChanged(int)', self._onWindowSpinBoxChanged)
        ui.levelSlider.connect('valueChanged(int)', self._onLevelSliderChanged)
        ui.levelSpinBox.connect('valueChanged(int)', self._onLevelSpinBoxChanged)

        # Brush / Erase toggles + diameter + shape
        ui.brushToolButton.connect('toggled(bool)', self.onBrushToggled)
        ui.eraseToolButton.connect('toggled(bool)', self.onEraseToggled)
        ui.brushDiameterSlider.connect('valueChanged(int)', self._onBrushDiameterSliderChanged)
        ui.brushDiameterSpinBox.connect('valueChanged(int)', self._onBrushDiameterSpinBoxChanged)
        ui.brushSphereCheckBox.connect('toggled(bool)', lambda _: self._applyBrushParams())

        # Point placement widgets — activating either place widget must deactivate
        # the active stroke handler.  Connected via Qt signal (not the VTK observer)
        # so it fires immediately when the user clicks the place button.
        ui.positivePrompts.activeMarkupsFiducialPlaceModeChanged.connect(
            self._onPlaceModeChanged)
        ui.negativePrompts.activeMarkupsFiducialPlaceModeChanged.connect(
            self._onPlaceModeChanged)

    # -------------------------
    # Observers
    # -------------------------
    def _observeMarkupsNodes(self):
        # removeObservers() wipes ALL VTK observers added via addObserver,
        # including the parameterNode → updateGUIFromParameterNode observer set
        # in setParameterNode.  Re-add it immediately so GUI updates keep working.
        self.removeObservers()
        if self._parameterNode:
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )

        posNode, negNode = self.logic.getPromptNodes(self._parameterNode)

        for node in [posNode, negNode]:
            if node:
                self.addObserver(
                    node,
                    slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                    self._onPointConfirmed
                )
                self.addObserver(
                    node,
                    slicer.vtkMRMLMarkupsNode.PointRemovedEvent,
                    self._onPointRemoved
                )

        # Re-add slice-node observers that removeObservers() wiped.
        self._connectSliceObservers()

    def _onPointConfirmed(self, caller=None, event=None):
        """PointPositionDefinedEvent — a placement was just confirmed by the user.

        Treats the point as a superpixel brush/erase stroke: runs the SPX model,
        finds the label at the click position, and writes the selection directly
        through SegmentTracker.write_slice() — the same single write path used
        by BrushHandler and EraseHandler.  The resulting MaskChange is stored in
        history immediately (no async timer needed).
        """
        if caller is None:
            return

        self._resolveActiveView()
        n = caller.GetNumberOfControlPoints()
        if n == 0:
            return
        cp_id = caller.GetNthControlPointID(n - 1)
        change = self.logic.commit_point(self, caller, cp_id)
        self._history.append(['point', change, caller, cp_id])
        log.debug('[Widget] point confirmed — change=%s  history=%d',
                  change is not None, len(self._history))

    def _onPointRemoved(self, caller=None, event=None):
        """PointRemovedEvent — a prompt point was manually deleted.

        Finds the removed point's history entry by comparing current cp_ids
        against history, then reverses its MaskChange through the single write
        path.  Paused during Ctrl+Z and clearPrompts (those paths handle their
        own mask reversal explicitly).
        """
        if self.ctrl.is_paused:
            return
        current_ids = {caller.GetNthControlPointID(i)
                       for i in range(caller.GetNumberOfControlPoints())}
        for i in range(len(self._history) - 1, -1, -1):
            entry = self._history[i]
            if entry[0] == 'point' and entry[2] is caller and entry[3] not in current_ids:
                change = entry[1]
                del self._history[i]
                if change is not None:
                    self.logic.reverse_change(self, change)
                return

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

    def _sync_wl_widgets(self, peer, value):
        """Copy *value* to *peer* (blocking signals) then run W/L change logic."""
        peer.blockSignals(True)
        peer.setValue(value)
        peer.blockSignals(False)
        self._onWLControlChanged()

    def _onWindowSliderChanged(self, value):
        self._sync_wl_widgets(self.ui.windowSpinBox, value)

    def _onWindowSpinBoxChanged(self, value):
        self._sync_wl_widgets(self.ui.windowSlider, value)

    def _onLevelSliderChanged(self, value):
        self._sync_wl_widgets(self.ui.levelSpinBox, value)

    def _onLevelSpinBoxChanged(self, value):
        self._sync_wl_widgets(self.ui.levelSlider, value)

    # -------------------------
    # Brush tool
    # -------------------------

    def _segEditor(self):
        """Return the Segment Editor's qMRMLSegmentEditorWidget, or None."""
        try:
            return slicer.modules.segmenteditor.widgetRepresentation().self().editor
        except Exception:
            return None

    def _applyBrushParams(self):
        """Push diameter and shape to the currently active Paint or Erase effect."""
        editor = self._segEditor()
        if not editor:
            return
        effect = editor.activeEffect()
        if effect and effect.name in ("Paint", "Erase"):
            effect.setParameter("BrushAbsoluteDiameter",
                                str(self.ui.brushDiameterSpinBox.value))
            effect.setParameter("BrushDiameterIsAbsolute", "1")
            effect.setParameter("BrushSphere",
                                "1" if self.ui.brushSphereCheckBox.isChecked() else "0")

    def onBrushToggled(self, checked: bool):
        if checked:
            BrushHandler().attach(self)
        elif isinstance(self._active_handler, BrushHandler):
            self._active_handler.detach(self)
            PointHandler().attach(self)

    def onEraseToggled(self, checked: bool):
        if checked:
            EraseHandler().attach(self)
        elif isinstance(self._active_handler, EraseHandler):
            self._active_handler.detach(self)
            PointHandler().attach(self)

    def _onBrushDiameterSliderChanged(self, value):
        self.ui.brushDiameterSpinBox.blockSignals(True)
        self.ui.brushDiameterSpinBox.setValue(value)
        self.ui.brushDiameterSpinBox.blockSignals(False)
        self._applyBrushParams()

    def _onBrushDiameterSpinBoxChanged(self, value):
        self.ui.brushDiameterSlider.blockSignals(True)
        self.ui.brushDiameterSlider.setValue(value)
        self.ui.brushDiameterSlider.blockSignals(False)
        self._applyBrushParams()

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
        visible = self.modelFamily.VISIBLE_BUTTONS if self.modelFamily else frozenset()
        ALL_BUTTONS = {
            'assignLabel2D', 'assignLabel3D',
            'expandSelectedLabelButton', 'showSPXBoundaryCheckBox',
            'runAutomaticSegmentation', 'goToMarkupsButton', 'samMaskDropdown',
        }
        for name in ALL_BUTTONS:
            getattr(self.ui, name).setVisible(name in visible)

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
        # View is now auto-detected from mouse position; hide the manual dropdown.
        self.ui.sliceViewDropdown.setVisible(False)

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

        posNode, negNode = self.logic.getPromptNodes(self._parameterNode)
        volumeNode, segNode = self.logic.getVolumeAndSegmentation(self._parameterNode)

        self.ctrl.pause()
        try:
            self.ui.positivePrompts.setCurrentNode(posNode)
            self.ui.negativePrompts.setCurrentNode(negNode)
        finally:
            self.ctrl.resume()

        self.ui.sourceVolumeSelector.setCurrentNode(volumeNode)
        self.ui.segmentationNodeSelector.setCurrentNode(segNode)
        self.ui.segmentSelector.setCurrentNode(segNode)
        self.ui.addSegmentButton.setEnabled(segNode is not None)

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
            segNode.CreateClosedSurfaceRepresentation()

        self.logic.setPromptNodes(
            self._parameterNode,
            self.ui.positivePrompts.currentNode(),
            self.ui.negativePrompts.currentNode(),
        )

        self.logic.setVolumeAndSegmentation(self._parameterNode, volumeNode, segNode)

        self._parameterNode.Modified()  # triggers updateGUIFromParameterNode via observer
    
    def getUserParameters(self):
        return parse_user_parameters(self.ui.paramTextEdit.toPlainText())
    
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
        if not self.modelFamily or not self.modelFamily.variant:
            return
        try:
            self.logic.on_confirm_model(self)
        except ValueError as exc:
            slicer.util.warningDisplay(str(exc))
            return
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
        if not segmentID or self.ctrl.creating_segment:
            return

        # Keep the Segment Editor's active tool in sync with the selected segment.
        if self.ui.brushToolButton.isChecked() or self.ui.eraseToolButton.isChecked():
            editor = self._segEditor()
            if editor:
                editor.setCurrentSegmentID(segmentID)
                self.ctrl.brush_in_progress = False

        # clearPrompts resets the prompt stack and preview IDs,
        # recreates fresh prompt nodes, and re-wires the markups widgets.
        self.ctrl.pause()
        try:
            self.clearPrompts()
        finally:
            self.ctrl.resume()
        segNode = self.ui.segmentSelector.currentNode()
        if segNode and segmentID:
            self._history.clear()
            if isinstance(self._active_handler, StrokeHandler):
                self._active_handler.reset(self)

    def clearPrompts(self):
        self._history.clear()
        if isinstance(self._active_handler, StrokeHandler):
            self._active_handler.reset(self)

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
        # Placement mode is now active; register PointHandler directly without
        # calling attach() (which would re-enter the mode switch we just did).
        self._active_handler = PointHandler()

    def onUndo(self):
        log.debug('[Widget] Undo pressed — history depth %d', len(self._history))

        # Flush any in-flight stroke so it lands in history before we pop.
        if isinstance(self._active_handler, StrokeHandler):
            self._active_handler.flush(self)

        if not self._history:
            return

        entry = self._history.pop()
        action_type = entry[0]

        # --- Brush / Erase / Expand → reverse the stored delta ---
        if action_type in ('brush', 'erase', 'expand'):
            change = entry[1]
            self.logic.reverse_change(self, change)

        # --- Point → remove the control point, then reverse its mask delta ---
        elif action_type == 'point':
            _, change, node, cp_id = entry

            _, negNode = self.logic.getPromptNodes(self._parameterNode)
            is_negative = (node is negNode)

            # Pause so _onPointRemoved does not fire a render mid-undo.
            self.ctrl.pause()
            try:
                idx = node.GetControlPointIndexByID(cp_id)
                if idx >= 0:
                    node.RemoveNthControlPoint(idx)

                # If the node is now empty, recreate it to reset the ID
                # counter — otherwise the next placement cursor shows "N+1"
                # instead of "1".
                remaining = node.GetNumberOfControlPoints()
                if remaining == 0:
                    new_node = self.logic.recreate_prompt_node(
                        self._parameterNode, is_negative
                    )
                    self._observeMarkupsNodes()
                    interactionNode = slicer.app.applicationLogic().GetInteractionNode()
                    interactionNode.SwitchToViewTransformMode()
                    if is_negative:
                        self.ui.negativePrompts.setCurrentNode(new_node)
                    else:
                        self.ui.positivePrompts.setCurrentNode(new_node)
                        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
                        selectionNode.SetActivePlaceNodeID(new_node.GetID())
                        selectionNode.SetActivePlaceNodeClassName(new_node.GetClassName())
                        interactionNode.SwitchToPersistentPlaceMode()
            finally:
                self.ctrl.resume()

            self.logic.reverse_change(self, change)

    # -------------------------
    # SPX Boundary Overlay  (Q)
    # -------------------------
    def _get_composite_node(self, view_name):
        """Return the slice composite node for *view_name*, or None if unavailable."""
        sw = slicer.app.layoutManager().sliceWidget(view_name)
        return sw.sliceLogic().GetSliceCompositeNode() if sw else None

    def _hideSPXBoundary(self):
        """Remove the SPX boundary label from the slice view it was shown on."""
        if not self._spx_boundary_visible:
            return
        if self._spx_boundary_view:
            composite = self._get_composite_node(self._spx_boundary_view)
            if composite:
                composite.SetLabelVolumeID("")
        self._spx_boundary_visible = False
        self._spx_boundary_view    = None
        self.ui.showSPXBoundaryCheckBox.blockSignals(True)
        self.ui.showSPXBoundaryCheckBox.setChecked(False)
        self.ui.showSPXBoundaryCheckBox.blockSignals(False)

    def onToggleSPXBoundary(self, _checked=None):
        """Q key / checkbox handler — show or hide the SPX superpixel boundary overlay.

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
        try:
            boundary_2d, axis, sliceIndex = self.logic.compute_spx_boundary(self)
        except ValueError as exc:
            slicer.util.warningDisplay(f"SPX boundary: {exc}")
            # Button may have been auto-toggled to checked by a click — revert it.
            self.ui.showSPXBoundaryCheckBox.blockSignals(True)
            self.ui.showSPXBoundaryCheckBox.setChecked(False)
            self.ui.showSPXBoundaryCheckBox.blockSignals(False)
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
            if sourceArray is None:
                slicer.util.warningDisplay("Cannot read volume data — SPX boundary not shown.")
                return
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
        composite = self._get_composite_node(viewName)
        if not composite:
            slicer.util.warningDisplay("Cannot access slice view — SPX boundary not shown.")
            return
        composite.SetLabelVolumeID(self._spx_boundary_node.GetID())
        composite.SetLabelOpacity(0.8)

        self._spx_boundary_visible = True
        self._spx_boundary_view    = viewName
        self.ui.showSPXBoundaryCheckBox.blockSignals(True)
        self.ui.showSPXBoundaryCheckBox.setChecked(True)
        self.ui.showSPXBoundaryCheckBox.blockSignals(False)

    def onAddSegment(self, *args):
        # Cache and detach the active stroke handler before creating the segment.
        # clearPrompts() (triggered via onSegmentChanged → setCurrentSegmentID)
        # sets _active_handler = PointHandler() directly without going through
        # the detach lifecycle.  We flush + detach first so the handler is
        # cleanly removed, then restore it in the finally block after creation.
        prior_handler_class = (
            type(self._active_handler)
            if isinstance(self._active_handler, StrokeHandler)
            else None
        )
        if self._active_handler is not None:
            self._active_handler.detach(self)

        segment_created = False
        self.ctrl.pause()
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
            segmentID = segmentation.AddEmptySegment(next_segment_name(existing))

            self.ui.segmentSelector.setCurrentSegmentID(segmentID)
            segment_created = True

        finally:
            self.ctrl.resume()
            if prior_handler_class is not None:
                prior_handler_class().attach(self)

        if segment_created:
            self._history.clear()

    def onRemoveSegment(self, *args):
        self.ctrl.pause()
        try:
            segNode = self.ui.segmentationNodeSelector.currentNode()
            segmentID = self.ui.segmentSelector.currentSegmentID()

            if not segNode or not segmentID:
                slicer.util.warningDisplay("No segment selected.")
                return

            segNode.GetSegmentation().RemoveSegment(segmentID)

        finally:
            self.ctrl.resume()
        
        if segNode and segmentID:
            self._history.clear()
            if isinstance(self._active_handler, StrokeHandler):
                self._active_handler.reset(self)

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
            segNode.CreateClosedSurfaceRepresentation()

            self.logic.setVolumeAndSegmentation(self._parameterNode, volumeNode, segNode)
            self._parameterNode.Modified()

        return segNode

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

        # Discover all TestCase subclasses in the module automatically so that
        # adding a new test class to SegmentHumanBodyTest.py is sufficient to
        # include it in the "Reload and Test" run.
        suite = unittest.TestLoader().loadTestsFromModule(ext)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise Exception(
                f'{len(result.failures) + len(result.errors)} test(s) failed — '
                'see the Python console for details'
            )
