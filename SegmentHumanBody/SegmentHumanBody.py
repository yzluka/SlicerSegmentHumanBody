import qt, vtk, slicer
import logging
import numpy as np
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin

from core.modelFamilies import BaseModelFamily, SAMFamily, SPXModelFamily, AutoModelFamily, FAMILY_REGISTRY
from core.utils import (
    call_if_exists,
    write_slice_to_volume,
    next_segment_name,
    parse_user_parameters,
)
from core.modelRegistry import ModelRegistry
from core._renderer import _SliceViewMouseFilter
from core._logic import SegmentHumanBodyLogic
from core._controller import RenderController

log = logging.getLogger(__name__)

#
# Module
#

class SegmentHumanBody(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = 'SegmentHumanBody (Optimized)'
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
        self.ctrl = RenderController(self)   # central state machine for all render flags
        self._parameterNode = None
        self.modelFamily = None
        self.currentViewName = None  # default

        # Single unified action history for Ctrl+Z.  Each entry is a list:
        #   ['brush',  change]                  — Paint/Erase stroke
        #   ['expand', change]                  — expand operation
        #   ['point',  change, node, cp_id]     — confirmed prompt control point
        # ``change`` is a MaskChange (or None when the action produced no net
        # mask change).  Undo pops the last entry and calls reverse_delta if
        # a change is present; for 'point' entries the control point is also
        # removed from the markup node.  Lists (not tuples) are used so that
        # the 'point' path can fill ``change`` after the async render completes.
        self._history = []
        self._undo_shortcut = None

        # Before-state captured at brush-stroke start (no pre_stroke wrapper).
        # Holds (axis, idx, slice_copy) while a stroke is in progress; None otherwise.
        self._stroke_before = None

        # SPX boundary overlay
        self._spx_boundary_node    = None   # vtkMRMLLabelMapVolumeNode
        self._spx_boundary_visible = False
        self._spx_boundary_view    = None   # view name the label is currently set on
        self._spx_boundary_shortcut = None
        self._expand_shortcut    = None

        self._toolValidatorTimer = qt.QTimer()
        self._toolValidatorTimer.timeout.connect(self._enforceToolConsistency)
        self._brushMouseFilter = None  # _SliceViewMouseFilter installed on slicer.app

    # -------------------------
    # Setup / Cleanup
    # -------------------------
    def cleanup(self):
        """Called by Slicer when the module is unloaded.  Remove the app-level
        Qt event filter so it does not reference a dead widget."""
        if self._brushMouseFilter:
            slicer.app.removeEventFilter(self._brushMouseFilter)
            self._brushMouseFilter = None
        if self._toolValidatorTimer.isActive():
            self._toolValidatorTimer.stop()
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
        self._toolValidatorTimer.start(100)

        # Install a Qt application-level event filter to detect brush stroke
        # boundaries.  Qt events fire before VTK processes them, so this works
        # even when the Segment Editor Paint/Erase effect absorbs VTK events.
        self._brushMouseFilter = _SliceViewMouseFilter(self)
        slicer.app.installEventFilter(self._brushMouseFilter)

        log.debug('[Setup complete]')

    def _onBrushStrokeStart(self):
        if not (self.ui.brushToolButton.isChecked() or self.ui.eraseToolButton.isChecked()):
            return

        if not self.ctrl.brush_in_progress:
            # If a prior stroke's commit timer is still pending (extremely rare
            # race between rapid clicks), flush it now so history stays ordered.
            if self._stroke_before is not None:
                self._commitPendingStroke()

            axis, idx, before = self.logic.capture_current_slice(self)
            if before is not None:
                self._stroke_before = (axis, idx, before)
            self.ctrl.brush_in_progress = True
            log.debug('[Widget] brush stroke start — history depth %d', len(self._history))
            self.logic.reset_render_state()

    def _onBrushStrokeEnd(self):
        if not self.ctrl.brush_in_progress:
            return
        self.ctrl.brush_in_progress = False

        # Our Qt event filter fires BEFORE Qt delivers the event to the VTK
        # render window, so the Paint effect has not called apply() yet.
        # A 0-ms timer fires AFTER the current event is fully dispatched
        # (including VTK's synchronous processing of the mouse release, which
        # triggers the Paint effect's natural apply()), so the labelmap is
        # committed by the time _commitPendingStroke reads it.
        if self._stroke_before is not None:
            qt.QTimer.singleShot(0, self._commitPendingStroke)

    def _commitPendingStroke(self):
        """Read the post-stroke Slicer state, compute the delta, record in history.

        Called from a 0-ms QTimer (scheduled in _onBrushStrokeEnd) so that
        VTK's Paint-effect apply() has already committed the stroke before we
        read the labelmap.  Also called synchronously when the E shortcut or
        Ctrl+Z needs an immediate commit.
        """
        if self._stroke_before is None:
            return
        axis, idx, before = self._stroke_before
        self._stroke_before = None
        change = self.logic.commit_stroke(self, axis, idx, before)
        if change is not None:
            self._history.append(['brush', change])
        log.debug('[Widget] stroke committed — change=%s  history=%d',
                  change is not None, len(self._history))

    def _onExpand(self):
        """Run expand and record the result in history.

        Shared by the E shortcut and the Expand button.  Returns immediately
        when pre-conditions fail (handled inside ``on_expand``).
        """
        change = self.logic.on_expand(self)
        if change is not None:
            self._history.append(['expand', change])

    def _onExpandShortcut(self):
        """Handle the E hotkey: exit brush → expand → re-enter brush.

        Three cases are handled:

        1. Mouse still held (brush_in_progress): VTK will never receive a
           mouse-release, so we call apply() manually to flush the buffered
           stroke, then commit synchronously.

        2. Pending 0-ms timer commit (mouse just released before the timer
           fired): flush _commitPendingStroke synchronously so the stroke
           is in history and in the tracker before expand reads the mask.

        3. No pending stroke: brush is active but idle — expand runs directly.
        """
        brush_active = self.ui.brushToolButton.isChecked()
        erase_active = self.ui.eraseToolButton.isChecked()

        if brush_active or erase_active:
            prior_tool = "brush" if brush_active else "erase"

            if self.ctrl.brush_in_progress:
                # Case 1: VTK will never get the mouse release — flush manually.
                editor = self._segEditor()
                if editor:
                    effect = editor.activeEffect()
                    if effect:
                        try:
                            effect.self().apply()
                        except Exception:
                            pass
                self.ctrl.brush_in_progress = False

            # Cases 1 + 2: commit any pending before-state synchronously.
            if self._stroke_before is not None:
                self._commitPendingStroke()

            self._setTool(None)
            self._onExpand()
            self._setTool(prior_tool)
        else:
            self._onExpand()

    def _enforceToolConsistency(self):
        editor = self._segEditor()
        if not editor:
            return

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()

        posNode, negNode = self.logic.getPromptNodes(self._parameterNode)

        activeEffect = editor.activeEffect().name if editor.activeEffect() else None
        activePlaceNodeID = selectionNode.GetActivePlaceNodeID()
        mode = interactionNode.GetCurrentInteractionMode()

        # --- If placing points → brush must be OFF ---
        if mode == interactionNode.Place and activePlaceNodeID in (
            posNode.GetID() if posNode else None,
            negNode.GetID() if negNode else None,
        ):
            if self.ui.brushToolButton.isChecked() or self.ui.eraseToolButton.isChecked():
                self._forceDeactivateBrush()

        # --- If brush is active → UI must match ---
        elif activeEffect in ("Paint", "Erase"):
            self._syncBrushUI(activeEffect)

        # --- If no tool active → UI must be OFF ---
        elif activeEffect is None:
            if self.ui.brushToolButton.isChecked() or self.ui.eraseToolButton.isChecked():
                self._forceDeactivateBrush()
    
    def _forceDeactivateBrush(self):
        self._setTool(None)

    def _syncBrushUI(self, effectName):
        self.ui.brushToolButton.blockSignals(True)
        self.ui.eraseToolButton.blockSignals(True)

        self.ui.brushToolButton.setChecked(effectName == "Paint")
        self.ui.eraseToolButton.setChecked(effectName == "Erase")

        self.ui.brushToolButton.blockSignals(False)
        self.ui.eraseToolButton.blockSignals(False)

    def _setTool(self, tool: str):
        editor = self._segEditor()

        # --- turn everything OFF (NO SIGNALS) ---
        self.ui.brushToolButton.blockSignals(True)
        self.ui.eraseToolButton.blockSignals(True)

        self.ui.brushToolButton.setChecked(False)
        self.ui.eraseToolButton.setChecked(False)

        self.ui.brushToolButton.blockSignals(False)
        self.ui.eraseToolButton.blockSignals(False)

        if editor:
            editor.setActiveEffectByName("")

        # --- apply selected tool ---
        if tool == "brush":
            self.ui.brushToolButton.blockSignals(True)
            self.ui.brushToolButton.setChecked(True)
            self.ui.brushToolButton.blockSignals(False)

            self._pausePromptPlacement()

            if editor:
                self._activateBrushEffect("Paint", self.ui.brushToolButton)

        elif tool == "erase":
            self.ui.eraseToolButton.blockSignals(True)
            self.ui.eraseToolButton.setChecked(True)
            self.ui.eraseToolButton.blockSignals(False)

            self._pausePromptPlacement()

            if editor:
                self._activateBrushEffect("Erase", self.ui.eraseToolButton)


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
        # Observe slice node changes so that scrolling re-draws the prompt result.
        self._connectSliceObservers()

    def _connectSliceObservers(self):
        """Add VTK observers on all three slice nodes.

        When the active slice view scrolls to a new slice the prompt result
        must be redrawn.  Observing ``ModifiedEvent`` on every slice node and
        checking whether the *active* view actually changed slice triggers the
        right render without polling.
        """
        lm = slicer.app.layoutManager()
        for viewName in ("Red", "Green", "Yellow"):
            sw = lm.sliceWidget(viewName)
            if sw:
                self.addObserver(sw.mrmlSliceNode(),
                                 vtk.vtkCommand.ModifiedEvent,
                                 self._onSliceNodeModified)

    def _onSliceNodeModified(self, caller=None, event=None):
        """Re-render when the active slice view scrolls to a new slice.

        Fires for every ``vtkMRMLSliceNode.ModifiedEvent`` (pan, zoom, scroll,
        window/level adjust from the display node, etc.).  We only act when:
          - not paused
          - the modified node belongs to the currently active view
          - the slice index actually differs from the last-rendered index

        Intentionally does NOT write to ``_last_render_key`` — that field is
        owned by ``onRender`` alone.  Writing None here before ``_triggerRender``
        would break the guard: if the triggered render is then dropped (because
        ``_isRendering`` is True) the key stays None, causing every subsequent
        slice-node event to bypass the guard until a render finally succeeds.
        Instead we rely on the fact that ``onRender`` already compares the full
        render key (which includes sliceIndex) and skips when nothing changed.
        """
        if self.ctrl.is_paused:
            return
        if not self.modelFamily or not self.modelFamily.model:
            return
        volumeNode = self.ui.sourceVolumeSelector.currentNode()
        if not volumeNode:
            return
        # Only react to the slice view we are rendering for.
        sw = slicer.app.layoutManager().sliceWidget(self.currentViewName)
        if not sw or sw.mrmlSliceNode() is not caller:
            return
        try:
            axis, sliceIndex = self.logic.getAxisAndSlice(self, volumeNode)
        except Exception:
            return
        # Skip if the slice hasn't moved since the last render.
        # When last_key is None (after a reset) we always proceed so the
        # committed state is refreshed on the first scroll after a reset.
        last_key = self.logic._last_render_key
        if (last_key is not None
                and last_key[2] == axis and last_key[3] == sliceIndex):
            return
        self._triggerRender()

    def _preloadSegmentEditor(self):
        """Silently initialize the Segment Editor module widget if not done yet.

        The widget is only created the first time the module is shown.  We trigger
        that creation by switching to it and immediately back — both happen in the
        same call stack before the event loop repaints, so the user sees no flash.
        Also hooks activeEffectChanged so a right-click exit of the paint/erase
        effect is reflected in our Brush / Erase toggle buttons.
        """
        if slicer.modules.segmenteditor.widgetRepresentation() is None:
            slicer.util.selectModule('SegmentEditor')
            slicer.util.selectModule(self.moduleName)

        editor = self._segEditor()
        if editor:
            editor.connect('activeEffectChanged()', self._onEditorEffectChanged)

    def _onEditorEffectChanged(self):
        if self.ctrl.activating_brush:
            return

        editor = self._segEditor()
        if not editor:
            return

        effect = editor.activeEffect()
        active_name = effect.name if effect else None

        # --- Sync UI state ---
        if active_name not in ("Paint", "Erase"):
            self._setTool(None)


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
        ui.sliceViewDropdown.connect('currentTextChanged(QString)', self.onSliceViewChanged)
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

    def _onPointConfirmed(self, caller=None, event=None):
        """PointPositionDefinedEvent — a placement was just confirmed by the user.

        At the moment this fires, the just-confirmed point is always the last
        one (index n-1): Slicer raises PointPositionDefinedEvent before creating
        the next placement cursor, so the new cursor's PointAddedEvent always
        follows this one.

        Push the confirmed point to the undo stack, then trigger a render so
        the result is computed without waiting for the next event-loop tick.
        """
        if caller is None:
            return

        n = caller.GetNumberOfControlPoints()
        if n > 0:
            cp_id = caller.GetNthControlPointID(n - 1)
            # Create the history entry now; fill the MaskChange after the render.
            entry = ['point', None, caller, cp_id]
            self._history.append(entry)
            # Clear the logic's last-change staging area before triggering the
            # render so we can detect whether the render produced a new change.
            self.logic._last_change = None
            # Render fires first; capture fires second (FIFO 0-ms timers).
            qt.QTimer.singleShot(0, self._triggerRender)
            qt.QTimer.singleShot(0, lambda: self._capturePointChange(entry))
        else:
            qt.QTimer.singleShot(0, self._triggerRender)

    def _onPointRemoved(self, caller=None, event=None):
        """PointRemovedEvent — a prompt point was deleted.

        The session's base_mask is unchanged; the next render recomputes from
        base + remaining points and commits the correct result directly.

        Guard: skip when paused (e.g. during clearPrompts or onUndo — the undo
        path removes the markup point itself, pausing to suppress this handler,
        then resets render state and triggers its own render explicitly).
        """
        if self.ctrl.is_paused:
            return
        qt.QTimer.singleShot(0, self._triggerRender)

    def _capturePointChange(self, entry):
        """Fill the MaskChange field of a 'point' history entry after the render fires.

        Scheduled as a 0-ms QTimer *after* the render timer queued in
        _onPointConfirmed, so ``logic._last_change`` is already populated when
        this runs (FIFO timer ordering within the same event-loop tick).
        """
        entry[1] = self.logic._last_change
        log.debug('[Widget] point change captured: %s', entry[1] is not None)

    def _triggerRender(self):
        """Request a single immediate render via the controller.

        Used when confirmed point placements, point removals, undos, and slice
        scrolls need the display updated without waiting for the 100ms timer.
        Re-entrancy and pending-render bookkeeping are handled by ctrl.request_render.
        """
        self.ctrl.request_render()

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

    def _pausePromptPlacement(self):
        """Switch the interaction node to view-transform mode.

        Called when brush or erase is activated so that left-clicks in slice
        views go to the paint effect rather than placing markup points.
        """
        slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()


    def _activateBrushEffect(self, effect_name: str, button):
        """Shared setup for Paint / Erase: guard nodes, sync Segment Editor, activate."""
        editor = self._segEditor()
        if editor is None:
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
            return

        volNode, segNode = self.logic.getVolumeAndSegmentation(self._parameterNode)

        if effect_name == "Paint" and segNode and \
                segNode.GetSegmentation().GetNumberOfSegments() == 0:
            self.onAddSegment()

        if not volNode or not segNode:
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
            return

        self.ctrl.activating_brush = True
        try:
            editor.setSegmentationNode(segNode)
            self.ctrl.brush_in_progress = False
            editor.setSourceVolumeNode(volNode)
            editor.setUndoEnabled(True)
            editor.setMaximumNumberOfUndoStates(50)
            segID = self.ui.segmentSelector.currentSegmentID()
            if segID:
                editor.setCurrentSegmentID(segID)
            editor.setActiveEffectByName(effect_name)
            self._applyBrushParams()
        finally:
            self.ctrl.activating_brush = False

        # Segment Editor node setup can reset the slice composite label layer.
        # Re-apply the SPX boundary overlay if it was active.
        if self._spx_boundary_visible and self._spx_boundary_node:
            composite = self._get_composite_node(self._spx_boundary_view)
            if composite:
                composite.SetLabelVolumeID(self._spx_boundary_node.GetID())
                composite.SetLabelOpacity(0.8)


    def onBrushToggled(self, checked: bool):
        if checked:
            self._setTool("brush")
        else:
            self._setTool(None)

    def onEraseToggled(self, checked: bool):
        if checked:
            self._setTool("erase")
        else:
            self._setTool(None)

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

        posNode, negNode = self.logic.getPromptNodes(self._parameterNode)
        volumeNode, segNode = self.logic.getVolumeAndSegmentation(self._parameterNode)

        self.ui.positivePrompts.setCurrentNode(posNode)
        self.ui.negativePrompts.setCurrentNode(negNode)

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

        # The working mask and render key are segment-specific — switching
        # segments must invalidate both so the next render reads fresh data.
        self.logic.reset_render_state()

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
            self._stroke_before = None

    def clearPrompts(self):
        self._history.clear()
        self._stroke_before = None

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
        log.debug('[Widget] Undo pressed — history depth %d', len(self._history))

        # If a 0-ms commit timer is pending (mouse just released, timer hasn't
        # fired) flush it now so the stroke lands in history before we pop.
        if self._stroke_before is not None:
            self._commitPendingStroke()

        if not self._history:
            return

        entry = self._history.pop()
        action_type = entry[0]

        # --- Brush / Expand → reverse the stored delta ---
        if action_type in ('brush', 'expand'):
            change = entry[1]
            self.logic.reverse_change(self, change)

        # --- Point → remove the control point, then reverse its mask delta ---
        elif action_type == 'point':
            _, change, node, cp_id = entry

            # Pause so _onPointRemoved does not fire a render mid-undo.
            self.ctrl.pause()
            try:
                idx = node.GetControlPointIndexByID(cp_id)
                if idx >= 0:
                    node.RemoveNthControlPoint(idx)
            finally:
                self.ctrl.resume()

            self.logic.reverse_change(self, change)

        # --- Reset session / render key so the next render starts fresh ---
        self.logic.reset_render_state()

        qt.QTimer.singleShot(0, self._triggerRender)
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

        finally:
            self.ctrl.resume()

        self._history.clear()
        self._stroke_before = None

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
            self._stroke_before = None

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
