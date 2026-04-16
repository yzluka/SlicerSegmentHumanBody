import qt, vtk, slicer
import logging
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
    select_spx_labels,
    collect_confirmed_points,
    collect_preview_points,
    VIEW_TO_AXIS,
    AXIS_TO_IJK_COMPONENT,
    ras_to_ijk_2d,
    next_segment_name,
    parse_user_parameters,
)
from core.modelRegistry import ModelRegistry


log = logging.getLogger(__name__)

POS_NODE = 'positivePromptPointsNode'
NEG_NODE = 'negativePromptPointsNode'
INPUT_VOLUME = "InputVolume"
SEGMENTATION = "Segmentation"

# Config shared by ensurePromptNodesExist and recreatePromptNodes.
# Each entry: ref_name → (RGB color, display label)
_PROMPT_NODE_CONFIGS = {
    POS_NODE: ([0, 1, 0], 'positive'),
    NEG_NODE: ([1, 0, 0], 'negative'),
}


def _node_records(node):
    """Return [(status, cp_id, position), …] for all control points in *node*.

    Produces the format expected by collect_confirmed_points /
    collect_preview_points.  Returns an empty list when *node* is None.
    This avoids duplicating the same comprehension in onRender and on_expand.
    """
    if not node:
        return []
    return [
        (node.GetNthControlPointPositionStatus(i),
         node.GetNthControlPointID(i),
         node.GetNthControlPointPosition(i))
        for i in range(node.GetNumberOfControlPoints())
    ]

#
# Module
#

class SegmentHumanBody(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = 'SegmentHumanBody (Optimized)'
        self.parent.categories = ['Segmentation']


class _SliceViewMouseFilter(qt.QObject):
    """Qt application-level event filter for detecting brush stroke boundaries.

    Installed on slicer.app (QApplication) so it fires for every Qt mouse
    event in the entire application — before any VTK interactor observer,
    and before the Segment Editor Paint/Erase effect can set the VTK abort
    flag.  This is the only reliable hook point: VTK-level LeftButtonPressEvent
    observers are silently skipped when an active Segment Editor effect has
    already consumed the event at higher priority.

    The callback guard (brushToolButton/eraseToolButton.isChecked) is checked
    inside _onBrushStrokeStart, so this filter is a no-op during any non-brush
    interaction.  The button is NOT yet toggled when its own MouseButtonPress
    fires (Qt buttons toggle on release), so clicking the brush button itself
    never triggers a spurious undo entry.
    """
    def __init__(self, slicerWidget):
        super().__init__()
        self._slicerWidget = slicerWidget

    def eventFilter(self, obj, event):
        t = event.type()
        if t == qt.QEvent.MouseButtonPress and event.button() == qt.Qt.LeftButton:
            self._slicerWidget._onBrushStrokeStart()
        elif t == qt.QEvent.MouseButtonRelease and event.button() == qt.Qt.LeftButton:
            self._slicerWidget._onBrushStrokeEnd()
        return False  # never consume — always propagate to the target widget

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
            return

        self.widget._isRendering = True
        try:
            self.widget.logic.onRender(self.widget.modelFamily, self.widget)

        except Exception as e:
            log.error(f"[Renderer Error] {e}")
            self.stop()
            slicer.util.errorDisplay(f"Rendering stopped:\n{str(e)}")
            self.widget.setInteractiveState(False)

        finally:
            self.widget._isRendering = False
#
# Widget
#

class SegmentHumanBodyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        super().__init__(parent)
        VTKObservationMixin.__init__(self)

        self._preview_mask = None
        
        self.logic = SegmentHumanBodyLogic()
        self._parameterNode = None
        self.modelFamily = None
        self._isInteractive = False   # True while the render loop is running
        self.currentViewName = None  # default
        self._isRendering = False
        self._pauseRender = False
        self._creating_segment = False  # suppresses onSegmentChanged during programmatic segment creation

        # Unified action stack for Ctrl+Z.  Each entry is a tuple whose first
        # element is the action type:
        #   ('brush', snapshot)           — a Paint/Erase stroke
        #   ('point', node, cp_id, snapshot) — a confirmed prompt control point
        #   ('expand', snapshot)          — an expand operation
        # snapshot = (segNodeID, segmentID, axis, sliceIndex, ndarray)
        # Actions are pushed in order; Ctrl+Z pops and reverses the most recent.
        self._undo_stack = []
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
        self._expand_shortcut    = None

        # Guard flag: True while _activateBrushEffect is running its setup
        # sequence (setSegmentationNode / setSourceVolumeNode / setActiveEffectByName).
        # _onEditorEffectChanged returns immediately when this is set so that
        # spurious activeEffectChanged(None) signals emitted during Segment Editor
        # node wiring do not uncheck the brush button or reset _prompt_base_mask.
        self._activatingBrushEffect = False
        self._brushInProgress = False
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

        # Q — toggle SPX superpixel boundary overlay
        self._spx_boundary_shortcut = qt.QShortcut(qt.QKeySequence("Q"), uiWidget)
        self._spx_boundary_shortcut.connect('activated()', self.onToggleSPXBoundary)

        # E — expand selected label
        self._expand_shortcut = qt.QShortcut(qt.QKeySequence("E"), uiWidget)
        self._expand_shortcut.connect(
            'activated()', lambda: self.bind('on_expand')())

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

        if not self._brushInProgress:
            # Capture the pre-stroke slice state so onUndo can restore it
            # directly, without delegating to the Segment Editor's separate
            # undo stack.  This keeps all undo entries in the unified stack.
            snapshot = self._captureSegmentationState()
            self._undo_stack.append(('brush', snapshot))
            self._brushInProgress = True
            log.debug(f"Brush stroke start — undo stack depth {len(self._undo_stack)}")

            self.logic.reset_render_state()
            self.logic.reset_prompt_base_mask()

    def _onBrushStrokeEnd(self):
        if self._brushInProgress:
            self._brushInProgress = False
            log.debug("Brush stroke end")

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
        editor = self._segEditor()

        self.ui.brushToolButton.blockSignals(True)
        self.ui.eraseToolButton.blockSignals(True)

        self.ui.brushToolButton.setChecked(False)
        self.ui.eraseToolButton.setChecked(False)

        self.ui.brushToolButton.blockSignals(False)
        self.ui.eraseToolButton.blockSignals(False)

        if editor:
            editor.setActiveEffectByName("")

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
        if self._activatingBrushEffect:
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

        model_button_connections = [
            ('assignLabel2D', 'on_assign_2d'),
            ('assignLabel3D', 'on_assign_3d'),
            ('expandSelectedLabelButton', 'on_expand'),
            ('runAutomaticSegmentation', 'on_automatic_segmentation'),
        ]

        for ui_name, method_name in model_button_connections:
            getattr(ui, ui_name).connect(
                'clicked(bool)',
                self.bind(method_name, target="logic")
            )

        widget_button_connections = [
            ('goToMarkupsButton', self.on_go_to_markups),
            ('confirmModelSelection', self.onConfirmClicked),
            ('addSegmentButton', self.onAddSegment),
            ('removeSegmentButton', self.onRemoveSegment),
            ('applyWindowLevelButton', self.onApplyWindowLevel),
        ]

        for ui_name, method in widget_button_connections:
            getattr(ui, ui_name).connect('clicked(bool)', method)

        # Checkboxes — use toggled(bool) so the checked state drives the action.
        ui.interactiveModeCheckBox.connect('toggled(bool)', self.onInteractiveModeToggled)
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
                    slicer.vtkMRMLMarkupsNode.PointAddedEvent,
                    self._onPointAdded
                )
                self.addObserver(
                    node,
                    slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                    self._onPointConfirmed
                )

    def _onPointAdded(self, caller=None, event=None):
        """Track newly added prompt points for the hover-preview system.

        If the new point is in PositionPreview state (Slicer's placement cursor
        before the user clicks on a slice), mark it as unconfirmed so the render
        loop ignores it until PointPositionDefinedEvent fires.

        Confirmed points are NOT pushed to _undo_stack here.  They are pushed
        in _onPointConfirmed so that the hover cursor (which also fires
        PointAddedEvent in PositionPreview state and then is overwritten by the
        real placement) never pollutes the undo stack.
        """
        if caller is None:
            return
        n = caller.GetNumberOfControlPoints()
        if n > 0:
            cp_id = caller.GetNthControlPointID(n - 1)
            status = caller.GetNthControlPointPositionStatus(n - 1)
            if status == slicer.vtkMRMLMarkupsNode.PositionPreview:
                self._preview_cp_ids.add(cp_id)

    def _onPointConfirmed(self, caller=None, event=None):
        """PointPositionDefinedEvent — a placement was just confirmed by the user.

        Push confirmed preview IDs to _undo_stack BEFORE evicting them from
        _preview_cp_ids.  This ensures each confirmed point gets exactly one
        entry in the undo stack — hover cursors that were already evicted from
        _preview_cp_ids (and thus tracked in _onPointAdded's else branch) are
        not double-counted because they are no longer in _preview_cp_ids.

        Any new placement cursor created afterward (multi-place mode) will be
        re-added to _preview_cp_ids when its own PointAddedEvent fires, which
        always follows PointPositionDefinedEvent in Slicer's event ordering.

        After updating preview state, trigger a single immediate render so the
        newly confirmed point is processed without waiting for the next timer tick.
        """
        snapshot = self._captureSegmentationState()
        
        if self.logic._prompt_base_mask is None:
            self.logic.reset_prompt_base_mask()

        if caller is None:
            return
        # Push IDs that were tracked as preview cursors and just got confirmed.
        for cp_id in list(self._preview_cp_ids):
            if caller.GetControlPointIndexByID(cp_id) >= 0:
                self._undo_stack.append(('point', caller, cp_id, snapshot))

        # Evict all IDs belonging to this node from the preview set.
        for i in range(caller.GetNumberOfControlPoints()):
            self._preview_cp_ids.discard(caller.GetNthControlPointID(i))
        # Defer the render to the next event-loop iteration so we are not
        # calling into MRML scene modifications from within a VTK observer
        # callback (which would cause nested event processing and a crash).
        qt.QTimer.singleShot(0, self._triggerRender)

    def _triggerRender(self):
        """Fire a single immediate render call.

        Used when confirmed point placements or undos need the display updated
        right away, without relying on the 100ms hover-preview timer.
        Guards against re-entrancy with the same flags as SegmentationRenderer.
        """
        if self._pauseRender or self._isRendering:
            return
        if not self.modelFamily or not self.modelFamily.model:
            return
        self._isRendering = True
        try:
            self.logic.onRender(self.modelFamily, self)
        except Exception as e:
            log.error(f"[TriggerRender Error] {e}")
        finally:
            self._isRendering = False

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

        editor.setSegmentationNode(segNode)
        self._brushInProgress = False
        editor.setSourceVolumeNode(volNode)
        editor.setUndoEnabled(True)
        editor.setMaximumNumberOfUndoStates(50)
        segID = self.ui.segmentSelector.currentSegmentID()
        if segID:
            editor.setCurrentSegmentID(segID)
        editor.setActiveEffectByName(effect_name)
        self._applyBrushParams()

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
        mapping = [
            ('assignLabel2D', 'on_assign_2d'),
            ('assignLabel3D', 'on_assign_3d'),
            ('interactiveModeCheckBox', 'on_enter_interactive'),
            ('expandSelectedLabelButton', 'on_expand'),
            ('showSPXBoundaryCheckBox', 'on_expand'),  # same family as expand
            ('runAutomaticSegmentation', 'on_automatic_segmentation'),
            ('goToMarkupsButton', 'on_go_to_markups'),
            ('samMaskDropdown', 'get_requested_mask'),
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
        self.ui.interactiveModeCheckBox.blockSignals(True)
        self.ui.interactiveModeCheckBox.setChecked(is_running)
        self.ui.interactiveModeCheckBox.blockSignals(False)

    def onInteractiveModeToggled(self, checked: bool):
        """Checkbox handler — enter or stop interactive mode."""
        if checked:
            self.bind('on_enter_interactive')()
        else:
            self.bind('on_stop_interactive')()

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
        if not segmentID or self._creating_segment:
            return

        # Keep the Segment Editor's active tool in sync with the selected segment.
        if self.ui.brushToolButton.isChecked() or self.ui.eraseToolButton.isChecked():
            editor = self._segEditor()
            if editor:
                editor.setCurrentSegmentID(segmentID)
                
                self._brushInProgress = False

        # The working mask and render key are segment-specific — switching
        # segments must invalidate both so the next render reads fresh data.
        self.logic.reset_render_state()

        # clearPrompts resets the prompt stack and preview IDs,
        # recreates fresh prompt nodes, and re-wires the markups widgets.
        self._pauseRender = True
        try:
            self.clearPrompts()
        finally:
            self._pauseRender = False
        segNode = self.ui.segmentSelector.currentNode()
        if segNode and segmentID:
            self._undo_stack.clear()
            
    
    def clearPrompts(self):
        # Clear tracking state FIRST so any PointAddedEvent that fires when
        # the new nodes are wired to the markups widgets (the widget may
        # auto-create a placement cursor on an empty node) is recorded in
        # _preview_cp_ids and excluded from the render loop.
        self._undo_stack.clear()
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
        log.debug(f"Undo pressed — stack depth {len(self._undo_stack)}")
        editor = self._segEditor()

        # --- Ensure any active brush stroke is committed first ---
        if editor and editor.activeEffect():
            try:
                editor.activeEffect().self().apply()
            except Exception:
                pass

        if not self._undo_stack:
            return

        action = self._undo_stack.pop()
        action_type = action[0]

        # --- Brush → restore pre-stroke snapshot (same path as point/expand) ---
        if action_type == 'brush':
            _, snapshot = action
            if snapshot:
                self._restoreSegmentation(snapshot)

        # --- Point ---
        elif action_type == 'point':
            _, node, cp_id, snapshot = action

            idx = node.GetControlPointIndexByID(cp_id)
            if idx >= 0:
                node.RemoveNthControlPoint(idx)

            self._restoreSegmentation(snapshot)

        # --- Expand ---
        elif action_type == 'expand':
            _, snapshot = action
            self._restoreSegmentation(snapshot)

        # --- Reset model/render state ---
        self.logic.reset_render_state()
        self.logic.reset_prompt_base_mask()
        self.logic._last_render_key = None
        self.logic._preview_mask = None

        qt.QTimer.singleShot(0, self._triggerRender)
    # -------------------------
    # SPX Boundary Overlay  (Q)
    # -------------------------
    def _restoreSegmentation(self, snapshot):
        """Restore a 2-D slice from *snapshot* and sync the Logic working mask.

        The working mask is updated through _get_working_mask so the cache is
        always the single source of truth — no parallel sync paths.
        """
        if snapshot is None:
            return

        segNodeID, segmentID, axis, sliceIndex, slice_2d = snapshot

        segNode = slicer.mrmlScene.GetNodeByID(segNodeID)
        volumeNode = self.ui.sourceVolumeSelector.currentNode()

        if not segNode or not volumeNode:
            return

        working = self.logic._get_working_mask(segNode, segmentID, volumeNode)
        write_slice_to_volume(working, slice_2d, axis, sliceIndex)
        slicer.util.updateSegmentBinaryLabelmapFromArray(working, segNode, segmentID, volumeNode)
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
        boundary_2d, axis, sliceIndex, err = self.logic.compute_spx_boundary(self)
        if boundary_2d is None:
            slicer.util.warningDisplay(f"SPX boundary: {err}")
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
        composite.SetLabelVolumeID(self._spx_boundary_node.GetID())
        composite.SetLabelOpacity(0.8)

        self._spx_boundary_visible = True
        self._spx_boundary_view    = viewName
        self.ui.showSPXBoundaryCheckBox.blockSignals(True)
        self.ui.showSPXBoundaryCheckBox.setChecked(True)
        self.ui.showSPXBoundaryCheckBox.blockSignals(False)

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
            segmentID = segmentation.AddEmptySegment(next_segment_name(existing))

            self.ui.segmentSelector.setCurrentSegmentID(segmentID)

        finally:
            self._pauseRender = False
        
        self._undo_stack.clear()
    
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
        
        if segNode and segmentID:
            self._undo_stack.clear()
    
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
    def _captureSegmentationState(self):
        segNode = self.ui.segmentSelector.currentNode()
        segmentID = self.ui.segmentSelector.currentSegmentID()
        volumeNode = self.ui.sourceVolumeSelector.currentNode()

        if not segNode or not segmentID or not volumeNode:
            return None

        axis, sliceIndex = self.logic.getAxisAndSlice(self, volumeNode)

        mask3d = slicer.util.arrayFromSegmentBinaryLabelmap(
            segNode, segmentID, volumeNode
        )
        sliceMask = get_slice_from_volume(mask3d, axis, sliceIndex)

        return (
            segNode.GetID(),
            segmentID,
            axis,
            sliceIndex,
            sliceMask.copy()
        )

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


        # --- Prompt base mask ---
        # Full 3D snapshot of the segment taken when the FIRST confirmed prompt
        # point is rendered.  Each render computes:
        #   result = (base_slice | pos_region) & ~neg_region
        # so pos points add to pre-existing painted data and neg points erase
        # from it.  Removing all prompts restores the base state.
        # Reset by reset_render_state() (segment change, model confirm, etc.).
        self._prompt_base_mask = None

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
        self._prompt_base_mask = None
        self._preview_mask = None
        # W/L is intentionally NOT reset here — it is a user preference that
        # should persist across segment/volume changes until explicitly changed.

    def _get_working_mask(self, segNode, segmentID, volumeNode):
        """Return the cached 3-D working mask, reading from Slicer on a miss.

        The mask is keyed by (segNodeID, segmentID).  On a cache hit the same
        numpy array is returned so callers can update it in-place.
        """
        segment_key = (segNode.GetID(), segmentID)
        if self._working_mask is None or self._working_mask_segment != segment_key:
            raw = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentID, volumeNode)
            self._working_mask = raw.copy()
            self._working_mask_segment = segment_key
        return self._working_mask

    def reset_prompt_base_mask(self):
        """Invalidate the prompt base mask and render key.

        Call this when the segment's painted content changes outside of the
        prompt render loop (e.g. after a brush stroke is committed) so that
        the next prompt render re-snapshots the up-to-date segment state as
        the additive/subtractive base.
        """
        self._prompt_base_mask = None
        self._last_render_key = None

    def set_window_level(self, window, level):
        """Confirm W/L values for model inference.
        Subsequent calls to onRender / on_expand will normalize each slice
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

    def _make_prompt_node(self, parameterNode, ref_name, color, label):
        """Remove any existing node for *ref_name*, create a fresh one, and
        register it on *parameterNode*.  Fresh nodes have a label counter of 0
        so the placement cursor is always labeled 'Positive 1' / 'Negative 1'.
        """
        old = parameterNode.GetNodeReference(ref_name)
        if old:
            slicer.mrmlScene.RemoveNode(old)
        node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', label)
        node.CreateDefaultDisplayNodes()
        dn = node.GetDisplayNode()
        dn.SetSelectedColor(*color)
        dn.SetColor(*color)
        dn.SetActiveColor(*color)
        node.SetHideFromEditors(True)
        parameterNode.SetNodeReferenceID(ref_name, node.GetID())
        return node

    def ensurePromptNodesExist(self, parameterNode):
        """Create prompt nodes for any reference slot that is currently empty."""
        for ref_name, (color, label) in _PROMPT_NODE_CONFIGS.items():
            if not parameterNode.GetNodeReference(ref_name):
                self._make_prompt_node(parameterNode, ref_name, color, label)

    def recreatePromptNodes(self, parameterNode):
        """Replace both prompt nodes with brand-new ones (counter reset to 0)."""
        for ref_name, (color, label) in _PROMPT_NODE_CONFIGS.items():
            self._make_prompt_node(parameterNode, ref_name, color, label)

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

        pos_points = collect_confirmed_points(_node_records(posNode), preview_ids)
        neg_points = collect_confirmed_points(_node_records(negNode), preview_ids)

        # --- Hover preview ---
        # Include the unconfirmed placement cursor only when the interactive
        # timer is running.  Slicer tracks cursor position by updating the
        # PositionPreview point continuously, making it the live hover point.
        if widget._isInteractive:
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

            # Lazily snapshot the segment state the first time confirmed points
            # arrive.  This becomes the additive/subtractive base: pos prompts
            # add SPX regions on top, neg prompts erase from it.  Removing all
            # prompts restores this base.

            if self._prompt_base_mask is None:
                segNode = widget.ui.segmentSelector.currentNode()
                segmentID = widget.ui.segmentSelector.currentSegmentID()
                if segNode and segmentID:
                    raw = slicer.util.arrayFromSegmentBinaryLabelmap(
                        segNode, segmentID, volumeNode)
                    self._prompt_base_mask = raw.copy()

        has_base = self._prompt_base_mask is not None

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

        base_slice = (
            get_slice_from_volume(self._prompt_base_mask, axis, sliceIndex)
            if has_base else None
        )

        self._restore_working_mask(widget)

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
            self.previewResult(widget, result, axis, sliceIndex)

    def _restore_working_mask(self, widget):
        """Push the cached working mask back to Slicer, undoing any preview write.

        Called at the start of each render cycle to undo the previous preview
        overlay before re-computing it.  Uses _get_working_mask so a cache miss
        (e.g. first render after a segment switch) performs one full Slicer read.
        """
        volumeNode = widget.ui.sourceVolumeSelector.currentNode()
        segNode = widget.ui.segmentSelector.currentNode()
        segmentID = widget.ui.segmentSelector.currentSegmentID()

        if not volumeNode or not segNode or not segmentID:
            return

        working = self._get_working_mask(segNode, segmentID, volumeNode)
        slicer.util.updateSegmentBinaryLabelmapFromArray(working, segNode, segmentID, volumeNode)

    def previewResult(self, widget, mask2d, axis, sliceIndex):
        """Write a preview overlay to Slicer without touching the working mask.

        The preview is intentionally written on top of the Slicer labelmap so
        the user sees the model's suggestion in the slice view.  _restore_working_mask
        undoes it before the next render cycle so the working mask stays clean.
        """
        volumeNode = widget.ui.sourceVolumeSelector.currentNode()
        segNode = widget.ui.segmentSelector.currentNode()
        segmentID = widget.ui.segmentSelector.currentSegmentID()

        if not volumeNode or not segNode or not segmentID:
            return

        self._preview_mask = (mask2d, axis, sliceIndex)

        base = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentID, volumeNode)
        preview = base.copy()
        write_slice_to_volume(preview, mask2d, axis, sliceIndex)
        slicer.util.updateSegmentBinaryLabelmapFromArray(preview, segNode, segmentID, volumeNode)
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
            # Block the segmentSelector signal AND set the _creating_segment flag
            # before AddEmptySegment fires its VTK event.  qMRMLSegmentSelectorWidget
            # reacts to that VTK event synchronously and would emit currentSegmentChanged,
            # triggering onSegmentChanged → clearPrompts() in the middle of a render.
            widget._creating_segment = True
            widget.ui.segmentSelector.blockSignals(True)
            try:
                segmentID = segNode.GetSegmentation().AddEmptySegment("Segment_1")
                widget.ui.segmentSelector.setCurrentSegmentID(segmentID)
                widget.ui.addSegmentButton.setEnabled(True)
            finally:
                widget.ui.segmentSelector.blockSignals(False)
                widget._creating_segment = False

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

        working = self._get_working_mask(segNode, segmentID, volumeNode)
        write_slice_to_volume(working, mask2d, axis, sliceIndex)
        slicer.util.updateSegmentBinaryLabelmapFromArray(working, segNode, segmentID, volumeNode)

    def compute_spx_boundary(self, widget):
        """Compute SPX superpixel boundary pixels for the current slice.

        Reuses the SPX label-map cache when available (no extra forward pass
        if the user is already in interactive mode or has expandd).  Falls
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

        # Always go through on_expand so its cache key (which includes
        # img.shape) is validated against the current axis/slice.  Bypassing
        # it with a raw _cache_labels check causes a shape mismatch when the
        # user switches slice planes (e.g. Red → Green) after the model ran.
        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        img = get_slice_from_volume(volumeArray, axis, sliceIndex)
        img = self._apply_wl_to_slice(img)
        params = widget.getUserParameters()
        if params is None:
            return None, None, None, "Invalid model parameters."
        labels = modelFamily.on_expand(img=img, **params)

        if labels is None:
            return None, None, None, "SPX model returned no labels for this slice."

        return spx_boundary_mask(labels), axis, sliceIndex, None

    def on_confirm_model(self, widget):
        if not widget.modelFamily:
            return

        widget.modelFamily.confirm_model()
        
    def on_expand(self, widget):

        editor = widget._segEditor()
        if editor and editor.activeEffect():
            try:
                editor.activeEffect().self().apply()  # force commit stroke
            except Exception:
                pass

        self.reset_prompt_base_mask()


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

        neg_points_ras = collect_confirmed_points(_node_records(negNode), preview_ids)
        neg_ijk = self._ras_to_ijk(
            volumeNode, {"positive": [], "negative": neg_points_ras}, axis
        )["negative"]

        # Delegate to the family so the correct algorithm and user params are
        # used, and the SPX label cache is consulted before recomputing.
        labels = call_if_exists(modelFamily, 'on_expand', img=img, **params)

        if labels is None:
            slicer.util.warningDisplay("This model does not support propagation.")
            return

        self.expandSegWithSPX(widget, segNode, segmentID, volumeNode, labels, axis, sliceIndex, neg_points=neg_ijk)

    
    def on_enter_interactive(self, widget):
        if not widget.renderer:
            widget.renderer = SegmentationRenderer(widget)

        widget.setInteractiveState(True)

        # Start from a clean render state.
        self.reset_render_state()

        # Interactive mode = hover preview always on; start the 100ms timer.
        widget.renderer.start()

    def on_stop_interactive(self, widget):
        if widget.renderer:
            widget.renderer.stop()
            widget.setInteractiveState(False)

        self.reset_render_state()

    def on_assign_2d(self, widget):
        call_if_exists(widget.modelFamily, 'on_assign_2d')

    def on_assign_3d(self, widget):
        call_if_exists(widget.modelFamily, 'on_assign_3d')
    
    def on_automatic_segmentation(self, widget):
        call_if_exists(widget.modelFamily, 'on_automatic_segmentation')


    def _ras_to_ijk(self, volumeNode, scrib, axis):
        # Extract the 4×4 RAS-to-IJK matrix from VTK into numpy, then delegate
        # the batch point conversion to the Slicer-agnostic ras_to_ijk_2d helper.
        vtk_mat = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(vtk_mat)
        mat = np.array([[vtk_mat.GetElement(r, c) for c in range(4)]
                        for r in range(4)])
        return {
            "positive": ras_to_ijk_2d(mat, scrib["positive"], axis),
            "negative": ras_to_ijk_2d(mat, scrib["negative"], axis),
        }

    def getAxisAndSlice(self, widget, volumeNode=None):
        viewName = widget.currentViewName
        axis = VIEW_TO_AXIS.get(viewName, 0)

        lm = slicer.app.layoutManager()
        sliceWidget = lm.sliceWidget(viewName)

        if volumeNode is not None:
            # Convert the slice plane's RAS origin to the volume's IJK space so
            # that the index is correct regardless of the volume's spacing/origin.
            sliceNode = sliceWidget.mrmlSliceNode()
            sliceToRAS = sliceNode.GetSliceToRAS()
            ras = [sliceToRAS.GetElement(r, 3) for r in range(3)]
            rasToIjk = vtk.vtkMatrix4x4()
            volumeNode.GetRASToIJKMatrix(rasToIjk)
            ijk = rasToIjk.MultiplyPoint(ras + [1])
            sliceIndex = int(round(ijk[AXIS_TO_IJK_COMPONENT[axis]]))
        else:
            logic = sliceWidget.sliceLogic()
            sliceIndex = logic.GetSliceIndexFromOffset(logic.GetSliceOffset()) - 1

        return axis, sliceIndex
    


    def expandSegWithSPX(self, widget, segNode, segmentID, volumeNode, labels, axis, sliceIndex, neg_points=None):
        # Snapshot before any modification so Ctrl+Z can restore.
        snapshot = widget._captureSegmentationState()
        if snapshot:
            widget._undo_stack.append(('expand', snapshot))

        expanded = select_spx_labels(
            labels,
            get_slice_from_volume(
                slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segmentID, volumeNode),
                axis, sliceIndex,
            ),
            neg_points=neg_points,
        )

        # Write the new slice into the working mask and push to Slicer.
        # _get_working_mask initialises the cache if needed; writing into the
        # returned array keeps it as the single source of truth.
        working = self._get_working_mask(segNode, segmentID, volumeNode)
        write_slice_to_volume(working, expanded, axis, sliceIndex)
        slicer.util.updateSegmentBinaryLabelmapFromArray(working, segNode, segmentID, volumeNode)


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
