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

        # Core state
        self.logic           = SegmentHumanBodyLogic()
        self.ctrl            = WidgetState(self)
        self._parameterNode  = None
        self.modelFamily     = None
        self.currentViewName = None

        # Undo history — each entry is a list:
        #   ['brush',  change]               — Paint stroke
        #   ['erase',  change]               — Erase stroke
        #   ['expand', change]               — Expand operation
        #   ['point',  change, node, cp_id]  — confirmed prompt control point
        # change is a MaskChange or None.  Lists (not tuples) so the 'point'
        # path can back-fill change after an async render completes.
        self._history = []

        # Currently active input handler (BrushHandler / EraseHandler /
        # PointHandler), or None before setup completes.  Mutual exclusion is
        # enforced by InputHandler._detach_current_tool_if_exists.
        self._active_handler = None

        # Keyboard shortcuts — assigned in setup(), parented to the UI widget.
        self._undo_shortcut         = None
        self._expand_shortcut       = None
        self._spx_boundary_shortcut = None
        self._segments_shortcut     = None

        # SPX boundary overlay state
        self._spx_boundary_node    = None   # vtkMRMLLabelMapVolumeNode
        self._spx_boundary_visible = False
        self._spx_boundary_view    = None   # view name the label is currently on

        # Segment visibility toggle state
        self._saved_segments_visible   = False  # other segments (checkbox)
        self._current_segment_visible  = True   # segment being worked on (V)

    # -------------------------
    # Lifecycle
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

        self.model_classes = FAMILY_REGISTRY

        self.initializeUI()
        self.connectSignals()

        self.ui.positivePrompts.setNodeSelectorVisible(False)
        self.ui.negativePrompts.setNodeSelectorVisible(False)

        # Keyboard shortcuts — parented to uiWidget so they are active only
        # while this panel is visible.
        self._undo_shortcut = qt.QShortcut(qt.QKeySequence("Ctrl+Z"), uiWidget)
        self._undo_shortcut.connect('activated()', self.onUndo)

        self._expand_shortcut = qt.QShortcut(qt.QKeySequence("E"), uiWidget)
        self._expand_shortcut.connect('activated()', self._onExpandShortcut)

        self._spx_boundary_shortcut = qt.QShortcut(qt.QKeySequence("Q"), uiWidget)
        self._spx_boundary_shortcut.connect('activated()', self.onToggleSPXBoundary)

        self._segments_shortcut = qt.QShortcut(qt.QKeySequence("V"), uiWidget)
        self._segments_shortcut.connect('activated()', lambda: self.onToggleCurrentSegment())

        qt.QTimer.singleShot(0, self._initializeAfterSetup)
        log.debug('[Setup complete]')

    def cleanup(self):
        """Called by Slicer when the module is unloaded."""
        if self._active_handler:
            self._active_handler.detach(self)
        super().cleanup()

    def _initializeAfterSetup(self):
        if not slicer.mrmlScene:
            return

        nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLMarkupsFiducialNode')
        log.debug('[Existing markups nodes]:')
        for i in range(nodes.GetNumberOfItems()):
            log.debug(f' - {nodes.GetItemAsObject(i).GetName()}')

        self.initializeParameterNode()
        self.setParameterNode(self._parameterNode)
        self.onModelFamilyChanged()

        qt.QTimer.singleShot(0, self.updateGUIFromParameterNode)
        qt.QTimer.singleShot(0, self._preloadSegmentEditor)

    def _preloadSegmentEditor(self):
        # Switch to Segment Editor and back so its widget is fully initialised
        # before the user clicks Brush — both calls happen in the same stack
        # frame so the user sees no visible flash.
        if slicer.modules.segmenteditor.widgetRepresentation() is None:
            slicer.util.selectModule('SegmentEditor')
            slicer.util.selectModule(self.moduleName)

    # -------------------------
    # UI
    # -------------------------

    def initializeUI(self):
        dropdowns = ['modelFamilyDropdown', 'samMaskDropdown', 'modelVariantDropdown']
        for name in dropdowns:
            if hasattr(self.ui, name):
                getattr(self.ui, name).blockSignals(True)

        self.ui.modelFamilyDropdown.clear()
        self.ui.modelFamilyDropdown.addItems(list(self.model_classes.keys()))

        self.ui.samMaskDropdown.clear()
        self.ui.samMaskDropdown.addItems(['Mask-1', 'Mask-2', 'Mask-3'])
        # Slice view is auto-detected from mouse position; hide the manual dropdown.
        self.ui.sliceViewDropdown.setVisible(False)

        for name in dropdowns:
            if hasattr(self.ui, name):
                getattr(self.ui, name).blockSignals(False)

    def updateUIVisibility(self):
        visible = self.modelFamily.VISIBLE_BUTTONS if self.modelFamily else frozenset()
        ALL_BUTTONS = {
            'assignLabel2D', 'assignLabel3D',
            'expandSelectedLabelButton', 'showSPXBoundaryCheckBox',
            'runAutomaticSegmentation', 'goToMarkupsButton', 'samMaskDropdown',
        }
        for name in ALL_BUTTONS:
            getattr(self.ui, name).setVisible(name in visible)

    # -------------------------
    # Signals & Observers
    # -------------------------

    def connectSignals(self):
        ui = self.ui

        for ui_name, method_name in [
            ('assignLabel2D',            'on_assign_2d'),
            ('assignLabel3D',            'on_assign_3d'),
            ('runAutomaticSegmentation', 'on_automatic_segmentation'),
        ]:
            getattr(ui, ui_name).connect('clicked(bool)', self.bind(method_name, target="model"))

        ui.expandSelectedLabelButton.connect('clicked(bool)', lambda _=None: self._onExpand())

        for ui_name, method in [
            ('goToMarkupsButton',      self.on_go_to_markups),
            ('confirmModelSelection',  self.onConfirmClicked),
            ('addSegmentButton',       self.onAddSegment),
            ('removeSegmentButton',    self.onRemoveSegment),
            ('applyWindowLevelButton', self.onApplyWindowLevel),
        ]:
            getattr(ui, ui_name).connect('clicked(bool)', method)

        ui.showSPXBoundaryCheckBox.connect('toggled(bool)', self.onToggleSPXBoundary)
        ui.showCurrentSegmentCheckBox.connect('toggled(bool)', self.onToggleCurrentSegment)
        ui.showSegmentsCheckBox.connect('toggled(bool)', self.onToggleSavedSegments)

        ui.modelFamilyDropdown.connect('currentIndexChanged(int)', self.onModelFamilyChanged)
        ui.modelVariantDropdown.connect('currentIndexChanged(int)', self.onVariantChanged)
        ui.sourceVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        ui.segmentationNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        ui.segmentSelector.connect("currentSegmentChanged(QString)", self.onSegmentChanged)

        ui.windowSlider.connect('valueChanged(int)', self._onWindowSliderChanged)
        ui.windowSpinBox.connect('valueChanged(int)', self._onWindowSpinBoxChanged)
        ui.levelSlider.connect('valueChanged(int)', self._onLevelSliderChanged)
        ui.levelSpinBox.connect('valueChanged(int)', self._onLevelSpinBoxChanged)

        ui.brushToolButton.connect('toggled(bool)', self.onBrushToggled)
        ui.eraseToolButton.connect('toggled(bool)', self.onEraseToggled)
        ui.brushDiameterSlider.connect('valueChanged(int)', self._onBrushDiameterSliderChanged)
        ui.brushDiameterSpinBox.connect('valueChanged(int)', self._onBrushDiameterSpinBoxChanged)
        ui.brushSphereCheckBox.connect('toggled(bool)', lambda _: self._applyBrushParams())

        # Activating either place widget must deactivate the active stroke handler.
        # Connected via Qt signal (not VTK observer) so it fires immediately.
        ui.positivePrompts.activeMarkupsFiducialPlaceModeChanged.connect(self._onPlaceModeChanged)
        ui.negativePrompts.activeMarkupsFiducialPlaceModeChanged.connect(self._onPlaceModeChanged)

    def _observeMarkupsNodes(self):
        # removeObservers() wipes ALL VTK observers added via addObserver,
        # including the parameterNode → updateGUIFromParameterNode observer set
        # in setParameterNode.  Re-add it immediately so GUI updates keep working.
        self.removeObservers()
        if self._parameterNode:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent,
                             self.updateGUIFromParameterNode)

        posNode, negNode = self.logic.getPromptNodes(self._parameterNode)
        for node in [posNode, negNode]:
            if node:
                self.addObserver(node, slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                                 self._onPointConfirmed)
                self.addObserver(node, slicer.vtkMRMLMarkupsNode.PointRemovedEvent,
                                 self._onPointRemoved)

        self._connectSliceObservers()

    def _connectSliceObservers(self):
        # Re-registered here so it survives the removeObservers() call inside
        # _observeMarkupsNodes.
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if interactionNode:
            self.addObserver(interactionNode, vtk.vtkCommand.ModifiedEvent,
                             self._onInteractionModeChanged)

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
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent,
                                self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent,
                             self.updateGUIFromParameterNode)
            self._observeMarkupsNodes()
        qt.QTimer.singleShot(0, self.updateGUIFromParameterNode)

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if not self._parameterNode:
            return

        posNode, negNode = self.logic.getPromptNodes(self._parameterNode)
        volumeNode, segNode = self.logic.getVolumeAndSegmentation(self._parameterNode)

        # Pause so programmatic setCurrentNode calls do not spuriously fire
        # activeMarkupsFiducialPlaceModeChanged and activate PointHandler.
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

        # Apply visibility rules to the newly active segmentation.
        currentID = self.ui.segmentSelector.currentSegmentID() if segNode else None
        self._apply_saved_segments_visibility(exclude=currentID)
        self._current_segment_visible = True
        if segNode and currentID:
            dn = segNode.GetDisplayNode()
            if dn:
                dn.SetSegmentVisibility(currentID, True)
        self.ui.showCurrentSegmentCheckBox.blockSignals(True)
        self.ui.showCurrentSegmentCheckBox.setChecked(True)
        self.ui.showCurrentSegmentCheckBox.blockSignals(False)
        self.ui.showSegmentsCheckBox.blockSignals(True)
        self.ui.showSegmentsCheckBox.setChecked(self._saved_segments_visible)
        self.ui.showSegmentsCheckBox.blockSignals(False)

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        if not self._parameterNode:
            return

        volumeNode = self.ui.sourceVolumeSelector.currentNode()
        self._syncWLSlidersFromVolume(volumeNode)
        segNode = self.ui.segmentationNodeSelector.currentNode()

        # Auto-create a segmentation node when a volume is selected but none
        # exists yet — selecting a volume is the natural trigger for this.
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

    # -------------------------
    # Model selection
    # -------------------------

    def onModelFamilyChanged(self, *args):
        self.setConfirmState(False)
        ModelClass = self.model_classes.get(self.ui.modelFamilyDropdown.currentText,
                                            BaseModelFamily)
        self.modelFamily = ModelClass()
        self.updateModelVariants()
        if hasattr(self.modelFamily, "VARIANTS") and self.modelFamily.VARIANTS:
            self.modelFamily.variant = self.modelFamily.VARIANTS[0]
        self.updateUIVisibility()
        self.updateParamPlaceholder()
        if not isinstance(self.modelFamily, SPXModelFamily):
            self._hideSPXBoundary()

    def updateModelVariants(self):
        dropdown = self.ui.modelVariantDropdown
        dropdown.blockSignals(True)
        dropdown.clear()
        variants = (self.modelFamily.VARIANTS
                    if self.modelFamily and hasattr(self.modelFamily, "VARIANTS")
                    else ["None"])
        dropdown.addItems(variants)
        if variants:
            dropdown.setCurrentIndex(0)
        dropdown.blockSignals(False)

    def onVariantChanged(self, *args):
        self.setConfirmState(False)
        if not self.modelFamily:
            return
        self.modelFamily.variant = self.ui.modelVariantDropdown.currentText
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

    def getUserParameters(self):
        return parse_user_parameters(self.ui.paramTextEdit.toPlainText())

    def updateParamPlaceholder(self):
        if not self.modelFamily or not self.modelFamily.variant:
            self.ui.paramTextEdit.setPlaceholderText("Select a model variant.")
            return
        try:
            key = (self.modelFamily._get_model_key()
                   if hasattr(self.modelFamily, "_get_model_key")
                   else self.modelFamily.variant)
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
        self.ui.docLinkLabel.setText(
            f'<a href="{url}">View documentation</a>' if url else ""
        )

    def on_go_to_markups(self, *args):
        slicer.util.selectModule('Markups')

    def bind(self, method_name, target="logic"):
        if target == "logic":
            return lambda _=None: getattr(self.logic, method_name)(self)
        elif target == "model":
            return lambda _=None: call_if_exists(self.modelFamily, method_name)
        else:
            return getattr(self, method_name)

    # -------------------------
    # Segment management
    # -------------------------

    def getOrCreateSegmentationNode(self):
        volumeNode = self.ui.sourceVolumeSelector.currentNode()
        segNode    = self.ui.segmentationNodeSelector.currentNode()
        if not segNode and self._parameterNode:
            _, segNode = self.logic.getVolumeAndSegmentation(self._parameterNode)
        if not segNode and volumeNode:
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segNode.CreateDefaultDisplayNodes()
            segNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
            segNode.CreateClosedSurfaceRepresentation()
            self.logic.setVolumeAndSegmentation(self._parameterNode, volumeNode, segNode)
            self._parameterNode.Modified()
        return segNode

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
            segNode   = self.ui.segmentationNodeSelector.currentNode()
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

    def onSegmentChanged(self, segmentID):
        if not segmentID or self.ctrl.creating_segment:
            return
        if self.ui.brushToolButton.isChecked() or self.ui.eraseToolButton.isChecked():
            editor = self._segEditor()
            if editor:
                editor.setCurrentSegmentID(segmentID)
                self.ctrl.brush_in_progress = False
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
            # Apply saved-segment rule to all others; always show the new current.
            self._apply_saved_segments_visibility(exclude=segmentID)
            self._current_segment_visible = True
            dn = segNode.GetDisplayNode()
            if dn:
                dn.SetSegmentVisibility(segmentID, True)
            self.ui.showCurrentSegmentCheckBox.blockSignals(True)
            self.ui.showCurrentSegmentCheckBox.setChecked(True)
            self.ui.showCurrentSegmentCheckBox.blockSignals(False)

    def clearPrompts(self):
        self._history.clear()
        if isinstance(self._active_handler, StrokeHandler):
            self._active_handler.reset(self)

        # Recreate fresh markup nodes (counter=0) so the placement cursor is
        # always "Positive 1" / "Negative 1".  Reusing and clearing old nodes
        # leaves the internal counter at N, causing the next cursor to show N+1.
        self.logic.recreatePromptNodes(self._parameterNode)
        self._observeMarkupsNodes()

        posNode, negNode = self.logic.getPromptNodes(self._parameterNode)

        # Switch to ViewTransform before wiring nodes.  setCurrentNode on an
        # empty node inherits the widget's current placement state — calling it
        # while Place mode is active would auto-create an unwanted cursor.
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SwitchToViewTransformMode()

        self.ui.negativePrompts.setCurrentNode(negNode)
        self.ui.positivePrompts.setCurrentNode(posNode)

        # Start persistent placement on the POSITIVE node only so the
        # "Positive 1" tracking cursor appears without touching the negative
        # widget's placement state.
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetActivePlaceNodeID(posNode.GetID())
        selectionNode.SetActivePlaceNodeClassName(posNode.GetClassName())
        interactionNode.SwitchToPersistentPlaceMode()
        # Register PointHandler directly — calling attach() would re-enter the
        # mode switch we just performed.
        self._active_handler = PointHandler()

    # -------------------------
    # Interaction mode
    # -------------------------

    def _onPlaceModeChanged(self, active: bool):
        """Qt slot: fires when the user clicks either markup place-widget button."""
        if not active or self.ctrl.is_paused:
            return
        if not isinstance(self._active_handler, PointHandler):
            PointHandler().attach(self)

    def _onInteractionModeChanged(self, caller=None, event=None):
        """VTK observer fallback: activate point mode when Slicer enters Place mode."""
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

    # -------------------------
    # Point events
    # -------------------------

    def _onPointConfirmed(self, caller=None, event=None):
        """PointPositionDefinedEvent — a placement was just confirmed by the user."""
        if caller is None:
            return
        self._resolveActiveView()
        n = caller.GetNumberOfControlPoints()
        if n == 0:
            return
        cp_id  = caller.GetNthControlPointID(n - 1)
        change = self.logic.commit_point(self, caller, cp_id)
        self._history.append(['point', change, caller, cp_id])
        log.debug('[Widget] point confirmed — change=%s  history=%d',
                  change is not None, len(self._history))

    def _onPointRemoved(self, caller=None, event=None):
        """PointRemovedEvent — a prompt point was manually deleted by the user."""
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

    # -------------------------
    # Window / Level
    # -------------------------

    def _syncWLSlidersFromVolume(self, volumeNode):
        if not volumeNode:
            return
        displayNode = volumeNode.GetScalarVolumeDisplayNode()
        if not displayNode:
            return
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
        self._resetWLButton()
        self.logic.set_window_level(None, None)

    def _resetWLButton(self):
        btn = self.ui.applyWindowLevelButton
        btn.setText("Apply Window / Level")
        btn.setEnabled(True)

    def _updateDisplayNodeWL(self):
        volumeNode = self.ui.sourceVolumeSelector.currentNode()
        if not volumeNode:
            return
        displayNode = volumeNode.GetScalarVolumeDisplayNode()
        if not displayNode:
            return
        # AutoWindowLevelOff is required — without it Slicer silently overrides
        # any manually set W/L with its own computed range on the next render.
        displayNode.AutoWindowLevelOff()
        displayNode.SetWindow(self.ui.windowSlider.value)
        displayNode.SetLevel(self.ui.levelSlider.value)

    def _onWLControlChanged(self, _=None):
        self._updateDisplayNodeWL()
        btn = self.ui.applyWindowLevelButton
        if not btn.isEnabled():
            self.logic.set_window_level(None, None)
            self._resetWLButton()

    def _sync_wl_widgets(self, peer, value):
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

    def onApplyWindowLevel(self, _=None):
        """Confirm W/L for model inference; volume scalar data is never modified."""
        self.logic.set_window_level(
            self.ui.windowSpinBox.value,
            self.ui.levelSpinBox.value,
        )
        btn = self.ui.applyWindowLevelButton
        btn.setText("W/L Applied")
        btn.setEnabled(False)

    # -------------------------
    # Expand  (E)
    # -------------------------

    def _onExpand(self):
        """Run expand and push the result to history."""
        self._resolveActiveView()
        change = self.logic.on_expand(self)
        if change is not None:
            self._history.append(['expand', change])

    def _onExpandShortcut(self):
        """E hotkey: flush any active stroke → expand → restore the tool."""
        active = self._active_handler
        prior_stroke_class = type(active) if isinstance(active, StrokeHandler) else None
        if prior_stroke_class:
            active.detach(self)
        self._onExpand()
        if prior_stroke_class:
            prior_stroke_class().attach(self)

    # -------------------------
    # Undo  (Ctrl+Z)
    # -------------------------

    def _add_history(self, entry):
        """Append *entry* to the undo history (called by input handlers)."""
        self._history.append(entry)

    def _resolveActiveView(self):
        """Update currentViewName to whichever slice view the cursor is over."""
        lm = slicer.app.layoutManager()
        for viewName in ("Red", "Green", "Yellow"):
            sw = lm.sliceWidget(viewName)
            if sw and sw.sliceView().underMouse():
                self.currentViewName = viewName
                return

    def onUndo(self):
        log.debug('[Widget] Undo pressed — history depth %d', len(self._history))
        if isinstance(self._active_handler, StrokeHandler):
            self._active_handler.flush(self)
        if not self._history:
            return

        entry       = self._history.pop()
        action_type = entry[0]

        if action_type in ('brush', 'erase', 'expand'):
            self.logic.reverse_change(self, entry[1])

        elif action_type == 'point':
            _, change, node, cp_id = entry
            _, negNode   = self.logic.getPromptNodes(self._parameterNode)
            is_negative  = (node is negNode)

            self.ctrl.pause()
            try:
                idx = node.GetControlPointIndexByID(cp_id)
                if idx >= 0:
                    node.RemoveNthControlPoint(idx)
                # Recreate the node when empty to reset the ID counter,
                # otherwise the next placement cursor shows "N+1" instead of "1".
                if node.GetNumberOfControlPoints() == 0:
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
        """Return the slice composite node for *view_name*, or None."""
        sw = slicer.app.layoutManager().sliceWidget(view_name)
        return sw.sliceLogic().GetSliceCompositeNode() if sw else None

    def _hideSPXBoundary(self):
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
        """Q key / checkbox: show or hide the SPX superpixel boundary overlay."""
        if not isinstance(self.modelFamily, SPXModelFamily):
            return

        if self._spx_boundary_visible:
            self._hideSPXBoundary()
            return

        try:
            boundary_2d, axis, sliceIndex = self.logic.compute_spx_boundary(self)
        except ValueError as exc:
            slicer.util.warningDisplay(f"SPX boundary: {exc}")
            self.ui.showSPXBoundaryCheckBox.blockSignals(True)
            self.ui.showSPXBoundaryCheckBox.setChecked(False)
            self.ui.showSPXBoundaryCheckBox.blockSignals(False)
            return

        volumeNode = self.ui.sourceVolumeSelector.currentNode()

        if (self._spx_boundary_node is None
                or not slicer.mrmlScene.IsNodePresent(self._spx_boundary_node)):
            self._spx_boundary_node = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLLabelMapVolumeNode', 'SPX Boundaries'
            )
            self._spx_boundary_node.CreateDefaultDisplayNodes()
            sourceArray = slicer.util.arrayFromVolume(volumeNode)
            if sourceArray is None:
                slicer.util.warningDisplay("Cannot read volume data — SPX boundary not shown.")
                return
            slicer.util.updateVolumeFromArray(self._spx_boundary_node,
                                              np.zeros(sourceArray.shape, dtype=np.uint8))
            ijkToRAS = vtk.vtkMatrix4x4()
            volumeNode.GetIJKToRASMatrix(ijkToRAS)
            self._spx_boundary_node.SetIJKToRASMatrix(ijkToRAS)
            # The default "Labels" color table is read-only; create a small
            # User-type table so we can assign label 1 → yellow.
            colorNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLColorTableNode')
            colorNode.SetTypeToUser()
            colorNode.SetNumberOfColors(2)
            colorNode.SetColor(0, 0.0, 0.0, 0.0, 0.0)  # 0 → transparent
            colorNode.SetColor(1, 1.0, 1.0, 0.0, 1.0)  # 1 → yellow, opaque
            self._spx_boundary_node.GetDisplayNode().SetAndObserveColorNodeID(
                colorNode.GetID()
            )

        boundaryArray = slicer.util.arrayFromVolume(self._spx_boundary_node)
        boundaryArray[:] = 0
        write_slice_to_volume(boundaryArray, boundary_2d, axis, sliceIndex)
        self._spx_boundary_node.GetImageData().Modified()
        self._spx_boundary_node.Modified()

        viewName  = self.currentViewName
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

    # -------------------------
    # Segment Visibility
    # -------------------------

    def _apply_saved_segments_visibility(self, exclude=None):
        """Set visibility of every segment except `exclude` to _saved_segments_visible."""
        segNode = self.ui.segmentationNodeSelector.currentNode()
        if not segNode:
            return
        dn = segNode.GetDisplayNode()
        if not dn:
            return
        seg = segNode.GetSegmentation()
        for i in range(seg.GetNumberOfSegments()):
            sid = seg.GetNthSegmentID(i)
            if sid != exclude:
                dn.SetSegmentVisibility(sid, self._saved_segments_visible)

    def onToggleSavedSegments(self, visible=None):
        """Checkbox toggled(bool) handler — show/hide all saved (non-current) segments."""
        if visible is None:
            visible = not self._saved_segments_visible
        self._saved_segments_visible = visible
        currentID = self.ui.segmentSelector.currentSegmentID()
        self._apply_saved_segments_visibility(exclude=currentID)
        self.ui.showSegmentsCheckBox.blockSignals(True)
        self.ui.showSegmentsCheckBox.setChecked(visible)
        self.ui.showSegmentsCheckBox.blockSignals(False)

    def onToggleCurrentSegment(self, visible=None):
        """V hotkey or checkbox — toggle visibility of only the segment currently being worked on."""
        if visible is None:
            visible = not self._current_segment_visible
        segNode = self.ui.segmentationNodeSelector.currentNode()
        segmentID = self.ui.segmentSelector.currentSegmentID()
        if segNode and segmentID:
            dn = segNode.GetDisplayNode()
            if dn:
                dn.SetSegmentVisibility(segmentID, visible)
            self._current_segment_visible = visible
        else:
            visible = self._current_segment_visible   # snap back
        self.ui.showCurrentSegmentCheckBox.blockSignals(True)
        self.ui.showCurrentSegmentCheckBox.setChecked(visible)
        self.ui.showCurrentSegmentCheckBox.blockSignals(False)


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
