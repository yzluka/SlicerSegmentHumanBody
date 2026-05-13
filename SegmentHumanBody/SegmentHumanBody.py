import qt, vtk, slicer
import logging
import numpy as np
import vtk.util.numpy_support as _vtk_ns
from contextlib import contextmanager
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin

import tempfile

from core.utils import next_segment_name
from core.modelFamilies import FAMILY_REGISTRY
from core._mouse_recorder import get_recorder
from core._input import StrokeHandler, BrushHandler, EraseHandler, PointHandler
from core._audio_recorder import StandaloneAudioRecorder, merge_chunks_to_wav

log = logging.getLogger(__name__)


@contextmanager
def _suppress_vtk_warnings():
    previous = vtk.vtkObject.GetGlobalWarningDisplay()
    vtk.vtkObject.GlobalWarningDisplayOff()
    try:
        yield
    finally:
        if previous:
            vtk.vtkObject.GlobalWarningDisplayOn()
        else:
            vtk.vtkObject.GlobalWarningDisplayOff()

# MRML parameter-node reference keys
_INPUT_VOLUME = 'InputVolume'
_SEGMENTATION = 'Segmentation'
_GEOMETRY_SIGNATURE_ATTR = 'SegmentHumanBody.referenceGeometrySignature'

# Segment tag keys — store the markup node IDs per segment natively
_POS_TAG = 'SegmentHumanBody.posNodeID'
_NEG_TAG = 'SegmentHumanBody.negNodeID'

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
        self.parent.acknowledgementText = 'Developed at Duke University.'


#
# Widget
#

class SegmentHumanBodyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Pure wrapper around Slicer's native Segment Editor.

    No model loading, no custom stroke tracking, no undo history.
    Every tool delegates directly to the Segment Editor effect.
    """

    # Widgets that belong to model features.
    # Hidden in this branch — layout preserved for future re-enabling.
    _HIDDEN_WIDGETS = frozenset({
        'modelFamilyDropdown', 'modelVariantDropdown',
        'confirmModelSelection', 'paramTextEdit', 'docLinkLabel',
        'assignLabel2D', 'assignLabel3D', 'runAutomaticSegmentation',
        'expandSelectedLabelButton', 'showSPXBoundaryCheckBox',
        'goToMarkupsButton', 'samMaskDropdown', 'sliceViewDropdown',
        'exportAnnotationLogButton', 'importAnnotationLogButton',
    })

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = SegmentHumanBodyLogic()
        self.modelFamily = FAMILY_REGISTRY['Default']('Identity')
        self.modelFamily.confirm_model()
        self._parameterNode         = None
        self._savedSegmentsVisible  = False
        self._currentSegmentVisible = True
        self.currentViewName        = 'Red'
        self._recorder              = get_recorder()
        self._eof_widget            = None   # borrowed EffectsOptionsFrame
        self._eof_orig_parent       = None   # original parent to return it to
        self._active_handler        = None   # current InputHandler subclass instance
        self._attaching_handler     = None   # set during InputHandler.attach() to suppress spurious detach events
        self._active_prompt_widget  = None   # positivePrompts or negativePrompts last activated
        self._suppressing_place_mode = False  # True while segment creation is in progress
        self._observed_segmentation = None   # vtkMRMLSegmentationNode being tracked
        self._observed_seg_obj      = None   # its vtkSegmentation (holds the event)
        self._observed_segment_ids  = set()  # current IDs for lifecycle recording
        self._observed_segment_names = {}    # last known names for removal events
        self._recorded_prompt_node_ids = set()
        self._active_point_drags = {}
        self._pending_point_confirmations = {}
        self._recorded_prompt_point_cache = {}
        self._recently_placed = {}
        self._syncing_parameter_node_to_ui = False
        self._reverting_source_volume = False
        self._last_source_volume_node = None
        self._volume_origin_scan_pending = False
        self._volume_origin_scan_timer = None
        self._volume_geometry_warning_signature = None
        self._recording_saved = True
        self._audio_recorder: StandaloneAudioRecorder | None = None
        self._audio_temp_dir: str | None = None
        self._audio_test_active: bool = False
        self._audio_test_recorder: StandaloneAudioRecorder | None = None
        self._audio_test_timer: qt.QTimer | None = None
        self._audio_test_chunks: list | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def setup(self):
        super().setup()

        uiWidget = slicer.util.loadUI(self.resourcePath('UI/SegmentHumanBody.ui'))
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        self.ui.sourceVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.segmentationNodeSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.segmentSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.segmentSelector.segmentationNodeSelectorVisible = False

        for name in self._HIDDEN_WIDGETS:
            w = getattr(self.ui, name, None)
            if w:
                w.setVisible(False)

        # Show the prompt widgets but hide the node-selector dropdowns inside
        # them — the user places points via the place button, not via the selector.
        self.ui.positivePrompts.setNodeSelectorVisible(False)
        self.ui.negativePrompts.setNodeSelectorVisible(False)
        self._set_prompt_nodes(None, None)  # configure placement mode; nodes wired later

        self._connectSignals(uiWidget)
        self._observeVolumeImports()
        self.initializeParameterNode()
        self._update_record_ui()
        self._recorder.context_fn = self._recorder_context
        self._recorder.on_record_appended = self._onRecorderAppended
        qt.QTimer.singleShot(0, self._preloadSegmentEditor)

    def cleanup(self):
        self._recorder.context_fn = None
        self._recorder.on_record_appended = None
        self.removeObservers()
        self._deactivateEffect()
        self._returnEffectsOptionsFrame()

    def enter(self):
        self.initializeParameterNode()

    def exit(self):
        self._deactivateEffect()
        self._returnEffectsOptionsFrame()

    def _preloadSegmentEditor(self):
        if slicer.modules.segmenteditor.widgetRepresentation() is None:
            slicer.util.selectModule('SegmentEditor')
            slicer.util.selectModule(self.moduleName)

    def _observeVolumeImports(self):
        for event_name in ('NodeAddedEvent', 'EndImportEvent', 'EndBatchProcessEvent'):
            event = getattr(slicer.vtkMRMLScene, event_name, None)
            if event is not None:
                self.addObserver(slicer.mrmlScene, event, self._schedule_volume_origin_scan)
        self._schedule_volume_origin_scan()

    def _schedule_volume_origin_scan(self, *_):
        self._volume_origin_scan_pending = True
        if self._volume_origin_scan_timer is None:
            timer = qt.QTimer()
            timer.setSingleShot(True)
            timer.connect('timeout()', self._normalize_loaded_volume_origins)
            self._volume_origin_scan_timer = timer
        self._volume_origin_scan_timer.start(1500)

    def _normalize_loaded_volume_origins(self):
        self._volume_origin_scan_pending = False
        self.logic.normalize_scalar_volume_names_from_filenames()
        current = self.ui.sourceVolumeSelector.currentNode()
        current_key = self.logic.volume_storage_key(current)
        replacements = self.logic.replace_duplicate_scalar_volume_nodes()
        if current_key in replacements:
            self.ui.sourceVolumeSelector.setCurrentNode(replacements[current_key])
        stats = self.logic.scene_volume_geometry_statistics()
        if stats['group_count'] > 1:
            warning_signature = stats['signature']
            if warning_signature != self._volume_geometry_warning_signature:
                self._volume_geometry_warning_signature = warning_signature
                slicer.util.warningDisplay(
                    'Loaded volumes have inconsistent geometry.\n\n'
                    f'{stats["summary"]}\n\n'
                    'Loading is allowed. Origin differences are ignored and '
                    'compatible zero-origin volumes will still be normalized, '
                    'but shape, spacing, or orientation differences may not '
                    'align as one sequence set.')
        elif stats['group_count'] == 1:
            self._volume_geometry_warning_signature = None
        self.logic.normalize_compatible_scene_volume_origins()

    @staticmethod
    def _configureUnlimitedPlacement(markups_widget):
        """Put a qSlicerSimpleMarkupsWidget in persistent multi-point placement.

        Slicer 4.x used setMaximumNumberOfMarkups; 5.x exposes the inner
        qMRMLMarkupsPlaceWidget directly.  We try both so the code is version-
        agnostic.
        """
        # Slicer 4.x / some 5.x builds
        if hasattr(markups_widget, 'setMaximumNumberOfMarkups'):
            markups_widget.setMaximumNumberOfMarkups(-1)
            return
        # Slicer 5.x: find the internal qSlicerMarkupsPlaceWidget (may be nested).
        # PlaceMultipleMarkupsType enum: ShowPlaceMultipleMarkupsOption=0,
        # HidePlaceMultipleMarkupsOption=1, ForcePlaceSingleMarkup=2,
        # ForcePlaceMultipleMarkups=3.
        for child in markups_widget.findChildren(qt.QWidget):
            if hasattr(child, 'setPlaceMultipleMarkups'):
                force = getattr(child, 'ForcePlaceMultipleMarkups', 3)
                child.setPlaceMultipleMarkups(force)
                return
        log.warning('[setup] Could not configure continuous markup placement on %s',
                    markups_widget.objectName)

    def _set_prompt_nodes(self, pos_node, neg_node):
        """Wire both markup widgets to the given nodes and re-apply placement config.

        Negative is configured first so that positive ends up as the last-touched
        widget — Slicer makes the last setCurrentNode call the active placement target.
        Place-mode states are preserved only when a PointHandler is currently active;
        otherwise they are forced to [False, False] so programmatic node rewiring
        (e.g. segment switch, segment delete) cannot accidentally re-activate
        point-placement mode when no tool is selected.
        """
        if isinstance(self._active_handler, PointHandler):
            self._set_prompt_nodes_preserving_place_mode(pos_node, neg_node)
        else:
            self._set_prompt_nodes_preserving_place_mode(
                pos_node, neg_node, force_states=[False, False])

    def _markup_place_widgets(self):
        """Return the inner place widgets for positive/negative prompt widgets."""
        result = []
        for markup_widget in (self.ui.positivePrompts, self.ui.negativePrompts):
            place = None
            for child in markup_widget.findChildren(qt.QWidget):
                if hasattr(child, 'setPlaceModeEnabled'):
                    place = child
                    break
            result.append((markup_widget, place))
        return result

    def _prompt_place_states(self):
        states = []
        for _, place in self._markup_place_widgets():
            active = False
            if place is not None and hasattr(place, 'placeModeEnabled'):
                value = place.placeModeEnabled
                active = bool(value() if callable(value) else value)
            states.append(active)
        return states

    def _set_prompt_place_states(self, states):
        for (_, place), active in zip(self._markup_place_widgets(), states):
            if place is not None:
                place.setPlaceModeEnabled(bool(active))

    def _set_prompt_widget_place_mode(self, markup_widget, active):
        if markup_widget is None:
            return
        old = self._suppressing_place_mode
        self._suppressing_place_mode = True
        try:
            for child in markup_widget.findChildren(qt.QWidget):
                if hasattr(child, 'setPlaceModeEnabled'):
                    child.setPlaceModeEnabled(bool(active))
                    break
        finally:
            self._suppressing_place_mode = old

    def _deactivate_prompt_place_mode(self):
        old = self._suppressing_place_mode
        self._suppressing_place_mode = True
        try:
            self._set_prompt_place_states([False, False])
        finally:
            self._suppressing_place_mode = old

    def _set_prompt_nodes_preserving_place_mode(self, pos_node, neg_node,
                                                 force_states=None):
        states = force_states if force_states is not None else self._prompt_place_states()
        old = self._suppressing_place_mode
        self._suppressing_place_mode = True
        try:
            self.ui.negativePrompts.setCurrentNode(neg_node)
            if neg_node is not None:
                self._configureUnlimitedPlacement(self.ui.negativePrompts)
            self.ui.positivePrompts.setCurrentNode(pos_node)
            if pos_node is not None:
                self._configureUnlimitedPlacement(self.ui.positivePrompts)
            # Restore inside suppression: place-state signals must never escape
            # into _onPlaceModeChanged during programmatic node rewiring.
            self._set_prompt_place_states(states)
        finally:
            self._suppressing_place_mode = old
        self._observe_prompt_node_for_recording(pos_node, is_negative=False)
        self._observe_prompt_node_for_recording(neg_node, is_negative=True)

    def _ensure_current_prompt_nodes(self):
        seg = self.ui.segmentationNodeSelector.currentNode()
        segID = self.ui.segmentSelector.currentSegmentID()
        if not seg or not segID:
            return None, None
        pos_node, neg_node = self.logic.create_segment_prompt_nodes(seg, segID)
        self._set_prompt_nodes(pos_node, neg_node)
        return pos_node, neg_node

    # ------------------------------------------------------------------ #
    # Signal wiring                                                        #
    # ------------------------------------------------------------------ #

    def _connectSignals(self, uiWidget):
        ui = self.ui
        ui.sourceVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)',
                                        self._onSourceVolumeSelectorChanged)
        ui.segmentationNodeSelector.connect('currentNodeChanged(vtkMRMLNode*)',
                                            self._onSegmentationSelectorChanged)
        ui.segmentSelector.connect('currentSegmentChanged(QString)',
                                   self._onSegmentIDChanged)
        ui.addSegmentButton.connect('clicked(bool)', self._onAddSegment)
        ui.removeSegmentButton.connect('clicked(bool)', self._onRemoveSegment)
        ui.clearVolumesButton.connect('clicked(bool)', self._onClearLoadedVolumes)
        ui.brushToolButton.connect('toggled(bool)', self._onBrushToggled)
        ui.eraseToolButton.connect('toggled(bool)', self._onEraseToggled)
        ui.overwriteModeDropdown.connect('currentIndexChanged(int)', self._onOverwriteModeChanged)
        ui.windowSlider.connect('valueChanged(int)',  self._onWindowSliderChanged)
        ui.windowSpinBox.connect('valueChanged(int)', self._onWindowSpinBoxChanged)
        ui.levelSlider.connect('valueChanged(int)',   self._onLevelSliderChanged)
        ui.levelSpinBox.connect('valueChanged(int)',  self._onLevelSpinBoxChanged)
        ui.applyWindowLevelButton.connect('clicked(bool)', self._onApplyWindowLevel)
        ui.showCurrentSegmentCheckBox.connect('toggled(bool)', self._onToggleCurrentSegment)
        ui.showSegmentsCheckBox.connect('toggled(bool)', self._onToggleSavedSegments)

        ui.recordButton.connect('clicked(bool)',            self.onRecord)
        ui.recordEventOnlyButton.connect('clicked(bool)',  self.onRecordEventOnly)
        ui.stopRecordButton.connect('clicked(bool)',        self.onStopRecord)
        ui.exportRecordButton.connect('clicked(bool)',      self.onExportRecord)
        ui.recordTestAudioButton.connect('clicked(bool)',  self.onRecordTestAudio)
        ui.playTestAudioButton.connect('clicked(bool)',    self.onPlayTestAudio)
        self._populate_audio_devices()

        # Record volume / seg changes when active.
        ui.sourceVolumeSelector.connect(
            'currentNodeChanged(vtkMRMLNode*)',
            self._onRecordedVolumeChanged,
        )
        ui.segmentationNodeSelector.connect(
            'currentNodeChanged(vtkMRMLNode*)',
            lambda node: (self._recorder.record_volume_changed(f'seg:{node.GetName()}' if node else None)
                          if self._recorder.is_active else None),
        )

        # Connect point-placement mode changes so PointHandler runs the segment guard.
        # Pass the source widget so _onPlaceModeChanged can track which list is active.
        for markup_widget in (ui.positivePrompts, ui.negativePrompts):
            for child in markup_widget.findChildren(qt.QWidget):
                if hasattr(child, 'setPlaceMultipleMarkups'):
                    child.connect(
                        'activeMarkupsFiducialPlaceModeChanged(bool)',
                        lambda active, w=markup_widget: self._onPlaceModeChanged(active, w),
                    )
                    break

        sc = qt.QShortcut
        sc(qt.QKeySequence('Ctrl+Z'),       uiWidget).connect('activated()', self._onUndo)
        sc(qt.QKeySequence('Ctrl+Shift+Z'), uiWidget).connect('activated()', self._onRedo)
        sc(qt.QKeySequence('A'), uiWidget).connect(
            'activated()', lambda: self._select_relative_volume(1))
        sc(qt.QKeySequence('W'), uiWidget).connect(
            'activated()', lambda: self._select_relative_volume(-1))
        sc(qt.QKeySequence('Z'), uiWidget).connect(
            'activated()', lambda: self._select_relative_segment(-1))
        sc(qt.QKeySequence('C'), uiWidget).connect(
            'activated()', lambda: self._select_relative_segment(1))
        sc(qt.QKeySequence('Q'), uiWidget).connect(
            'activated()', self._toggle_current_segment_visibility)
        sc(qt.QKeySequence('S'), uiWidget).connect(
            'activated()', self._toggle_saved_segments_visibility)

    # ------------------------------------------------------------------ #
    # Parameter node                                                       #
    # ------------------------------------------------------------------ #

    def initializeParameterNode(self):
        pn = self.logic.getParameterNode()
        if pn is None:
            pn = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode',
                                                    self.moduleName)
        if pn is self._parameterNode:
            return
        if self._parameterNode:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent,
                                self._onParameterNodeModified)
        self._parameterNode = pn
        self.addObserver(pn, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
        qt.QTimer.singleShot(0, self._onParameterNodeModified)

    def _onParameterNodeModified(self, *_):
        if not self._parameterNode:
            return
        if not hasattr(self, 'ui'):
            return
        if self._syncing_parameter_node_to_ui:
            return
        self._syncing_parameter_node_to_ui = True
        try:
            vol = self._parameterNode.GetNodeReference(_INPUT_VOLUME)
            seg = self._parameterNode.GetNodeReference(_SEGMENTATION)
            self.ui.sourceVolumeSelector.setCurrentNode(vol)
            self.ui.segmentationNodeSelector.setCurrentNode(seg)
            if seg:
                self.ui.segmentSelector.setCurrentNode(seg)
            self.ui.addSegmentButton.setEnabled(vol is not None)
            self._syncWLFromVolume(vol)
            self._sync_selected_nodes_to_views()
            self._last_source_volume_node = vol
            self.ui.showCurrentSegmentCheckBox.blockSignals(True)
            self.ui.showCurrentSegmentCheckBox.setChecked(True)
            self.ui.showCurrentSegmentCheckBox.blockSignals(False)
            self.ui.showSegmentsCheckBox.blockSignals(True)
            self.ui.showSegmentsCheckBox.setChecked(self._savedSegmentsVisible)
            self.ui.showSegmentsCheckBox.blockSignals(False)
            segID = self.ui.segmentSelector.currentSegmentID() if seg else None
            self.logic.set_saved_segments_visibility(
                seg, segID, self._savedSegmentsVisible)
            if seg and segID:
                pos_node, neg_node = self.logic.create_segment_prompt_nodes(seg, segID)
            else:
                pos_node, neg_node = None, None
            self._set_prompt_nodes(pos_node, neg_node)
        finally:
            self._syncing_parameter_node_to_ui = False

    def _set_parameter_node_reference(self, role, node):
        if not self._parameterNode:
            return
        self._syncing_parameter_node_to_ui = True
        try:
            self._parameterNode.SetNodeReferenceID(
                role, node.GetID() if node else '')
            self._parameterNode.Modified()
        finally:
            self._syncing_parameter_node_to_ui = False

    def _onSourceVolumeSelectorChanged(self, *_):
        if self._syncing_parameter_node_to_ui:
            return
        if not self._parameterNode:
            return
        if self._reverting_source_volume:
            return
        previous_vol = self._last_source_volume_node
        vol = self.ui.sourceVolumeSelector.currentNode()
        mismatch_action = self._resolve_geometry_mismatch_for_volume(vol)
        if mismatch_action == 'cancel':
            self._reverting_source_volume = True
            try:
                self.ui.sourceVolumeSelector.setCurrentNode(self._last_source_volume_node)
            finally:
                self._reverting_source_volume = False
            return
        self._set_parameter_node_reference(_INPUT_VOLUME, vol)
        self.ui.addSegmentButton.setEnabled(vol is not None)
        self._syncWLFromVolume(vol)
        if vol is not None:
            self.logic.normalize_volume_origin_from_compatible_scene_volume(vol)
            self.logic.show_volume_in_slice_views(vol, fit=False, propagate=False)
        self._last_source_volume_node = vol
        if mismatch_action == 'create':
            self._create_and_select_segmentation_for_volume(vol)

    def _onSegmentationSelectorChanged(self, *_):
        if self._syncing_parameter_node_to_ui:
            return
        if not self._parameterNode:
            return
        seg = self.ui.segmentationNodeSelector.currentNode()
        # Externally loaded segmentation nodes may have no display node, which
        # causes qMRMLSegmentsModel to warn. Ensure one exists before the
        # segment selector tries to render the segment list.
        if seg and not seg.GetDisplayNode():
            seg.CreateDefaultDisplayNodes()
        self._set_parameter_node_reference(_SEGMENTATION, seg)
        self._rewire_segmentation_observer(seg)
        self._sync_selected_nodes_to_views()

    def _onNodeSelectorChanged(self, *_):
        """Compatibility path for tests or external callers using the old hook."""
        if self._syncing_parameter_node_to_ui:
            return
        self._onSourceVolumeSelectorChanged()
        self._onSegmentationSelectorChanged()

    def _onClearLoadedVolumes(self, *_):
        count = self.logic.clear_scalar_volume_nodes()
        self._last_source_volume_node = None
        self._volume_geometry_warning_signature = None
        if self._parameterNode:
            self._set_parameter_node_reference(_INPUT_VOLUME, None)
        self.ui.sourceVolumeSelector.setCurrentNode(None)
        self.ui.addSegmentButton.setEnabled(False)
        if count:
            slicer.util.infoDisplay(f'Removed {count} loaded volume(s).')
        else:
            slicer.util.infoDisplay('No loaded volumes to remove.')

    def _resolve_geometry_mismatch_for_volume(self, vol):
        seg = self.ui.segmentationNodeSelector.currentNode()
        if vol is None or seg is None:
            return 'keep'
        if self.logic.segmentation_matches_volume_geometry(seg, vol):
            return 'keep'

        box = qt.QMessageBox()
        box.setIcon(qt.QMessageBox.Warning)
        box.setWindowTitle('Geometry mismatch')
        box.setText(
            'The selected segmentation uses a different reference geometry '
            'than the selected volume.')
        box.setInformativeText(
            'Create a new empty segmentation for this volume, keep the current '
            'segmentation, or cancel the volume switch?')
        create_button = box.addButton('Create New Segmentation', qt.QMessageBox.AcceptRole)
        keep_button = box.addButton('Keep Current Segmentation', qt.QMessageBox.DestructiveRole)
        cancel_button = box.addButton('Cancel Switch', qt.QMessageBox.RejectRole)
        box.setDefaultButton(create_button)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is create_button:
            return 'create'
        if clicked is cancel_button:
            return 'cancel'
        return 'keep'

    def _create_and_select_segmentation_for_volume(self, vol):
        if vol is None:
            return None
        with _suppress_vtk_warnings():
            seg = self.logic.create_segmentation_for_volume(vol)
            self.ui.segmentationNodeSelector.setCurrentNode(seg)
            self.ui.segmentSelector.setCurrentNode(seg)
        return seg

    # ------------------------------------------------------------------ #
    # View / selection sync                                                #
    # ------------------------------------------------------------------ #

    def _sync_selected_nodes_to_views(self, sync_editor=True, fit=False, propagate=False):
        vol = self.ui.sourceVolumeSelector.currentNode()
        seg = self.ui.segmentationNodeSelector.currentNode()
        if vol is not None:
            self.logic.show_volume_in_slice_views(vol, fit=fit, propagate=propagate)
        if seg is not None and not seg.GetDisplayNode():
            seg.CreateDefaultDisplayNodes()
        if sync_editor and vol is not None and seg is not None:
            editor = self.logic.get_segment_editor()
            seg_id = self.ui.segmentSelector.currentSegmentID()
            self.logic.setup_editor_nodes(editor, vol, seg, seg_id)

    def _select_relative_volume(self, offset):
        nodes = self.logic.scalar_volume_nodes()
        if not nodes:
            slicer.util.warningDisplay('No source volumes are loaded.')
            return
        current = self.ui.sourceVolumeSelector.currentNode()
        current_id = current.GetID() if current is not None else None
        node_ids = [node.GetID() for node in nodes]
        if current_id in node_ids:
            idx = node_ids.index(current_id)
            target = nodes[(idx + offset) % len(nodes)]
        else:
            target = nodes[0 if offset >= 0 else -1]
        self.ui.sourceVolumeSelector.setCurrentNode(target)

    def _select_relative_segment(self, offset):
        seg = self.ui.segmentationNodeSelector.currentNode()
        if not seg:
            slicer.util.warningDisplay('No segmentation selected.')
            return
        ids = self.logic.segment_ids(seg)
        if not ids:
            slicer.util.warningDisplay('No segments are available.')
            return
        current = self.ui.segmentSelector.currentSegmentID()
        if current in ids:
            idx = ids.index(current)
            target = ids[(idx + offset) % len(ids)]
        else:
            target = ids[0 if offset >= 0 else -1]
        self.ui.segmentSelector.setCurrentSegmentID(target)

    def _toggle_current_segment_visibility(self):
        self.ui.showCurrentSegmentCheckBox.toggle()

    def _toggle_saved_segments_visibility(self):
        self.ui.showSegmentsCheckBox.toggle()

    # ------------------------------------------------------------------ #
    # Segment lifecycle sync                                               #
    # ------------------------------------------------------------------ #

    def _rewire_segmentation_observer(self, seg):
        """Keep exactly one lifecycle observer set on the active segmentation.

        Segment events live on vtkSegmentation (the inner object), not on the
        vtkMRMLSegmentationNode, so we observe GetSegmentation() directly.
        """
        new_obj = seg.GetSegmentation() if seg is not None else None
        old_obj = self._observed_seg_obj
        if old_obj is new_obj:
            return
        if old_obj is not None:
            for event, callback in (
                (old_obj.SegmentAdded, self._onSegmentAdded),
                (old_obj.SegmentRemoved, self._onSegmentRemoved),
                (old_obj.SegmentModified, self._onSegmentModified),
            ):
                self.removeObserver(old_obj, event, callback)
        self._observed_segmentation = seg
        self._observed_seg_obj      = new_obj
        self._observed_segment_ids  = self._segment_ids(seg)
        self._observed_segment_names = self._segment_names(seg)
        if new_obj is not None:
            self.addObserver(new_obj, new_obj.SegmentAdded,
                             self._onSegmentAdded)
            self.addObserver(new_obj, new_obj.SegmentRemoved,
                             self._onSegmentRemoved)
            self.addObserver(new_obj, new_obj.SegmentModified,
                             self._onSegmentModified)

    @staticmethod
    def _segment_ids(seg):
        if seg is None:
            return set()
        segmentation = seg.GetSegmentation()
        return {
            segmentation.GetNthSegmentID(i)
            for i in range(segmentation.GetNumberOfSegments())
        }

    @staticmethod
    def _segment_names(seg):
        if seg is None:
            return {}
        segmentation = seg.GetSegmentation()
        names = {}
        for i in range(segmentation.GetNumberOfSegments()):
            sid = segmentation.GetNthSegmentID(i)
            seg_obj = segmentation.GetSegment(sid)
            names[sid] = seg_obj.GetName() if seg_obj else sid
        return names

    def _onSegmentAdded(self, caller, event, callData=None):
        if self._observed_segmentation is None:
            return
        current_ids = self._segment_ids(self._observed_segmentation)
        added = set()
        if callData is not None:
            added.add(str(callData))
        added.update(current_ids - self._observed_segment_ids)
        self._observed_segment_ids = current_ids
        self._observed_segment_names = self._segment_names(self._observed_segmentation)
        for seg_id in sorted(sid for sid in added if sid in current_ids):
            self.logic.create_segment_prompt_nodes(self._observed_segmentation, seg_id)
            self._record_segment_created(self._observed_segmentation, seg_id)

    def _onSegmentRemoved(self, caller, event, callData=None):
        removed = set()
        if callData is not None:
            removed.add(str(callData))
        if self._observed_segmentation is not None:
            current_ids = self._segment_ids(self._observed_segmentation)
            removed.update(self._observed_segment_ids - current_ids)
            self._observed_segment_ids = current_ids
            current_names = self._segment_names(self._observed_segmentation)
        else:
            current_names = {}
        for seg_id in sorted(removed):
            seg_name = self._observed_segment_names.get(seg_id, seg_id)
            self._record_segment_removed_by_name(seg_id, seg_name)
        self._observed_segment_names = current_names

    def _onSegmentModified(self, caller, event, callData=None):
        """Rename markup nodes when their owning segment is renamed."""
        seg_id = str(callData) if callData is not None else None
        if not seg_id or self._observed_segmentation is None:
            return
        current_name = self._segment_name(self._observed_segmentation, seg_id)
        old_name = self._observed_segment_names.get(seg_id)
        if old_name == current_name:
            return
        self._observed_segment_names[seg_id] = current_name
        if getattr(self._recorder, 'is_active', False):
            self._recorder.record_segment_renamed(
                seg_id, old_name or seg_id, current_name or seg_id)
        self.logic.sync_prompt_node_names(self._observed_segmentation, seg_id)

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def onRecord(self, *_):
        if not self._prepare_recording_restart():
            return
        place_states = self._prompt_place_states()
        if self._recorder.is_active:
            self._recorder.stop()
        self._recorder.clear()
        vol = self._parameterNode.GetNodeReference(_INPUT_VOLUME) if self._parameterNode else None
        seg = self._parameterNode.GetNodeReference(_SEGMENTATION) if self._parameterNode else None
        self._recorder.start(
            volume_node       = vol,
            segmentation_name = seg.GetName() if seg else None,
            volume_sequences  = self._volume_sequence_metadata(),
        )
        self._audio_temp_dir = tempfile.mkdtemp(prefix='shb_audio_')
        self._audio_recorder = StandaloneAudioRecorder(
            sample_rate_hz=44100, device=self._selected_audio_device())
        try:
            self._audio_recorder.start(self._audio_temp_dir, prefix='audio')
        except Exception:
            self._audio_recorder = None
        self._recording_saved = False
        self._update_record_ui()
        self._set_prompt_place_states(place_states)

    def _prepare_recording_restart(self):
        if len(self._recorder) <= 0 or self._recording_saved:
            return True
        choice = self._prompt_unsaved_recording()
        if choice == 'discard':
            return True
        if choice == 'save':
            return self._save_recording_to_user_path()
        return False

    def _prompt_unsaved_recording(self):
        box = qt.QMessageBox()
        box.setWindowTitle('Unsaved Recording')
        box.setText('The current recording has not been saved.')
        box.setInformativeText('Save it before starting a new recording?')
        save_button = box.addButton('Save', qt.QMessageBox.AcceptRole)
        discard_button = box.addButton('Discard', qt.QMessageBox.DestructiveRole)
        cancel_button = box.addButton('Cancel', qt.QMessageBox.RejectRole)
        box.setDefaultButton(save_button)
        box.exec_()
        clicked = box.clickedButton()
        if clicked == save_button:
            return 'save'
        if clicked == discard_button:
            return 'discard'
        if clicked == cancel_button:
            return 'cancel'
        return 'cancel'

    def _onRecorderAppended(self):
        self._recording_saved = False
        self._update_record_ui()
        records = self._recorder.records
        if records and records[-1].event_type == 'press':
            # A new mouse press is an explicit interaction boundary. Any fresh
            # placed point is now eligible for real removal/displacement.
            self._arm_recorded_prompt_points_for_deletion()
        elif records and records[-1].event_type == 'release':
            self._confirm_all_pending_points()

    def _onRecordedVolumeChanged(self, node):
        if not self._recorder.is_active:
            return
        self._recorder.set_volume_node(node)
        self._recorder.record_volume_changed(
            node.GetName() if node else None,
            volume_id=node.GetID() if node else None,
            sequence_index=self._volume_sequence_index(node),
        )
        qt.QTimer.singleShot(0, self._recorder.refresh_visual_state)

    def _volume_sequence_metadata(self):
        if not hasattr(self, 'logic') or not hasattr(self.logic, 'scalar_volume_nodes'):
            return []
        result = []
        for idx, node in enumerate(self.logic.scalar_volume_nodes()):
            item = {
                'index': idx,
                'id': node.GetID(),
                'name': node.GetName(),
            }
            origin = self.logic.volume_origin(node)
            if origin is not None:
                item['origin'] = list(origin)
            image = node.GetImageData()
            if image is not None:
                item['dimensions'] = list(image.GetDimensions())
            item['spacing'] = list(node.GetSpacing())
            result.append(item)
        return result

    def _volume_sequence_index(self, node):
        if node is None:
            return None
        node_id = node.GetID()
        for idx, candidate in enumerate(self.logic.scalar_volume_nodes()):
            if candidate.GetID() == node_id:
                return idx
        return None

    def onRecordEventOnly(self, *_):
        if not self._prepare_recording_restart():
            return
        place_states = self._prompt_place_states()
        if self._recorder.is_active:
            self._recorder.stop()
        self._recorder.clear()
        vol = self._parameterNode.GetNodeReference(_INPUT_VOLUME) if self._parameterNode else None
        seg = self._parameterNode.GetNodeReference(_SEGMENTATION) if self._parameterNode else None
        self._recorder.start(
            volume_node       = vol,
            segmentation_name = seg.GetName() if seg else None,
            volume_sequences  = self._volume_sequence_metadata(),
        )
        self._audio_recorder = None
        self._audio_temp_dir = None
        self._recording_saved = False
        self._update_record_ui()
        self._set_prompt_place_states(place_states)

    def onStopRecord(self, *_):
        if not self._recorder.is_active:
            return
        self._recorder.stop()
        if self._audio_recorder is not None and self._audio_recorder.is_active:
            self._audio_recorder.stop()
        self._update_record_ui()

    def onExportRecord(self, *_):
        self._save_recording_to_user_path()

    def onRecordTestAudio(self, *_):
        if self._audio_test_active:
            self._stop_audio_test()
            return
        self._audio_test_active = True
        self._audio_test_recorder = StandaloneAudioRecorder(device=self._selected_audio_device())
        test_dir = tempfile.mkdtemp(prefix='shb_audiotest_')
        self._set_record_buttons_enabled(False)
        self.ui.recordTestAudioButton.setText('Stop Recording')
        self.ui.recordTestAudioButton.setEnabled(True)
        self.ui.playTestAudioButton.setEnabled(False)
        try:
            self._audio_test_recorder.start(test_dir, prefix='test')
        except Exception:
            self._cleanup_audio_test()
            slicer.util.warningDisplay('No microphone available for audio test.')
            return

    def onPlayTestAudio(self, *_):
        if not self._audio_test_chunks:
            return
        try:
            import wave as _wave
            import sounddevice as sd
            combined = bytearray()
            sr = self._audio_test_chunks[0].sample_rate_hz
            for chunk in self._audio_test_chunks:
                with _wave.open(chunk.path, 'rb') as w:
                    combined.extend(w.readframes(w.getnframes()))
            if not combined:
                slicer.util.warningDisplay('Test recording is empty — nothing to play.')
                return
            audio = np.frombuffer(bytes(combined), dtype='<i2').astype('float32') / 32768.0
            log.info(f'Playing test audio: {len(audio)} samples at {sr} Hz '
                     f'({len(audio)/sr:.1f}s)')
            self.ui.playTestAudioButton.setEnabled(False)
            sd.stop()
            sd.play(audio, sr, blocking=False)
            duration_ms = int(len(audio) / sr * 1000) + 500
            timer = qt.QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self.ui.playTestAudioButton.setEnabled(True))
            timer.start(duration_ms)
            self._audio_test_timer = timer
        except Exception as exc:
            slicer.util.warningDisplay(f'Audio playback failed:\n{exc}')
            self.ui.playTestAudioButton.setEnabled(True)

    def _stop_audio_test(self):
        chunks = []
        if self._audio_test_recorder is not None and self._audio_test_recorder.is_active:
            chunks = self._audio_test_recorder.stop()
        if self._audio_test_timer is not None:
            self._audio_test_timer.stop()
            self._audio_test_timer = None
        self._audio_test_active = False
        self._audio_test_recorder = None
        if chunks:
            self._audio_test_chunks = chunks
        self.ui.recordTestAudioButton.setText('Record Test')
        self._set_record_buttons_enabled(True)
        self.ui.playTestAudioButton.setEnabled(bool(self._audio_test_chunks))
        self._update_record_ui()

    def _cleanup_audio_test(self):
        self._audio_test_active = False
        self._audio_test_recorder = None
        self._audio_test_timer = None
        self.ui.recordTestAudioButton.setText('Record Test')
        self.ui.playTestAudioButton.setEnabled(bool(self._audio_test_chunks))
        self._set_record_buttons_enabled(True)
        self._update_record_ui()

    def _populate_audio_devices(self):
        cb = self.ui.audioDeviceComboBox
        cb.clear()
        cb.addItem('Default Device', -1)
        try:
            import sounddevice as sd
            for idx, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    cb.addItem(dev['name'], idx)
        except Exception:
            pass

    def _selected_audio_device(self):
        idx = self.ui.audioDeviceComboBox.currentIndex
        device = self.ui.audioDeviceComboBox.itemData(idx)
        return None if device == -1 else device

    def _set_record_buttons_enabled(self, enabled: bool):
        self.ui.recordButton.setEnabled(enabled)
        self.ui.recordEventOnlyButton.setEnabled(enabled)
        self.ui.stopRecordButton.setEnabled(enabled)
        self.ui.exportRecordButton.setEnabled(enabled)
        self.ui.recordTestAudioButton.setEnabled(enabled)
        self.ui.playTestAudioButton.setEnabled(enabled and bool(self._audio_test_chunks))

    def _save_recording_to_user_path(self):
        if self._recorder.is_active:
            self._recorder.stop()
            self._update_record_ui()
        if self._audio_recorder is not None and self._audio_recorder.is_active:
            self._audio_recorder.stop()
        path = qt.QFileDialog.getSaveFileName(None, 'Save Recording', '', 'JSON files (*.json)')
        if not path:
            return False
        if not path.endswith('.json'):
            path += '.json'
        base = path[:-5] if path.endswith('.json') else path
        try:
            self._recorder.save_to_file(path)
            self._recording_saved = True
            self._update_record_ui()
            wav_saved = False
            if self._audio_recorder is not None:
                chunks = self._audio_recorder.chunks
                if chunks:
                    merge_chunks_to_wav(chunks, base + '.wav')
                    wav_saved = True
            msg = (
                f'Recording saved:\n'
                f'  {base}_raw.json\n'
                f'  {base}.json\n'
                f'  {base}_summary.txt'
            )
            if wav_saved:
                msg += f'\n  {base}.wav'
            slicer.util.infoDisplay(msg)
            return True
        except Exception as exc:
            slicer.util.errorDisplay(f'Failed to save recording:\n{exc}')
            return False

    def _update_record_ui(self):
        if self._audio_test_active:
            return
        ui         = self.ui
        is_active  = self._recorder.is_active
        has_events = len(self._recorder) > 0

        ui.recordButton.setVisible(True)
        ui.recordButton.setEnabled(True)
        ui.recordButton.setText('Restart (Evt+Audio)' if has_events else 'Event & Audio')
        ui.recordEventOnlyButton.setVisible(True)
        ui.recordEventOnlyButton.setEnabled(True)
        ui.recordEventOnlyButton.setText('Restart (Evt)' if has_events else 'Event Only')
        ui.stopRecordButton.setVisible(is_active)
        ui.exportRecordButton.setEnabled(is_active or has_events)
        ui.recordTestAudioButton.setEnabled(not is_active)
        ui.recordTestAudioButton.setText('Record Test')
        ui.playTestAudioButton.setEnabled(not is_active and bool(self._audio_test_chunks))

        if is_active:
            status = f'Recording... ({len(self._recorder)} events)'
        elif has_events:
            status = f'Recorded: {len(self._recorder)} events'
        else:
            status = ''
        if has_events and not self._recording_saved:
            status = f'{status} (unsaved)' if status else 'Unsaved recording'
        ui.recordStatusLabel.setText(status)

    def _recorder_context(self, view_name=None) -> dict:
        seg_id = self.ui.segmentSelector.currentSegmentID()
        if self.ui.brushToolButton.isChecked():
            tool = 'brush'
        elif self.ui.eraseToolButton.isChecked():
            tool = 'erase'
        elif isinstance(self._active_handler, PointHandler):
            tool = 'point'
        else:
            tool = None
        # Resolve active slice so exported records have the active view context.
        pn = self._parameterNode
        vol = pn.GetNodeReference(_INPUT_VOLUME) if pn else None
        seg = self.ui.segmentationNodeSelector.currentNode()
        active_view = view_name or self.currentViewName
        axis, slice_idx = self.logic.active_slice_info(active_view, vol)
        seg_name = None
        if seg is not None and seg_id:
            seg_obj = seg.GetSegmentation().GetSegment(seg_id)
            seg_name = seg_obj.GetName() if seg_obj is not None else None
        if tool in ('brush', 'erase'):
            editor = self.logic.get_segment_editor()
            pn_ed  = editor.mrmlSegmentEditorNode() if editor else None
            diam = float(pn_ed.GetAttribute('BrushAbsoluteDiameter') or 10) if pn_ed else 10.0
            brush_radius_mm = diam / 2.0
        else:
            brush_radius_mm = None
        return {
            'segment_id':      seg_id,
            'seg_name':        seg_name,
            'volume_id':       vol.GetID() if vol is not None else None,
            'volume_name':     vol.GetName() if vol is not None else None,
            'segmentation_id': seg.GetID() if seg is not None else None,
            'segmentation_name': seg.GetName() if seg is not None else None,
            'tool':            tool,
            'view_name':       active_view,
            'axis':            axis,
            'slice_idx':       slice_idx,
            'brush_radius_mm': brush_radius_mm,
        }

    def _observe_prompt_node_for_recording(self, node, is_negative):
        if node is None:
            return
        key = (node.GetID(), bool(is_negative))
        if key in self._recorded_prompt_node_ids:
            return
        defined_event = slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent

        @vtk.calldata_type(vtk.VTK_INT)
        def on_point_defined(caller, event_id, callData=None, neg=bool(is_negative)):
            self._onPromptPointDefinedForRecording(caller, neg, callData)

        self.addObserver(
            node, defined_event,
            on_point_defined,
        )
        for event_name, phase in (
            ('PointStartInteractionEvent', 'start'),
            ('PointModifiedEvent', 'move'),
            ('PointEndInteractionEvent', 'end'),
        ):
            event = getattr(slicer.vtkMRMLMarkupsNode, event_name, None)
            if event is None:
                continue

            @vtk.calldata_type(vtk.VTK_INT)
            def on_point_interaction(caller, event_id, callData=None,
                                     neg=bool(is_negative), ph=phase):
                self._onPromptPointDragForRecording(caller, neg, ph, callData)

            self.addObserver(
                node, event,
                on_point_interaction,
            )
        removed_event = getattr(slicer.vtkMRMLMarkupsNode, 'PointRemovedEvent', None)
        if removed_event is not None:
            @vtk.calldata_type(vtk.VTK_INT)
            def on_point_removed(caller, event_id, callData=None,
                                 neg=bool(is_negative)):
                self._onPromptPointRemovedForRecording(caller, neg, callData)

            self.addObserver(
                node, removed_event,
                on_point_removed,
            )
        self._recorded_prompt_node_ids.add(key)

    def _onPromptPointDefinedForRecording(self, node, is_negative, callData=None):
        if not self._recorder.is_active:
            return

        seg_id = self._segment_id_for_prompt_node(node)
        if not seg_id:
            raise RuntimeError(
                f'_onPromptPointDefinedForRecording: node {node.GetID()} '
                'has no segment tag — was it created outside create_segment_prompt_nodes?')
        idx = self._control_point_index_from_call_data(node, callData)
        if idx < 0:
            idx = self._last_defined_control_point_index(node)
        if idx < 0:
            return
        ras = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(idx, ras)
        point_id = self._control_point_id(node, idx)
        point_name = self._control_point_name(node, idx)
        pend_key = (node.GetID(), point_id)
        self._pending_point_confirmations[pend_key] = {
            'segment_id': seg_id,
            'ras': list(ras),
            'is_negative': bool(is_negative),
            'point_index': idx,
            'point_id': point_id,
            'point_name': point_name,
            'view_name': self.currentViewName,
            'node': node,
        }
        if not (self._recorder._active_mouse_press or self._qt_left_button_is_down()):
            self._confirm_pending_point(node, idx)

    def _onPromptPointDragForRecording(self, node, is_negative, phase, callData=None):
        if not self._recorder.is_active:
            return
        # Throttle 'move' events before any node access — PointModifiedEvent fires
        # very frequently and most are rate-limited.  Tests also rely on this guard
        # happening before GetID()/GetNumberOfControlPoints().
        if phase == 'move' and not self._recorder.should_sample_point_drag(phase):
            return
        # Use event callData only. Falling back to the last-defined index tracks
        # the wrong point when the user drags anything except the highest index.
        idx = self._drag_control_point_index_from_call_data(node, callData)
        key = (node.GetID(), bool(is_negative))
        if phase == 'start':
            if idx < 0 or not self._is_control_point_defined(node, idx):
                # If start cannot identify a point, PointModifiedEvent can still
                # auto-start once it reports a concrete control-point index.
                return
            pend_key = (node.GetID(), self._control_point_id(node, idx))
            if pend_key in self._pending_point_confirmations:
                # Just-created point: placement motion, not a relocation.
                return
        else:
            if key not in self._active_point_drags:
                if phase == 'move' and idx >= 0:
                    # Fallback for missed start events: PointModifiedEvent also
                    # carries integer point-index callData when observed correctly.
                    mouse_pressed = (self._recorder._active_mouse_press or
                                     self._qt_left_button_is_down())
                    pend_key = (node.GetID(), self._control_point_id(node, idx))
                    if (mouse_pressed and
                            self._is_control_point_defined(node, idx) and
                            pend_key not in self._pending_point_confirmations):
                        ras_start = [0.0, 0.0, 0.0]
                        node.GetNthControlPointPositionWorld(idx, ras_start)
                        self._active_point_drags[key] = {
                            'idx': idx,
                            'point_id': self._control_point_id(node, idx),
                            'start_ras': list(ras_start),
                        }
                        # fall through to record this move
                    else:
                        return
                elif phase == 'end':
                    self._confirm_pending_point(node, idx)
                    return
                else:
                    return
            drag = self._active_point_drags[key]
            idx = drag['idx'] if isinstance(drag, dict) else drag
        if not self._recorder.should_sample_point_drag(phase):
            return
        if idx >= node.GetNumberOfControlPoints():
            return
        if not self._is_control_point_defined(node, idx):
            return
        seg_id = self._segment_id_for_prompt_node(node)
        if not seg_id:
            raise RuntimeError(
                f'_onPromptPointDragForRecording: node {node.GetID()} '
                'has no segment tag — was it created outside create_segment_prompt_nodes?')
        ras = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(idx, ras)
        point_id = self._control_point_id(node, idx)
        point_name = self._control_point_name(node, idx)
        if phase == 'start':
            self._active_point_drags[key] = {
                'idx': idx,
                'point_id': point_id,
                'start_ras': list(ras),
            }
        elif phase == 'end':
            drag = self._active_point_drags.get(key)
            start_ras = drag.get('start_ras') if isinstance(drag, dict) else None
            if start_ras is not None and _ras_positions_close(start_ras, ras):
                self._active_point_drags.pop(key, None)
                return
        self._cache_recorded_prompt_point(
            node, seg_id, ras, point_id, point_name)
        self._recorder.record_point_drag(
            phase, seg_id, ras, is_negative, view_name=self.currentViewName,
            point_index=idx, point_id=point_id, point_name=point_name)
        if phase == 'end':
            self._active_point_drags.pop(key, None)

    def _confirm_pending_point(self, node, idx):
        if idx < 0:
            idx = self._last_defined_control_point_index(node)
        if idx < 0 or idx >= node.GetNumberOfControlPoints():
            return
        if not self._is_control_point_defined(node, idx):
            return
        point_id = self._control_point_id(node, idx)
        pend_key = (node.GetID(), point_id)
        pending = self._pending_point_confirmations.pop(pend_key, None)
        if not pending:
            return
        ras = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(idx, ras)
        pending['ras'] = list(ras)
        pending['point_id'] = point_id
        pending['point_name'] = self._control_point_name(node, idx)
        self._cache_recorded_prompt_point(
            node, pending['segment_id'], ras,
            pending['point_id'], pending['point_name'], fresh_placement=True)
        self._recently_placed[(node.GetID(), point_id)] = True
        self._recorder.record_point_placed(
            pending['segment_id'], pending['ras'], pending['is_negative'],
            view_name=pending['view_name'], point_index=pending['point_index'],
            point_id=pending['point_id'], point_name=pending['point_name'])

    def _schedule_pending_point_confirmation(self, node, is_negative, idx):
        # Kept for tests/older call sites. Confirmation is event-driven:
        # point-defined after release confirms immediately, otherwise the
        # recorder's release boundary confirms pending placements.
        return

    def _confirm_all_pending_points(self):
        for pending in list(self._pending_point_confirmations.values()):
            node = pending.get('node')
            if node is None:
                continue
            self._confirm_pending_point(node, pending['point_index'])

    def _onPromptPointRemovedForRecording(self, node, is_negative, callData=None):
        if not self._recorder.is_active:
            return
        node_id = node.GetID()

        # Identify the removed point by diffing the cache against the current node.
        # PointRemovedEvent fires after removal+renumber, so callData's index maps
        # to a survivor, not the deleted point. The diff is the only stable approach.
        current_pt_ids = {
            node.GetNthControlPointID(i)
            for i in range(node.GetNumberOfControlPoints())
        }
        removed_pt_id = next(
            (ptid for (nid, ptid) in self._recorded_prompt_point_cache
             if nid == node_id and ptid not in current_pt_ids),
            None,
        )
        if removed_pt_id is None:
            # Also check pending confirmations (point removed before being confirmed)
            for pend_key in list(self._pending_point_confirmations):
                if pend_key[0] == node_id and pend_key[1] not in current_pt_ids:
                    self._pending_point_confirmations.pop(pend_key, None)
                    return
            return

        cache_key = (node_id, removed_pt_id)

        # Suppress if this point is the subject of an active drag (null-tool
        # remove+recreate, or PointStartInteractionEvent-tracked drag).
        drag_key = (node_id, bool(is_negative))
        active_drag = self._active_point_drags.get(drag_key)
        if isinstance(active_drag, dict) and active_drag.get('point_id') == removed_pt_id:
            return

        cached = self._recorded_prompt_point_cache.pop(cache_key, None)
        if not cached:
            return

        # Check the real Qt button state in addition to the VTK-event-based flag,
        # because Slicer's PointRemovedEvent can fire before the VTK
        # LeftButtonPressEvent reaches our observer priority.
        mouse_is_pressed = (self._recorder._active_mouse_press or
                            self._qt_left_button_is_down())
        if mouse_is_pressed:
            # null-tool drag: Slicer removes then recreates the point.
            # Re-insert cache so drag recording still works; record with flag so
            # interpreter can suppress this from the compact log.
            self._recorded_prompt_point_cache[cache_key] = cached
            self._recorder.record_point_removed(
                cached['segment_id'], cached.get('ras'), bool(is_negative),
                view_name=self.currentViewName,
                point_id=cached.get('point_id'),
                point_name=cached.get('point_name'),
                mouse_pressed=True)
            return
        # Suppress Slicer's internal remove/recreate signal after placement until
        # the next explicit mouse press arms the point for real removal.
        if cached.get('fresh_placement') or self._recently_placed.get(cache_key):
            self._recorded_prompt_point_cache[cache_key] = cached
            return
        self._recorder.record_point_removed(
            cached['segment_id'], cached.get('ras'), bool(is_negative),
            view_name=self.currentViewName,
            point_id=cached.get('point_id'),
            point_name=cached.get('point_name'))

    def _cache_recorded_prompt_point(self, node, segment_id, ras, point_id,
                                     point_name=None, fresh_placement=False):
        cache_key = (node.GetID(), point_id)
        self._recorded_prompt_point_cache[cache_key] = {
            'segment_id': segment_id,
            'ras': list(ras),
            'point_id': point_id,
            'point_name': point_name,
        }
        if fresh_placement:
            self._recorded_prompt_point_cache[cache_key]['fresh_placement'] = True

    def _arm_recorded_prompt_points_for_deletion(self):
        self._recently_placed.clear()
        for cached in self._recorded_prompt_point_cache.values():
            cached.pop('fresh_placement', None)

    @staticmethod
    def _control_point_index_from_call_data(node, callData=None):
        if callData is not None:
            idx = int(str(callData))
            if 0 <= idx < node.GetNumberOfControlPoints():
                return idx
        return SegmentHumanBodyWidget._last_defined_control_point_index(node)

    @staticmethod
    def _control_point_id(node, idx):
        return node.GetNthControlPointID(idx)

    @staticmethod
    def _control_point_name(node, idx):
        value = node.GetNthControlPointLabel(idx)
        return str(value) if value else SegmentHumanBodyWidget._control_point_id(node, idx)

    @staticmethod
    def _is_control_point_defined(node, idx):
        return node.GetNthControlPointPositionStatus(idx) == slicer.vtkMRMLMarkupsNode.PositionDefined

    @staticmethod
    def _last_defined_control_point_index(node):
        defined = slicer.vtkMRMLMarkupsNode.PositionDefined
        for idx in range(node.GetNumberOfControlPoints() - 1, -1, -1):
            if node.GetNthControlPointPositionStatus(idx) == defined:
                return idx
        return -1

    @staticmethod
    def _drag_control_point_index_from_call_data(node, callData=None):
        """Return the control point index from callData, or -1 if unavailable.

        Unlike _control_point_index_from_call_data, does NOT fall back to the
        last-defined index — that fallback gives the wrong point when callData
        is absent for PointStartInteractionEvent.
        """
        if callData is not None:
            try:
                idx = int(str(callData))
            except (ValueError, TypeError):
                return -1
            if 0 <= idx < node.GetNumberOfControlPoints():
                return idx
        return -1

    @staticmethod
    def _qt_left_button_is_down():
        try:
            return bool(qt.QApplication.mouseButtons() & qt.Qt.LeftButton)
        except Exception:
            return False

    def _segment_id_for_prompt_node(self, node):
        seg = self.ui.segmentationNodeSelector.currentNode()
        if not seg or not node:
            return ''
        node_id = node.GetID()
        segmentation = seg.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            sid = segmentation.GetNthSegmentID(i)
            seg_obj = segmentation.GetSegment(sid)
            for tag in (_POS_TAG, _NEG_TAG):
                value = vtk.reference('')
                seg_obj.GetTag(tag, value)
                if str(value) == node_id:
                    return sid
        return ''

    # ------------------------------------------------------------------ #
    # Segment Editor effects                                               #
    # ------------------------------------------------------------------ #

    def _deactivateEffect(self):
        handler = self._active_handler
        if handler is not None:
            handler.detach(self)
        else:
            # Fallback: no handler registered — clean up the editor directly.
            self._returnEffectsOptionsFrame()
            editor = self.logic.get_segment_editor()
            if editor:
                editor.setActiveEffectByName('')
            for btn in (self.ui.brushToolButton, self.ui.eraseToolButton):
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Segment management                                                   #
    # ------------------------------------------------------------------ #

    def _onAddSegment(self, *_):
        pn = self._parameterNode
        vol, seg = self.logic.getVolumeAndSegmentation(pn)
        if not vol:
            vol = self.ui.sourceVolumeSelector.currentNode()
        if not seg:
            seg = self.ui.segmentationNodeSelector.currentNode()
        # Auto-create segmentation when a volume is present but none exists yet.
        if vol and not seg:
            with _suppress_vtk_warnings():
                seg = self.logic.create_segmentation_for_volume(vol)
                self.ui.segmentationNodeSelector.blockSignals(True)
                self.ui.segmentationNodeSelector.setCurrentNode(seg)
                self.ui.segmentationNodeSelector.blockSignals(False)
                self.ui.segmentSelector.setCurrentNode(seg)
                self.ui.addSegmentButton.setEnabled(True)
            if pn:
                pn.SetNodeReferenceID(_SEGMENTATION, seg.GetID())
        if not seg:
            slicer.util.warningDisplay('Please select a volume first.')
            return
        self._rewire_segmentation_observer(seg)
        # Cache the active handler before creation, detach cleanly, then restore.
        # Suppress place-mode signals during creation so setCurrentNode calls inside
        # _set_prompt_nodes don't spuriously re-activate the wrong widget.
        prev_was_point  = isinstance(self._active_handler, PointHandler)
        prev_stroke_cls = type(self._active_handler) if isinstance(self._active_handler, StrokeHandler) else None
        prev_prompt_widget = self._active_prompt_widget if prev_was_point else None
        if prev_was_point and prev_prompt_widget is None:
            raise RuntimeError('_onAddSegment: PointHandler active with no _active_prompt_widget')
        if self._active_handler is not None:
            self._active_handler.detach(self)
        self._suppressing_place_mode = True
        try:
            with _suppress_vtk_warnings():
                new_id = self.logic.add_segment(seg)
                if new_id:
                    self.ui.segmentSelector.setCurrentSegmentID(new_id)
        finally:
            self._suppressing_place_mode = False
            if prev_stroke_cls is not None:
                prev_stroke_cls().attach(self)
            elif prev_was_point:
                for child in prev_prompt_widget.findChildren(qt.QWidget):
                    if hasattr(child, 'setPlaceModeEnabled'):
                        child.setPlaceModeEnabled(True)
                        break
            else:
                self._ensure_current_prompt_nodes()

    def _onRemoveSegment(self, *_):
        seg   = self.ui.segmentationNodeSelector.currentNode()
        segID = self.ui.segmentSelector.currentSegmentID()
        if not seg or not segID:
            slicer.util.warningDisplay('No segment selected.')
            return
        next_id = self._segment_id_before(seg, segID)
        self._record_action('remove_segment')
        # Cache the active handler, detach before node rewiring, then restore.
        # Removing markup nodes from the scene while they are wired to the prompt
        # widgets fires activeMarkupsFiducialPlaceModeChanged, which would
        # otherwise install a PointHandler and discard whatever tool was active.
        prev_was_point  = isinstance(self._active_handler, PointHandler)
        prev_stroke_cls = type(self._active_handler) if isinstance(self._active_handler, StrokeHandler) else None
        prev_prompt_widget = self._active_prompt_widget if prev_was_point else None
        if prev_was_point and prev_prompt_widget is None:
            raise RuntimeError('_onRemoveSegment: PointHandler active with no _active_prompt_widget')
        if self._active_handler is not None:
            self._active_handler.detach(self)
        self._suppressing_place_mode = True
        try:
            with _suppress_vtk_warnings():
                # Disconnect widgets before deleting the nodes.  If nodes are
                # deleted while the widget still observes them, Slicer's C++
                # qSlicerSimpleMarkupsWidget calls setPlaceModeEnabled(True)
                # internally to "restart" placement — bypassing our Python
                # suppression flag and activating the interaction node.
                self._set_prompt_nodes(None, None)
                self.logic.delete_segment_prompt_nodes(seg, segID)
                self.logic.remove_segment(seg, segID)
                if next_id:
                    self.ui.segmentSelector.setCurrentSegmentID(next_id)
        finally:
            self._suppressing_place_mode = False
            if prev_stroke_cls is not None:
                prev_stroke_cls().attach(self)
            elif prev_was_point:
                if next_id:
                    for child in prev_prompt_widget.findChildren(qt.QWidget):
                        if hasattr(child, 'setPlaceModeEnabled'):
                            child.setPlaceModeEnabled(True)
                            break
            else:
                self._ensure_current_prompt_nodes()

    @staticmethod
    def _segment_id_before(seg_node, segment_id):
        segmentation = seg_node.GetSegmentation()
        ids = [
            segmentation.GetNthSegmentID(i)
            for i in range(segmentation.GetNumberOfSegments())
        ]
        if segment_id not in ids:
            return ''
        idx = ids.index(segment_id)
        if idx > 0:
            return ids[idx - 1]
        return ids[1] if len(ids) > 1 else ''

    def _onSegmentIDChanged(self, segmentID):
        seg = self.ui.segmentationNodeSelector.currentNode()
        if not segmentID or not seg:
            # No segment selected — clear the prompt widgets.
            self._set_prompt_nodes(None, None)
            return
        if self._recorder.is_active and hasattr(self._recorder, 'record_segment_selected'):
            self._recorder.record_segment_selected(
                segmentID, self._segment_name(seg, segmentID))
        pos_node, neg_node = self.logic.create_segment_prompt_nodes(seg, segmentID)
        self._set_prompt_nodes(pos_node, neg_node)

        editor = self.logic.get_segment_editor()
        if editor and editor.currentSegmentID() != segmentID:
            editor.setCurrentSegmentID(segmentID)
        self.logic.set_saved_segments_visibility(seg, segmentID,
                                                 self._savedSegmentsVisible)
        self._currentSegmentVisible = True
        self.logic.set_segment_visibility(seg, segmentID, True)
        self.ui.showCurrentSegmentCheckBox.blockSignals(True)
        self.ui.showCurrentSegmentCheckBox.setChecked(True)
        self.ui.showCurrentSegmentCheckBox.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Brush / Erase tools                                                  #
    # ------------------------------------------------------------------ #

    def _applyOverwriteMode(self):
        pn_ed = slicer.mrmlScene.GetSingletonNode('SegmentEditor', 'vtkMRMLSegmentEditorNode')
        if not pn_ed:
            return
        cls   = slicer.vtkMRMLSegmentEditorNode
        modes = [cls.OverwriteNone, cls.OverwriteVisibleSegments, cls.OverwriteAllSegments]
        idx   = self.ui.overwriteModeDropdown.currentIndex
        mode  = modes[idx] if 0 <= idx < len(modes) else cls.OverwriteNone
        pn_ed.SetOverwriteMode(mode)

    def _onOverwriteModeChanged(self, _index):
        self._applyOverwriteMode()

    def _onPlaceModeChanged(self, active, src_widget):
        if self._suppressing_place_mode:
            return
        if active:
            self._deactivateEffect()
            self._active_prompt_widget = src_widget
            PointHandler().attach(self)
        elif isinstance(self._active_handler, PointHandler):
            self._active_handler.detach(self)
            self._active_prompt_widget = None

    def _onStrokeToggled(self, handler_cls, checked):
        if checked:
            handler_cls().attach(self)
            self._applyOverwriteMode()
        elif isinstance(self._active_handler, handler_cls):
            self._active_handler.detach(self)

    def _onBrushToggled(self, checked):
        self._onStrokeToggled(BrushHandler, checked)

    def _onEraseToggled(self, checked):
        self._onStrokeToggled(EraseHandler, checked)

    def _borrowEffectsOptionsFrame(self):
        """Move the Segment Editor's EffectsOptionsFrame into our panel.

        The frame already contains the native Paint/Erase options (Diameter
        slider, Absolute, Sphere, Edit in 3D view, Color Smudge) and updates
        live because it IS Slicer's own control.  We keep the original parent
        so _returnEffectsOptionsFrame() can put it back when we exit.
        """
        editor = self.logic.get_segment_editor()
        if not editor:
            return
        if self._eof_widget is None:
            eof = editor.findChild(qt.QWidget, 'EffectsOptionsFrame')
            if eof is None:
                log.warning('[BrushOptions] EffectsOptionsFrame not found')
                return
            self._eof_widget      = eof
            self._eof_orig_parent = eof.parentWidget()
        container = self.ui.brushOptionsContainer
        container.layout().addWidget(self._eof_widget)
        self._eof_widget.setVisible(True)
        container.setVisible(True)

    def _returnEffectsOptionsFrame(self):
        """Return the EffectsOptionsFrame to the editor widget."""
        if self._eof_widget is None:
            return
        self.ui.brushOptionsContainer.setVisible(False)
        if self._eof_orig_parent is not None:
            self._eof_orig_parent.layout().addWidget(self._eof_widget)
        self._eof_widget      = None
        self._eof_orig_parent = None

    # ------------------------------------------------------------------ #
    # Undo / Redo — delegated to the Segment Editor's built-in stack      #
    # ------------------------------------------------------------------ #

    def _onUndo(self):
        self._record_action('onUndo')
        editor = self.logic.get_segment_editor()
        if editor:
            editor.undo()

    def _onRedo(self):
        self._record_action('onRedo')
        editor = self.logic.get_segment_editor()
        if editor:
            editor.redo()

    # ------------------------------------------------------------------ #
    # Window / Level                                                       #
    # ------------------------------------------------------------------ #

    def _syncPair(self, target, value):
        target.blockSignals(True)
        target.setValue(value)
        target.blockSignals(False)

    def _syncWLFromVolume(self, vol):
        """Pre-fill W/L sliders from the volume's current display settings."""
        window, level = self.logic.get_window_level(vol)
        if window is None:
            return
        for widget, value in (
            (self.ui.windowSlider,  window),
            (self.ui.windowSpinBox, window),
            (self.ui.levelSlider,   level),
            (self.ui.levelSpinBox,  level),
        ):
            widget.blockSignals(True)
            widget.setValue(max(widget.minimum, min(widget.maximum, value)))
            widget.blockSignals(False)

    def _onWindowSliderChanged(self, value):
        self._syncPair(self.ui.windowSpinBox, value)
        self._onApplyWindowLevel()

    def _onWindowSpinBoxChanged(self, value):
        self._syncPair(self.ui.windowSlider, value)
        self._onApplyWindowLevel()

    def _onLevelSliderChanged(self, value):
        self._syncPair(self.ui.levelSpinBox, value)
        self._onApplyWindowLevel()

    def _onLevelSpinBoxChanged(self, value):
        self._syncPair(self.ui.levelSlider, value)
        self._onApplyWindowLevel()

    def _onApplyWindowLevel(self, *_):
        vol = self.ui.sourceVolumeSelector.currentNode()
        self.logic.apply_window_level(vol,
                                      self.ui.windowSpinBox.value,
                                      self.ui.levelSpinBox.value)

    # ------------------------------------------------------------------ #
    # Recording helpers                                                    #
    # ------------------------------------------------------------------ #

    def _record_action(self, name: str):
        if self._recorder.is_active:
            self._recorder.record_action(name)

    @staticmethod
    def _segment_name(seg_node, segment_id):
        seg_obj = seg_node.GetSegmentation().GetSegment(segment_id)
        if seg_obj is None:
            raise RuntimeError(
                f'_segment_name: segment {segment_id!r} not found in segmentation node')
        return seg_obj.GetName()

    def _record_segment_created(self, seg_node, segment_id):
        if self._recorder.is_active and segment_id:
            self._recorder.record_segment_created(
                segment_id, self._segment_name(seg_node, segment_id))

    def _record_segment_removed(self, seg_node, segment_id):
        if self._recorder.is_active and segment_id:
            self._recorder.record_segment_removed(
                segment_id, self._segment_name(seg_node, segment_id))

    def _record_segment_removed_by_name(self, segment_id, seg_name):
        if self._recorder.is_active and segment_id:
            self._recorder.record_segment_removed(segment_id, seg_name or segment_id)

    # ------------------------------------------------------------------ #
    # Segment visibility                                                   #
    # ------------------------------------------------------------------ #

    def _onToggleCurrentSegment(self, visible):
        self._currentSegmentVisible = visible
        seg   = self.ui.segmentationNodeSelector.currentNode()
        segID = self.ui.segmentSelector.currentSegmentID()
        self.logic.set_segment_visibility(seg, segID, visible)

    def _onToggleSavedSegments(self, visible):
        self._savedSegmentsVisible = visible
        seg   = self.ui.segmentationNodeSelector.currentNode()
        segID = self.ui.segmentSelector.currentSegmentID()
        self.logic.set_saved_segments_visibility(seg, segID, visible)


#
# Logic
#

class SegmentHumanBodyLogic(ScriptedLoadableModuleLogic):
    """Business logic for the native-editor-wrapper.

    Zero references to self.ui or Qt widget state — only MRML nodes and
    Slicer services.  The Widget reads UI state and calls these methods.
    """

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------ #
    # Scene / view helpers                                                 #
    # ------------------------------------------------------------------ #

    def scalar_volume_nodes(self):
        nodes = []
        collection = slicer.mrmlScene.GetNodesByClass('vtkMRMLScalarVolumeNode')
        collection.UnRegister(None)
        for i in range(collection.GetNumberOfItems()):
            nodes.append(collection.GetItemAsObject(i))
        return nodes

    def clear_scalar_volume_nodes(self):
        nodes = list(self.scalar_volume_nodes())
        for node in nodes:
            slicer.mrmlScene.RemoveNode(node)
        return len(nodes)

    def replace_duplicate_scalar_volume_nodes(self):
        """Keep the newest node for each loaded file path and remove older duplicates."""
        by_key = {}
        replacements = {}
        for node in self.scalar_volume_nodes():
            key = self.volume_storage_key(node)
            if key is None:
                continue
            by_key.setdefault(key, []).append(node)
        for key, nodes in by_key.items():
            if len(nodes) < 2:
                continue
            keep = nodes[-1]
            preserved_name = self.volume_filename(keep) or self.volume_filename(nodes[0]) or nodes[0].GetName()
            for old in nodes[:-1]:
                slicer.mrmlScene.RemoveNode(old)
            keep.SetName(preserved_name)
            replacements[key] = keep
        return replacements

    def normalize_scalar_volume_names_from_filenames(self):
        for node in self.scalar_volume_nodes():
            filename = self.volume_filename(node)
            if filename:
                node.SetName(filename)

    @staticmethod
    def volume_storage_key(vol):
        if vol is None:
            return None
        storage = vol.GetStorageNode()
        if storage is None:
            return None
        filename = storage.GetFileName()
        if not filename:
            return None
        return filename.replace('\\', '/').lower()

    @staticmethod
    def volume_filename(vol):
        if vol is None:
            return None
        storage = vol.GetStorageNode()
        if storage is None:
            return None
        filename = storage.GetFileName()
        if not filename:
            return None
        return filename.replace('\\', '/').rsplit('/', 1)[-1]

    def volume_geometry_signature(self, vol):
        if vol is None:
            return None
        image = vol.GetImageData()
        dims = image.GetDimensions() if image is not None else None
        spacing = tuple(round(float(value), 4) for value in vol.GetSpacing())
        matrix = vtk.vtkMatrix4x4()
        vol.GetIJKToRASMatrix(matrix)
        orientation = []
        for col in range(3):
            length = sum(matrix.GetElement(row, col) ** 2 for row in range(3)) ** 0.5
            if length == 0:
                orientation.extend([0.0, 0.0, 0.0])
            else:
                orientation.extend(
                    round(float(matrix.GetElement(row, col) / length), 4)
                    for row in range(3)
                )
        return repr((dims, spacing, tuple(orientation)))

    def volume_reference_geometry_string(self, vol):
        if vol is None:
            return None
        try:
            return slicer.vtkSlicerSegmentationsModuleLogic.GetReferenceImageGeometryParameterFromVolumeNode(vol)
        except Exception:
            return None

    def segmentation_reference_geometry_string(self, seg):
        if seg is None:
            return None
        try:
            name = slicer.vtkSegmentationConverter.GetReferenceImageGeometryParameterName()
            return seg.GetSegmentation().GetConversionParameter(name)
        except Exception:
            return None

    def segmentation_matches_volume_geometry(self, seg, vol):
        if seg is None or vol is None:
            return True
        vol_sig = self.volume_geometry_signature(vol)
        seg_sig = seg.GetAttribute(_GEOMETRY_SIGNATURE_ATTR)
        if seg_sig and vol_sig:
            return seg_sig == vol_sig

        return True

    def normalize_volume_origin_from_compatible_scene_volume(self, vol):
        if vol is None:
            return
        current_origin = self.volume_origin(vol)
        if current_origin is None or not self.origin_is_zero(current_origin):
            return
        signature = self.volume_geometry_signature(vol)
        if signature is None:
            return
        reference = self.first_nonzero_origin_volume_for_signature(signature, vol)
        if reference is None:
            return
        reference_origin = self.volume_origin(reference)
        if reference_origin is None or self.origin_is_zero(reference_origin):
            return
        vol.SetOrigin(reference_origin)
        vol.Modified()

    def normalize_compatible_scene_volume_origins(self):
        groups = {}
        for node in self.scalar_volume_nodes():
            signature = self.volume_geometry_signature(node)
            if signature is None:
                continue
            groups.setdefault(signature, []).append(node)

        for nodes in groups.values():
            reference_origin = None
            for node in nodes:
                origin = self.volume_origin(node)
                if origin is not None and not self.origin_is_zero(origin):
                    reference_origin = origin
                    break
            if reference_origin is None:
                continue
            for node in nodes:
                origin = self.volume_origin(node)
                if origin is not None and self.origin_is_zero(origin):
                    node.SetOrigin(reference_origin)
                    node.Modified()

    def scene_volume_geometry_statistics(self):
        nodes = self.scalar_volume_nodes()
        groups = {}
        for node in nodes:
            signature = self.volume_geometry_signature(node)
            if signature is None:
                continue
            groups.setdefault(signature, []).append(node)
        lines = []
        warning_signature_parts = []
        for idx, (signature, group_nodes) in enumerate(groups.items(), start=1):
            origin_counts = self._origin_count_summary(group_nodes)
            files = '; '.join(self.volume_display_path(node) for node in group_nodes)
            geom = self.volume_geometry_summary(group_nodes[0], include_name=False)
            lines.append(
                f'Group {idx}: {len(group_nodes)} volume(s); {geom}; '
                f'{origin_counts}; files: {files}')
            warning_signature_parts.append(
                f'{signature}:{",".join(node.GetID() for node in group_nodes)}')
        return {
            'group_count': len(groups),
            'summary': '\n'.join(lines) if lines else 'No scalar volumes loaded.',
            'signature': '|'.join(warning_signature_parts),
        }

    def _origin_count_summary(self, nodes):
        zero_count = 0
        nonzero_count = 0
        for node in nodes:
            origin = self.volume_origin(node)
            if origin is None:
                continue
            if self.origin_is_zero(origin):
                zero_count += 1
            else:
                nonzero_count += 1
        return f'origins: {nonzero_count} non-zero, {zero_count} zero'

    def volume_geometry_summary(self, vol, include_name=True):
        if vol is None:
            return '<none>'
        image = vol.GetImageData()
        dims = image.GetDimensions() if image is not None else None
        spacing = tuple(round(float(value), 4) for value in vol.GetSpacing())
        orientation = self.volume_orientation_summary(vol)
        prefix = f'{vol.GetName()} ' if include_name else ''
        return f'{prefix}shape={dims} spacing={spacing} orientation={orientation}'

    def volume_display_path(self, vol):
        if vol is None:
            return '<none>'
        storage = vol.GetStorageNode()
        if storage is not None:
            filename = storage.GetFileName()
            if filename:
                return filename
        return vol.GetName()

    def volume_orientation_summary(self, vol):
        matrix = vtk.vtkMatrix4x4()
        vol.GetIJKToRASMatrix(matrix)
        columns = []
        for col in range(3):
            length = sum(matrix.GetElement(row, col) ** 2 for row in range(3)) ** 0.5
            if length == 0:
                columns.append((0.0, 0.0, 0.0))
            else:
                columns.append(tuple(
                    round(float(matrix.GetElement(row, col) / length), 3)
                    for row in range(3)
                ))
        return columns

    def first_nonzero_origin_volume_for_signature(self, signature, exclude_node=None):
        for node in self.scalar_volume_nodes():
            if node is exclude_node:
                continue
            if self.volume_geometry_signature(node) != signature:
                continue
            origin = self.volume_origin(node)
            if origin is not None and not self.origin_is_zero(origin):
                return node
        return None

    @staticmethod
    def volume_origin(vol):
        if vol is None:
            return None
        try:
            return tuple(float(value) for value in vol.GetOrigin())
        except Exception:
            return None

    @staticmethod
    def origin_is_zero(origin, tolerance=1e-4):
        return all(abs(float(value)) <= tolerance for value in origin)

    def show_volume_in_slice_views(self, vol, fit=False, propagate=False):
        if vol is None:
            return
        app_logic = slicer.app.applicationLogic()
        selection = app_logic.GetSelectionNode()
        if selection is not None:
            selection.SetReferenceActiveVolumeID(vol.GetID())
        if propagate:
            app_logic.PropagateVolumeSelection(0)

        lm = slicer.app.layoutManager()
        if lm is None:
            return
        for view_name in ('Red', 'Green', 'Yellow'):
            widget = lm.sliceWidget(view_name)
            if widget is None:
                continue
            composite = widget.mrmlSliceCompositeNode()
            if composite is not None and composite.GetBackgroundVolumeID() != vol.GetID():
                composite.SetBackgroundVolumeID(vol.GetID())
            logic = widget.sliceLogic() if fit else None
            if logic is not None:
                logic.FitSliceToAll()

    def segment_ids(self, seg_node):
        if not seg_node:
            return []
        segmentation = seg_node.GetSegmentation()
        return [
            segmentation.GetNthSegmentID(i)
            for i in range(segmentation.GetNumberOfSegments())
        ]

    # ------------------------------------------------------------------ #
    # Segment Editor access                                                #
    # ------------------------------------------------------------------ #

    def get_segment_editor(self):
        """Return the shared qMRMLSegmentEditorWidget."""
        return slicer.modules.segmenteditor.widgetRepresentation().self().editor

    def setup_editor_nodes(self, editor, vol, seg, segment_id=None):
        """Point the Segment Editor at vol/seg/segment_id.

        Only fires the heavyweight setters when the value actually changed so
        that Slicer's slice-refitting pipeline is not re-queued on every click.
        """
        if not editor or not vol or not seg:
            return
        if editor.segmentationNode() is not seg:
            editor.setSegmentationNode(seg)
        if editor.sourceVolumeNode() is not vol:
            editor.setSourceVolumeNode(vol)
        editor.setUndoEnabled(True)
        editor.setMaximumNumberOfUndoStates(50)
        if segment_id and editor.currentSegmentID() != segment_id:
            editor.setCurrentSegmentID(segment_id)

    # ------------------------------------------------------------------ #
    # Node access                                                          #
    # ------------------------------------------------------------------ #

    def getVolumeAndSegmentation(self, parameterNode):
        """Return (volumeNode, segmentationNode) from *parameterNode*."""
        if parameterNode is None:
            return None, None
        vol = parameterNode.GetNodeReference(_INPUT_VOLUME)
        seg = parameterNode.GetNodeReference(_SEGMENTATION)
        return vol, seg

    def create_segment_prompt_nodes(self, seg_node, segment_id):
        """Create pos/neg markup nodes for *segment_id* and store IDs in segment tags.

        Returns ``(pos_node, neg_node)``.  Idempotent: if tags already point to
        valid scene nodes, those nodes are returned without creating new ones.
        """
        existing = self.get_segment_prompt_nodes(seg_node, segment_id)
        if existing[0] is not None:
            return existing
        seg_obj = seg_node.GetSegmentation().GetSegment(segment_id)
        if seg_obj is None:
            return None, None
        seg_name = seg_obj.GetName()
        result = []
        for color, suffix, tag in (
            ((0.0, 1.0, 0.0), 'pos', _POS_TAG),
            ((1.0, 0.0, 0.0), 'neg', _NEG_TAG),
        ):
            node = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLMarkupsFiducialNode', f'{seg_name}-{suffix}')
            node.CreateDefaultDisplayNodes()
            dn = node.GetDisplayNode()
            dn.SetSelectedColor(*color)
            dn.SetColor(*color)
            dn.SetActiveColor(*color)
            node.SetMaximumNumberOfControlPoints(255)
            seg_obj.SetTag(tag, node.GetID())
            result.append(node)
        return tuple(result)

    def get_segment_prompt_nodes(self, seg_node, segment_id):
        """Return ``(pos_node, neg_node)`` for *segment_id*, or ``(None, None)``."""
        if not seg_node or not segment_id:
            return None, None
        seg_obj = seg_node.GetSegmentation().GetSegment(segment_id)
        if seg_obj is None:
            return None, None
        nodes = []
        for tag in (_POS_TAG, _NEG_TAG):
            value = vtk.reference('')
            seg_obj.GetTag(tag, value)
            node_id = str(value)
            node = slicer.mrmlScene.GetNodeByID(node_id) if node_id else None
            nodes.append(node)
        return tuple(nodes)

    def delete_segment_prompt_nodes(self, seg_node, segment_id):
        """Remove the pos/neg markup nodes for *segment_id* from the scene."""
        pos_node, neg_node = self.get_segment_prompt_nodes(seg_node, segment_id)
        for node in (pos_node, neg_node):
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)

    def sync_prompt_node_names(self, seg_node, segment_id):
        """Rename pos/neg markup nodes to match the current segment name."""
        seg_obj = seg_node.GetSegmentation().GetSegment(segment_id)
        if not seg_obj:
            return
        seg_name = seg_obj.GetName()
        pos_node, neg_node = self.get_segment_prompt_nodes(seg_node, segment_id)
        for node, suffix in ((pos_node, 'pos'), (neg_node, 'neg')):
            if node is not None:
                new_name = f'{seg_name}-{suffix}'
                if node.GetName() != new_name:
                    node.SetName(new_name)

    # ------------------------------------------------------------------ #
    # Segmentation / segment CRUD                                          #
    # ------------------------------------------------------------------ #

    def create_segmentation_for_volume(self, vol):
        """Create and return a new segmentation node linked to *vol*."""
        with _suppress_vtk_warnings():
            name = slicer.mrmlScene.GetUniqueNameByString('Segmentation')
            seg = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLSegmentationNode', name)
            seg.CreateDefaultDisplayNodes()
            seg.SetReferenceImageGeometryParameterFromVolumeNode(vol)
            signature = self.volume_geometry_signature(vol)
            if signature:
                seg.SetAttribute(_GEOMETRY_SIGNATURE_ATTR, signature)
        return seg

    def add_segment(self, seg_node) -> str:
        """Add a new empty segment, create its prompt nodes, and return its ID."""
        segmentation = seg_node.GetSegmentation()
        existing_ids = {
            segmentation.GetNthSegmentID(i)
            for i in range(segmentation.GetNumberOfSegments())
        }
        existing_names = {
            segmentation.GetNthSegment(i).GetName()
            for i in range(segmentation.GetNumberOfSegments())
        }
        # vtkMRMLSegmentationDisplayNode observes vtkSegmentation directly (VTK-level),
        # so StartModify doesn't suppress the transient "0 connections" warning that fires
        # when the 3D pipeline tries to render an empty segment during creation.
        with _suppress_vtk_warnings():
            mod = seg_node.StartModify()
            try:
                segmentation.AddEmptySegment(next_segment_name(existing_names))
                # The new segment is the one whose ID wasn't in the pre-call set.
                for i in range(segmentation.GetNumberOfSegments()):
                    sid = segmentation.GetNthSegmentID(i)
                    if sid not in existing_ids:
                        self.create_segment_prompt_nodes(seg_node, sid)
                        return sid
                return ''
            finally:
                seg_node.EndModify(mod)

    def remove_segment(self, seg_node, segment_id):
        """Remove *segment_id* from *seg_node*."""
        with _suppress_vtk_warnings():
            seg_node.GetSegmentation().RemoveSegment(segment_id)

    # ------------------------------------------------------------------ #
    # Segment visibility                                                   #
    # ------------------------------------------------------------------ #

    def _set_triplet_visibility(self, seg_node, dn, segment_id, visible):
        dn.SetSegmentVisibility(segment_id, visible)
        for node in self.get_segment_prompt_nodes(seg_node, segment_id):
            if node is not None:
                node.SetDisplayVisibility(int(visible))

    def set_saved_segments_visibility(self, seg_node, exclude_id, visible):
        """Set visibility of every segment triplet except *exclude_id*."""
        if not seg_node:
            return
        dn = seg_node.GetDisplayNode()
        if not dn:
            return
        segmentation = seg_node.GetSegmentation()
        with _suppress_vtk_warnings():
            for i in range(segmentation.GetNumberOfSegments()):
                sid = segmentation.GetNthSegmentID(i)
                if sid != exclude_id:
                    self._set_triplet_visibility(seg_node, dn, sid, visible)

    def set_segment_visibility(self, seg_node, segment_id, visible):
        """Set visibility of a single segment triplet."""
        if not seg_node or not segment_id:
            return
        dn = seg_node.GetDisplayNode()
        if dn:
            with _suppress_vtk_warnings():
                self._set_triplet_visibility(seg_node, dn, segment_id, visible)

    # ------------------------------------------------------------------ #
    # Window / Level                                                       #
    # ------------------------------------------------------------------ #

    def get_window_level(self, vol):
        """Return ``(window, level)`` ints from *vol*'s display node, or ``(None, None)``."""
        if not vol:
            return None, None
        dn = vol.GetDisplayNode()
        if not dn:
            return None, None
        return int(dn.GetWindow()), int(dn.GetLevel())

    def apply_window_level(self, vol, window, level):
        """Write *window* / *level* to *vol*'s display node."""
        if not vol:
            return
        dn = vol.GetDisplayNode()
        if not dn:
            return
        dn.SetAutoWindowLevel(0)
        dn.SetWindow(window)
        dn.SetLevel(level)

    # ------------------------------------------------------------------ #
    # Slice-index resolution                                               #
    # ------------------------------------------------------------------ #

    def active_slice_info(self, view_name: str, volume_node) -> tuple:
        """Return ``(axis, slice_idx)`` for *view_name* and *volume_node*.

        Returns ``(axis, None)`` when the view or volume is unavailable.
        """
        from core.utils import VIEW_TO_AXIS, AXIS_TO_IJK_COMPONENT
        axis = VIEW_TO_AXIS.get(view_name)
        if axis is None or volume_node is None:
            return axis, None
        lm = slicer.app.layoutManager()
        sw = lm.sliceWidget(view_name)
        if not sw:
            return axis, None

        xy_to_ras = sw.mrmlSliceNode().GetXYToRAS()
        ras = [xy_to_ras.GetElement(i, 3) for i in range(3)]

        m = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(m)
        ijk = [
            sum(m.GetElement(r, c) * (ras[c] if c < 3 else 1.0)
                for c in range(4))
            for r in range(3)
        ]

        comp      = AXIS_TO_IJK_COMPONENT[axis]
        slice_idx = int(round(ijk[comp]))
        dims      = volume_node.GetImageData().GetDimensions()  # (I, J, K)
        max_idx   = [dims[2], dims[1], dims[0]][axis]
        slice_idx = max(0, min(slice_idx, max_idx - 1))
        return axis, slice_idx

    # ------------------------------------------------------------------ #
    # Internal: zero-copy VTK buffer access                               #
    # ------------------------------------------------------------------ #

    def _vtk_view(self, seg_node, segment_id):
        """Return ``(numpy_view, vtkImageData)`` for the segment's binary labelmap.

        Shape is ``(K, J, I)`` — zero-copy writable view into Slicer's buffer.
        Returns ``(None, None)`` on any failure.
        """
        lm = seg_node.GetBinaryLabelmapInternalRepresentation(segment_id)
        if lm is None:
            return None, None
        img  = lm.GetImageData()
        ext  = img.GetExtent()
        if ext[0] != 0 or ext[2] != 0 or ext[4] != 0:
            return None, None
        dims = img.GetDimensions()
        if dims[0] <= 1 or dims[1] <= 1 or dims[2] <= 1:
            return None, None
        flat = _vtk_ns.vtk_to_numpy(img.GetPointData().GetScalars())
        if flat.max() > 1:
            return None, None
        return flat.reshape(dims[2], dims[1], dims[0]), img

    # ------------------------------------------------------------------ #
    # Public read API                                                      #
    # ------------------------------------------------------------------ #

    def segment_slice(self, seg_node, segment_id, volume_node,
                      axis: int, slice_idx: int) -> 'np.ndarray | None':
        """Return a 2-D ``uint8`` binary mask at *axis/slice_idx*.

        Fast path: zero-copy VTK read + one ``ndarray.copy()``.
        Slow fallback: ``arrayFromSegmentBinaryLabelmap``.
        Returns ``None`` on failure.
        """
        from core.utils import get_slice_from_volume
        view, _ = self._vtk_view(seg_node, segment_id)
        if view is not None:
            return get_slice_from_volume(view, axis, slice_idx).copy()
        raw = slicer.util.arrayFromSegmentBinaryLabelmap(
            seg_node, segment_id, volume_node)
        return get_slice_from_volume(raw, axis, slice_idx).copy()

    def segment_mask(self, seg_node, segment_id,
                     volume_node) -> 'np.ndarray | None':
        """Return a full 3-D ``uint8`` binary mask copy for the segment.

        Prefer :meth:`segment_slice` for single-slice operations.
        Returns ``None`` on failure.
        """
        view, _ = self._vtk_view(seg_node, segment_id)
        if view is not None:
            return view.copy()
        raw = slicer.util.arrayFromSegmentBinaryLabelmap(
            seg_node, segment_id, volume_node)
        return raw.copy()

    def current_segment_slice(self, pn, view_name: str,
                               segment_id: str) -> 'tuple | None':
        """Return ``(mask_2d, axis, slice_idx)`` for the given state, or ``None``.

        Parameters
        ----------
        pn          : vtkMRMLScriptedModuleNode  (the widget's parameter node)
        view_name   : str   e.g. 'Red'
        segment_id  : str
        """
        vol, seg = self.getVolumeAndSegmentation(pn)
        if vol is None or seg is None or not segment_id:
            return None
        axis, slice_idx = self.active_slice_info(view_name, vol)
        if slice_idx is None:
            return None
        mask = self.segment_slice(seg, segment_id, vol, axis, slice_idx)
        return (mask, axis, slice_idx) if mask is not None else None


def _ras_positions_close(a, b, tolerance=1e-4):
    if a is None or b is None or len(a) < 3 or len(b) < 3:
        return False
    tol2 = float(tolerance) * float(tolerance)
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)) <= tol2


#
# Test
#

class SegmentHumanBodyTest(ScriptedLoadableModuleTest):
    """Entry point for the 3D Slicer "Reload and Test" button."""

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

        suite  = unittest.TestLoader().loadTestsFromModule(ext)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise Exception(
                f'{len(result.failures) + len(result.errors)} test(s) failed — '
                'see the Python console for details'
            )
