import qt, vtk, slicer
import logging
import numpy as np
import vtk.util.numpy_support as _vtk_ns
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin

from core.utils import next_segment_name
from core._mouse_recorder import get_recorder
from core._input import StrokeHandler, BrushHandler, EraseHandler, PointHandler

log = logging.getLogger(__name__)

# MRML parameter-node reference keys
_INPUT_VOLUME = 'InputVolume'
_SEGMENTATION = 'Segmentation'

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
        self._parameterNode         = None
        self._savedSegmentsVisible  = False
        self._currentSegmentVisible = True
        self.currentViewName        = 'Red'
        self._recorder              = get_recorder()
        self._loaded_record         = None
        self._replay_engine         = None
        self._eof_widget            = None   # borrowed EffectsOptionsFrame
        self._eof_orig_parent       = None   # original parent to return it to
        self._active_handler        = None   # current InputHandler subclass instance
        self._active_prompt_widget  = None   # positivePrompts or negativePrompts last activated
        self._suppressing_place_mode = False  # True while segment creation is in progress
        self._observed_segmentation = None   # vtkMRMLSegmentationNode being tracked
        self._observed_seg_obj      = None   # its vtkSegmentation (holds the event)

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
        self.initializeParameterNode()
        self._update_record_ui()
        self._recorder.context_fn = self._recorder_context
        qt.QTimer.singleShot(0, self._preloadSegmentEditor)

    def cleanup(self):
        self._recorder.context_fn = None
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
        """
        self.ui.negativePrompts.setCurrentNode(neg_node)
        self._configureUnlimitedPlacement(self.ui.negativePrompts)
        self.ui.positivePrompts.setCurrentNode(pos_node)
        self._configureUnlimitedPlacement(self.ui.positivePrompts)

    # ------------------------------------------------------------------ #
    # Signal wiring                                                        #
    # ------------------------------------------------------------------ #

    def _connectSignals(self, uiWidget):
        ui = self.ui
        ui.sourceVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)',
                                        self._onNodeSelectorChanged)
        ui.segmentationNodeSelector.connect('currentNodeChanged(vtkMRMLNode*)',
                                            self._onNodeSelectorChanged)
        ui.segmentSelector.connect('currentSegmentChanged(QString)',
                                   self._onSegmentIDChanged)
        ui.addSegmentButton.connect('clicked(bool)', self._onAddSegment)
        ui.removeSegmentButton.connect('clicked(bool)', self._onRemoveSegment)
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

        ui.recordButton.connect('clicked(bool)',       self.onRecord)
        ui.stopRecordButton.connect('clicked(bool)',   self.onStopRecord)
        ui.exportRecordButton.connect('clicked(bool)', self.onExportRecord)
        ui.loadRecordButton.connect('clicked(bool)',   self.onLoadRecord)
        ui.replayRecordButton.connect('clicked(bool)', self.onReplayRecord)

        # Record volume / seg changes when active.
        ui.sourceVolumeSelector.connect(
            'currentNodeChanged(vtkMRMLNode*)',
            lambda node: (self._recorder.record_volume_changed(node.GetName() if node else None)
                          if self._recorder.is_active else None),
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
        sc(qt.QKeySequence('V'), uiWidget).connect(
            'activated()', lambda: ui.showCurrentSegmentCheckBox.toggle())
        sc(qt.QKeySequence('A'), uiWidget).connect('activated()', self._onAddSegment)

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
        vol = self._parameterNode.GetNodeReference(_INPUT_VOLUME)
        seg = self._parameterNode.GetNodeReference(_SEGMENTATION)
        self.ui.sourceVolumeSelector.setCurrentNode(vol)
        self.ui.segmentationNodeSelector.setCurrentNode(seg)
        if seg:
            self.ui.segmentSelector.setCurrentNode(seg)
        self.ui.addSegmentButton.setEnabled(vol is not None)
        self._syncWLFromVolume(vol)
        self.ui.showCurrentSegmentCheckBox.blockSignals(True)
        self.ui.showCurrentSegmentCheckBox.setChecked(True)
        self.ui.showCurrentSegmentCheckBox.blockSignals(False)
        self.ui.showSegmentsCheckBox.blockSignals(True)
        self.ui.showSegmentsCheckBox.setChecked(self._savedSegmentsVisible)
        self.ui.showSegmentsCheckBox.blockSignals(False)
        segID = self.ui.segmentSelector.currentSegmentID() if seg else None
        self.logic.set_saved_segments_visibility(seg, segID, self._savedSegmentsVisible)
        if seg and segID:
            pos_node, neg_node = self.logic.get_segment_prompt_nodes(seg, segID)
        else:
            pos_node, neg_node = None, None
        self._set_prompt_nodes(pos_node, neg_node)

    def _onNodeSelectorChanged(self, *_):
        if not self._parameterNode:
            return
        vol = self.ui.sourceVolumeSelector.currentNode()
        seg = self.ui.segmentationNodeSelector.currentNode()
        # Externally loaded segmentation nodes may have no display node, which
        # causes qMRMLSegmentsModel to warn. Ensure one exists before the
        # segment selector tries to render the segment list.
        if seg and not seg.GetDisplayNode():
            seg.CreateDefaultDisplayNodes()
        self._parameterNode.SetNodeReferenceID(
            _INPUT_VOLUME, vol.GetID() if vol else '')
        self._parameterNode.SetNodeReferenceID(
            _SEGMENTATION, seg.GetID() if seg else '')
        self._parameterNode.Modified()
        self._syncWLFromVolume(vol)
        self._rewire_segmentation_observer(seg)

    # ------------------------------------------------------------------ #
    # Segment-rename sync                                                  #
    # ------------------------------------------------------------------ #

    def _rewire_segmentation_observer(self, seg):
        """Keep exactly one SegmentModified observer on the active segmentation.

        The event lives on vtkSegmentation (the inner object), not on
        vtkMRMLSegmentationNode, so we observe GetSegmentation() directly.
        """
        new_obj = seg.GetSegmentation() if seg is not None else None
        old_obj = self._observed_seg_obj
        if old_obj is new_obj:
            return
        if old_obj is not None:
            self.removeObserver(old_obj, old_obj.SegmentModified,
                                self._onSegmentModified)
        self._observed_segmentation = seg
        self._observed_seg_obj      = new_obj
        if new_obj is not None:
            self.addObserver(new_obj, new_obj.SegmentModified,
                             self._onSegmentModified)

    def _onSegmentModified(self, caller, event, callData=None):
        """Rename markup nodes when their owning segment is renamed."""
        seg_id = str(callData) if callData is not None else None
        if not seg_id or self._observed_segmentation is None:
            return
        self.logic.sync_prompt_node_names(self._observed_segmentation, seg_id)

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def onRecord(self, *_):
        if self._recorder.is_active:
            return
        vol = self._parameterNode.GetNodeReference(_INPUT_VOLUME) if self._parameterNode else None
        seg = self._parameterNode.GetNodeReference(_SEGMENTATION) if self._parameterNode else None
        self._recorder.start(
            volume_node       = vol,
            segmentation_name = seg.GetName() if seg else None,
        )
        self._update_record_ui()

    def onStopRecord(self, *_):
        if not self._recorder.is_active:
            return
        self._recorder.stop()
        self._update_record_ui()

    def onExportRecord(self, *_):
        if self._recorder.is_active:
            self._recorder.stop()
            self._update_record_ui()
        path = qt.QFileDialog.getSaveFileName(None, 'Save Recording', '', 'JSON files (*.json)')
        if not path:
            return
        if not path.endswith('.json'):
            path += '.json'
        try:
            self._recorder.save_to_file(path)
            slicer.util.infoDisplay(f'Recording saved to:\n{path}')
        except Exception as exc:
            slicer.util.errorDisplay(f'Failed to save recording:\n{exc}')

    def onLoadRecord(self, *_):
        from core._mouse_recorder import MouseEventRecorder
        path = qt.QFileDialog.getOpenFileName(None, 'Load Recording', '', 'JSON files (*.json)')
        if not path:
            return
        try:
            self._loaded_record = MouseEventRecorder.load_from_file(path)
            slicer.util.infoDisplay(f'Loaded {len(self._loaded_record)} events from:\n{path}')
        except Exception as exc:
            slicer.util.errorDisplay(f'Failed to load recording:\n{exc}')
        self._update_record_ui()

    def onReplayRecord(self, *_):
        if self._loaded_record is None:
            slicer.util.warningDisplay('No recording loaded. Use Load first.')
            return
        from core._replay import ReplayEngine
        if self._replay_engine is not None and self._replay_engine.is_running:
            self._replay_engine.stop()
        self._replay_engine = ReplayEngine()
        self._replay_engine.start(self._loaded_record, self, on_done=self._on_replay_done)

    def _on_replay_done(self):
        self._update_record_ui()
        slicer.util.infoDisplay('Replay complete.')

    def _update_record_ui(self):
        ui        = self.ui
        is_active = self._recorder.is_active
        has_events = len(self._recorder) > 0

        replay_ok     = False
        replay_reason = ''
        if self._loaded_record is not None and not is_active:
            vol = (self._parameterNode.GetNodeReference(_INPUT_VOLUME)
                   if self._parameterNode else None)
            replay_ok, replay_reason = self._loaded_record.matches_volume(vol)

        ui.recordButton.setVisible(not is_active)
        ui.stopRecordButton.setVisible(is_active)
        ui.exportRecordButton.setEnabled(is_active or has_events)
        ui.replayRecordButton.setEnabled(replay_ok)

        if is_active:
            status = f'Recording... ({len(self._recorder)} events)'
        elif self._loaded_record is not None:
            n = len(self._loaded_record)
            status = (f'Loaded: {n} events — ready to replay' if replay_ok
                      else f'Loaded: {n} events — {replay_reason}')
        elif has_events:
            status = f'Recorded: {len(self._recorder)} events'
        else:
            status = ''
        ui.recordStatusLabel.setText(status)

    def _recorder_context(self) -> dict:
        seg_id = self.ui.segmentSelector.currentSegmentID()
        if self.ui.brushToolButton.isChecked():
            tool = 'brush'
        elif self.ui.eraseToolButton.isChecked():
            tool = 'erase'
        else:
            tool = None
        # Resolve active slice so replay can re-paint strokes on the right slice.
        pn = self._parameterNode
        vol = pn.GetNodeReference(_INPUT_VOLUME) if pn else None
        axis, slice_idx = self.logic.active_slice_info(self.currentViewName, vol)
        if tool in ('brush', 'erase'):
            editor = self.logic.get_segment_editor()
            pn_ed  = editor.mrmlSegmentEditorNode() if editor else None
            try:
                diam = float(pn_ed.GetAttribute('BrushAbsoluteDiameter') or 10)
            except Exception:
                diam = 10.0
            brush_radius_mm = diam / 2.0
        else:
            brush_radius_mm = None
        return {
            'segment_id':      seg_id,
            'tool':            tool,
            'view_name':       self.currentViewName,
            'axis':            axis,
            'slice_idx':       slice_idx,
            'brush_radius_mm': brush_radius_mm,
        }

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
        # Cache the active handler before creation, detach cleanly, then restore.
        # Suppress place-mode signals during creation so setCurrentNode calls inside
        # _set_prompt_nodes don't spuriously re-activate the wrong widget.
        prev_stroke_cls    = type(self._active_handler) if isinstance(self._active_handler, StrokeHandler) else None
        prev_prompt_widget = self._active_prompt_widget  if isinstance(self._active_handler, PointHandler)  else None
        if self._active_handler is not None:
            self._active_handler.detach(self)
        self._suppressing_place_mode = True
        try:
            new_id = self.logic.add_segment(seg)
            if new_id:
                self.ui.segmentSelector.setCurrentSegmentID(new_id)
        finally:
            self._suppressing_place_mode = False
            if prev_stroke_cls is not None:
                prev_stroke_cls().attach(self)
            elif prev_prompt_widget is not None:
                for child in prev_prompt_widget.findChildren(qt.QWidget):
                    if hasattr(child, 'setPlaceModeEnabled'):
                        child.setPlaceModeEnabled(True)
                        break

    def _onRemoveSegment(self, *_):
        seg   = self.ui.segmentationNodeSelector.currentNode()
        segID = self.ui.segmentSelector.currentSegmentID()
        if not seg or not segID:
            slicer.util.warningDisplay('No segment selected.')
            return
        self.logic.delete_segment_prompt_nodes(seg, segID)
        self.logic.remove_segment(seg, segID)

    def _onSegmentIDChanged(self, segmentID):
        seg = self.ui.segmentationNodeSelector.currentNode()
        if not segmentID or not seg:
            # No segment selected — clear the prompt widgets.
            self._set_prompt_nodes(None, None)
            return
        pos_node, neg_node = self.logic.get_segment_prompt_nodes(seg, segmentID)
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
        editor = self.logic.get_segment_editor()
        if editor:
            editor.undo()

    def _onRedo(self):
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
    # Segment Editor access                                                #
    # ------------------------------------------------------------------ #

    def get_segment_editor(self):
        """Return the shared qMRMLSegmentEditorWidget, or None."""
        try:
            return slicer.modules.segmenteditor.widgetRepresentation().self().editor
        except Exception:
            return None

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
        seg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        seg.CreateDefaultDisplayNodes()
        seg.SetReferenceImageGeometryParameterFromVolumeNode(vol)
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
        vtk.vtkObject.GlobalWarningDisplayOff()
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
            vtk.vtkObject.GlobalWarningDisplayOn()

    def remove_segment(self, seg_node, segment_id):
        """Remove *segment_id* from *seg_node*."""
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
        try:
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
        except Exception as exc:
            log.debug('[Logic._vtk_view] %s', exc)
            return None, None

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
        try:
            raw = slicer.util.arrayFromSegmentBinaryLabelmap(
                seg_node, segment_id, volume_node)
            return get_slice_from_volume(raw, axis, slice_idx).copy()
        except Exception as exc:
            log.debug('[Logic.segment_slice] fallback failed: %s', exc)
            return None

    def segment_mask(self, seg_node, segment_id,
                     volume_node) -> 'np.ndarray | None':
        """Return a full 3-D ``uint8`` binary mask copy for the segment.

        Prefer :meth:`segment_slice` for single-slice operations.
        Returns ``None`` on failure.
        """
        view, _ = self._vtk_view(seg_node, segment_id)
        if view is not None:
            return view.copy()
        try:
            raw = slicer.util.arrayFromSegmentBinaryLabelmap(
                seg_node, segment_id, volume_node)
            return raw.copy()
        except Exception as exc:
            log.debug('[Logic.segment_mask] fallback failed: %s', exc)
            return None

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
