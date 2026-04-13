import qt, vtk, slicer
import logging, ast
import numpy as np
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin

from core.modelFamilies import BaseModelFamily, SAMFamily, SPXModelFamily, AutoModelFamily
from core.utils import call_if_exists, get_slice_from_volume, write_slice_to_volume
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
        self.currentViewName = None  # default
        self._isRendering = False
        self._pauseRender = False

        # Undo tracking
        self._interactive_point_stack = []  # (node, controlPointID) — interactive mode
        self._undo_shortcut = None


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
        
        ]

        for ui_name, method in widget_button_connections:
            getattr(ui, ui_name).connect('clicked(bool)', method)

        ui.modelFamilyDropdown.connect('currentIndexChanged(int)', self.onModelFamilyChanged)
        ui.modelVariantDropdown.connect('currentIndexChanged(int)', self.onVariantChanged)
        ui.sliceViewDropdown.connect('currentTextChanged(QString)', self.onSliceViewChanged)
        ui.sourceVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        ui.segmentationNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        ui.segmentSelector.connect("currentSegmentChanged(QString)", self.onSegmentChanged)

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

    def _onMarkupsModified(self, caller=None, event=None):
        if not self._parameterNode:
            return

        self.logic.updateParameterNodeFromMarkups(
            self._parameterNode,
            self.ui.positivePrompts.currentNode(),
            self.ui.negativePrompts.currentNode(),
        )

    def _onPointAdded(self, caller=None, event=None):
        """Record each newly added prompt point so Ctrl+Z can remove it."""
        if caller is None:
            return
        n = caller.GetNumberOfControlPoints()
        if n > 0:
            cp_id = caller.GetNthControlPointID(n - 1)
            self._interactive_point_stack.append((caller, cp_id))

    def onSliceViewChanged(self, viewName):
        self.currentViewName = viewName

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
        segNode = self.ui.segmentationNodeSelector.currentNode()

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

        # Prompt-point history is per-session; clear it on segment switch so
        # Ctrl+Z cannot remove points that belong to a different segment.
        self._interactive_point_stack.clear()

        self._pauseRender = True
        try:
            self.clearPrompts()
        finally:
            self._pauseRender = False
    
    def clearPrompts(self):
        posNode = self.ui.positivePrompts.currentNode()
        negNode = self.ui.negativePrompts.currentNode()

        if posNode:
            posNode.RemoveAllControlPoints()

        if negNode:
            negNode.RemoveAllControlPoints()

        self._interactive_point_stack.clear()

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

    def setDefaultParameters(self, parameterNode):
        pass

    def reset_render_state(self):
        """Clear per-segment caches.  Call whenever the active segment or
        volume changes so the next render reads fresh data from Slicer."""
        self._last_render_key = None
        self._working_mask = None
        self._working_mask_segment = None

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

        # --- Extract points ---
        pos_points = [
            posNode.GetNthControlPointPosition(i)
            for i in range(posNode.GetNumberOfControlPoints())
        ] if posNode else []

        neg_points = [
            negNode.GetNthControlPointPosition(i)
            for i in range(negNode.GetNumberOfControlPoints())
        ] if negNode else []

        if not pos_points and not neg_points:
            return

        # --- Get image ---
        volumeNode = widget.ui.sourceVolumeSelector.currentNode()
        if not volumeNode:
            return

        axis, sliceIndex = self.getAxisAndSlice(widget)

        # --- Build render key before any expensive work ---
        # pos/neg points are VTK objects; tuple(p) makes them hashable.
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

        # SPXModelFamily.onRender is already cached (no model.forward call on
        # cache hit), but applyResult still writes the full 3D labelmap to
        # Slicer on every tick.  Skip it when nothing has changed.
        if render_key == self._last_render_key:
            return

        # --- Compute result ---
        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        img = get_slice_from_volume(volumeArray, axis, sliceIndex)

        scribbles_ijk = self._ras_to_ijk(volumeNode, {
            "positive": pos_points,
            "negative": neg_points,
        }, axis)

        result = call_if_exists(
            modelFamily,
            "onRender",
            img=img,
            pos_points=scribbles_ijk["positive"],
            neg_points=scribbles_ijk["negative"],
            **params
        )

        # --- Apply result and record key ---
        if result is not None:
            self.applyResult(widget, result, axis, sliceIndex)
            self._last_render_key = render_key
    
    def applyResult(self, widget, mask2d, axis, sliceIndex):
        volumeNode = widget.ui.sourceVolumeSelector.currentNode()
        if not volumeNode:
            return

        segNode = widget.ui.segmentSelector.currentNode()
        segmentID = widget.ui.segmentSelector.currentSegmentID()

        # Auto-create a segmentation node if the user hasn't selected one yet.
        if not segNode:
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segNode.CreateDefaultDisplayNodes()
            segNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
            self.setVolumeAndSegmentation(widget._parameterNode, volumeNode, segNode)
            # Update both selectors without triggering downstream signal chains.
            widget.ui.segmentationNodeSelector.blockSignals(True)
            widget.ui.segmentationNodeSelector.setCurrentNode(segNode)
            widget.ui.segmentationNodeSelector.blockSignals(False)
            widget.ui.segmentSelector.blockSignals(True)
            widget.ui.segmentSelector.setCurrentNode(segNode)
            widget.ui.segmentSelector.blockSignals(False)

        # Auto-create the first segment if the segmentation is empty.
        if not segmentID:
            segmentID = segNode.GetSegmentation().AddEmptySegment("Segment_1")
            # Block signals so onSegmentChanged does not clear the prompt points
            # that triggered this render.
            widget.ui.segmentSelector.blockSignals(True)
            widget.ui.segmentSelector.setCurrentSegmentID(segmentID)
            widget.ui.segmentSelector.blockSignals(False)
            widget.ui.addSegmentButton.setEnabled(True)

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

        if not segmentID:
            slicer.util.warningDisplay("No segment selected.")
            return

        params = widget.getUserParameters()

        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        axis, sliceIndex = self.getAxisAndSlice(widget)
        img = get_slice_from_volume(volumeArray, axis, sliceIndex)

        # Convert negative prompt points to 2-D IJK for this slice so they can
        # be used to subtract regions in expandSegWithSPX.
        _, negNode = self.getPromptNodes(widget._parameterNode)
        neg_points_ras = [
            negNode.GetNthControlPointPosition(i)
            for i in range(negNode.GetNumberOfControlPoints())
        ] if negNode else []
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

        # Force a fresh read from Slicer on the first render: the user may
        # have edited the segment outside of interactive mode.
        self.reset_render_state()

        widget.renderer.start()
        widget.setInteractiveState(True)

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
        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)

        def convert(points):
            ijk_pts = []

            for p in points:
                ras = list(p) + [1]
                ijk = rasToIjk.MultiplyPoint(ras)

                i, j, k = int(ijk[0]), int(ijk[1]), int(ijk[2])

                if axis == 0:        # Red (Z slice)
                    pt2d = [i, j]
                elif axis == 1:      # Yellow (Y slice)
                    pt2d = [i, k]
                elif axis == 2:      # Green (X slice)
                    pt2d = [j, k]

                ijk_pts.append(pt2d)

            return ijk_pts

        return {
            "positive": convert(scrib["positive"]),
            "negative": convert(scrib["negative"])
        }

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
            for x, y in neg_points:
                if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                    selected_labels.discard(int(labels[y, x]))

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
    
