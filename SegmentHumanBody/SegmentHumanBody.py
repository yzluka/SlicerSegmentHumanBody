import glob
import json
import os
import pickle
from PIL import Image

import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin

from models import cfg
from model_manager import ModelManager
from segmentation_utils import (
    bfs_connected_component,
    combine_multiple_masks,
    get_annotation_slice,
    get_slice_accessor_dimension,
    get_volume_slice,
    set_volume_slice,
)

#
# SegmentHumanBody
#

args = cfg.parse_args()
args.if_mask_decoder_adapter = True
args.if_encoder_adapter = True
args.decoder_adapt_depth = 2


class SegmentHumanBody(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "SegmentHumanBody"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Zafer Yildiz (Mazurowski Lab, Duke University)"]
        self.parent.helpText = """
The SegmentHumanBody module aims to assist its users in segmenting medical data by integrating
the <a href="https://github.com/facebookresearch/segment-anything">Segment Anything Model (SAM)</a>
developed by Meta.<br>
<br>
See more information in <a href="https://github.com/mazurowski-lab/SlicerSegmentHumanBody">module documentation</a>.
"""
        self.parent.acknowledgementText = """
This file was originally developed by Zafer Yildiz (Mazurowski Lab, Duke University).
"""


class SegmentHumanBodyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    MODEL_SPX = "SPX-Assisted Annotation"
    MODEL_BONE = "SegmentAnyBone"
    MODEL_BREAST = "Breast Segmentation Model"
    MODEL_MUSCLE = "SegmentAnyMuscle"
    MODEL_CT = "CT Segmentation"
    MODEL_SAM2 = "SLM-SAM 2"

    def __init__(self, parent=None):
        super().__init__(parent)
        VTKObservationMixin.__init__(self)

        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self._controllerUpdating = False

        self.slicesFolder = self.resourcePath("UI") + "/../../../slices"
        self.featuresFolder = self.resourcePath("UI") + "/../../../features"
        self.annotationMasksFolder = self.resourcePath("UI") + "/../../../annotations"

        self.modelVersion = "vit_t"
        self.modelCheckpoint = self.resourcePath("UI") + "/../../models/bone_sam.pth"
        self.mask_threshold = 0

        self.masks = None
        self.init_masks = None
        self.producedMask = None
        self.currentlySegmenting = False
        self.featuresAreExtracted = False

        self.volume = None
        self.volumeShape = None
        self.imageShape = None
        self.nofSlices = 0
        self.sliceAccessorDimension = 2

        self.segmentIdToSegmentationMask = {}

        self.model_manager = ModelManager(self)
        self.model_manager.initialize_environment()

        self.device = self.model_manager.device
        self.sam = None
        self.sam2AnnotationTool = None

        self.changeModel(self.MODEL_BREAST)
    def _safeSetParameter(self, key, value):
        if self._parameterNode is None:
            return

        current = self._parameterNode.GetParameter(key)
        if current == value:
            return

        self._controllerUpdating = True
        try:
            self._parameterNode.SetParameter(key, value)
        finally:
            self._controllerUpdating = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SegmentHumanBody.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = SegmentHumanBodyLogic()

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.positivePrompts.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.positivePrompts.markupsPlaceWidget().setPlaceModePersistency(True)
        self.ui.negativePrompts.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.negativePrompts.markupsPlaceWidget().setPlaceModePersistency(True)

        self.ui.goToSegmentEditorButton.connect("clicked(bool)", self.onGoToSegmentEditor)
        self.ui.goToMarkupsButton.connect("clicked(bool)", self.onGoToMarkups)
        self.ui.runAutomaticSegmentation.connect("clicked(bool)", self.onAutomaticSegmentation)
        self.ui.assignLabel2D.connect("clicked(bool)", self.onAssignLabel2D)
        self.ui.assignLabel3D.connect("clicked(bool)", self.onAssignLabelIn3D)
        self.ui.startInferenceForSAM2ToolButton.connect("clicked(bool)", self.sam2AnnotationToolInference)
        self.ui.segmentButton.connect("clicked(bool)", self.onStartSegmentation)
        self.ui.stopSegmentButton.connect("clicked(bool)", self.onStopSegmentButton)
        self.ui.segmentationDropDown.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.maskDropDown.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.modelDropDown.connect("currentIndexChanged(int)", self.onModelChanged)

        self.ui.modelDropDown.blockSignals(True)
        self.ui.maskDropDown.blockSignals(True)
        self.ui.segmentationDropDown.blockSignals(True)

        self.initializeParameterNode()

        self.ui.modelDropDown.blockSignals(False)
        self.ui.maskDropDown.blockSignals(False)
        self.ui.segmentationDropDown.blockSignals(False)

    def cleanup(self):
        self.removeObservers()

    def enter(self):
        pass

    def exit(self):
        if self._parameterNode:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )

    def onSceneStartClose(self, caller, event):
        self.setParameterNode(None)
    
    def onModelChanged(self, index):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        if index < 0:
            return  # ignore transient invalid state

        newMethodName = self.ui.modelDropDown.itemText(index)

        if self._parameterNode.GetParameter("SAMCurrentModel") == newMethodName:
            return

        self._safeSetParameter("SAMCurrentModel", newMethodName)

        # apply effects
        self.changeModel(newMethodName)
        self.render_Widgets_by_Method(newMethodName)

    def onSceneEndClose(self, caller, event):
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())

        if not self._parameterNode.GetNodeReferenceID("positivePromptPointsNode"):
            newPromptPointNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "positive"
            )
            newPromptPointNode.GetDisplayNode().SetSelectedColor(0, 1, 0)
            self._parameterNode.SetNodeReferenceID(
                "positivePromptPointsNode", newPromptPointNode.GetID()
            )

        if not self._parameterNode.GetNodeReferenceID("negativePromptPointsNode"):
            newPromptPointNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "negative"
            )
            newPromptPointNode.GetDisplayNode().SetSelectedColor(1, 0, 0)
            self._parameterNode.SetNodeReferenceID(
                "negativePromptPointsNode", newPromptPointNode.GetID()
            )

        self.ui.positivePrompts.setCurrentNode(
            self._parameterNode.GetNodeReference("positivePromptPointsNode")
        )
        self.ui.negativePrompts.setCurrentNode(
            self._parameterNode.GetNodeReference("negativePromptPointsNode")
        )

        if not self._parameterNode.GetNodeReferenceID("SAMSegmentationNode"):
            self.samSegmentationNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", "Segmentation"
            )
            self.samSegmentationNode.CreateDefaultDisplayNodes()
            firstSegmentId = self.samSegmentationNode.GetSegmentation().AddEmptySegment("bone")

            self._parameterNode.SetNodeReferenceID(
                "SAMSegmentationNode", self.samSegmentationNode.GetID()
            )
            self._parameterNode.SetParameter("SAMCurrentSegment", firstSegmentId)
            self._parameterNode.SetParameter("SAMCurrentMask", "Mask-1")
            self._parameterNode.SetParameter("SAMCurrentModel", self.MODEL_BONE)

        self.ui.segmentationDropDown.blockSignals(True)
        self.ui.maskDropDown.blockSignals(True)
        self.ui.modelDropDown.blockSignals(True)

        self.ui.segmentationDropDown.clear()
        segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")
        if segmentationNode:
            seg = segmentationNode.GetSegmentation()
            for i in range(seg.GetNumberOfSegments()):
                self.ui.segmentationDropDown.addItem(seg.GetNthSegment(i).GetName())


        self.ui.maskDropDown.clear()
        for i in range(2):
            self.ui.maskDropDown.addItem(f"Mask-{i+1}")

        self.ui.modelDropDown.clear()
        self.ui.modelDropDown.addItems([
            self.MODEL_SPX,
            self.MODEL_BONE,
            self.MODEL_BREAST,
            self.MODEL_MUSCLE,
            self.MODEL_CT,
            self.MODEL_SAM2,
        ])

        self.ui.ctSegmentationModelDropdown.clear()
        self.ui.ctSegmentationModelDropdown.addItems([
            "Custom",
            "2D",
            "3D",
            "Both",
        ])

        if self._parameterNode.GetParameter("SAMCurrentSegment"):
            segId = self._parameterNode.GetParameter("SAMCurrentSegment")
            segName = segmentationNode.GetSegmentation().GetSegment(segId).GetName()
            self.ui.segmentationDropDown.setCurrentText(segName)

        if self._parameterNode.GetParameter("SAMCurrentMask"):
            self.ui.maskDropDown.setCurrentText(
                self._parameterNode.GetParameter("SAMCurrentMask")
            )

        if self._parameterNode.GetParameter("SAMCurrentModel"):
            self.ui.modelDropDown.setCurrentText(
                self._parameterNode.GetParameter("SAMCurrentModel")
            )

        self.render_Widgets_by_Method(self._parameterNode.GetParameter("SAMCurrentModel"))

        self.ui.segmentationDropDown.blockSignals(False)
        self.ui.maskDropDown.blockSignals(False)
        self.ui.modelDropDown.blockSignals(False)

    def setParameterNode(self, inputParameterNode):
        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        if self._parameterNode is not None and self.hasObserver(
            self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode
        ):
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )

        self._parameterNode = inputParameterNode

        if self._parameterNode is not None:
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )

        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode or self._controllerUpdating:   
            return

        if not slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode"):
            return

        self._updatingGUIFromParameterNode = True

        try:
            if self._parameterNode.GetNodeReferenceID("SAMSegmentationNode"):
                segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")

                self.ui.segmentationDropDown.clear()
                for i in range(segmentationNode.GetSegmentation().GetNumberOfSegments()):
                    segmentName = segmentationNode.GetSegmentation().GetNthSegment(i).GetName()
                    self.ui.segmentationDropDown.addItem(segmentName)

                if self._parameterNode.GetParameter("SAMCurrentSegment"):
                    currentSeg = segmentationNode.GetSegmentation().GetSegment(
                        self._parameterNode.GetParameter("SAMCurrentSegment")
                    )
                    if currentSeg:
                        self.ui.segmentationDropDown.setCurrentText(currentSeg.GetName())

                if self._parameterNode.GetParameter("SAMCurrentMask"):
                    self.ui.maskDropDown.setCurrentText(
                        self._parameterNode.GetParameter("SAMCurrentMask")
                    )

                modelName = self._parameterNode.GetParameter("SAMCurrentModel")
                if modelName:
                    index = self.ui.modelDropDown.findText(modelName)
                    if index >= 0:
                        self.ui.modelDropDown.setCurrentIndex(index)

            self.render_Widgets_by_Method(modelName)

        finally:
            self._updatingGUIFromParameterNode = False


    def render_minimalist_layout(self):
        self.ui.assignLabel2D.hide()
        self.ui.assignLabel3D.hide()
        self.ui.goToMarkupsButton.hide()
        self.ui.segmentButton.hide()
        self.ui.stopSegmentButton.hide()
        self.ui.segmentationDropDown.hide()
        self.ui.maskDropDown.hide()
        self.ui.ctSegmentationModelDropdown.hide()
        self.ui.startTrainingForSAM2ToolButton.hide()
        self.ui.startInferenceForSAM2ToolButton.hide()
        self.ui.runAutomaticSegmentation.hide()

    def render_interactive_seg_layout(self):
        self.ui.assignLabel2D.show()
        self.ui.assignLabel3D.show()
        self.ui.goToMarkupsButton.show()
        self.ui.segmentButton.show()
        self.ui.stopSegmentButton.show()
        self.ui.segmentationDropDown.show()
        self.ui.maskDropDown.show()

    def render_Widgets_by_Method(self, methodName):
        self.render_minimalist_layout()
        
        if methodName == self.MODEL_SPX:
            self.ui.runAutomaticSegmentation.show()
        elif methodName in (self.MODEL_BREAST, self.MODEL_MUSCLE):
            self.ui.runAutomaticSegmentation.show()
        elif methodName == self.MODEL_BONE:
            self.render_interactive_seg_layout()
        elif methodName == self.MODEL_CT:
            self.ui.ctSegmentationModelDropdown.show()
            self.ui.runAutomaticSegmentation.show()
        elif methodName == self.MODEL_SAM2:
            self.ui.ctSegmentationModelDropdown.show()
            self.ui.startInferenceForSAM2ToolButton.show()
        else:
            print("Model dropdown:", self.ui.modelDropDown.currentIndex, repr(self.ui.modelDropDown.currentText))
            raise ValueError(f"Invalid method name: {methodName}")

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        if not self._parameterNode.GetNodeReference("SAMSegmentationNode") or not hasattr(self, "volumeShape"):
            return

        wasModified = self._parameterNode.StartModify()

        segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode").GetSegmentation()

        segId = segmentationNode.GetSegmentIdBySegmentName(self.ui.segmentationDropDown.currentText)
        self._safeSetParameter("SAMCurrentSegment", segId)

        if segId not in self.segmentIdToSegmentationMask and self.volumeShape is not None:
            self.segmentIdToSegmentationMask[segId] = np.zeros(self.volumeShape, dtype=bool)

        self._safeSetParameter("SAMCurrentMask", self.ui.maskDropDown.currentText)

        self._parameterNode.EndModify(wasModified)
    
    

    def changeModel(self, modelName):
        self.modelName = modelName

        if modelName == self.MODEL_SAM2:
            self.initializeVariables()
        elif modelName == self.MODEL_BONE:
            self.sam = self.model_manager.build_bone_predictor(args, self.modelVersion, self.modelCheckpoint)
        elif modelName in (self.MODEL_MUSCLE, self.MODEL_BREAST, self.MODEL_CT, self.MODEL_SPX):
            pass

    def initializeVariables(self):
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())
            else:
                slicer.util.warningDisplay("You need to add data first to start segmentation!")
                return False

        self.volume = slicer.util.arrayFromVolume(self._parameterNode.GetNodeReference("InputVolume"))
        self.volumeShape = self.volume.shape

        currentSeg = self._parameterNode.GetParameter("SAMCurrentSegment")
        if currentSeg and currentSeg not in self.segmentIdToSegmentationMask:
            self.segmentIdToSegmentationMask[currentSeg] = np.zeros(self.volumeShape, dtype=bool)

        self.sliceAccessorDimension = get_slice_accessor_dimension(
            self._parameterNode.GetNodeReference("InputVolume")
        )

        sampleInputImage = get_volume_slice(self.volume, self.sliceAccessorDimension, 0)
        self.nofSlices = self.volume.shape[self.sliceAccessorDimension]
        self.imageShape = sampleInputImage.shape
        return True

    def createSlices(self):
        if not os.path.exists(self.slicesFolder):
            os.makedirs(self.slicesFolder)

        for filename in glob.glob(self.slicesFolder + "/*"):
            os.remove(filename)

        for sliceIndex in range(self.nofSlices):
            sliceImage = get_volume_slice(self.volume, self.sliceAccessorDimension, sliceIndex)
            np.save(self.slicesFolder + "/" + f"slice_{sliceIndex}", sliceImage)

    def createAnnotationMasks(self):
        if not os.path.exists(self.annotationMasksFolder):
            os.makedirs(self.annotationMasksFolder)

        for filename in glob.glob(self.annotationMasksFolder + "/*"):
            os.remove(filename)

        volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")
        segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName("Segment_1")

        slicer.util.saveNode(volumeNode, self.annotationMasksFolder + "/volume.nii.gz")
        segmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)

        sliceAnnotationsArr = []
        for sliceIndex in range(self.nofSlices):
            sliceAnnotationImage = get_annotation_slice(segmentArray, self.sliceAccessorDimension, sliceIndex)
            sliceAnnotationsArr.append(sliceAnnotationImage)
            np.save(self.annotationMasksFolder + "/" + f"slice_{sliceIndex}", sliceAnnotationImage)

        import nibabel

        maskVolume = np.stack(sliceAnnotationsArr, axis=-1)
        np.save(self.annotationMasksFolder + "/mask.npy", maskVolume)
        np.save(
            self.annotationMasksFolder + "/volume.npy",
            nibabel.load(self.annotationMasksFolder + "/volume.nii.gz").get_fdata(),
        )

        self.sam2AnnotationTool = self.model_manager.build_sam2_annotation_tool()

    def sam2AnnotationToolInference(self):
        self.createAnnotationMasks()

        img_vol_data, mask_vol_data = self.sam2AnnotationTool.load_data(
            img_path=self.annotationMasksFolder + "/volume.npy",
            mask_path=self.annotationMasksFolder + "/mask.npy",
        )

        predictedVolume = self.sam2AnnotationTool.inference(
            data_save_directory=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/example_test_data/",
            img_save_directory=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/example_test_data/images/",
            mask_save_directory=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/example_test_data/masks/",
            volume_name="volume",
            img_vol_data=img_vol_data,
            mask_vol_data=mask_vol_data,
            ann_frame_idx=self.getIndexOfCurrentSlice(),
            checkpoint_folder=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/sam2_logs/configs/sam2.1_training/lstm_sam2.1_hiera_t.yaml/checkpoints",
            cfg_folder=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/sam2/configs/sam2.1",
        )

        segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")
        segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName("Segment_1")

        if segmentId not in self.segmentIdToSegmentationMask:
            self.segmentIdToSegmentationMask[segmentId] = np.zeros(self.volumeShape, dtype=bool)

        for sliceIndex in range(self.getTotalNumberOfSlices()):
            sliceMask = predictedVolume[:, :, sliceIndex].astype(bool)
            set_volume_slice(
                self.segmentIdToSegmentationMask[segmentId],
                self.sliceAccessorDimension,
                sliceIndex,
                sliceMask,
            )

        slicer.util.updateSegmentBinaryLabelmapFromArray(
            self.segmentIdToSegmentationMask[segmentId],
            self._parameterNode.GetNodeReference("SAMSegmentationNode"),
            segmentId,
            self._parameterNode.GetNodeReference("InputVolume"),
        )

    def createFeatures(self):
        if not os.path.exists(self.featuresFolder):
            os.makedirs(self.featuresFolder)

        for filename in glob.glob(self.featuresFolder + "/*"):
            os.remove(filename)

        for filename in os.listdir(self.slicesFolder):
            image = np.load(self.slicesFolder + "/" + filename)
            image = (255 * (image - np.min(image)) / (np.ptp(image) + 1e-8)).astype(np.uint8)
            image = np.stack([image] * 3, axis=-1)
            self.sam.set_image(image)

            with open(self.featuresFolder + "/" + os.path.splitext(filename)[0] + "_features.pkl", "wb") as f:
                pickle.dump(self.sam.features, f)

    def initializeSegmentationProcess(self):
        self.positivePromptPointsNode = self._parameterNode.GetNodeReference("positivePromptPointsNode")
        self.negativePromptPointsNode = self._parameterNode.GetNodeReference("negativePromptPointsNode")

        self.volumeRasToIjk = vtk.vtkMatrix4x4()
        self.volumeIjkToRas = vtk.vtkMatrix4x4()
        self._parameterNode.GetNodeReference("InputVolume").GetRASToIJKMatrix(self.volumeRasToIjk)
        self._parameterNode.GetNodeReference("InputVolume").GetIJKToRASMatrix(self.volumeIjkToRas)

    def getIndexOfCurrentSlice(self):
        redView = slicer.app.layoutManager().sliceWidget("Red")
        redViewLogic = redView.sliceLogic()
        return redViewLogic.GetSliceIndexFromOffset(redViewLogic.GetSliceOffset()) - 1

    def getTotalNumberOfSlices(self):
        return self.volume.shape[self.sliceAccessorDimension]

    def onGoToSegmentEditor(self):
        slicer.util.selectModule("SegmentEditor")

    def onGoToMarkups(self):
        slicer.util.selectModule("Markups")

    def onStopSegmentButton(self):
        self.currentlySegmenting = False
        self.masks = None
        self.init_masks = None

        for i in range(self.positivePromptPointsNode.GetNumberOfControlPoints()):
            self.positivePromptPointsNode.SetNthControlPointVisibility(i, False)

        for i in range(self.negativePromptPointsNode.GetNumberOfControlPoints()):
            self.negativePromptPointsNode.SetNthControlPointVisibility(i, False)

        roiList = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for roiNode in roiList:
            slicer.mrmlScene.RemoveNode(roiNode)

    def _store_mask_for_segment(self, segmentId, sliceIndex, mask):
        if segmentId not in self.segmentIdToSegmentationMask:
            self.segmentIdToSegmentationMask[segmentId] = np.zeros(self.volumeShape, dtype=bool)
        set_volume_slice(
            self.segmentIdToSegmentationMask[segmentId],
            self.sliceAccessorDimension,
            sliceIndex,
            mask.astype(bool),
        )

    def _update_segment_in_scene(self, segmentId):
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            self.segmentIdToSegmentationMask[segmentId],
            self._parameterNode.GetNodeReference("SAMSegmentationNode"),
            segmentId,
            self._parameterNode.GetNodeReference("InputVolume"),
        )

    def _collect_visible_prompt_points(self, markupsNode):
        points = []
        nofPoints = markupsNode.GetNumberOfControlPoints()

        for i in range(nofPoints):
            if markupsNode.GetNthControlPointVisibility(i):
                pointRAS = [0, 0, 0]
                markupsNode.GetNthControlPointPositionWorld(i, pointRAS)
                pointIJK = [0, 0, 0, 1]
                self.volumeRasToIjk.MultiplyPoint(np.append(pointRAS, 1.0), pointIJK)
                pointIJK = [int(round(c)) for c in pointIJK[0:3]]

                if self.sliceAccessorDimension == 2:
                    points.append([pointIJK[1], pointIJK[2]])
                elif self.sliceAccessorDimension == 1:
                    points.append([pointIJK[0], pointIJK[2]])
                else:
                    points.append([pointIJK[0], pointIJK[1]])
        return points

    def getLabelOfPromptPoint(self, promptPoint):
        currentSliceIndex = self.getIndexOfCurrentSlice()

        for segmentId, segmentMask in self.segmentIdToSegmentationMask.items():
            currentMask = get_annotation_slice(segmentMask, self.sliceAccessorDimension, currentSliceIndex)
            if currentMask[promptPoint[1], promptPoint[0]]:
                return segmentId

        return None

    def onAssignLabel2D(self):
        self.initializeSegmentationProcess()

        labelAssigned = False
        promptPoints = self._collect_visible_prompt_points(self.positivePromptPointsNode)

        if promptPoints:
            promptPointToAssignLabel = promptPoints[-1]
            currentSliceIndex = self.getIndexOfCurrentSlice()
            segmentationIdToBeUpdated = self.getLabelOfPromptPoint(promptPointToAssignLabel)

            if segmentationIdToBeUpdated is None:
                qt.QTimer.singleShot(100, self.onAssignLabel2D)
                return

            currentMask = get_annotation_slice(
                self.segmentIdToSegmentationMask[segmentationIdToBeUpdated],
                self.sliceAccessorDimension,
                currentSliceIndex,
            )

            componentMask = bfs_connected_component(currentMask, promptPointToAssignLabel)

            targetSeg = self._parameterNode.GetParameter("SAMCurrentSegment")
            if targetSeg not in self.segmentIdToSegmentationMask:
                self.segmentIdToSegmentationMask[targetSeg] = np.zeros(self.volumeShape, dtype=bool)

            set_volume_slice(
                self.segmentIdToSegmentationMask[targetSeg],
                self.sliceAccessorDimension,
                currentSliceIndex,
                componentMask,
            )

            sourceSlice = get_annotation_slice(
                self.segmentIdToSegmentationMask[segmentationIdToBeUpdated],
                self.sliceAccessorDimension,
                currentSliceIndex,
            )
            sourceSlice[componentMask == True] = False
            set_volume_slice(
                self.segmentIdToSegmentationMask[segmentationIdToBeUpdated],
                self.sliceAccessorDimension,
                currentSliceIndex,
                sourceSlice,
            )

            self._update_segment_in_scene(segmentationIdToBeUpdated)
            self._update_segment_in_scene(targetSeg)

            labelAssigned = True
            for i in range(self.positivePromptPointsNode.GetNumberOfControlPoints()):
                self.positivePromptPointsNode.SetNthControlPointVisibility(i, False)

        if not labelAssigned:
            qt.QTimer.singleShot(100, self.onAssignLabel2D)

    def onAssignLabelIn3D(self):
        self.initializeSegmentationProcess()

        labelAssigned = False
        promptPoints = self._collect_visible_prompt_points(self.positivePromptPointsNode)

        if promptPoints:
            promptPointToAssignLabel = promptPoints[-1]
            segmentationIdToBeUpdated = self.getLabelOfPromptPoint(promptPointToAssignLabel)

            if segmentationIdToBeUpdated is None:
                qt.QTimer.singleShot(100, self.onAssignLabelIn3D)
                return

            targetSeg = self._parameterNode.GetParameter("SAMCurrentSegment")
            if targetSeg not in self.segmentIdToSegmentationMask:
                self.segmentIdToSegmentationMask[targetSeg] = np.zeros(self.volumeShape, dtype=bool)

            sliceIndicesThatContainObject = []
            for sliceIndex in range(self.getTotalNumberOfSlices()):
                currentMask = get_annotation_slice(
                    self.segmentIdToSegmentationMask[segmentationIdToBeUpdated],
                    self.sliceAccessorDimension,
                    sliceIndex,
                )
                if currentMask[promptPointToAssignLabel[1], promptPointToAssignLabel[0]]:
                    sliceIndicesThatContainObject.append(sliceIndex)

            for sliceIndex in sliceIndicesThatContainObject:
                currentMask = get_annotation_slice(
                    self.segmentIdToSegmentationMask[segmentationIdToBeUpdated],
                    self.sliceAccessorDimension,
                    sliceIndex,
                )

                componentMask = bfs_connected_component(currentMask, promptPointToAssignLabel)
                set_volume_slice(
                    self.segmentIdToSegmentationMask[targetSeg],
                    self.sliceAccessorDimension,
                    sliceIndex,
                    componentMask,
                )

                currentMask[componentMask == True] = False
                set_volume_slice(
                    self.segmentIdToSegmentationMask[segmentationIdToBeUpdated],
                    self.sliceAccessorDimension,
                    sliceIndex,
                    currentMask,
                )

            self._update_segment_in_scene(segmentationIdToBeUpdated)
            self._update_segment_in_scene(targetSeg)

            labelAssigned = True
            for i in range(self.positivePromptPointsNode.GetNumberOfControlPoints()):
                self.positivePromptPointsNode.SetNthControlPointVisibility(i, False)

        if not labelAssigned:
            qt.QTimer.singleShot(100, self.onAssignLabelIn3D)

    def onAutomaticSegmentation(self):
        if not self.initializeVariables():
            return

        if self.modelName == self.MODEL_BONE:
            self._run_automatic_bone_segmentation()
        elif self.modelName == self.MODEL_MUSCLE:
            self._run_automatic_muscle_segmentation()
        elif self.modelName == self.MODEL_CT:
            self._run_automatic_ct_segmentation()
        else:
            self._run_automatic_breast_segmentation()

    def _run_automatic_bone_segmentation(self):
        currentSegment = self._parameterNode.GetParameter("SAMCurrentSegment")
        currentSliceIndex = self.getIndexOfCurrentSlice()
        previouslyProducedMask = get_annotation_slice(
            self.segmentIdToSegmentationMask[currentSegment],
            self.sliceAccessorDimension,
            currentSliceIndex,
        )

        if np.any(previouslyProducedMask):
            segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")
            currentLabel = segmentationNode.GetSegmentation().GetSegment(currentSegment).GetName()

            confirmed = slicer.util.confirmOkCancelDisplay(
                f"Are you sure you want to re-annotate {currentLabel} for the current slice? "
                f"All of your previous annotation for {currentLabel} in the current slice will be removed!",
                windowTitle="Warning",
            )
            if not confirmed:
                return

        if not self.featuresAreExtracted:
            self.extractFeatures()
            self.featuresAreExtracted = True

        roiList = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for roiNode in roiList:
            slicer.mrmlScene.RemoveNode(roiNode)

        for currentSliceIndex in range(self.getTotalNumberOfSlices()):
            with open(self.featuresFolder + "/slice_" + str(currentSliceIndex) + "_features.pkl", "rb") as f:
                self.sam.features = pickle.load(f)

            self.init_masks, _, _ = self.sam.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=True,
                return_logits=False,
            )

            self.init_masks = self.init_masks > self.mask_threshold
            if self.init_masks is not None:
                if self._parameterNode.GetParameter("SAMCurrentMask") == "Mask-1":
                    producedMask = self.init_masks[1][:]
                else:
                    producedMask = self.init_masks[0][:]
            else:
                producedMask = np.full(self.sam.original_size, False)

            self._store_mask_for_segment(currentSegment, currentSliceIndex, producedMask)

        self._update_segment_in_scene(currentSegment)

    def _run_automatic_muscle_segmentation(self):
        self.muscleSegmentId = self.samSegmentationNode.GetSegmentation().AddEmptySegment("muscle")
        self.createSlices()

        slices = []
        sliceShape = None
        for currentSliceIndex in range(self.getTotalNumberOfSlices()):
            sliceData = np.load(self.slicesFolder + "/" + f"slice_{currentSliceIndex}.npy")
            sliceShape = sliceData.shape
            img = Image.fromarray(sliceData).convert("RGB")
            slices.append(img)

        for currentSliceIndex in range(self.getTotalNumberOfSlices()):
            img = slices[currentSliceIndex]
            img = self.model_manager.transforms.Resize((1024, 1024))(img)
            img = self.model_manager.transforms.ToTensor()(img)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = self.model_manager.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )(img)
            img = img.unsqueeze(0)

            checkpointLoc = self.resourcePath("UI") + "/../../models/segment_any_muscle/segmentanymuscle.pth"
            mask = self.model_manager.seg_any_muscle_pred_one_image(
                img,
                checkpointLocation=checkpointLoc,
            )
            self.model_manager.save_image(mask, self.slicesFolder + "/" + f"slice_{currentSliceIndex}_pred.png")
            mask = self.model_manager.transforms.Resize((sliceShape[0], sliceShape[1]))(mask.unsqueeze(0))
            mask = mask.detach().cpu().numpy().astype(bool)

            self._store_mask_for_segment(self.muscleSegmentId, currentSliceIndex, mask[0, 0])

        self._update_segment_in_scene(self.muscleSegmentId)

    def _run_automatic_ct_segmentation(self):
        volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        inputPath = self.resourcePath("UI") + "/../../models/ct_segmentation/demo/example.nii.gz"
        outputPath = self.resourcePath("UI") + "/../../models/ct_segmentation/results"
        slicer.util.saveNode(volumeNode, inputPath)

        os.system("python C:/Users/zafry/SlicerSegmentHumanBody/SegmentHumanBody/models/ct_segmentation/predict_muscle_fat.py")
        slicer.util.loadSegmentation(self.resourcePath("UI") + "/../../models/ct_segmentation/results/example_segmentation.nii.gz")

        ctSegmentationNode = slicer.mrmlScene.GetFirstNodeByName("example_segmentation.nii.gz")
        ctSegmentationNode.GetSegmentation().GetNthSegment(0).SetName("muscle")
        ctSegmentationNode.GetSegmentation().GetNthSegment(1).SetName("subcutaneous_fat")
        ctSegmentationNode.GetSegmentation().GetNthSegment(2).SetName("visceral_fat")
        ctSegmentationNode.GetSegmentation().GetNthSegment(3).SetName("muscular_fat")

        with open(outputPath + "/metrics.txt") as f:
            data = f.read()
        js = json.loads(data)

        slicer.util.confirmOkCancelDisplay(
            text=json.dumps(js, indent=4),
            windowTitle="Metrics",
        )

    def _run_automatic_breast_segmentation(self):
        self.breastTissueSegmentId = self.samSegmentationNode.GetSegmentation().AddEmptySegment("breast tissue")
        self.breastVesselSegmentId = self.samSegmentationNode.GetSegmentation().AddEmptySegment("breast vessel")

        self.createSlices()
        slices = []
        for currentSliceIndex in range(self.getTotalNumberOfSlices()):
            sliceData = np.load(self.slicesFolder + "/" + f"slice_{currentSliceIndex}.npy")
            slices.append(sliceData)

        volume = np.stack(slices, axis=2)
        checkpointPath = self.resourcePath("UI") + "/../../models/breast_model/vnet_with_aug.pth"
        predictedVolume = self.model_manager.breastModelPredict(volume, checkpointPath, self.device)

        for sliceIndex in range(self.getTotalNumberOfSlices()):
            tissueMask = predictedVolume[2, :, :, sliceIndex].astype(bool)
            vesselMask = predictedVolume[1, :, :, sliceIndex].astype(bool)

            self._store_mask_for_segment(self.breastTissueSegmentId, sliceIndex, tissueMask)
            self._store_mask_for_segment(self.breastVesselSegmentId, sliceIndex, vesselMask)

        self._update_segment_in_scene(self.breastTissueSegmentId)
        self._update_segment_in_scene(self.breastVesselSegmentId)

    def onStartSegmentation(self):
        if not self.initializeVariables():
            return

        currentSegment = self._parameterNode.GetParameter("SAMCurrentSegment")
        currentSliceIndex = self.getIndexOfCurrentSlice()
        previouslyProducedMask = get_annotation_slice(
            self.segmentIdToSegmentationMask[currentSegment],
            self.sliceAccessorDimension,
            currentSliceIndex,
        )

        if np.any(previouslyProducedMask):
            segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")
            currentLabel = segmentationNode.GetSegmentation().GetSegment(currentSegment).GetName()

            confirmed = slicer.util.confirmOkCancelDisplay(
                f"Are you sure you want to re-annotate {currentLabel} for the current slice? "
                f"All of your previous annotation for {currentLabel} in the current slice will be removed!",
                windowTitle="Warning",
            )
            if not confirmed:
                return

        if not self.featuresAreExtracted:
            self.extractFeatures()
            self.featuresAreExtracted = True

        roiList = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for roiNode in roiList:
            slicer.mrmlScene.RemoveNode(roiNode)

        self.currentlySegmenting = True
        self.initializeSegmentationProcess()
        self.collectPromptInputsAndPredictSegmentationMask()
        self.updateSegmentationScene()

    def updateSegmentationScene(self):
        if self.currentlySegmenting:
            currentSegment = self._parameterNode.GetParameter("SAMCurrentSegment")
            currentSliceIndex = self.getIndexOfCurrentSlice()
            self._store_mask_for_segment(currentSegment, currentSliceIndex, self.producedMask)
            self._update_segment_in_scene(currentSegment)

        qt.QTimer.singleShot(100, self.updateSegmentationScene)

    def combineMultipleMasks(self, masks):
        return combine_multiple_masks(masks)

    def collectPromptInputsAndPredictSegmentationMask(self):
        if self.currentlySegmenting:
            self.isTherePromptBoxes = False
            self.isTherePromptPoints = False
            currentSliceIndex = self.getIndexOfCurrentSlice()

            positivePromptPointList = self._collect_visible_prompt_points(self.positivePromptPointsNode)
            negativePromptPointList = self._collect_visible_prompt_points(self.negativePromptPointsNode)

            promptPointCoordinations = positivePromptPointList + negativePromptPointList
            promptPointLabels = [1] * len(positivePromptPointList) + [0] * len(negativePromptPointList)

            if len(promptPointCoordinations) != 0:
                self.isTherePromptPoints = True

            boxList = []
            roiBoxes = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")

            for roiBox in roiBoxes:
                boxBounds = np.zeros(6)
                roiBox.GetBounds(boxBounds)
                minBoundaries = self.volumeRasToIjk.MultiplyPoint([boxBounds[0], boxBounds[2], boxBounds[4], 1])
                maxBoundaries = self.volumeRasToIjk.MultiplyPoint([boxBounds[1], boxBounds[3], boxBounds[5], 1])

                if self.sliceAccessorDimension == 2:
                    boxList.append([maxBoundaries[1], maxBoundaries[2], minBoundaries[1], minBoundaries[2]])
                elif self.sliceAccessorDimension == 1:
                    boxList.append([maxBoundaries[0], maxBoundaries[2], minBoundaries[0], minBoundaries[2]])
                else:
                    boxList.append([maxBoundaries[0], maxBoundaries[1], minBoundaries[0], minBoundaries[1]])

            if len(boxList) != 0:
                self.isTherePromptBoxes = True

            with open(self.featuresFolder + "/" + f"slice_{currentSliceIndex}_features.pkl", "rb") as f:
                self.sam.features = pickle.load(f)

            if self.isTherePromptBoxes and not self.isTherePromptPoints:
                inputBoxes = self.model_manager.torch.tensor(boxList, device=self.device)
                transformedBoxes = self.sam.transform.apply_boxes_torch(inputBoxes, self.imageShape)

                self.masks, _, _ = self.sam.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformedBoxes,
                    multimask_output=True,
                    return_logits=False,
                )

                self.masks = self.masks.cpu().numpy()
                self.masks = combine_multiple_masks(self.masks)

            elif self.isTherePromptPoints and not self.isTherePromptBoxes:
                self.masks, _, _ = self.sam.predict(
                    point_coords=np.array(promptPointCoordinations),
                    point_labels=np.array(promptPointLabels),
                    multimask_output=True,
                    return_logits=False,
                )

            elif self.isTherePromptBoxes and self.isTherePromptPoints:
                self.masks, _, _ = self.sam.predict(
                    point_coords=np.array(promptPointCoordinations),
                    point_labels=np.array(promptPointLabels),
                    box=np.array(boxList[0]),
                    multimask_output=True,
                    return_logits=False,
                )
            else:
                self.masks = None

            if self.masks is not None:
                self.masks = self.masks > self.mask_threshold

            if self.masks is not None:
                if self._parameterNode.GetParameter("SAMCurrentMask") == "Mask-1":
                    self.producedMask = self.masks[1][:]
                elif self._parameterNode.GetParameter("SAMCurrentMask") == "Mask-2":
                    self.producedMask = self.masks[0][:]
            else:
                self.producedMask = np.full(self.sam.original_size, False)

            qt.QTimer.singleShot(100, self.collectPromptInputsAndPredictSegmentationMask)

    def extractFeatures(self):
        with slicer.util.MessageDialog("Please wait until SAM has processed the input."):
            with slicer.util.WaitCursor():
                self.createSlices()
                self.createFeatures()

        print("Features are extracted. You can start segmentation by placing prompt points or ROIs (boundary boxes)!")


class SegmentHumanBodyLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()

    def setDefaultParameters(self, parameterNode):
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")