import os
import shutil


class ModelManager:
    def __init__(self, widget):
        self.widget = widget

        self.device = None
        self.torch = None
        self.transforms = None
        self.save_image = None

        self.sab_model_registry = None
        self.SabPredictor = None
        self.seg_any_muscle_pred_one_image = None
        self.SAM2AnnotationTool = None
        self.breastModelPredict = None
        self.ct_segmentator = None

    def initialize_environment(self):
        self._show_license_warning()
        self._ensure_torch()
        self._ensure_python_packages()
        self._ensure_repo_assets()
        self._ensure_model_files()
        self._import_runtime_modules()
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        print("Working on", self.device)

    def _show_license_warning(self):
        import slicer

        slicer.util.warningDisplay(
            "The model and the extension is licensed under the CC BY-NC 4.0 license!\n\n"
            "Please also note that this software is developed for research purposes and is not intended "
            "for clinical use yet. Users should exercise caution and are advised against employing it "
            "immediately in clinical or medical settings."
        )

    def _ensure_torch(self):
        import slicer

        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            raise RuntimeError("You need to install PyTorch extension from the Extensions Manager.")

        minimumTorchVersion = "2.0.0"
        minimumTorchVisionVersion = "0.15.0"
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()

        if not torchLogic.torchInstalled():
            if slicer.util.confirmOkCancelDisplay(
                "PyTorch Python package is required. Would you like to install it now? "
                "(it may take several minutes)"
            ):
                torch_module = torchLogic.installTorch(
                    askConfirmation=True,
                    forceComputationBackend="cu117",
                    torchVersionRequirement=f">={minimumTorchVersion}",
                    torchvisionVersionRequirement=f">={minimumTorchVisionVersion}",
                )
                if torch_module is None:
                    raise ValueError("You need to install PyTorch to use SegmentHumanBody!")
        else:
            from packaging import version

            if version.parse(torchLogic.torch.__version__) < version.parse(minimumTorchVersion):
                raise ValueError(
                    f"PyTorch version {torchLogic.torch.__version__} is not compatible with this module. "
                    f"Minimum required version is {minimumTorchVersion}."
                )

    def _ensure_python_packages(self):
        packages = [
            ("timm", "timm"),
            ("einops", "einops"),
            ("pandas", "pandas"),
            ("batchgenerators", "batchgenerators"),
            ("nnunetv2", "nnunetv2"),
            ("pywintypes", "pypiwin32"),
            ("blosc2", "blosc2"),
            ("tensordict", "tensordict"),
            ("tensorboard", "tensorboard"),
            ("nibabel", "nibabel"),
            ("nrrd", "pynrrd"),
            ("submitit", "submitit"),
            ("hydra", "hydra-core"),
            ("iopath", "iopath"),
            ("ruamel.yaml", "ruamel.yaml"),
            ("gdown", "gdown"),
            ("git", "gitpython"),
        ]

        for import_name, pip_name in packages:
            self._ensure_package(import_name, pip_name)

        try:
            from segment_anything import sam_model_registry, SamPredictor  # noqa: F401
        except ModuleNotFoundError:
            import slicer
            if slicer.util.confirmOkCancelDisplay(
                "'segment-anything' is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install(
                    "https://github.com/facebookresearch/segment-anything/archive/"
                    "6fdee8f2727f4506cfbbe553e23b895e27956588.zip"
                )

        try:
            from segment_anything import sam_model_registry, SamPredictor  # noqa: F401
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem installing 'segment-anything'. Please try again.")

    def _ensure_package(self, import_name, pip_name):
        import slicer

        try:
            __import__(import_name)
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                f"'{pip_name}' package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install(pip_name)

        try:
            __import__(import_name)
        except ModuleNotFoundError:
            raise RuntimeError(f"There is a problem installing '{pip_name}'. Please try again.")

    def _ensure_repo_assets(self):
        import git

        os.system("git config --global core.longpaths true")

        copyFolder = os.path.normpath(self.widget.resourcePath("UI") + "/../../../repo_copy")
        if not os.path.exists(copyFolder):
            os.makedirs(copyFolder, exist_ok=True)
            git.Repo.clone_from(
                "https://github.com/mazurowski-lab/SlicerSegmentHumanBody",
                copyFolder,
            )

        modelRoot = os.path.normpath(self.widget.resourcePath("UI") + "/../../models")
        self._move_if_missing(
            os.path.join(copyFolder, "SegmentHumanBody", "models", "breast_model"),
            os.path.join(modelRoot, "breast_model"),
        )
        self._move_if_missing(
            os.path.join(copyFolder, "SegmentHumanBody", "models", "ct_segmentation"),
            os.path.join(modelRoot, "ct_segmentation"),
        )
        self._move_if_missing(
            os.path.join(copyFolder, "SegmentHumanBody", "models", "sam2_annotation_tool"),
            os.path.join(modelRoot, "sam2_annotation_tool"),
        )

    def _move_if_missing(self, src, dst):
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.move(src, dst)

    def _ensure_model_files(self):
        import gdown
        import slicer

        modelRoot = os.path.normpath(self.widget.resourcePath("UI") + "/../../models")

        breast_checkpoint = os.path.join(modelRoot, "breast_model", "vnet_with_aug.pth")
        if not os.path.exists(breast_checkpoint):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use breast segmentation model? Click OK to install it now!"
            ):
                url = "https://drive.google.com/uc?id=1IQu_8hYnvAR1_GSKpzkf5l7mqlcQpq3j"
                os.makedirs(os.path.dirname(breast_checkpoint), exist_ok=True)
                gdown.download(url, breast_checkpoint, quiet=False)

        ct_checkpoint = os.path.join(
            modelRoot,
            "ct_segmentation",
            "nnUNetTrainer__nnUNetResEncUNetXLPlans__2d",
            "fold_5",
            "checkpoint_final.pth",
        )
        if not os.path.exists(ct_checkpoint):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use CT Segmentation model? Click OK to install it now!"
            ):
                url = "https://drive.google.com/uc?id=10Pt3nLXxx9nbikhsrTiMH2HQ5p9-GiLB"
                os.makedirs(os.path.dirname(ct_checkpoint), exist_ok=True)
                gdown.download(url, ct_checkpoint, quiet=False)

        sam2_checkpoint = os.path.join(
            modelRoot,
            "sam2_annotation_tool",
            "sam2_logs",
            "configs",
            "sam2.1_training",
            "lstm_sam2.1_hiera_t.yaml",
            "checkpoints",
            "checkpoint.pt",
        )
        if not os.path.exists(sam2_checkpoint):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SLM-SAM2 model? Click OK to install it now!"
            ):
                url = "https://drive.google.com/uc?id=1uTL1KWjYTIf27_Rs-H5Umr3PogfX1lyB"
                os.makedirs(os.path.dirname(sam2_checkpoint), exist_ok=True)
                gdown.download(url, sam2_checkpoint, quiet=False)

    def _import_runtime_modules(self):
        import torch
        from torchvision import transforms
        from torchvision.utils import save_image

        from models.sam import sam_model_registry as sab_model_registry
        from models.sam import SamPredictor as SabPredictor
        from models.segment_any_muscle.main import predict_one_image as seg_any_muscle_pred_one_image
        from models.sam2_annotation_tool.annotation_tool import SAM2AnnotationTool
        from models.breast_model.predict_mask_singleimage import (
            breast_model_predict_volume as breastModelPredict,
        )
        from models.ct_segmentation import predict_muscle_fat as ct_segmentator

        self.torch = torch
        self.transforms = transforms
        self.save_image = save_image
        self.sab_model_registry = sab_model_registry
        self.SabPredictor = SabPredictor
        self.seg_any_muscle_pred_one_image = seg_any_muscle_pred_one_image
        self.SAM2AnnotationTool = SAM2AnnotationTool
        self.breastModelPredict = breastModelPredict
        self.ct_segmentator = ct_segmentator

    def build_bone_predictor(self, args, modelVersion, modelCheckpoint):
        model = self.sab_model_registry[modelVersion](
            args,
            checkpoint=modelCheckpoint,
            num_classes=2,
        )
        model.to(device=self.device).eval()
        return self.SabPredictor(model)

    def build_sam2_annotation_tool(self):
        return self.SAM2AnnotationTool(
            ckpt_folder=self.widget.resourcePath("UI") + "/../../models/sam2_annotation_tool/checkpoints",
            cfg_folder=self.widget.resourcePath("UI") + "/../../models/sam2_annotation_tool/sam2/configs/sam2.1_training",
            search_path=self.widget.resourcePath("UI") + "/../../models/sam2_annotation_tool/configs/sam2.1_training",
        )