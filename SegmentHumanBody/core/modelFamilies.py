from .modelRegistry import ModelRegistry

class BaseModelFamily:
    VARIANTS = ['None']

    def __init__(self, variant=None):
        self.variant = variant
        self.model = None
        print(f"INIT CALLED: {type(self).__name__}")

    def confirm_model(self):
        if not self.variant:
            print("[Confirm] No variant selected")
            return

        print(f"[Confirm] {type(self).__name__} → {self.variant}")

        self.model = ModelRegistry.get_model(self.variant)


# ------------------------
# Interactive Model
# ------------------------

class InteractiveModelFamily(BaseModelFamily):
    VARIANTS = [
        'SAM-VIT-H','SAM-ViT-L','SAM-ViT-B',
        'sam2_hiera_l','sam2_hiera_b+','sam2_hiera_s','sam2_hiera_t'
    ]

    def on_assign_2d(self, **kwargs):
        print("[Interactive] assign 2D")

    def on_assign_3d(self, **kwargs):
        print("[Interactive] assign 3D")

    def on_enter_interactive(self, **kwargs):
        print("[Interactive] start")

    def on_stop_interactive(self, **kwargs):
        print("[Interactive] stop")
    
    def get_requested_mask(self, **kwargs):
        pass

# ------------------------
# SPX Model
# ------------------------

class SPXModelFamily(BaseModelFamily):
    VARIANTS = ['SLIC', 'Felzenszwalb','Naive_Grid']

    MODEL_MAP = {
        'SLIC': None,
        'Felzenszwalb': None,
        'Naive_Grid': 'SPX_Tester'
    }

    def confirm_model(self):
        model_key = self._get_model_key()
        print(f"[SPX] Loading model: {model_key}")
        self.model = ModelRegistry.get_model(model_key)

    def on_propagate(self, **kwargs):
        if not self.model:
            raise RuntimeError("Model not confirmed")

        return self.model.forward(**kwargs)

    def _get_model_key(self):
        if not self.variant:
            raise ValueError("No variant selected")

        if self.variant not in self.MODEL_MAP:
            raise ValueError(f"Unknown variant: {self.variant}")

        return self.MODEL_MAP[self.variant]


# ------------------------
# Auto Model
# ------------------------

class AutoModelFamily(BaseModelFamily):
    VARIANTS = ['BreastCT', 'PE_SEG']

    def on_automatic_segmentation(self, **kwargs):
        if not self.model:
            raise RuntimeError("Model not confirmed")

        if "img" not in kwargs:
            raise ValueError("Missing required argument: img")

        return self.model.forward(**kwargs)