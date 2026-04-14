import numpy as np
from .modelRegistry import ModelRegistry

class BaseModelFamily:
    VARIANTS = ['None']

    def __init__(self, variant=None):
        self.variant = variant
        self.model = None
        #print(f"INIT CALLED: {type(self).__name__}")

    def confirm_model(self):
        if not self.variant:
            #print("[Confirm] No variant selected")
            return

        #print(f"[Confirm] {type(self).__name__} → {self.variant}")

        self.model = ModelRegistry.get_model(self.variant)


# ------------------------
# Interactive Model
# ------------------------

class SAMFamily(BaseModelFamily):
    VARIANTS = [
        'SAM-VIT-H','SAM-ViT-L','SAM-ViT-B',
        'sam2_hiera_l','sam2_hiera_b+','sam2_hiera_s','sam2_hiera_t'
    ]

    def on_enter_interactive(self, **kwargs):
        pass

    def on_stop_interactive(self, **kwargs):
        pass
    
    def get_requested_mask(self, **kwargs):
        pass
    def onRender(self, **kwargs):
        pass

# ------------------------
# SPX Model
# ------------------------

class SPXModelFamily(BaseModelFamily):

    MODEL_MAP = {
        'SLIC-2D': 'SPX_SLIC2D',
        'Felzenszwalb-2D': 'SPX_Felzenszwalb2D',
        'Naive_Grid-2D': 'SPX_Tester2D'
    }
    VARIANTS = sorted(list(MODEL_MAP.keys()))

    def __init__(self, variant=None):
        super().__init__(variant)
        # Cache the last computed superpixel label map so re-renders on the
        # same slice with the same parameters skip the expensive forward pass.
        self._cache_key = None
        self._cache_labels = None

    def _get_model_key(self):
        if not self.variant:
            raise ValueError("No variant selected")

        if self.variant not in self.MODEL_MAP:
            raise ValueError(f"Unknown variant: {self.variant}")

        return self.MODEL_MAP[self.variant]

    def confirm_model(self):
        model_key = self._get_model_key()
        self.model = ModelRegistry.get_model(model_key)
        # A new model produces different labels for the same image — drop the cache.
        self._cache_key = None
        self._cache_labels = None

    def _make_cache_key(self, img, kwargs):
        # img.ctypes.data is the pointer to the first byte of the slice in the
        # volume buffer.  For a view (which get_slice_from_volume always returns),
        # this pointer uniquely identifies the slice position within the volume
        # without copying any pixel data.
        try:
            params = frozenset(kwargs.items())
        except TypeError:
            params = str(sorted(kwargs.items()))
        return (img.ctypes.data, img.shape, img.dtype.str, params)

    def on_propagate(self, **kwargs):
        if not self.model:
            raise RuntimeError("Model not confirmed")

        img = kwargs.get('img')
        if img is None:
            raise ValueError("on_propagate requires 'img' keyword argument")

        # Strip 'img' so only algorithm params reach model.forward and the cache key.
        model_kwargs = {k: v for k, v in kwargs.items() if k != 'img'}

        # Reuse the label map already in cache when the user has been working
        # on this slice in interactive mode — avoids a redundant forward pass.
        key = self._make_cache_key(img, model_kwargs)
        if key != self._cache_key:
            self._cache_labels = self.model.forward(img=img, **model_kwargs)
            self._cache_key = key

        return self._cache_labels

    def on_enter_interactive(self, **kwargs):
        pass

    def on_stop_interactive(self, **kwargs):
        pass

    def onRender(self, img, pos_points, neg_points, **kwargs):
        if not self.model:
            return None

        # Pop base_mask before building the SPX label-map cache key.
        # The label map depends only on img content and algorithm params,
        # not on the pre-existing painted slice.
        base_mask = kwargs.pop('base_mask', None)

        key = self._make_cache_key(img, kwargs)
        if key != self._cache_key:
            self._cache_labels = self.model.forward(img=img, **kwargs)
            self._cache_key = key

        labels = self._cache_labels

        # Collect labels under positive and negative prompts.
        pos_labels = set()
        for x, y in pos_points:
            if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                pos_labels.add(labels[y, x])

        neg_labels = set()
        for x, y in neg_points:
            if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                neg_labels.add(labels[y, x])

        # Neg has priority: a label under a neg point is never added by pos.
        pos_only_labels = pos_labels - neg_labels

        if base_mask is not None:
            # Additive / subtractive mode — used in interactive sessions that
            # have a pre-existing painted region:
            #   result = (base_slice | pos_region) & ~neg_region
            # Neg points erase from whatever is painted, including data that
            # predates the current session; pos points add to it.
            pos_region = (np.isin(labels, list(pos_only_labels))
                          if pos_only_labels else np.zeros(labels.shape, dtype=bool))
            neg_region = (np.isin(labels, list(neg_labels))
                          if neg_labels else np.zeros(labels.shape, dtype=bool))
            return np.where(neg_region, 0,
                            np.where(pos_region, 1, base_mask)).astype(np.uint8)

        # Classic mode (no base mask): derive result entirely from prompts.
        if not pos_only_labels:
            return None

        return np.isin(labels, list(pos_only_labels)).astype(np.uint8)


# ------------------------
# Auto Model
# ------------------------

class AutoModelFamily(BaseModelFamily):
    VARIANTS = ['BreastCT', 'PE_SEG']

    def on_assign_2d(self, **kwargs):
        #print("[Interactive] assign 2D")
        pass

    def on_assign_3d(self, **kwargs):
        #print("[Interactive] assign 3D")
        pass

    def on_automatic_segmentation(self, **kwargs):
        if not self.model:
            raise RuntimeError("Model not confirmed")

        if "img" not in kwargs:
            raise ValueError("Missing required argument: img")

        return self.model.forward(**kwargs)