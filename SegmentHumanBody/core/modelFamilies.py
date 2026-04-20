"""Model family classes for the SegmentHumanBody Slicer module.

Each family encapsulates one "style" of segmentation and controls:
  - which model variants are available (``VARIANTS``)
  - which UI buttons should be visible (``VISIBLE_BUTTONS``)
  - how inference is triggered (``onRender``, ``on_expand``, etc.)

UI button visibility is driven solely by ``VISIBLE_BUTTONS``; adding a new
button to a family requires only adding its UI name to that set.

``FAMILY_REGISTRY`` maps display-name strings to family classes and is the
single source of truth used by the widget's model-family dropdown.
"""

import numpy as np
from .modelRegistry import ModelRegistry
from .utils import labels_at_points


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseModelFamily:
    """Root class shared by all model families.

    Subclasses should override ``VARIANTS``, ``VISIBLE_BUTTONS``, and any
    inference methods they support.
    """

    VARIANTS: list = ['None']

    # Set of UI widget names that should be *visible* for this family.
    # The widget's updateUIVisibility() iterates a fixed list of managed
    # buttons and shows/hides each based on membership in this set.
    VISIBLE_BUTTONS: frozenset = frozenset()

    def __init__(self, variant=None):
        self.variant = variant
        self.model = None

    def confirm_model(self):
        """Load the model for the currently selected variant.

        Raises
        ------
        ValueError
            If no variant has been selected.
        """
        if not self.variant:
            raise ValueError(
                "No model variant selected. Choose a variant before confirming."
            )
        self.model = ModelRegistry.get_model(self.variant)


# ---------------------------------------------------------------------------
# SAM family
# ---------------------------------------------------------------------------

class SAMFamily(BaseModelFamily):
    """SAM-based interactive segmentation (not yet fully implemented)."""

    VARIANTS = [
        'SAM-VIT-H', 'SAM-ViT-L', 'SAM-ViT-B',
        'sam2_hiera_l', 'sam2_hiera_b+', 'sam2_hiera_s', 'sam2_hiera_t',
    ]

    VISIBLE_BUTTONS = frozenset({
        'goToMarkupsButton',
        'samMaskDropdown',
    })

    def get_requested_mask(self, **kwargs):
        """Placeholder — controls samMaskDropdown visibility."""
        pass

    def onRender(self, **kwargs):
        """Placeholder — SAM interactive rendering not yet implemented."""
        return None


# ---------------------------------------------------------------------------
# SPX family
# ---------------------------------------------------------------------------

class SPXModelFamily(BaseModelFamily):
    """Superpixel-assisted annotation family."""

    MODEL_MAP = {
        'SLIC-2D':          'SPX_SLIC2D',
        'Felzenszwalb-2D':  'SPX_Felzenszwalb2D',
        'Naive_Grid-2D':    'SPX_Tester2D',
    }
    VARIANTS = sorted(list(MODEL_MAP.keys()))

    VISIBLE_BUTTONS = frozenset({
        'expandSelectedLabelButton',
        'showSPXBoundaryCheckBox',
        'goToMarkupsButton',
    })

    def __init__(self, variant=None):
        super().__init__(variant)
        # Cache the last computed superpixel label map so re-renders on the
        # same slice with the same parameters skip the expensive forward pass.
        self._cache_key    = None
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
        # A new model produces different labels for the same image — drop cache.
        self._cache_key    = None
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

    def on_expand(self, **kwargs):
        if not self.model:
            raise RuntimeError("Model not confirmed")

        img = kwargs.get('img')
        if img is None:
            raise ValueError("on_expand requires 'img' keyword argument")

        # Strip 'img' so only algorithm params reach model.forward and cache key.
        model_kwargs = {k: v for k, v in kwargs.items() if k != 'img'}

        # Reuse cached labels when the user has been working on this slice in
        # interactive mode — avoids a redundant forward pass.
        key = self._make_cache_key(img, model_kwargs)
        if key != self._cache_key:
            self._cache_labels = self.model.forward(img=img, **model_kwargs)
            self._cache_key = key

        return self._cache_labels

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
        pos_labels = labels_at_points(pos_points, labels)
        neg_labels = labels_at_points(neg_points, labels)

        # Neg has priority: a label under a neg point is never added by pos.
        pos_only_labels = pos_labels - neg_labels

        if base_mask is not None:
            # Additive / subtractive mode — used in interactive sessions that
            # have a pre-existing painted region:
            #   result = (base_slice | pos_region) & ~neg_region
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


# ---------------------------------------------------------------------------
# Auto family
# ---------------------------------------------------------------------------

class AutoModelFamily(BaseModelFamily):
    """Automated (non-interactive) segmentation family."""

    VARIANTS = ['BreastCT', 'PE_SEG']

    VISIBLE_BUTTONS = frozenset({
        'assignLabel2D',
        'assignLabel3D',
        'runAutomaticSegmentation',
    })

    def on_assign_2d(self, **kwargs):
        pass

    def on_assign_3d(self, **kwargs):
        pass

    def on_automatic_segmentation(self, **kwargs):
        if not self.model:
            raise RuntimeError("Model not confirmed")
        if "img" not in kwargs:
            raise ValueError("Missing required argument: img")
        return self.model.forward(**kwargs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Single source of truth for the widget's model-family dropdown.
# Keys are the display names shown in the UI; values are family classes.
# Add a new family by adding one entry here — no widget edits required.
FAMILY_REGISTRY: dict = {
    'None':                      BaseModelFamily,
    'SAM-Style':                 SAMFamily,
    'SPX-Assisted Annotation':   SPXModelFamily,
    'Auto':                      AutoModelFamily,
}
