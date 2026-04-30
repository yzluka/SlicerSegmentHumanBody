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
        'positivePrompts', 'positivePromptLabel',
        'negativePrompts', 'negativePromptLabel',
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
        'positivePrompts', 'positivePromptLabel',
        'negativePrompts', 'negativePromptLabel',
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
        'positivePrompts', 'positivePromptLabel',
        'negativePrompts', 'negativePromptLabel',
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
# TimedMarker family
# ---------------------------------------------------------------------------

class TimedAnnotatorFamily(BaseModelFamily):
    """Thin delegate shell for the TimedMarker workflow.

    All logic lives in TimedAnnotatorModel (core/models/timed_annotator.py),
    which is loaded from ModelRegistry on confirm.  The family exposes only
    the hook interface the widget expects and forwards each call to the model.
    """

    VARIANTS: list = []   # no model weights; auto-confirmed on family switch

    VISIBLE_BUTTONS = frozenset({
        'exportAnnotationLogButton', 'importAnnotationLogButton',
        'positivePrompts', 'positivePromptLabel',
    })

    def confirm_model(self):
        self.model = ModelRegistry.get_model('TimedAnnotatorModel')

    def on_segment_created(self, segment_id, seg_name, segmentation_node=None):
        self.model.on_segment_created(segment_id, seg_name, segmentation_node=segmentation_node)

    def on_point_confirmed(self, segment_id, ras, cp_id, is_negative=False,
                           volume_node=None, segmentation_node=None):
        self.model.on_point_confirmed(segment_id, ras, cp_id, is_negative,
                                      volume_node=volume_node, segmentation_node=segmentation_node)

    def on_point_undone(self, cp_id):
        self.model.on_point_undone(cp_id)

    def sync_visibility(self, current_seg_id, current_visible, saved_visible):
        self.model.sync_visibility(current_seg_id, current_visible, saved_visible)

    def export_data(self):
        return self.model.export_data()

    def on_export(self, widget):
        self.model.on_export(widget)

    def on_import(self, widget):
        self.model.on_import(widget)


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
    'TimedMarker':               TimedAnnotatorFamily,
}
