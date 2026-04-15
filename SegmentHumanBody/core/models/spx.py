"""Concrete superpixel (SPX) algorithm implementations.

Each class exposes:
  - ``PARAM_HINT``  — example parameter string shown in the UI
  - ``DOC_URL``     — link to upstream documentation (optional)
  - ``forward(**kwargs)`` — pops ``img`` from kwargs; returns an integer label map
"""

import numpy as np


class SPX_Tester2D:
    """Naive uniform-grid superpixels — useful for debugging the SPX pipeline.

    Divides the image into a regular gh × gw grid.  Each cell receives a
    unique integer label.  Fully vectorised; no Python-level loops.
    """

    DOC_URL = None
    PARAM_HINT = 'gh=9, gw=9'

    def forward(self, **kwargs):
        img = kwargs["img"]
        H, W = img.shape[:2]
        gh = int(kwargs.get('gh', 9))
        gw = int(kwargs.get('gw', 9))

        y_coords = np.linspace(0, gh, H, endpoint=False).astype(np.int32)
        x_coords = np.linspace(0, gw, W, endpoint=False).astype(np.int32)

        # Broadcast to (H, W) without any Python loops.
        return (y_coords[:, np.newaxis] * gw + x_coords[np.newaxis, :] + 1).astype(np.int32)


class SPX_SLIC2D:
    """SLIC superpixel segmentation via scikit-image."""

    DOC_URL = "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic"
    PARAM_HINT = "n_segments=100, compactness=10, sigma=1"

    def __init__(self):
        from skimage.segmentation import slic
        self._slic = slic

    def forward(self, **kwargs):
        img = kwargs.pop("img")
        if img is None:
            raise ValueError("Missing required argument: img")
        # CT slices are 2-D grayscale; slic defaults to channel_axis=-1
        # (multi-channel) which raises on a plain 2-D array.
        if img.ndim == 2:
            kwargs.setdefault('channel_axis', None)
        return self._slic(img, **kwargs)


class SPX_Felzenszwalb2D:
    """Felzenszwalb graph-based superpixel segmentation via scikit-image."""

    DOC_URL = "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb"
    PARAM_HINT = "scale=100, sigma=0.5, min_size=50"

    def __init__(self):
        from skimage.segmentation import felzenszwalb
        self._felzenszwalb = felzenszwalb

    def forward(self, **kwargs):
        img = kwargs.pop("img")
        if img is None:
            raise ValueError("Missing required argument: img")
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        return self._felzenszwalb(img, **kwargs)
