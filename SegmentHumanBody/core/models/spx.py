"""Concrete superpixel (SPX) algorithm implementations.

Each subclass of ``SPXModel`` must:
  - define ``PARAM_HINT``  — example parameter string shown in the UI
  - define ``DOC_URL``     — link to upstream documentation (or None)
  - implement ``forward(img, **kwargs)`` — returns an integer label map

Callers receive a label map of the same H×W shape as the input image.
"""

import numpy as np
from abc import ABC, abstractmethod
from .._deps import DependencyCheck


class SPXModel(ABC):
    """Abstract base for all SPX algorithm implementations.

    Enforces the interface contract so that adding a new algorithm requires
    only subclassing this class and registering it in ModelRegistry.
    """

    PARAM_HINT: str = ''
    DOC_URL: str | None = None

    @abstractmethod
    def forward(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """Run the superpixel algorithm on *img*.

        Parameters
        ----------
        img : (H, W) ndarray — 2-D image slice.
        **kwargs — algorithm-specific parameters (e.g. n_segments, sigma).

        Returns
        -------
        (H, W) ndarray of non-negative integers: one value per pixel, same
        value for pixels belonging to the same superpixel region.
        """


class SPX_Tester2D(SPXModel):
    """Naive uniform-grid superpixels — useful for debugging the SPX pipeline.

    Divides the image into a regular gh × gw grid.  Each cell receives a
    unique integer label.  Fully vectorised; no Python-level loops.
    """

    DOC_URL = None
    PARAM_HINT = 'gh=9, gw=9'

    def forward(self, img: np.ndarray, **kwargs) -> np.ndarray:
        H, W = img.shape[:2]
        gh = int(kwargs.get('gh', 9))
        gw = int(kwargs.get('gw', 9))

        y_coords = np.linspace(0, gh, H, endpoint=False).astype(np.int32)
        x_coords = np.linspace(0, gw, W, endpoint=False).astype(np.int32)

        # Broadcast to (H, W) without any Python loops.
        return (y_coords[:, np.newaxis] * gw + x_coords[np.newaxis, :] + 1).astype(np.int32)


class SPX_SLIC2D(SPXModel):
    """SLIC superpixel segmentation via scikit-image."""

    DOC_URL = "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic"
    PARAM_HINT = "n_segments=100, compactness=10, sigma=1"

    def __init__(self):
        DependencyCheck.require_package('skimage', display_name='scikit-image')
        from skimage.segmentation import slic
        self._slic = slic

    def forward(self, img: np.ndarray, **kwargs) -> np.ndarray:
        if img is None:
            raise ValueError("Missing required argument: img")
        # CT slices are 2-D grayscale; slic defaults to channel_axis=-1
        # (multi-channel) which raises on a plain 2-D array.
        if img.ndim == 2:
            kwargs.setdefault('channel_axis', None)
        return self._slic(img, **kwargs)


class SPX_Felzenszwalb2D(SPXModel):
    """Felzenszwalb graph-based superpixel segmentation via scikit-image."""

    DOC_URL = "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb"
    PARAM_HINT = "scale=100, sigma=0.5, min_size=50"

    def __init__(self):
        DependencyCheck.require_package('skimage', display_name='scikit-image')
        from skimage.segmentation import felzenszwalb
        self._felzenszwalb = felzenszwalb

    def forward(self, img: np.ndarray, **kwargs) -> np.ndarray:
        if img is None:
            raise ValueError("Missing required argument: img")
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            # Constant-intensity slice: normalise to a flat zero image so
            # felzenszwalb receives a valid [0, 1] input.  The resulting
            # label map will be a single region (all pixels equal), which is
            # the correct degenerate output for a uniform image.
            img = np.zeros_like(img)
        return self._felzenszwalb(img, **kwargs)
