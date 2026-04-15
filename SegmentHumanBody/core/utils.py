"""Core utility functions shared across the SegmentHumanBody module.

Boundary detection and connected-component extraction use scikit-image where
available; a pure-numpy fallback is provided for environments that do not have
scikit-image installed (though those environments cannot run the SPX models
either, since they also require scikit-image).
"""

import ast
import numpy as np


# ---------------------------------------------------------------------------
# Slice view / coordinate constants
# ---------------------------------------------------------------------------

# Maps 3D Slicer slice view names to axis indices used by get/write_slice.
VIEW_TO_AXIS = {"Red": 0, "Green": 1, "Yellow": 2}

# axis → (x_col, y_col) indices into an IJK triple (I=0, J=1, K=2).
# Used when projecting 3-D IJK voxel coordinates onto a 2-D slice plane.
#   axis=0 Red/axial:       slice = vol[k,:,:]  → 2D pt = (I, J)
#   axis=1 Green/coronal:   slice = vol[:,j,:]  → 2D pt = (I, K)
#   axis=2 Yellow/sagittal: slice = vol[:,:,i]  → 2D pt = (J, K)
AXIS_TO_XY_COLS = {0: (0, 1), 1: (0, 2), 2: (1, 2)}

# axis → which IJK component gives the current slice index.
#   axis=0 (axial):    K = ijk[2]
#   axis=1 (coronal):  J = ijk[1]
#   axis=2 (sagittal): I = ijk[0]
AXIS_TO_IJK_COMPONENT = {0: 2, 1: 1, 2: 0}


# ---------------------------------------------------------------------------
# RAS → IJK 2-D projection  (pure numpy — no VTK / Slicer dependency)
# ---------------------------------------------------------------------------

def ras_to_ijk_2d(mat_4x4: np.ndarray, points, axis: int) -> list:
    """Convert RAS-space 3-D points to 2-D slice coordinates in IJK space.

    Parameters
    ----------
    mat_4x4 : (4, 4) numpy array — the volume's RAS-to-IJK affine matrix.
    points : sequence of (x, y, z) triples in RAS space.
    axis : int — slice axis (0 axial, 1 coronal, 2 sagittal).

    Returns
    -------
    List of [x, y] pairs in 2-D slice space (column-first / OpenCV convention).
    Empty list when *points* is empty.
    """
    if not points:
        return []
    xc, yc = AXIS_TO_XY_COLS[axis]
    pts   = np.array(points, dtype=np.float64)          # (N, 3)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])    # (N, 4)
    ijk   = (mat_4x4 @ pts_h.T).T[:, :3].astype(int)   # (N, 3)
    return ijk[:, [xc, yc]].tolist()


# ---------------------------------------------------------------------------
# Segment name helper
# ---------------------------------------------------------------------------

def next_segment_name(existing_names) -> str:
    """Return the lowest 'Segment_N' name not already in *existing_names*."""
    index = 1
    while f"Segment_{index}" in existing_names:
        index += 1
    return f"Segment_{index}"


# ---------------------------------------------------------------------------
# Parameter string parsing
# ---------------------------------------------------------------------------

def parse_user_parameters(text: str) -> dict:
    """Parse a model hyper-parameter string into a dict.

    Accepts two formats:
      - Dict literal:  {"n_segments": 100, "sigma": 5.0}
      - key=value:     n_segments=100, sigma=5.0    (matches PARAM_HINT style)

    Returns an empty dict for empty / whitespace-only input.
    Raises ValueError with a descriptive message on parse failure.
    """
    text = text.strip()
    if not text:
        return {}
    # Try dict-literal first (JSON-compatible)
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    # Fall back to key=value pairs
    try:
        items = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            key, value = part.split("=", 1)
            items.append(f"'{key.strip()}': {value.strip()}")
        return ast.literal_eval("{" + ", ".join(items) + "}")
    except Exception as e:
        raise ValueError(f"Invalid parameter format:\n{str(e)}")


# ---------------------------------------------------------------------------
# Volume slice helpers
# ---------------------------------------------------------------------------

def get_slice_from_volume(volume_array: np.ndarray, axis: int, slice_index: int) -> np.ndarray:
    """Return a 2-D view of *volume_array* at *slice_index* along *axis*.

    The returned array is a **numpy view** — mutations propagate back to the
    source array without any data copy.
    """
    idx = [slice(None), slice(None), slice(None)]
    idx[axis] = slice_index
    return volume_array[tuple(idx)]


def write_slice_to_volume(target_array: np.ndarray, slice_data: np.ndarray,
                          axis: int, slice_index: int) -> None:
    """Write *slice_data* into *target_array* at *slice_index* along *axis* in-place."""
    idx = [slice(None), slice(None), slice(None)]
    idx[axis] = slice_index
    target_array[tuple(idx)] = slice_data


# ---------------------------------------------------------------------------
# Generic helper
# ---------------------------------------------------------------------------

def call_if_exists(obj, method_name: str, *args, **kwargs):
    """Call ``obj.method_name(*args, **kwargs)`` if the method exists; return None otherwise."""
    if obj and hasattr(obj, method_name):
        return getattr(obj, method_name)(*args, **kwargs)


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def apply_window_level(img: np.ndarray, window, level) -> np.ndarray:
    """Clip *img* to [level − window/2, level + window/2] and scale to [0, 255].

    Returns a ``uint8`` array.  If *window* or *level* is ``None`` the original
    array is returned unchanged (no normalization).

    This function never modifies the source volume — it always operates on a
    float32 copy of the slice.
    """
    if window is None or level is None:
        return img
    lo = float(level) - float(window) / 2.0
    hi = float(level) + float(window) / 2.0
    clipped = np.clip(img.astype(np.float32), lo, hi)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Markups point-collection helpers (Slicer-agnostic)
# ---------------------------------------------------------------------------

# Status constants mirror vtkMRMLMarkupsNode — duplicated here so the helpers
# can be tested without importing Slicer.
POSITION_UNDEFINED = 0
POSITION_PREVIEW   = 1
POSITION_DEFINED   = 2


def collect_confirmed_points(point_records, preview_ids):
    """Return positions of control points that represent confirmed placements.

    Parameters
    ----------
    point_records : iterable of (status: int, cp_id: str, position)
        Each tuple describes one control point.  ``status`` uses the
        POSITION_* constants above.  ``position`` is whatever the caller
        stored (e.g. a 3-tuple of RAS coordinates).
    preview_ids : set
        IDs of control points that are currently tracked as unconfirmed
        placement cursors (populated by _onPointAdded in the widget).

    A point is *excluded* when:
    * status == POSITION_UNDEFINED  (no position at all)
    * status == POSITION_PREVIEW **and** the ID is in preview_ids
      (this is the active placement cursor before the user clicks)

    A point is *included* when:
    * status == POSITION_DEFINED  (normal confirmed state), or
    * status == POSITION_PREVIEW **and** the ID is NOT in preview_ids
      (some Slicer builds leave confirmed points at PositionPreview;
      a point whose ID was removed from preview_ids by _onPointConfirmed
      is treated as confirmed even if its status stayed at Preview)
    """
    result = []
    for status, cp_id, position in point_records:
        if status == POSITION_UNDEFINED:
            continue
        if status == POSITION_PREVIEW and cp_id in preview_ids:
            continue
        result.append(position)
    return result


def collect_preview_points(point_records, preview_ids):
    """Return positions of unconfirmed placement cursors.

    These are PositionPreview points whose ID is still in *preview_ids*,
    i.e. the user clicked '+' but has not yet clicked on a slice.
    Used by the hover-preview render path.
    """
    return [
        position for status, cp_id, position in point_records
        if status == POSITION_PREVIEW and cp_id in preview_ids
    ]


def labels_at_points(points, labels: np.ndarray) -> set:
    """Return the set of unique label values under the given (x, y) points.

    Uses numpy fancy indexing — O(n_points) cost, no Python loop.
    Out-of-bounds points are silently ignored.

    Parameters
    ----------
    points : sequence of (x, y) pairs  (column-first / OpenCV convention)
    labels : 2-D integer array of shape (H, W)
    """
    if not points:
        return set()
    pts = np.asarray(points, dtype=np.intp)   # (N, 2)
    xs, ys = pts[:, 0], pts[:, 1]
    H, W = labels.shape[:2]
    valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    return set(labels[ys[valid], xs[valid]].tolist())


def spx_boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Return a ``uint8`` mask that is 1 at superpixel boundaries, 0 elsewhere.

    Uses ``skimage.segmentation.find_boundaries`` with 4-connectivity when
    scikit-image is available; falls back to a pure-numpy equivalent otherwise.
    """
    try:
        from skimage.segmentation import find_boundaries
        return find_boundaries(labels, connectivity=1, mode='thick').astype(np.uint8)
    except ImportError:
        b = np.zeros(labels.shape, dtype=np.uint8)
        b[1:]    |= (labels[1:]    != labels[:-1]   ).view(np.uint8)
        b[:-1]   |= (labels[:-1]   != labels[1:]    ).view(np.uint8)
        b[:, 1:] |= (labels[:, 1:] != labels[:, :-1]).view(np.uint8)
        b[:, :-1]|= (labels[:, :-1]!= labels[:, 1:] ).view(np.uint8)
        return b


def extract_connected_component(mask: np.ndarray, point_xy) -> np.ndarray:
    """Return a boolean mask containing the 4-connected region of *mask* that
    includes the pixel at *point_xy* = (x, y).

    Uses ``skimage.morphology.flood`` when available; falls back to
    ``scipy.ndimage.binary_propagation`` otherwise.
    """
    x, y = point_xy
    if not mask[y, x]:
        return np.zeros_like(mask, dtype=bool)
    try:
        from skimage.morphology import flood
        return flood(mask, (y, x), connectivity=1)
    except ImportError:
        from scipy.ndimage import binary_propagation
        seed = np.zeros_like(mask, dtype=bool)
        seed[y, x] = True
        return binary_propagation(seed, mask=mask)
