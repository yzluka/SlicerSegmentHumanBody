"""PromptSession — single source of truth for one annotation round.

A session spans from the first confirmed prompt point to the next
segment/model/brush change.  Its sole responsibility is to hold an
immutable snapshot of the segment state at session start so that
removing any combination of prompt points can always recompute the
correct result from scratch — without any snapshot-per-point bookkeeping.

Design principles
-----------------
* No Slicer imports — pure numpy so it is instantiable and testable
  entirely outside of 3D Slicer.
* Immutable base — ``base_mask`` is copied at construction and never
  mutated.  All render calls recompute from it, guaranteeing that the
  committed segment converges to the correct state regardless of the
  order in which points are added or removed.
* No point lists — prompt points are owned by the MRML markup nodes and
  read on demand by ``_logic.py``.  The session only holds what cannot
  be reconstructed from those nodes: the pre-session segment state.
"""

from __future__ import annotations
import numpy as np
from .utils import get_slice_from_volume


class PromptSession:
    """Immutable base-mask snapshot for one annotation session.

    Parameters
    ----------
    base_mask_3d : numpy.ndarray
        The segment's 3-D binary labelmap *before* any prompts in this
        session were placed.  A copy is taken immediately so that later
        Slicer writes cannot corrupt the baseline.
    """

    def __init__(self, base_mask_3d: np.ndarray) -> None:
        self.base_mask: np.ndarray = base_mask_3d.copy()

    def base_slice(self, axis: int, slice_idx: int) -> np.ndarray:
        """Read-only view of ``base_mask`` at the given slice position."""
        return get_slice_from_volume(self.base_mask, axis, slice_idx)
