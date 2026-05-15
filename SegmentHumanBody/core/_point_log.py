"""Per-segment point log — pure Python, no Slicer/VTK imports.

Stores an ordered list of control-point entries keyed by segment ID.
Each entry is a plain dict:
    {
        'ras':    [float, float, float],  # RAS world coordinates
        'is_neg': bool,                   # True = negative prompt
        'cp_id':  str,                    # Slicer control-point GUID
    }

Usage
-----
Logic owns one PointLog instance (self.point_log).

Real-time path (VTK observers on PointAddedEvent / PointRemovedEvent):
  Widget._onPromptPointAdded  → point_log.append()
  Widget._onPromptPointRemoved → point_log.sync_removed()

Segment-switch path (captures position drags which fire no add/remove):
  Logic.snapshot_segment_points() → point_log.save()
  Logic.restore_segment_points()  → repopulates nodes with stable cp_ids
"""

import copy


class PointLog:
    """Ordered per-segment store of control-point entries."""

    def __init__(self):
        self._data: dict = {}   # segment_id (str) → list[dict]

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def save(self, segment_id: str, entries) -> None:
        """Replace the full entry list for *segment_id*."""
        self._data[segment_id] = list(entries)

    def append(self, segment_id: str, entry: dict) -> None:
        """Append one entry to *segment_id* — O(1), used by the PointAdded observer."""
        self._data.setdefault(segment_id, []).append(entry)

    def sync_removed(self, segment_id: str, is_neg: bool,
                     present_cp_ids: set) -> None:
        """Drop entries for *segment_id* that are no longer in the node.

        Called by the PointRemoved observer.  Only touches entries whose
        ``is_neg`` matches — the other polarity's entries are left intact.
        """
        entries = self._data.get(segment_id)
        if not entries:
            return
        self._data[segment_id] = [
            e for e in entries
            if e['is_neg'] != is_neg or e['cp_id'] in present_cp_ids
        ]

    def remove_segment(self, segment_id: str) -> None:
        """Drop all entries for *segment_id* (called when the segment is deleted)."""
        self._data.pop(segment_id, None)

    def clear(self) -> None:
        """Remove all entries for all segments."""
        self._data.clear()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get(self, segment_id: str) -> list:
        """Return a shallow copy of the entry list for *segment_id*."""
        return list(self._data.get(segment_id, []))

    def all_segments(self) -> list:
        """Return all segment IDs that have been tracked (including empty)."""
        return list(self._data.keys())

    def export(self) -> dict:
        """Deep copy of the full log (safe for JSON serialisation)."""
        return copy.deepcopy(self._data)
