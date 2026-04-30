# Upcoming Sprint — TODO

## ~~TODO 1: Dependency checking with lazy evaluation and live cached results~~ ✓ DONE

`core/_deps.py` — `DependencyCheck` class with process-scoped `_cache` dict.
`require_package` / `require_file` raise on failure; `check_package` / `check_file`
return `(ok, message)` without raising.  SPX model constructors (`SPX_SLIC2D`,
`SPX_Felzenszwalb2D`) call `DependencyCheck.require_package('skimage')` in
`__init__`.  Covered by `tests/test_deps.py`.

---

## ~~TODO 2: "A" key creates a new segment~~ ✓ DONE

`_tab_shortcut` registered in `setup()` with key `A`, calls `onAddSegment`.

---

## ~~TODO 3: Annotation log model family~~ ✓ DONE

`TimedAnnotatorFamily` in `core/modelFamilies.py` + `TimedAnnotatorModel` in
`core/models/timed_annotator.py`.  Registered as `'TimedMarker'` in `FAMILY_REGISTRY`.
Per-segment persistent `vtkMRMLMarkupsFiducialNode` mirror nodes with palette colors.
Nested JSON export/import with legacy flat-list compatibility.  Undo wired via
`on_point_undone`.  Export/Import buttons wired via `on_export` / `on_import`.

---

## TODO 4: Brush and erase support in TimedAnnotatorFamily

**Goal**
Allow the user to paint or erase segmentation masks while in the TimedMarker
family, and have those stroke events recorded in the annotation log (with
timestamps) alongside point events.

**Design**
- `TimedAnnotatorModel` should expose `on_stroke_committed(segment_id,
  source, timestamp)` (where `source` is `'brush'` or `'erase'`).
- The widget's `StrokeHandler._do_commit` (or a family hook called from it)
  appends a `{'type': 'stroke', ...}` entry to the log.
- Undo (`Ctrl+Z`) of a stroke must also remove the corresponding log entry.
- `TimedAnnotatorFamily.VISIBLE_BUTTONS` should expose the brush/erase
  toolbar buttons when this is implemented.

**Where to look**
- `core/models/timed_annotator.py` → add `on_stroke_committed`
- `core/_input.py` → `StrokeHandler._do_commit` (hook call site)
- `SegmentHumanBody.py` → `onUndo` brush/erase branch (log removal)

---

## ~~TODO 5: Per-segment point colors in TimedAnnotatorFamily~~ ✓ DONE

`_PALETTE` (8 colors) + `_color_for(segment_id)` in `TimedAnnotatorModel`; assigned once per segment in creation order, cycling; `_seg_colors` dict makes assignment deterministic across import/export.

---

## TODO 6: Import must create missing segments before replaying points

**Problem**
`on_import` / `load_from_json` adds mirror markup nodes and log entries for
each point in the JSON, but it never creates the corresponding segments in
the MRML segmentation node.  The result is orphaned annotation points that
are not linked to any actual segment the user can paint into.

**Design**
- Before replaying points, collect the unique `segment_id` values in the
  imported data.
- For each `segment_id` that does not already exist in the current
  segmentation node, create it (e.g. via
  `segmentation.GetSegmentation().AddEmptySegment(segment_id)`).
- Fire `on_segment_created` for each newly created segment so the log
  records the segment entry with a timestamp.
- The segment name shown in the UI should come from the JSON if a `seg_name`
  field is present (add it to the export format), otherwise fall back to the
  `segment_id` string.
- If no segmentation node is currently selected in the widget, surface a
  clear error via `slicer.util.errorDisplay` rather than silently dropping
  the import.

**Where to look**
- `core/models/timed_annotator.py` → `on_import`, `load_from_json`
  (need access to the widget/segmentation node — may require passing the
  widget reference through, or moving segment-creation logic to a family hook
  that the widget calls with scene access)
- `SegmentHumanBody.py` → `on_import` connection (widget has the
  segmentation node reference; consider passing it to `on_import`)
- Export format: add `seg_name` field to `export_data` output so round-trips
  preserve human-readable names

---

## ~~TODO 7: Restructure JSON export to a nested per-segment format~~ ✓ DONE

**Problem**
The current export is a flat list where every point carries its own
`segment_id` field:

```json
[
  {"segment_id": "seg-1", "coord_ras": [x, y, z], "timestamp": "..."},
  {"segment_id": "seg-1", "coord_ras": [x, y, z], "timestamp": "..."},
  {"segment_id": "seg-2", "coord_ras": [x, y, z], "timestamp": "..."}
]
```

This repeats `segment_id` on every row, makes it hard to query all points
for one segment, and does not have a natural place for per-segment metadata
(name, color, creation time).

**Design**
Switch to a nested dict keyed by segment:

```json
{
  "segments": {
    "seg-1": {
      "seg_name": "Segment_1",
      "points": [
        {"coord_ras": [x, y, z], "timestamp": "..."},
        ...
      ]
    },
    "seg-2": {
      "seg_name": "Segment_2",
      "points": [...]
    }
  }
}
```

- `export_data` in `TimedAnnotatorModel` builds this structure instead of a
  flat list.
- `load_from_json` accepts both the new nested format and the old flat list
  (detected by checking whether the top-level value is a `dict` or a `list`)
  so existing exported files remain importable.
- The top-level wrapper dict also leaves room for future metadata fields
  (e.g. `"version"`, `"volume_name"`, `"exported_at"`).

**Where to look**
- `core/models/timed_annotator.py` → `export_data`, `load_from_json`
- `tests/test_families.py` → update export-format assertions

---

## ~~TODO 8: Timestamps record last-modification time, not creation time~~ ✓ DONE

`PointEndInteractionEvent` + `PointRemovedEvent` observers added in `_mirror_to_node`. `_on_mirror_point_moved` updates `coord_ras` + `timestamp` in-place on drag; `_on_mirror_point_removed` drops orphaned log entries so export stays in sync with direct node edits.

---

## TODO 9: Restore SetHideFromEditors(True) on mirror nodes to recover per-click performance

**Problem**
`SetHideFromEditors(True)` was removed from `_mirror_to_node` to allow users
to drag/delete annotation points via the Markups module panel.  Without it,
every `AddControlPoint` call on a mirror node triggers Slicer's full markup
UI pipeline (panel refresh, active-node selection events, observer chains),
causing noticeable per-click slowdown.

**Key fact**
`SetHideFromEditors` only controls visibility in Slicer's **module panel
lists**.  Points are still fully visible and draggable/deletable directly in
the slice and 3D views even with the flag set — the user does not need the
panel to interact with them.

**Fix**
Restore `node.SetHideFromEditors(True)` in `_mirror_to_node`.  Verify that
drag and delete still work in the slice/3D views after the change.

**Where to look**
- `core/models/timed_annotator.py` → `_mirror_to_node`
