# TOFIX.md — Restoration Roadmap (`feature/native-editor-wrapper`)

This branch deliberately strips model-family logic and replaces the custom
stroke tracker with Slicer's native Segment Editor.  The table below records
everything that was omitted, partially broken, or intentionally deferred so
that nothing is forgotten when functionality is restored.

**Guiding rule for every fix**: prefer re-using or delegating to a 3D Slicer
native tool over writing custom logic.  Only add custom code where Slicer has
no equivalent.

---

## Priority 1 — Broken / Silently Wrong

### 1.1  Recording replay fails silently
**Symptom**: `ReplayEngine._do_stroke` skips every stroke because
`_seg_id_map` is always empty.  
**Root cause**: `_recorder.record_segment_created(segment_id, ...)` is never
called in this branch.  On the main branch it is called from `onAddSegment`.  
**Fix**: Call `self._recorder.record_segment_created(segment_id, seg_name)`
inside `_onAddSegment` (after Slicer creates the segment) and
`self._recorder.record_segment_changed(old_id, new_id)` inside
`_onSegmentIDChanged`.  Both methods already exist in `_mouse_recorder.py`.

### 1.2  Point confirmation does nothing
**Symptom**: Placing a positive/negative prompt point has no effect on the
segment mask — points are purely decorative.  
**Root cause**: `_onPointConfirmed` and the VTK observer that calls it were
removed when the model-family system was stripped.  
**Fix options** (pick one or combine):  
- Wire a `PointModifiedEvent` observer → call a new `Logic.apply_point()`
  that uses Slicer's `SegmentEditorEffects` (Grow from seeds, Paint) rather
  than the custom SPX path.  
- Restore the SPX path from main branch as an optional family so the widget
  stays model-agnostic.  
**Note**: `PointLog` already records which segment each point belongs to and
is ready to feed a confirmation handler.

---

## Priority 2 — Hidden UI That Had Working Functionality

All items below are hidden via `_HIDDEN_WIDGETS` in `SegmentHumanBodyWidget`.
The UI elements exist in `SegmentHumanBody.ui`; only the Python wiring is
missing.

### 2.1  Export / Import Annotation Log
**Widgets**: `exportAnnotationLogButton`, `importAnnotationLogButton`  
**Status**: `PointLog.export()` exists.  No handler, no file dialog, no
snapshot-before-export call.  
**Fix**: Add `_onExportPointLog` (snapshot current segment first, then
`point_log.export()` → JSON file) and `_onImportPointLog` (load JSON →
`point_log.save()` per segment + call `restore_segment_points` for active
segment).  Wire to the existing buttons and un-hide them.

### 2.2  Slice-view selector
**Widget**: `sliceViewDropdown`  
**Status**: `Logic.active_slice_info(view_name, vol)` already resolves
axis/slice from any view name.  Dropdown was wired to `currentViewName` on
main branch.  
**Fix**: Un-hide, populate with `['Red', 'Green', 'Yellow']`, connect
`currentIndexChanged` → `self.currentViewName = ...`.

### 2.3  Assign Label (2D / 3D)
**Widgets**: `assignLabel2D`, `assignLabel3D`  
**Status**: Fully functional on main branch via `logic.commit_point()` /
`logic.expandSegWithSPX()`.  Removed with model-family system.  
**Fix**: Restore when model families are re-enabled (see Priority 3).

### 2.4  Create Box Prompt
**Widget**: `goToMarkupsButton`  
**Status**: Navigated to the Markups module for ROI placement.  Trivial to
restore.  
**Fix**: Un-hide, connect `clicked` →
`slicer.util.selectModule('Markups')`.

### 2.5  SAM mask selector
**Widget**: `samMaskDropdown`  
**Status**: Populated by `SAMFamily.confirm_model()` on main branch.  
**Fix**: Restore with SAM family (see Priority 3).

---

## Priority 3 — Model Family System (Deferred Until Model Weights Available)

The entire `FAMILY_REGISTRY` in `core/modelFamilies.py` is intact on the
main branch.  This branch hides the family UI but keeps the registry
importable.

| Widget (hidden) | Depends on |
|---|---|
| `modelFamilyDropdown` | `FAMILY_REGISTRY` |
| `modelVariantDropdown` | active family `MODEL_MAP` |
| `confirmModelSelection` | `family.confirm_model()` |
| `paramTextEdit` | `family.PARAM_HINT` |
| `docLinkLabel` | `family.DOC_URL` |
| `expandSelectedLabelButton` | `SPXModelFamily.on_expand()` |
| `showSPXBoundaryCheckBox` | `utils.spx_boundary_mask()` |
| `runAutomaticSegmentation` | `AutoModelFamily` |

**Restore path**:
1. Import `FAMILY_REGISTRY` and wire `modelFamilyDropdown`.
2. Restore `updateUIVisibility()` from main branch — it reads
   `family.VISIBLE_BUTTONS` and shows/hides widgets accordingly.
3. Restore `confirmModelSelection` handler → `family.confirm_model()`.
4. Restore `expandSelectedLabelButton` / E-hotkey → `logic.expandSegWithSPX()`.
5. Restore `showSPXBoundaryCheckBox` / Q-hotkey → `utils.spx_boundary_mask()`.

---

## Priority 4 — Minor Gaps and Polish

### 4.1  Point log: position drags not captured in real time
**Impact**: If the user drags a point and then calls `point_log.export()`
without switching segments, the exported `ras` is the original placement
position.  
**Fix**: Call `logic.snapshot_segment_points(seg_id, pos_node, neg_node)`
at the top of any export handler before `point_log.export()`.  No
architectural change needed.

### 4.2  `SetNthControlPointID` compatibility
**Location**: `Logic.restore_segment_points()`  
**Status**: Wrapped in `try/except AttributeError`.  If the API is absent,
cp_ids renumber on each restore and `sync_removed` may miss stale entries.  
**Fix**: Verify once in Slicer 5.10 console:
```python
n = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
idx = n.AddControlPoint([0,0,0])
n.SetNthControlPointID(idx, 'test-id')
print(n.GetNthControlPointID(idx))   # expect 'test-id'
slicer.mrmlScene.RemoveNode(n)
```
If it fails, replace with the log-update approach (store new cp_id from
`GetNthControlPointID(idx)` and call `point_log.save(segment_id,
new_entries)`).

### 4.3  Undo/redo scope
**Current**: Ctrl+Z / Ctrl+Shift+Z delegate to Slicer's Segment Editor undo
stack.  This is correct for brush/erase strokes.  
**Gap**: Point placements (markup node additions) are not in the Segment
Editor undo stack — Ctrl+Z after placing a point undoes the last brush
stroke instead.  
**Fix**: When point confirmation is restored (1.2), push a custom undo entry
or use Slicer's `vtkMRMLMarkupsNode` undo support.
