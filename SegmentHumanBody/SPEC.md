# SPEC.md — Behavior Specification

Authoritative checklist of expected behaviors.  Verify **every item** after
any change to `SegmentHumanBody.py`, `core/_point_log.py`, or related UI.

---

## 0  Input Handler Abstraction

All interactive input modes — Brush, Erase, and Point placement — are
managed through the `InputHandler` hierarchy in `core/_input.py`.

| Rule | Detail |
|---|---|
| Mutual exclusion | Activating any handler automatically detaches the previous one via `_detach_current()` |
| Unified segment guard | `InputHandler._ensure_segment()` is called inside `_detach_current()` for every handler; no subclass has its own null-segment check |
| Auto-create on attach | If no segmentation/segment exists when a handler attaches, it is created before any Slicer element is installed |
| Auto-switch on create | After any segment creation (via `+`, `A`, or handler guard), the selector always switches to the newly created segment |
| Per-segment markup nodes | Each segment owns one pos + one neg `vtkMRMLMarkupsFiducialNode`; their IDs stored in segment tags; switching segments calls `setCurrentNode()` on the widgets |
| Point isolation | Points placed are stored directly in the active segment's markup node; switching segments swaps which nodes the widgets display |

### Handler classes

| Class | Trigger | What it does |
|---|---|---|
| `BrushHandler` | Brush button toggled on | Activates Segment Editor Paint effect; deactivates Erase if active |
| `EraseHandler` | Erase button toggled on | Activates Segment Editor Erase effect; deactivates Brush if active |
| `PointHandler` | Point placement mode entered | Runs segment guard only; placement itself is delegated to `qSlicerSimpleMarkupsWidget` |

---

## 1  Volume / Segmentation Selection

| # | Action | Expected result |
|---|---|---|
| 1.1 | Select a volume with no segmentation present | Segmentation node auto-created and linked to the volume; one initial segment auto-created; `segmentSelector` shows that segment |
| 1.2 | Select a volume when a segmentation already exists | Existing segmentation kept; no extra segment created |
| 1.3 | Clear the volume selector | Parameter node updated; no crash |
| 1.4 | Switch to a different volume | W/L sliders sync to the new volume's display node |

---

## 2  Segment Management

| # | Action | Expected result |
|---|---|---|
| 2.1 | Press `+` button with a segmentation present | New segment added; selector **switches** to the new segment automatically |
| 2.2 | Press `+` after volume selected but before segmentation created | Segmentation auto-created, segment added, selector switches to it, button enabled |
| 2.3 | Press `+` with no volume and no segmentation | Warning dialog; no crash |
| 2.4 | Press `A` hotkey | Same as 2.1 |
| 2.5 | Activate Brush or Erase with no segment | Handler creates segmentation+segment, selector switches, then effect activates |
| 2.6 | Press `-` button with a segment selected | Segment removed; markup nodes cleared if it was the current segment |
| 2.7 | Press `-` button with no segment selected | Warning dialog; no crash |
| 2.8 | Remove the last segment | Segment selector shows nothing; markup nodes cleared |
| 2.9 | Any new segment always triggers auto-switch | True whether created via `+`, `A`, or handler guard |

---

## 3  Brush / Erase Tools

| # | Action | Expected result |
|---|---|---|
| 3.1 | Click **Brush** with no volume | Warning dialog; button stays unchecked |
| 3.2 | Click **Brush** with volume but no segmentation | Segmentation and segment both auto-created; Paint effect activates |
| 3.2b | Click **Brush** with segmentation but no segments | Segment auto-created; Paint effect activates |
| 3.3 | Click **Brush** normally | Paint effect activates in Segment Editor; brush options panel appears |
| 3.4 | Click **Erase** while Brush is active | Brush deactivates; Erase effect activates |
| 3.5 | Click **Brush** while Erase is active | Erase deactivates; Paint effect activates |
| 3.6 | Deactivate Brush (uncheck) | Effect deactivated; brush options panel hidden |
| 3.7 | Ctrl+Z while brush is active | Segment Editor undo triggered (brush stroke reversed) |

---

## 4  Segment Switching

| # | Action | Expected result |
|---|---|---|
| 4.1 | Switch from segment A to segment B | A's points snapshotted; B's points loaded into markup nodes; markup nodes show B's points only |
| 4.2 | Switch back to segment A | A's points restored (including any positions changed since last visit) |
| 4.3 | Switch segment | Incoming segment always made visible; `showCurrentSegmentCheckBox` snaps to checked |
| 4.4 | Switch segment with saved-segments hidden | Only the incoming segment becomes visible; all others remain hidden |
| 4.5 | `_onParameterNodeModified` fires | Prompt widgets wired to current segment's nodes via `get_segment_prompt_nodes`; no snapshot/restore needed |

---

## 5  Prompt Points (per-segment native markup nodes)

Each segment owns exactly one pair of `vtkMRMLMarkupsFiducialNode` nodes (positive
and negative).  Their scene IDs are stored in the segment's tags under
`SegmentHumanBody.posNodeID` and `SegmentHumanBody.negNodeID`.  Switching segments
calls `setCurrentNode()` on the two `qSlicerSimpleMarkupsWidget`s — no snapshot/
restore, no Python point log.

| # | Action | Expected result |
|---|---|---|
| 5.1 | Place a positive prompt point | Point stored directly in the segment's positive markup node; `positivePrompts` widget stays in placement mode |
| 5.2 | Place a negative prompt point | Point stored in the segment's negative markup node |
| 5.3 | Delete a prompt point | Markup node control point removed natively; no extra bookkeeping |
| 5.4 | Switch segments, return | `positivePrompts`/`negativePrompts` widgets switch to the returning segment's nodes via `setCurrentNode()`; all previously placed points visible immediately |
| 5.5 | Remove a segment | `Logic.delete_segment_prompt_nodes` removes both markup nodes from the scene; no stale nodes remain |
| 5.6 | Create a new segment | `Logic.add_segment` calls `create_segment_prompt_nodes`; tags written before the selector switches |

---

## 6  Window / Level

| # | Action | Expected result |
|---|---|---|
| 6.1 | Select a volume | W/L sliders pre-filled from volume's display node |
| 6.2 | Move Window slider | SpinBox syncs; W/L applied to volume display node immediately |
| 6.3 | Edit Window spin box | Slider syncs; W/L applied immediately |
| 6.4 | Move Level slider / spin box | Same as 6.2 / 6.3 for level |
| 6.5 | Click **Apply Window / Level** | Same W/L explicitly written to display node |

---

## 7  Segment Visibility

| # | Action | Expected result |
|---|---|---|
| 7.1 | Check **Show Saved Segments** | All segments except current become visible |
| 7.2 | Uncheck **Show Saved Segments** | All segments except current become hidden |
| 7.3 | Press `V` hotkey | Current segment visibility toggles; `showCurrentSegmentCheckBox` reflects state |
| 7.4 | Uncheck **Show Current Seg** | Current segment hidden in slice views |
| 7.5 | Switch segment | Incoming current segment always made visible; checkbox snaps to checked |

---

## 8  Recording

| # | Action | Expected result |
|---|---|---|
| 8.1 | Click **Start Record** | Recording starts; **Stop Record** button appears; status label updates |
| 8.2 | Click **Stop Record** | Recording stops; **Export** button enabled |
| 8.3 | Click **Export** | File dialog; JSON saved; confirmation shown |
| 8.4 | Click **Load** | File dialog; recording loaded; status label shows event count |
| 8.5 | Load recording matching current volume | **Replay** button enabled |
| 8.6 | Load recording for different volume | **Replay** button disabled with reason |
| 8.7 | Click **Replay** | Replay engine runs; "Replay complete." dialog shown when done |

---

## 9  Undo / Redo

| # | Action | Expected result |
|---|---|---|
| 9.1 | Ctrl+Z with brush stroke | Segment Editor undo: brush stroke reversed |
| 9.2 | Ctrl+Shift+Z after undo | Segment Editor redo: stroke re-applied |
| 9.3 | Ctrl+Z with no history | No crash; no visible effect |

---

## 10  Lifecycle

| # | Action | Expected result |
|---|---|---|
| 10.1 | Switch away from module (`exit()`) | Active Segment Editor effect deactivated; EffectsOptionsFrame returned |
| 10.2 | Return to module (`enter()`) | Parameter node re-initialized; prompt nodes rewired |
| 10.3 | Scene closed / Slicer quit | No dangling observers; no Python crash |
