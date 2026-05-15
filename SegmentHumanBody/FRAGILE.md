# FRAGILE.md — Known Code Fragility

Spots in the codebase that work correctly today but are sensitive to specific
assumptions. Each entry describes the fragility and its safe precondition so
a future refactor knows what contract to preserve or harden.

## Planned improvements

- **Expandable trajectory in `_caption` / `_summary`:** Collapsed spans show only
  the summary line. To let a reader cross-reference back to the compact process
  log (`*.json`), emit the event ID range as an indented line in `_span_text()`.
  `_span()` already stores `start_id` and `end_id` on every span (line 479 of
  `TimeLogSummarizer.py`), and events are consumed sequentially with no gaps, so
  every event with `id` between `start_id` and `end_id` inclusive belongs to the
  span. No new ID collection is needed — just emit `  events={start_id}–{end_id}`
  below the summary line.

---

## Checklist before touching any of these areas

- [ ] `_active_prompt_widget` is never `None` when `PointHandler.detach()` can be called
- [ ] Slicer window layout does not change mid-recording (no resize, dock, undock)
- [ ] Brush press inference is only a fallback — verify the normal press fires in the target Slicer version
- [ ] `tempfile.mkdtemp()` resolves to a local (non-network) drive on the target machine
- [ ] The audio subprocess still writes mono PCM16 if its recording parameters change
- [ ] Both `posNode` and `negNode` have `PointRemovedEvent` observers wired before any removal can occur
- [ ] All Qt widget comparisons use `==`, never `is`

---

---

## 1. `_input.py` — `PointHandler._on_detach` with `None` prompt widget

**Location:** `core/_input.py:187`

`_on_detach` calls `widget._set_prompt_widget_place_mode(widget._active_prompt_widget, False)`
unconditionally. If `_active_prompt_widget` is `None` at detach time (e.g. the
segment was removed between attach and detach), this passes `None` into
`_set_prompt_widget_place_mode`, which may silently no-op or raise inside Slicer
depending on the Qt binding state.

`_on_attach` raises `RuntimeError` when the widget is `None`, but `_on_detach`
has no guard. The detach path is harder to control because it is also called
during mutual-exclusion teardown.

**Safe precondition:** `_active_prompt_widget` is always set before `attach()`
and is not cleared until after `detach()` completes.

---

## 2. `_mouse_recorder.py` — Stale `xy_to_ijk` matrix cache

**Location:** `core/_mouse_recorder.py` — per-view `_xy_to_ijk` cache updated
in `_all_slice_visual_state`.

The DataProbe-style `xy_to_ijk` matrix is captured once at recording start and
on `view_changed` events. If the user resizes the Slicer window or docks/undocks
a panel between those events, the cached matrix drifts from the live transform.
Subsequent IJK coordinates derived from that cache will be slightly wrong without
any warning.

**Safe precondition:** The Slicer window layout does not change during a
recording session, or view-changed events are fired reliably on any resize that
affects the slice view geometry.

---

## 3. `_mouse_recorder.py` — Inferred brush press on missed initial event

**Location:** `core/_mouse_recorder.py` — `boundary_source` inference.

If Slicer drops the initial mouse-press event (can happen when the cursor
enters the view already pressed), the recorder infers a `press` boundary before
the first observed drag/release and sets `payload.boundary_source`. The inferred
press timestamp and IJK are copied from the first observed move, which may be
slightly later and spatially offset from the true press location.

**Safe precondition:** Slicer delivers the initial press reliably, which it does
under normal operation. The fallback is only triggered on fast entry or unusual
focus transitions.

---

## 4. `SegmentHumanBody.py` — `_AudioSubprocess` sentinel-file stop protocol

**Location:** `SegmentHumanBody.py:_AudioSubprocess.stop()`

The subprocess is told to stop by writing a sentinel file, which the subprocess
polls every 50 ms. On a slow or network-backed temp directory (e.g. a
redirected `%TEMP%` on Windows), filesystem notification latency can delay
the stop by hundreds of milliseconds, causing the saved WAV to include more
trailing audio than the intended 150 ms drain.

`stop()` waits up to 10 s then kills the process, so it never hangs, but the
WAV length may be unpredictable on slow storage.

**Safe precondition:** `tempfile.mkdtemp()` resolves to a local (non-network)
filesystem, which is the default on Windows and macOS.

---

## 5. `SegmentHumanBody.py` — `_finalize_wav` assumes PCM16 mono

**Location:** `SegmentHumanBody.py:_finalize_wav()`

The byte-arithmetic for prewarm trim and pause silencing uses
`frame_size = params.sampwidth * params.nchannels` from the WAV header.
This is correct for the 22050 Hz mono PCM16 WAV written by `_audio_subprocess.py`.
If a different subprocess or device produces a non-PCM or multi-channel WAV,
`frame_size` will be wrong and the trim/silence will land on incorrect byte
boundaries — corrupting the file without raising an error.

**Safe precondition:** The WAV written by `_audio_subprocess.py` is always
mono PCM16. Any future change to the subprocess recording format must also
update `_finalize_wav`.

---

## 6. `_point_log.py` — `sync_removed` only cleans one polarity per call

**Location:** `core/_point_log.py:sync_removed()`

`sync_removed(segment_id, is_neg, present_cp_ids)` only removes stale entries
whose `is_neg` matches the `is_neg` argument. A caller that needs to reconcile
both positive and negative entries must call it twice with `is_neg=False` then
`is_neg=True`.

The current widget only ever calls it once per polarity (once per
`PointRemovedEvent` on the positive node, once per the negative node), so both
polarities are eventually cleaned. But a caller that only calls it once for one
polarity will accumulate stale entries of the other polarity silently.

**Safe precondition:** Both markup nodes (`posNode` and `negNode`) have
`PointRemovedEvent` observers wired, and both fire reliably on every removal.

---

## 7. `_input.py` — PythonQt widget identity comparison

**Location:** `core/_input.py` and `SegmentHumanBody.py` wherever `widget.ui.*`
objects are compared.

PythonQt creates a new wrapper object each time a Qt widget is accessed through
the attribute chain. This means `widget.ui.someButton is widget.ui.someButton`
is `False`. Comparisons must use `==` (which compares the underlying C++ pointer)
rather than `is`. Using `is` will always be `False` and silently skip the
intended branch.

This has already caused one regression (the original `_PauseBlockFilter`
implementation failed to match the Resume button). Any future code that stores
a Qt widget reference and later compares it against a freshly-accessed attribute
must use `==`.

**Safe precondition:** All widget identity checks use `==`, never `is`.
