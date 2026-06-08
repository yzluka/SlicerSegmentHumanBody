"""Input handlers for SegmentHumanBodyWidget (native-editor-wrapper branch).

InputHandler    — base: volume check, mutual-exclusion, segment-existence guard,
                  and active-handler registration.  Subclasses override _on_attach.
StrokeHandler   — shared Brush/Erase logic (effect activation, UI sync, stroke hooks).
                  Installs VTK LeftButtonPress/Release observers on all slice views.
                  Subclasses override _on_stroke_press / _post_process_stroke (both
                  are no-ops by default) to add per-stroke behaviour.
BrushHandler    — Paint effect base (no brush geometry setting).
Brush2DHandler  — Paint with a 2-D disc brush (BrushSphere=0); current default UI brush.
Brush3DHandler  — Paint with a 3-D sphere brush (BrushSphere=1); skeleton for future 3D models.
EraseHandler    — Erase effect base (no brush geometry setting).
Erase2DHandler  — Erase with a 2-D disc brush; current default UI erase.
Erase3DHandler  — Erase with a 3-D sphere brush; skeleton for future 3D models.
SpxBrushHandler — Brush2DHandler + SPX snapping: after each stroke a labelmap diff identifies
                  which superpixels were touched, then fills them completely via apply_spx_label.
SpxEraseHandler — Erase2DHandler + SPX snapping: same diff approach, erases touched superpixels.
PointHandler    — prompt-point mode (no extra setup beyond the base guard).

Design rules
------------
* attach() in the base class enforces: volume check → detach previous →
  ensure segment → register self → call _on_attach().  No subclass repeats
  any of these steps.
* The segment-existence guard lives exclusively in _ensure_segment() —
  no subclass needs its own null-segment check.
* Handlers never import qt or reference widget.ui directly for business
  logic; they only call widget.logic.* and widget.ui widget names for
  button-state sync.
"""

import logging
import slicer

log = logging.getLogger(__name__)

# Set True to print per-phase timing for every SPX stroke to the Python console.
SPX_DEBUG_TIMING = True


class _T:
    """Minimal inline phase timer.  Usage:
        t = _T()
        ... phase A ...
        t.mark('A')
        ... phase B ...
        t.mark('B')
        print(t)
    """
    def __init__(self):
        import time
        self._t0 = self._last = time.perf_counter()
        self._phases: list = []

    def mark(self, name: str) -> float:
        import time
        now = time.perf_counter()
        ms = (now - self._last) * 1000
        self._phases.append((name, ms))
        self._last = now
        return ms

    def total_ms(self) -> float:
        import time
        return (time.perf_counter() - self._t0) * 1000

    def __str__(self):
        parts = '  '.join(f'{n}={v:.1f}ms' for n, v in self._phases)
        return f'{parts}  | total={self.total_ms():.1f}ms'


class InputHandler:
    """Base class for interactive input modes."""

    TOOL_NAME: str | None = None  # set by each subclass

    # ------------------------------------------------------------------ #
    # Guard helpers                                                        #
    # ------------------------------------------------------------------ #

    def _detach_current(self, widget) -> None:
        """Flush and detach the active handler."""
        current = widget._active_handler
        if current is not None and current is not self:
            current.detach(widget)

    def _ensure_segment(self, widget) -> str:
        """Create segmentation/segment if missing; switch to the new segment.

        Delegates to widget._onAddSegment so that creation, UI wiring, and
        the auto-switch all live in one place.  Returns the active segment ID
        after the guard completes (empty string when no volume is selected).
        """
        pn = getattr(widget, '_parameterNode', None)
        vol, seg = widget.logic.getVolumeAndSegmentation(pn)
        if not vol:
            vol = widget.ui.sourceVolumeSelector.currentNode()
        if not vol:
            return ''
        if not seg:
            seg = widget.ui.segmentationNodeSelector.currentNode()
        if not seg or seg.GetSegmentation().GetNumberOfSegments() == 0:
            widget._onAddSegment()
        else:
            widget._ensure_current_prompt_nodes()
        return widget.ui.segmentSelector.currentSegmentID()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def attach(self, widget) -> None:
        """Volume check → detach previous → ensure segment → register → _on_attach."""
        if not widget.ui.sourceVolumeSelector.currentNode():
            slicer.util.warningDisplay('Please select a volume first.')
            self._on_attach_cancelled(widget)
            return
        widget._attaching_handler = self
        self._detach_current(widget)
        self._ensure_segment(widget)
        widget._active_handler = self
        self._on_attach(widget)
        widget._attaching_handler = None
        widget._recorder.record_tool_selected(
            self.TOOL_NAME, widget.ui.segmentSelector.currentSegmentID())

    def _on_attach(self, widget) -> None:
        """Override in subclasses to install tool-specific resources."""

    def _on_attach_cancelled(self, widget) -> None:
        """Override in subclasses to reset UI when attach is aborted."""

    def detach(self, widget) -> None:
        """_on_detach → unregister.  Subclasses override _on_detach, not detach."""
        self._on_detach(widget)
        if widget._active_handler is self:
            widget._active_handler = None
        # Only record "no tool" when deactivating without a replacement —
        # _attaching_handler is set during a tool switch to suppress this.
        if widget._attaching_handler is None:
            widget._recorder.record_tool_selected(
                None, widget.ui.segmentSelector.currentSegmentID())

    def _on_detach(self, widget) -> None:
        """Override in subclasses for tool-specific teardown."""


class StrokeHandler(InputHandler):
    """Shared logic for Paint and Erase handlers.

    Subclasses set EFFECT ('Paint' or 'Erase') and BUTTON_NAME (the name
    of the checkable QPushButton in widget.ui that controls this handler).

    VTK LeftButtonPress/Release observers are installed on all three slice
    views during _on_attach and removed in _on_detach.  On each press the
    current slice axis and index are captured and passed to _on_stroke_press;
    on release they are retrieved and passed to _post_process_stroke (deferred
    via QTimer.singleShot so Paint has committed its stroke first).  Both hooks
    are no-ops by default.
    """

    EFFECT: str = ''
    BUTTON_NAME: str = ''

    def __init__(self):
        super().__init__()
        self._stroke_obs: dict     = {}   # vn → (iv, press_tag, release_tag)
        self._stroke_context: dict = {}   # vn → (axis, slice_idx)
        self._widget_ref           = None

    def _on_attach(self, widget) -> None:
        vol    = widget.ui.sourceVolumeSelector.currentNode()
        seg    = widget.ui.segmentationNodeSelector.currentNode()
        seg_id = widget.ui.segmentSelector.currentSegmentID()
        editor = widget.logic.get_segment_editor()
        widget.logic.setup_editor_nodes(editor, vol, seg, seg_id)
        if editor:
            editor.setActiveEffectByName(self.EFFECT)
            # Re-assert segment ID: setActiveEffectByName can trigger deferred
            # MRML events that reset currentSegmentID to empty.
            if seg_id:
                editor.setCurrentSegmentID(seg_id)
            widget._borrowEffectsOptionsFrame()
            slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()
        self._sync_buttons(widget, active=True)
        self._widget_ref = widget
        self._install_stroke_observers(widget)

    def _on_attach_cancelled(self, widget) -> None:
        self._sync_buttons(widget, active=False)

    def _on_detach(self, widget) -> None:
        self._remove_stroke_observers()
        widget._returnEffectsOptionsFrame()
        editor = widget.logic.get_segment_editor()
        if editor:
            cur = editor.activeEffect()
            if cur and cur.name == self.EFFECT:
                editor.setActiveEffectByName('')
        self._sync_buttons(widget, active=False)

    def _sync_buttons(self, widget, active: bool) -> None:
        """Set the brush/erase toggle buttons to reflect current handler state."""
        for name in ('brushToolButton', 'eraseToolButton'):
            btn = getattr(widget.ui, name, None)
            if btn:
                btn.blockSignals(True)
                btn.setChecked(active and name == self.BUTTON_NAME)
                btn.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Stroke observer infrastructure                                       #
    # ------------------------------------------------------------------ #

    def _install_stroke_observers(self, widget):
        lm = slicer.app.layoutManager()
        if lm is None:
            return
        for vn in ('Red', 'Green', 'Yellow'):
            sw = lm.sliceWidget(vn)
            sv = sw.sliceView() if sw else None
            iv = sv.interactor() if sv else None
            if iv is None:
                continue
            p = iv.AddObserver('LeftButtonPressEvent',
                               lambda e, n, v=vn: self._handle_stroke_press(v),
                               1000.0)
            r = iv.AddObserver('LeftButtonReleaseEvent',
                               lambda e, n, v=vn: self._handle_stroke_release(v),
                               1000.0)
            self._stroke_obs[vn] = (iv, p, r)

    def _remove_stroke_observers(self):
        for vn, (iv, p, r) in self._stroke_obs.items():
            iv.RemoveObserver(p)
            iv.RemoveObserver(r)
        self._stroke_obs = {}

    def _handle_stroke_press(self, view_name):
        from .utils import VIEW_TO_AXIS
        editor = self._widget_ref.logic.get_segment_editor()
        if editor is None:
            return
        vol = editor.sourceVolumeNode()
        if vol is None:
            return
        axis      = VIEW_TO_AXIS.get(view_name, 2)
        slice_idx = self._get_stroke_slice_idx(view_name, vol, axis)
        if slice_idx is None:
            return
        self._stroke_context[view_name] = (axis, slice_idx)
        self._on_stroke_press(view_name, axis, slice_idx)

    def _handle_stroke_release(self, view_name):
        ctx = self._stroke_context.pop(view_name, None)
        if ctx is None:
            return
        axis, slice_idx = ctx
        # Defer to the next Qt event-loop tick so Slicer's Paint/Erase effect
        # has fully committed the stroke to the labelmap before we read it.
        import qt
        qt.QTimer.singleShot(
            0, lambda: self._post_process_stroke(view_name, axis, slice_idx))

    def _get_stroke_slice_idx(self, view_name, vol, axis):
        import vtk
        lm = slicer.app.layoutManager()
        sw = lm.sliceWidget(view_name) if lm else None
        if sw is None:
            return None
        sliceToRAS = sw.mrmlSliceNode().GetSliceToRAS()
        rasToIJK   = vtk.vtkMatrix4x4()
        vol.GetRASToIJKMatrix(rasToIJK)
        ras = [sliceToRAS.GetElement(r, 3) for r in range(3)] + [1.0]
        ijk = [0.0, 0.0, 0.0, 0.0]
        rasToIJK.MultiplyPoint(ras, ijk)
        # numpy axis 0 → K → ijk[2]; axis 1 → J → ijk[1]; axis 2 → I → ijk[0]
        return int(round(ijk[{0: 2, 1: 1, 2: 0}[axis]]))

    # ------------------------------------------------------------------ #
    # Stroke hooks (no-op by default; override in subclasses)             #
    # ------------------------------------------------------------------ #

    def _on_stroke_press(self, view_name: str, axis: int, slice_idx: int) -> None:
        """Override to capture pre-stroke state. No-op by default."""

    def _post_process_stroke(self, view_name: str, axis: int, slice_idx: int) -> None:
        """Override to apply post-stroke processing. No-op by default."""


class BrushHandler(StrokeHandler):
    """Activates the Segment Editor Paint effect (no brush geometry setting)."""
    EFFECT      = 'Paint'
    BUTTON_NAME = 'brushToolButton'
    TOOL_NAME   = 'brush'


class Brush2DHandler(BrushHandler):
    """Paint effect with a flat 2-D disc brush (BrushSphere=0)."""

    def _on_attach(self, widget) -> None:
        super()._on_attach(widget)
        editor = widget.logic.get_segment_editor()
        if editor is None:
            return
        effect = editor.activeEffect()
        if effect:
            effect.setParameter('BrushSphere', '0')


class Brush3DHandler(BrushHandler):
    """Paint effect with a 3-D sphere brush (BrushSphere=1). Skeleton for future 3D models."""

    def _on_attach(self, widget) -> None:
        super()._on_attach(widget)
        editor = widget.logic.get_segment_editor()
        if editor is None:
            return
        effect = editor.activeEffect()
        if effect:
            effect.setParameter('BrushSphere', '1')


class EraseHandler(StrokeHandler):
    """Activates the Segment Editor Erase effect (no brush geometry setting)."""
    EFFECT      = 'Erase'
    BUTTON_NAME = 'eraseToolButton'
    TOOL_NAME   = 'erase'


class Erase2DHandler(EraseHandler):
    """Erase effect with a flat 2-D disc brush (BrushSphere=0)."""

    def _on_attach(self, widget) -> None:
        super()._on_attach(widget)
        editor = widget.logic.get_segment_editor()
        if editor is None:
            return
        effect = editor.activeEffect()
        if effect:
            effect.setParameter('BrushSphere', '0')


class Erase3DHandler(EraseHandler):
    """Erase effect with a 3-D sphere brush (BrushSphere=1). Skeleton for future 3D models."""

    def _on_attach(self, widget) -> None:
        super()._on_attach(widget)
        editor = widget.logic.get_segment_editor()
        if editor is None:
            return
        effect = editor.activeEffect()
        if effect:
            effect.setParameter('BrushSphere', '1')


class PointHandler(InputHandler):
    """Segment guard for prompt-point placement mode.

    Actual placement is managed by qSlicerSimpleMarkupsWidget.  This handler
    exists so that entering point-placement mode runs the same unified
    segment-existence guard as brush and erase, ensuring every placed point
    is immediately associated with a valid segment.

    widget._active_prompt_widget must be set to the source markup widget
    BEFORE attach() is called.  Attaching without it is a programming error
    and raises RuntimeError immediately so the broken call site is visible.
    """

    TOOL_NAME = 'point'

    def _on_attach(self, widget) -> None:
        if widget._active_prompt_widget is None:
            raise RuntimeError(
                'PointHandler._on_attach: widget._active_prompt_widget is None. '
                'Set _active_prompt_widget to the triggering markup widget '
                'before calling PointHandler().attach().'
            )
        widget._set_prompt_widget_place_mode(widget._active_prompt_widget, True)

    def _on_detach(self, widget) -> None:
        widget._set_prompt_widget_place_mode(widget._active_prompt_widget, False)
        widget._deactivate_prompt_place_mode()


# ======================================================================== #
# SPX stroke handlers — shared helper functions                            #
# ======================================================================== #

def _spx_get_vol_mats(handler, vol):
    """Return (ijk_to_ras_vtk, ras_to_ijk_vtk) cached per volume ID in handler._mat_cache."""
    import vtk
    vol_id = vol.GetID()
    if vol_id not in handler._mat_cache:
        ijk_to_ras = vtk.vtkMatrix4x4()
        vol.GetIJKToRASMatrix(ijk_to_ras)
        ras_to_ijk = vtk.vtkMatrix4x4()
        vol.GetRASToIJKMatrix(ras_to_ijk)
        handler._mat_cache[vol_id] = (ijk_to_ras, ras_to_ijk)
    return handler._mat_cache[vol_id]


def _spx_read_native_slice(seg_node, seg_id, ref_vol, axis, slice_idx,
                           ras_to_ijk=None):
    """Read a 2-D bool slice from the segment's native vtkOrientedImageData.

    Uses vtk_to_numpy (a zero-copy view into VTK memory) and immediately reads the
    relevant 2-D slice — no MRML node creation and no full-volume resampling.

    Slicer stores the compact binary labelmap with its position encoded in the
    labelmap's IJK-to-RAS world matrix, not in the VTK extent values.  The extent
    origin (li0, lj0, lk0) may be 0 (compact/world-encoded) or a non-zero offset
    depending on the Slicer version and how the labelmap was created.  We resolve
    the reference-volume IJK position by mapping the labelmap's (0,0,0) voxel through
    its world matrix, then back through the reference volume's RAS-to-IJK matrix.

    Returns all-zeros if the segment has no binary labelmap yet (first stroke).
    """
    import numpy as np
    from vtk.util.numpy_support import vtk_to_numpy
    import vtk

    rdims = ref_vol.GetImageData().GetDimensions()  # (ni, nj, nk)
    out_shape = {0: (rdims[1], rdims[0]),   # K-slice: (nj, ni)
                 1: (rdims[2], rdims[0]),   # J-slice: (nk, ni)
                 2: (rdims[2], rdims[1])}[axis]  # I-slice: (nk, nj)
    out = np.zeros(out_shape, dtype=bool)

    import vtkSegmentationCorePython as vtkSegmentationCore
    seg_obj  = seg_node.GetSegmentation().GetSegment(seg_id)
    if seg_obj is None:
        return out
    rep_name = vtkSegmentationCore.vtkSegmentationConverter \
                   .GetBinaryLabelmapRepresentationName()
    labelmap = seg_obj.GetRepresentation(rep_name)
    if labelmap is None:
        return out
    scalars = labelmap.GetPointData().GetScalars()
    if scalars is None or scalars.GetNumberOfTuples() == 0:
        return out

    # Map labelmap-local voxel (0,0,0) → reference-volume IJK.
    # Works for both 0-based (compact, world-encoded origin) and offset extents.
    lm_to_world = vtk.vtkMatrix4x4()
    labelmap.GetImageToWorldMatrix(lm_to_world)
    if ras_to_ijk is None:
        ras_to_ijk = vtk.vtkMatrix4x4()
        ref_vol.GetRASToIJKMatrix(ras_to_ijk)
    origin_ras = [0.0, 0.0, 0.0, 1.0]
    lm_to_world.MultiplyPoint([0.0, 0.0, 0.0, 1.0], origin_ras)
    ref_ijk = [0.0, 0.0, 0.0, 0.0]
    ras_to_ijk.MultiplyPoint(origin_ras, ref_ijk)
    ref_i = int(round(ref_ijk[0]))  # ref-vol I of labelmap voxel (0,0,0)
    ref_j = int(round(ref_ijk[1]))
    ref_k = int(round(ref_ijk[2]))

    ext = labelmap.GetExtent()   # labelmap-local (li0, li1, lj0, lj1, lk0, lk1)
    li0, li1, lj0, lj1, lk0, lk1 = ext
    ni_s = li1 - li0 + 1
    nj_s = lj1 - lj0 + 1
    nk_s = lk1 - lk0 + 1
    arr = vtk_to_numpy(scalars).reshape(nk_s, nj_s, ni_s)  # view, no copy

    def _place(raw2d, dest_r0, dest_c0):
        """Write raw2d into out at (dest_r0, dest_c0), clamping to volume bounds."""
        h, w = raw2d.shape
        r0 = max(0, dest_r0); r1 = min(out.shape[0], dest_r0 + h)
        c0 = max(0, dest_c0); c1 = min(out.shape[1], dest_c0 + w)
        if r0 < r1 and c0 < c1:
            out[r0:r1, c0:c1] = raw2d[r0 - dest_r0:r1 - dest_r0,
                                       c0 - dest_c0:c1 - dest_c0].astype(bool)

    if axis == 0:   # K-slice: out shape (nj, ni)
        arr_k = (slice_idx - ref_k) - lk0
        if not (0 <= arr_k < nk_s):
            return out
        _place(arr[arr_k, :, :], ref_j + lj0, ref_i + li0)
    elif axis == 1:  # J-slice: out shape (nk, ni)
        arr_j = (slice_idx - ref_j) - lj0
        if not (0 <= arr_j < nj_s):
            return out
        _place(arr[:, arr_j, :], ref_k + lk0, ref_i + li0)
    else:            # I-slice: out shape (nk, nj)
        arr_i = (slice_idx - ref_i) - li0
        if not (0 <= arr_i < ni_s):
            return out
        _place(arr[:, :, arr_i], ref_k + lk0, ref_j + lj0)

    return out


def _spx_on_stroke_press(handler, view_name, axis, slice_idx):
    """Snapshot the pre-stroke 2-D slice from the segment's native labelmap."""
    if SPX_DEBUG_TIMING:
        t = _T()
    editor = handler._widget_ref.logic.get_segment_editor()
    if editor is None:
        return
    vol    = editor.sourceVolumeNode()
    seg    = editor.segmentationNode()
    seg_id = editor.currentSegmentID()
    if not (vol and seg and seg_id):
        return
    _ijk_to_ras, ras_to_ijk = _spx_get_vol_mats(handler, vol)
    if SPX_DEBUG_TIMING:
        t.mark('get_vol_mats')
    handler._pre_stroke_slices[view_name] = \
        _spx_read_native_slice(seg, seg_id, vol, axis, slice_idx,
                               ras_to_ijk=ras_to_ijk)
    if SPX_DEBUG_TIMING:
        t.mark('read_native_slice')
        print(f'[SPX press {view_name}] {t}')


def _spx_post_process_stroke(handler, view_name, axis, slice_idx):
    """Diff pre/post stroke, apply all touched SPX labels in one write, no snapshot."""
    import numpy as np
    from .utils import get_slice_from_volume
    from ._logic import apply_spx_labels_batch

    if SPX_DEBUG_TIMING:
        t = _T()

    pre_slice = handler._pre_stroke_slices.pop(view_name, None)
    if pre_slice is None:
        return
    editor = handler._widget_ref.logic.get_segment_editor()
    if editor is None:
        return
    vol    = editor.sourceVolumeNode()
    seg    = editor.segmentationNode()
    seg_id = editor.currentSegmentID()
    if not (vol and seg and seg_id):
        return

    ijk_to_ras, ras_to_ijk = _spx_get_vol_mats(handler, vol)
    post_slice = _spx_read_native_slice(seg, seg_id, vol, axis, slice_idx,
                                        ras_to_ijk=ras_to_ijk)
    if SPX_DEBUG_TIMING:
        t.mark('read_native_slice')

    changed    = post_slice != pre_slice
    if not changed.any():
        if SPX_DEBUG_TIMING:
            print(f'[SPX post {view_name}] no change — early exit  {t}')
        return

    # Handler-level label cache: skip arrayFromVolume + on_expand on repeat strokes.
    params = handler._widget_ref.getUserParameters() or {}
    try:
        params_frozen = frozenset(params.items())
    except TypeError:
        params_frozen = str(sorted(params.items()))
    label_key = (vol.GetID(), axis, int(slice_idx), params_frozen)

    labels = handler._spx_label_cache.get(label_key)
    if labels is None:
        vol_arr = slicer.util.arrayFromVolume(vol)
        img     = get_slice_from_volume(vol_arr, axis, slice_idx)
        labels  = handler._widget_ref.modelFamily.on_expand(
            volume_node=vol, axis=axis, slice_idx=int(slice_idx),
            img=img, **params)
        if labels is None:
            return
        handler._spx_label_cache[label_key] = labels
        if SPX_DEBUG_TIMING:
            t.mark('label_cache=MISS+on_expand')
    else:
        if SPX_DEBUG_TIMING:
            t.mark('label_cache=HIT')

    touched_ids = sorted(int(lid) for lid in np.unique(labels[changed]) if lid != 0)
    if not touched_ids:
        if SPX_DEBUG_TIMING:
            print(f'[SPX post {view_name}] no touched ids  {t}')
        return

    # Build one combined mask covering all touched superpixels.
    combined = np.zeros_like(labels, dtype=np.uint8)
    for lid in touched_ids:
        combined |= (labels == lid)

    # Crop to the tight bounding box so the modifier only covers the painted region.
    rows, cols = np.where(combined)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    if SPX_DEBUG_TIMING:
        t.mark(f'build_mask(npx={int(combined.sum())},bbox={r1-r0+1}x{c1-c0+1})')

    apply_spx_labels_batch(
        handler._widget_ref, touched_ids, combined[r0:r1+1, c0:c1+1],
        axis, int(slice_idx), handler._ADDITIVE, view_name,
        row_offset=r0, col_offset=c0,
        ijk_to_ras_mat=ijk_to_ras, modifier_box=handler._modifier_box)
    if SPX_DEBUG_TIMING:
        t.mark(f'apply_batch(nlabels={len(touched_ids)})')
        print(f'[SPX post {view_name}] {t}')


def _spx_lock_radius(handler, widget):
    """Lock the brush to an absolute fixed diameter; keep BrushDiameterIsRelative=0."""
    editor = widget.logic.get_segment_editor()
    if editor is None:
        return
    effect = editor.activeEffect()
    if effect is None:
        return
    effect.setParameter('BrushAbsoluteDiameter',
                        str(handler._SPX_BRUSH_DIAMETER_MM))
    effect.setParameter('BrushDiameterIsRelative', '0')


def _spx_sync_buttons(handler, widget, active):
    for btn_name in ('spxBrushToolButton', 'spxEraseToolButton'):
        btn = getattr(widget.ui, btn_name, None)
        if btn:
            btn.blockSignals(True)
            btn.setChecked(active and btn_name == handler.BUTTON_NAME)
            btn.blockSignals(False)


def _spx_preallocate_for_widget(handler, widget):
    """Pre-expand the active segment's labelmap to full volume at SPX activation time."""
    from ._logic import _spx_preallocate_full_labelmap
    editor = widget.logic.get_segment_editor()
    if editor is None:
        return
    vol    = editor.sourceVolumeNode()
    seg    = editor.segmentationNode()
    seg_id = editor.currentSegmentID()
    if not (vol and seg and seg_id):
        return
    ijk_to_ras, _ = _spx_get_vol_mats(handler, vol)
    _spx_preallocate_full_labelmap(seg, seg_id, vol, ijk_to_ras)


# ======================================================================== #
# SPX handlers                                                              #
# ======================================================================== #

class SpxBrushHandler(Brush2DHandler):
    """Paint with SPX snapping: after each stroke a labelmap diff identifies
    which superpixels were touched, then fills them completely via apply_spx_label.
    """

    BUTTON_NAME            = 'spxBrushToolButton'
    TOOL_NAME              = 'spx_brush'
    _ADDITIVE              = True
    _SPX_BRUSH_DIAMETER_MM = 10.0

    def __init__(self):
        super().__init__()
        self._pre_stroke_slices: dict = {}   # vn → 2D pre-stroke bool array
        self._mat_cache: dict = {}           # vol_id → (ijk_to_ras, ras_to_ijk) vtk matrices
        self._modifier_box: list = [None]    # [vtkOrientedImageData] reused across strokes
        self._spx_label_cache: dict = {}     # label_key → labels array (one slot per axis/slice)

    def _on_attach(self, widget) -> None:
        from .modelFamilies import SPXModelFamily
        if not isinstance(widget.modelFamily, SPXModelFamily) or not widget.modelFamily.model:
            raise RuntimeError(
                'SpxBrushHandler requires an active SPXModelFamily with a confirmed model.')
        super()._on_attach(widget)
        _spx_lock_radius(self, widget)
        # Defer pre-expansion to the next Qt event-loop tick so that all deferred
        # MRML events queued by setActiveEffectByName are processed first.
        import qt
        qt.QTimer.singleShot(0, lambda: _spx_preallocate_for_widget(self, widget))

    def _on_stroke_press(self, view_name, axis, slice_idx):
        _spx_on_stroke_press(self, view_name, axis, slice_idx)

    def _post_process_stroke(self, view_name, axis, slice_idx):
        _spx_post_process_stroke(self, view_name, axis, slice_idx)

    def _sync_buttons(self, widget, active: bool) -> None:
        _spx_sync_buttons(self, widget, active)


class SpxEraseHandler(Erase2DHandler):
    """Erase with SPX snapping: after each stroke a labelmap diff identifies
    which superpixels were touched, then erases them completely via apply_spx_label.
    """

    BUTTON_NAME            = 'spxEraseToolButton'
    TOOL_NAME              = 'spx_erase'
    _ADDITIVE              = False
    _SPX_BRUSH_DIAMETER_MM = 10.0

    def __init__(self):
        super().__init__()
        self._pre_stroke_slices: dict = {}   # vn → 2D pre-stroke bool array
        self._mat_cache: dict = {}           # vol_id → (ijk_to_ras, ras_to_ijk) vtk matrices
        self._modifier_box: list = [None]    # [vtkOrientedImageData] reused across strokes
        self._spx_label_cache: dict = {}     # label_key → labels array (one slot per axis/slice)

    def _on_attach(self, widget) -> None:
        from .modelFamilies import SPXModelFamily
        if not isinstance(widget.modelFamily, SPXModelFamily) or not widget.modelFamily.model:
            raise RuntimeError(
                'SpxEraseHandler requires an active SPXModelFamily with a confirmed model.')
        super()._on_attach(widget)
        _spx_lock_radius(self, widget)
        import qt
        qt.QTimer.singleShot(0, lambda: _spx_preallocate_for_widget(self, widget))

    def _on_stroke_press(self, view_name, axis, slice_idx):
        _spx_on_stroke_press(self, view_name, axis, slice_idx)

    def _post_process_stroke(self, view_name, axis, slice_idx):
        _spx_post_process_stroke(self, view_name, axis, slice_idx)

    def _sync_buttons(self, widget, active: bool) -> None:
        _spx_sync_buttons(self, widget, active)
