"""Input handlers for SegmentHumanBodyWidget.

_SliceViewMouseFilter  — callback-based Qt event filter for stroke boundaries.
InputHandler           — base class: attach / detach / flush lifecycle.
StrokeHandler          — owns mouse filter + stroke state for Paint / Erase.
BrushHandler           — Paint effect, tracks all strokes.
EraseHandler           — Erase effect, skips no-op (non-removing) strokes.
"""

import logging
import numpy as np
import qt
import slicer

log = logging.getLogger(__name__)


class _SliceViewMouseFilter(qt.QObject):
    """Qt application-level event filter for stroke boundary detection.

    Callback-based: takes on_press / on_release callables so the owner does
    not need to implement a specific interface.  Returns False always so
    events are never consumed.
    """

    def __init__(self, on_press, on_release):
        super().__init__()
        self._on_press   = on_press
        self._on_release = on_release

    def eventFilter(self, obj, event):
        t = event.type()
        try:
            if t == qt.QEvent.MouseButtonPress and event.button() == qt.Qt.LeftButton:
                self._on_press()
            elif t == qt.QEvent.MouseButtonRelease and event.button() == qt.Qt.LeftButton:
                self._on_release()
        except Exception as exc:
            log.error('[MouseFilter] %s', exc)
        return False


class InputHandler:
    """Base class for interactive input modes."""

    def attach(self, widget):
        """Install this handler on *widget*."""

    def detach(self, widget):
        """Remove this handler and clean up any pending state."""

    def flush(self, widget):
        """Commit any pending state synchronously."""


class StrokeHandler(InputHandler):
    """Paint / Erase session handler.

    Owns the Qt mouse filter and the per-stroke before-state snapshot.
    Replaces the widget-level ``_brushMouseFilter``, ``_toolValidatorTimer``,
    and ``_stroke_before`` fields.

    Subclasses set EFFECT / SOURCE and may override ``_should_track``.
    """

    EFFECT: str = ''   # 'Paint' or 'Erase'
    SOURCE: str = ''   # 'brush' or 'erase'

    def __init__(self):
        self._stroke_before = None   # (axis, idx, before) while stroke is live
        self._mouse_filter  = None
        self._effect_cb     = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def attach(self, widget):
        """Activate the Segment Editor effect and install event listeners."""
        self._activate_effect(widget)

        self._mouse_filter = _SliceViewMouseFilter(
            on_press   = lambda: self._on_stroke_start(widget),
            on_release = lambda: self._on_stroke_end(widget),
        )
        slicer.app.installEventFilter(self._mouse_filter)

        editor = widget._segEditor()
        if editor:
            self._effect_cb = lambda: self._on_effect_changed(widget)
            editor.connect('activeEffectChanged()', self._effect_cb)

    def detach(self, widget):
        """Flush any pending stroke, deactivate the effect, remove listeners."""
        self.flush(widget)

        if self._mouse_filter:
            slicer.app.removeEventFilter(self._mouse_filter)
            self._mouse_filter = None

        editor = widget._segEditor()
        if editor and self._effect_cb:
            editor.disconnect('activeEffectChanged()', self._effect_cb)
        self._effect_cb = None

        if editor:
            editor.setActiveEffectByName("")

        ui = widget.ui
        for btn in (ui.brushToolButton, ui.eraseToolButton):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)

    def reset(self, widget):
        """Discard any pending before-state without committing."""
        self._stroke_before = None
        widget.ctrl.brush_in_progress = False

    def flush(self, widget):
        """Force-apply and commit any in-progress or pending stroke."""
        if widget.ctrl.brush_in_progress:
            editor = widget._segEditor()
            if editor:
                effect = editor.activeEffect()
                if effect:
                    try:
                        effect.self().apply()
                    except Exception as exc:
                        log.warning('[StrokeHandler] apply() failed: %s', exc)
            widget.ctrl.brush_in_progress = False

        self._do_commit(widget)

    # ------------------------------------------------------------------ #
    # Event callbacks                                                      #
    # ------------------------------------------------------------------ #

    def _on_stroke_start(self, widget):
        if not (widget.ui.brushToolButton.isChecked()
                or widget.ui.eraseToolButton.isChecked()):
            return
        if widget.ctrl.brush_in_progress:
            return

        # Resolve the active view NOW — the cursor is over a slice view at
        # the moment of the click, so underMouse() is reliable here.
        widget._resolveActiveView()

        # Rapid double-click: flush any uncommitted before-state first.
        if self._stroke_before is not None:
            self._do_commit(widget)

        axis, idx, before = widget.logic.capture_current_slice(widget)
        if before is not None:
            self._stroke_before = (axis, idx, before)
        widget.ctrl.brush_in_progress = True
        widget.logic._last_render_key = None   # invalidate render key; _session_base preserved for commit_stroke
        log.debug('[%s] stroke start — history %d', self.SOURCE, len(widget._history))

    def _on_stroke_end(self, widget):
        if not widget.ctrl.brush_in_progress:
            return
        widget.ctrl.brush_in_progress = False
        if self._stroke_before is not None:
            # 0-ms timer fires after VTK's Paint effect apply() commits the stroke.
            qt.QTimer.singleShot(0, lambda: self._do_commit(widget))

    def _do_commit(self, widget):
        if self._stroke_before is None:
            return
        axis, idx, before = self._stroke_before
        self._stroke_before = None
        change = widget.logic.commit_stroke(widget, axis, idx, before, self.SOURCE)
        if change is not None and self._should_track(change):
            widget._add_history([self.SOURCE, change])
        log.debug('[%s] committed — change=%s  history=%d',
                  self.SOURCE, change is not None, len(widget._history))

    def _on_effect_changed(self, widget):
        """Detach when the Segment Editor effect is changed externally."""
        if widget.ctrl.activating_brush:
            return
        editor = widget._segEditor()
        effect = editor.activeEffect() if editor else None
        if (effect.name if effect else None) != self.EFFECT:
            widget._set_stroke_handler(None)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _should_track(self, change) -> bool:
        """Return True when this stroke should be pushed to history."""
        return True

    def _activate_effect(self, widget):
        """Configure Segment Editor and activate EFFECT; sync UI buttons."""
        editor = widget._segEditor()
        if editor is None:
            return

        volNode, segNode = widget.logic.getVolumeAndSegmentation(widget._parameterNode)

        if self.EFFECT == "Paint" and segNode and \
                segNode.GetSegmentation().GetNumberOfSegments() == 0:
            widget.onAddSegment()

        if not volNode or not segNode:
            return

        widget.ctrl.activating_brush = True
        try:
            editor.setSegmentationNode(segNode)
            widget.ctrl.brush_in_progress = False
            editor.setSourceVolumeNode(volNode)
            editor.setUndoEnabled(True)
            editor.setMaximumNumberOfUndoStates(50)
            segID = widget.ui.segmentSelector.currentSegmentID()
            if segID:
                editor.setCurrentSegmentID(segID)
            editor.setActiveEffectByName(self.EFFECT)
            widget._applyBrushParams()
        finally:
            widget.ctrl.activating_brush = False

        # Restore SPX overlay if editor node setup displaced it.
        if widget._spx_boundary_visible and widget._spx_boundary_node:
            composite = widget._get_composite_node(widget._spx_boundary_view)
            if composite:
                composite.SetLabelVolumeID(widget._spx_boundary_node.GetID())
                composite.SetLabelOpacity(0.8)

        # Switch interaction to view-transform so clicks paint, not place markups.
        slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()

        # Reflect the active tool in the UI buttons.
        ui = widget.ui
        for btn, active in ((ui.brushToolButton, self.SOURCE == 'brush'),
                            (ui.eraseToolButton, self.SOURCE == 'erase')):
            btn.blockSignals(True)
            btn.setChecked(active)
            btn.blockSignals(False)


class BrushHandler(StrokeHandler):
    """Paint (additive) stroke handler."""
    EFFECT = 'Paint'
    SOURCE = 'brush'


class EraseHandler(StrokeHandler):
    """Erase stroke handler — skips strokes that removed no positive pixels."""
    EFFECT = 'Erase'
    SOURCE = 'erase'

    def _should_track(self, change) -> bool:
        return bool(np.any(change.delta < 0))
