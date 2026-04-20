"""RenderController — central state machine for all rendering-related flags.

Replaces the scattered boolean semaphores that previously lived directly on
SegmentHumanBodyWidget:

  _isRendering          → ctrl.is_rendering   (read-only property)
  _pauseRender          → ctrl.is_paused       (read-only property)
                           ctrl.pause() / ctrl.resume()
  _render_pending       → internal to ctrl
  _activatingBrushEffect→ ctrl.activating_brush
  _brushInProgress      → ctrl.brush_in_progress
  _creating_segment     → ctrl.creating_segment

Having all flags in one place makes every state transition visible and
eliminates race conditions that arose from flags being toggled from multiple
uncoordinated call sites.
"""

import logging
import qt

log = logging.getLogger(__name__)


class RenderController:
    """Central state machine for rendering and interaction flags.

    Parameters
    ----------
    widget : SegmentHumanBodyWidget
        The owning widget.  The controller calls back into the widget only
        through the ``logic`` and ``ui`` attributes it already exposes.
    """

    def __init__(self, widget):
        self._widget = widget

        # Core render-loop flags
        self._rendering = False
        self._pending   = False
        self._pause_depth = 0   # supports nested pause() / resume() calls

        # Interaction / tool mode flags
        self.activating_brush  = False   # True while _activateBrushEffect is running
        self.brush_in_progress = False   # True between stroke-start and stroke-end
        self.creating_segment  = False   # suppresses onSegmentChanged during add

    # ------------------------------------------------------------------
    # Read-only state properties
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        """True when at least one ``pause()`` call is outstanding."""
        return self._pause_depth > 0

    @property
    def is_rendering(self) -> bool:
        """True while ``request_render`` is executing its render callback."""
        return self._rendering

    # ------------------------------------------------------------------
    # Pause / resume (nestable)
    # ------------------------------------------------------------------

    def pause(self):
        """Increment the pause counter.

        While paused, ``request_render`` is a no-op and event handlers that
        check ``ctrl.is_paused`` skip their work.  Call ``resume()`` exactly
        once for each ``pause()`` call (typically in a try/finally block).
        """
        self._pause_depth += 1

    def resume(self):
        """Decrement the pause counter.  Renders resume when it reaches zero."""
        if self._pause_depth > 0:
            self._pause_depth -= 1

    # ------------------------------------------------------------------
    # Render request
    # ------------------------------------------------------------------

    def request_render(self, on_render=None, on_error=None):
        """Request a single render pass, handling re-entrancy and pause state.

        Safe to call from any context — including from within a render pass
        already in progress.  In that case the request is queued and fired
        after the current pass completes, so no render is ever silently lost.

        Parameters
        ----------
        on_render : callable, optional
            Zero-argument callable that performs the actual render.  Defaults
            to ``widget.logic.onRender(widget.modelFamily, widget)``.
        on_error : callable(Exception), optional
            Called when ``on_render`` raises.  Defaults to logging only.
            Called when ``on_render`` raises.  Defaults to logging only.
        """
        w = self._widget
        if self._pause_depth > 0 or not w.modelFamily or not w.modelFamily.model:
            return
        if self._rendering:
            self._pending = True
            return

        if on_render is None:
            on_render = lambda: w.logic.onRender(w.modelFamily, w)

        self._pending  = False
        self._rendering = True
        try:
            on_render()
        except Exception as exc:
            if on_error:
                on_error(exc)
            else:
                log.error(f"[RenderController] render error: {exc}")
        finally:
            self._rendering = False

        if self._pending:
            # Capture current callbacks in the closure so the deferred call
            # uses the same on_render / on_error pair.
            _on_render = on_render
            _on_error  = on_error
            qt.QTimer.singleShot(0, lambda: self.request_render(_on_render, _on_error))

