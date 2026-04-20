"""Widget state for the SegmentHumanBody Slicer module.

WidgetState — render-loop gating and tool-mode flags.
"""

import logging
import qt

log = logging.getLogger(__name__)


class WidgetState:
    """Render-loop gating and tool-mode flags for SegmentHumanBodyWidget.

    Centralises all boolean semaphores so every state transition is visible
    in one place, eliminating race conditions from flags scattered across
    the widget.

    Parameters
    ----------
    widget : SegmentHumanBodyWidget
        The owning widget.
    """

    def __init__(self, widget):
        self._widget = widget

        # Render-loop flags
        self._rendering   = False
        self._pending     = False
        self._pause_depth = 0   # nestable: each pause() must be paired with resume()

        # Tool-mode flags (read and written directly by the widget)
        self.activating_brush  = False   # True while StrokeHandler._activate_effect is running
        self.brush_in_progress = False   # True between stroke-start and stroke-end
        self.creating_segment  = False   # suppresses onSegmentChanged during add

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        """True when at least one ``pause()`` call is outstanding."""
        return self._pause_depth > 0

    @property
    def is_rendering(self) -> bool:
        """True while ``request_render`` is executing its callback."""
        return self._rendering

    # ------------------------------------------------------------------
    # Pause / resume (nestable)
    # ------------------------------------------------------------------

    def pause(self):
        """Increment the pause counter.

        While paused, ``request_render`` is a no-op and event handlers that
        check ``is_paused`` skip their work.  Call ``resume()`` exactly once
        for each ``pause()`` — typically in a try/finally block.
        """
        self._pause_depth += 1

    def resume(self):
        """Decrement the pause counter.  Renders resume when it reaches zero."""
        if self._pause_depth > 0:
            self._pause_depth -= 1

    # ------------------------------------------------------------------
    # Render dispatch
    # ------------------------------------------------------------------

    def request_render(self, on_render=None, on_error=None):
        """Request a single render pass, handling re-entrancy and pause state.

        Safe to call from any context, including from within an active render
        pass.  In that case the request is queued and fired after the current
        pass completes so no render is silently lost.

        Parameters
        ----------
        on_render : callable, optional
            Zero-argument callable that performs the render.  Defaults to
            ``widget.logic.onRender(widget.modelFamily, widget)``.
        on_error : callable(Exception), optional
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

        self._pending   = False
        self._rendering = True
        try:
            on_render()
        except Exception as exc:
            if on_error:
                on_error(exc)
            else:
                log.error('[WidgetState] render error: %s', exc)
        finally:
            self._rendering = False

        if self._pending:
            # Bind callbacks as default args to avoid closure-capture bugs when
            # request_render is called again before the timer fires.
            qt.QTimer.singleShot(
                0,
                lambda r=on_render, e=on_error: self.request_render(r, e),
            )
