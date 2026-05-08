"""Input handlers for SegmentHumanBodyWidget (native-editor-wrapper branch).

InputHandler   — base: volume check, mutual-exclusion, segment-existence guard,
                 and active-handler registration.  Subclasses override _on_attach.
StrokeHandler  — shared Brush/Erase logic (effect activation, UI sync).
BrushHandler   — Paint effect.
EraseHandler   — Erase effect.
PointHandler   — prompt-point mode (no extra setup beyond the base guard).

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
    """

    EFFECT: str = ''
    BUTTON_NAME: str = ''

    def _on_attach(self, widget) -> None:
        vol    = widget.ui.sourceVolumeSelector.currentNode()
        seg    = widget.ui.segmentationNodeSelector.currentNode()
        seg_id = widget.ui.segmentSelector.currentSegmentID()
        editor = widget.logic.get_segment_editor()
        widget.logic.setup_editor_nodes(editor, vol, seg, seg_id)
        if editor:
            editor.setActiveEffectByName(self.EFFECT)
            widget._borrowEffectsOptionsFrame()
            slicer.app.applicationLogic().GetInteractionNode().SwitchToViewTransformMode()
        self._sync_buttons(widget, active=True)

    def _on_attach_cancelled(self, widget) -> None:
        self._sync_buttons(widget, active=False)

    def _on_detach(self, widget) -> None:
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


class BrushHandler(StrokeHandler):
    """Activates the Segment Editor Paint effect."""
    EFFECT = 'Paint'
    BUTTON_NAME = 'brushToolButton'
    TOOL_NAME = 'brush'


class EraseHandler(StrokeHandler):
    """Activates the Segment Editor Erase effect."""
    EFFECT = 'Erase'
    BUTTON_NAME = 'eraseToolButton'
    TOOL_NAME = 'erase'


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
