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

    # ------------------------------------------------------------------ #
    # Guard helpers                                                        #
    # ------------------------------------------------------------------ #

    def _detach_current(self, widget) -> None:
        """Flush and detach the active handler."""
        current = getattr(widget, '_active_handler', None)
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
        elif hasattr(widget, '_ensure_current_prompt_nodes'):
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
        self._detach_current(widget)
        self._ensure_segment(widget)
        widget._active_handler = self
        self._on_attach(widget)

    def _on_attach(self, widget) -> None:
        """Override in subclasses to install tool-specific resources."""

    def _on_attach_cancelled(self, widget) -> None:
        """Override in subclasses to reset UI when attach is aborted."""

    def detach(self, widget) -> None:
        """_on_detach → unregister.  Subclasses override _on_detach, not detach."""
        self._on_detach(widget)
        if getattr(widget, '_active_handler', None) is self:
            widget._active_handler = None

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


class EraseHandler(StrokeHandler):
    """Activates the Segment Editor Erase effect."""
    EFFECT = 'Erase'
    BUTTON_NAME = 'eraseToolButton'


class PointHandler(InputHandler):
    """Segment guard for prompt-point placement mode.

    Actual placement is managed by qSlicerSimpleMarkupsWidget.  This handler
    exists so that entering point-placement mode runs the same unified
    segment-existence guard as brush and erase, ensuring every placed point
    is immediately associated with a valid segment.  No extra setup is needed
    beyond what the base class already provides.
    """

    def _on_attach(self, widget) -> None:
        if hasattr(widget, '_ensure_current_prompt_nodes'):
            widget._ensure_current_prompt_nodes()
        if hasattr(widget, '_set_prompt_widget_place_mode'):
            widget._set_prompt_widget_place_mode(
                getattr(widget, '_active_prompt_widget', None), True)

    def _on_detach(self, widget) -> None:
        if hasattr(widget, '_set_prompt_widget_place_mode'):
            widget._set_prompt_widget_place_mode(
                getattr(widget, '_active_prompt_widget', None), False)
        if hasattr(widget, '_deactivate_prompt_place_mode'):
            widget._deactivate_prompt_place_mode()
