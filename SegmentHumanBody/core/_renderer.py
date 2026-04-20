"""Rendering helpers for the SegmentHumanBody Slicer module.

_SliceViewMouseFilter — Qt application-level event filter that detects the
start and end of brush strokes before any VTK interactor can absorb them.
"""

import logging
import qt

log = logging.getLogger(__name__)


class _SliceViewMouseFilter(qt.QObject):
    """Qt application-level event filter for detecting brush stroke boundaries.

    Installed on slicer.app (QApplication) so it fires for every Qt mouse
    event in the entire application — before any VTK interactor observer,
    and before the Segment Editor Paint/Erase effect can set the VTK abort
    flag.  This is the only reliable hook point: VTK-level LeftButtonPressEvent
    observers are silently skipped when an active Segment Editor effect has
    already consumed the event at higher priority.

    The callback guard (brushToolButton/eraseToolButton.isChecked) is checked
    inside _onBrushStrokeStart, so this filter is a no-op during any non-brush
    interaction.  The button is NOT yet toggled when its own MouseButtonPress
    fires (Qt buttons toggle on release), so clicking the brush button itself
    never triggers a spurious undo entry.
    """

    def __init__(self, slicerWidget):
        super().__init__()
        self._slicerWidget = slicerWidget

    def eventFilter(self, obj, event):
        t = event.type()
        try:
            if t == qt.QEvent.MouseButtonPress and event.button() == qt.Qt.LeftButton:
                self._slicerWidget._onBrushStrokeStart()
            elif t == qt.QEvent.MouseButtonRelease and event.button() == qt.Qt.LeftButton:
                self._slicerWidget._onBrushStrokeEnd()
        except Exception as exc:
            log.error("[MouseFilter] brush event handler raised: %s", exc)
        return False  # never consume — always propagate to the target widget


