"""
Unit tests for _applyOverwriteMode.

These run under PythonSlicer.exe -m pytest (no live Slicer app, so
slicer.mrmlScene and slicer.vtkMRMLSegmentEditorNode are unavailable).

_FakeEditorNodeClass intentionally omits PaintAllowedEverywhere and
PaintAllowedOutsideAllSegments so any accidental use of cls.<MaskConstant>
raises AttributeError — exactly what the real Slicer class does.

For integration tests that verify the real MRML node state after
_applyOverwriteMode(), see Testing/Python/SegmentHumanBodyTest.py
(class OverwriteModeApplyTest), which runs under Slicer.exe --no-main-window.
"""
import types
import SegmentHumanBody as segment_module
from SegmentHumanBody import SegmentHumanBodyWidget


class _RecordingNode:
    """Captures SetMaskMode / SetOverwriteMode call arguments."""
    def __init__(self):
        self.mask_mode_calls = []
        self.overwrite_mode_calls = []

    def SetMaskMode(self, mode):
        self.mask_mode_calls.append(mode)

    def SetOverwriteMode(self, mode):
        self.overwrite_mode_calls.append(mode)

    def __bool__(self):
        return True


class _FakeEditorNodeClass:
    """Mimics vtkMRMLSegmentEditorNode with only the attributes Slicer exposes.

    PaintAllowedEverywhere / PaintAllowedOutsideAllSegments are absent because
    the real Slicer class does not expose them as Python attributes either.
    A test that accidentally uses cls.PaintAllowedEverywhere will raise
    AttributeError here, matching the real failure mode.
    """
    OverwriteNone        = 2
    OverwriteAllSegments = 0


class _Dropdown:
    def __init__(self, index):
        self.currentIndex = index


def _patched_widget(monkeypatch, dropdown_index):
    node = _RecordingNode()
    fake_slicer = types.SimpleNamespace(
        mrmlScene=types.SimpleNamespace(GetSingletonNode=lambda *_: node),
        vtkMRMLSegmentEditorNode=_FakeEditorNodeClass,
    )
    monkeypatch.setattr(segment_module, 'slicer', fake_slicer)
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget.ui = types.SimpleNamespace(overwriteModeDropdown=_Dropdown(dropdown_index))
    return widget, node


# ---------------------------------------------------------------------------
# Mode 0 — Coexist
# ---------------------------------------------------------------------------

def test_coexist_sets_mask_everywhere_and_overwrite_none(monkeypatch):
    widget, node = _patched_widget(monkeypatch, 0)
    widget._applyOverwriteMode()
    assert node.mask_mode_calls    == [0]                                 # PaintAllowedEverywhere
    assert node.overwrite_mode_calls == [_FakeEditorNodeClass.OverwriteNone]


# ---------------------------------------------------------------------------
# Mode 1 — Aggressive
# ---------------------------------------------------------------------------

def test_aggressive_sets_mask_everywhere_and_overwrite_all(monkeypatch):
    widget, node = _patched_widget(monkeypatch, 1)
    widget._applyOverwriteMode()
    assert node.mask_mode_calls    == [0]                                 # PaintAllowedEverywhere
    assert node.overwrite_mode_calls == [_FakeEditorNodeClass.OverwriteAllSegments]


# ---------------------------------------------------------------------------
# Mode 2 — Defensive
# ---------------------------------------------------------------------------

def test_defensive_sets_mask_outside_all_segments(monkeypatch):
    widget, node = _patched_widget(monkeypatch, 2)
    widget._applyOverwriteMode()
    assert node.mask_mode_calls    == [3]                                 # PaintAllowedOutsideAllSegments
    assert node.overwrite_mode_calls == [_FakeEditorNodeClass.OverwriteNone]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_out_of_range_index_falls_back_to_coexist(monkeypatch):
    widget, node = _patched_widget(monkeypatch, 99)
    widget._applyOverwriteMode()
    assert node.mask_mode_calls    == [0]
    assert node.overwrite_mode_calls == [_FakeEditorNodeClass.OverwriteNone]


def test_no_op_when_singleton_absent(monkeypatch):
    fake_slicer = types.SimpleNamespace(
        mrmlScene=types.SimpleNamespace(GetSingletonNode=lambda *_: None),
        vtkMRMLSegmentEditorNode=_FakeEditorNodeClass,
    )
    monkeypatch.setattr(segment_module, 'slicer', fake_slicer)
    widget = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    widget.ui = types.SimpleNamespace(overwriteModeDropdown=_Dropdown(0))
    widget._applyOverwriteMode()   # must not raise
