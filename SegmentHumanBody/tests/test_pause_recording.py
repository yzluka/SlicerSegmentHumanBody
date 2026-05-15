"""Tests for the recording pause/resume feature.

Covers:
  - MouseEventRecorder pause/resume state and is_paused property
  - _on_mouse guard: events dropped while paused, button state still tracked
  - Widget _do_resume_recording: interval accumulation across multiple cycles
  - Widget _update_record_ui: pause button enabled/disabled state
  - Widget _finalize_wav: prewarm lead-in trim and pause interval silencing
"""
import datetime
import struct
import wave

import pytest

from core._mouse_recorder import MouseEventRecorder, PRESS, RELEASE, MOVE
import core._mouse_recorder as recorder_mod
from SegmentHumanBody import SegmentHumanBodyWidget


# ── shared helpers ────────────────────────────────────────────────────────────

def _dt(offset_sec=0.0):
    return datetime.datetime(2026, 1, 1) + datetime.timedelta(seconds=offset_sec)


def _make_wav(path, sr, samples):
    """Write a mono PCM16 WAV with the given int16 sample list."""
    data = struct.pack(f'<{len(samples)}h', *samples)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data)


def _read_wav_samples(path):
    """Read a WAV back as a list of int16 values."""
    with wave.open(str(path), 'rb') as wf:
        data = wf.readframes(wf.getnframes())
    return list(struct.unpack(f'<{len(data) // 2}h', data))


# ── stubs ─────────────────────────────────────────────────────────────────────

class _W:
    """Minimal widget attribute stub."""
    def __init__(self):
        self.enabled = None
        self.text = ''
    def setEnabled(self, v): self.enabled = v
    def setText(self, v):    self.text = v
    def isChecked(self):     return False
    def blockSignals(self, v): pass


class _PauseUI:
    def __init__(self):
        self.recordToggleButton  = _W()
        self.pauseRecordButton   = _W()
        self.recordMouseKeyCheckBox = _W()
        self.recordAudioCheckBox = _W()
        self.audioDeviceComboBox = _W()
        self.exportRecordButton  = _W()
        self.recordStatusLabel   = _W()


class _PausableRecorder:
    def __init__(self, *, active=True):
        self.is_active = active
        self._paused   = False
        self.resumed   = 0

    @property
    def is_paused(self):
        return self.is_active and self._paused

    def pause(self):
        if self.is_active:
            self._paused = True

    def resume(self):
        self._paused = False
        self.resumed += 1

    def __len__(self):
        return 0


class _FakeAudioSubprocess:
    def __init__(self, start_time):
        self.start_time = start_time


def _pause_widget(recording_start, pause_start, intervals=None):
    w = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    w._recorder            = _PausableRecorder(active=True)
    w._recording_start_time = recording_start
    w._pause_start_time    = pause_start
    w._pause_intervals     = list(intervals or [])
    w._audio_recorder      = None
    w._audio_only_mode     = False
    w.ui                   = _PauseUI()
    return w


def _finalize_widget(recording_start, audio_start=None, pause_intervals=None):
    w = SegmentHumanBodyWidget.__new__(SegmentHumanBodyWidget)
    w._recording_start_time = recording_start
    w._audio_recorder = (
        _FakeAudioSubprocess(audio_start) if audio_start is not None else None
    )
    w._pause_intervals = list(pause_intervals or [])
    return w


# ── recorder pause state ──────────────────────────────────────────────────────

def test_recorder_is_not_paused_initially():
    assert MouseEventRecorder().is_paused is False


def test_recorder_pause_is_noop_when_not_active():
    recorder = MouseEventRecorder()
    recorder.pause()
    assert recorder.is_paused is False


def test_recorder_pause_sets_paused_when_active(monkeypatch):
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda volume_node=None: {})
    recorder = MouseEventRecorder()
    recorder.start()

    recorder.pause()

    assert recorder.is_paused is True
    assert recorder.is_active is True


def test_recorder_resume_clears_paused(monkeypatch):
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda volume_node=None: {})
    recorder = MouseEventRecorder()
    recorder.start()
    recorder.pause()

    recorder.resume()

    assert recorder.is_paused is False
    assert recorder.is_active is True


def test_recorder_is_not_paused_after_stop(monkeypatch):
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda volume_node=None: {})
    recorder = MouseEventRecorder()
    recorder.start()
    recorder.pause()

    recorder.stop()

    assert recorder.is_active is False
    assert recorder.is_paused is False


# ── _on_mouse guard ───────────────────────────────────────────────────────────

def test_on_mouse_drops_events_while_paused(monkeypatch):
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda volume_node=None: {})
    recorder = MouseEventRecorder()
    recorder.start()
    recorder._active_region_gate = None  # accept any XY
    n_before = len(recorder.records)

    recorder.pause()
    recorder._on_mouse('Red', (10, 20), PRESS)
    recorder._on_mouse('Red', (11, 21), MOVE)
    recorder._on_mouse('Red', (12, 22), RELEASE)

    assert len(recorder.records) == n_before


def test_on_mouse_tracks_press_state_while_paused(monkeypatch):
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda volume_node=None: {})
    recorder = MouseEventRecorder()
    recorder.start()
    recorder._active_region_gate = None

    recorder.pause()
    assert recorder._active_mouse_press is False

    recorder._on_mouse('Red', (10, 10), PRESS)
    assert recorder._active_mouse_press is True

    recorder._on_mouse('Red', (10, 10), RELEASE)
    assert recorder._active_mouse_press is False


def test_on_mouse_records_again_after_resume(monkeypatch):
    monkeypatch.setattr(recorder_mod, '_all_slice_visual_state', lambda volume_node=None: {})
    recorder = MouseEventRecorder()
    recorder.start()
    recorder._active_region_gate = None
    n_before = len(recorder.records)

    recorder.pause()
    recorder._on_mouse('Red', (10, 10), PRESS)  # dropped
    n_paused = len(recorder.records)

    recorder.resume()
    recorder._on_mouse('Red', (10, 10), PRESS)  # should record

    assert n_paused == n_before
    assert len(recorder.records) > n_before


# ── widget pause interval accumulation ───────────────────────────────────────

def test_do_resume_appends_pause_interval():
    now = datetime.datetime.now()
    widget = _pause_widget(
        recording_start=now - datetime.timedelta(seconds=30),
        pause_start=now - datetime.timedelta(seconds=20),
    )

    widget._do_resume_recording()

    assert len(widget._pause_intervals) == 1
    start_sec, end_sec = widget._pause_intervals[0]
    assert abs(start_sec - 10.0) < 0.5   # pause was 10s into recording
    assert end_sec > start_sec


def test_do_resume_clears_pause_start_time():
    now = datetime.datetime.now()
    widget = _pause_widget(
        recording_start=now - datetime.timedelta(seconds=30),
        pause_start=now - datetime.timedelta(seconds=20),
    )

    widget._do_resume_recording()

    assert widget._pause_start_time is None


def test_do_resume_calls_recorder_resume():
    now = datetime.datetime.now()
    widget = _pause_widget(
        recording_start=now - datetime.timedelta(seconds=30),
        pause_start=now - datetime.timedelta(seconds=20),
    )

    widget._do_resume_recording()

    assert widget._recorder.resumed == 1


def test_multiple_pause_resume_cycles_accumulate_all_intervals():
    now = datetime.datetime.now()
    rec_start = now - datetime.timedelta(seconds=60)

    widget = _pause_widget(recording_start=rec_start, pause_start=None)

    for pause_offset in (10, 25, 45):
        widget._pause_start_time = rec_start + datetime.timedelta(seconds=pause_offset)
        widget._do_resume_recording()

    assert len(widget._pause_intervals) == 3
    starts = [s for s, _ in widget._pause_intervals]
    assert abs(starts[0] - 10.0) < 0.5
    assert abs(starts[1] - 25.0) < 0.5
    assert abs(starts[2] - 45.0) < 0.5


def test_do_resume_skips_interval_when_recording_start_missing():
    widget = _pause_widget(recording_start=None, pause_start=_dt(5))

    widget._do_resume_recording()

    assert widget._pause_intervals == []


# ── _update_record_ui pause button state ─────────────────────────────────────

def test_pause_button_enabled_during_active_recording():
    widget = _pause_widget(recording_start=_dt(0), pause_start=None)
    # recorder is active and not paused

    widget._update_record_ui()

    assert widget.ui.pauseRecordButton.enabled is True


def test_pause_button_disabled_when_not_recording():
    widget = _pause_widget(recording_start=_dt(0), pause_start=None)
    widget._recorder.is_active = False

    widget._update_record_ui()

    assert widget.ui.pauseRecordButton.enabled is False


def test_pause_button_disabled_when_recorder_is_paused():
    widget = _pause_widget(recording_start=_dt(0), pause_start=_dt(5))
    widget._recorder.pause()

    widget._update_record_ui()

    assert widget.ui.pauseRecordButton.enabled is False


def test_status_label_shows_paused_while_paused():
    widget = _pause_widget(recording_start=_dt(0), pause_start=_dt(5))
    widget._recorder.pause()

    widget._update_record_ui()

    assert 'Paused' in widget.ui.recordStatusLabel.text


# ── _finalize_wav ─────────────────────────────────────────────────────────────

_SR = 100  # 100 samples/sec keeps frame arithmetic exact


def test_finalize_wav_unchanged_when_no_prewarm_no_pauses(tmp_path):
    wav = tmp_path / 'rec.wav'
    samples = list(range(1, _SR + 1))
    _make_wav(wav, _SR, samples)

    _finalize_widget(
        recording_start=_dt(0),
        audio_start=_dt(0),
    )._finalize_wav(str(wav))

    assert _read_wav_samples(wav) == samples


def test_finalize_wav_trims_prewarm_lead_in(tmp_path):
    wav = tmp_path / 'rec.wav'
    # 110 frames: first 10 are prewarm (0.1 s before recording start)
    samples = list(range(110))
    _make_wav(wav, _SR, samples)

    _finalize_widget(
        recording_start=_dt(0.1),
        audio_start=_dt(0.0),
    )._finalize_wav(str(wav))

    assert _read_wav_samples(wav) == list(range(10, 110))


def test_finalize_wav_silences_single_pause_interval(tmp_path):
    wav = tmp_path / 'rec.wav'
    samples = [1000] * _SR
    _make_wav(wav, _SR, samples)

    _finalize_widget(
        recording_start=_dt(0),
        audio_start=_dt(0),
        pause_intervals=[(0.2, 0.5)],
    )._finalize_wav(str(wav))

    result = _read_wav_samples(wav)
    assert all(v == 1000 for v in result[:20]),  'pre-pause audio should be intact'
    assert all(v == 0    for v in result[20:50]), 'pause interval should be silenced'
    assert all(v == 1000 for v in result[50:]),  'post-pause audio should be intact'


def test_finalize_wav_silences_multiple_pause_intervals(tmp_path):
    wav = tmp_path / 'rec.wav'
    samples = [999] * _SR
    _make_wav(wav, _SR, samples)

    _finalize_widget(
        recording_start=_dt(0),
        audio_start=_dt(0),
        pause_intervals=[(0.1, 0.2), (0.6, 0.8)],
    )._finalize_wav(str(wav))

    result = _read_wav_samples(wav)
    assert all(v == 999 for v in result[:10])
    assert all(v == 0   for v in result[10:20])
    assert all(v == 999 for v in result[20:60])
    assert all(v == 0   for v in result[60:80])
    assert all(v == 999 for v in result[80:])


def test_finalize_wav_trim_and_silence_combined(tmp_path):
    wav = tmp_path / 'rec.wav'
    # 110 frames; first 10 are prewarm, rest are recording
    samples = [500] * 110
    _make_wav(wav, _SR, samples)

    _finalize_widget(
        recording_start=_dt(0.1),          # 10-frame prewarm
        audio_start=_dt(0.0),
        pause_intervals=[(0.2, 0.4)],      # silence frames 20–40 of trimmed
    )._finalize_wav(str(wav))

    result = _read_wav_samples(wav)
    assert len(result) == 100
    assert all(v == 500 for v in result[:20])
    assert all(v == 0   for v in result[20:40])
    assert all(v == 500 for v in result[40:])


def test_finalize_wav_no_trim_without_audio_recorder(tmp_path):
    wav = tmp_path / 'rec.wav'
    samples = list(range(_SR))
    _make_wav(wav, _SR, samples)

    _finalize_widget(
        recording_start=_dt(0.5),
        audio_start=None,               # no audio recorder → no trim
    )._finalize_wav(str(wav))

    assert _read_wav_samples(wav) == list(range(_SR))


def test_finalize_wav_no_trim_when_audio_started_after_recording(tmp_path):
    """Negative trim_sec is clamped to zero — no audio discarded."""
    wav = tmp_path / 'rec.wav'
    samples = list(range(50))
    _make_wav(wav, _SR, samples)

    _finalize_widget(
        recording_start=_dt(0.0),
        audio_start=_dt(0.1),          # audio started after recording (edge case)
    )._finalize_wav(str(wav))

    assert _read_wav_samples(wav) == list(range(50))
