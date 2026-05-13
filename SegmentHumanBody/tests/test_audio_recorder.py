import datetime as dt
import json
import wave

import numpy as np

from core._audio_recorder import StandaloneAudioRecorder, pcm16_from_array


def _clock():
    current = dt.datetime(2026, 5, 12, 12, 0, 0, tzinfo=dt.timezone.utc)

    def now():
        return current

    return now


def test_pcm16_from_float_array_clips_and_scales():
    samples = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float32)

    pcm = np.frombuffer(pcm16_from_array(samples), dtype='<i2')

    assert pcm.tolist() == [-32767, -32767, 0, 32767, 32767]


def test_ingest_frames_writes_timestamped_wav_chunk(tmp_path):
    recorder = StandaloneAudioRecorder(
        sample_rate_hz=4,
        channels=1,
        chunk_seconds=1.0,
        clock=_clock(),
    )
    recorder.prepare_output(tmp_path, prefix='case 01 audio')
    frames = np.array([[0.0], [0.25], [0.5], [0.75]], dtype=np.float32)

    recorder.ingest_frames(frames)
    chunks = recorder.stop()

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.id == 1
    assert chunk.frame_count == 4
    assert chunk.sample_rate_hz == 4
    assert chunk.channels == 1
    assert chunk.path.endswith('.wav')
    assert 'case_01_audio_0001_' in chunk.path

    with wave.open(chunk.path, 'rb') as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 4
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 4


def test_ingest_frames_splits_chunks_by_duration(tmp_path):
    recorder = StandaloneAudioRecorder(
        sample_rate_hz=4,
        channels=1,
        chunk_seconds=0.5,
        clock=_clock(),
    )
    recorder.prepare_output(tmp_path, prefix='audio')

    recorder.ingest_frames(np.zeros((5, 1), dtype=np.float32))
    chunks = recorder.stop()

    assert [chunk.frame_count for chunk in chunks] == [2, 2, 1]
    assert [round(chunk.duration_seconds, 3) for chunk in chunks] == [0.5, 0.5, 0.25]


def test_manifest_records_whisper_friendly_format(tmp_path):
    recorder = StandaloneAudioRecorder(sample_rate_hz=16000, channels=1, clock=_clock())
    recorder.prepare_output(tmp_path, prefix='audio')
    recorder.ingest_frames(np.zeros((160, 1), dtype=np.float32))
    recorder.stop()

    manifest_path = tmp_path / 'manifest.json'
    recorder.save_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    assert manifest['type'] == 'audio_recording'
    assert manifest['format'] == 'wav_pcm16'
    assert manifest['sample_rate_hz'] == 16000
    assert manifest['channels'] == 1
    assert manifest['chunks'][0]['duration_seconds'] == 0.01
