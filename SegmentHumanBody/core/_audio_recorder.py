"""Standalone timestamped microphone recorder for Whisper-style audio input.

This module is intentionally independent from Slicer and the mouse/process
recorder. It uses ``sounddevice`` only when recording starts, so importing this
file does not require microphone/audio dependencies.
"""

from __future__ import annotations

import datetime as _dt
import json
import queue
import threading
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    """Metadata for one timestamped WAV chunk."""

    id: int
    path: str
    start_time: str
    end_time: str
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate_hz <= 0:
            return 0.0
        return self.frame_count / float(self.sample_rate_hz)


class StandaloneAudioRecorder:
    """Record microphone input to timestamped WAV chunks.

    The default output format is 16 kHz, mono, 16-bit PCM WAV, which is a
    practical format for local Whisper pipelines. The recorder does not run
    Whisper; it only produces audio files and timing metadata for a downstream
    transcriber.
    """

    def __init__(
        self,
        sample_rate_hz: int = 16000,
        channels: int = 1,
        chunk_seconds: float = 30.0,
        device=None,
        clock: Callable[[], _dt.datetime] | None = None,
    ):
        if sample_rate_hz <= 0:
            raise ValueError('sample_rate_hz must be positive')
        if channels <= 0:
            raise ValueError('channels must be positive')
        if chunk_seconds <= 0:
            raise ValueError('chunk_seconds must be positive')
        self.sample_rate_hz = int(sample_rate_hz)
        self.channels = int(channels)
        self.chunk_seconds = float(chunk_seconds)
        self.device = device
        self._clock = clock or (lambda: _dt.datetime.now(_dt.timezone.utc))
        self._stream = None
        self._active = False
        self._queue: queue.Queue[tuple[_dt.datetime, bytes, int]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._output_dir: Path | None = None
        self._prefix = 'audio'
        self._chunk_id = 0
        self._chunks: list[AudioChunk] = []
        self._pending = bytearray()
        self._pending_start: _dt.datetime | None = None
        self._pending_frames = 0

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def chunks(self) -> tuple[AudioChunk, ...]:
        return tuple(self._chunks)

    def start(self, output_dir, prefix: str = 'audio') -> None:
        """Start microphone capture.

        Raises ImportError if ``sounddevice`` is unavailable in the current
        Python environment.
        """
        if self._active:
            raise RuntimeError('audio recorder is already active')
        import sounddevice as sd

        self.prepare_output(output_dir, prefix)
        self._active = True
        self._worker = threading.Thread(target=self._drain_queue, daemon=True)
        self._worker.start()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate_hz,
            channels=self.channels,
            dtype='float32',
            device=self.device,
            callback=self._on_audio,
        )
        self._stream.start()

    def prepare_output(self, output_dir, prefix: str = 'audio') -> None:
        """Configure chunk output without starting microphone capture."""
        if self._active:
            raise RuntimeError('cannot change output while recording')
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._prefix = _safe_prefix(prefix)

    def stop(self) -> list[AudioChunk]:
        """Stop capture, flush the last partial chunk, and return chunk metadata."""
        if not self._active:
            self._flush_pending(self._clock())
            return list(self._chunks)
        self._active = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._queue.put((self._clock(), b'', 0))
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None
        self._flush_pending(self._clock())
        return list(self._chunks)

    def save_manifest(self, path) -> None:
        """Write a JSON manifest for recorded chunks."""
        manifest = {
            'type': 'audio_recording',
            'format': 'wav_pcm16',
            'sample_rate_hz': self.sample_rate_hz,
            'channels': self.channels,
            'chunks': [asdict(chunk) | {'duration_seconds': chunk.duration_seconds}
                       for chunk in self._chunks],
        }
        Path(path).write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def ingest_frames(self, frames, timestamp: _dt.datetime | None = None) -> None:
        """Process already-captured frames.

        This is useful for tests and for callers that own microphone capture.
        ``frames`` may be PCM16 bytes or an array-like object convertible to
        float/int samples.
        """
        ts = timestamp or self._clock()
        pcm, frame_count = self._frames_to_pcm16(frames)
        if frame_count <= 0:
            return
        self._append_pcm(ts, pcm, frame_count)

    def _on_audio(self, indata, frames, time_info, status) -> None:
        pcm, frame_count = self._frames_to_pcm16(indata)
        self._queue.put((self._clock(), pcm, frame_count))

    def _drain_queue(self) -> None:
        while self._active or not self._queue.empty():
            try:
                ts, pcm, frame_count = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame_count > 0:
                self._append_pcm(ts, pcm, frame_count)
        self._flush_pending(self._clock())

    def _append_pcm(self, timestamp: _dt.datetime, pcm: bytes, frame_count: int) -> None:
        if self._pending_start is None:
            self._pending_start = timestamp
        self._pending.extend(pcm)
        self._pending_frames += frame_count
        max_frames = int(round(self.sample_rate_hz * self.chunk_seconds))
        while self._pending_frames >= max_frames:
            bytes_per_frame = self.channels * 2
            split_bytes = max_frames * bytes_per_frame
            chunk_pcm = bytes(self._pending[:split_bytes])
            del self._pending[:split_bytes]
            start = self._pending_start
            end = start + _dt.timedelta(seconds=max_frames / self.sample_rate_hz)
            self._write_chunk(chunk_pcm, max_frames, start, end)
            self._pending_frames -= max_frames
            self._pending_start = end if self._pending_frames else None

    def _flush_pending(self, timestamp: _dt.datetime) -> None:
        if not self._pending or self._pending_start is None:
            return
        frame_count = self._pending_frames
        end = self._pending_start + _dt.timedelta(
            seconds=frame_count / self.sample_rate_hz)
        self._write_chunk(bytes(self._pending), frame_count, self._pending_start, end)
        self._pending.clear()
        self._pending_frames = 0
        self._pending_start = None

    def _write_chunk(
        self,
        pcm: bytes,
        frame_count: int,
        start: _dt.datetime,
        end: _dt.datetime,
    ) -> None:
        if self._output_dir is None:
            raise RuntimeError('output directory is not configured')
        self._chunk_id += 1
        name = f'{self._prefix}_{self._chunk_id:04d}_{_stamp(start)}.wav'
        path = self._output_dir / name
        write_wav_pcm16(path, pcm, self.sample_rate_hz, self.channels)
        self._chunks.append(AudioChunk(
            id=self._chunk_id,
            path=str(path),
            start_time=start.isoformat(timespec='milliseconds'),
            end_time=end.isoformat(timespec='milliseconds'),
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            sample_width_bytes=2,
            frame_count=frame_count,
        ))

    def _frames_to_pcm16(self, frames) -> tuple[bytes, int]:
        if isinstance(frames, bytes):
            bytes_per_frame = self.channels * 2
            if len(frames) % bytes_per_frame != 0:
                raise ValueError('PCM byte length is not aligned to frame size')
            return frames, len(frames) // bytes_per_frame
        arr = np.asarray(frames)
        if arr.ndim == 1:
            arr = arr.reshape((-1, self.channels))
        if arr.ndim != 2 or arr.shape[1] != self.channels:
            raise ValueError(
                f'frames must have shape (n, {self.channels}) or flat mono data')
        return pcm16_from_array(arr), int(arr.shape[0])


def pcm16_from_array(frames) -> bytes:
    """Convert float [-1, 1] or integer audio samples to little-endian PCM16."""
    arr = np.asarray(frames)
    if np.issubdtype(arr.dtype, np.floating):
        clipped = np.clip(arr, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype('<i2')
    elif arr.dtype == np.int16:
        pcm = arr.astype('<i2', copy=False)
    else:
        pcm = np.clip(arr, -32768, 32767).astype('<i2')
    return pcm.tobytes()


def write_wav_pcm16(path, pcm: bytes, sample_rate_hz: int, channels: int) -> None:
    """Write PCM16 bytes to a WAV container."""
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm)


def _safe_prefix(prefix: str) -> str:
    result = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in prefix)
    return result.strip('_') or 'audio'


def _stamp(ts: _dt.datetime) -> str:
    return ts.strftime('%Y%m%dT%H%M%S%f')[:-3]
