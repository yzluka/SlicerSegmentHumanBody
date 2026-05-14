"""Standalone audio recording process.

Launched as a subprocess by SegmentHumanBody. Records microphone audio to a
single WAV file. Stops when a sentinel file appears at --stop-file path.
Writes a result JSON to --result-file on clean exit.
"""
import argparse
import datetime as _dt
import json
import queue
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wav_path', help='Output WAV file path')
    ap.add_argument('--sample-rate', type=int, default=22050)
    ap.add_argument('--channels', type=int, default=1)
    ap.add_argument('--device', type=int, default=None)
    ap.add_argument('--stop-file', required=True)
    ap.add_argument('--ready-file', required=True)
    ap.add_argument('--result-file', required=True)
    args = ap.parse_args()

    wav_path = Path(args.wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path = Path(args.stop_file)
    ready_path = Path(args.ready_file)
    result_path = Path(args.result_file)

    audio_q: queue.Queue = queue.Queue()
    start_time = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='milliseconds')

    def _callback(indata, frames, time_info, status):
        audio_q.put(np.clip(indata, -1.0, 1.0).copy())

    def _drain(wav_file):
        total = 0
        while not audio_q.empty():
            chunk = audio_q.get_nowait()
            pcm = (chunk * 32767.0).astype('<i2').tobytes()
            wav_file.writeframes(pcm)
            total += len(chunk)
        return total

    frames_written = 0
    stream_kw = dict(
        samplerate=args.sample_rate,
        channels=args.channels,
        dtype='float32',
        callback=_callback,
    )
    if args.device is not None:
        stream_kw['device'] = args.device

    with wave.open(str(wav_path), 'wb') as wav_file:
        wav_file.setnchannels(args.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(args.sample_rate)

        with sd.InputStream(**stream_kw):
            ready_path.touch()  # signal to parent that stream is open and capturing
            while not stop_path.exists():
                time.sleep(0.05)
                frames_written += _drain(wav_file)
            # drain buffered audio after stop signal
            time.sleep(0.15)
            frames_written += _drain(wav_file)

    end_time = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='milliseconds')
    result_path.write_text(json.dumps({
        'ok': True,
        'wav_path': str(wav_path),
        'start_time': start_time,
        'end_time': end_time,
        'sample_rate_hz': args.sample_rate,
        'channels': args.channels,
        'frames': frames_written,
    }), encoding='utf-8')


if __name__ == '__main__':
    import sys
    try:
        main()
    except Exception as exc:
        print(f'audio subprocess error: {exc}', file=sys.stderr)
        sys.exit(1)
