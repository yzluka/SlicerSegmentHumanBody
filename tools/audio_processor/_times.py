"""Timestamp parsing and formatting utilities."""
from __future__ import annotations

import datetime
import re
from pathlib import Path


def _parse_iso(s: str | None) -> datetime.datetime | None:
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _first_timestamp(ev: dict) -> str | None:
    ts = ev.get('timestamp') or ev.get('start_time')
    if isinstance(ts, list):
        return ts[0] if ts else None
    return ts


def parse_audio_start_from_wav_name(wav_path: str) -> datetime.datetime | None:
    """Parse recording-start time from WAV filenames like ``name_20240101T120000000.wav``."""
    stem = Path(wav_path).stem
    m = re.search(r'_(\d{8}T\d{9})$', stem)
    if not m:
        return None
    try:
        # format: YYYYMMDDTHHMMSSmmm  (9 digits after T = HHMMSS + 3ms)
        return datetime.datetime.strptime(m.group(1), '%Y%m%dT%H%M%S%f')
    except ValueError:
        return None


def _hms(iso: str) -> str:
    """Return HH:MM:SS from an ISO datetime string."""
    try:
        return datetime.datetime.fromisoformat(iso).strftime('%H:%M:%S')
    except (ValueError, TypeError):
        return '??:??:??'


def _time_range_hms(start_iso: str, end_iso: str) -> str:
    a, b = _hms(start_iso), _hms(end_iso)
    return a if a == b else f'{a}–{b}'
