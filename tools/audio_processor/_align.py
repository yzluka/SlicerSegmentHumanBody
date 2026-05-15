"""Phrase-to-annotation-span alignment."""
from __future__ import annotations

import datetime

from _times import _parse_iso  # type: ignore[import]


def _overlaps(
    span: dict,
    abs_start: datetime.datetime,
    abs_end: datetime.datetime,
) -> bool:
    s = _parse_iso(span.get('start_time'))
    e = _parse_iso(span.get('end_time')) or s
    if s is None:
        return False
    return s <= abs_end and e >= abs_start


def _span_summary(span: dict) -> dict:
    out: dict = {}
    for key in ('type', 'text', 'tool', 'segment', 'view',
                'start_time', 'end_time', 'trajectory'):
        if span.get(key) is not None:
            out[key] = span[key]
    return out


def align(
    segments: list[dict],
    spans: list[dict],
    audio_start: datetime.datetime,
    audio_offset: float = 0.0,
) -> list[dict]:
    """Add ``abs_start``, ``abs_end``, and ``linked_spans`` to each segment."""
    result = []
    for seg in segments:
        abs_start = audio_start + datetime.timedelta(seconds=seg['start'] - audio_offset)
        abs_end   = audio_start + datetime.timedelta(seconds=seg['end']   - audio_offset)
        linked = [
            {'index': i, **_span_summary(sp)}
            for i, sp in enumerate(spans)
            if _overlaps(sp, abs_start, abs_end)
        ]
        result.append({
            **seg,
            'abs_start': abs_start.isoformat(),
            'abs_end':   abs_end.isoformat(),
            'linked_spans': linked,
        })
    return result
