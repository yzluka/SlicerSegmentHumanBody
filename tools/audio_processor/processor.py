"""Core transcription and annotation-alignment logic.

Standalone — no Slicer dependency.
Inputs:
  - annotation JSON (annotation_summary or annotation_process type)
  - WAV file
Outputs:
  - list of transcript segments, each linked to overlapping annotation spans
"""
from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_annotation_json(json_path: str) -> dict:
    with open(json_path, encoding='utf-8') as f:
        return json.load(f)


def get_spans(data: dict) -> list[dict]:
    """Return span-like dicts with start_time/end_time from annotation JSON.

    Handles both annotation_summary (has ``spans``) and annotation_process
    (has ``events`` — converts them to minimal span dicts).
    """
    dtype = data.get('type', '')
    if dtype == 'SegmentHumanBody.annotation_summary':
        return [s for s in (data.get('spans') or []) if isinstance(s, dict)]

    if dtype == 'SegmentHumanBody.annotation_process':
        events = [e for e in (data.get('events') or []) if isinstance(e, dict)]
        return [
            {
                'type': e.get('event', 'event'),
                'start_time': _first_timestamp(e),
                'end_time': _first_timestamp(e),
                'text': e.get('event', ''),
                'tool': e.get('tool'),
                'segment': e.get('segment'),
            }
            for e in events
        ]

    return []


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(
    wav_path: str,
    model_size: str = 'base',
    device: str = 'auto',
    language: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Transcribe *wav_path* with faster-whisper.

    Returns ``(segments, info)`` where each segment is
    ``{start, end, text, words?}``.
    """
    from faster_whisper import WhisperModel  # type: ignore[import]

    # Resolve device/compute_type
    if device == 'auto':
        try:
            import torch  # type: ignore[import]
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'

    compute_type = 'float16' if device == 'cuda' else 'int8'

    if progress_cb:
        progress_cb(f'Loading model "{model_size}" on {device} ({compute_type})…')

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    if progress_cb:
        progress_cb('Transcribing…')

    lang_arg = language if language and language != 'auto' else None
    raw_segments, info = model.transcribe(wav_path, language=lang_arg, beam_size=5)

    segments: list[dict] = []
    for seg in raw_segments:
        text = seg.text.strip()
        if not text:
            continue
        entry: dict = {'start': round(seg.start, 3), 'end': round(seg.end, 3), 'text': text}
        segments.append(entry)
        if progress_cb:
            progress_cb(f'  [{seg.start:6.1f}s]  {text}')

    info_dict = {
        'language': info.language,
        'language_probability': round(info.language_probability, 3),
        'duration': round(info.duration, 3),
    }
    if progress_cb:
        progress_cb(
            f'Done — language: {info.language} '
            f'(p={info.language_probability:.2f}), '
            f'duration: {info.duration:.1f}s'
        )
    return segments, info_dict


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _overlaps(span: dict, abs_start: datetime.datetime, abs_end: datetime.datetime) -> bool:
    s = _parse_iso(span.get('start_time'))
    e = _parse_iso(span.get('end_time')) or s
    if s is None:
        return False
    return s <= abs_end and e >= abs_start


def _span_summary(span: dict) -> dict:
    out: dict = {}
    for key in ('type', 'text', 'tool', 'segment', 'view', 'start_time', 'end_time'):
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


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def process(
    json_path: str,
    wav_path: str,
    model_size: str = 'base',
    device: str = 'auto',
    language: str | None = None,
    audio_offset: float = 0.0,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Full pipeline: load → transcribe → align → return result dict."""
    _log = progress_cb or (lambda _: None)

    _log('Loading annotation data…')
    data = load_annotation_json(json_path)
    meta = data.get('metadata') or {}

    # Determine audio start time (prefer filename timestamp, fallback to JSON metadata)
    audio_start = parse_audio_start_from_wav_name(wav_path)
    if audio_start is None:
        audio_start = _parse_iso(meta.get('start_time'))
    if audio_start is None:
        audio_start = datetime.datetime.now()
        _log('Warning: could not determine audio start time — timestamps will be approximate.')

    spans = get_spans(data)
    _log(f'Loaded {len(spans)} annotation spans.')

    segments, info = transcribe(wav_path, model_size, device, language, _log)

    _log(f'Aligning {len(segments)} transcript segments to spans…')
    aligned = align(segments, spans, audio_start, audio_offset)

    return {
        'type': 'SegmentHumanBody.audio_transcript',
        'metadata': meta,
        'transcription_info': info,
        'audio_start_time': audio_start.isoformat(),
        'audio_offset_seconds': audio_offset,
        'source_json': str(json_path),
        'source_wav': str(wav_path),
        'segments': aligned,
    }


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def format_text_report(result: dict) -> str:
    lines: list[str] = []
    meta = result.get('metadata') or {}

    lines.append('Audio Transcript + Annotation Alignment Report')
    lines.append('=' * 50)
    if meta.get('start_time'):
        lines.append(f"Recording start : {meta['start_time']}")
    info = result.get('transcription_info') or {}
    if info.get('language'):
        lines.append(f"Language        : {info['language']} (confidence {info.get('language_probability', '?')})")
    if info.get('duration'):
        lines.append(f"Audio duration  : {info['duration']:.1f}s")
    lines.append('')

    for seg in result.get('segments', []):
        lines.append(f"[{seg['start']:6.1f}s – {seg['end']:6.1f}s]  {seg['text']}")
        for sp in seg.get('linked_spans', []):
            parts = [sp.get('type', ''), sp.get('tool', ''), sp.get('segment', '')]
            detail = '  |  '.join(p for p in parts if p)
            note   = sp.get('text', '')
            lines.append(f"    → {detail}")
            if note and note != detail:
                # wrap long notes
                lines.append(f"      {note[:120]}")
        if not seg.get('linked_spans'):
            lines.append('    → (no annotation activity in this window)')
        lines.append('')

    return '\n'.join(lines)
