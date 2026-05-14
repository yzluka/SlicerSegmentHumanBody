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

from cleaner import clean_phrases  # type: ignore[import]

# TimeLogSummarizer lives in SegmentHumanBody/core/ — pure Python, no Slicer dep.
# Import it so annotation_process JSON gets the same rich spans as _summary.txt.
import sys as _sys
_core = Path(__file__).parent.parent.parent / 'SegmentHumanBody' / 'core'
if _core.exists() and str(_core) not in _sys.path:
    _sys.path.insert(0, str(_core))
try:
    from TimeLogInterpreter import TimeLogInterpreter as _Interpreter  # type: ignore[import]
    from TimeLogSummarizer import TimeLogSummarizer as _Summarizer  # type: ignore[import]
    _HAS_SUMMARIZER = True
except ImportError:
    _HAS_SUMMARIZER = False


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
    """Return rich spans from annotation JSON.

    annotation_summary  → spans used directly (already have formatted text).
    annotation_process  → run TimeLogSummarizer inline to produce the same
                          rich spans as _summary.txt (coordinates, volume,
                          tool, segment). No file is written.
    """
    dtype = data.get('type', '')
    if dtype == 'SegmentHumanBody.annotation_summary':
        return [s for s in (data.get('spans') or []) if isinstance(s, dict)]

    if dtype == 'SegmentHumanBody.annotation_process':
        if _HAS_SUMMARIZER:
            export = _Summarizer(data).export()
            return [s for s in (export.get('spans') or []) if isinstance(s, dict)]

    if dtype == 'SegmentHumanBody.annotation_raw':
        if _HAS_SUMMARIZER:
            process = _Interpreter(data).export()
            export = _Summarizer(process).export()
            return [s for s in (export.get('spans') or []) if isinstance(s, dict)]

    return []


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(
    wav_path: str,
    model_size: str = 'base',
    device: str = 'auto',
    language: str | None = None,
    initial_prompt: str | None = None,
    temperature: float = 0.0,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Transcribe *wav_path* with faster-whisper at word-level granularity.

    Returns ``(words, info)`` where each word is
    ``{start, end, word, probability}``.
    """
    from faster_whisper import WhisperModel  # type: ignore[import]

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
    raw_segments, info = model.transcribe(
        wav_path,
        language=lang_arg,
        initial_prompt=initial_prompt or None,
        temperature=temperature,
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    words: list[dict] = []
    for seg in raw_segments:
        for w in (seg.words or []):
            word_text = w.word.strip()
            if not word_text:
                continue
            words.append({
                'start': round(w.start, 3),
                'end':   round(w.end,   3),
                'word':  word_text,
                'probability': round(w.probability, 3),
            })
            if progress_cb:
                progress_cb(f'  [{w.start:6.1f}s]  {word_text}')

    info_dict = {
        'language': info.language,
        'language_probability': round(info.language_probability, 3),
        'duration': round(info.duration, 3),
    }
    if progress_cb:
        progress_cb(
            f'Done — {len(words)} words, language: {info.language} '
            f'(p={info.language_probability:.2f}), '
            f'duration: {info.duration:.1f}s'
        )
    return words, info_dict


# ---------------------------------------------------------------------------
# Word → phrase merging
# ---------------------------------------------------------------------------

def merge_words_to_phrases(
    words: list[dict],
    spans: list[dict],
    audio_start: datetime.datetime,
    audio_offset: float = 0.0,
    silence_gap: float = 0.35,
) -> list[dict]:
    """Group words into phrases using silence gaps and annotation boundaries.

    A split is inserted between word[i] and word[i+1] when:
    - the silence between them is >= silence_gap seconds, OR
    - any annotation span starts or ends within that inter-word gap.

    Returns segments with ``{start, end, text, words}``.
    """
    if not words:
        return []

    # Collect all annotation event boundary times as audio-relative seconds
    event_times: list[float] = []
    for span in spans:
        for key in ('start_time', 'end_time'):
            t = _parse_iso(span.get(key))
            if t is not None:
                rel = (t - audio_start).total_seconds() + audio_offset
                event_times.append(rel)

    phrases: list[list[dict]] = []
    current: list[dict] = [words[0]]

    for i in range(1, len(words)):
        gap_start = words[i - 1]['end']
        gap_end   = words[i]['start']

        split = (gap_end - gap_start) >= silence_gap or any(
            gap_start < et < gap_end for et in event_times
        )

        if split:
            phrases.append(current)
            current = [words[i]]
        else:
            current.append(words[i])

    if current:
        phrases.append(current)

    return [
        {
            'start': round(g[0]['start'], 3),
            'end':   round(g[-1]['end'],  3),
            'text':  ' '.join(w['word'] for w in g),
            'words': g,
        }
        for g in phrases
    ]


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
    for key in ('type', 'text', 'tool', 'segment', 'view', 'start_time', 'end_time', 'trajectory'):
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
    initial_prompt: str | None = None,
    temperature: float = 0.0,
    audio_offset: float = 0.0,
    silence_gap: float = 0.35,
    corrections_dir: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Full pipeline: load → transcribe → save whisper JSON → merge → clean → align."""
    _log = progress_cb or (lambda _: None)

    _log('Loading annotation data…')
    data = load_annotation_json(json_path)
    meta = data.get('metadata') or {}

    audio_start = parse_audio_start_from_wav_name(wav_path)
    if audio_start is None:
        audio_start = _parse_iso(meta.get('start_time'))
    if audio_start is None:
        audio_start = datetime.datetime.now()
        _log('Warning: could not determine audio start time — timestamps will be approximate.')

    spans = get_spans(data)
    _log(f'Loaded {len(spans)} annotation spans.')

    words, info = transcribe(wav_path, model_size, device, language,
                             initial_prompt, temperature, _log)

    # Save raw whisper word output as an intermediate file so phrase merging
    # can be re-run with different settings without re-running the model.
    wav_stem = Path(wav_path).stem
    whisper_json_path = Path(json_path).parent / f'whisper_{wav_stem}.json'
    whisper_data = {
        'type': 'SegmentHumanBody.whisper_words',
        'transcription_info': info,
        'audio_start_time': audio_start.isoformat(),
        'source_wav': str(wav_path),
        'words': words,
    }
    with open(whisper_json_path, 'w', encoding='utf-8') as _f:
        json.dump(whisper_data, _f, indent=2, ensure_ascii=False)
    _log(f'Saved whisper words → {whisper_json_path}')

    _log(f'Merging {len(words)} words into phrases (silence_gap={silence_gap}s)…')
    segments = merge_words_to_phrases(words, spans, audio_start, audio_offset, silence_gap)
    segments = clean_phrases(segments, corrections_dir)
    flagged = sum(1 for s in segments if s.get('needs_review'))
    _log(f'{len(segments)} phrases formed; {flagged} flagged for review.')

    _log(f'Aligning phrases to spans…')
    aligned = align(segments, spans, audio_start, audio_offset)

    return {
        'type': 'SegmentHumanBody.audio_transcript',
        'metadata': meta,
        'transcription_info': info,
        'audio_start_time': audio_start.isoformat(),
        'audio_offset_seconds': audio_offset,
        'silence_gap_seconds': silence_gap,
        'source_json': str(json_path),
        'source_wav': str(wav_path),
        'segments': aligned,
        'spans': spans,
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

    # Which span indices are linked to at least one audio phrase
    linked_indices: set[int] = set()
    for seg in result.get('segments', []):
        for sp in seg.get('linked_spans', []):
            if sp.get('index') is not None:
                linked_indices.add(sp['index'])

    # Orphaned spans: annotation events with no overlapping audio
    orphaned: list[tuple[int, dict]] = [
        (i, span)
        for i, span in enumerate(result.get('spans', []))
        if i not in linked_indices
    ]

    # Merge audio segments and orphaned spans into one time-ordered stream
    items: list[tuple[str, object]] = (
        [('audio', seg) for seg in result.get('segments', [])] +
        [('span', (i, span)) for i, span in orphaned]
    )

    def _sort_key(item: tuple) -> str:
        if item[0] == 'audio':
            return item[1].get('abs_start', '')  # type: ignore[union-attr]
        _, (_, span) = item
        return span.get('start_time') or ''

    items.sort(key=_sort_key)

    def _render_span_detail(sp: dict) -> None:
        parts = [sp.get('type', ''), sp.get('tool', ''), sp.get('segment', '')]
        detail = '  |  '.join(p for p in parts if p)
        note = sp.get('text', '')
        lines.append(f"    → {detail}")
        if note and note != detail:
            lines.append(f"      {note}")
        traj = sp.get('trajectory')
        if traj:
            pts = ', '.join(
                '(' + ','.join(str(int(round(float(v)))) for v in pt[:3]) + ')'
                for pt in traj
            )
            lines.append(f"      trajectory=[{pts}]")

    current_date: str | None = None

    for kind, payload in items:
        if kind == 'audio':
            seg = payload  # type: ignore[assignment]
            item_date = (seg.get('abs_start') or '')[:10]
        else:
            _, span = payload  # type: ignore[misc]
            item_date = (span.get('start_time') or '')[:10]

        if item_date and item_date != current_date:
            if current_date is not None:
                lines.append(f'--- {item_date} ---')
                lines.append('')
            current_date = item_date

        if kind == 'audio':
            seg = payload  # type: ignore[assignment]
            display = seg.get('cleaned_text') or seg['text']
            flag = ' [!]' if seg.get('needs_review') else ''
            t = _time_range_hms(seg.get('abs_start', ''), seg.get('abs_end', ''))
            lines.append(f"[{t}]  {display}{flag}")
            for sp in seg.get('linked_spans', []):
                _render_span_detail(sp)
            if not seg.get('linked_spans'):
                lines.append('    → (no annotation activity in this window)')
        else:
            _, span = payload  # type: ignore[misc]
            t = _hms(span.get('start_time') or '')
            lines.append(f"[{t}]  (no audio)")
            _render_span_detail(span)
        lines.append('')

    return '\n'.join(lines)


def _hms(iso: str) -> str:
    """Return HH:MM:SS from an ISO datetime string."""
    try:
        return datetime.datetime.fromisoformat(iso).strftime('%H:%M:%S')
    except (ValueError, TypeError):
        return '??:??:??'


def _time_range_hms(start_iso: str, end_iso: str) -> str:
    a, b = _hms(start_iso), _hms(end_iso)
    return a if a == b else f'{a}–{b}'


def _fmt_span_block(span: dict) -> str:
    """Render a span identically to how _summary.txt renders it."""
    if _HAS_SUMMARIZER:
        try:
            from TimeLogSummarizer import _span_text  # type: ignore[import]
            return _span_text(span)
        except Exception:
            pass
    # Fallback: mirror _span_text logic manually
    text = span.get('text', '').strip()
    trajectory = span.get('trajectory')
    if trajectory:
        pts = ', '.join(
            '(' + ','.join(str(int(round(float(v)))) for v in pt[:3]) + ')'
            for pt in trajectory
        )
        text += f'\n  trajectory=[{pts}]'
    return text


def format_caption_report(result: dict) -> str:
    """_summary.txt layout enriched with timestamped audio captions.

    Span blocks are identical to _summary.txt; overlapping audio phrases are
    appended as  [HH:MM:SS–HH:MM:SS] spoken text  lines.
    Audio phrases that don't overlap any annotation span appear as standalone
    audio-only blocks, so the file is the complete transpose of _transcript.txt.
    """
    # span index → list of (abs_start, abs_end, display_text)
    span_audio: dict[int, list[tuple[str, str, str]]] = {}
    # audio segment indices that are linked to at least one span
    linked_seg_indices: set[int] = set()
    for si, seg in enumerate(result.get('segments', [])):
        for linked in seg.get('linked_spans', []):
            idx = linked.get('index')
            if idx is not None:
                span_audio.setdefault(idx, []).append(
                    (seg.get('abs_start', ''), seg.get('abs_end', ''),
                     seg.get('cleaned_text') or seg['text'])
                )
                linked_seg_indices.add(si)

    meta = result.get('metadata') or {}
    info = result.get('transcription_info') or {}

    # Header matches _summary.txt exactly (plus language line)
    header_lines: list[str] = []
    if meta.get('start_time'):
        try:
            dt = datetime.datetime.fromisoformat(meta['start_time'])
            header_lines.append(f"Recording: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except (ValueError, TypeError):
            header_lines.append(f"Recording: {meta['start_time']}")
    if info.get('language'):
        header_lines.append(
            f"Language: {info['language']} (p={info.get('language_probability', '?')})"
        )

    current_date: str | None = (
        header_lines[0][10:20] if header_lines else None
    )

    # Build time-ordered items: annotation spans + orphaned audio phrases
    # Each item: ('span', i, span_dict) or ('audio', seg_dict)
    items: list[tuple] = []
    for i, span in enumerate(result.get('spans', [])):
        items.append(('span', i, span))
    for si, seg in enumerate(result.get('segments', [])):
        if si not in linked_seg_indices:
            items.append(('audio', seg))

    def _item_time(item: tuple) -> str:
        if item[0] == 'span':
            return item[2].get('start_time') or ''
        return item[1].get('abs_start', '')

    items.sort(key=_item_time)

    parts: list[str] = ['\n'.join(header_lines)] if header_lines else []

    # Buffer for consecutive orphaned audio segments (no annotation overlap)
    audio_buf: list[dict] = []

    def _flush_audio_buf() -> None:
        if not audio_buf:
            return
        t = _time_range_hms(
            audio_buf[0].get('abs_start', ''),
            audio_buf[-1].get('abs_end', ''),
        )
        words = ' '.join(
            (s.get('cleaned_text') or s.get('text', ''))
            + (' [!]' if s.get('needs_review') else '')
            for s in audio_buf
        )
        parts.append(f'[{t}] {words}\n  (no annotation activity)')
        audio_buf.clear()

    for item in items:
        if item[0] == 'span':
            _, i, span = item
            item_date = (span.get('start_time') or '')[:10]
        else:
            _, seg = item
            item_date = (seg.get('abs_start') or '')[:10]

        if item_date and item_date != current_date:
            if current_date is not None:
                _flush_audio_buf()
                parts.append(f'--- {item_date} ---')
            current_date = item_date

        if item[0] == 'span':
            _flush_audio_buf()
            block = _fmt_span_block(span)
            phrases = span_audio.get(i)
            if phrases:
                for abs_start, abs_end, text in phrases:
                    block += f'\n  [{_time_range_hms(abs_start, abs_end)}] {text}'
            parts.append(block)
        else:
            audio_buf.append(seg)

    _flush_audio_buf()
    return '\n\n'.join(parts) + '\n'
