"""Text and caption report formatting.

format_text_report   → _transcript.txt  (audio-centric, each phrase + linked spans)
format_caption_report → _caption.txt    (span-centric, each span + overlapping audio)
"""
from __future__ import annotations

import datetime

from _annotation import _HAS_SUMMARIZER  # type: ignore[import]
from _times import _hms, _time_range_hms  # type: ignore[import]


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


def format_text_report(result: dict) -> str:
    lines: list[str] = []
    meta = result.get('metadata') or {}

    lines.append('Audio Transcript + Annotation Alignment Report')
    lines.append('=' * 50)
    if meta.get('start_time'):
        lines.append(f"Recording start : {meta['start_time']}")
    info = result.get('transcription_info') or {}
    if info.get('language'):
        lines.append(
            f"Language        : {info['language']} "
            f"(confidence {info.get('language_probability', '?')})"
        )
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
