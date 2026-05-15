"""Pipeline orchestrator for audio transcription and annotation alignment.

Standalone — no Slicer dependency.

Two-stage public API
---------------------
Stage 1  transcribe_and_phrase(...)
             WAV → whisper_{stem}.json + phrases_{stem}.txt only.
             Light: no cleaning, no alignment, no report files.

Stage 2  apply_phrase_corrections(...)   reconcile user edits → refined words
         process(...)                    full pipeline: merge → clean → align
         format_text_report(result)      render _transcript.txt
         format_caption_report(result)   render _caption.txt

Internal modules
----------------
_times.py       timestamp parsing & formatting
_annotation.py  JSON loading, span extraction, summarizer bootstrap
_transcribe.py  faster-whisper transcription
_phrases.py     word→phrase merge, editable txt I/O, corrections
_align.py       phrase-to-span alignment
_reports.py     report formatting
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from cleaner import clean_phrases  # type: ignore[import]

from _annotation import load_annotation_json, get_spans  # type: ignore[import]
from _times import parse_audio_start_from_wav_name, _parse_iso  # type: ignore[import]
from _transcribe import transcribe  # type: ignore[import]
from _phrases import (  # type: ignore[import]
    merge_words_to_phrases,
    write_phrases_txt,
    apply_phrase_corrections,
)
from _align import align  # type: ignore[import]
from _reports import format_text_report, format_caption_report  # type: ignore[import]

import datetime

# Re-export everything app.py calls directly on the `processor` module.
__all__ = [
    'transcribe_and_phrase',
    'process',
    'apply_phrase_corrections',
    'format_text_report',
    'format_caption_report',
]


def transcribe_and_phrase(
    json_path: str,
    wav_path: str,
    model_size: str = 'base',
    device: str = 'auto',
    language: str | None = None,
    initial_prompt: str | None = None,
    temperature: float = 0.0,
    audio_offset: float = 0.0,
    silence_gap: float = 0.35,
    progress_cb: Callable[[str], None] | None = None,
) -> None:
    """Stage 1: transcribe WAV and write an editable phrases file.

    Writes next to *json_path*:
      ``whisper_{wav_stem}.json``  — word-level Whisper output
      ``phrases_{wav_stem}.txt``   — one merged phrase per line, ready to edit

    Does not run cleaning, alignment, or report generation.
    """
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

    wav_stem = Path(wav_path).stem

    words, info = transcribe(wav_path, model_size, device, language,
                             initial_prompt, temperature, _log)

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
    _log(f'Saved whisper words → {whisper_json_path.name}')

    _log(f'Merging {len(words)} words into phrases (silence_gap={silence_gap}s)…')
    segments = merge_words_to_phrases(words, spans, audio_start, audio_offset, silence_gap)
    _log(f'{len(segments)} phrases formed.')

    phrases_txt_path = Path(json_path).parent / f'phrases_{wav_stem}.txt'
    write_phrases_txt(segments, str(phrases_txt_path))
    _log(f'Saved editable phrases → {phrases_txt_path.name}')


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
    words_override: list[dict] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Full pipeline: load → transcribe → save whisper JSON → merge → clean → align.

    If *words_override* is provided, transcription is skipped and those words
    are used directly (e.g. after applying phrase corrections via
    ``apply_phrase_corrections``).
    """
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

    wav_stem = Path(wav_path).stem
    whisper_json_path = Path(json_path).parent / f'whisper_{wav_stem}.json'

    if words_override is not None:
        words = words_override
        info: dict = {}
        if whisper_json_path.exists():
            try:
                with open(whisper_json_path, encoding='utf-8') as _f:
                    info = json.load(_f).get('transcription_info') or {}
            except Exception:
                pass
        _log(f'Using {len(words)} pre-loaded words (skipping transcription).')
    else:
        words, info = transcribe(wav_path, model_size, device, language,
                                 initial_prompt, temperature, _log)

        # Save raw whisper word output so phrase merging can be re-run without
        # re-running the model, and so apply_phrase_corrections has a source.
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

    phrases_txt_path = Path(json_path).parent / f'phrases_{wav_stem}.txt'
    write_phrases_txt(segments, str(phrases_txt_path))
    _log(f'Saved editable phrases → {phrases_txt_path}')

    _log('Aligning phrases to spans…')
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
