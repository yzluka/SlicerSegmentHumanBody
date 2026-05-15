"""Word-to-phrase merging and the phrase-correction workflow.

Phrase corrections workflow
---------------------------
1. ``merge_words_to_phrases``  groups Whisper words into phrases.
2. ``write_phrases_txt``       writes an editable one-phrase-per-line file.
3. User edits the file in any text editor.
4. ``apply_phrase_corrections`` reconciles edits back into a refined word list
   and writes ``whisper_{stem}_refined.json``.
"""
from __future__ import annotations

import datetime
import json
import re
from difflib import SequenceMatcher

from _times import _parse_iso  # type: ignore[import]


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


def write_phrases_txt(phrases: list[dict], output_path: str) -> None:
    """Write one-phrase-per-line editable file from merged phrases.

    Format per data line: ``{n}  [{start:.3f}–{end:.3f}]  {text}``
    The index and bracketed timestamps are machine-read; edit only the text
    after the closing ``]``.  Do not add or remove lines.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('# Edit text after the ] on each line. Do not add or remove lines.\n')
        for i, phrase in enumerate(phrases, 1):
            text = phrase.get('cleaned_text') or phrase.get('text', '')
            f.write(f"{i}  [{phrase['start']:.3f}–{phrase['end']:.3f}]  {text}\n")


def read_phrases_txt(txt_path: str) -> list[dict]:
    """Parse an edited phrases file.

    Returns list of ``{index, start, end, text}`` dicts, one per data line.
    Comment lines (``#``) and blank lines are ignored.
    """
    result: list[dict] = []
    _pat = re.compile(r'^(\d+)\s+\[([0-9.]+)–([0-9.]+)\]\s*(.*)', re.UNICODE)
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if line.startswith('#') or not line.strip():
                continue
            m = _pat.match(line)
            if not m:
                continue
            result.append({
                'index': int(m.group(1)),
                'start': float(m.group(2)),
                'end':   float(m.group(3)),
                'text':  m.group(4).strip(),
            })
    return result


def _apply_word_diff(phrase_words: list[dict], corrected: str) -> list[dict]:
    """Align corrected text to original word entries using edit-distance alignment.

    SequenceMatcher finds the longest matching subsequences between original
    word tokens and corrected tokens.  For each opcode block:

    equal   — unchanged tokens: keep original word entries verbatim (timing,
              probability, no is_correction flag).
    replace — n originals → m corrected: emit m synthetic entries distributing
              the original time span proportionally; each carries the average
              probability of the replaced originals.
    delete  — original words removed: emit nothing.
    insert  — new tokens with no original: distribute in the gap between the
              adjacent original words.
    """
    orig_tokens = [w['word'] for w in phrase_words]
    corr_tokens = corrected.split()
    result: list[dict] = []

    for tag, i1, i2, j1, j2 in SequenceMatcher(
        None, orig_tokens, corr_tokens, autojunk=False
    ).get_opcodes():
        orig_slice = phrase_words[i1:i2]
        corr_slice = corr_tokens[j1:j2]

        if tag == 'equal':
            result.extend(orig_slice)
            continue

        if not corr_slice:  # delete — drop original words
            continue

        # replace or insert: determine the time span to distribute across
        if orig_slice:
            span_start = orig_slice[0]['start']
            span_end   = orig_slice[-1]['end']
            avg_prob   = round(
                sum(w.get('probability') or 0 for w in orig_slice) / len(orig_slice), 3
            )
        else:
            # insert: use the gap between the surrounding original words
            prev_end   = phrase_words[i1 - 1]['end']   if i1 > 0                  else (phrase_words[0]['start'] if phrase_words else 0.0)
            next_start = phrase_words[i1]['start']      if i1 < len(phrase_words)  else prev_end
            span_start, span_end, avg_prob = prev_end, next_start, None

        if len(corr_slice) == 1:
            result.append({
                'start': span_start,
                'end':   span_end,
                'word':  corr_slice[0],
                'probability': avg_prob,
                'is_correction': True,
            })
        else:
            # Distribute time proportionally across the corrected tokens
            step = (span_end - span_start) / len(corr_slice)
            for k, token in enumerate(corr_slice):
                result.append({
                    'start': round(span_start + k * step,       3),
                    'end':   round(span_start + (k + 1) * step, 3),
                    'word':  token,
                    'probability': avg_prob,
                    'is_correction': True,
                })

    return result


def apply_phrase_corrections(
    whisper_json_path: str,
    phrases_txt_path: str,
    output_path: str,
) -> list[dict]:
    """Reconcile user edits in phrases txt with raw Whisper word output.

    For each phrase:
    - Text unchanged → keep original word entries verbatim.
    - Text changed   → edit-distance alignment via ``_apply_word_diff``:
        equal blocks   keep original word entries (timing + probability intact).
        replace blocks emit synthetic entries reusing the original time span,
                       distributed proportionally across corrected tokens.
        delete blocks  drop the removed original words.
        insert blocks  emit synthetic entries in the gap between adjacent words.
    - Text empty     → drop the phrase (deletion).

    Returns the refined words list and writes ``output_path``.
    """
    with open(whisper_json_path, encoding='utf-8') as f:
        whisper_data = json.load(f)

    words: list[dict] = list(whisper_data.get('words') or [])
    phrases = read_phrases_txt(phrases_txt_path)
    _EPS = 0.005  # seconds: tolerance for float boundary matching

    refined: list[dict] = []
    for phrase in phrases:
        p_start, p_end = phrase['start'], phrase['end']
        corrected = phrase['text']

        phrase_words = [
            w for w in words
            if w['start'] >= p_start - _EPS and w['end'] <= p_end + _EPS
        ]
        original_text = ' '.join(w['word'] for w in phrase_words)

        if not corrected:
            pass  # deleted phrase — emit nothing
        elif corrected == original_text:
            refined.extend(phrase_words)
        else:
            refined.extend(_apply_word_diff(phrase_words, corrected))

    out_data = {
        **{k: v for k, v in whisper_data.items() if k != 'words'},
        'type': 'SegmentHumanBody.whisper_words_refined',
        'source_whisper_json': str(whisper_json_path),
        'source_phrases_txt': str(phrases_txt_path),
        'words': refined,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    return refined
