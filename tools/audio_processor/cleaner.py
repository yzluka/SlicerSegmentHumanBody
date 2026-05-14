"""Post-processing layer for transcribed annotation recordings.

A corrections directory contains *.json files:
  annotation_actions.json   {"regex_pattern": "replacement", ...}
  anatomy.json              {"regex_pattern": "replacement", ...}
  _review_patterns.json     ["pattern", ...]   (array, not a dict)

Pass any directory path to clean_phrases(), or None to skip post-processing.
Duplicate keys across files within the same directory emit a warning; last file wins.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path


def load_dir(directory: str | Path) -> tuple[dict[str, str], list[str]]:
    """Load corrections and review patterns from all *.json files in *directory*.

    Files named _review_patterns.json are treated as a list[str] of regex patterns.
    All other *.json files must be dict[str, str] (pattern → replacement).
    Duplicate keys emit a warning; last file wins.
    """
    folder = Path(directory)
    corrections: dict[str, str] = {}
    review_patterns: list[str] = []

    for path in sorted(folder.glob('*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            warnings.warn(f'cleaner: could not load {path}: {exc}')
            continue

        if path.name == '_review_patterns.json':
            if isinstance(data, list):
                review_patterns.extend(str(p) for p in data)
            else:
                warnings.warn(f'cleaner: {path.name} should be a JSON array')
            continue

        if not isinstance(data, dict):
            warnings.warn(f'cleaner: {path.name} should be a JSON object')
            continue

        for key in data:
            if key in corrections:
                warnings.warn(
                    f'cleaner: duplicate key {key!r} in {path.name} — overrides earlier definition'
                )
        corrections.update(data)

    return corrections, review_patterns


def clean_text(
    text: str,
    corrections: dict[str, str],
    review_patterns: list[str],
) -> tuple[str, list[dict], bool]:
    """Apply corrections then check review_patterns.

    Returns (cleaned_text, corrections_applied, needs_review).
    """
    cleaned = text
    applied: list[dict] = []

    for pattern, replacement in corrections.items():
        new_text = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        if new_text != cleaned:
            applied.append({'pattern': pattern, 'replacement': replacement})
            cleaned = new_text

    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = cleaned.replace(' ,', ',').replace(' .', '.')

    needs_review = any(
        re.search(p, cleaned, flags=re.IGNORECASE)
        for p in review_patterns
    )
    return cleaned, applied, needs_review


def clean_phrases(
    phrases: list[dict],
    corrections_dir: str | Path | None = None,
) -> list[dict]:
    """Apply a corrections directory to each phrase dict.

    corrections_dir=None → return phrases unchanged (no post-processing).
    Adds optional keys only when relevant:
      cleaned_text  – present when text was changed
      corrections   – list of applied {'pattern', 'replacement'} dicts
      needs_review  – True when a review pattern matched
    """
    if not corrections_dir:
        return phrases

    corrections, review_patterns = load_dir(corrections_dir)

    result: list[dict] = []
    for phrase in phrases:
        raw = phrase.get('text', '')
        cleaned, applied, needs_review = clean_text(raw, corrections, review_patterns)
        out = dict(phrase)
        if cleaned != raw:
            out['cleaned_text'] = cleaned
        if applied:
            out['corrections'] = applied
        if needs_review:
            out['needs_review'] = True
        result.append(out)
    return result
