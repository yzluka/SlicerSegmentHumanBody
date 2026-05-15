"""Annotation JSON loading and span extraction.

Handles all three annotation types:
  annotation_summary  — spans used directly.
  annotation_process  — summarised inline via TimeLogSummarizer.
  annotation_raw      — interpreted then summarised inline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# TimeLogSummarizer lives in SegmentHumanBody/core/ — pure Python, no Slicer dep.
_core = Path(__file__).parent.parent.parent / 'SegmentHumanBody' / 'core'
if _core.exists() and str(_core) not in sys.path:
    sys.path.insert(0, str(_core))
try:
    from TimeLogInterpreter import TimeLogInterpreter as _Interpreter  # type: ignore[import]
    from TimeLogSummarizer import TimeLogSummarizer as _Summarizer  # type: ignore[import]
    _HAS_SUMMARIZER = True
except ImportError:
    _HAS_SUMMARIZER = False


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
