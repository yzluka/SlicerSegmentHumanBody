"""faster-whisper transcription at word-level granularity."""
from __future__ import annotations

from typing import Callable


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
        import ctranslate2  # type: ignore[import]
        device = 'cuda' if ctranslate2.get_cuda_device_count() > 0 else 'cpu'

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
