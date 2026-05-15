"""Minimal tkinter GUI for the SegmentHumanBody audio processor.

Two-stage workflow
------------------
Stage 1 (Transcribe): JSON + WAV → whisper_{stem}.json + phrases_{stem}.txt
                       + _transcript.json / _transcript.txt / _caption.txt
Stage 2 (Correct):    user edits phrases_{stem}.txt in any text editor →
                       Apply Corrections → whisper_{stem}_refined.json +
                       updated _transcript / _caption outputs

Run with:
    python tools/audio_processor/app.py
    python -m tools.audio_processor   (from repo root)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, ttk

# Ensure this directory is importable (needed when run as a script directly)
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_wav(json_path: str) -> str:
    """Find the WAV whose filename timestamp best matches metadata.start_time in the JSON.

    WAV filenames follow the pattern ``{base}_{YYYYMMDDTHHMMSSMMM}.wav`` where
    the timestamp comes from _recording_start_time in the Slicer widget, which
    is set a few milliseconds before metadata.start_time is written.  We pick
    the candidate whose parsed timestamp is closest to metadata.start_time and
    within a 5-second tolerance.  Falls back to first candidate if the JSON
    cannot be parsed or no timestamp-bearing WAV is found.
    """
    import datetime
    import json as _json
    import re

    p = Path(json_path)
    stem = p.stem
    if stem.endswith('_raw'):
        stem = stem[:-4]

    candidates = list(p.parent.glob(f'{stem}*.wav'))
    if not candidates:
        return ''
    if len(candidates) == 1:
        return str(candidates[0])

    # Read start_time from the JSON metadata.
    start_dt = None
    try:
        data = _json.loads(p.read_text(encoding='utf-8'))
        start_str = (data.get('metadata') or {}).get('start_time')
        if start_str:
            start_dt = datetime.datetime.fromisoformat(start_str)
    except Exception:
        pass

    if start_dt is None:
        return str(candidates[0])

    # Match each candidate whose name ends with _{YYYYMMDDTHHMMSSMMM}.wav.
    # %f accepts 1-6 digits; 3-digit milliseconds are padded to microseconds.
    best: Path | None = None
    best_delta: float | None = None
    for wav in candidates:
        m = re.search(r'_(\d{8}T\d{9})\.wav$', wav.name)
        if not m:
            continue
        try:
            wav_dt = datetime.datetime.strptime(m.group(1), '%Y%m%dT%H%M%S%f')
        except ValueError:
            continue
        delta = abs((wav_dt - start_dt).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = wav

    if best is not None and best_delta <= 5.0:
        return str(best)

    return str(candidates[0])


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('Audio Processor')
        self.minsize(560, 540)
        self._build_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        p = {'padx': 6, 'pady': 3}

        # ---- Files ---------------------------------------------------
        grp_files = ttk.LabelFrame(self, text='Files')
        grp_files.pack(fill='x', padx=8, pady=(6, 2))

        ttk.Label(grp_files, text='JSON:').grid(row=0, column=0, sticky='e', **p)
        self._json_var = tk.StringVar()
        ttk.Entry(grp_files, textvariable=self._json_var).grid(
            row=0, column=1, sticky='ew', **p)
        ttk.Button(grp_files, text='Browse…', command=self._browse_json).grid(
            row=0, column=2, **p)

        ttk.Label(grp_files, text='WAV:').grid(row=1, column=0, sticky='e', **p)
        self._wav_var = tk.StringVar()
        ttk.Entry(grp_files, textvariable=self._wav_var).grid(
            row=1, column=1, sticky='ew', **p)
        ttk.Button(grp_files, text='Browse…', command=self._browse_wav).grid(
            row=1, column=2, **p)

        grp_files.columnconfigure(1, weight=1)

        # ---- Step 1 · Transcribe -------------------------------------
        grp_s1 = ttk.LabelFrame(self, text='Step 1 · Transcribe')
        grp_s1.pack(fill='x', padx=8, pady=2)

        ttk.Label(grp_s1, text='Model:').grid(row=0, column=0, sticky='e', **p)
        self._model_var = tk.StringVar(value='base')
        ttk.Combobox(
            grp_s1, textvariable=self._model_var,
            values=['tiny', 'base', 'small', 'medium', 'large-v3'],
            state='readonly', width=12,
        ).grid(row=0, column=1, sticky='w', **p)

        ttk.Label(grp_s1, text='Device:').grid(row=0, column=2, sticky='e', **p)
        self._device_var = tk.StringVar(value='auto')
        ttk.Combobox(
            grp_s1, textvariable=self._device_var,
            values=['auto', 'cuda', 'cpu'],
            state='readonly', width=8,
        ).grid(row=0, column=3, sticky='w', **p)

        ttk.Label(grp_s1, text='Language:').grid(row=1, column=0, sticky='e', **p)
        self._lang_var = tk.StringVar(value='auto')
        ttk.Entry(grp_s1, textvariable=self._lang_var, width=8).grid(
            row=1, column=1, sticky='w', **p)

        ttk.Label(grp_s1, text='Audio offset (s):').grid(row=1, column=2, sticky='e', **p)
        self._offset_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(
            grp_s1, textvariable=self._offset_var,
            from_=-120.0, to=120.0, increment=0.5, width=7,
        ).grid(row=1, column=3, sticky='w', **p)

        ttk.Label(grp_s1, text='Phrase gap (s):').grid(row=2, column=0, sticky='e', **p)
        self._gap_var = tk.DoubleVar(value=0.35)
        ttk.Spinbox(
            grp_s1, textvariable=self._gap_var,
            from_=0.05, to=5.0, increment=0.05, width=7,
        ).grid(row=2, column=1, sticky='w', **p)

        ttk.Label(grp_s1, text='Corrections:').grid(row=2, column=2, sticky='e', **p)
        self._corrections_var = tk.StringVar()
        ttk.Entry(grp_s1, textvariable=self._corrections_var, width=14).grid(
            row=2, column=3, sticky='ew', **p)
        ttk.Button(grp_s1, text='Browse…', command=self._browse_corrections).grid(
            row=2, column=4, **p)

        ttk.Label(
            grp_s1,
            text='Offset: positive = WAV started that many seconds before annotation start.',
            foreground='gray',
        ).grid(row=3, column=0, columnspan=5, sticky='w', padx=6)

        grp_prompt = ttk.LabelFrame(grp_s1, text='Initial Prompt (optional)')
        grp_prompt.grid(row=4, column=0, columnspan=5, sticky='ew', padx=6, pady=(4, 2))
        grp_s1.columnconfigure(3, weight=1)

        self._prompt_text = tk.Text(grp_prompt, height=3, wrap='word')
        sb_p = ttk.Scrollbar(grp_prompt, command=self._prompt_text.yview)
        self._prompt_text.configure(yscrollcommand=sb_p.set)
        self._prompt_text.pack(side='left', fill='both', expand=True, padx=(4, 0), pady=3)
        sb_p.pack(side='right', fill='y', pady=3)

        self._run_btn = ttk.Button(grp_s1, text='Transcribe', command=self._on_run)
        self._run_btn.grid(row=5, column=0, columnspan=5, pady=(4, 6))

        # ---- Step 2 · Text Fixed — Generate Reports ------------------
        grp_s2 = ttk.LabelFrame(self, text='Step 2 · Text Fixed — Generate Reports')
        grp_s2.pack(fill='x', padx=8, pady=2)

        ttk.Label(
            grp_s2,
            text=(
                'Open phrases_{stem}.txt in any text editor and fix transcription mistakes.\n'
                'Each line is one spoken phrase: edit only the text after the ] bracket.\n'
                'Do not change line numbers or the [start–end] timestamps — they are used\n'
                'to match your edits back to the original word timings.\n'
                'To remove a phrase entirely, clear its text (leave the line otherwise intact).\n'
                'When the text looks correct, click Generate Reports to align and export.'
            ),
            justify='left',
        ).grid(row=0, column=0, columnspan=3, sticky='w', padx=6, pady=(4, 4))

        ttk.Label(grp_s2, text='Phrases file:').grid(row=1, column=0, sticky='e', **p)
        self._phrases_path_var = tk.StringVar()
        ttk.Entry(grp_s2, textvariable=self._phrases_path_var, state='readonly').grid(
            row=1, column=1, sticky='ew', **p)
        ttk.Button(grp_s2, text='Open', command=self._open_phrases_file).grid(
            row=1, column=2, **p)
        grp_s2.columnconfigure(1, weight=1)

        self._apply_btn = ttk.Button(
            grp_s2, text='Generate Reports', command=self._on_apply_corrections,
        )
        self._apply_btn.grid(row=2, column=0, columnspan=3, pady=(2, 6))

        # ---- Log -----------------------------------------------------
        grp_log = ttk.LabelFrame(self, text='Log')
        grp_log.pack(fill='both', expand=True, padx=8, pady=(2, 8))

        self._log_text = tk.Text(grp_log, height=12, wrap='word', state='disabled')
        sb = ttk.Scrollbar(grp_log, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)
        self._log_text.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        # Keep phrases path display in sync with file selections
        self._json_var.trace_add('write', self._update_phrases_path)
        self._wav_var.trace_add('write', self._update_phrases_path)

    # ------------------------------------------------------------------
    # File pickers
    # ------------------------------------------------------------------

    def _browse_json(self) -> None:
        path = filedialog.askopenfilename(
            title='Select annotation JSON',
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
        )
        if path:
            self._json_var.set(path)
            if not self._wav_var.get():
                wav = _auto_wav(path)
                if wav:
                    self._wav_var.set(wav)

    def _browse_wav(self) -> None:
        path = filedialog.askopenfilename(
            title='Select WAV file',
            filetypes=[('WAV files', '*.wav'), ('All files', '*.*')],
        )
        if path:
            self._wav_var.set(path)

    def _browse_corrections(self) -> None:
        path = filedialog.askdirectory(
            title='Select corrections directory (leave blank to skip)')
        if path:
            self._corrections_var.set(path)

    # ------------------------------------------------------------------
    # Phrases path helpers
    # ------------------------------------------------------------------

    def _update_phrases_path(self, *_) -> None:
        """Recompute the expected phrases file path from current JSON/WAV."""
        json_path = self._json_var.get().strip()
        wav_path  = self._wav_var.get().strip()
        if json_path and wav_path:
            wav_stem = Path(wav_path).stem
            self._phrases_path_var.set(
                str(Path(json_path).parent / f'phrases_{wav_stem}.txt')
            )
        else:
            self._phrases_path_var.set('')

    def _open_phrases_file(self) -> None:
        path = self._phrases_path_var.get().strip()
        if not path:
            self._append_log('Set JSON and WAV paths first.')
            return
        if not Path(path).exists():
            self._append_log(f'File not found: {path}\nRun Step 1 (Transcribe) first.')
            return
        try:
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', path], check=False)
            else:
                subprocess.run(['xdg-open', path], check=False)
        except Exception as exc:
            self._append_log(f'Could not open file: {exc}')

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _append_log(self, msg: str) -> None:
        def _do() -> None:
            self._log_text.configure(state='normal')
            self._log_text.insert('end', msg + '\n')
            self._log_text.see('end')
            self._log_text.configure(state='disabled')
        self.after(0, _do)

    # ------------------------------------------------------------------
    # Stage 1 — Transcribe
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        json_path = self._json_var.get().strip()
        wav_path  = self._wav_var.get().strip()
        if not json_path or not wav_path:
            self._append_log('ERROR: Both JSON and WAV paths are required.')
            return

        self._run_btn.configure(state='disabled')
        self._apply_btn.configure(state='disabled')

        def _worker() -> None:
            try:
                import processor  # local import; sys.path already set
                processor.transcribe_and_phrase(
                    json_path=json_path,
                    wav_path=wav_path,
                    model_size=self._model_var.get(),
                    device=self._device_var.get(),
                    language=(
                        self._lang_var.get()
                        if self._lang_var.get() not in ('', 'auto')
                        else None
                    ),
                    initial_prompt=self._prompt_text.get('1.0', 'end').strip() or None,
                    audio_offset=self._offset_var.get(),
                    silence_gap=self._gap_var.get(),
                    progress_cb=self._append_log,
                )
                wav_stem = Path(wav_path).stem
                self._append_log(
                    f'\nStep 1 complete.'
                    f'\nReview phrases_{wav_stem}.txt and edit if needed,'
                    f'\nthen click Apply Corrections to align and generate output files.'
                )

            except Exception as exc:
                self._append_log(f'\nERROR: {exc}')
                import traceback
                self._append_log(traceback.format_exc())
            finally:
                self.after(0, lambda: self._run_btn.configure(state='normal'))
                self.after(0, lambda: self._apply_btn.configure(state='normal'))

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Stage 2 — Apply Corrections
    # ------------------------------------------------------------------

    def _on_apply_corrections(self) -> None:
        json_path = self._json_var.get().strip()
        wav_path  = self._wav_var.get().strip()
        if not json_path or not wav_path:
            self._append_log('ERROR: Both JSON and WAV paths are required.')
            return

        wav_stem     = Path(wav_path).stem
        out_dir      = Path(json_path).parent
        whisper_json = out_dir / f'whisper_{wav_stem}.json'
        phrases_txt  = out_dir / f'phrases_{wav_stem}.txt'
        refined_json = out_dir / f'whisper_{wav_stem}_refined.json'

        if not whisper_json.exists():
            self._append_log(
                f'ERROR: {whisper_json.name} not found — run Step 1 (Transcribe) first.')
            return
        if not phrases_txt.exists():
            self._append_log(
                f'ERROR: {phrases_txt.name} not found — run Step 1 (Transcribe) first.')
            return

        self._run_btn.configure(state='disabled')
        self._apply_btn.configure(state='disabled')

        def _worker() -> None:
            try:
                import processor

                self._append_log(f'Reading edits from {phrases_txt.name}…')
                refined_words = processor.apply_phrase_corrections(
                    str(whisper_json), str(phrases_txt), str(refined_json),
                )
                self._append_log(f'Saved refined words → {refined_json.name}')

                result = processor.process(
                    json_path=json_path,
                    wav_path=wav_path,
                    model_size=self._model_var.get(),
                    device=self._device_var.get(),
                    language=(
                        self._lang_var.get()
                        if self._lang_var.get() not in ('', 'auto')
                        else None
                    ),
                    audio_offset=self._offset_var.get(),
                    silence_gap=self._gap_var.get(),
                    corrections_dir=self._corrections_var.get().strip() or None,
                    words_override=refined_words,
                    progress_cb=self._append_log,
                )

                base = Path(json_path)
                if base.suffix == '.json':
                    base = base.with_suffix('')
                out_json    = base.with_name(base.name + '_transcript.json')
                out_txt     = base.with_name(base.name + '_transcript.txt')
                out_caption = base.with_name(base.name + '_caption.txt')

                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                with open(out_txt, 'w', encoding='utf-8') as f:
                    f.write(processor.format_text_report(result))
                with open(out_caption, 'w', encoding='utf-8') as f:
                    f.write(processor.format_caption_report(result))

                self._append_log(
                    f'\nSaved:\n  {refined_json}\n  {out_json}\n  {out_txt}\n  {out_caption}'
                )

            except Exception as exc:
                self._append_log(f'\nERROR: {exc}')
                import traceback
                self._append_log(traceback.format_exc())
            finally:
                self.after(0, lambda: self._run_btn.configure(state='normal'))
                self.after(0, lambda: self._apply_btn.configure(state='normal'))

        threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
