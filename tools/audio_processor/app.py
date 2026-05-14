"""Minimal tkinter GUI for the SegmentHumanBody audio processor.

Run with:
    python tools/audio_processor/app.py
    python -m tools.audio_processor   (from repo root)
"""
from __future__ import annotations

import json
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
        self.minsize(540, 420)
        self._result: dict | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        p = {'padx': 6, 'pady': 3}

        # ---- File section ----------------------------------------
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

        # ---- Options section -------------------------------------
        grp_opts = ttk.LabelFrame(self, text='Options')
        grp_opts.pack(fill='x', padx=8, pady=2)

        ttk.Label(grp_opts, text='Model:').grid(row=0, column=0, sticky='e', **p)
        self._model_var = tk.StringVar(value='base')
        ttk.Combobox(
            grp_opts, textvariable=self._model_var,
            values=['tiny', 'base', 'small', 'medium', 'large-v3'],
            state='readonly', width=12,
        ).grid(row=0, column=1, sticky='w', **p)

        ttk.Label(grp_opts, text='Device:').grid(row=0, column=2, sticky='e', **p)
        self._device_var = tk.StringVar(value='auto')
        ttk.Combobox(
            grp_opts, textvariable=self._device_var,
            values=['auto', 'cuda', 'cpu'],
            state='readonly', width=8,
        ).grid(row=0, column=3, sticky='w', **p)

        ttk.Label(grp_opts, text='Language:').grid(row=1, column=0, sticky='e', **p)
        self._lang_var = tk.StringVar(value='auto')
        ttk.Entry(grp_opts, textvariable=self._lang_var, width=8).grid(
            row=1, column=1, sticky='w', **p)

        ttk.Label(grp_opts, text='Audio offset (s):').grid(
            row=1, column=2, sticky='e', **p)
        self._offset_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(
            grp_opts, textvariable=self._offset_var,
            from_=-120.0, to=120.0, increment=0.5, width=7,
        ).grid(row=1, column=3, sticky='w', **p)

        ttk.Label(grp_opts, text='Phrase gap (s):').grid(
            row=2, column=0, sticky='e', **p)
        self._gap_var = tk.DoubleVar(value=0.35)
        ttk.Spinbox(
            grp_opts, textvariable=self._gap_var,
            from_=0.05, to=5.0, increment=0.05, width=7,
        ).grid(row=2, column=1, sticky='w', **p)

        ttk.Label(grp_opts, text='Corrections:').grid(row=2, column=2, sticky='e', **p)
        self._corrections_var = tk.StringVar()
        ttk.Entry(grp_opts, textvariable=self._corrections_var, width=16).grid(
            row=2, column=3, sticky='ew', **p)
        ttk.Button(grp_opts, text='Browse…', command=self._browse_corrections).grid(
            row=2, column=4, **p)

        ttk.Label(
            grp_opts,
            text='Offset: positive = WAV started that many seconds before annotation start.',
            foreground='gray',
        ).grid(row=3, column=0, columnspan=4, sticky='w', padx=6)

        # ---- Prompt section --------------------------------------
        grp_prompt = ttk.LabelFrame(self, text='Initial Prompt (optional — bias Whisper toward domain vocabulary)')
        grp_prompt.pack(fill='x', padx=8, pady=2)

        self._prompt_text = tk.Text(grp_prompt, height=4, wrap='word')
        sb_p = ttk.Scrollbar(grp_prompt, command=self._prompt_text.yview)
        self._prompt_text.configure(yscrollcommand=sb_p.set)
        self._prompt_text.pack(side='left', fill='both', expand=True, padx=(4, 0), pady=3)
        sb_p.pack(side='right', fill='y', pady=3)

        # ---- Run button ------------------------------------------
        self._run_btn = ttk.Button(self, text='Run', command=self._on_run)
        self._run_btn.pack(pady=(6, 2))

        # ---- Log section -----------------------------------------
        grp_log = ttk.LabelFrame(self, text='Log')
        grp_log.pack(fill='both', expand=True, padx=8, pady=(2, 8))

        self._log_text = tk.Text(grp_log, height=14, wrap='word', state='disabled')
        sb = ttk.Scrollbar(grp_log, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)
        self._log_text.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

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
        path = filedialog.askdirectory(title='Select corrections directory (leave blank to skip)')
        if path:
            self._corrections_var.set(path)

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
    # Run
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        json_path = self._json_var.get().strip()
        wav_path  = self._wav_var.get().strip()
        if not json_path or not wav_path:
            self._append_log('ERROR: Both JSON and WAV paths are required.')
            return

        self._run_btn.configure(state='disabled')
        self._result = None

        def _worker() -> None:
            try:
                import processor  # local import; sys.path already set
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
                    initial_prompt=self._prompt_text.get('1.0', 'end').strip() or None,
                    audio_offset=self._offset_var.get(),
                    silence_gap=self._gap_var.get(),
                    corrections_dir=self._corrections_var.get().strip() or None,
                    progress_cb=self._append_log,
                )
                self._result = result

                # Save outputs next to the JSON file
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

                self._append_log(f'\nSaved:\n  {out_json}\n  {out_txt}\n  {out_caption}')

            except Exception as exc:
                self._append_log(f'\nERROR: {exc}')
                import traceback
                self._append_log(traceback.format_exc())
            finally:
                self.after(0, lambda: self._run_btn.configure(state='normal'))

        threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
