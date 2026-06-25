# CLAUDE.md

Guidance for working in this repository.

## Overview

Local speech-to-text tool built on `microsoft/VibeVoice-ASR-HF`. It is a single-file
Python app (`vibevoice_asr.py`) that runs as a resident, arrow-key-driven Rich TUI.
The model is loaded once per session and stays in memory; the user picks actions
(transcribe a file, record from mic, change settings, pick input device) from in-app
menus. There are no CLI arguments — every option lives in the interactive menus.

Recognized text is written to `output/<source>_<timestamp>.txt`. Simplified Chinese in
the result is automatically converted to Traditional (Taiwan vocabulary) before display
and file output.

## Running

- Normal launch: double-click `run.bat` (activates the `vibevoice-asr` conda env, sets
  UTF-8, runs the app).
- Manual: `conda activate vibevoice-asr` then `python vibevoice_asr.py`.
- The interactive arrow-key UI requires Windows + a real TTY + Rich. Without a TTY the
  app falls back to a numbered text-input mode automatically.

## Environment setup (important quirks)

- GPU torch: on RTX 50-series (Blackwell, sm_120) torch/torchaudio MUST come from the
  CUDA 12.8 index (`--extra-index-url https://download.pytorch.org/whl/cu128`), otherwise
  pip installs a CPU or incompatible build. See `environment.yml`.
- `xz` from conda-forge is required so `liblzma.dll` is present on Windows; the default
  channel build can be missing it, which breaks `librosa` import.
- Dependencies are split: `environment.yml` (conda env + torch via cu128 pip index) and
  `requirements.txt` (pure-pip deps). `opencc` (Simplified→Traditional) is required and
  listed in `requirements.txt`.
- Model files live in `models/VibeVoice-ASR-HF/`. If absent, the app offers to download
  from the Hub on first run.

## Architecture (`vibevoice_asr.py`, single file)

Organized top-to-bottom into clear sections:

1. **UI layer (Rich)** — `ui_print` / `ui_warning` / `ui_success` / `ui_panel`, plus the
   full-screen pieces. All dynamic surfaces (menus, progress) use `screen=True`
   (alternate-screen, full repaint) so they survive terminal resize without leaving
   artifacts. Do NOT switch them back to `screen=False`.
   - `fullscreen_progress(...)` — context manager yielding a `Progress` driven by an
     outer alt-screen `Live`. Progress shows: bar line, a status line, and CPU/RAM/GPU/
     VRAM each on its own line as a `ProgressBar` (`render_resource_bars`).
   - `_message_sink` / `_capture_messages` — while a full-screen `Live` is active, `ui_*`
     messages are captured into an in-panel log instead of printing (which would be
     erased by the repaint).
   - `select_menu` — alt-screen arrow-key/number menu; returns index or `None` on Esc.
2. **Audio I/O** — `record_fixed`, `record_until_enter` (sounddevice), `load_audio_file`
   (soundfile fast path, else librosa), all resampled to `TARGET_SR` (24 kHz).
3. **Model load / inference** — `build_model` (4bit/8bit/none via bitsandbytes; tries
   `sdpa` attention, falls back to default), `transcribe`, `transcribe_windowed`.
4. **OOM handling** — long audio is split into bounded windows
   (`DEFAULT_WINDOW_SECONDS`). On CUDA OOM the code retries with smaller tokenizer
   `chunk_size` (`_chunk_retry_sizes`) and, if needed, halves the window recursively.
   This OOM/windowing logic is load-bearing — preserve it.
5. **Simplified→Traditional** — `to_traditional` (lazy OpenCC `s2twp`, degrades
   gracefully if OpenCC missing) and `convert_result_to_traditional` (mutates result
   `text` and each parsed segment `Content`). Applied in `run_transcription` after
   transcription, before `show_result`.
6. **Output** — `write_transcript` (plain text; optional speaker/timestamp block when
   `with_timestamps`), `default_output_path`.
7. **Flow / main loop** — `Settings` dataclass holds runtime-adjustable options; `state`
   dict caches the loaded model (reloaded only when quant mode changes). `main()` drives
   the top-level menu.

## Conventions

- Keep changes confined to the UI layer when fixing UI issues; the inference/OOM backend
  should stay untouched unless explicitly required.
- VibeVoice tokenizer requires `chunk_size` to be a multiple of `TOKENIZER_HOP_LENGTH`
  (3200); use `_normalize_chunk_size`.
- Comments and in-app strings are Traditional Chinese; match the surrounding style.
- Use relative paths; `_display_path` renders paths relative to the project root.
