# VibeVoice ASR CLI

A local, resident speech-to-text tool built on the
[`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF)
model. It runs as an arrow-key driven, full-screen terminal UI (Rich). The model
is loaded once per session and stays resident in memory; you pick actions —
transcribe a file, record from the microphone, change settings, choose an input
device — from in-app menus. There are no command-line arguments; every option
lives in the interactive menus.

Recognized text is written to `output/<timestamp>_<source>.txt`. Simplified
Chinese in the result is automatically converted to Traditional Chinese (Taiwan
vocabulary) before display and file output.

## Features

- Resident TUI with arrow-key / number menus; the model is loaded once and kept
  warm for the whole session.
- Two input sources: existing audio files, or live microphone recording
  (fixed duration, or press-Enter start / stop).
- Long-form audio by design: single-pass transcription (the model handles up to
  ~60 minutes in one pass), with automatic fallback to bounded windowing on GPU
  out-of-memory.
- Optional speaker labels and timestamps (structured "who / when / what" output).
- Automatic Simplified -> Traditional Chinese conversion (Taiwan vocabulary, via
  OpenCC).
- 4-bit / 8-bit / no quantization (bitsandbytes), switchable at runtime.
- Broad audio format support via ffmpeg decoding: wav, mp3, flac, m4a, aac, wma,
  ogg, opus.
- Live CPU / RAM / GPU / VRAM monitor while loading and transcribing.

## Requirements

- OS: Windows 10 / 11. The interactive UI targets Windows and a real terminal;
  without a TTY it falls back to a numbered text-input mode.
- GPU: an NVIDIA CUDA GPU. ~16 GB VRAM is comfortable for long audio at 4-bit;
  less VRAM still works for shorter clips or smaller windows.
  - RTX 50-series (Blackwell, sm_120) requires the CUDA 12.8 (cu128) PyTorch
    build. The bundled `environment.yml` already pins this index.
- Disk: ~16 GB free for the model (downloaded on first use) plus working space.
- Software: Miniconda or Anaconda. `ffmpeg` is installed automatically as part of
  the conda environment.

## Installation and first run

1. Clone the repository:
   ```
   git clone https://github.com/yuyan2123/VibeVoice-asr-cli.git
   cd VibeVoice-asr-cli
   ```
2. Double-click `run.bat` (or run it from a terminal).
   - On first launch it detects that the `vibevoice-asr` conda environment does
     not exist and creates it automatically from `environment.yml`. This is a
     one-time step and can take several minutes (it downloads the cu128 PyTorch
     build). Miniconda/Anaconda must be installed beforehand.
3. The model is **not** bundled with this repository. On the first transcription
   the app offers to download `microsoft/VibeVoice-ASR-HF` (~16 GB) from the
   Hugging Face Hub. The model is public, so no account or token is required —
   just confirm the prompt and it downloads into `models/`.

### Manual environment setup (alternative)

```
conda env create -f environment.yml
conda activate vibevoice-asr
python vibevoice_asr.py
```

If PyTorch was installed as a CPU-only build, reinstall it from the cu128 index:

```
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch torchaudio
```

## Output

- Transcripts: `output/<timestamp>_<source>.txt`. The timestamp comes first, so
  files sort chronologically by name.
- Microphone recordings (when kept): `recordings/`.
- `models/`, `output/`, `recordings/` and logs are local-only and excluded from
  version control.
