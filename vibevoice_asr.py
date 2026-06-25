#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VibeVoice ASR 本地語音辨識工具。

支援兩種輸入來源:
  1. 麥克風即時錄音(固定秒數或按 Enter 起訖)。
  2. 載入磁碟上既有的音訊檔(wav / mp3 / flac / m4a 等)。

辨識完成後會把純文字結果寫入 output/ 下的 .txt 檔;
若加上 --with-timestamps,會額外附上「語者 + 時間戳記」的結構化內容。
"""

import datetime
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

try:
    import msvcrt
except ImportError:  # 非 Windows 平台沒有 msvcrt,改用後援文字輸入
    msvcrt = None

import numpy as np

# 確保標準輸出為 UTF-8:在 cp950/Big5 語系且輸出被導向檔案或管線時,
# 避免列印中文或特殊裝置名稱時拋出 UnicodeEncodeError。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 降低 CUDA 記憶體碎片,減少顯存不足(OOM)機率。必須在匯入 torch 前設定。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message=".*feature_extractor_class.*")
warnings.filterwarnings("ignore", message=".*expandable_segments not supported.*")
warnings.filterwarnings("ignore", message=".*_check_is_size will be removed.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

# VibeVoice 聲學/語意 tokenizer 固定以 24kHz 運作,所有音訊都先統一到此取樣率。
TARGET_SR = 24000
MODEL_ID = "microsoft/VibeVoice-ASR-HF"
TOKENIZER_HOP_LENGTH = 3200
DEFAULT_CHUNK_SIZE = 480000
MIN_CHUNK_SIZE = 32000
DEFAULT_WINDOW_SECONDS = 600.0
MIN_WINDOW_SECONDS = 60.0

# 生成(generate)相關上限與防護。
# 模型自帶的 generation_config.json 預設 max_new_tokens=32768;先前固定 2048 會把
# 長音訊/長分段的輸出硬生生截斷(JSON 收不了尾),故改為「依音訊長度動態估算、以此為上限」。
DEFAULT_MAX_NEW_TOKENS = 32768       # 對齊模型預設,作為單次/分段的 token 上限
MIN_MAX_NEW_TOKENS = 512             # 動態估算下限,避免極短音訊估太小
EST_TOKENS_PER_SECOND = 25          # 密集多人對話(JSON+內容)的保守估值,用於動態估算
# repetition_penalty:阻止 greedy 解碼陷入「000…」這類暴衝重複;設 1.0 表示關閉。
GEN_REPETITION_PENALTY = 1.1

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "output")
RECORDING_DIR = os.path.join(THIS_DIR, "recordings")
MODEL_DIR = os.path.join(THIS_DIR, "models", "VibeVoice-ASR-HF")
# 專案內若已下載模型則優先用本地路徑,否則回退到 HuggingFace Hub 名稱(首次執行會自動下載)。
DEFAULT_MODEL = MODEL_DIR if os.path.isdir(MODEL_DIR) else MODEL_ID


try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.progress_bar import ProgressBar
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    RICH_AVAILABLE = True
except Exception:
    box = None
    Console = None
    Group = None
    Live = None
    Panel = None
    Progress = None
    ProgressBar = None
    Confirm = None
    Prompt = None
    Table = None
    RICH_AVAILABLE = False


USE_RICH = RICH_AVAILABLE
console = Console() if RICH_AVAILABLE else None

# 全螢幕 Live(替身畫面)啟用時,訊息改寫入此緩衝區並顯示在畫面內,
# 避免直接 print 被整頁重繪洗掉造成殘影。
_message_sink = None


@contextmanager
def _capture_messages(sink):
    """暫時把 ui_* 訊息導向 sink 串列,離開時還原。"""
    global _message_sink
    prev = _message_sink
    _message_sink = sink
    try:
        yield
    finally:
        _message_sink = prev


def _strip_rich_markup(message):
    return re.sub(r"\[/?[^\]]+\]", "", str(message))


def ui_print(message="", style=None):
    """優先使用 Rich；缺套件時退回一般 print。"""
    if _message_sink is not None and USE_RICH:
        _message_sink.append(message if style is None else f"[{style}]{message}[/{style}]")
        return
    if USE_RICH:
        console.print(message, style=style)
    else:
        print(_strip_rich_markup(message))


def ui_warning(message):
    if USE_RICH:
        ui_print(f"[bold yellow]警告[/bold yellow] {message}")
    else:
        print(f"警告: {message}")


def ui_success(message):
    if USE_RICH:
        ui_print(f"[bold green]✓[/bold green] {message}")
    else:
        print(f"成功: {message}")


def ui_error(message):
    if USE_RICH:
        ui_print(f"[bold red]錯誤[/bold red] {message}")
    else:
        print(f"錯誤: {message}")
def ui_rule(title):
    if USE_RICH:
        console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)


def ui_panel(title, body, border_style="cyan"):
    if USE_RICH:
        console.print(
            Panel(
                body,
                title=f"[bold]{title}[/bold]",
                border_style=border_style,
                box=box.ROUNDED,
            )
        )
    else:
        print(f"\n[{title}]")
        print(_strip_rich_markup(body))


@contextmanager
def ui_status(message):
    if USE_RICH:
        with console.status(message, spinner="dots"):
            yield
    else:
        print(_strip_rich_markup(message))
        yield


def format_duration(seconds):
    seconds = float(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:04.1f}s"
    if minutes:
        return f"{minutes}m {sec:04.1f}s"
    return f"{sec:.1f}s"


def _app_header_renderable():
    return Panel(
        "本地語音辨識工具\n[dim]支援音訊檔、麥克風錄音、長音訊自動分段與 OOM 自動降載。[/dim]",
        title="[bold]VibeVoice ASR[/bold]",
        border_style="bright_cyan",
        box=box.ROUNDED,
    )


def print_app_header():
    if USE_RICH:
        console.print(_app_header_renderable())
    else:
        print("\n[VibeVoice ASR]")
        print("本地語音辨識工具")


def make_progress():
    # 進度列只放會自動縮放的欄位(spinner / 描述 / bar / 百分比 / 時間),
    # status 與 resource 兩個較長的欄位改由 fullscreen_progress 拆成獨立行呈現。
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def collect_resource_stats():
    """回傳 CPU/RAM/GPU/VRAM 使用率的結構化數值,供進度面板畫成進度條。"""
    stats = {"cpu": None, "ram": None, "gpu": None, "vram": None, "vram_label": ""}
    try:
        import psutil

        stats["cpu"] = psutil.cpu_percent(interval=None)
        stats["ram"] = psutil.virtual_memory().percent
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        first = result.stdout.strip().splitlines()[0]
        gpu_util, mem_used, mem_total = [int(part.strip()) for part in first.split(",")[:3]]
        stats["gpu"] = gpu_util
        stats["vram"] = (mem_used / mem_total * 100) if mem_total else 0
        stats["vram_label"] = f"{mem_used}/{mem_total}MiB"
    except Exception:
        pass

    return stats


def _resource_level_style(value):
    """依使用率高低回傳顏色:綠 → 黃 → 紅。"""
    if value is None:
        return "grey50"
    if value >= 85:
        return "red"
    if value >= 60:
        return "yellow"
    return "green"


def render_resource_bars(stats):
    """把資源使用率畫成多行進度條(每項一行:標籤 + bar + 百分比)。"""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="left", no_wrap=True)   # 標籤
    grid.add_column(no_wrap=True)                   # 進度條
    grid.add_column(justify="right", no_wrap=True)  # 百分比
    grid.add_column(justify="left", no_wrap=True)   # 附註(VRAM 容量)

    # CPU/GPU 為使用率(固定色),RAM/VRAM 為記憶體壓力(依高低變色,接近滿載示警)。
    rows = [
        ("CPU", stats.get("cpu"), "cyan", ""),
        ("RAM", stats.get("ram"), _resource_level_style(stats.get("ram")), ""),
        ("GPU", stats.get("gpu"), "magenta", ""),
        ("VRAM", stats.get("vram"), _resource_level_style(stats.get("vram")), stats.get("vram_label") or ""),
    ]
    for label, value, style, extra in rows:
        completed = float(value) if value is not None else 0
        bar = ProgressBar(
            total=100,
            completed=completed,
            width=26,
            complete_style=style,
            finished_style=style,
        )
        pct = f"[{style}]{value:>3.0f}%[/{style}]" if value is not None else "[grey50] --%[/grey50]"
        grid.add_row(f"[bold]{label:<4}[/bold]", bar, pct, f"[dim]{extra}[/dim]" if extra else "")
    return grid


class ResourceMonitor:
    """Background updater for Rich progress resource fields during blocking GPU calls."""

    def __init__(self, progress, task_id, interval=1.0):
        self.progress = progress
        self.task_id = task_id
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def start(self):
        if not USE_RICH:
            return
        try:
            import psutil

            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.progress.update(self.task_id, resource=collect_resource_stats())
            except Exception:
                return


@contextmanager
def fullscreen_progress(title="處理中", subtitle=None):
    """以替身畫面(alt-screen)全螢幕顯示進度。

    每次刷新都從畫面原點整頁重繪,因此縮放視窗時不會留下殘影;
    期間的 ui_* 訊息會被收集到面板內顯示,而非直接 print 被洗掉。
    回傳尚未啟動自身 Live 的 Progress 物件,由外層 Live 統一驅動重繪。
    """
    progress = make_progress()
    log = []

    def render():
        # 進度列為第一行;status 一行;CPU/RAM/GPU/VRAM 各自一行並以進度條呈現。
        inner = [progress]
        for task in progress.tasks:
            status = task.fields.get("status")
            resource = task.fields.get("resource")
            if status:
                inner.append(f"[dim]{status}[/dim]")
            if isinstance(resource, dict):
                inner.append(render_resource_bars(resource))
        body = [
            Panel(
                Group(*inner),
                title=f"[bold]{title}[/bold]",
                subtitle=f"[dim]{subtitle}[/dim]" if subtitle else None,
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        ]
        if log:
            body.append(
                Panel(
                    Group(*log[-8:]),
                    title="[bold]訊息[/bold]",
                    border_style="yellow",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
        return Group(*body)

    with _capture_messages(log):
        with Live(
            get_renderable=render,
            console=console,
            screen=True,
            auto_refresh=True,
            refresh_per_second=8,
            vertical_overflow="crop",
        ):
            yield progress


def _display_path(path):
    """回傳可讀路徑;跨磁碟機時 relpath 會丟例外,改用絕對路徑。"""
    try:
        return os.path.relpath(path, THIS_DIR)
    except ValueError:
        return os.path.abspath(path)


# --------------------------------------------------------------------------- #
# 音訊輸入:麥克風錄音與檔案載入
# --------------------------------------------------------------------------- #
def record_fixed(seconds, samplerate=TARGET_SR, device=None):
    """錄製固定秒數的單聲道音訊,回傳 float32 numpy 陣列。"""
    import sounddevice as sd

    ui_print(f"[bold cyan]開始錄音[/bold cyan] 持續 {seconds} 秒...(請對著麥克風說話)")
    audio = sd.rec(
        int(seconds * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    ui_success("錄音結束。")
    return audio.reshape(-1)


def record_until_enter(samplerate=TARGET_SR, device=None):
    """按一次 Enter 開始、再按一次 Enter 停止的互動式錄音。"""
    import queue
    import threading

    import sounddevice as sd

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            ui_warning(f"音訊狀態: {status}")
        q.put(indata.copy())

    Prompt.ask("按下 Enter 開始錄音") if USE_RICH else input("按下 Enter 開始錄音...")
    stop_event = threading.Event()

    def wait_for_enter():
        input()
        stop_event.set()

    frames = []
    with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device,
        callback=callback,
    ):
        ui_print("[bold red]錄音中[/bold red]... 再次按下 Enter 即可停止。")
        threading.Thread(target=wait_for_enter, daemon=True).start()
        while not stop_event.is_set():
            try:
                frames.append(q.get(timeout=0.1))
            except queue.Empty:
                continue
        while not q.empty():
            frames.append(q.get())

    ui_success("錄音結束。")
    if not frames:
        raise RuntimeError("沒有錄到任何音訊資料。")
    audio = np.concatenate(frames, axis=0)
    return audio.reshape(-1)


def _decode_with_ffmpeg(path, samplerate=TARGET_SR):
    """用 ffmpeg 直接把任意格式解碼成單聲道 float32 PCM,並重採樣到目標取樣率。

    主要用於 libsndfile 無法解碼的格式(m4a/aac/wma 等),取代 librosa 已棄用的
    audioread 後備路徑。系統找不到 ffmpeg、或解碼失敗時回傳 None,交由呼叫端後備。
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    cmd = [
        ffmpeg,
        "-nostdin",
        "-loglevel", "error",
        "-i", path,
        "-f", "f32le",            # 原始 32-bit float little-endian PCM
        "-acodec", "pcm_f32le",
        "-ac", "1",               # 單聲道
        "-ar", str(samplerate),   # 重採樣到目標取樣率
        "-",                      # 輸出到 stdout
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    # frombuffer 產生的是唯讀檢視,複製一份成可寫且自有記憶體的陣列。
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_audio_file(path, samplerate=TARGET_SR):
    """載入任意音訊檔,轉成單聲道並重採樣到目標取樣率。

    解碼優先序:
      1. soundfile 快速路徑(取樣率已等於目標,免重採樣);
      2. ffmpeg 直接解碼(支援 m4a/aac/wma 等 libsndfile 無法解的格式並完成重採樣);
      3. librosa 後備(僅在系統無 ffmpeg 時使用)。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到音訊檔:{path}")

    try:
        import soundfile as sf

        info = sf.info(path)
        if info.samplerate == samplerate:
            audio, _ = sf.read(path, dtype="float32", always_2d=True)
            if audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            else:
                audio = audio[:, 0]
            return audio.astype(np.float32, copy=False)
    except Exception:
        pass

    audio = _decode_with_ffmpeg(path, samplerate)
    if audio is not None:
        return audio

    import librosa

    audio, _ = librosa.load(path, sr=samplerate, mono=True)
    return audio.astype(np.float32)


def save_wav(path, audio, samplerate=TARGET_SR):
    """把 numpy 音訊存成 WAV 檔。"""
    import soundfile as sf

    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, samplerate)
    ui_success(f"已儲存錄音: {_display_path(path)}")



def prompt_yes_no(message, default=False):
    if USE_RICH:
        return Confirm.ask(message, default=default)
    if not sys.stdin.isatty():
        return default
    suffix = "Y/n" if default else "y/N"
    answer = input(f"{message} [{suffix}]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes", "是", "好")


def download_model(model_id):
    """Download a HuggingFace model into models/ with Rich progress."""
    from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    ui_print(f"[bold cyan]準備下載模型[/bold cyan] {model_id}")
    api = HfApi()
    info = api.model_info(model_id, files_metadata=True)
    siblings = [item for item in info.siblings if not item.rfilename.endswith(".lock")]
    total_size = sum((getattr(item, "size", None) or 0) for item in siblings)
    total_files = len(siblings)

    if USE_RICH:
        total = total_size if total_size > 0 else total_files
        with fullscreen_progress("下載模型") as progress:
            task_id = progress.add_task(
                "下載模型",
                total=total,
                status=f"0/{total_files} 檔案",
                resource=collect_resource_stats(),
            )
            with ResourceMonitor(progress, task_id, interval=1.5):
                for index, item in enumerate(siblings, start=1):
                    filename = item.rfilename
                    size = getattr(item, "size", None) or 0
                    progress.update(task_id, status=f"{index}/{total_files} {filename}")
                    hf_hub_download(repo_id=model_id, filename=filename, local_dir=MODEL_DIR)
                    progress.update(task_id, advance=size if total_size > 0 else 1)
    else:
        for index, item in enumerate(siblings, start=1):
            print(f"下載模型檔案 {index}/{total_files}: {item.rfilename}")
            hf_hub_download(repo_id=model_id, filename=item.rfilename, local_dir=MODEL_DIR)

    ui_success(f"模型已下載到: {_display_path(MODEL_DIR)}")
    return MODEL_DIR


# --------------------------------------------------------------------------- #
# 模型載入與推論
# --------------------------------------------------------------------------- #
def build_model(model_path, quant):
    """載入處理器與模型。quant 可為 '4bit' / '8bit' / 'none'。"""
    import torch
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except Exception:
        pass

    load_kwargs = {"device_map": "auto"}

    if quant in ("4bit", "8bit"):
        try:
            from transformers import BitsAndBytesConfig

            if quant == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as exc:  # bitsandbytes 不可用時自動退回 bf16
            ui_warning(f"量化設定失敗({exc}),改以 bfloat16 全精度載入。")
            quant = "none"

    if quant == "none":
        load_kwargs["dtype"] = torch.bfloat16

    # 使用 sdpa 注意力:避免 eager 對長序列具現化 O(N^2) 的注意力矩陣而爆顯存。
    load_kwargs["attn_implementation"] = "sdpa"

    ui_print(f"[bold cyan]載入模型[/bold cyan] {model_path}  [dim]量化模式: {quant}[/dim]")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_path)
    active_attn = "sdpa"
    try:
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    except (ValueError, TypeError) as exc:
        ui_warning("sdpa 注意力不支援此模型，已改用預設注意力。")
        load_kwargs.pop("attn_implementation", None)
        active_attn = "預設"
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    model.eval()
    ui_success(f"模型已載入: 裝置={model.device}, dtype={model.dtype}, 注意力={active_attn}, 耗時 {time.time() - t0:.1f}s")
    return processor, model


def _is_cuda_oom(exc, torch_module):
    """Return True for CUDA OOM errors raised directly or wrapped as RuntimeError."""
    if isinstance(exc, torch_module.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)


def _normalize_chunk_size(chunk_size):
    """Use a valid tokenizer chunk size; VibeVoice requires multiples of 3200 samples."""
    if not chunk_size:
        return None
    chunk_size = int(chunk_size)
    if chunk_size < TOKENIZER_HOP_LENGTH:
        return TOKENIZER_HOP_LENGTH
    return max(TOKENIZER_HOP_LENGTH, (chunk_size // TOKENIZER_HOP_LENGTH) * TOKENIZER_HOP_LENGTH)


def _chunk_retry_sizes(chunk_size):
    """Largest-to-smallest tokenizer chunk candidates for OOM retry."""
    first_candidate = _normalize_chunk_size(chunk_size)
    candidates = [first_candidate, 256000, 128000, 64000, MIN_CHUNK_SIZE]
    if first_candidate is None:
        candidates[0] = None

    sizes = []
    for size in candidates:
        size = _normalize_chunk_size(size)
        if size not in sizes:
            sizes.append(size)
    return sizes or [None]


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")
_CONTENT_RE = re.compile(r'"Content"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _estimate_max_new_tokens(audio_sec, ceiling):
    """依音訊長度估算所需的 max_new_tokens,並以 ceiling(使用者設定值)為上限。

    先前固定 2048,對 600 秒分段或單次長音訊都太小,會把輸出截斷;改成隨長度放大。
    """
    if ceiling is None or ceiling <= 0:
        ceiling = DEFAULT_MAX_NEW_TOKENS
    est = int(audio_sec * EST_TOKENS_PER_SECOND) + 256
    return max(MIN_MAX_NEW_TOKENS, min(est, ceiling))


def _text_from_parsed(parsed):
    """從已解析的分段重建純文字(只取 Content),確保輸出不含模板標記或 JSON。"""
    if not parsed:
        return None
    parts = []
    for seg in parsed:
        if isinstance(seg, dict):
            content = seg.get("Content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
    return "".join(parts) if parts else None


def _sanitize_raw_text(raw):
    """最後手段:去除 <|im_start|> 等特殊標記;若仍是 JSON 形態,抽出 Content 拼成純文字。"""
    if not raw:
        return ""
    cleaned = _SPECIAL_TOKEN_RE.sub("", raw).strip()
    if '"Content"' in cleaned:
        contents = _CONTENT_RE.findall(cleaned)
        if contents:
            return "".join(
                c.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").strip()
                for c in contents
            )
    if cleaned.startswith("assistant"):
        cleaned = cleaned[len("assistant"):].lstrip()
    return cleaned.strip()


def _looks_unclean(text):
    """判斷 transcription_only 是否其實退化成原始解碼(含模板標記或裸 JSON)。"""
    if not text:
        return True
    stripped = text.lstrip()
    return "<|" in text or stripped.startswith("[{") or stripped.startswith("assistant")


def _decode_results(processor, generated_ids):
    """Decode defensively; short/truncated generations may not be valid JSON.

    text 一律回傳乾淨純文字:優先用 transcription_only,失敗或退化成原始 JSON 時,
    改以已解析分段的 Content 重建,再不行才回退到去標記後的原始文字。
    """
    raw = processor.decode(generated_ids)[0]
    try:
        parsed = processor.decode(generated_ids, return_format="parsed")[0]
    except Exception:
        parsed = None
    try:
        text = processor.decode(generated_ids, return_format="transcription_only")[0]
    except Exception:
        text = None
    if _looks_unclean(text):
        text = _text_from_parsed(parsed) or _sanitize_raw_text(raw)
    return raw, parsed, text


def transcribe(processor, model, audio, prompt=None, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, chunk_size=None, report=True):
    """對單段音訊執行辨識,回傳 raw / parsed / text 三種結果。"""
    import torch

    request = {"audio": audio}
    if prompt:
        request["prompt"] = prompt

    inputs = processor.apply_transcription_request(**request).to(model.device, model.dtype)

    audio_sec = len(audio) / TARGET_SR
    # max_new_tokens 視為上限,實際依音訊長度動態估算,避免長段被截斷又不浪費。
    effective_max_new = _estimate_max_new_tokens(audio_sec, max_new_tokens)
    base_gen_kwargs = {"max_new_tokens": effective_max_new, "do_sample": False}
    if GEN_REPETITION_PENALTY and GEN_REPETITION_PENALTY != 1.0:
        base_gen_kwargs["repetition_penalty"] = GEN_REPETITION_PENALTY

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.time()
    last_oom = None
    output_ids = None
    chunk_sizes = _chunk_retry_sizes(chunk_size)
    for attempt, current_chunk_size in enumerate(chunk_sizes, start=1):
        gen_kwargs = dict(base_gen_kwargs)
        if current_chunk_size:
            gen_kwargs["acoustic_tokenizer_chunk_size"] = current_chunk_size

        try:
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **gen_kwargs)
            break
        except Exception as exc:
            if not _is_cuda_oom(exc, torch):
                raise
            last_oom = exc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt == len(chunk_sizes):
                raise
            next_size = chunk_sizes[attempt]
            ui_warning(f"顯存不足(OOM),改用較小的 chunk-size={next_size} 重試...")

    if output_ids is None and last_oom:
        raise last_oom
    elapsed = time.time() - t0

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]

    raw, parsed, text = _decode_results(processor, generated_ids)

    rtf = elapsed / audio_sec if audio_sec > 0 else float("nan")
    if report:
        ui_success(f"辨識完成: 音訊 {format_duration(audio_sec)}, 推論 {format_duration(elapsed)}, RTF={rtf:.2f}")
    return {"raw": raw, "parsed": parsed, "text": text, "elapsed": elapsed, "audio_sec": audio_sec, "rtf": rtf}


def _offset_parsed_segments(parsed, offset_seconds):
    """Offset parsed timestamps from chunk-relative time to full-audio time."""
    if not parsed:
        return None

    adjusted = []
    for seg in parsed:
        if not isinstance(seg, dict):
            return None
        item = dict(seg)
        for key in ("Start", "End"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                item[key] = round(float(value) + offset_seconds, 2)
        adjusted.append(item)
    return adjusted


def transcribe_windowed(
    processor,
    model,
    audio,
    prompt=None,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    chunk_size=None,
    window_seconds=DEFAULT_WINDOW_SECONDS,
):
    """Split long audio into bounded windows to avoid long-context GPU OOM."""
    window_samples = int(window_seconds * TARGET_SR)
    if window_samples <= 0 or len(audio) <= window_samples:
        try:
            if USE_RICH:
                with fullscreen_progress("語音辨識") as progress:
                    task_id = progress.add_task(
                        "語音辨識",
                        total=1,
                        status=f"單段 {format_duration(len(audio) / TARGET_SR)}",
                        resource=collect_resource_stats(),
                    )
                    with ResourceMonitor(progress, task_id):
                        result = transcribe(
                            processor,
                            model,
                            audio,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            chunk_size=chunk_size,
                            report=False,
                        )
                    progress.update(task_id, advance=1, status=f"完成 RTF={result.get('rtf', float('nan')):.2f}", resource=collect_resource_stats())
                    return result
            return transcribe(
                processor,
                model,
                audio,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                chunk_size=chunk_size,
            )
        except Exception as exc:
            import torch

            can_split = (
                window_samples > MIN_WINDOW_SECONDS * TARGET_SR
                and len(audio) > MIN_WINDOW_SECONDS * TARGET_SR
                and _is_cuda_oom(exc, torch)
            )
            if not can_split:
                raise
            smaller_window = max(MIN_WINDOW_SECONDS, window_seconds / 2)
            ui_warning(f"此段仍發生 OOM,改用 {smaller_window:g}s 分段重試。")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return transcribe_windowed(
                processor,
                model,
                audio,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                chunk_size=chunk_size,
                window_seconds=smaller_window,
            )

    total_windows = (len(audio) + window_samples - 1) // window_samples
    ui_print(
        f"[bold cyan]音訊較長，啟用分段辨識[/bold cyan]: 每段 {window_seconds:g}s，"
        f"共 {total_windows} 段。若要先嘗試單次處理請加 --window-seconds 0"
    )

    results = []
    parsed_segments = []
    raw_parts = []
    text_parts = []

    def run_window(index, start, progress=None, task_id=None):
        end = min(start + window_samples, len(audio))
        start_sec = start / TARGET_SR
        end_sec = end / TARGET_SR
        status = f"{index}/{total_windows}  {format_duration(start_sec)} - {format_duration(end_sec)}"
        if progress is not None:
            progress.update(task_id, status=status)
        else:
            ui_print(f"\n[bold]分段 {index}/{total_windows}[/bold] {start_sec:.1f}s - {end_sec:.1f}s")
        try:
            result = transcribe(
                processor,
                model,
                audio[start:end],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                chunk_size=chunk_size,
                report=progress is None,
            )
        except Exception as exc:
            import torch

            smaller_window = max(MIN_WINDOW_SECONDS, window_seconds / 2)
            if not _is_cuda_oom(exc, torch) or smaller_window >= window_seconds:
                raise
            ui_warning(f"分段 {index} 發生 OOM,改用 {smaller_window:g}s 子分段重試。")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = transcribe_windowed(
                processor,
                model,
                audio[start:end],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                chunk_size=chunk_size,
                window_seconds=smaller_window,
            )
        if progress is not None:
            progress.update(task_id, advance=1, status=f"完成 {index}/{total_windows}  RTF={result.get('rtf', float('nan')):.2f}")
        results.append(result)
        if result["raw"]:
            raw_parts.append(result["raw"].strip())
        if result["text"]:
            text_parts.append(result["text"].strip())
        adjusted = _offset_parsed_segments(result["parsed"], start_sec)
        if adjusted:
            parsed_segments.extend(adjusted)

    if USE_RICH:
        with fullscreen_progress("分段辨識", subtitle=f"每段 {window_seconds:g}s · 共 {total_windows} 段") as progress:
            task_id = progress.add_task("分段辨識", total=total_windows, status="準備中", resource=collect_resource_stats())
            with ResourceMonitor(progress, task_id):
                for index, start in enumerate(range(0, len(audio), window_samples), start=1):
                    run_window(index, start, progress, task_id)
    else:
        for index, start in enumerate(range(0, len(audio), window_samples), start=1):
            run_window(index, start)

    return {
        "raw": "\n".join(raw_parts),
        "parsed": parsed_segments or None,
        "text": " ".join(part for part in text_parts if part),
        "chunks": results,
    }


# --------------------------------------------------------------------------- #
# 簡轉繁(OpenCC s2twp:簡體 → 台灣正體並轉台灣慣用詞)
# --------------------------------------------------------------------------- #
_opencc_converter = None
_opencc_failed = False


def to_traditional(text):
    """把簡體字轉為繁體(台灣慣用詞)。OpenCC 不可用時原樣回傳。"""
    global _opencc_converter, _opencc_failed
    if not text or _opencc_failed:
        return text
    if _opencc_converter is None:
        try:
            import opencc

            _opencc_converter = opencc.OpenCC("s2twp")
        except Exception as exc:
            _opencc_failed = True
            ui_warning(f"無法載入 OpenCC,略過簡轉繁:{exc}")
            return text
    try:
        return _opencc_converter.convert(text)
    except Exception:
        return text


def convert_result_to_traditional(result):
    """就地把辨識結果的文字與分段內容轉成繁體。"""
    if not result:
        return result
    if result.get("text"):
        result["text"] = to_traditional(result["text"])
    parsed = result.get("parsed")
    if parsed:
        for seg in parsed:
            if isinstance(seg, dict) and isinstance(seg.get("Content"), str):
                seg["Content"] = to_traditional(seg["Content"])
    return result


# --------------------------------------------------------------------------- #
# 結果輸出
# --------------------------------------------------------------------------- #
def write_transcript(result, output_path, source_label, with_timestamps=False):
    """把辨識結果寫入 .txt 檔。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = [result["text"].strip()]

    if with_timestamps and result["parsed"]:
        lines.append("")
        lines.append("=" * 60)
        lines.append("語者 / 時間戳記")
        lines.append("=" * 60)
        for seg in result["parsed"]:
            start = seg.get("Start", "?")
            end = seg.get("End", "?")
            speaker = seg.get("Speaker", "?")
            content = seg.get("Content", "")
            lines.append(f"[{start:>7} - {end:>7}] 語者{speaker}:{content}")

    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    ui_success(f"已輸出辨識結果: {_display_path(output_path)}")


def default_output_path(source_label):
    """依時間戳記與來源名稱產生預設輸出檔名(時間戳在前,確保依檔名排序即為時間順序)。"""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c for c in source_label if c.isalnum() or c in ("-", "_")) or "audio"
    return os.path.join(OUTPUT_DIR, f"{stamp}_{safe}.txt")


# --------------------------------------------------------------------------- #
# 互動式介面:設定、鍵盤選單、操作流程
# --------------------------------------------------------------------------- #
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus")


@dataclass
class Settings:
    """執行期可即時調整的設定;不必重啟程式。"""

    quant: str = "4bit"
    with_timestamps: bool = False
    prompt: str = None
    window_seconds: float = 0.0  # 0 表示單次優先(對齊模型 60 分鐘單次設計),OOM 才退回分段
    chunk_size: int = DEFAULT_CHUNK_SIZE
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS  # 視為上限,實際依音訊長度動態估算
    preview_chars: int = 1200
    keep_recording: bool = True
    device: int = None


def _interactive():
    """是否可用方向鍵即時選單(需 Windows 終端機 + Rich + TTY)。"""
    return (
        msvcrt is not None
        and USE_RICH
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )


def read_key():
    """讀取單一按鍵,回傳標準化代碼:UP/DOWN/ENTER/ESC/BACKSPACE 或原始字元。"""
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):  # 方向鍵等特殊鍵會先回傳前綴,再讀一次取得實際鍵碼
        nxt = msvcrt.getwch()
        return {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}.get(nxt, "")
    if ch in ("\r", "\n"):
        return "ENTER"
    if ch == "\x1b":
        return "ESC"
    if ch == "\x03":  # Ctrl+C
        raise KeyboardInterrupt
    if ch == "\x08":
        return "BACKSPACE"
    return ch


def pause(message="按任意鍵返回..."):
    ui_print(f"[dim]{message}[/dim]")
    if _interactive():
        read_key()
    else:
        try:
            input()
        except EOFError:
            pass


def _render_menu(title, items, index, subtitle):
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", no_wrap=True)
    table.add_column()
    for i, item in enumerate(items):
        label = item[0] if isinstance(item, tuple) else item
        hint = item[1] if isinstance(item, tuple) and len(item) > 1 else ""
        selected = i == index
        marker = "[bold cyan]›[/bold cyan]" if selected else " "
        number = f"[cyan]{i + 1}[/cyan]" if selected else f"[dim]{i + 1}[/dim]"
        text = f"[bold reverse] {label} [/bold reverse]" if selected else f" {label} "
        if hint:
            text += f"  [dim]{hint}[/dim]"
        table.add_row(f"{marker} {number}", text)
    return Panel(
        table,
        title=f"[bold]{title}[/bold]",
        subtitle=f"[dim]{subtitle}[/dim]" if subtitle else None,
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def select_menu(title, items, subtitle="↑↓ 移動 · Enter 確認 · 數字快選 · Esc 返回", start=0, header=None):
    """方向鍵 / 數字鍵選單。回傳選項索引;按 Esc 或選「返回」回傳 None。

    以替身畫面(screen=True)整頁重繪,縮放視窗時不會殘留亂碼。
    header 為可選的上方 renderable(例如主選單的標題面板)。
    """
    n = len(items)
    if not _interactive():
        if header is not None and USE_RICH:
            console.print(header)
        ui_rule(title)
        for i, item in enumerate(items):
            label = item[0] if isinstance(item, tuple) else item
            ui_print(f"  {i + 1}) {label}")
        raw = input("輸入編號(直接 Enter 返回): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw) - 1
        return None

    state = {"index": max(0, min(start, n - 1))}

    def render():
        menu = _render_menu(title, items, state["index"], subtitle)
        return Group(header, menu) if header is not None else menu

    with Live(
        get_renderable=render,
        console=console,
        screen=True,
        auto_refresh=True,
        refresh_per_second=10,
        vertical_overflow="crop",
    ):
        while True:
            key = read_key()
            if key == "UP":
                state["index"] = (state["index"] - 1) % n
            elif key == "DOWN":
                state["index"] = (state["index"] + 1) % n
            elif key == "ENTER":
                return state["index"]
            elif key == "ESC":
                return None
            elif key.isdigit() and key != "0" and int(key) <= n:
                return int(key) - 1


def ask_text(prompt, default=None):
    if USE_RICH:
        return Prompt.ask(prompt, default="" if default is None else default)
    raw = input(f"{prompt}: ").strip()
    return raw or (default or "")


def _ask_int(prompt, current):
    raw = ask_text(prompt, default=str(current)).strip()
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        ui_warning("輸入無效,維持原值。")
        return current


def _ask_float(prompt, current):
    raw = ask_text(prompt, default=str(current)).strip()
    try:
        return float(raw)
    except (TypeError, ValueError):
        ui_warning("輸入無效,維持原值。")
        return current


def _fmt_mtime(path):
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%m-%d %H:%M")
    except OSError:
        return ""


def _list_audio(folder, limit=8):
    if not os.path.isdir(folder):
        return []
    files = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(AUDIO_EXTS)
    ]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[:limit]


def _open_folder(path):
    try:
        os.startfile(path)  # Windows 專用,開啟檔案總管
    except Exception as exc:
        ui_warning(f"無法開啟資料夾:{exc}")


def _copy_text(text):
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", "$input | Set-Clipboard"],
            input=text,
            text=True,
            encoding="utf-8",
            check=True,
        )
        ui_success("已複製辨識結果到剪貼簿。")
    except Exception as exc:
        ui_warning(f"複製失敗:{exc}")


# --------------------------------------------------------------------------- #
# 模型快取:整個工作階段只載入一次,量化模式變更時才重載
# --------------------------------------------------------------------------- #
def resolve_model_path():
    """回傳可用的模型路徑;本地不存在時詢問是否下載。失敗回傳 None。"""
    if os.path.isdir(MODEL_DIR):
        return MODEL_DIR
    if prompt_yes_no(
        f"找不到本地模型 {_display_path(MODEL_DIR)},是否下載 {MODEL_ID}?",
        default=True,
    ):
        return download_model(MODEL_ID)
    ui_error("未取得模型,無法辨識。")
    return None


def _unload_model(state):
    if state.get("model") is None:
        return
    state["model"] = None
    state["processor"] = None
    state["quant"] = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def ensure_model(settings, state):
    """確保已載入符合目前量化設定的模型;必要時重載。回傳 (processor, model)。"""
    if state.get("model") is not None and state.get("quant") == settings.quant:
        return state["processor"], state["model"]
    if state.get("model") is not None:
        ui_print("[dim]量化模式已變更,重新載入模型...[/dim]")
        _unload_model(state)

    model_path = resolve_model_path()
    if not model_path:
        return None, None
    processor, model = build_model(model_path, settings.quant)
    state.update(processor=processor, model=model, quant=settings.quant)
    return processor, model


# --------------------------------------------------------------------------- #
# 辨識流程
# --------------------------------------------------------------------------- #
def print_run_config(settings, source_label, audio, output_path):
    audio_sec = len(audio) / TARGET_SR
    if not USE_RICH:
        ui_print(f"來源: {source_label}  長度: {format_duration(audio_sec)}  輸出: {_display_path(output_path)}")
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("來源", source_label)
    table.add_row("音訊長度", format_duration(audio_sec))
    table.add_row("量化模式", settings.quant)
    table.add_row("時間戳記", "開" if settings.with_timestamps else "關")
    if settings.prompt:
        table.add_row("熱詞", settings.prompt)
    table.add_row("分段", "單次優先" if settings.window_seconds <= 0 else f"{settings.window_seconds:g}s")
    table.add_row("輸出", _display_path(output_path))
    console.print(Panel(table, title="[bold]本次設定[/bold]", border_style="cyan", box=box.ROUNDED, padding=(1, 2)))


def run_transcription(settings, state, audio, source_label):
    processor, model = ensure_model(settings, state)
    if model is None:
        pause()
        return

    output_path = default_output_path(source_label)
    print_run_config(settings, source_label, audio, output_path)

    try:
        result = transcribe_windowed(
            processor,
            model,
            audio,
            prompt=settings.prompt or None,
            max_new_tokens=settings.max_new_tokens,
            chunk_size=settings.chunk_size,
            window_seconds=settings.window_seconds,
        )
    except Exception as exc:
        import torch

        audio_sec = len(audio) / TARGET_SR
        if (
            _is_cuda_oom(exc, torch)
            and settings.window_seconds <= 0
            and audio_sec > MIN_WINDOW_SECONDS
        ):
            ui_warning(f"單次處理發生 OOM,自動改用 {DEFAULT_WINDOW_SECONDS:g}s 分段重試。")
            try:
                result = transcribe_windowed(
                    processor,
                    model,
                    audio,
                    prompt=settings.prompt or None,
                    max_new_tokens=settings.max_new_tokens,
                    chunk_size=settings.chunk_size,
                    window_seconds=DEFAULT_WINDOW_SECONDS,
                )
            except Exception as exc2:
                ui_error(f"辨識失敗:{exc2}")
                pause()
                return
        else:
            ui_error(f"辨識失敗:{exc}")
            pause()
            return

    # 辨識結果自動簡轉繁(台灣慣用詞),預覽與輸出檔都套用。
    with ui_status("[cyan]簡轉繁轉換中...[/cyan]"):
        convert_result_to_traditional(result)

    show_result(settings, result, output_path, source_label)


def show_result(settings, result, output_path, source_label):
    ui_rule("辨識結果")
    preview = result["text"].strip()
    if settings.preview_chars != 0:
        if settings.preview_chars > 0 and len(preview) > settings.preview_chars:
            preview = preview[: settings.preview_chars].rstrip() + "\n\n[dim]...(畫面預覽已截斷,完整內容見輸出檔)[/dim]"
        ui_panel("結果預覽", preview or "(空)", border_style="green")

    write_transcript(result, output_path, source_label, with_timestamps=settings.with_timestamps)

    action = select_menu(
        "接下來",
        [
            ("返回主選單", ""),
            ("開啟輸出資料夾", _display_path(OUTPUT_DIR)),
            ("複製辨識結果", ""),
        ],
        subtitle="↑↓ 選擇 · Enter 確認",
    )
    if action == 1:
        _open_folder(OUTPUT_DIR)
    elif action == 2:
        _copy_text(result["text"].strip())


# --------------------------------------------------------------------------- #
# 輸入流程:音訊檔 / 麥克風
# --------------------------------------------------------------------------- #
def choose_audio_file():
    """選擇音訊檔:可手動輸入/拖放路徑,或從 recordings/ 快速挑選。"""
    recents = _list_audio(RECORDING_DIR)
    items = [("輸入或拖放檔案路徑", "")]
    for path in recents:
        items.append((os.path.basename(path), _fmt_mtime(path)))
    items.append(("返回", ""))

    idx = select_menu("選擇音訊檔", items)
    if idx is None or idx == len(items) - 1:
        return None
    if idx == 0:
        raw = ask_text("音訊檔路徑(可直接把檔案拖進視窗)")
        return raw.strip().strip('"').strip() or None
    return recents[idx - 1]


def flow_file(settings, state):
    path = choose_audio_file()
    if not path:
        return
    ui_print(f"[bold cyan]載入音訊檔[/bold cyan] {path}")
    try:
        with ui_status("[cyan]讀取與重採樣中...[/cyan]"):
            audio = load_audio_file(path)
    except Exception as exc:
        ui_error(f"載入失敗:{exc}")
        pause()
        return
    if audio.size == 0:
        ui_error("音訊內容為空。")
        pause()
        return
    source_label = os.path.splitext(os.path.basename(path))[0]
    run_transcription(settings, state, audio, source_label)


def flow_record(settings, state):
    mode = select_menu(
        "麥克風錄音",
        [
            ("按 Enter 起訖", "不限長度,適合長段談話"),
            ("錄固定秒數", ""),
            ("返回", ""),
        ],
    )
    if mode is None or mode == 2:
        return

    try:
        if mode == 0:
            audio = record_until_enter(device=settings.device)
        else:
            seconds = _ask_float("錄音秒數", 10)
            audio = record_fixed(seconds, device=settings.device)
    except Exception as exc:
        ui_error(f"錄音失敗:{exc}")
        pause()
        return

    if audio.size == 0:
        ui_error("沒有錄到任何音訊。")
        pause()
        return

    if settings.keep_recording:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_wav(os.path.join(RECORDING_DIR, f"recording_{stamp}.wav"), audio)

    run_transcription(settings, state, audio, "recording")


# --------------------------------------------------------------------------- #
# 設定與裝置
# --------------------------------------------------------------------------- #
def settings_menu(settings):
    quant_order = ["4bit", "8bit", "none"]
    while True:
        items = [
            ("量化模式", f"{settings.quant}  (循環切換)"),
            ("附加語者/時間戳記", "開" if settings.with_timestamps else "關"),
            ("熱詞 / 提示", settings.prompt or "(無)"),
            ("分段秒數", "單次優先" if settings.window_seconds <= 0 else f"{settings.window_seconds:g}s"),
            ("chunk-size", str(settings.chunk_size)),
            ("產生上限 max-new-tokens", str(settings.max_new_tokens)),
            ("畫面預覽字數", "不顯示" if settings.preview_chars == 0 else str(settings.preview_chars)),
            ("保留麥克風錄音檔", "是" if settings.keep_recording else "否"),
            ("返回", ""),
        ]
        idx = select_menu("設定", items, subtitle="↑↓ 選擇 · Enter 修改 · Esc 返回")
        if idx is None or idx == len(items) - 1:
            return
        if idx == 0:
            settings.quant = quant_order[(quant_order.index(settings.quant) + 1) % len(quant_order)]
        elif idx == 1:
            settings.with_timestamps = not settings.with_timestamps
        elif idx == 2:
            text = ask_text("熱詞/提示(留空表示無)", default=settings.prompt or "").strip()
            settings.prompt = text or None
        elif idx == 3:
            settings.window_seconds = _ask_float("分段秒數(0 表示先嘗試單次)", settings.window_seconds)
        elif idx == 4:
            settings.chunk_size = _normalize_chunk_size(_ask_int("chunk-size", settings.chunk_size)) or DEFAULT_CHUNK_SIZE
        elif idx == 5:
            settings.max_new_tokens = _ask_int("max-new-tokens", settings.max_new_tokens)
        elif idx == 6:
            settings.preview_chars = _ask_int("畫面預覽字數(0 表示不顯示)", settings.preview_chars)
        elif idx == 7:
            settings.keep_recording = not settings.keep_recording


def devices_menu(settings):
    try:
        import sounddevice as sd

        devices = [
            (i, d)
            for i, d in enumerate(sd.query_devices())
            if d.get("max_input_channels", 0) > 0
        ]
    except Exception as exc:
        ui_error(f"無法列出音訊裝置:{exc}")
        pause()
        return

    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None

    items = [("使用系統預設裝置", "目前" if settings.device is None else "")]
    for i, dev in devices:
        tags = []
        if i == default_in:
            tags.append("系統預設")
        if settings.device == i:
            tags.append("目前")
        items.append((f"[{i}] {dev['name']}", "  ".join(tags)))
    items.append(("返回", ""))

    idx = select_menu("麥克風輸入裝置", items, subtitle="Enter 設為錄音裝置 · Esc 返回")
    if idx is None or idx == len(items) - 1:
        return
    settings.device = None if idx == 0 else devices[idx - 1][0]
    ui_success(f"已選擇錄音裝置:{'系統預設' if settings.device is None else settings.device}")
    pause()


# --------------------------------------------------------------------------- #
# 主程式
# --------------------------------------------------------------------------- #
def main():
    settings = Settings()
    state = {}

    if not _interactive():
        ui_warning("目前環境不支援即時方向鍵選單,已改用文字輸入模式。")

    if not USE_RICH:
        print_app_header()

    try:
        while True:
            choice = select_menu(
                "VibeVoice ASR",
                [
                    ("辨識音訊檔", "wav / mp3 / flac / m4a ..."),
                    ("麥克風錄音辨識", ""),
                    ("設定", "量化 / 時間戳記 / 熱詞 ..."),
                    ("麥克風輸入裝置", ""),
                    ("離開", ""),
                ],
                subtitle="↑↓ 移動 · Enter 確認 · 數字快選 · Esc 離開",
                header=_app_header_renderable() if USE_RICH else None,
            )
            if choice is None or choice == 4:
                break
            if choice == 0:
                flow_file(settings, state)
            elif choice == 1:
                flow_record(settings, state)
            elif choice == 2:
                settings_menu(settings)
            elif choice == 3:
                devices_menu(settings)
    except KeyboardInterrupt:
        ui_print("\n[dim]已中斷。[/dim]")
    finally:
        _unload_model(state)

    ui_print("[dim]再見。[/dim]")


if __name__ == "__main__":
    main()


