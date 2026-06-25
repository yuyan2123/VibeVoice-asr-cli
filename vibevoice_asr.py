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
    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress
    from rich.prompt import Confirm, Prompt
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except Exception:
    box = None
    Align = None
    Console = None
    Group = None
    Live = None
    Panel = None
    Progress = None
    Spinner = None
    Confirm = None
    Prompt = None
    Table = None
    Text = None
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
    # 僅作為進度 / 已用 / 預估剩餘的狀態追蹤;實際畫面由 fullscreen_progress
    # 以自訂雙欄儀表板繪製,不使用 Progress 內建欄位渲染。
    return Progress(console=console)


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


def _pressure_style(value):
    """記憶體壓力配色:平時 dim 不搶眼,偏高轉黃,接近滿載轉紅(語意色只給示警)。"""
    if value is None:
        return "grey50"
    if value >= 85:
        return "red"
    if value >= 60:
        return "yellow"
    return "grey70"


def _fmt_clock(seconds):
    """秒數轉 MM:SS(超過一小時轉 H:MM:SS);無值或非數時回傳佔位。"""
    if seconds is None:
        return "--:--"
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "--:--"
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "--:--"
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _progress_bar_text(percentage, width):
    """主進度條:面板上唯一的強調色塊,實心粗筆畫。"""
    completed = max(0.0, min(100.0, float(percentage or 0)))
    n = max(0, min(width, int(round(completed / 100 * width))))
    return (
        f"[grey42]▕[/grey42][bright_cyan]{'█' * n}[/bright_cyan]"
        f"[grey30]{'░' * (width - n)}[/grey30][grey42]▏[/grey42]"
    )


def _resource_line(stats):
    """資源使用率壓成一行 dim 文字(次要資訊;只有記憶體吃緊才以黃/紅示警)。"""

    def cell(label, value, pressure=False):
        if value is None:
            return f"[grey42]{label} --[/grey42]"
        style = _pressure_style(value) if pressure else "grey70"
        return f"[grey50]{label}[/grey50] [{style}]{value:.0f}%[/{style}]"

    line = "   ".join(
        [
            cell("CPU", stats.get("cpu")),
            cell("RAM", stats.get("ram"), pressure=True),
            cell("GPU", stats.get("gpu")),
            cell("VRAM", stats.get("vram"), pressure=True),
        ]
    )
    vram_label = stats.get("vram_label")
    if vram_label:
        line += f"   [grey42]{vram_label}[/grey42]"
    return line


def render_dashboard(task, spinner, width=52):
    """單欄、有階層的進度面板:進度條與轉錄文字為主角,時間/資源退為次要 dim 資訊。

    視覺階層:強調色只給「進度條」與「目前語者」;狀態用語意色(完成綠、記憶體吃緊黃/紅);
    其餘時間、資源一律 dim。支援辨識任務(kind=asr)與下載任務(kind=download)。
    """
    fields = task.fields
    kind = fields.get("kind", "asr")
    stats = fields.get("resource") if isinstance(fields.get("resource"), dict) else None
    pct = task.percentage or 0.0
    elapsed = _fmt_clock(task.elapsed)
    remaining = task.time_remaining
    stage = fields.get("stage") or ("下載中" if kind == "download" else "辨識中")

    # 1) 標題列:spinner + 狀態(語意色)+ 段數 ……右側 RTF(次要)
    if stage == "完成":
        status = "[bold green]● 完成[/bold green]"
    else:
        status = f"[bold bright_cyan]{stage}[/bold bright_cyan]"
    if kind != "download" and fields.get("seg"):
        status += f"  [grey58]段 {fields['seg']}[/grey58]"
    rtf = fields.get("rtf")
    rtf_text = f"[grey58]RTF {rtf:.2f}[/grey58]" if isinstance(rtf, (int, float)) and rtf == rtf else ""
    head = Table.grid(expand=True)
    head.add_column(no_wrap=True)
    head.add_column(ratio=1)
    head.add_column(justify="right", no_wrap=True)
    head.add_row(spinner, " " + status, rtf_text)

    blocks = [head]

    # 2) 主進度條(面板唯一強調)+ 百分比
    bar_w = max(12, width - 7)
    blocks.append(_progress_bar_text(pct, bar_w) + f" [bold]{pct:>3.0f}%[/bold]")

    # 3) 次要資訊列(時間 / 檔案;dim)
    if kind == "download":
        meta = f"檔案 {fields.get('status') or '—'}   已用 {elapsed}"
    else:
        rem = f"約 {_fmt_clock(remaining)}" if remaining else "—"
        parts = [p for p in (fields.get("span"),) if p]
        parts.append(f"已用 {elapsed}")
        parts.append(f"剩餘 {rem}")
        meta = "   ".join(parts)
    blocks.append(Text(meta, style="grey58", no_wrap=True, overflow="ellipsis"))

    # 4) 即時轉錄(主要前景):語者為強調,內容為前景白;最多兩句
    stream_text = fields.get("stream")
    if kind != "download" and stage == "辨識中" and stream_text:
        spk = fields.get("stream_speaker")
        blocks.append("")
        if spk:
            blocks.append(Text(f"語者{spk}", style="bold bright_cyan"))
        for ln in stream_text.split("\n"):
            blocks.append(Text(ln, style="white"))

    # 5) 資源(次要,單行 dim)
    if stats is not None:
        blocks.append("")
        blocks.append(_resource_line(stats))

    return Group(*blocks)


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
    spinner = Spinner("dots", style="cyan")  # 持久實例,跨刷新才會持續轉動

    def render():
        panel_w = min(max(console.width - 4, 40), 62)
        tasks = progress.tasks
        if tasks:
            content = render_dashboard(tasks[0], spinner, panel_w - 6)
        else:
            placeholder = Table.grid(padding=(0, 1))
            placeholder.add_column(no_wrap=True)
            placeholder.add_column(no_wrap=True)
            placeholder.add_row(spinner, "[dim]準備中...[/dim]")
            content = placeholder
        body = [
            Align.center(
                Panel(
                    content,
                    title=f"[bold]{title}[/bold]",
                    subtitle=f"[dim]{subtitle}[/dim]" if subtitle else None,
                    border_style="cyan",
                    box=box.ROUNDED,
                    padding=(1, 2),
                    width=panel_w,
                )
            )
        ]
        if log:
            body.append(
                Align.center(
                    Panel(
                        Group(*log[-8:]),
                        title="[bold]訊息[/bold]",
                        border_style="grey42",
                        box=box.ROUNDED,
                        padding=(0, 1),
                        width=panel_w,
                    )
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
                kind="download",
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

    # 只對文字骨幹(Qwen2)啟用 sdpa;聲學/語意 tokenizer 是卷積、不支援 sdpa。
    # 必須用 dict 精準指定子設定:若傳字串 "sdpa",transformers 會試圖把它套到「所有」
    # 子模型,卷積 tokenizer 不支援就拋 ValueError,導致整個退回 eager。
    # eager 會對長序列具現化 O(N^2) 的注意力矩陣(30 分鐘音訊約 13700 tokens、注意力矩陣
    # 可達數 GB),長音訊必爆顯存;sdpa 走記憶體高效 kernel(O(N)),是長音訊不 OOM 的關鍵。
    load_kwargs["attn_implementation"] = {"text_config": "sdpa"}

    ui_print(f"[bold cyan]載入模型[/bold cyan] {model_path}  [dim]量化模式: {quant}[/dim]")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_path)
    active_attn = "sdpa(text)"
    try:
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    except (ValueError, TypeError) as exc:
        ui_warning(f"sdpa 注意力設定失敗({exc}),改用預設(eager)注意力。")
        load_kwargs.pop("attn_implementation", None)
        active_attn = "eager(預設)"
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
# 即時預覽用:依句末標點切句(逗號/頓號不算句末),最後一段允許是未收尾的半句。
_SENT_RE = re.compile(r"[^。．.!?！？\n]*[。．.!?！？\n]|[^。．.!?！？\n]+")


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


def _make_json_array_stopper(tokenizer, prompt_len):
    """建立「頂層 JSON 轉錄陣列一收尾就停止生成」的停止條件。

    模型在 4-bit 下,轉完該段 JSON 後常不穩定吐出結束符;在較高的 max_new_tokens 下會
    一路空吐到接近上限(多吐的內容會被解析器丟掉,卻白白浪費大量時間)。此停止條件以
    JSON 中括號深度追蹤,在頂層陣列收尾(深度歸零)時立即停止;字串內的 '['/']'
    (例如 "[Noise]")以 in_str 狀態正確忽略。若模型始終未收尾,則維持原本 max_new_tokens
    上限——屬純粹的提早停止,不會截斷或破壞正確性。狀態化,故每次 generate 都要重建。
    """
    from transformers import StoppingCriteria

    class _JsonArrayStopper(StoppingCriteria):
        def __init__(self):
            self.depth = 0
            self.in_str = False
            self.esc = False
            self.started = False
            self.done = False
            self.scanned = 0

        def __call__(self, input_ids, scores=None, **kwargs):
            import torch

            gen = input_ids[0, prompt_len:]
            new = gen[self.scanned:]
            if new.numel():
                for ch in tokenizer.decode(new, skip_special_tokens=False):
                    if self.in_str:
                        if self.esc:
                            self.esc = False
                        elif ch == "\\":
                            self.esc = True
                        elif ch == '"':
                            self.in_str = False
                        continue
                    if ch == '"':
                        self.in_str = True
                    elif ch == "[":
                        self.depth += 1
                        self.started = True
                    elif ch == "]":
                        self.depth -= 1
                        if self.started and self.depth == 0:
                            self.done = True
                self.scanned = gen.numel()
            return torch.tensor([self.done], device=input_ids.device)

    return _JsonArrayStopper()


class _StreamingDecoder:
    """逐步把新 token 解碼成乾淨文字(UTF-8 安全),並抽出目前正在生成的 Content。

    模型輸出是帶語者/時間戳的 JSON 陣列;此處以字元狀態機只取出 Content 值,
    串成「目前已辨識內容」,並回傳尾端約一句話寬度餵給顯示回呼(節流避免過度刷新)。
    """

    def __init__(self, tokenizer, on_text, max_sentences=2, line_chars=40,
                 keep_chars=400, min_interval=0.1):
        self.tok = tokenizer
        self.on_text = on_text
        self.max_sentences = max_sentences   # 畫面最多保留幾句(滿了清掉最舊的)
        self.line_chars = line_chars         # 單句過長時保留的尾端字數
        self.keep_chars = keep_chars         # 僅保留最近這麼多字,足夠切出末兩句即可
        self.min_interval = min_interval
        self.tok_scanned = 0
        self.cache = []          # 尚未 flush 的 token(可能含半個多位元組字)
        # JSON 內容抽取狀態
        self.in_str = False
        self.esc = False
        self.value_mode = False
        self.capturing = False
        self.last_key = None
        self.key_candidate = None
        self.cur = []            # 目前字串的字元
        self.tail = ""           # 已完成 Content 的有界尾段(只留最近 keep_chars 字)
        # 語者:依出現順序對應成 1、2…(與輸出檔的對話稿一致)
        self.pending_speaker = None
        self.current_speaker = None
        self.speaker_map = {}
        self._dirty = False
        self._last_emit = 0.0

    def _feed_char(self, ch):
        if self.in_str:
            if self.esc:
                self.esc = False
                self.cur.append(ch)
            elif ch == "\\":
                self.esc = True
            elif ch == '"':
                self.in_str = False
                s = "".join(self.cur)
                if self.value_mode:
                    if self.capturing:
                        self.tail = (self.tail + s)[-self.keep_chars :]
                        self.capturing = False
                        self._dirty = True
                    elif self.last_key == "Speaker":
                        self.pending_speaker = s
                    self.last_key = None
                    self.value_mode = False
                else:
                    self.key_candidate = s
            else:
                self.cur.append(ch)
                if self.capturing:
                    self._dirty = True
            return
        if ch == '"':
            self.in_str = True
            self.cur = []
            self.value_mode = self.last_key is not None
            self.capturing = self.value_mode and self.last_key == "Content"
            if self.capturing and self.pending_speaker is not None:
                if self.pending_speaker not in self.speaker_map:
                    self.speaker_map[self.pending_speaker] = len(self.speaker_map) + 1
                self.current_speaker = self.speaker_map[self.pending_speaker]
        elif ch == ":":
            self.last_key = self.key_candidate
        elif ch == "{":
            self.last_key = None
            self.key_candidate = None
            self.pending_speaker = None
        elif ch in ",}[]":
            self.last_key = None
            self.key_candidate = None

    def _current_text(self):
        # 只取有界尾段 + 目前正在生成的半句,切句後保留最近 max_sentences 句。
        partial = "".join(self.cur[-self.keep_chars :]) if self.capturing else ""
        text = (self.tail + partial)[-self.keep_chars :].strip()
        if not text:
            return ""
        sentences = [s.strip() for s in _SENT_RE.findall(text) if s.strip()]
        lines = []
        for s in sentences[-self.max_sentences :]:
            if len(s) > self.line_chars:
                s = "…" + s[-self.line_chars :]
            lines.append(s)
        return "\n".join(lines)

    def _maybe_emit(self):
        if not self._dirty:
            return
        now = time.time()
        if now - self._last_emit < self.min_interval:
            return
        self._last_emit = now
        self._dirty = False
        try:
            self.on_text(self._current_text(), self.current_speaker)
        except Exception:
            pass

    def push(self, gen_ids):
        total = gen_ids.shape[0] if hasattr(gen_ids, "shape") else len(gen_ids)
        if total <= self.tok_scanned:
            return
        new = gen_ids[self.tok_scanned :]
        self.tok_scanned = total
        new_list = new.tolist() if hasattr(new, "tolist") else list(new)
        self.cache.extend(new_list)
        decoded = self.tok.decode(self.cache, skip_special_tokens=False)
        if decoded.endswith("�"):
            return  # 半個多位元組字,等下一步補齊再 flush
        self.cache = []
        for ch in decoded:
            self._feed_char(ch)
        self._maybe_emit()


def _make_stream_criteria(tokenizer, prompt_len, on_text):
    """只負責即時顯示、永不停止的 StoppingCriteria(與原停止條件並存,不影響其行為)。"""
    from transformers import StoppingCriteria
    import torch

    decoder = _StreamingDecoder(tokenizer, on_text)

    class _StreamCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores=None, **kwargs):
            try:
                decoder.push(input_ids[0, prompt_len:])
            except Exception:
                pass
            return torch.tensor([False], device=input_ids.device)

    return _StreamCriteria()


def transcribe(processor, model, audio, prompt=None, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, chunk_size=None, report=True, stream_callback=None):
    """對單段音訊執行辨識,回傳 raw / parsed / text 三種結果。"""
    import torch

    request = {"audio": audio}
    if prompt:
        request["prompt"] = prompt

    inputs = processor.apply_transcription_request(**request).to(model.device, model.dtype)
    prompt_len = inputs["input_ids"].shape[1]

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
    from transformers import StoppingCriteriaList

    for attempt, current_chunk_size in enumerate(chunk_sizes, start=1):
        gen_kwargs = dict(base_gen_kwargs)
        if current_chunk_size:
            gen_kwargs["acoustic_tokenizer_chunk_size"] = current_chunk_size
        # 轉錄陣列收尾即停,避免高上限下空吐浪費時間;停止條件有狀態,每次嘗試都重建。
        _criteria = [_make_json_array_stopper(processor.tokenizer, prompt_len)]
        # 另掛一個「只顯示、永不停止」的條件做即時逐字預覽,不影響上面的停止行為。
        if stream_callback is not None:
            _criteria.append(_make_stream_criteria(processor.tokenizer, prompt_len, stream_callback))
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(_criteria)

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
    stream=False,
):
    """Split long audio into bounded windows to avoid long-context GPU OOM."""
    window_samples = int(window_seconds * TARGET_SR)
    if window_samples <= 0 or len(audio) <= window_samples:
        try:
            if USE_RICH:
                with fullscreen_progress("語音辨識") as progress:
                    dur = len(audio) / TARGET_SR
                    task_id = progress.add_task(
                        "語音辨識",
                        total=1,
                        kind="asr",
                        seg=None,
                        span=f"{_fmt_clock(0)} – {_fmt_clock(dur)}",
                        rtf=None,
                        stage="辨識中",
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
                            stream_callback=(lambda t, spk: progress.update(task_id, stream=t, stream_speaker=spk)) if stream else None,
                        )
                    progress.update(
                        task_id,
                        advance=1,
                        stage="完成",
                        stream="",
                        stream_speaker=None,
                        rtf=result.get("rtf", float("nan")),
                        resource=collect_resource_stats(),
                    )
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
                stream=stream,
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
        if progress is not None:
            progress.update(
                task_id,
                stage="辨識中",
                seg=f"{index}/{total_windows}",
                span=f"{_fmt_clock(start_sec)} – {_fmt_clock(end_sec)}",
                stream="",
                stream_speaker=None,
            )
        else:
            ui_print(f"\n[bold]分段 {index}/{total_windows}[/bold] {start_sec:.1f}s - {end_sec:.1f}s")
        try:
            sink = (lambda t, spk: progress.update(task_id, stream=t, stream_speaker=spk)) if (stream and progress is not None) else None
            result = transcribe(
                processor,
                model,
                audio[start:end],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                chunk_size=chunk_size,
                report=progress is None,
                stream_callback=sink,
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
                stream=stream,
            )
        if progress is not None:
            progress.update(
                task_id,
                advance=1,
                stage="完成",
                seg=f"{index}/{total_windows}",
                stream="",
                stream_speaker=None,
                rtf=result.get("rtf", float("nan")),
            )
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
            task_id = progress.add_task(
                "分段辨識",
                total=total_windows,
                kind="asr",
                seg=None,
                span=None,
                rtf=None,
                stage="準備中",
                resource=collect_resource_stats(),
            )
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
def _speaker_numbering(parsed):
    """依出現順序把原始語者標記對應成 1、2…(供對話稿與時間戳記區塊共用)。"""
    mapping = {}
    for seg in parsed or []:
        if isinstance(seg, dict) and seg.get("Speaker") is not None:
            key = str(seg.get("Speaker"))
            if key not in mapping:
                mapping[key] = len(mapping) + 1
    return mapping


def _speaker_labeled_text(parsed):
    """把分段整理成對話稿:合併同語者連續段,語者依出現順序標成 1、2…。

    僅在偵測到兩位以上語者時回傳對話稿,否則回傳 None(交由呼叫端用純文字)。
    """
    numbering = _speaker_numbering(parsed)
    if len(numbering) < 2:
        return None
    turns = []  # [(語者編號, 內容)],合併同語者連續段
    for seg in parsed:
        if not isinstance(seg, dict):
            continue
        content = seg.get("Content")
        if not isinstance(content, str) or not content.strip():
            continue
        num = numbering.get(str(seg.get("Speaker")), "?")
        content = content.strip()
        if turns and turns[-1][0] == num:
            turns[-1][1] += content
        else:
            turns.append([num, content])
    return "\n".join(f"語者{num}：{text}" for num, text in turns)


def _render_body(result, speaker_labels):
    """依設定回傳正文:多語者且開啟語者標示時用對話稿,否則用合併純文字。"""
    if speaker_labels:
        labeled = _speaker_labeled_text(result.get("parsed"))
        if labeled:
            return labeled
    return result["text"].strip()


def write_transcript(result, output_path, source_label, with_timestamps=False, speaker_labels=False):
    """把辨識結果寫入 .txt 檔。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = [_render_body(result, speaker_labels)]

    if with_timestamps and result["parsed"]:
        lines.append("")
        lines.append("=" * 60)
        lines.append("語者 / 時間戳記")
        lines.append("=" * 60)
        numbering = _speaker_numbering(result["parsed"])
        for seg in result["parsed"]:
            start = seg.get("Start", "?")
            end = seg.get("End", "?")
            speaker = numbering.get(str(seg.get("Speaker")), seg.get("Speaker", "?"))
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
    window_seconds: float = DEFAULT_WINDOW_SECONDS  # 超過此長度即分段;短音訊(<=600s)仍單次處理。長音訊單次在小顯存會 OOM 空轉,故預設分段
    chunk_size: int = DEFAULT_CHUNK_SIZE
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS  # 視為上限,實際依音訊長度動態估算
    preview_chars: int = 1200
    keep_recording: bool = True
    device: int = None
    stream_preview: bool = True   # 辨識過程中即時顯示逐字內容
    speaker_labels: bool = True   # 兩人以上時,正文/預覽以「語者N:」對話稿呈現


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
    """辨識同一段音訊;回傳 'home'(返回主選單)或 'again'(再辨識一個)。

    使用者在結果頁選「換設定重辨」時,於此迴圈內沿用同一段音訊重跑。
    """
    while True:
        processor, model = ensure_model(settings, state)
        if model is None:
            pause()
            return "home"

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
                stream=settings.stream_preview,
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
                        stream=settings.stream_preview,
                    )
                except Exception as exc2:
                    ui_error(f"辨識失敗:{exc2}")
                    pause()
                    return "home"
            else:
                ui_error(f"辨識失敗:{exc}")
                pause()
                return "home"

        # 辨識結果自動簡轉繁(台灣慣用詞),預覽與輸出檔都套用。
        with ui_status("[cyan]簡轉繁轉換中...[/cyan]"):
            convert_result_to_traditional(result)

        decision = show_result(settings, result, output_path, source_label)
        if decision == "redo":
            continue  # 換設定重辨:沿用同一段音訊重跑(設定可能已在 _redo_adjust 調整)
        return decision


def render_result_summary(settings, result, output_path, source_label):
    """辨識完成摘要面板(與『本次設定』同一套視覺)。"""
    text = result["text"].strip()
    segs = len(result["parsed"]) if result.get("parsed") else 0
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="cyan", no_wrap=True)
    table.add_column(style="white")
    speakers = {
        str(s.get("Speaker"))
        for s in (result.get("parsed") or [])
        if isinstance(s, dict) and s.get("Speaker") is not None
    }
    table.add_row("來源", source_label)
    table.add_row("字數", str(len(text)))
    if segs:
        table.add_row("段落", str(segs))
    if len(speakers) >= 2:
        table.add_row("語者", f"{len(speakers)} 人")
    table.add_row("時間戳記", "開" if settings.with_timestamps else "關")
    table.add_row("輸出", _display_path(output_path))
    return Panel(table, title="[bold]辨識完成[/bold]", border_style="green", box=box.ROUNDED, padding=(1, 2))


def _result_oneline(result, source_label, output_path):
    """『接下來』選單上方的精簡摘要列。"""
    chars = len(result["text"].strip())
    return Panel(
        f"[green]● 辨識完成[/green]    [cyan]來源[/cyan] {source_label}    "
        f"[cyan]字數[/cyan] {chars}    [cyan]輸出[/cyan] {_display_path(output_path)}",
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 2),
    )


def _redo_adjust(settings):
    """『換設定重辨』前的快速調整;回傳 True 表示開始重辨,False 表示取消。"""
    quant_order = ["4bit", "8bit", "none"]
    while True:
        items = [
            ("開始重新辨識", "套用下列設定,沿用同一段音訊"),
            (f"時間戳記:{'開' if settings.with_timestamps else '關'}", "Enter 切換"),
            (f"量化模式:{settings.quant}", "Enter 循環 4bit → 8bit → none"),
            ("分段秒數", "單次優先" if settings.window_seconds <= 0 else f"{settings.window_seconds:g}s"),
            ("熱詞 / 提示", settings.prompt or "(無)"),
            ("取消", ""),
        ]
        idx = select_menu("換設定重辨", items, subtitle="↑↓ 選擇 · Enter 確認 · Esc 取消")
        if idx is None or idx == len(items) - 1:
            return False
        if idx == 0:
            return True
        if idx == 1:
            settings.with_timestamps = not settings.with_timestamps
        elif idx == 2:
            settings.quant = quant_order[(quant_order.index(settings.quant) + 1) % len(quant_order)]
        elif idx == 3:
            settings.window_seconds = _ask_float("分段秒數(0 表示先嘗試單次)", settings.window_seconds)
        elif idx == 4:
            text = ask_text("熱詞/提示(留空表示無)", default=settings.prompt or "").strip()
            settings.prompt = text or None


def show_result(settings, result, output_path, source_label):
    """顯示結果摘要與預覽;回傳下一步決策:home / again / redo。"""
    write_transcript(
        result,
        output_path,
        source_label,
        with_timestamps=settings.with_timestamps,
        speaker_labels=settings.speaker_labels,
    )

    if USE_RICH:
        console.print(render_result_summary(settings, result, output_path, source_label))
    else:
        ui_rule("辨識結果")

    if settings.preview_chars != 0:
        preview = _render_body(result, settings.speaker_labels)
        if settings.preview_chars > 0 and len(preview) > settings.preview_chars:
            preview = preview[: settings.preview_chars].rstrip() + "\n\n[dim]...(畫面預覽已截斷,完整內容見輸出檔)[/dim]"
        ui_panel("結果預覽", preview or "(空)", border_style="cyan")

    header = _result_oneline(result, source_label, output_path) if USE_RICH else None
    while True:
        action = select_menu(
            "接下來",
            [
                ("返回主選單", ""),
                ("換設定重辨", "同一段音訊,改設定再跑一次"),
                ("再辨識一個", "回到來源選擇"),
                ("開啟輸出資料夾", _display_path(OUTPUT_DIR)),
                ("複製辨識結果", "純文字到剪貼簿"),
            ],
            subtitle="↑↓ 選擇 · Enter 確認 · Esc 返回主選單",
            header=header,
        )
        if action is None or action == 0:
            return "home"
        if action == 1:
            if _redo_adjust(settings):
                return "redo"
        elif action == 2:
            return "again"
        elif action == 3:
            _open_folder(OUTPUT_DIR)
        elif action == 4:
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
    while True:
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
            continue
        if audio.size == 0:
            ui_error("音訊內容為空。")
            pause()
            continue
        source_label = os.path.splitext(os.path.basename(path))[0]
        if run_transcription(settings, state, audio, source_label) != "again":
            return


def flow_record(settings, state):
    while True:
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
            continue

        if audio.size == 0:
            ui_error("沒有錄到任何音訊。")
            pause()
            continue

        if settings.keep_recording:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_wav(os.path.join(RECORDING_DIR, f"recording_{stamp}.wav"), audio)

        if run_transcription(settings, state, audio, "recording") != "again":
            return


# --------------------------------------------------------------------------- #
# 設定與裝置
# --------------------------------------------------------------------------- #
def settings_menu(settings):
    # 量化模式、時間戳記已上移為主畫面快速切換,此處只保留其餘參數。
    while True:
        items = [
            ("熱詞 / 提示", settings.prompt or "(無)"),
            ("分段秒數", "單次優先" if settings.window_seconds <= 0 else f"{settings.window_seconds:g}s"),
            ("chunk-size", str(settings.chunk_size)),
            ("產生上限 max-new-tokens", str(settings.max_new_tokens)),
            ("畫面預覽字數", "不顯示" if settings.preview_chars == 0 else str(settings.preview_chars)),
            ("語者標示", "開(兩人以上)" if settings.speaker_labels else "關"),
            ("辨識即時預覽", "開" if settings.stream_preview else "關"),
            ("保留麥克風錄音檔", "是" if settings.keep_recording else "否"),
            ("返回", ""),
        ]
        idx = select_menu("其他設定", items, subtitle="↑↓ 選擇 · Enter 修改 · Esc 返回")
        if idx is None or idx == len(items) - 1:
            return
        if idx == 0:
            text = ask_text("熱詞/提示(留空表示無)", default=settings.prompt or "").strip()
            settings.prompt = text or None
        elif idx == 1:
            settings.window_seconds = _ask_float("分段秒數(0 表示先嘗試單次)", settings.window_seconds)
        elif idx == 2:
            settings.chunk_size = _normalize_chunk_size(_ask_int("chunk-size", settings.chunk_size)) or DEFAULT_CHUNK_SIZE
        elif idx == 3:
            settings.max_new_tokens = _ask_int("max-new-tokens", settings.max_new_tokens)
        elif idx == 4:
            settings.preview_chars = _ask_int("畫面預覽字數(0 表示不顯示)", settings.preview_chars)
        elif idx == 5:
            settings.speaker_labels = not settings.speaker_labels
        elif idx == 6:
            settings.stream_preview = not settings.stream_preview
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


def _device_label(device):
    """主畫面用的麥克風裝置標籤。"""
    if device is None:
        return "系統預設"
    return f"輸入裝置 #{device}"


def _model_state_text(settings, state):
    """模型載入狀態(含量化是否與目前設定一致)。"""
    if state.get("model") is None:
        return "[grey50]○ 未載入[/grey50]"
    if state.get("quant") != settings.quant:
        return f"[yellow]● 已載入 {state.get('quant')} · 下次以 {settings.quant} 重載[/yellow]"
    return "[green]● 已載入[/green]"


def render_status_header(settings, state):
    """常駐狀態列:模型 / 麥克風 / 關鍵設定一眼可見,隨操作即時更新。"""
    prompt = settings.prompt.strip() if settings.prompt else ""
    if len(prompt) > 16:
        prompt = prompt[:16] + "…"
    window = "單次優先" if settings.window_seconds <= 0 else f"{settings.window_seconds:g}s"
    ts = "[green]開[/green]" if settings.with_timestamps else "[dim]關[/dim]"
    line1 = (
        f"[cyan]模型[/cyan] {settings.quant} · {_model_state_text(settings, state)}"
        f"     [cyan]麥克風[/cyan] {_device_label(settings.device)}"
    )
    line2 = (
        f"[cyan]分段[/cyan] {window}    "
        f"[cyan]時間戳記[/cyan] {ts}    "
        f"[cyan]熱詞[/cyan] {prompt or '無'}"
    )
    return Panel(
        Group(line1, line2),
        title="[bold]VibeVoice ASR[/bold]",
        subtitle="[dim]本地語音辨識 · 模型常駐記憶體[/dim]",
        border_style="bright_cyan",
        box=box.ROUNDED,
        padding=(0, 2),
    )


def flow_load_model(settings, state):
    """從主畫面預先載入(或依量化變更重載)模型,讓首次辨識不必再等。"""
    if state.get("model") is not None and state.get("quant") == settings.quant:
        return  # 已是目前量化的模型,狀態列已顯示「已載入」,無需動作
    processor, model = ensure_model(settings, state)
    if model is None:
        pause()
        return
    ui_success(f"模型已載入({settings.quant})。")
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

    quant_order = ["4bit", "8bit", "none"]
    cursor = 0
    try:
        while True:
            loaded = state.get("model") is not None
            if loaded and state.get("quant") != settings.quant:
                model_label = "重新載入模型"
                model_hint = f"目前 {state.get('quant')} → 改用 {settings.quant} 重載"
            elif loaded:
                model_label = "重新載入模型"
                model_hint = f"已載入 {state.get('quant')}"
            else:
                model_label = "載入模型"
                model_hint = f"先載入 {settings.quant}(否則首次辨識才載入)"

            items = [
                ("辨識音訊檔", "wav / mp3 / flac / m4a ..."),
                ("麥克風錄音辨識", ""),
                (model_label, model_hint),
                (f"時間戳記:{'開' if settings.with_timestamps else '關'}", "Enter 切換"),
                (f"量化模式:{settings.quant}", "Enter 循環 4bit → 8bit → none"),
                ("其他設定", "分段 / chunk / 熱詞 / 預覽 ..."),
                ("麥克風輸入裝置", _device_label(settings.device)),
                ("離開", ""),
            ]
            choice = select_menu(
                "主選單",
                items,
                subtitle="↑↓ 移動 · Enter 確認 · 數字快選 · Esc 離開",
                start=cursor,
                header=render_status_header(settings, state) if USE_RICH else None,
            )
            if choice is None or choice == 7:
                break
            cursor = choice
            if choice == 0:
                flow_file(settings, state)
            elif choice == 1:
                flow_record(settings, state)
            elif choice == 2:
                flow_load_model(settings, state)
            elif choice == 3:
                settings.with_timestamps = not settings.with_timestamps
            elif choice == 4:
                settings.quant = quant_order[(quant_order.index(settings.quant) + 1) % len(quant_order)]
            elif choice == 5:
                settings_menu(settings)
            elif choice == 6:
                devices_menu(settings)
    except KeyboardInterrupt:
        ui_print("\n[dim]已中斷。[/dim]")
    finally:
        _unload_model(state)

    ui_print("[dim]再見。[/dim]")


if __name__ == "__main__":
    main()


