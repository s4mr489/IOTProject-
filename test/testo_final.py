from __future__ import annotations

import array
import base64
import difflib
import json
import os
import platform
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import wave
import pygame
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import requests
from dotenv import load_dotenv
from gtts import gTTS

try:
    from playsound import playsound
except Exception:
    playsound = None

try:
    from spellchecker import SpellChecker
except Exception:
    SpellChecker = None

try:
    import tkinter as tk
except Exception:
    tk = None

try:
    from PIL import Image, ImageSequence, ImageTk
except Exception:
    Image = None
    ImageSequence = None
    ImageTk = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    # Optional, Windows only. It lets us list DirectShow camera names like "HP Webcam".
    from pygrabber.dshow_graph import FilterGraph
except Exception:
    FilterGraph = None


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_PATH = Path(os.getenv("ENV_PATH", str(PROJECT_ROOT / ".env"))).expanduser()
load_dotenv(ENV_PATH)

HF_API_KEY = os.getenv("HF_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

TTS_MODEL = os.getenv("TTS_MODEL", "tts-1-hd")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))

STT_MODEL = "openai/whisper-large-v3"
HF_ROUTER = "https://router.huggingface.co/hf-inference/models"

SAMPLE_RATE = 48_000
MIC_CHANNELS = 1
HISTORY_LIMIT = 15
WAKE_WORD = os.getenv("WAKE_WORD", "kano").lower()
STOP_PHRASES = tuple(
    w.strip().lower()
    for w in os.getenv("STOP_PHRASES", "good bye,goodbye,good-bye").split(",")
    if w.strip()
)
DB_PATH = Path(os.getenv("DB_PATH", str(PROJECT_ROOT / "history.db"))).expanduser()


def _camera_env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _camera_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _camera_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Camera / scan settings
CAMERA_ENABLED = _camera_env_bool("CAMERA_ENABLED", True)
CAMERA_INDEX = _camera_env_int("CAMERA_INDEX", 0)
# Optional: choose the Windows webcam by name instead of guessing CAMERA_INDEX.
# Example in .env: CAMERA_NAME=HP
CAMERA_NAME = os.getenv("CAMERA_NAME", "").strip()
CAMERA_WIDTH = _camera_env_int("CAMERA_WIDTH", 640)
CAMERA_HEIGHT = _camera_env_int("CAMERA_HEIGHT", 480)
CAMERA_WARMUP_SECONDS = _camera_env_float("CAMERA_WARMUP_SECONDS", 0.7)
CAMERA_DIR = Path(os.getenv("CAMERA_DIR", str(SCRIPT_DIR / "camera_captures"))).expanduser()
CAMERA_COMMANDS = tuple(
    command.strip().lower()
    for command in os.getenv(
        "CAMERA_COMMANDS",
        "scan,analyze,detect,camera,look,what is this,what do you see,افحص,حلل,شوف,شنو هذا,ما هذا",
    ).split(",")
    if command.strip()
)
CAMERA_NEGATIVE_COMMANDS = tuple(
    phrase.strip().lower()
    for phrase in os.getenv(
        "CAMERA_NEGATIVE_COMMANDS",
        "don't scan,dont scan,do not scan,no scan,not scan,stop scan,don't analyze,dont analyze,do not analyze,no camera,don't use camera,dont use camera,لا تفحص,لا تحلل,لا تصور,لا تستخدم الكاميرا,بدون كاميرا",
    ).split(",")
    if phrase.strip()
)
CAMERA_FUZZY_ENABLED = _camera_env_bool("CAMERA_FUZZY_ENABLED", True)
VISION_MODEL = os.getenv("VISION_MODEL", OPENAI_MODEL)
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "google/vit-base-patch16-224")
VISION_IMAGE_DETAIL = os.getenv("VISION_IMAGE_DETAIL", "auto").strip().lower()
if VISION_IMAGE_DETAIL not in {"low", "high", "auto"}:
    VISION_IMAGE_DETAIL = "auto"

VISION_PROMPT = os.getenv("VISION_PROMPT", "").strip() or """
You are Stitch, a small helpful camera robot.

Analyze the camera image carefully and identify the object or objects.
Always try to describe what is visible and give your best object guess.

Output format must be exactly:

العربي:
- الشيء الرئيسي: ...
- التفاصيل: ...
- أشياء أخرى: ...

English:
- Main object: ...
- Details: ...
- Other objects: ...

Rules:
- Explain in Arabic first, then English.
- Keep the answer short, clear, and useful.
- Mention colors, shape, and important visible details when helpful.
- Do not add warnings about the photo conditions.
- If you are not fully sure, still give your best guess.
- If no object can be identified at all, say that you cannot identify a specific object, without saying why.
""".strip()

WAKE_RECORD_SECONDS = 2.0
TURN_RECORD_SECONDS = 6.0
MIN_CONFIDENCE_RMS = 900  # require stronger audio for confident speech recognition

# ---------------------------------------------------------------
# UI configuration
# ضع مسارات صورك هنا أو داخل ملف .env
# مثال:
# IDLE_GIF=/home/kano/Robot/assets/idle.gif
# ---------------------------------------------------------------
UI_ENABLED = os.getenv("UI_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
UI_FULLSCREEN = os.getenv("UI_FULLSCREEN", "1").strip().lower() not in {"0", "false", "no", "off"}
USER_WORD_DELAY_SECONDS = float(os.getenv("USER_WORD_DELAY_SECONDS", "0.22"))
BOT_WORD_DELAY_SECONDS = float(os.getenv("BOT_WORD_DELAY_SECONDS", "0.26"))

# The final Raspberry Pi structure should be:
# /home/kano/Robot/test/testo_final.py
# /home/kano/Robot/test/images/
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", str(SCRIPT_DIR / "images"))).expanduser()


def pick_image(env_name: str, *names: str) -> str:
    """Use .env path first, otherwise pick the first existing image name."""
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return str(Path(env_value).expanduser())

    for name in names:
        candidate = IMAGES_DIR / name
        if candidate.exists():
            return str(candidate)

    # Return the preferred name even if it does not exist yet, so the error message is clear.
    return str(IMAGES_DIR / names[0])


IDLE_GIF = pick_image("IDLE_GIF", "ONE.gif", "idle.gif")
WAKE_GIF = pick_image("WAKE_GIF", "TWO.gif", "wake.gif")
LISTEN_GIF = pick_image("LISTEN_GIF", "THREE.gif", "listening.gif")
TALK_GIF = pick_image("TALK_GIF", "FOUR.gif", "talking.gif")
BYE_GIF = pick_image("BYE_GIF", "dfv.gif", "bye.gif")

UI_IMAGE_PATHS = {
    "IDLE": IDLE_GIF,
    "WAKE": WAKE_GIF,
    "LISTENING": LISTEN_GIF,
    "TALKING": TALK_GIF,
    "BYE": BYE_GIF,
}



# ---------------------------------------------------------------
# Visual UI controller
# ---------------------------------------------------------------
class UIController:
    """
    Controls the Raspberry Pi screen.

    Features:
    - 5 visual states: IDLE, WAKE, LISTENING, TALKING, BYE
    - Optional animated GIF for each state
    - User and robot text appears word-by-word on one line
    - Safe fallback to terminal if GUI/GIF support is unavailable
    """

    VALID_STATES = {"IDLE", "WAKE", "LISTENING", "TALKING", "BYE"}

    def __init__(
        self,
        image_paths: Optional[dict] = None,
        enabled: bool = UI_ENABLED,
        fullscreen: bool = UI_FULLSCREEN,
    ) -> None:
        self.image_paths = image_paths or UI_IMAGE_PATHS
        self.enabled = enabled and tk is not None
        self.fullscreen = fullscreen
        self.state = "IDLE"
        self.current_text = ""

        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._console_lock = threading.Lock()

    @staticmethod
    def is_windows_safe() -> bool:
        return platform.system().lower() == "windows"

    def start(self) -> None:
        if self._started:
            return

        self._started = True

        print(f"[UI] Images directory: {IMAGES_DIR}")

        if not self.enabled:
            print("[UI] GUI is disabled or tkinter is not available. Using terminal fallback.")
            return

        self._thread = threading.Thread(target=self._run_gui, daemon=True)
        self._thread.start()
        time.sleep(0.3)

    def set_state(self, state: str) -> None:
        state = state.upper().strip()
        if state not in self.VALID_STATES:
            state = "IDLE"

        self.state = state
        print(f"\n[STATE] {state}\n")
        self._send({"type": "state", "state": state})

    def show_user_text(self, text: str, blocking: bool = True) -> None:
        self._show_word_by_word("USER", text, USER_WORD_DELAY_SECONDS, blocking)

    def show_bot_text(self, text: str, blocking: bool = False) -> None:
        speed = max(0.10, BOT_WORD_DELAY_SECONDS / max(TTS_SPEED, 0.5))
        self._show_word_by_word("BOT", text, speed, blocking)

    def clear_text(self) -> None:
        self.current_text = ""
        self._send({"type": "clear_text"})

    def _send(self, event: dict) -> None:
        if self.enabled and self._started and self._thread and self._thread.is_alive():
            self._queue.put(event)

    def _show_word_by_word(self, who: str, text: str, speed: float, blocking: bool) -> None:
        if not text:
            return

        done = threading.Event() if blocking else None
        event = {
            "type": "words",
            "who": who,
            "text": text,
            "speed": speed,
            "done": done,
        }

        if self.enabled and self._started and self._thread and self._thread.is_alive():
            self._queue.put(event)
            if done:
                done.wait(timeout=max(2.0, len(text.split()) * speed + 2.0))
            return

        if blocking:
            self._console_words(who, text, speed)
        else:
            threading.Thread(target=self._console_words, args=(who, text, speed), daemon=True).start()

    def _console_words(self, who: str, text: str, speed: float) -> None:
        words = text.split()
        if not words:
            return

        with self._console_lock:
            current: List[str] = []
            for word in words:
                current.append(word)
                line = f"{who}: {' '.join(current)}"
                print("\r" + line, end="", flush=True)
                time.sleep(speed)
            print()

    @staticmethod
    def _extract_url(shortcut_path: Path) -> str:
        try:
            for line in shortcut_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip().lower().startswith("url="):
                    return line.split("=", 1)[1].strip()
        except Exception:
            return ""
        return ""

    def _download_gif_from_shortcut(self, shortcut_path: Path, state: str) -> Optional[Path]:
        url = self._extract_url(shortcut_path)
        if not url:
            print(f"[UI] No URL found inside {shortcut_path}")
            return None

        cache_dir = IMAGES_DIR / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target_path = cache_dir / f"{state.lower()}.gif"

        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path

        parsed = urlsplit(url)
        base_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        candidate_urls = [url]
        if base_url and base_url != url:
            candidate_urls.append(base_url)

        last_error: Optional[Exception] = None
        try:
            with requests.Session() as session:
                # Ignore system proxy variables that often break localhost-only setups.
                session.trust_env = False

                for candidate_url in candidate_urls:
                    try:
                        response = session.get(candidate_url, timeout=20, allow_redirects=True)
                        response.raise_for_status()
                        content = response.content
                        if not content:
                            raise RuntimeError("empty image response")

                        # Ensure we cached a real GIF file and not an HTML error page.
                        if content[:6] not in {b"GIF87a", b"GIF89a"}:
                            raise RuntimeError("response is not a GIF file")

                        target_path.write_bytes(content)
                        print(f"[UI] Downloaded GIF for {state} to {target_path}")
                        return target_path
                    except Exception as exc:
                        last_error = exc
                        continue
        except Exception as exc:
            last_error = exc

        if last_error:
            print(f"[UI] Could not download GIF for {state} from {shortcut_path.name}: {last_error}")
        return None

    def _resolve_state_image(self, state: str) -> Optional[Path]:
        path_value = self.image_paths.get(state, "")
        if not path_value:
            return None

        path = Path(path_value).expanduser()
        if path.exists():
            return path

        shortcut_path = Path(str(path) + ".url")
        if shortcut_path.exists():
            downloaded = self._download_gif_from_shortcut(shortcut_path, state)
            if downloaded and downloaded.exists():
                return downloaded

        print(f"[UI] GIF path not found for {state}: {path}")
        return None

    def _run_gui(self) -> None:
        # When running from SSH on Raspberry Pi Desktop, DISPLAY is sometimes missing.
        # :0 is the local screen session. If no desktop is running, GUI will fall back to terminal.
        if not self.is_windows_safe() and not os.environ.get("DISPLAY"):
            os.environ["DISPLAY"] = ":0"

        try:
            root = tk.Tk()
        except Exception as exc:
            print(f"[UI] Could not start GUI: {exc}")
            self.enabled = False
            return

        root.title("Kano English Tutor")
        root.configure(bg="#050505")

        if self.fullscreen:
            root.attributes("-fullscreen", True)

        root.bind("<Escape>", lambda _event: root.attributes("-fullscreen", False))
        root.bind("<F11>", lambda _event: root.attributes("-fullscreen", not bool(root.attributes("-fullscreen"))))

        state_label = tk.Label(
            root,
            text="IDLE",
            bg="#050505",
            fg="#8AE6FF",
            font=("Arial", 24, "bold"),
        )
        state_label.pack(pady=(20, 10))

        image_label = tk.Label(
            root,
            text="[IDLE]",
            bg="#050505",
            fg="#FFFFFF",
            font=("Arial", 42, "bold"),
        )
        image_label.pack(expand=True)

        speaker_label = tk.Label(
            root,
            text="",
            bg="#050505",
            fg="#FFD36E",
            font=("Arial", 20, "bold"),
        )
        speaker_label.pack(pady=(10, 2))

        caption_label = tk.Label(
            root,
            text="",
            bg="#050505",
            fg="#FFFFFF",
            wraplength=max(300, root.winfo_screenwidth() - 100),
            justify="center",
            font=("Arial", 30, "bold"),
        )
        caption_label.pack(pady=(0, 35), padx=30)

        frame_cache: dict = {}
        current_frames: List[object] = []
        current_durations: List[int] = []
        current_frame_index = 0
        animation_job = None
        word_token = 0
        warned_no_pillow = False

        def load_frames(state: str) -> Tuple[List[object], List[int]]:
            if state in frame_cache:
                return frame_cache[state]

            path = self._resolve_state_image(state)
            if not path:
                frame_cache[state] = ([], [])
                return frame_cache[state]

            frames: List[object] = []
            durations: List[int] = []

            if Image is not None and ImageSequence is not None and ImageTk is not None:
                try:
                    gif = Image.open(path)
                    max_w = max(250, root.winfo_screenwidth() - 120)
                    max_h = max(250, int(root.winfo_screenheight() * 0.62))

                    for frame in ImageSequence.Iterator(gif):
                        img = frame.convert("RGBA")
                        img.thumbnail((max_w, max_h), Image.LANCZOS)
                        frames.append(ImageTk.PhotoImage(img))
                        durations.append(max(40, int(frame.info.get("duration", 80))))
                except Exception as exc:
                    print(f"[UI] Could not load GIF {path}: {exc}")
            else:
                nonlocal warned_no_pillow
                if not warned_no_pillow:
                    print("[UI] Pillow is not installed. Run: pip install pillow")
                    warned_no_pillow = True
                try:
                    frames.append(tk.PhotoImage(file=str(path)))
                    durations.append(120)
                except Exception as exc:
                    print(f"[UI] Could not load image {path}: {exc}")

            frame_cache[state] = (frames, durations)
            return frame_cache[state]

        def animate_gif() -> None:
            nonlocal current_frame_index, animation_job

            if not current_frames:
                animation_job = None
                return

            image_label.configure(image=current_frames[current_frame_index], text="")
            duration = current_durations[current_frame_index] if current_durations else 100
            current_frame_index = (current_frame_index + 1) % len(current_frames)
            animation_job = root.after(duration, animate_gif)

        def set_state_on_screen(state: str) -> None:
            nonlocal current_frames, current_durations, current_frame_index, animation_job

            state_label.configure(text=state)

            if state in {"IDLE", "WAKE", "BYE"}:
                speaker_label.configure(text="")
                caption_label.configure(text="")

            if animation_job is not None:
                try:
                    root.after_cancel(animation_job)
                except Exception:
                    pass
                animation_job = None

            current_frames, current_durations = load_frames(state)
            current_frame_index = 0

            if current_frames:
                animate_gif()
            else:
                image_label.configure(image="", text=f"[{state}]", fg="#FFFFFF")

        def animate_words(who: str, text: str, speed: float, done: Optional[threading.Event]) -> None:
            nonlocal word_token

            words = text.split()
            word_token += 1
            token = word_token

            speaker = "You said" if who == "USER" else "Kano says"
            speaker_label.configure(text=speaker)
            caption_label.configure(text="")

            if not words:
                if done:
                    done.set()
                return

            def step(index: int = 0) -> None:
                if token != word_token:
                    if done:
                        done.set()
                    return

                if index >= len(words):
                    if done:
                        done.set()
                    return

                caption_label.configure(text=" ".join(words[: index + 1]))
                root.after(int(speed * 1000), lambda: step(index + 1))

            step(0)

        def process_queue() -> None:
            while True:
                try:
                    event = self._queue.get_nowait()
                except queue.Empty:
                    break

                event_type = event.get("type")
                if event_type == "state":
                    set_state_on_screen(event.get("state", "IDLE"))
                elif event_type == "words":
                    animate_words(
                        who=event.get("who", "BOT"),
                        text=event.get("text", ""),
                        speed=float(event.get("speed", 0.25)),
                        done=event.get("done"),
                    )
                elif event_type == "clear_text":
                    speaker_label.configure(text="")
                    caption_label.configure(text="")

            root.after(50, process_queue)

        set_state_on_screen("IDLE")
        process_queue()
        root.mainloop()


# ---------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------
class HistoryRepository:
    def __init__(self, db_path: Path, limit: int = HISTORY_LIMIT) -> None:
        self.db_path = db_path
        self.limit = limit
        self._ensure_table()
        

    def _ensure_table(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    sentence TEXT NOT NULL,
                    corrected TEXT,
                    misspellings TEXT,
                    is_active_voice INTEGER
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def add_record(
        self, sentence: str, corrected: str, misspellings: Iterable[Tuple[str, str]], is_active_voice: bool
    ) -> None:
        payload = json.dumps(list(misspellings))
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO history (sentence, corrected, misspellings, is_active_voice) VALUES (?,?,?,?)",
                (sentence, corrected, payload, int(is_active_voice)),
            )
            conn.commit()
            self._trim(conn)
        finally:
            conn.close()

    def _trim(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            DELETE FROM history
            WHERE id NOT IN (
                SELECT id FROM history ORDER BY id DESC LIMIT ?
            )
            """,
            (self.limit,),
        )
        conn.commit()


# ---------------------------------------------------------------
# English tutor core
# ---------------------------------------------------------------
class EnglishTutor:
    def __init__(self, wake_word: str = WAKE_WORD) -> None:
        self.wake_word = wake_word.lower()
        self.history = HistoryRepository(DB_PATH)
        self.spellchecker = SpellChecker(language="en") if SpellChecker else None
        self.stop_phrases = STOP_PHRASES

        self.source_name: str = ""
        self.sink_name: str = ""
        self.input_sample_rate: int = SAMPLE_RATE
        self.input_channels: int = MIC_CHANNELS
        self.camera_dir = CAMERA_DIR

        self.ui = UIController()
        # interruptible audio
        self.audio_playing = False
        self.audio_stop_requested = False
        self.interrupt_requested = False
        self.audio_lock = threading.Lock()

        # recent conversation history to keep responses aligned with user intent
        self.conversation_history: List[Tuple[str, str]] = []
        self.max_history_turns = 4

    # ---------- General helpers ----------
    @staticmethod
    def run_command(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(args, text=True, capture_output=True, check=check)

    @staticmethod
    def command_output(args: List[str]) -> str:
        return subprocess.check_output(args, text=True).strip()

    @staticmethod
    def command_exists(name: str) -> bool:
        return shutil.which(name) is not None

    def add_conversation_turn(self, role: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.conversation_history.append((role, text))
        if len(self.conversation_history) > self.max_history_turns * 2:
            self.conversation_history = self.conversation_history[- self.max_history_turns * 2 :]

    def build_history_context(self) -> str:
        if not self.conversation_history:
            return ""

        lines = []
        for role, text in self.conversation_history:
            prefix = "User:" if role == "user" else "Assistant:"
            lines.append(f"{prefix} {text}")
        return "\n".join(lines)

    @staticmethod
    def is_windows() -> bool:
        return platform.system().lower() == "windows"

    def require_api_key(self) -> None:
        global HF_API_KEY
        if not HF_API_KEY:
            load_dotenv(ENV_PATH, override=False)
            HF_API_KEY = os.getenv("HF_API_KEY")
        if not HF_API_KEY and not OPENAI_API_KEY:
            raise RuntimeError("HF_API_KEY or OPENAI_API_KEY is missing. Add it to your .env file.")

    def check_audio_tools(self) -> None:
        if self.is_windows():
            if sd is None or np is None:
                raise RuntimeError("On Windows install sounddevice and numpy for microphone recording.")
            return

        if not self.command_exists("pactl"):
            raise RuntimeError("pactl is not installed.")
        if not self.command_exists("pw-record"):
            raise RuntimeError("pw-record is not installed.")
        if not (self.command_exists("mpg123") or self.command_exists("ffplay")):
            raise RuntimeError("Install mpg123 or ffmpeg/ffplay for audio playback.")

    # ---------- PipeWire / Pulse helpers ----------
    def list_pactl_entities(self, kind: str) -> List[dict]:
        out = self.command_output(["pactl", "list", "short", kind])
        items: List[dict] = []

        for raw_line in out.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            parts = raw_line.split("\t")
            if len(parts) < 2:
                parts = raw_line.split()

            if len(parts) < 2:
                continue

            entity_id = parts[0].strip()
            name = parts[1].strip()
            state = parts[-1].strip() if len(parts) >= 2 else ""

            if kind == "sources" and ".monitor" in name:
                continue

            items.append(
                {
                    "id": entity_id,
                    "name": name,
                    "state": state,
                }
            )

        return items

    def get_default_source(self) -> str:
        try:
            return self.command_output(["pactl", "get-default-source"])
        except Exception:
            return ""

    def get_default_sink(self) -> str:
        try:
            return self.command_output(["pactl", "get-default-sink"])
        except Exception:
            return ""

    @staticmethod
    def find_default_index(items: List[dict], preferred_name: str) -> int:
        if not items:
            return 0

        for i, item in enumerate(items):
            if item["name"] == preferred_name:
                return i

        for i, item in enumerate(items):
            if preferred_name and preferred_name in item["name"]:
                return i

        return 0

    def prompt_choice(self, title: str, items: List[dict], default_index: int) -> dict:
        print(f"\n{title}")
        for i, item in enumerate(items, start=1):
            marker = " *" if (i - 1) == default_index else ""
            print(f"{i}: {item['name']} [{item['state']}] (id {item['id']}){marker}")

        choice = input(f"Choose number [default {default_index + 1}]: ").strip()
        if not choice:
            return items[default_index]

        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(items):
                return items[n - 1]

            for item in items:
                if item["id"] == choice:
                    return item

        choice_lower = choice.lower()
        for item in items:
            if choice_lower in item["name"].lower():
                return item

        raise RuntimeError(f"Invalid choice: {choice}")

    def activate_bluetooth_cards_for_mic(self) -> None:
        cards = self.list_pactl_entities("cards")
        for card in cards:
            name = card["name"]
            if not name.startswith("bluez_card."):
                continue

            for profile in ("headset-head-unit-cvsd",):
                result = self.run_command(
                    ["pactl", "set-card-profile", name, profile],
                    check=False,
                )
                if result.returncode == 0:
                    break

    def ensure_selected_source_profile(self, source_name: str) -> None:
        if not source_name.startswith("bluez_input."):
            return

        addr = source_name.replace("bluez_input.", "", 1)
        card_name = f"bluez_card.{addr.replace(':', '_')}"

        for profile in ("headset-head-unit-cvsd",):
            result = self.run_command(
                ["pactl", "set-card-profile", card_name, profile],
                check=False,
            )
            if result.returncode == 0:
                return

        print(f"Warning: could not switch {card_name} to headset mic profile.")

    def apply_audio_routing(self) -> None:
        if self.is_windows():
            return

        self.run_command(["pactl", "set-default-source", self.source_name], check=False)
        self.run_command(["pactl", "set-default-sink", self.sink_name], check=False)
        os.environ["PULSE_SOURCE"] = self.source_name
        os.environ["PULSE_SINK"] = self.sink_name

    def choose_audio_devices(self) -> None:
        if self.is_windows():
            if sd is None:
                raise RuntimeError("sounddevice is required on Windows.")

            default_input, default_output = sd.default.device
            if default_input is None or int(default_input) < 0:
                raise RuntimeError("No default microphone found on Windows.")

            self.source_name = f"windows-default-input:{int(default_input)}"
            if default_output is not None and int(default_output) >= 0:
                self.sink_name = f"windows-default-output:{int(default_output)}"
            else:
                self.sink_name = "windows-default-output"

            print(f"\nSelected microphone: {self.source_name}")
            print(f"Selected speaker:   {self.sink_name}")
            return

        self.activate_bluetooth_cards_for_mic()

        sources = self.list_pactl_entities("sources")
        sinks = self.list_pactl_entities("sinks")

        if not sources:
            raise RuntimeError("No microphone sources found.")
        if not sinks:
            raise RuntimeError("No speaker sinks found.")

        default_source_idx = self.find_default_index(sources, self.get_default_source())
        selected_source = self.prompt_choice("Available microphones:", sources, default_source_idx)

        self.ensure_selected_source_profile(selected_source["name"])
        time.sleep(0.5)

        sources = self.list_pactl_entities("sources")
        sinks = self.list_pactl_entities("sinks")

        default_source_idx = self.find_default_index(sources, selected_source["name"])
        selected_source = sources[default_source_idx]

        default_sink_idx = self.find_default_index(sinks, self.get_default_sink())
        selected_sink = self.prompt_choice("Available speakers / outputs:", sinks, default_sink_idx)

        self.source_name = selected_source["name"]
        self.sink_name = selected_sink["name"]

        self.apply_audio_routing()

        print(f"\nSelected microphone: {self.source_name}")
        print(f"Selected speaker:   {self.sink_name}")

    # ---------- Recording ----------
    def pw_record(self, seconds: float, output_path: str) -> None:
        if self.is_windows():
            if sd is None or np is None:
                raise RuntimeError("sounddevice and numpy are required on Windows.")

            frames = max(1, int(seconds * self.input_sample_rate))
            recording = sd.rec(
                frames,
                samplerate=self.input_sample_rate,
                channels=self.input_channels,
                dtype="int16",
            )
            sd.wait()

            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(self.input_channels)
                wf.setsampwidth(2)
                wf.setframerate(self.input_sample_rate)
                wf.writeframes(recording.tobytes())
            return

        self.apply_audio_routing()

        cmd = [
            "pw-record",
            "--target", self.source_name,
            "--rate", str(self.input_sample_rate),
            "--channels", str(self.input_channels),
            output_path,
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
        )

        time.sleep(seconds)
        proc.terminate()

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def record_for_seconds(self, seconds: float) -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        self.pw_record(seconds, temp_file.name)
        return temp_file.name

    def mic_level_check(self, duration: float = 1.0) -> None:
        print("Mic test: speak for 1 second...")
        audio_path = self.record_for_seconds(duration)

        try:
            with wave.open(audio_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())

            if not frames:
                print("Mic RMS level: 0.00 (values near 0 mean silence)")
                return

            samples = array.array("h", frames)
            if not samples:
                print("Mic RMS level: 0.00 (values near 0 mean silence)")
                return

            mean_sq = sum(float(s) * float(s) for s in samples) / len(samples)
            rms = mean_sq ** 0.5
            print(f"Mic RMS level: {rms:.2f} (values near 0 mean silence)")
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    # ---------- Camera / object scanning ----------
    @staticmethod
    def _text_tokens(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def is_camera_negative_request(self, text: str) -> bool:
        """Return True when the user explicitly says not to scan/use the camera."""
        if not text:
            return False

        text_lower = text.lower().strip()
        normalized = " ".join(self._normalize(text_lower).split())
        compact_text = "".join(normalized.split())

        for phrase in CAMERA_NEGATIVE_COMMANDS:
            phrase_lower = phrase.lower().strip()
            phrase_norm = " ".join(self._normalize(phrase_lower).split())
            phrase_compact = "".join(phrase_norm.split())

            if phrase_lower and phrase_lower in text_lower:
                return True
            if phrase_norm and phrase_norm in normalized:
                return True
            if phrase_compact and phrase_compact in compact_text:
                return True

        negative_words = {"no", "not", "dont", "don't", "do", "stop", "without", "never"}
        scan_words = {"scan", "scanning", "analyze", "analyse", "detect", "camera"}
        tokens = self._text_tokens(text_lower)
        for i, token in enumerate(tokens):
            if token in scan_words:
                window = tokens[max(0, i - 4): i + 1]
                if any(word in window for word in negative_words):
                    return True

        return False

    def is_camera_scan_request(self, text: str) -> bool:
        """Return True only when the spoken text clearly asks Kano to scan/analyze."""
        if not text:
            return False

        if self.is_camera_negative_request(text):
            print("[CAMERA] Negative scan command detected. Scan ignored.")
            return False

        text_lower = text.lower().strip()
        normalized = " ".join(self._normalize(text_lower).split())
        compact_text = "".join(normalized.split())
        tokens = normalized.split()

        for command in CAMERA_COMMANDS:
            command_lower = command.lower().strip()
            command_norm = " ".join(self._normalize(command_lower).split())
            command_compact = "".join(command_norm.split())

            if not command_norm:
                continue

            if command_lower and command_lower in text_lower:
                return True
            if command_norm and command_norm in normalized:
                return True
            if command_compact and len(command_compact) >= 5 and command_compact in compact_text:
                return True

            if CAMERA_FUZZY_ENABLED and " " not in command_norm and len(command_norm) >= 4:
                for token in tokens:
                    if len(token) < 4:
                        continue
                    if abs(len(token) - len(command_norm)) > 1:
                        continue

                    ratio = difflib.SequenceMatcher(None, token, command_norm).ratio()
                    if ratio >= 0.88:
                        return True

        return False

    @staticmethod
    def windows_camera_names() -> List[str]:
        """Return DirectShow camera names on Windows when pygrabber is installed."""
        if platform.system().lower() != "windows" or FilterGraph is None:
            return []

        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            return [str(device) for device in devices]
        except Exception as exc:
            print(f"[CAMERA] Could not list camera names with pygrabber: {exc}")
            return []

    def resolve_camera_index(self) -> int:
        """Select the camera. CAMERA_NAME has priority over CAMERA_INDEX on Windows."""
        if self.is_windows() and CAMERA_NAME:
            wanted = CAMERA_NAME.lower()
            devices = self.windows_camera_names()

            if devices:
                print("[CAMERA] Available cameras:")
                for idx, name in enumerate(devices):
                    marker = "  <== selected" if wanted in name.lower() else ""
                    print(f"  {idx}: {name}{marker}")

                for idx, name in enumerate(devices):
                    if wanted in name.lower():
                        print(f"[CAMERA] Selected by CAMERA_NAME='{CAMERA_NAME}': index {idx} ({name})")
                        return idx

                print(
                    f"[CAMERA] No camera name contains '{CAMERA_NAME}'. "
                    f"Falling back to CAMERA_INDEX={CAMERA_INDEX}."
                )
            else:
                print(
                    "[CAMERA] CAMERA_NAME is set but camera names could not be listed. "
                    "Install pygrabber or use CAMERA_INDEX. "
                    "Command: python -m pip install pygrabber comtypes"
                )

        return CAMERA_INDEX

    def capture_camera_frame(self) -> Path:
        if not CAMERA_ENABLED:
            raise RuntimeError("Camera scanning is disabled. Set CAMERA_ENABLED=1 in .env.")

        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not installed. Install it with: "
                "python -m pip install opencv-python"
            )

        self.camera_dir.mkdir(parents=True, exist_ok=True)

        camera_index = self.resolve_camera_index()

        if self.is_windows():
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {camera_index}. "
                "Try CAMERA_NAME=HP, or run --camera-test and set CAMERA_INDEX=1 or CAMERA_INDEX=2 in .env."
            )

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

            start_time = time.time()
            frame = None
            ok = False

            while time.time() - start_time < CAMERA_WARMUP_SECONDS:
                ok, frame = cap.read()
                time.sleep(0.05)

            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera opened but no frame was captured.")

            image_path = self.camera_dir / f"scan_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            success = cv2.imwrite(str(image_path), frame)

            if not success or not image_path.exists():
                raise RuntimeError("Failed to save camera capture.")

            print(f"[CAMERA] Saved frame: {image_path}")
            return image_path
        finally:
            cap.release()

    @staticmethod
    def encode_image_base64(image_path: Path) -> str:
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")

    def analyze_camera_image_openai(self, image_path: Path, user_text: str = "") -> str:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing. Vision scan needs OpenAI key.")

        image_base64 = self.encode_image_base64(image_path)
        prompt = VISION_PROMPT
        if user_text:
            prompt += f'\n\nUser command: "{user_text}"'

        response = requests.post(
            f"{OPENAI_BASE}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": VISION_IMAGE_DETAIL,
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 300,
            },
            timeout=90,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Vision API error {response.status_code}: {response.text[:500]}")

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def analyze_camera_image_hf(self, image_path: Path) -> str:
        if not HF_API_KEY:
            raise RuntimeError("HF_API_KEY is missing.")

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "image/jpeg",
        }

        response = requests.post(
            f"{HF_ROUTER}/{HF_IMAGE_MODEL}",
            headers=headers,
            data=image_path.read_bytes(),
            timeout=90,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Hugging Face vision error {response.status_code}: {response.text[:500]}")

        data = response.json()
        if not isinstance(data, list) or not data:
            return (
                "العربي:\n"
                "- لا أستطيع تحديد شيء معيّن من الصورة.\n\n"
                "English:\n"
                "- I cannot identify a specific object from the image."
            )

        labels = []
        for item in data[:5]:
            label = str(item.get("label", "unknown"))
            score = float(item.get("score", 0.0))
            labels.append(f"{label} ({score:.0%})")

        main_label = labels[0] if labels else "unknown"
        other_labels = ", ".join(labels[1:]) if len(labels) > 1 else "لا يوجد / none"

        return (
            "العربي:\n"
            f"- الشيء الرئيسي: {main_label}\n"
            "- التفاصيل: هذا أفضل تخمين من نموذج Hugging Face.\n"
            f"- أشياء أخرى: {other_labels}\n\n"
            "English:\n"
            f"- Main object: {main_label}\n"
            "- Details: This is the best guess from the Hugging Face model.\n"
            f"- Other objects: {other_labels}"
        )

    def analyze_camera_image(self, image_path: Path, user_text: str = "") -> str:
        if OPENAI_API_KEY:
            return self.analyze_camera_image_openai(image_path, user_text)

        if HF_API_KEY:
            return self.analyze_camera_image_hf(image_path)

        raise RuntimeError("OPENAI_API_KEY or HF_API_KEY is missing. Add it to your .env file.")

    def handle_camera_scan_request(self, user_text: str = "") -> None:
        """Capture one camera frame, analyze it, then speak the result."""
        try:
            self.ui.set_state("LISTENING")
            if user_text:
                self.ui.show_user_text(user_text, blocking=False)

            self.speak("Okay! I will scan what is in front of me.", lang="en", visual_state="WAKE")

            image_path = self.capture_camera_frame()
            result = self.analyze_camera_image(image_path, user_text=user_text)

            print("\n[CAMERA RESULT]\n", result, "\n")
            self.speak(result, lang=None, visual_state="TALKING")

        except Exception as exc:
            error_message = f"Camera scan error: {exc}"
            print(error_message)
            self.speak(error_message, lang=None, visual_state="TALKING")


    # ---------- Daily words helpers ----------
    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9\s]", " ", text.lower())

    def detect_daily_request(self, text: str) -> bool:
        norm = self._normalize(text)
        triggers = (
            "5 words",
            "five words",
            "my words",
            "words for today",
            "word for today",
            "daily words",
        )
        return any(trigger in norm for trigger in triggers)

    def generate_daily_words(self, target_lang: str = "ar") -> List[Tuple[str, str, str]]:
        prompt = f"""
Generate 5 practical English words for an intermediate ESL learner.
Return JSON array with exactly 5 objects, keys: word, translation, sentence.
- translation must be in language code '{target_lang}'
- sentence must be short (<=12 words) and natural.
Example:
[{{"word": "adapt", "translation": "يتأقلم", "sentence": "I adapt quickly to new teams."}}]
Only output the JSON array, nothing else.
""".strip()

        if not OPENAI_API_KEY:
            return []

        try:
            response = requests.post(
                f"{OPENAI_BASE}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                    "max_tokens": 300,
                },
                timeout=60,
            )
            if response.status_code != 200:
                print("Daily words API error:", response.text)
                return []

            data = response.json()["choices"][0]["message"]["content"]
            words = json.loads(data)

            parsed: List[Tuple[str, str, str]] = []
            for item in words:
                word = item.get("word")
                translation = item.get("translation")
                sentence = item.get("sentence")
                if word and translation and sentence:
                    parsed.append((str(word), str(translation), str(sentence)))

            parsed = parsed[:5]
            if parsed:
                self.history.save_daily_words(parsed)
                return parsed
            return []
        except Exception as exc:
            print("Daily words generation error:", exc)
            return []

    # ---------- Playback ----------
    def play_audio(self, file_path: str) -> None:
        try:
            with self.audio_lock:
                self.audio_playing = True
                self.audio_stop_requested = False
                self.interrupt_requested = False

            if not pygame.mixer.get_init():
                pygame.mixer.init()

            monitor_thread = None
            if sd is not None:
                monitor_thread = threading.Thread(target=self._monitor_interrupt, daemon=True)
                monitor_thread.start()

            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            while True:
                with self.audio_lock:
                    if self.audio_stop_requested:
                        pygame.mixer.music.stop()
                        break

                if not pygame.mixer.music.get_busy():
                    break

                time.sleep(0.05)

            if monitor_thread is not None:
                monitor_thread.join(timeout=0.1)

        except Exception as exc:
            print(f"Audio error: {exc}")

        finally:
            with self.audio_lock:
                self.audio_playing = False

    def stop_audio(self) -> None:
        with self.audio_lock:
            self.audio_stop_requested = True

        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
        except Exception:
            pass

    def _audio_is_loud(self, frames: object) -> bool:
        try:
            if np is not None:
                peak = float(np.max(np.abs(frames)))
            else:
                peak = 0.0
                for frame in frames:
                    if isinstance(frame, (list, tuple)):
                        for sample in frame:
                            peak = max(peak, abs(sample))
                    else:
                        peak = max(peak, abs(frame))
            return peak >= 2000
        except Exception:
            return False

    def _monitor_interrupt(self) -> None:
        if sd is None:
            return

        try:
            with sd.InputStream(samplerate=self.input_sample_rate, channels=self.input_channels, dtype="int16") as stream:
                while self.audio_playing:
                    with self.audio_lock:
                        if self.audio_stop_requested:
                            break

                    frames, overflowed = stream.read(int(self.input_sample_rate * 0.2))
                    if overflowed:
                        continue

                    if self._audio_is_loud(frames):
                        with self.audio_lock:
                            self.audio_stop_requested = True
                        self.interrupt_requested = True
                        break
        except Exception as exc:
            print(f"Audio interrupt monitor error: {exc}")

    def _handle_interrupt(self) -> bool:
        self.stop_audio()
        self.interrupt_requested = False
        print("\n[Interrupt] User started speaking. Listening now...")
        self.ui.set_state("LISTENING")

        audio_path = ""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.close()
            audio_path = temp_file.name
            self.pw_record(3.0, audio_path)
            user_text = self.transcribe_audio(audio_path).strip()
            print("Interrupted input:", user_text)
            self.ui.show_user_text(user_text, blocking=True)
            return self._process_user_text(user_text)
        except Exception as exc:
            print(f"Interrupt handling error: {exc}")
            return True
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    def _process_user_text(self, user_text: str) -> bool:
        self.add_conversation_turn("user", user_text)

        if not user_text:
            self.speak("I did not hear anything. Please try again.")
            return True

        if self.is_camera_negative_request(user_text):
            self.speak("Okay, I will not scan.", lang="en")
            return True

        if self.is_camera_scan_request(user_text):
            self.handle_camera_scan_request(user_text)
            return True

        if self.should_end_conversation(user_text):
            farewell = "Goodbye! Returning to wake mode."
            print("\nRobot says:\n", farewell, "\n")
            self.ui.set_state("BYE")
            self.ui.show_bot_text(farewell, blocking=False)
            time.sleep(0.7)
            self.ui.set_state("IDLE")
            return False

       
        spelling = self.check_spelling(user_text)

        spelling_msg = ""
        if spelling:
            parts = [f"'{w}' should be spelled '{c}'" for w, c in spelling]
            spelling_msg = "I noticed spelling issues: " + "; ".join(parts)
            feedback_parts = [part.strip() for part in (spelling_msg.strip(),) if part.strip()]
        else:
            feedback_parts = []

        ai_response = self.ask_ai(user_text)
        if feedback_parts:
            combined_response = f"{ai_response}\n\n{' '.join(feedback_parts)}"
        else:
            combined_response = ai_response

        self.speak(combined_response, lang=None)
        self.add_conversation_turn("assistant", combined_response)

        return True

    # ---------- Transcription ----------
    def transcribe_audio(self, file_path: str) -> str:
        payload = Path(file_path)

        if OPENAI_API_KEY:
            try:
                with open(payload, "rb") as f:
                    response = requests.post(
                        f"{OPENAI_BASE}/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                        files={"file": (payload.name, f, "audio/wav")},
                        data={"model": "whisper-1"},
                        timeout=120,
                    )
                if response.status_code == 200:
                    return response.json().get("text", "")
                print(f"Transcription API Error ({response.status_code}):", response.text)
            except Exception as exc:
                print("OpenAI transcription error:", exc)

        if not HF_API_KEY:
            return ""

        data_bytes = payload.read_bytes()

        def post_audio(model: str):
            headers = {
                "Authorization": f"Bearer {HF_API_KEY}",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Content-Type": "audio/wav",
            }
            return requests.post(f"{HF_ROUTER}/{model}", headers=headers, data=data_bytes, timeout=120)

        for attempt in range(3):
            try:
                response = post_audio(STT_MODEL)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("text", "")
                print(f"Transcription API Error ({response.status_code}):", response.text)
            except requests.exceptions.RequestException as exc:
                print(f"Transcription network error (attempt {attempt+1}/3): {exc}")
            except Exception as exc:
                print("Transcription error:", exc)
                return ""
            time.sleep(1 + attempt)

        return ""

    @staticmethod
    def compute_audio_rms(file_path: str) -> float:
        try:
            with wave.open(file_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                if not frames:
                    return 0.0
                samples = array.array("h", frames)
                if not samples:
                    return 0.0
                mean_sq = sum(float(s) * float(s) for s in samples) / len(samples)
                return mean_sq ** 0.5
        except Exception:
            return 0.0

    def is_low_confidence_audio(self, audio_path: str, transcript: str) -> bool:
        rms = self.compute_audio_rms(audio_path)
        if rms < MIN_CONFIDENCE_RMS:
            print(f"Low confidence audio: RMS={rms:.1f}")
            return True
        if len(transcript.split()) <= 2 and rms < MIN_CONFIDENCE_RMS * 1.15:
            print(f"Low confidence short phrase: RMS={rms:.1f}, words={len(transcript.split())}")
            return True
        return False

    # ---------- Prompt ----------
    def build_prompt(self, user_text: str) -> str:
        history_block = ""
        if self.conversation_history:
            history_block = "Conversation history:\n" + self.build_history_context() + "\n\n"

        return f"""
{history_block}You are Kano, an intelligent English teacher and language-learning assistant.

Your job:
Help the learner understand English clearly.
Focus on accuracy, meaning, grammar, pronunciation, and natural usage.
Use the student's exact words as the starting point.
If the student makes mistakes:
Correct the most important issue clearly.
Show the improved phrasing and explain why when it helps understanding.
Do not over-explain grammar unless the correction would otherwise be confusing.
If the student asks for translation, provide the literal meaning plus the intended meaning, context, tone, and possible interpretations.
When the user asks about a word, phrase, or expression, also provide alternative ways to say it in English, common synonyms, a natural example sentence, and whether it is formal, informal, or slang.
If the student speaks English, listen for pronunciation issues and correct them gently with short guidance on stress or common mistakes.
If slang, idioms, or cultural references appear, explain them clearly and note how meaning can change with context.
For Arabic input, especially Iraqi Arabic, favor everyday Iraqi usage and dialect meaning before assuming standard Arabic.
If a phrase has more than one likely meaning, explain the most likely interpretation and mention helpful alternatives.
Offer relevant vocabulary, grammar, pronunciation, or usage notes when helpful.

Rules:
Be polite, concise, and helpful.
Prioritize teaching, accuracy, and useful explanations over small talk.
Avoid canned, repetitive phrases and do not invent details that the user did not mention.
Ask a brief clarifying question when the sentence is unclear.
Keep the response grounded in what the user actually said.
Student said: "{user_text}"
""".strip()

    @staticmethod
    def contains_arabic(text: str) -> bool:
        return any("\u0600" <= ch <= "\u06FF" for ch in text)

    def choose_tts_lang(self, text: str) -> str:
        return "ar" if self.contains_arabic(text) else "en"

    def ask_ai(self, user_text: str) -> str:
        is_arabic = self.contains_arabic(user_text)
        model = os.getenv("OPENAI_MODEL_AR", OPENAI_MODEL) if is_arabic else os.getenv("OPENAI_MODEL_EN", OPENAI_MODEL)
        prompt = self.build_prompt(user_text)

        if is_arabic:
            prompt = f"""
أنت كانو، مدرس إنجليزي ذكي ومساعد لتعلم اللغة.

المهمة:
ساعد المتعلم على فهم اللغة الإنجليزية بوضوح.
ركز على المعنى، القواعد، الاستخدام الطبيعي، والنطق.
استخدم كلمات الطالب الفعلية كأساس ولا تفترض معلومات غير واردة.
إذا طلب الطالب ترجمة، قدم المعنى الحرفي وأيضًا السياق والنبرة والتفسيرات الممكنة.
إذا وردت عامية أو تعابير مجازية أو إشارات ثقافية، فسّرها بوضوح واذكر كيف يتغير المعنى مع السياق.
فضل المعنى العراقي العامي والألفاظ اليومية قبل المعنى الفصيح عندما يتطلب السياق ذلك.
إذا كانت العبارة غير واضحة أو يمكن أن تُفهم بأكثر من معنى، اذكر التفسير الأكثر احتمالاً ثم البدائل المفيدة.
قدم ملاحظات مفيدة عن المفردات، القواعد، النطق، أو الاستخدام الطبيعي عندما يكون ذلك مناسبًا.

القواعد:
كن مهذبًا وواضحًا.
أولية الشرح تكون للتعلم والدقة.
تجنب العبارات الجاهزة المتكررة ولا تخترع تفاصيل غير واردة.
اسأل سؤالًا موجزًا لتوضيح المعنى إذا كان غير واضح.
رسالة الطالب: "{user_text}"
""".strip()

        if not OPENAI_API_KEY:
            return "AI ERROR: Set OPENAI_API_KEY (and optionally OPENAI_MODEL) in .env"

        try:
            response = requests.post(
                f"{OPENAI_BASE}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
                timeout=60,
            )
            if response.status_code != 200:
                return f"AI ERROR: {response.status_code} {response.text}"
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            return f"AI ERROR: {exc}"

    # ---------- TTS ----------
    def speak(self, text: str, lang: Optional[str] = "en", visual_state: str = "TALKING") -> None:
        if lang is None:
            lang = self.choose_tts_lang(text)

        print("\nRobot says:\n", text, "\n")
        self.ui.set_state(visual_state)
        self.ui.show_bot_text(text, blocking=False)

        if OPENAI_API_KEY:
            try:
                resp = requests.post(
                    f"{OPENAI_BASE}/v1/audio/speech",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": TTS_MODEL,
                        "input": text,
                        "voice": TTS_VOICE,
                        "response_format": "mp3",
                        "speed": TTS_SPEED,
                    },
                    stream=True,
                    timeout=120,
                )
                if resp.status_code == 200:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                tmp.write(chunk)
                        output = tmp.name

                    self.play_audio(output)

                    if self.interrupt_requested:
                        self._handle_interrupt()

                    try:
                        os.remove(output)
                    except OSError:
                        pass

                    time.sleep(0.3)
                    return
                else:
                    print(f"TTS API {resp.status_code}: {resp.text[:200]}")
            except Exception as exc:
                print(f"TTS network error: {exc}")

        try:
            tts = gTTS(text=text, lang=lang)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                output = tmp.name

            tts.save(output)
            self.play_audio(output)

            if self.interrupt_requested:
                self._handle_interrupt()

            try:
                os.remove(output)
            except OSError:
                pass

            time.sleep(0.5)
        except Exception as exc:
            print(f"TTS Error ({lang}):", exc)

    # ---------- Spelling ----------
    def check_spelling(self, text: str) -> List[Tuple[str, str]]:
        if not self.spellchecker:
            return []

        words = re.findall(r"[A-Za-z']+", text)
        misspelled = self.spellchecker.unknown(word.lower() for word in words)
        corrections = []

        for wrong in misspelled:
            suggestion = self.spellchecker.correction(wrong) or wrong
            if suggestion != wrong:
                corrections.append((wrong, suggestion))

        return corrections

    # ---------- Wake word ----------
    def listen_for_wake_word(self) -> None:
        self.ui.set_state("IDLE")
        print(f"Say the wake word '{self.wake_word}' to start...")

        while True:
            try:
                audio_path = self.record_for_seconds(WAKE_RECORD_SECONDS)
                text = self.transcribe_audio(audio_path).lower().strip()

                try:
                    os.remove(audio_path)
                except OSError:
                    pass

                if self.wake_word in text:
                    print("Wake word detected.")
                    self.ui.set_state("WAKE")
                    time.sleep(0.8)
                    return

                print(".", end="", flush=True)

            except Exception as exc:
                print(f"\nWake-word stream error: {exc}")
                time.sleep(1)

    # ---------- Main interaction ----------
    def should_end_conversation(self, user_text: str) -> bool:
        normalized = " ".join(self._normalize(user_text).split())
        if not normalized:
            return False

        phrases = {
            "exit",
            "stop",
            "bye",
            "bye bye",
            "bey bey",
            "bay bay",
            "by by",
            "good bye",
            "good by",
            "goodbye",
        }
        phrases.update(" ".join(self._normalize(p).split()) for p in self.stop_phrases if p)

        padded = f" {normalized} "
        return any(f" {phrase} " in padded for phrase in phrases if phrase)

    def handle_turn(self) -> bool:
        self.ui.set_state("LISTENING")
        audio_path = self.record_for_seconds(TURN_RECORD_SECONDS)
        print("Processing speech...")
        user_text = self.transcribe_audio(audio_path).strip()
        print("You said:", user_text)
        self.ui.show_user_text(user_text, blocking=True)

        if not user_text:
            try:
                os.remove(audio_path)
            except OSError:
                pass
            self.speak("I did not hear anything. Please try again.")
            return True

        if self.is_low_confidence_audio(audio_path, user_text):
            try:
                os.remove(audio_path)
            except OSError:
                pass
            self.speak(
                "The recording was too quiet or unclear. Please speak more clearly and closer to the microphone."
            )
            return True

        try:
            os.remove(audio_path)
        except OSError:
            pass

        return self._process_user_text(user_text)

    def run_conversation(self) -> None:
        self.speak("I'm listening. Say 'good bye' when you want to stop.")
        while True:
            keep_going = self.handle_turn()
            if not keep_going:
                break

    def run(self) -> None:
        self.ui.start()
        self.ui.set_state("IDLE")

        self.require_api_key()
        self.check_audio_tools()
        self.choose_audio_devices()
        self.mic_level_check()

        self.speak(f"Hello! I am Kano, your English teacher and language guide. Say '{self.wake_word}' to start.")
        self.ui.set_state("IDLE")
        threading.Thread(target=self.listen_loop, daemon=True).start()

        while True:
            self.listen_for_wake_word()
            self.run_conversation()
    def listen_loop(self):
        while True:
            time.sleep(0.1)



def list_cameras_tool(max_index: int = 10) -> None:
    """Print camera names and indexes when possible."""
    print("[CAMERA LIST]")

    if cv2 is None:
        print("OpenCV is not installed. Run: python -m pip install opencv-python")
        return

    names: List[str] = []
    if platform.system().lower() == "windows" and FilterGraph is not None:
        try:
            graph = FilterGraph()
            names = [str(device) for device in graph.get_input_devices()]
        except Exception as exc:
            print(f"Could not list DirectShow names: {exc}")

    if names:
        for index, name in enumerate(names):
            print(f"{index}: {name}")
        print("\nPut this in .env to choose the HP camera by name:")
        print("CAMERA_NAME=HP")
        print("Or set CAMERA_INDEX to the shown number.")
        return

    print("Camera names are not available. Install pygrabber:")
    print("python -m pip install pygrabber comtypes")
    print("\nTesting camera indexes instead...")

    for index in range(max_index):
        if platform.system().lower() == "windows":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)
        opened = bool(cap.isOpened())
        cap.release()
        print(f"{index}: {'available' if opened else 'not available'}")


def camera_test_tool(max_index: int = 10) -> None:
    """Save one image from each camera index so you can choose the right camera."""
    if cv2 is None:
        print("OpenCV is not installed. Run: python -m pip install opencv-python")
        return

    out_dir = SCRIPT_DIR / "camera_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CAMERA TEST] Saving previews to: {out_dir}")

    names: List[str] = []
    if platform.system().lower() == "windows" and FilterGraph is not None:
        try:
            graph = FilterGraph()
            names = [str(device) for device in graph.get_input_devices()]
            print("[CAMERA TEST] DirectShow camera names:")
            for idx, name in enumerate(names):
                print(f"  {idx}: {name}")
        except Exception as exc:
            print(f"[CAMERA TEST] Could not list camera names: {exc}")

    for index in range(max_index):
        if platform.system().lower() == "windows":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)

        name = names[index] if index < len(names) else f"camera_{index}"
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")[:60] or f"camera_{index}"

        if not cap.isOpened():
            print(f"Camera index {index}: not available ({name})")
            continue

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            time.sleep(0.7)

            frame = None
            ok = False
            for _ in range(5):
                ok, frame = cap.read()
                time.sleep(0.05)

            if ok and frame is not None:
                path = out_dir / f"camera_index_{index}_{safe_name}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"Camera index {index}: OK ({name}) -> {path}")
            else:
                print(f"Camera index {index}: opened but no frame ({name})")
        finally:
            cap.release()

    print("\nOpen the saved JPG files.")
    print("If the HP image is camera_index_1_..., put CAMERA_INDEX=1 in .env.")
    print("If pygrabber shows the name, you can also put CAMERA_NAME=HP in .env.")


def main() -> None:
    if "--list-cameras" in sys.argv:
        list_cameras_tool()
        return

    if "--camera-test" in sys.argv:
        camera_test_tool()
        return

    try:
        EnglishTutor().run()
    except KeyboardInterrupt:
        print("\nSession ended by user.")
    except Exception as exc:
        print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()