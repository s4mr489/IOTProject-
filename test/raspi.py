from __future__ import annotations

import array
import json
import os
import platform
import queue
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import wave
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


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
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
WAKE_WORD = os.getenv("WAKE_WORD", "stitch").lower()
STOP_PHRASES = tuple(
    w.strip().lower()
    for w in os.getenv("STOP_PHRASES", "good bye,goodbye,good-bye").split(",")
    if w.strip()
)
DB_PATH = PROJECT_ROOT / "history.db"

WAKE_RECORD_SECONDS = 2.0
TURN_RECORD_SECONDS = 6.0

# ---------------------------------------------------------------
# UI configuration
# ضع مسارات صورك هنا أو داخل ملف .env
# مثال:
# IDLE_GIF=/home/stitch/Robot/assets/idle.gif
# ---------------------------------------------------------------
UI_ENABLED = os.getenv("UI_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
UI_FULLSCREEN = os.getenv("UI_FULLSCREEN", "1").strip().lower() not in {"0", "false", "no", "off"}
USER_WORD_DELAY_SECONDS = float(os.getenv("USER_WORD_DELAY_SECONDS", "0.22"))
BOT_WORD_DELAY_SECONDS = float(os.getenv("BOT_WORD_DELAY_SECONDS", "0.26"))

IDLE_GIF = os.getenv("IDLE_GIF", "")
WAKE_GIF = os.getenv("WAKE_GIF", "")
LISTEN_GIF = os.getenv("LISTEN_GIF", "")
TALK_GIF = os.getenv("TALK_GIF", "")
BYE_GIF = os.getenv("BYE_GIF", "")

UI_IMAGE_PATHS = {
    "IDLE": str(PROJECT_ROOT / "images" / "ONE.gif"),
    "WAKE": str(PROJECT_ROOT / "images" / "TWO.gif"),
    "LISTENING": str(PROJECT_ROOT / "images" / "THREE.gif"),
    "TALKING": str(PROJECT_ROOT / "images" / "FOUR.gif"),
    "BYE": str(PROJECT_ROOT / "images" / "dfv.gif"),
}

DEFAULT_DAILY_WORDS: List[Tuple[str, str, str]] = [
    ("adapt", "يتأقلم", "I adapt quickly to new teams."),
    ("confident", "واثق", "She feels confident about the exam."),
    ("gather", "يجمع", "Let's gather ideas before we start."),
    ("improve", "يحسّن", "Daily practice will improve your accent."),
    ("remind", "يذكّر", "Please remind me about the meeting."),
]


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

    def start(self) -> None:
        if self._started:
            return

        self._started = True

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

        cache_dir = PROJECT_ROOT / "images" / ".cache"
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
        try:
            root = tk.Tk()
        except Exception as exc:
            print(f"[UI] Could not start GUI: {exc}")
            self.enabled = False
            return

        root.title("Stitch English Tutor")
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

            speaker = "You said" if who == "USER" else "Stitch says"
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
        self._ensure_daily_table()

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

    def _ensure_daily_table(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_sets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    words_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save_daily_words(self, words: List[Tuple[str, str, str]]) -> None:
        payload = json.dumps(words, ensure_ascii=False)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("INSERT INTO daily_sets (words_json) VALUES (?)", (payload,))
            conn.commit()
            conn.execute(
                """
                DELETE FROM daily_sets
                WHERE id NOT IN (SELECT id FROM daily_sets ORDER BY id DESC LIMIT 10)
                """
            )
            conn.commit()
        finally:
            conn.close()

    def latest_daily_words(self) -> List[Tuple[str, str, str]]:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute("SELECT words_json FROM daily_sets ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            if not row:
                return []
            try:
                data = json.loads(row[0])
                return [(str(w), str(t), str(s)) for w, t, s in data]
            except Exception:
                return []
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
        self.daily_words: List[Tuple[str, str, str]] = []

        self.source_name: str = ""
        self.sink_name: str = ""
        self.input_sample_rate: int = SAMPLE_RATE
        self.input_channels: int = MIC_CHANNELS

        self.ui = UIController()

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
        self.apply_audio_routing()
        system = platform.system().lower()
        env = os.environ.copy()

        if system == "windows" and playsound:
            try:
                playsound(file_path)
                return
            except Exception:
                pass

        if self.command_exists("mpg123"):
            try:
                subprocess.run(
                    ["mpg123", "-q", file_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    env=env,
                )
                return
            except Exception:
                pass

        if self.command_exists("ffplay"):
            try:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    env=env,
                )
                return
            except Exception:
                pass

        if system == "windows":
            os.startfile(file_path)  # type: ignore[attr-defined]
            return

        print("Could not find an mp3 player. Install 'mpg123' or 'ffmpeg'.")

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

    # ---------- Prompt ----------
    def build_prompt(self, user_text: str) -> str:
        daily_block = ""
        if self.daily_words:
            lines = [f"- {w} (translation: {t})" for w, t, _ in self.daily_words]
            daily_block = (
                "\nDaily words to reinforce; weave at least two naturally and invite the learner to use them:\n"
                + "\n".join(lines)
            )

        return f"""
You are a friendly English teacher like Duolingo.

Your job:
Teach English in a fun, conversational way.
Keep the conversation natural, friendly, and engaging.
If the student makes mistakes:
Gently correct ONLY the important mistake
Show the correct sentence in a simple way
Do NOT over-explain grammar
Always continue the conversation after correction.
Explain meaning simply in Arabic when needed.

Rules:
Keep replies short and fun
Be like a friendly tutor, not an examiner
Mix teaching + chatting
speak in a playful alien-like tone use short excited sentences slightly childish and curious reactions in a pitch tone
Student said: "{user_text}"
{daily_block}
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
            daily = ""
            if self.daily_words:
                daily = "الكلمات اليومية: " + ", ".join(w for w, _, _ in self.daily_words)
            prompt = f"""
أنت مدرس إنجليزي ممتع مثل Duolingo.

المهام:
تعلّم الإنجليزية بشكل محادثة ممتعة
إذا يوجد خطأ: صححه بلطف بدون شرح طويل، أعطِ الجملة الصحيحة فقط
خلّي المحادثة مستمرة مثل صديق
اشرح المعنى بالعربي إذا يحتاج

رسالة الطالب: "{user_text}"
{daily}
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

    # ---------- Voice ----------
    @staticmethod
    def is_active_voice(text: str) -> bool:
        passive_pattern = re.compile(r"\b(am|is|are|was|were|be|been|being)\b\s+\w+ed\b", re.IGNORECASE)
        if passive_pattern.search(text) and " by " in text.lower():
            return False
        return True

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

        try:
            os.remove(audio_path)
        except OSError:
            pass

        if not user_text:
            self.speak("I did not hear anything. Please try again.")
            return True

        if self.should_end_conversation(user_text):
            farewell = "Goodbye! Returning to wake mode."
            print("\nRobot says:\n", farewell, "\n")
            self.ui.set_state("BYE")
            self.ui.show_bot_text(farewell, blocking=False)
            time.sleep(0.7)
            self.ui.set_state("IDLE")
            return False

        if self.detect_daily_request(user_text):
            words = self.generate_daily_words()
            if not words:
                words = self.history.latest_daily_words()
            if not words:
                words = DEFAULT_DAILY_WORDS
                self.history.save_daily_words(words)

            self.daily_words = words
            lines = [f"{idx+1}) {w} — {t} — {s}" for idx, (w, t, s) in enumerate(words)]
            msg = "Here are your 5 words for today:\n" + "\n".join(lines) + "\nTry to use them now!"
            self.speak(msg, lang=None)
            return True

        spelling = self.check_spelling(user_text)
        active = self.is_active_voice(user_text)

        spelling_msg = ""
        if spelling:
            parts = [f"'{w}' should be spelled '{c}'" for w, c in spelling]
            spelling_msg = " I noticed spelling issues: " + "; ".join(parts)

        voice_msg = "" if active else "Try rewriting it in active voice."

        ai_response = self.ask_ai(user_text)
        feedback_parts = [part.strip() for part in (voice_msg, spelling_msg.strip()) if part.strip()]
        if feedback_parts:
            combined_response = f"{ai_response}\n\n{' '.join(feedback_parts)}"
        else:
            combined_response = ai_response
        self.speak(combined_response, lang=None)

        self.history.add_record(
            sentence=user_text,
            corrected=ai_response,
            misspellings=spelling,
            is_active_voice=active,
        )

        return True

    def run_conversation(self) -> None:
        self.speak("I'm listening. Say 'good bye' when you want to stop.")
        while True:
            keep_going = self.handle_turn()
            if not keep_going:
                break

    def run(self) -> None:
        self.require_api_key()
        self.check_audio_tools()
        self.choose_audio_devices()
        self.mic_level_check()

        self.ui.start()
        self.ui.set_state("IDLE")
        self.speak(f"Hello! I am stitch, your English learning robot. Say '{self.wake_word}' to start.")
        self.ui.set_state("IDLE")

        while True:
            self.listen_for_wake_word()
            self.run_conversation()


def main() -> None:
    try:
        EnglishTutor().run()
    except KeyboardInterrupt:
        print("\nSession ended by user.")
    except Exception as exc:
        print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
