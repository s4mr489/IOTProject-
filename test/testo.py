from __future__ import annotations

import array
import json
import os
import platform
import re
import sqlite3
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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


# =========================================================
# UI ASSETS - Raspberry Pi paths
# ضع الصور داخل:
# /home/stitch/Robot/test/images
# =========================================================
IMAGES_DIR = Path("/home/stitch/Robot/test/images")

IDLE_GIF = str(IMAGES_DIR / "idle.gif")
WAKE_GIF = str(IMAGES_DIR / "wake.gif")
LISTEN_GIF = str(IMAGES_DIR / "listening.gif")
TALK_GIF = str(IMAGES_DIR / "talking.gif")
BYE_GIF = str(IMAGES_DIR / "bye.gif")


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
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
DB_PATH = Path(__file__).resolve().parents[1] / "history.db"

WAKE_RECORD_SECONDS = 2.0
TURN_RECORD_SECONDS = 6.0

DEFAULT_DAILY_WORDS: List[Tuple[str, str, str]] = [
    ("adapt", "يتأقلم", "I adapt quickly to new teams."),
    ("confident", "واثق", "She feels confident about the exam."),
    ("gather", "يجمع", "Let's gather ideas before we start."),
    ("improve", "يحسّن", "Daily practice will improve your accent."),
    ("remind", "يذكّر", "Please remind me about the meeting."),
]


# =========================================================
# UI CONTROLLER
# حالياً يطبع الحالة والنص.
# لاحقاً نربطه فعلياً مع GIF window.
# =========================================================
class UIController:
    def __init__(self) -> None:
        self.state = "IDLE"
        self.current_text = ""
        self._lock = threading.Lock()

        self.assets = {
            "IDLE": IDLE_GIF,
            "WAKE": WAKE_GIF,
            "LISTENING": LISTEN_GIF,
            "TALKING": TALK_GIF,
            "BYE": BYE_GIF,
        }

    def set_state(self, state: str) -> None:
        self.state = state
        gif_path = self.assets.get(state, "")
        print(f"\n[STATE] {state}")
        if gif_path:
            print(f"[GIF] {gif_path}")

    def show_user_text(self, text: str) -> None:
        self._word_by_word("USER", text, delay=0.18)

    def show_bot_text(self, text: str) -> None:
        self._word_by_word("BOT", text, delay=0.22)

    def show_bot_text_async(self, text: str) -> threading.Thread:
        thread = threading.Thread(target=self.show_bot_text, args=(text,), daemon=True)
        thread.start()
        return thread

    def _word_by_word(self, who: str, text: str, delay: float = 0.22) -> None:
        words = text.split()
        if not words:
            return

        with self._lock:
            self.current_text = ""

            print(f"\n{who}: ", end="", flush=True)

            for word in words:
                self.current_text += word + " "
                print(f"\r{who}: {self.current_text.strip()}", end="", flush=True)
                time.sleep(delay)

            print()


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
        self.ui = UIController()

        self.wake_word = wake_word.lower()
        self.history = HistoryRepository(DB_PATH)
        self.spellchecker = SpellChecker(language="en") if SpellChecker else None
        self.stop_phrases = STOP_PHRASES
        self.daily_words: List[Tuple[str, str, str]] = []

        self.source_name: str = ""
        self.sink_name: str = ""
        self.input_sample_rate: int = SAMPLE_RATE
        self.input_channels: int = MIC_CHANNELS

    # ---------- General helpers ----------
    @staticmethod
    def run_command(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(args, text=True, capture_output=True, check=check)

    @staticmethod
    def command_output(args: List[str]) -> str:
        return subprocess.check_output(args, text=True).strip()

    @staticmethod
    def command_exists(name: str) -> bool:
        return subprocess.run(
            ["bash", "-lc", f"command -v {name} >/dev/null 2>&1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0

    def require_api_key(self) -> None:
        global HF_API_KEY
        if not HF_API_KEY:
            load_dotenv(ENV_PATH, override=False)
            HF_API_KEY = os.getenv("HF_API_KEY")
        if not HF_API_KEY and not OPENAI_API_KEY:
            raise RuntimeError("HF_API_KEY or OPENAI_API_KEY is missing. Add it to your .env file.")

    def check_audio_tools(self) -> None:
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
        self.run_command(["pactl", "set-default-source", self.source_name], check=False)
        self.run_command(["pactl", "set-default-sink", self.sink_name], check=False)
        os.environ["PULSE_SOURCE"] = self.source_name
        os.environ["PULSE_SINK"] = self.sink_name

    def choose_audio_devices(self) -> None:
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
    def speak(self, text: str, lang: Optional[str] = "en") -> None:
        if lang is None:
            lang = self.choose_tts_lang(text)

        self.ui.set_state("TALKING")
        print("\nRobot says:\n", text, "\n")

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

                    text_thread = self.ui.show_bot_text_async(text)
                    self.play_audio(output)
                    text_thread.join()

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

            text_thread = self.ui.show_bot_text_async(text)
            self.play_audio(output)
            text_thread.join()

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
                    self.ui.set_state("WAKE")
                    print("Wake word detected.")
                    return

                print(".", end="", flush=True)

            except Exception as exc:
                print(f"\nWake-word stream error: {exc}")
                time.sleep(1)

    # ---------- Main interaction ----------
    def should_end_conversation(self, user_text: str) -> bool:
        normalized = user_text.lower()
        return any(phrase in normalized for phrase in (*self.stop_phrases, "exit", "stop"))

    def handle_turn(self) -> bool:
        self.ui.set_state("LISTENING")

        audio_path = self.record_for_seconds(TURN_RECORD_SECONDS)
        print("Processing speech...")
        user_text = self.transcribe_audio(audio_path).strip()
        print("You said:", user_text)

        try:
            os.remove(audio_path)
        except OSError:
            pass

        if user_text:
            self.ui.show_user_text(user_text)

        if not user_text:
            self.speak("I did not hear anything. Please try again.")
            return True

        if self.should_end_conversation(user_text):
            self.ui.set_state("BYE")
            self.speak("Goodbye! Returning to wake mode.", lang="en")
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

        voice_msg = "Great, your sentence is in active voice." if active else "Try rewriting it in active voice."

        ai_response = self.ask_ai(user_text)
        combined_response = f"{ai_response}\n\n{voice_msg}{spelling_msg}"
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

        self.ui.set_state("IDLE")
        self.speak(f"Hello! I am stitch, your English learning robot. Say '{self.wake_word}' to start.")

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