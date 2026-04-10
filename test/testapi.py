"""Voice English tutor with wake-word streaming, spelling checks, and history.

Key features:
- Object-oriented design (EnglishTutor + HistoryRepository helpers).
- SQLite storage of the last N interactions (sentence, corrected sentence, spelling issues, active/passive flag).
- Wake-word streaming listener that starts a session when you say the wake word.
- Active/passive voice detection heuristic.
- Spelling check with friendly correction (uses `pyspellchecker` if installed).
- Robust error handling (network, mic/device, transcription).
"""

from __future__ import annotations

import json
import os
import platform
import re
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import requests
import sounddevice as sd
from dotenv import load_dotenv
from gtts import gTTS
from scipy.io.wavfile import write

try:
    from playsound import playsound  # lightweight, no UI pop-up on Windows
except Exception:  # pragma: no cover
    playsound = None

try:
    from spellchecker import SpellChecker  # optional dependency
except Exception:  # pragma: no cover
    SpellChecker = None

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)
HF_API_KEY = os.getenv("HF_API_KEY")
MIC_DEVICE = os.getenv("MIC_DEVICE")  # optional: device name substring or index

STT_MODEL = "openai/whisper-large-v3"
HF_ROUTER = "https://router.huggingface.co/hf-inference/models"
HF_LEGACY = "https://api-inference.huggingface.co/models"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1-hd")  # higher quality; switch to tts-1 for speed
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")     # other options: verse, hollow, shimmer, etc.
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))

SAMPLE_RATE = 16_000
CLIP_SECONDS = 7
HISTORY_LIMIT = 15
WAKE_WORD = os.getenv("WAKE_WORD", "stitch").lower()
STOP_PHRASES = tuple(
    w.strip().lower() for w in os.getenv("STOP_PHRASES", "good bye,goodbye,good-bye").split(",") if w.strip()
)
DB_PATH = Path(__file__).resolve().parents[1] / "history.db"
MAX_RECORD_SECONDS = 20
SILENCE_HOLD_SECONDS = 0.9  # how long of silence to consider the utterance finished
MIN_ACTIVE_RMS = 300  # floor to avoid treating very low noise as speech
DEFAULT_DAILY_WORDS: List[Tuple[str, str, str]] = [
    ("adapt", "يتأقلم", "I adapt quickly to new teams."),
    ("confident", "واثق", "She feels confident about the exam."),
    ("gather", "يجمع", "Let's gather ideas before we start."),
    ("improve", "يحسّن", "Daily practice will improve your accent."),
    ("remind", "يذكّر", "Please remind me about the meeting."),
]


# ---------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------
class HistoryRepository:
    """SQLite repository that keeps the last N interactions."""

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
            # keep only last 10 sets
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

    def last(self, n: Optional[int] = None) -> List[Tuple]:
        n = n or self.limit
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                "SELECT ts, sentence, corrected, misspellings, is_active_voice FROM history ORDER BY id DESC LIMIT ?",
                (n,),
            )
            return cur.fetchall()
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
        self.daily_words: List[Tuple[str, str, str]] = []  # (word, translation, sentence)

    # ---------- Setup helpers ----------
    def require_api_key(self) -> None:
        global HF_API_KEY
        if not HF_API_KEY:
            load_dotenv(ENV_PATH, override=False)
            HF_API_KEY = os.getenv("HF_API_KEY")
        if not HF_API_KEY and not OPENAI_API_KEY:
            raise RuntimeError("HF_API_KEY or OPENAI_API_KEY is missing. Add it to your .env file.")

    def select_input_device(self) -> None:
        """Let the user pick the mic; reuses prior helper."""
        devices = sd.query_devices()

        def set_device(idx: int) -> None:
            sd.default.device = (idx, None)
            print(f"Using input device: {devices[idx]['name']} (index {idx})")

        suggested = str(MIC_DEVICE) if MIC_DEVICE is not None else None
        for idx, dev in enumerate(devices):
            if suggested:
                break
            if dev.get("max_input_channels", 0) > 0 and "soundcore" in str(dev.get("name", "")).lower():
                suggested = str(idx)
                break

        print("Available input devices:")
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                print(f"{idx}: {dev['name']} (inputs: {dev['max_input_channels']})")

        prompt_default = f"[default {suggested}]" if suggested is not None else "[press Enter for system default]"
        choice = input(f"Enter mic index to use {prompt_default}: ").strip()
        target = choice or suggested
        if target is not None:
            for idx, dev in enumerate(devices):
                if target == str(idx) or target.lower() in str(dev.get("name", "")).lower():
                    set_device(idx)
                    return
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        print(f"Using system default input device (index {default_in}).")

    def mic_level_check(self, duration: float = 1.0) -> None:
        print("Mic test: speak for 1 second...")
        fs = SAMPLE_RATE
        sd.default.samplerate = fs
        sd.default.channels = 1
        recording = sd.rec(int(duration * fs), dtype="int16")
        sd.wait()
        rms = (recording.astype("float32") ** 2).mean() ** 0.5
        print(f"Mic RMS level: {rms:.2f} (values near 0 mean silence)")

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
        except Exception as exc:  # pragma: no cover
            print("Daily words generation error:", exc)
            return []

    # ---------- Audio I/O ----------
    def record_until_silence(
        self,
        max_seconds: int = MAX_RECORD_SECONDS,
        silence_hold: float = SILENCE_HOLD_SECONDS,
        fs: int = SAMPLE_RATE,
    ) -> str:
        """
        Stream microphone audio and stop a little after the user finishes speaking.
        Uses a dynamic noise floor so it waits for your full sentence instead of cutting early.
        """
        sd.default.samplerate = fs
        sd.default.channels = 1

        block_dur = 0.2  # seconds
        block_frames = int(fs * block_dur)
        frames = []

        # Measure ambient noise for a moment to set a threshold
        try:
            warmup = sd.rec(int(fs * 0.5), dtype="int16")
            sd.wait()
            noise_rms = float((warmup.astype("float32") ** 2).mean() ** 0.5)
        except Exception:
            noise_rms = 0.0
        voice_threshold = max(MIN_ACTIVE_RMS, noise_rms * 3.5)

        print("Listening... (auto-stops after silence)")
        last_voice_time = None
        start_time = time.time()

        with sd.InputStream(
            samplerate=fs,
            channels=1,
            dtype="int16",
            blocksize=block_frames,
        ) as stream:
            while True:
                data, _ = stream.read(block_frames)
                frames.append(data.copy())

                rms = float((data.astype("float32") ** 2).mean() ** 0.5)
                now = time.time()
                if rms >= voice_threshold:
                    last_voice_time = now

                if last_voice_time is not None and (now - last_voice_time) >= silence_hold:
                    break

                if (now - start_time) >= max_seconds:
                    print("Max record time reached; stopping.")
                    break

        # save to wav
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_file.name, fs, np.concatenate(frames))
        return temp_file.name

    def play_audio(self, file_path: str) -> None:
        """Cross-platform mp3 playback with minimal dependencies."""
        system = platform.system().lower()
        if system == "windows" and playsound:
            try:
                playsound(file_path)
                return
            except Exception:
                pass

        for player in ("mpg123", "ffplay"):
            try:
                subprocess.run(
                    [player, "-nodisp", "-autoexit", file_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                return
            except Exception:
                continue

        if system == "windows":
            os.startfile(file_path)  # type: ignore[attr-defined]
            return

        print("Could not find an mp3 player. Install 'mpg123' or 'ffmpeg'.")

    # ---------- Transcription ----------
    def transcribe_audio(self, file_path: str) -> str:
        """Transcribe using OpenAI Whisper if key is present, else HF fallback."""
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
            except Exception as exc:  # pragma: no cover
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
                if response.status_code == 404 and "Not Found" in response.text:
                    print("Model not found on router; trying fallback to whisper-large-v3.")
                    return self.transcribe_audio_fallback(post_audio)
            except requests.exceptions.RequestException as exc:
                print(f"Transcription network error (attempt {attempt+1}/3): {exc}")
            except Exception as exc:  # pragma: no cover
                print("Transcription error:", exc)
                return ""
            time.sleep(1 + attempt)
        return ""

    @staticmethod
    def transcribe_audio_fallback(post_fn) -> str:
        fallback_model = "openai/whisper-large-v3"
        try:
            response = post_fn(fallback_model)
            if response.status_code == 200:
                return response.json().get("text", "")
            print(f"Fallback transcription error ({response.status_code}):", response.text)
        except Exception as exc:
            print("Fallback transcription error:", exc)
        return ""

    # ---------- LLM prompt ----------
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
speak in a palyfull alien-like tone use short excited sentencse slighty childish and curious ractions in a pitch tone
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

        # Choose per-language model with env overrides
        model = (
            os.getenv("OPENAI_MODEL_AR", OPENAI_MODEL) if is_arabic else os.getenv("OPENAI_MODEL_EN", OPENAI_MODEL)
        )

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
        except Exception as exc:  # pragma: no cover
            return f"AI ERROR: {exc}"

    # ---------- Text-to-speech ----------
    def speak(self, text: str, lang: Optional[str] = "en") -> None:
        if lang is None:
            lang = self.choose_tts_lang(text)
        print("\nRobot says:\n", text, "\n")

        # 1) Preferred: OpenAI neural TTS (much more natural)
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
                else:  # fall back if API fails
                    print(f"TTS API {resp.status_code}: {resp.text[:200]}")
            except Exception as exc:  # pragma: no cover
                print(f"TTS network error: {exc}")

        # 2) Fallback: gTTS
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
            time.sleep(0.5)  # small gap before next recording
        except Exception as exc:  # pragma: no cover
            print(f"TTS Error ({lang}):", exc)

    # ---------- Spelling ----------
    def check_spelling(self, text: str) -> List[Tuple[str, str]]:
        """Return list of (wrong, corrected) pairs."""
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

    # ---------- Voice (active/passive) ----------
    @staticmethod
    def is_active_voice(text: str) -> bool:
        """Simple heuristic: passive if 'be' verb + past participle + optional 'by'."""
        passive_pattern = re.compile(r"\b(am|is|are|was|were|be|been|being)\b\s+\w+ed\b", re.IGNORECASE)
        if passive_pattern.search(text) and " by " in text.lower():
            return False
        return True

    # ---------- Wake word streaming ----------
    def listen_for_wake_word(self) -> None:
        """Stream short buffers until the wake word is heard."""
        print(f"Say the wake word '{self.wake_word}' to start...")
        block_duration = 1.5  # seconds
        frames_per_block = int(SAMPLE_RATE * block_duration)
        buffer: List[np.ndarray] = []

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=frames_per_block,
        ) as stream:
            while True:
                try:
                    data, _ = stream.read(frames_per_block)
                    buffer.append(data.copy())
                    # keep only a few blocks to avoid unbounded growth
                    if len(buffer) > 4:
                        buffer.pop(0)

                    merged = np.concatenate(buffer)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    write(temp_file.name, SAMPLE_RATE, merged)
                    text = self.transcribe_audio(temp_file.name).lower().strip()
                    if self.wake_word in text:
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
        """Record, transcribe, spell-check, ask AI, store history. Returns False to end active convo."""
        audio_path = self.record_until_silence()
        print("Processing speech...")
        user_text = self.transcribe_audio(audio_path).strip()
        print("You said:", user_text)

        if not user_text:
            self.speak("I did not hear anything. Please try again.")
            return True

        # User asks for daily 5 words
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

        if self.should_end_conversation(user_text):
            self.speak("Goodbye! Returning to wake mode.", lang="en")
            return False
        return True

    def run_conversation(self) -> None:
        """Stay in active mode until the user says a stop phrase."""
        self.speak("I'm listening. Say 'good bye' when you want to stop.")
        while True:
            keep_going = self.handle_turn()
            if not keep_going:
                break

    def run(self) -> None:
        self.require_api_key()
        self.select_input_device()
        self.mic_level_check()
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
