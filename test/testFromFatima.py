import os
import sys
import time
import tempfile
import platform
import subprocess
from pathlib import Path
import requests
import sounddevice as sd
from dotenv import load_dotenv
from gtts import gTTS
import wave
import platform

# Prefer the official PyAudio module; fall back to PyAudioWPatch wheels for
# Python 3.14 where upstream wheels are not yet published.
try:
    import pyaudio  # type: ignore
except ImportError:  # pragma: no cover - runtime compatibility shim
    import pyaudiowpatch as pyaudio  # type: ignore
    sys.modules["pyaudio"] = pyaudio

import speech_recognition as sr

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
# Bias Whisper to Arabic by default; override via .env if needed
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "ar")

# Cross-platform mp3 playback (borrowed from testapi.py)
def play_audio(file_path: str) -> None:
    system = platform.system().lower()

    # Preferred: playsound on Windows to avoid opening a video player
    if system == "windows":
        try:
            from playsound import playsound
            playsound(file_path)
            return
        except Exception:
            pass  # fallback to other methods

    # On Linux, try mpg123 then fallback to ffplay
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

class ScorpionTutor:
    def __init__(self):
        self.history = [
            {"role": "system", "content": "You are an English teacher for Arabic speakers. Correct mistakes, explain in Arabic, and keep it conversational."}
        ]
        self.microphone_index = self.choose_microphone()
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
    def speak(self, text):
        lang = "ar" if any("\u0600" <= ch <= "\u06FF" for ch in text) else "en"
        print(f"\n[Scorpion]: {text}")

        # Use gTTS + play_audio (same approach as testapi.py) for better Arabic support.
        try:
            tts = gTTS(text=text, lang=lang)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tts.save(f.name)
                play_audio(f.name)
            return
        except Exception as e:
            print(f"TTS Error ({lang}): {e}")

    # (pyttsx3 voice selection removed; gTTS handles Arabic/English automatically)

    def choose_microphone(self):
        """Prompt once at startup to pick an input device."""
        try:
            names = sr.Microphone.list_microphone_names()
            if not names:
                print("No microphones detected; defaulting to system default.")
                return None
            print("\nAvailable microphones:")
            for i, name in enumerate(names):
                print(f"  [{i}] {name}")
            choice = input("Select mic index (press Enter for default): ").strip()
            if choice == "":
                return None
            idx = int(choice)
            if 0 <= idx < len(names):
                return idx
            print("Invalid index; using default.")
            return None
        except Exception as e:
            print(f"Mic selection error: {e}; using default.")
            return None

    def listen(self):
        with sr.Microphone(device_index=self.microphone_index) as source:
            print("\n[Listening...]")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio_data = self.recognizer.listen(source, timeout=10, phrase_time_limit=8)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_data.get_wav_data())
                    return f.name
            except sr.WaitTimeoutError:
                return None

    def transcribe(self, audio_path):
        if not audio_path: return ""
        try:
            payload = {"model": "whisper-1"}
            if STT_LANGUAGE:
                payload["language"] = STT_LANGUAGE
            with open(audio_path, "rb") as f:
                response = requests.post(
                    f"{OPENAI_BASE}/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data=payload
                )
            if response.status_code != 200:
                print(f"STT HTTP {response.status_code}: {response.text[:500]}")
                return ""
            data = response.json()
            text = data.get("text", "") if isinstance(data, dict) else ""
            if not text:
                print(f"STT empty result: {data}")
            return text
        except Exception as e:
            print(f"STT Error: {e}")
            return ""

    def get_ai_response(self, user_text):
        self.history.append({"role": "user", "content": user_text})
        try:
            response = requests.post(
                f"{OPENAI_BASE}/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": OPENAI_MODEL,
                    "messages": self.history,
                    "temperature": 0.7
                }
            )
            if response.status_code != 200:
                return f"Chat HTTP {response.status_code}: {response.text[:500]}" 
            data = response.json()
            reply = data["choices"][0]["message"]["content"]
            self.history.append({"role": "assistant", "content": reply})
            if len(self.history) > 11:
                self.history = [self.history[0]] + self.history[-10:]
            return reply
        except Exception as e:
            return f"Error: {e}"

    def run(self):
        self.speak("Hello! I am Scorpion. Let's practice English.")
        while True:
            audio_path = self.listen()
            if not audio_path:
                continue
            user_text = self.transcribe(audio_path)
            if not user_text.strip():
                continue
            print(f"[You]: {user_text}")
            if any(word in user_text.lower() for word in ["exit", "stop", "goodbye"]):
                self.speak("Goodbye!")
                break
            response = self.get_ai_response(user_text)
            self.speak(response)

if __name__ == "__main__":
    tutor = ScorpionTutor()
    tutor.run()
