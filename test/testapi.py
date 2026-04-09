"""Simple voice English tutor for Arabic speakers.

This script records a short utterance, sends it to a speech-to-text model
hosted on Hugging Face, asks a lightweight language model for feedback, and
plays the reply back using text-to-speech. It is designed to be runnable on a
Raspberry Pi 3 with a USB microphone and speaker/headphones.
"""

import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import sounddevice as sd
from dotenv import load_dotenv
from gtts import gTTS
from scipy.io.wavfile import write

try:
    from playsound import playsound  # lightweight, no UI pop-up on Windows
except Exception:  # pragma: no cover
    playsound = None

# ---------------------------------------------------------------
# Configuration

# ---------------------------------------------------------------
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)
HF_API_KEY = os.getenv("HF_API_KEY")
MIC_DEVICE = os.getenv("MIC_DEVICE")  # optional: device name substring or index


# Models chosen to be small and free via Hugging Face Inference API
STT_MODEL = "openai/whisper-large-v3"  # router-supported; raw audio/wav (HF fallback)
HF_ROUTER = "https://router.huggingface.co/hf-inference/models"
HF_LEGACY = "https://api-inference.huggingface.co/models"
# OpenAI-compatible settings (option 2)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

SAMPLE_RATE = 16_000
CLIP_SECONDS = 5


def require_api_key() -> None:
    """Ensure the Hugging Face token exists before making HF calls."""
    global HF_API_KEY
    if not HF_API_KEY:
        load_dotenv(ENV_PATH, override=False)
        HF_API_KEY = os.getenv("HF_API_KEY")
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY is missing. Add it to your .env file.")


# ---------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------
def select_input_device() -> None:
    """Always let the user pick the microphone at startup.

    - Shows all input-capable devices with indexes.
    - Pre-fills with MIC_DEVICE (env) if present; otherwise uses the current default.
    - User can press Enter to accept the pre-filled choice.
    """
    devices = sd.query_devices()

    def set_device(idx: int) -> None:
        sd.default.device = (idx, None)
        print(f"Using input device: {devices[idx]['name']} (index {idx})")

    # Determine suggested default
    suggested = None
    if MIC_DEVICE is not None:
        suggested = str(MIC_DEVICE)
    else:
        # auto-suggest Soundcore / headset mics if present
        for idx, dev in enumerate(devices):
            name = str(dev.get("name", "")).lower()
            if dev.get("max_input_channels", 0) > 0 and "soundcore" in name:
                suggested = str(idx)
                break

    # List options
    print("Available input devices:")
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            print(f"{idx}: {dev['name']} (inputs: {dev['max_input_channels']})")

    # Build prompt text
    prompt_default = f"[default {suggested}]" if suggested is not None else "[press Enter for system default]"
    choice = input(f"Enter mic index to use {prompt_default}: ").strip()

    target = choice or suggested
    if target is not None:
        # Try to match by index first, else substring
        for idx, dev in enumerate(devices):
            if target == str(idx) or target.lower() in str(dev.get("name", "")).lower():
                set_device(idx)
                return
        print(f"Choice '{target}' not found. Using system default mic.")

    # Fallback: leave whatever default is
    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    print(f"Using system default input device (index {default_in}).")


def mic_level_check(duration: float = 1.0) -> None:
    """Record a short clip and print RMS level so the user knows the mic is live."""
    print("Mic test: speak for 1 second...")
    fs = SAMPLE_RATE
    sd.default.samplerate = fs
    sd.default.channels = 1
    recording = sd.rec(int(duration * fs), dtype="int16")
    sd.wait()
    rms = (recording.astype("float32") ** 2).mean() ** 0.5
    print(f"Mic RMS level: {rms:.2f} (values near 0 mean silence)")


def record_audio(duration: int = CLIP_SECONDS, fs: int = SAMPLE_RATE) -> str:
    """Capture microphone audio and return a temporary wav path."""
    print("Robot: please speak now...")

    sd.default.samplerate = fs
    sd.default.channels = 1

    recording = sd.rec(int(duration * fs), dtype="int16")
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, fs, recording)
    return temp_file.name


def play_audio(file_path: str) -> None:
    """Cross-platform mp3 playback with minimal dependencies."""
    system = platform.system().lower()

    # Preferred: playsound on Windows to avoid opening a video player
    if system == "windows" and playsound:
        try:
            playsound(file_path)
            return
        except Exception:
            pass  # fallback to other methods

    # On Raspberry Pi (Linux), try mpg123 then fallback to ffplay
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


# ---------------------------------------------------------------
# Speech → Text
# ---------------------------------------------------------------
def transcribe_audio(file_path: str) -> str:
    """Transcribe using OpenAI whisper if key is present, else HF fallback."""
    payload = Path(file_path)

    # Preferred: OpenAI-compatible Whisper
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

    # Fallback: Hugging Face router
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
        return requests.post(
            f"{HF_ROUTER}/{model}",
            headers=headers,
            data=data_bytes,
            timeout=120,
        )

    for attempt in range(3):
        try:
            response = post_audio(STT_MODEL)
            if response.status_code == 200:
                data = response.json()
                return data.get("text", "")
            print(f"Transcription API Error ({response.status_code}):", response.text)
            if response.status_code == 404 and "Not Found" in response.text:
                print("Model not found on router; trying fallback to whisper-large-v3.")
                return transcribe_audio_fallback(post_audio)
        except requests.exceptions.RequestException as exc:
            print(f"Transcription network error (attempt {attempt+1}/3): {exc}")
        except Exception as exc:  # pragma: no cover - best-effort logging
            print("Transcription error:", exc)
            return ""
        time.sleep(1 + attempt)
    return ""


def transcribe_audio_fallback(post_fn) -> str:
    """Fallback: try posting to whisper-large-v3 explicitly."""
    fallback_model = "openai/whisper-large-v3"
    try:
        response = post_fn(fallback_model)
        if response.status_code == 200:
            return response.json().get("text", "")
        print(f"Fallback transcription error ({response.status_code}):", response.text)
    except Exception as exc:
        print("Fallback transcription error:", exc)
    return ""


# ---------------------------------------------------------------
# English tutor prompt
# ---------------------------------------------------------------
def build_prompt(user_text: str) -> str:
    """Create the teaching prompt sent to the LLM."""
    return f"""
You are an English teacher for Arabic speakers.

Student said: "{user_text}"

Do the following in one short reply:
- Correct the sentence.
- Explain mistakes in Arabic.
- Provide an improved sentence in English.
- Ask a follow-up question.
Keep the tone friendly and concise.
""".strip()


def contains_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def choose_tts_lang(text: str) -> str:
    """Return 'ar' if Arabic letters detected, else 'en'."""
    return "ar" if contains_arabic(text) else "en"


def ask_ai(user_text: str) -> str:
    """Use OpenAI-compatible chat completions (option 2)."""
    prompt = build_prompt(user_text)

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
                "model": OPENAI_MODEL,
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


# ---------------------------------------------------------------
# Text → Speech
# ---------------------------------------------------------------
def speak(text: str, lang: str = "en") -> None:
    """Print and play speech; auto-detect Arabic for TTS if not specified."""
    if lang is None:
        lang = choose_tts_lang(text)
    print("\nRobot says:\n", text, "\n")

    try:
        tts = gTTS(text=text, lang=lang)
        output = "voice.mp3"
        tts.save(output)
        play_audio(output)
    except Exception as exc:  # pragma: no cover
        print(f"TTS Error ({lang}):", exc)


# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------
def main() -> None:
    require_api_key()
    select_input_device()
    mic_level_check()
    speak("Hello! I am scorpion your English learning robot. Say something in English.")

    while True:
        audio_path = record_audio()
        print("Processing speech...")
        user_text = transcribe_audio(audio_path).strip()
        print("You said:", user_text)

        if not user_text:
            speak("I did not hear anything. Please try again.")
            continue

        response = ask_ai(user_text)
        speak(response, lang=None)

        if any(word in user_text.lower() for word in ("exit", "stop", "goodbye")):
            speak("Goodbye!", lang="en")
            break


if __name__ == "__main__":
    main()
