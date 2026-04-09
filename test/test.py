import os
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import pyttsx3
import time
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY is missing. Add it to your .env file (not committed).")

model = whisper.load_model("base")

def select_mic():
    devices = sd.query_devices()
    print("\n🎧 Available input devices:")
    input_devices = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            input_devices.append(i)
            print(f"  [{i}] {d['name']} ({d['max_input_channels']} ch)")
    print()
    while True:
        choice = input("Select mic index: ").strip()
        if choice.isdigit() and int(choice) in input_devices:
            print(f"✅ Using: {devices[int(choice)]['name']}\n")
            return int(choice)
        print("❌ Invalid choice, try again.")

# ✅ موديل مضمون يشتغل
API_URL = "https://router.huggingface.co/hf-inference/models/microsoft/phi-2"
headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

def speak(text):
    print("🤖 Robot:", text)

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.setProperty("volume", 1)

    engine.say(text)
    engine.runAndWait()
    engine.stop()

    time.sleep(0.5)

def record_audio(mic_index, duration=5, fs=16000):
    print("🎤 Speak now...")

    sd.default.device = (mic_index, None)
    sd.default.samplerate = fs
    sd.default.channels = 1

    recording = sd.rec(int(duration * fs), dtype="int16")
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, fs, recording)

    return temp_file.name

# ✅ دالة AI مصححة
def ask_ai(text):
    try:
        payload = {
            "inputs": text
        }

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        print("STATUS CODE:", response.status_code)
        print("RAW RESPONSE:", response.text)

        # Return the full error details if the request failed
        if response.status_code != 200:
            return f"API ERROR (HTTP {response.status_code}): {response.text}"

        try:
            data = response.json()
        except Exception as e:
            return f"Invalid JSON response: {response.text}"

        if isinstance(data, dict) and "error" in data:
            return f"API ERROR: {data['error']}"

        if isinstance(data, list):
            return data[0].get("generated_text", "No response")

        return str(data)

    except requests.exceptions.ConnectionError as e:
        return f"CONNECTION ERROR: {e}"
    except requests.exceptions.Timeout:
        return "TIMEOUT ERROR: API did not respond within 60 seconds"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"UNEXPECTED ERROR: {type(e).__name__}: {e}"
    

    
def main():
    mic_index = select_mic()
    speak("Hello! I am your AI English learning robot.")

    # while True:
    #     audio_path = record_audio(mic_index, duration=5)

    #     print("⏳ Processing...")
    #     result = model.transcribe(audio_path)

    #     user_text = result["text"].strip().lower()
    #     print("🧑 You:", user_text)

    #     if not user_text:
    #         speak("I did not hear anything. Please try again.")
    #         continue

    response = ask_ai("hi")
    speak(response)

    # if "goodbye" in user_text or "exit" in user_text:
    #     speak("Goodbye!")
        # break

if __name__ == "__main__":
    main()
