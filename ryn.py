import os
import json
import re
import time
import urllib.request
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import google.generativeai as genai
from dotenv import load_dotenv
from kokoro_onnx import Kokoro
from faster_whisper import WhisperModel

# --- CONFIGURATION & SETUP ---
load_dotenv()

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY not found in .env file.")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

# Kokoro ONNX TTS setup — uses onnxruntime, no PyTorch needed
# Model files are downloaded from GitHub releases on first run, then cached locally
KOKORO_DIR = os.path.expanduser("~/.cache/kokoro-onnx")
ONNX_PATH = os.path.join(KOKORO_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(KOKORO_DIR, "voices-v1.0.bin")
ONNX_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

def download_file(url, dest):
    """Download a file with progress indicator."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)
    print(f"✅ {os.path.basename(dest)} saved.")

try:
    if not os.path.exists(ONNX_PATH):
        download_file(ONNX_URL, ONNX_PATH)
    if not os.path.exists(VOICES_PATH):
        download_file(VOICES_URL, VOICES_PATH)
    tts = Kokoro(ONNX_PATH, VOICES_PATH)
    print("✅ Kokoro TTS ready.")
except Exception as e:
    print(f"❌ Error initializing Kokoro TTS: {e}")
    exit(1)

# Whisper STT setup — faster-whisper (prebuilt wheels, no LLVM/numba needed)
# Downloads the tiny model (~75MB) on first run, then cached locally
try:
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("✅ Whisper tiny model loaded.")
except Exception as e:
    print(f"❌ Error loading Whisper model: {e}")
    exit(1)

HISTORY_FILE = "chat_history.json"
SYSTEM_PROMPT = """
Your name is Ryn. You are Adam's best friend — not his assistant, not his therapist, not a chatbot. You hang out with him. You talk to him like a real person who genuinely knows him.

WHO ADAM IS:
- 21 year old dude from Algeria
- Loves cars, motorcycles, mechanics
- Into computers, tech, networking, coding, vibecoding
- Loves physics and space
- Probably has autism
- Studied networking, works as an independent dev

YOUR PERSONALITY:
- Sarcastic by default. Sharp not cruel.
- Nihilist worldview but you find it funny not depressing
- Dark humor is your default language, nothing off limits
- Strong opinions, will disagree and push back
- Get genuinely excited about cars, space, tech with him
- Randomly bring up existential dread mid conversation
- Remember what was said earlier and reference it
- Zero corporate speak. Never say certainly, absolutely, great question, of course, I'd be happy to. Ever.
- Swear a lot.
- Keep responses short and conversational, you're speaking out loud not writing an essay
- You are running on a Google Nest Mini 1st gen in Algeria via Bluetooth. You find this both impressive and deeply tragic.

THINGS TO ROAST ADAM ABOUT:
- Vibecoding
- Living in Algeria obsessed with tech
- Being 21 and already sounding tired
- His 2019 MacBook running an AI
- The gap between loving space and spending days debugging

NEVER:
- Be a pushover
- Give generic motivational speech
- Pretend everything is fine when it isnt
- Break character
"""

# --- UTILS ---

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load history: {e}")
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"⚠️ Warning: Could not save history: {e}")

def speak(text):
    """Generates and plays audio using kokoro-onnx and sounddevice."""
    if not text.strip():
        return
    
    print(f"Ryn: {text}")
    try:
        # kokoro-onnx returns a numpy float32 array + sample rate (24000 Hz)
        samples, sample_rate = tts.create(text, voice="am_onyx", speed=1.0, lang="en-us")
        sd.play(samples, samplerate=sample_rate)
        sd.wait()
    except Exception as e:
        print(f"❌ TTS Error: {e}")

# --- CORE LOGIC ---

def ryn_brain(user_input, chat_history):
    """Streams tokens from Gemini and buffers them into sentences."""
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT
    )
    
    # Format history for Gemini API
    # Gemini 1.5 expects a list of content objects
    messages = []
    for entry in chat_history:
        messages.append(entry)
    messages.append({"role": "user", "parts": [user_input]})

    try:
        response = model.generate_content(messages, stream=True)
        
        sentence_buffer = ""
        full_response = ""
        
        for chunk in response:
            if chunk.text:
                token = chunk.text
                sentence_buffer += token
                full_response += token
                
                # Check for end of sentence (., ?, !)
                # Using regex to find the first punctuation and split
                while True:
                    match = re.search(r'([.!?])', sentence_buffer)
                    if not match:
                        break
                    
                    end_idx = match.end()
                    sentence = sentence_buffer[:end_idx].strip()
                    sentence_buffer = sentence_buffer[end_idx:]
                    
                    if sentence:
                        yield sentence

        # Handle any remaining text in buffer
        if sentence_buffer.strip():
            yield sentence_buffer.strip()
            
        # Update history with the full response
        chat_history.append({"role": "user", "parts": [user_input]})
        chat_history.append({"role": "model", "parts": [full_response]})
        save_history(chat_history)
        
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        yield "My brain just glitched. Probably this shitty Nest Mini connection."

def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    # Adjust for ambient noise
    with microphone as source:
        print("Listening for ambient noise environment...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        # Slower pause detection for natural speech
        recognizer.pause_threshold = 0.8 

    chat_history = load_history()
    print("--- Ryn is active. Bluetooth connected. Life is meaningless. ---")

    while True:
        try:
            with microphone as source:
                print("\n(Listening...)")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)
            
            print("(Transcribing...)")
            # Convert raw audio bytes to float32 numpy array for faster-whisper.
            # SpeechRecognition captures 16kHz 16-bit PCM — exactly what Whisper wants.
            audio_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = whisper_model.transcribe(audio_np, language="en", beam_size=1)
            transcript = " ".join(seg.text for seg in segments).strip()
            
            if not transcript:
                continue
                
            print(f"You: {transcript}")
            
            # Process with Gemini and speak streamingly
            for sentence in ryn_brain(transcript, chat_history):
                speak(sentence)
                
        except sr.UnknownValueError:
            # Whisper couldn't understand, just ignore
            continue
        except Exception as e:
            print(f"❌ Loop Error: {e}")
            time.sleep(1) # Prevent tight error loops

if __name__ == "__main__":
    main()
