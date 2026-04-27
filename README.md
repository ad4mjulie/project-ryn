# Ryn 

A local, sarcastic AI voice assistant. No cloud STT, no cloud TTS, no bullshit.

Ryn lives on a Google Nest Mini via Bluetooth and talks to you.

## Stack

| Layer | Tech |
|-------|------|
| Speech-to-Text | Whisper tiny (local, via `openai-whisper`) |
| AI Brain | Gemini 1.5 Flash (streaming) |
| Text-to-Speech | Kokoro TTS (local, `am_onyx` voice) |
| Audio Output | `sounddevice` → Mac default output (Bluetooth speaker) |

## Setup

### 1. System Dependencies
```bash
brew install espeak-ng portaudio
```

### 2. Create & Activate Virtual Environment
```bash
~/.pyenv/versions/3.10.14/bin/python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch (CPU-only, faster download)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Python packages
```bash
pip install -r requirements.txt
```

### 5. Set your API key
Copy `.env.example` to `.env` and add your Gemini API key:
```bash
cp .env.example .env
# edit .env and paste your GEMINI_API_KEY
```
Get a free key at [aistudio.google.com](https://aistudio.google.com).

### 6. Run
```bash
source venv/bin/activate
python ryn.py
```
Make sure your Mac's default audio output is set to your Bluetooth speaker first.

## How it works

1. Listens to mic continuously via `SpeechRecognition`
2. Detects pause → transcribes locally with Whisper tiny
3. Sends transcript to Gemini 1.5 Flash with full conversation context
4. Buffers streaming tokens into sentences (splits on `.`, `!`, `?`)
5. Sends each sentence to Kokoro TTS as it arrives
6. Plays audio immediately through Mac default output
7. Saves full history to `chat_history.json` — Ryn remembers across sessions

## Notes

- First run downloads Whisper tiny (~75MB) and Kokoro model (~300MB) automatically
- `chat_history.json` is gitignored — your conversations stay local
- `.env` is gitignored — your API key stays local
