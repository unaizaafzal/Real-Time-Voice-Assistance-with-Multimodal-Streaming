# Real-Time Voice Assistant

A production-grade voice assistant built with FastAPI, Pipecat, Deepgram, Groq, and ElevenLabs. Streams audio from browser microphone through a full STT → LLM → TTS pipeline with sub-second response latency.

## Architecture

Browser (AudioWorklet) → WebSocket → FastAPI → Pipecat Pipeline:
- **Deepgram** (streaming STT): transcribes speech in real time using linear16 PCM audio
- **Groq / Llama 3.3 70B** (LLM): generates responses with conversation history
- **ElevenLabs** (WebSocket TTS): synthesizes speech and streams audio chunks back to browser

## Latency Budget

Measured end-to-end on localhost:

| Hop | Latency |
|-----|---------|
| ASR → first LLM token | ~470ms |
| LLM → first TTS audio | ~390ms |
| **Total time to first audio** | **~860ms** |

## Graceful Degradation

- **LLM timeout**: 8-second timeout on LLM calls with fallback message sent to user instead of dead air
- **Service errors**: error notifications sent to browser UI so user always knows what's happening
- **Echo cancellation**: browser-side echo cancellation prevents mic from picking up TTS output

## Stack

- FastAPI + WebSockets (server)
- Pipecat 1.4.0 (pipeline orchestration)
- AudioWorklet (browser mic capture, 16kHz PCM)
- Deepgram (streaming ASR)
- Groq / Llama 3.3 70B (LLM)
- ElevenLabs (WebSocket TTS)

## Running Locally

```bash
# Install dependencies
pip install fastapi uvicorn websockets pipecat-ai python-dotenv

# Set environment variables in .env
DEEPGRAM_API_KEY=...
GROQ_API_KEY=...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...

# Start the server
uvicorn pipeline_server:app --reload

# Serve the frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500/index.html` in Chrome.
