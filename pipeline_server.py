import asyncio
import os
import time
import json
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMTextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    Frame
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams
)

load_dotenv()

app = FastAPI()

# ─── Latency budget tracker ───────────────────────────────────────────────────
# Stores timestamps for each hop in the pipeline per turn.
# Printed after each complete turn so you can see exactly where time is spent.

class LatencyTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.speech_end_time = None        # when Deepgram fired final transcript
        self.first_llm_token_time = None   # when Groq returned first token
        self.first_tts_audio_time = None   # when ElevenLabs returned first audio byte

    def mark_speech_end(self):
        self.speech_end_time = time.time()

    def mark_first_llm_token(self):
        if self.first_llm_token_time is None:
            self.first_llm_token_time = time.time()

    def mark_first_tts_audio(self):
        if self.first_tts_audio_time is None:
            self.first_tts_audio_time = time.time()

    def print_budget(self):
        if not self.speech_end_time:
            return

        print("\n━━━━━━━━━━ LATENCY BUDGET ━━━━━━━━━━")

        if self.first_llm_token_time:
            asr_to_llm = (self.first_llm_token_time - self.speech_end_time) * 1000
            print(f"  ASR → first LLM token  : {asr_to_llm:.0f} ms")
        else:
            print("  ASR → first LLM token  : (no LLM token received)")

        if self.first_llm_token_time and self.first_tts_audio_time:
            llm_to_tts = (self.first_tts_audio_time - self.first_llm_token_time) * 1000
            print(f"  LLM → first TTS audio  : {llm_to_tts:.0f} ms")
        else:
            print("  LLM → first TTS audio  : (no TTS audio received)")

        if self.first_tts_audio_time:
            total = (self.first_tts_audio_time - self.speech_end_time) * 1000
            print(f"  Total time to first audio: {total:.0f} ms")

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        self.reset()


# ─── Serializer ───────────────────────────────────────────────────────────────

class RawAudioSerializer(FrameSerializer):
    async def setup(self, frame):
        pass

    async def deserialize(self, data: bytes | str):
        if isinstance(data, bytes) and len(data) > 0:
            return InputAudioRawFrame(
                audio=data,
                sample_rate=16000,
                num_channels=1
            )
        return None

    async def serialize(self, frame: Frame):
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        return None


# ─── Processors ───────────────────────────────────────────────────────────────

class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"[{time.time():.3f}] FINAL transcript: '{frame.text}'")
        elif isinstance(frame, InterimTranscriptionFrame):
            print(f"[{time.time():.3f}] interim: '{frame.text}'")

        await self.push_frame(frame, direction)


class TranscriptionToLLM(FrameProcessor):
    def __init__(self, latency: LatencyTracker, websocket: WebSocket):
        super().__init__()
        self._latency = latency
        self._websocket = websocket
        self._messages = [
            {
                "role": "system",
                "content": "You are a concise voice assistant. Answer questions directly in one or two sentences. Never ask clarifying questions. Never repeat the question back. Just answer immediately and stop."
            }
        ]
        self._current_assistant_response = ""

    async def _send_error_to_browser(self, message: str):
        """Send a text notification to the browser when something goes wrong."""
        try:
            await self._websocket.send_text(
                json.dumps({"type": "error", "message": message})
            )
        except Exception:
            pass

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._latency.mark_speech_end()
            print(f"[{time.time():.3f}] Sending to LLM: '{frame.text}'")

            self._messages.append({
                "role": "user",
                "content": frame.text
            })
            self._current_assistant_response = ""

            try:
                # Graceful degradation: 8 second timeout on LLM context push.
                # If Groq stalls, the user hears a fallback instead of silence.
                context = LLMContext(messages=self._messages)
                await asyncio.wait_for(
                    self.push_frame(LLMContextFrame(context=context)),
                    timeout=8.0
                )
            except asyncio.TimeoutError:
                print(f"[{time.time():.3f}] LLM TIMEOUT — sending fallback")
                await self._send_error_to_browser(
                    "I'm taking too long to think. Could you repeat that?"
                )
                # Remove the failed user message from history
                self._messages.pop()

        elif isinstance(frame, LLMTextFrame):
            self._latency.mark_first_llm_token()
            self._current_assistant_response += frame.text
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._current_assistant_response.strip():
                self._messages.append({
                    "role": "assistant",
                    "content": self._current_assistant_response
                })
                print(f"[{time.time():.3f}] Assistant response saved to history.")
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)


class TTSLatencyMarker(FrameProcessor):
    """Sits between ElevenLabs and the output transport.
    Marks the timestamp of the first audio byte for latency tracking."""

    def __init__(self, latency: LatencyTracker):
        super().__init__()
        self._latency = latency

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputAudioRawFrame):
            self._latency.mark_first_tts_audio()

        elif isinstance(frame, LLMFullResponseEndFrame):
            # Full turn complete — print the latency budget
            self._latency.print_budget()

        await self.push_frame(frame, direction)


class LLMResponseLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame):
            print(f"[{time.time():.3f}] LLM token: '{frame.text}'")
        elif isinstance(frame, LLMFullResponseStartFrame):
            print(f"[{time.time():.3f}] LLM response starting...")
        elif isinstance(frame, LLMFullResponseEndFrame):
            print(f"[{time.time():.3f}] LLM response complete.")

        await self.push_frame(frame, direction)


# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    latency = LatencyTracker()

    try:
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                serializer=RawAudioSerializer(),
                allowed_origins=[]
            )
        )

        deepgram = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            encoding="linear16",
            sample_rate=16000,
            settings=DeepgramSTTSettings(
                interim_results=True,
                utterance_end_ms=1500,
                endpointing=600,
                smart_format=True,
            )
        )

        groq = GroqLLMService(
            api_key=os.getenv("GROQ_API_KEY"),
            settings=GroqLLMService.Settings(model="llama-3.3-70b-versatile"),
        )

        elevenlabs = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            settings=ElevenLabsTTSService.Settings(
                voice=os.getenv("ELEVENLABS_VOICE_ID"),
            ),
            sample_rate=16000,
        )

        pipeline = Pipeline([
            transport.input(),
            deepgram,
            TranscriptionLogger(),
            TranscriptionToLLM(latency, websocket),
            groq,
            LLMResponseLogger(),
            elevenlabs,
            TTSLatencyMarker(latency),
            transport.output(),
        ])

        task = PipelineTask(pipeline)
        runner = PipelineRunner()

        await runner.run(task)

    except Exception as e:
        print(f"ERROR IN ENDPOINT: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()