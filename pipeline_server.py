import asyncio
import os
import time
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
    TextFrame,
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

# ─── Latency Budget Tracker ───────────────────────────────────────────────────
# Tracks timestamps at each hop so we can print a clean breakdown per turn.
class LatencyTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.speech_end_time = None        # when Deepgram fired final transcript
        self.llm_first_token_time = None   # when first LLM token arrived
        self.tts_first_audio_time = None   # when first TTS audio chunk arrived

    def record_speech_end(self):
        self.speech_end_time = time.time()

    def record_llm_first_token(self):
        if self.llm_first_token_time is None:
            self.llm_first_token_time = time.time()

    def record_tts_first_audio(self):
        if self.tts_first_audio_time is None:
            self.tts_first_audio_time = time.time()

    def print_budget(self):
        if not self.speech_end_time:
            return
        print("\n─── Latency Budget ───")
        if self.llm_first_token_time:
            asr_hop = self.llm_first_token_time - self.speech_end_time
            print(f"  ASR → LLM first token:     {asr_hop*1000:.0f}ms")
        if self.tts_first_audio_time and self.llm_first_token_time:
            tts_hop = self.tts_first_audio_time - self.llm_first_token_time
            print(f"  LLM first token → TTS audio: {tts_hop*1000:.0f}ms")
        if self.tts_first_audio_time:
            total = self.tts_first_audio_time - self.speech_end_time
            print(f"  Total time to first audio:   {total*1000:.0f}ms")
        print("──────────────────────\n")
        self.reset()


# ─── Serializer ───────────────────────────────────────────────────────────────
class RawAudioSerializer(FrameSerializer):
    def __init__(self, latency_tracker: LatencyTracker):
        self._tracker = latency_tracker
        self._first_audio_sent = False

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
            # Record when first TTS audio chunk goes out
            if not self._first_audio_sent:
                self._tracker.record_tts_first_audio()
                self._tracker.print_budget()
                self._first_audio_sent = True
            return frame.audio
        return None

    def reset_audio_flag(self):
        self._first_audio_sent = False


# ─── Transcription Logger ─────────────────────────────────────────────────────
class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"[{time.time():.3f}] FINAL transcript: '{frame.text}'")
        elif isinstance(frame, InterimTranscriptionFrame):
            print(f"[{time.time():.3f}] interim: '{frame.text}'")

        await self.push_frame(frame, direction)


# ─── Transcription → LLM with timeout handling ───────────────────────────────
class TranscriptionToLLM(FrameProcessor):
    LLM_TIMEOUT_SECONDS = 8  # if Groq doesn't respond in 8s, send fallback

    def __init__(self, latency_tracker: LatencyTracker, serializer: RawAudioSerializer):
        super().__init__()
        self._tracker = latency_tracker
        self._serializer = serializer
        self._messages = [
            {
                "role": "system",
                "content": "You are a concise voice assistant. Answer questions directly in one or two sentences. Never ask clarifying questions. Never repeat the question back. Just answer immediately and stop."
            }
        ]
        self._current_assistant_response = ""
        self._llm_task = None

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            # Record when speech ended — start of latency measurement
            self._tracker.record_speech_end()
            self._serializer.reset_audio_flag()

            print(f"[{time.time():.3f}] Sending to LLM: '{frame.text}'")
            self._messages.append({
                "role": "user",
                "content": frame.text
            })
            self._current_assistant_response = ""
            context = LLMContext(messages=self._messages)

            # Wrap the LLM push in a timeout — if Groq stalls, user hears
            # a fallback message instead of dead air
            try:
                await asyncio.wait_for(
                    self.push_frame(LLMContextFrame(context=context)),
                    timeout=self.LLM_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                print(f"[{time.time():.3f}] LLM TIMEOUT after {self.LLM_TIMEOUT_SECONDS}s — pushing fallback")
                # Push a TextFrame with fallback text so TTS speaks it
                await self.push_frame(TextFrame(
                    text="I'm having trouble responding right now. Could you repeat that?"
                ))

        elif isinstance(frame, LLMTextFrame):
            self._tracker.record_llm_first_token()
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


# ─── LLM Response Logger ──────────────────────────────────────────────────────
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


# ─── WebSocket Endpoint ───────────────────────────────────────────────────────
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        tracker = LatencyTracker()
        serializer = RawAudioSerializer(latency_tracker=tracker)

        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                serializer=serializer,
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

        # Graceful degradation: notify browser if Deepgram drops
        @transport.event_handler("on_client_disconnected")
        async def on_disconnect(t, ws):
            print(f"[{time.time():.3f}] Client disconnected — cleaning up.")

        pipeline = Pipeline([
            transport.input(),
            deepgram,
            TranscriptionLogger(),
            TranscriptionToLLM(latency_tracker=tracker, serializer=serializer),
            groq,
            LLMResponseLogger(),
            elevenlabs,
            transport.output(),
        ])

        task = PipelineTask(pipeline)
        runner = PipelineRunner()

        await runner.run(task)

    except Exception as e:
        print(f"[{time.time():.3f}] ERROR IN ENDPOINT: {type(e).__name__}: {e}")
        # Graceful degradation: try to notify browser before dying
        try:
            await websocket.send_text(f'{{"type": "error", "message": "Server error: {type(e).__name__}"}}')
        except Exception:
            pass
        import traceback
        traceback.print_exc()