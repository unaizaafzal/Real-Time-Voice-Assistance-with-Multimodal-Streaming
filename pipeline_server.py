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


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"[{time.time():.3f}] FINAL transcript: '{frame.text}'")
        elif isinstance(frame, InterimTranscriptionFrame):
            print(f"[{time.time():.3f}] interim: '{frame.text}'")

        await self.push_frame(frame, direction)


class TranscriptionToLLM(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._messages = [
    {
        "role": "system",
        "content": "You are a concise voice assistant. Answer questions directly in one or two sentences. Never ask clarifying questions. Never repeat the question back. Just answer immediately and stop."
    }
]
        self._current_assistant_response = ""

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            word_count = len(frame.text.strip().split())
            if word_count < 3:
                print(f"[{time.time():.3f}] Ignoring short fragment: '{frame.text}'")
                await self.push_frame(frame, direction)
                return

            print(f"[{time.time():.3f}] Sending to LLM: '{frame.text}'")
            self._messages.append({
                "role": "user",
                "content": frame.text
            })
            self._current_assistant_response = ""
            context = LLMContext(messages=self._messages)
            await self.push_frame(LLMContextFrame(context=context))

        elif isinstance(frame, LLMTextFrame):
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


@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
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
            TranscriptionToLLM(),
            groq,
            LLMResponseLogger(),
            elevenlabs,
            transport.output(),
        ])

        task = PipelineTask(pipeline)
        runner = PipelineRunner()

        await runner.run(task)

    except Exception as e:
        print(f"ERROR IN ENDPOINT: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()