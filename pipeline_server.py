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
        # Maintain conversation history across turns
        self._messages = [
            {
                "role": "system",
                "content": "You are a helpful voice assistant. Keep responses concise and conversational — two or three sentences maximum, since your response will be spoken aloud."
            }
        ]

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            print(f"[{time.time():.3f}] Sending to LLM: '{frame.text}'")

            # Add user turn to history
            self._messages.append({
                "role": "user",
                "content": frame.text
            })

            # Build context and push to Groq
            context = LLMContext(messages=self._messages)
            await self.push_frame(LLMContextFrame(context=context))
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
                utterance_end_ms=1000,
            )
        )

        groq = GroqLLMService(
    api_key=os.getenv("GROQ_API_KEY"),
    settings=GroqLLMService.Settings(model="llama-3.3-70b-versatile"),
    )


        pipeline = Pipeline([
            transport.input(),
            deepgram,
            TranscriptionLogger(),
            TranscriptionToLLM(),
            groq,
            LLMResponseLogger(),
            transport.output(),
        ])

        task = PipelineTask(pipeline)
        runner = PipelineRunner()

        await runner.run(task)

    except Exception as e:
        print(f"ERROR IN ENDPOINT: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()