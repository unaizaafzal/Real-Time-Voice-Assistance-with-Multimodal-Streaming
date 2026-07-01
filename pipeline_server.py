import asyncio
import time
from fastapi import FastAPI, WebSocket
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import InputAudioRawFrame, StartFrame, EndFrame, Frame
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams
)

app = FastAPI()


# A minimal serializer that converts raw bytes from the browser
# into InputAudioRawFrame objects that Pipecat can work with.
# This is the missing piece — without it, Pipecat silently drops everything.
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
        # No outgoing serialization needed yet
        return None


class LoggingProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        # Now correctly checking for InputAudioRawFrame, not AudioRawFrame
        if isinstance(frame, InputAudioRawFrame):
            print(f"[{time.time():.3f}] InputAudioRawFrame passing through: {len(frame.audio)} bytes")

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

        pipeline = Pipeline([
            transport.input(),
            LoggingProcessor(),
            transport.output(),
        ])

        task = PipelineTask(pipeline)
        runner = PipelineRunner()

        await runner.run(task)

    except Exception as e:
        print(f"ERROR IN ENDPOINT: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=RawAudioSerializer(),
            allowed_origins=[]  # empty list = allow all origins (local dev)
        )
    )

    pipeline = Pipeline([
        transport.input(),
        LoggingProcessor(),
        transport.output(),
    ])

    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    await runner.run(task)