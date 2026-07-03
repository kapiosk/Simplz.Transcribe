"""FastAPI server exposing a vLLM-Realtime-compatible WebSocket ASR API.

Protocol (subset of vLLM's /v1/realtime, so a GPU vLLM deployment is a
drop-in replacement for this sidecar):

  C->S {"type": "session.update", "model": "..."}            (optional)
  C->S {"type": "input_audio_buffer.append", "audio": "<b64 PCM16LE 16kHz mono>"}
  C->S {"type": "input_audio_buffer.commit", "final": true}  (end of audio)

  S->C {"type": "session.created", "model": "..."}
  S->C {"type": "transcription.delta", "delta": "..."}
  S->C {"type": "transcription.done", "text": "<full transcript>"}
  S->C {"type": "error", "error": "..."}
"""

import asyncio
import base64
import json
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from engine import AsrEvent, VoxtralEngine, VoxtralSession

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("asr.server")

app = FastAPI(title="Simplz.Transcribe ASR sidecar")
engine: VoxtralEngine | None = None


@app.on_event("startup")
def load_model() -> None:
    global engine
    engine = VoxtralEngine()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": engine.model_id}


async def _pump_events(ws: WebSocket, session: VoxtralSession) -> None:
    """Forward engine events to the client until the session finishes."""
    while True:
        event: AsrEvent = await asyncio.to_thread(session.events.get)
        if event.type == "delta":
            await ws.send_json({"type": "transcription.delta", "delta": event.text})
        elif event.type == "done":
            await ws.send_json({"type": "transcription.done", "text": event.text})
            return
        else:
            await ws.send_json({"type": "error", "error": event.text})
            return


@app.websocket("/v1/realtime")
async def realtime(ws: WebSocket) -> None:
    await ws.accept()

    session = engine.try_create_session()
    if session is None:
        await ws.send_json({"type": "error", "error": "busy: another transcription is in progress"})
        await ws.close()
        return

    await ws.send_json({"type": "session.created", "model": engine.model_id})
    pump = asyncio.create_task(_pump_events(ws, session))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "input_audio_buffer.append":
                session.feed_pcm16(base64.b64decode(msg["audio"]))
            elif msg_type == "input_audio_buffer.commit":
                if msg.get("final", True):
                    session.end_input()
                    break  # stop reading; wait for the pump to drain
            elif msg_type == "session.update":
                pass  # single fixed model/config in the sidecar
            else:
                await ws.send_json({"type": "error", "error": f"unknown message type: {msg_type}"})

        await pump
        await ws.close()
    except (WebSocketDisconnect, RuntimeError):
        logger.info("client disconnected mid-session, cancelling")
        session.cancel()
        pump.cancel()
    finally:
        session.cancel()  # no-op if already finished


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
