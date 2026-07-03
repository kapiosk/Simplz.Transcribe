"""Standalone sidecar test client.

Usage:  python test_client.py <audio.wav> [ws://localhost:8000/v1/realtime]

The wav must be PCM16 mono 16 kHz (convert with:
  ffmpeg -i input.any -ac 1 -ar 16000 -f wav sample16k.wav).
Streams it in ~250 ms chunks and prints transcription deltas as they arrive.

Requires: pip install websockets
"""

import asyncio
import base64
import json
import sys
import wave

import websockets

CHUNK_MS = 250


async def main() -> None:
    wav_path = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8000/v1/realtime"

    with wave.open(wav_path, "rb") as wav:
        assert wav.getnchannels() == 1 and wav.getframerate() == 16000 and wav.getsampwidth() == 2, \
            "expected PCM16 mono 16 kHz wav"
        pcm = wav.readframes(wav.getnframes())

    chunk_bytes = 16000 * 2 * CHUNK_MS // 1000

    async with websockets.connect(url, max_size=None) as ws:
        print(json.loads(await ws.recv()))  # session.created

        async def send() -> None:
            for off in range(0, len(pcm), chunk_bytes):
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[off:off + chunk_bytes]).decode(),
                }))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        sender = asyncio.create_task(send())
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "transcription.delta":
                print(msg["delta"], end="", flush=True)
            elif msg["type"] == "transcription.done":
                print(f"\n--- done ---\n{msg['text']}")
                break
            else:
                print(f"\n{msg}")
                break
        await sender


if __name__ == "__main__":
    asyncio.run(main())
