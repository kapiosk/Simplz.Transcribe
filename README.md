# Simplz.Transcribe

Local streaming speech-to-text web app. A C# (ASP.NET Core) frontend streams audio —
live from your **microphone** or from an uploaded **audio/video file** — to a
**Voxtral Realtime 4B** model ([`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602),
Apache 2.0) running fully locally in the same Docker stack. No cloud, no API keys.

```
Browser ── WS (binary PCM16 16 kHz) ──► web (ASP.NET Core :8080) ── WS /v1/realtime ──► asr (Python, Voxtral on CPU)
Browser ── POST file (SSE response) ──►   │ ffmpeg → 16 kHz mono PCM                        │
                                          ◄── partial / final transcript events ───────────┘
```

## Requirements

- Docker + docker compose (CPU only — works on x86-64 and Apple Silicon/arm64; no NVIDIA needed)
- **RAM for the asr container:** ~10 GB with the default `bfloat16`, ~18 GB with `float32`.
  On Docker Desktop, raise the memory limit in *Settings → Resources* accordingly.
- ~10 GB disk for the model cache volume (downloaded once on first start)

## Run

```sh
docker compose up --build
```

The first start downloads ~9 GB of model weights into the `hf-cache` volume and then loads
the model — watch progress with `docker compose logs -f asr`. The `web` service starts once
`asr` reports healthy. Then open **http://localhost:8085**.

- **Microphone**: click *Start recording*, speak, click *Stop*. Partial transcript streams in
  as it's decoded; the final transcript replaces it at the end.
- **File**: pick any audio/video file (anything ffmpeg can decode). The transcript streams in
  live while the file is processed.

### Test without a browser/microphone

```sh
curl -N --data-binary @sample.mp3 http://localhost:8085/api/transcribe-file
```

You'll see `data: {"type":"partial","text":"..."}` SSE events streaming, ending in a `final` event.

To test the ASR sidecar directly (bypassing the web app), see
[asr-sidecar/test_client.py](asr-sidecar/test_client.py).

## Expected performance (important)

Voxtral Realtime is a 4B-parameter model. There is currently **no production CPU serving
stack for it** (vLLM supports it on GPU only; llama.cpp support is still open —
[ggml-org/llama.cpp#20914](https://github.com/ggml-org/llama.cpp/issues/20914)). This stack
runs it with the official 🤗 transformers streaming API on CPU, which is typically
**2–10× slower than real time** depending on your CPU. Transcripts still stream in
incrementally — they just lag behind the audio. Quality is unaffected.

**Drop-in GPU upgrade:** the sidecar speaks the vLLM Realtime WebSocket protocol. If you have
a GPU machine running `vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602`, point the web app
at it and remove the sidecar — no code changes:

```yaml
  web:
    environment:
      - Asr__WebSocketUrl=ws://your-gpu-host:8000/v1/realtime
```

### Experimental: AMD GPU (Ryzen AI Max "Strix Halo", gfx1151) — Linux hosts only

The repo includes an all-Docker ROCm variant of the sidecar built against AMD's
[TheRock nightly PyTorch wheels](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) for
gfx1151. The container gets the iGPU via `/dev/kfd` + `/dev/dri`, which requires **Docker on
a Linux host** with the `amdgpu` kernel driver — Docker Desktop on Windows/macOS cannot pass
through the AMD iGPU.

```sh
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build
```

Untested/experimental: gfx1151 is still "preview" in ROCm. If the container can't see the
GPU, replace the `video`/`render` group names in `docker-compose.rocm.yml` with your host's
numeric GIDs (`getent group video render`). Rough math says the 4B model at bf16 should reach
real-time on the 8060S iGPU (memory-bandwidth-bound at roughly 2× the ~12.5 tok/s needed).

## Configuration

| Variable | Service | Default | Notes |
|---|---|---|---|
| `Asr__WebSocketUrl` | web | `ws://asr:8000/v1/realtime` | Any vLLM-Realtime-compatible endpoint |
| `Asr__Model` | web | `mistralai/Voxtral-Mini-4B-Realtime-2602` | Sent in `session.update` |
| `VOXTRAL_DTYPE` | asr | `bfloat16` | `float32` = more compatible, ~2× RAM |
| `VOXTRAL_DEVICE` | asr | `cpu` | `cuda` selects the GPU (incl. ROCm/HIP builds) |
| `VOXTRAL_MODEL` | asr | `mistralai/Voxtral-Mini-4B-Realtime-2602` | |

The streaming delay (480 ms) is fixed by the model's processor config.

## Limitations (v1)

- One transcription session at a time (the CPU model is serialized; concurrent requests get
  a `busy` error).
- Partial transcripts are append-only deltas (the model does not revise earlier text).
- Uploaded files are paced at up to 16× real time into the ASR engine.

## Development

```sh
dotnet build Simplz.Transcribe.slnx         # build the web app locally
docker compose build                        # build both images
```

The web app can run outside Docker (`dotnet run --project src/Simplz.Transcribe.Web`) if you
have ffmpeg on PATH and set `Asr__WebSocketUrl=ws://localhost:8000/v1/realtime` with the
sidecar's port 8000 published.

## License

[MIT](LICENSE). The Voxtral Realtime model weights are separately licensed under Apache 2.0
by Mistral AI.
