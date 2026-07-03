"""Voxtral Realtime streaming engine (CPU, transformers).

Wraps the experimental incremental streaming API of
`VoxtralRealtimeForConditionalGeneration` (transformers >= 5.2) behind a small
engine/session interface so an alternative backend (e.g. voxtral.cpp) can be
dropped in later without touching the WebSocket server.

Audio in: PCM16LE mono @ 16 kHz. Text out: incremental transcript deltas.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger("asr.engine")

MODEL_ID = os.environ.get("VOXTRAL_MODEL", "mistralai/Voxtral-Mini-4B-Realtime-2602")
DTYPE_NAME = os.environ.get("VOXTRAL_DTYPE", "bfloat16")  # bfloat16 | float32
DEVICE = os.environ.get("VOXTRAL_DEVICE", "cpu")  # "cuda" also selects ROCm/HIP GPUs

SAMPLE_RATE = 16_000


@dataclass
class AsrEvent:
    type: str  # "delta" | "done" | "error"
    text: str = ""


class VoxtralEngine:
    """Loads the model once; hands out one active session at a time."""

    def __init__(self) -> None:
        from transformers import (
            VoxtralRealtimeForConditionalGeneration,
            VoxtralRealtimeProcessor,
        )

        dtype = torch.bfloat16 if DTYPE_NAME == "bfloat16" else torch.float32
        logger.info("loading %s (dtype=%s, device=%s) — this can take a while on first run",
                    MODEL_ID, dtype, DEVICE)
        self.processor = VoxtralRealtimeProcessor.from_pretrained(MODEL_ID)
        # No device_map: plain load (device_map would require `accelerate`).
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype)
        if DEVICE != "cpu":
            self.model.to(DEVICE)
        self.model.eval()
        self._busy = threading.Lock()
        logger.info("model loaded")

    @property
    def model_id(self) -> str:
        return MODEL_ID

    def try_create_session(self) -> "VoxtralSession | None":
        """Returns a session, or None if another transcription is running."""
        if not self._busy.acquire(blocking=False):
            return None
        return VoxtralSession(self, release=self._busy.release)


class VoxtralSession:
    """One live transcription: feed PCM in, iterate events out.

    Follows the documented transformers streaming pattern: a background
    `model.generate` consumes a generator of per-chunk `input_features`; the
    generator blocks on an internal sample buffer that `feed()` appends to.
    """

    def __init__(self, engine: VoxtralEngine, release) -> None:
        self._engine = engine
        self._release = release
        self._proc = engine.processor
        self._model = engine.model

        # Growing sample buffer (amortized O(1) append; windows overlap, so
        # the whole session's audio is kept — ~230 MB/hour, acceptable for v1).
        self._buf = np.zeros(SAMPLE_RATE * 60, dtype=np.float32)
        self._len = 0
        self._ended = False
        self._cancelled = False
        self._cond = threading.Condition()

        self.events: queue.Queue[AsrEvent] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="voxtral-session", daemon=True)
        self._thread.start()

    # -- input side (called from the server) --------------------------------

    def feed_pcm16(self, data: bytes) -> None:
        """Append raw PCM16LE mono 16 kHz bytes."""
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        with self._cond:
            if self._ended:
                return
            self._append(samples)
            self._cond.notify_all()

    def end_input(self) -> None:
        """No more audio will arrive; pad and let generation drain."""
        with self._cond:
            if self._ended:
                return
            # Right padding required by the model to flush the last tokens.
            # (method in transformers 5.12, plain attribute in the docs — accept both)
            n_right = self._proc.num_right_pad_tokens
            if callable(n_right):
                n_right = n_right()
            self._append(np.zeros(n_right * self._proc.raw_audio_length_per_tok, dtype=np.float32))
            self._ended = True
            self._cond.notify_all()

    def _append(self, samples: np.ndarray) -> None:
        """Append under self._cond; doubles capacity as needed."""
        needed = self._len + len(samples)
        if needed > len(self._buf):
            grown = np.zeros(max(needed, len(self._buf) * 2), dtype=np.float32)
            grown[:self._len] = self._buf[:self._len]
            self._buf = grown
        self._buf[self._len:needed] = samples
        self._len = needed

    def cancel(self) -> None:
        """Client went away: stop consuming input and drop remaining work."""
        with self._cond:
            self._cancelled = True
            self._ended = True
            self._cond.notify_all()

    # -- internals -----------------------------------------------------------

    def _wait_span(self, start: int, length: int) -> np.ndarray | None:
        """Block until buf[start:start+length] is fully available.

        Returns None when input ended and no further full span exists
        (matching the reference loop, which only yields complete windows).
        """
        with self._cond:
            while True:
                if self._cancelled:
                    return None
                if self._len >= start + length:
                    return self._buf[start:start + length].copy()
                if self._ended:
                    return None
                self._cond.wait(timeout=0.1)

    def _first_chunk(self) -> np.ndarray | None:
        n = self._proc.num_samples_first_audio_chunk
        span = self._wait_span(0, n)
        if span is not None:
            return span
        if self._cancelled:
            return None
        # Very short input: everything (incl. right pad) fits in < first chunk.
        with self._cond:
            if self._len == 0:
                return None
            return np.pad(self._buf[:self._len], (0, n - self._len))

    def _run(self) -> None:
        try:
            self._transcribe()
        except Exception:  # surface, don't kill the server
            logger.exception("transcription session failed")
            self.events.put(AsrEvent("error", "transcription failed, see server logs"))
        finally:
            self._release()

    def _transcribe(self) -> None:
        from transformers import TextIteratorStreamer

        proc, model = self._proc, self._model

        first = self._first_chunk()
        if first is None:
            self.events.put(AsrEvent("done", ""))
            return

        first_inputs = proc(first, is_streaming=True, is_first_audio_chunk=True, return_tensors="pt")
        first_inputs = first_inputs.to(model.device, dtype=model.dtype)

        # The streaming delay (480 ms) is baked into the model's processor
        # config; the processor ignores attempts to override it.
        num_delay_tokens = first_inputs.get("num_delay_tokens", None)
        if num_delay_tokens is None:
            num_delay_tokens = 6  # model default (480 ms / 80 ms per token)

        def input_features_generator():
            yield first_inputs.input_features

            mel_frame_idx = proc.num_mel_frames_first_audio_chunk
            hop_length = proc.feature_extractor.hop_length
            win_length = proc.feature_extractor.win_length

            while True:
                start_idx = mel_frame_idx * hop_length - win_length // 2
                span = self._wait_span(start_idx, proc.num_samples_per_audio_chunk)
                if span is None:
                    return
                inputs = proc(span, is_streaming=True, is_first_audio_chunk=False, return_tensors="pt")
                inputs = inputs.to(model.device, dtype=model.dtype)
                yield inputs.input_features
                mel_frame_idx += proc.audio_length_per_tok

        # timeout: if generate crashes, iteration raises instead of hanging forever.
        streamer = TextIteratorStreamer(
            proc.tokenizer, skip_prompt=True, skip_special_tokens=True,
            clean_up_tokenization_spaces=True, timeout=600.0,
        )
        generate_kwargs = {
            "input_ids": first_inputs.input_ids,
            "input_features": input_features_generator(),
            "num_delay_tokens": num_delay_tokens,
            "streamer": streamer,
        }
        gen_thread = threading.Thread(
            target=model.generate, kwargs=generate_kwargs, name="voxtral-generate", daemon=True
        )
        gen_thread.start()

        pieces: list[str] = []
        for text_chunk in streamer:
            if text_chunk:
                pieces.append(text_chunk)
                if not self._cancelled:
                    self.events.put(AsrEvent("delta", text_chunk))
        gen_thread.join()
        self.events.put(AsrEvent("done", "".join(pieces)))
