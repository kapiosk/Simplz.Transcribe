"""Idempotent model download into the HF cache volume.

Runs before the server starts so the (large, one-time) download produces clear
progress logs and the server's model load afterwards is a fast cache hit.
"""

import os

from huggingface_hub import snapshot_download

MODEL_ID = os.environ.get("VOXTRAL_MODEL", "mistralai/Voxtral-Mini-4B-Realtime-2602")


def main() -> None:
    print(f"[download_model] ensuring {MODEL_ID} is present in HF cache "
          f"(HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')})", flush=True)
    path = snapshot_download(MODEL_ID)
    print(f"[download_model] model available at {path}", flush=True)


if __name__ == "__main__":
    main()
