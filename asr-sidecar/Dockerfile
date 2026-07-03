# Multi-arch (linux/amd64 + linux/arm64) CPU-only ASR sidecar.
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py engine.py server.py ./

ENV HF_HOME=/data/hf \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Download (idempotent, ~9 GB on first run) before serving so progress is visible
# in the logs; the healthcheck's start_period covers this window.
CMD ["sh", "-c", "python download_model.py && exec uvicorn server:app --host 0.0.0.0 --port 8000"]
