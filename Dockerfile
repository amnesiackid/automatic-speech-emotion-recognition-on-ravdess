FROM python:3.11-slim

WORKDIR /app

# System libraries for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image small
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[server]"

COPY server ./server
COPY frontend ./frontend

EXPOSE 5000
ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "server.app:app"]
