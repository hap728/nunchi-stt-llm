FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL_DIR=/models/faster-whisper

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app app
COPY run_api.py .
COPY run_worker.py .
COPY run_schedule.py .

RUN mkdir -p "${WHISPER_MODEL_DIR}"

EXPOSE 8090

ENTRYPOINT ["python", "run_api.py"]
