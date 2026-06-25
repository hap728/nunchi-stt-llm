#!/bin/bash

echo "Starting Ollama..."

nohup ollama serve > /tmp/ollama.log 2>&1 &

echo "Waiting for Ollama..."
sleep 5

curl http://localhost:11434/api/tags

echo "Ollama started."
echo "Starting STT worker..."

cd /workspace/nunchi-stt/stt-worker
source .venv/bin/activate

# 네가 실제 실행하는 파이썬 파일로 변경
python run_audio_to_schedule.py
