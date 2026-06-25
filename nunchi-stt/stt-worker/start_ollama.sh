#!/bin/bash

echo "Starting Ollama..."

# 모델 저장 위치를 /workspace로 지정
# RunPod에서 /workspace가 유지되는 구조면 모델 재다운로드를 줄일 수 있음
export OLLAMA_MODELS=/workspace/ollama_models
mkdir -p $OLLAMA_MODELS

# Ollama가 이미 실행 중이 아니면 실행
if ! pgrep -f "ollama serve" > /dev/null; then
    nohup /usr/local/bin/ollama serve > /tmp/ollama.log 2>&1 &
fi

echo "Waiting for Ollama..."

for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "Ollama is running."
        exit 0
    fi
    sleep 1
done

echo "Ollama failed to start."
cat /tmp/ollama.log
exit 1
