# Nunchi STT Pipeline

회의 음성을 텍스트로 전사하고, 전사 결과를 기반으로 일정 정보를 추출하는 STT/LLM 처리 파이프라인입니다.

## 개요

이 프로젝트는 사용자가 업로드한 회의 음성 파일을 NCP Object Storage에서 가져와 `faster-whisper` 기반 STT 모델로 전사하고, 전사 결과를 DB에 저장합니다. 이후 DB에 저장된 `raw_text`를 LLM 일정 추출 단계로 전달해 날짜, 시간, 일정명, 장소 등의 정보를 JSON 형태로 추출합니다.

```text
NCP Object Storage
-> STT Worker
-> Whisper 기반 전사
-> DB stt_results.raw_text 저장
-> LLM 일정 추출
-> 일정 결과 DB 저장
```

## 담당 범위

- Whisper 기반 STT 전사 파이프라인 구성
- 한국어 회의 음성에 맞춘 STT 옵션 조정
- 무음/저신뢰 구간 필터링 및 후처리
- NCP Object Storage 음성 다운로드 연동
- 기존 DB에 STT 결과 저장
- DB의 `raw_text`를 LLM 일정 추출로 연결
- 일정 추출 결과를 DB에 저장하는 구조 추가

## 주요 기능

- `recordings` 테이블에서 STT 대기 작업 조회
- NCP Object Storage의 `object_key`로 음성 파일 다운로드
- `faster-whisper` 기반 한국어 STT 전사
- `stt_results.raw_text` 및 `corrected_text` 저장
- STT 상태값 관리: `STT_PENDING`, `STT_PROCESSING`, `STT_DONE`, `STT_FAILED`
- Ollama 기반 LLM 일정 추출
- 긴 전사문 chunk 분리 및 overlap 처리
- 일정 추출 결과 JSON 저장

## 실행 환경

STT 전사는 RunPod L4 GPU 클라우드 환경에서 GPU 기반으로 실행하는 구성을 기준으로 정리했습니다. Whisper 모델은 `large-v3`를 사용하고, GPU 추론을 위해 `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16` 설정을 사용합니다.

```text
Cloud GPU: RunPod NVIDIA L4
STT Model: faster-whisper large-v3
Device: cuda
Compute Type: float16
```

GPU 환경에서는 `Dockerfile.gpu`를 사용할 수 있으며, 모델 파일은 `/models/faster-whisper` 경로에 준비되도록 구성했습니다.

## 디렉터리 구조

```text
nunchi-stt/
  stt-worker/
    app/
      api.py                  # FastAPI 엔드포인트
      config.py               # 환경 변수 설정
      db.py                   # DB 조회/저장 로직
      storage.py              # NCP Object Storage 연동
      transcriber.py          # Whisper STT 처리
      text_normalizer.py      # 전사문 후처리
      worker.py               # STT 워커 실행 흐름
      schedule_extractor.py   # LLM 일정 추출
      schedule_worker.py      # DB raw_text 기반 일정 추출 실행
    run_api.py
    run_worker.py
    run_schedule.py
    requirements.txt
    Dockerfile
    Dockerfile.gpu
```

## 환경 변수

실제 값은 `.env`에 작성하고, Git에는 올리지 않습니다. 예시는 `nunchi-stt/stt-worker/.env.example`을 참고합니다.

```text
DB_HOST=
DB_PORT=
DB_NAME=
DB_USERNAME=
DB_PASSWORD=

NCP_ENDPOINT=https://kr.object.ncloudstorage.com
NCP_REGION=kr-standard
NCP_BUCKET=
NCP_ACCESS_KEY=
NCP_SECRET_KEY=

WHISPER_MODEL_SIZE=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

LLM_SCHEDULE_ENABLED=true
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=exaone3.5:7.8b
SCHEDULE_RESULTS_TABLE=schedule_results
```

## 실행 방법

```powershell
cd nunchi-stt/stt-worker
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
python run_api.py
```

API 서버 기본 주소:

```text
http://localhost:8090
```

헬스 체크:

```text
GET /stt/health
```

특정 녹음 STT 처리:

```powershell
python run_worker.py --recording-id 1
```

이미 DB에 저장된 STT 결과로 일정 추출:

```powershell
python run_schedule.py --recording-id 1
```

## 연결 상태

현재 코드상 연결된 범위:

```text
bucket 음성 다운로드
-> STT 전사
-> DB raw_text 저장
-> raw_text 기반 LLM 일정 추출
-> 일정 결과 DB 저장
```

운영 시 확인할 항목:

- `.env`의 DB/NCP/Ollama 설정
- `recordings`, `stt_results` 테이블 컬럼명
- `SCHEDULE_RESULTS_TABLE`에 지정한 일정 결과 저장 테이블
- Ollama 서버 실행 여부
- LLM 모델 설치 여부

## 보안

`.env`, 가상환경, 음성 데이터, 모델 파일, 실험 결과 파일은 `.gitignore`로 제외했습니다. 실제 DB/NCP 키는 저장소에 커밋하지 않습니다.
