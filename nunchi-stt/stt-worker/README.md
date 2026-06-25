# STT Worker

`faster-whisper` 기반 STT 워커입니다. NCP Object Storage에 저장된 회의 음성을 다운로드해 텍스트로 전사하고, 전사 결과를 DB에 저장합니다. 선택적으로 DB에 저장된 `raw_text`를 LLM 일정 추출 단계로 연결합니다.

## 처리 흐름

```text
backend
-> recordings 저장 (STT_PENDING)
-> stt-worker /stt/process 호출 또는 polling
-> NCP Object Storage에서 음성 다운로드
-> Whisper STT 전사
-> stt_results.raw_text 저장
-> LLM 일정 추출
-> schedule_results 저장
```

## 실행 방식

- 이벤트 방식: backend가 `/stt/process`로 `recordingId`를 전달
- 폴링 방식: worker가 `STT_PENDING`, `STT_REQUESTED`, `UPLOADED` 상태를 주기적으로 조회
- 수동 방식: `run_worker.py` 또는 `run_schedule.py`로 특정 건 처리

## 파일 구성

```text
stt-worker/
  .env.example
  Dockerfile
  Dockerfile.gpu
  requirements.txt
  run_api.py
  run_worker.py
  run_schedule.py
  app/
    api.py
    config.py
    db.py
    storage.py
    transcriber.py
    text_normalizer.py
    worker.py
    schedule_extractor.py
    schedule_worker.py
```

## 환경 변수

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
WHISPER_MODEL_DIR=/models/faster-whisper
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_BATCH_SIZE=2

WORKER_POLL_INTERVAL_SEC=10
WORKER_BATCH_SIZE=3
WORKER_ENABLE_POLLING=true

LLM_SCHEDULE_ENABLED=true
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=exaone3.5:7.8b
LLM_REFERENCE_DATE=
LLM_REQUEST_TIMEOUT_SEC=300
SCHEDULE_RESULTS_TABLE=schedule_results
```

## 로컬 실행

```powershell
cd nunchi-stt/stt-worker
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
python run_api.py
```

기본 주소:

```text
http://localhost:8090
```

헬스 체크:

```text
GET /stt/health
```

STT 처리 요청:

```text
POST /stt/process
{
  "recordingId": 1
}
```

## 수동 실행

특정 녹음 1건 STT 처리:

```powershell
python run_worker.py --recording-id 1
```

대기 중인 건 배치 1회 처리:

```powershell
python run_worker.py --once
```

폴링 루프 실행:

```powershell
python run_worker.py
```

이미 DB에 저장된 STT 결과로 일정 추출:

```powershell
python run_schedule.py --recording-id 1
```

## DB 저장

STT 워커는 기본적으로 다음 컬럼을 기대합니다.

```text
recordings
- recording_id
- user_id
- object_key
- status
- retry_count
- created_at

stt_results
- recording_id
- user_id
- raw_text
- corrected_text
- processed_at
```

일정 추출 결과는 `.env`의 `SCHEDULE_RESULTS_TABLE` 값에 지정된 테이블로 저장합니다. JSON 저장 컬럼은 다음 이름 중 하나를 지원합니다.

```text
result_json
schedule_json
extracted_json
llm_result_json
calendar_events_json
```

## 상태 흐름

```text
STT_PENDING
-> STT_PROCESSING
-> STT_DONE
```

실패 시:

```text
STT_FAILED
```

LLM 일정 추출 상태 컬럼이 DB에 있으면 다음 상태도 업데이트합니다.

```text
LLM_PROCESSING
-> LLM_DONE
```

실패 시:

```text
LLM_FAILED
```
