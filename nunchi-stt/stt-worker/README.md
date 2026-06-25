# STT Worker

이 디렉터리는 `faster-whisper` 기반 STT 워커다.

지금 구조는 두 방식을 같이 쓴다.

- 이벤트 방식: backend가 `/stt/process`로 `recordingId`를 바로 전달
- 폴링 방식: worker가 주기적으로 `STT_PENDING` 건을 다시 확인해서 놓친 건을 복구

즉, 정상 흐름은 이벤트로 빠르게 처리하고, 폴링은 복구용 안전망으로 동작한다.

## 처리 흐름

```text
frontend
-> backend /api/recordings 업로드
-> recordings 저장 (STT_PENDING)
-> backend -> stt-worker /stt/process 호출
-> stt-worker 전사
-> stt_results.raw_text 저장
-> recordings.status = STT_DONE
```

이벤트 요청이 실패하거나 worker가 놓친 건이 있으면 폴링 루프가 `STT_PENDING` 건을 다시 처리한다.

## 파일 구성

```text
stt-worker/
  .env
  .env.example
  Dockerfile
  Dockerfile.gpu
  requirements.txt
  run_api.py
  run_worker.py
  app/
    api.py
    config.py
    db.py
    storage.py
    transcriber.py
    worker.py
```

## 환경 변수

`.env.example` 기준:

```text
WORKER_POLL_INTERVAL_SEC=10
WORKER_BATCH_SIZE=3
WORKER_ENABLE_POLLING=true
```

- `WORKER_POLL_INTERVAL_SEC`: 폴링 주기(초)
- `WORKER_BATCH_SIZE`: 한 번에 가져올 대기 건수
- `WORKER_ENABLE_POLLING`: `true`면 API 서버와 함께 폴링 루프도 실행

## 로컬 실행

```powershell
cd "C:\Users\smhrd\OneDrive\Desktop\ai-hub 회의 데이터\service\stt-worker"
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

이벤트 처리 요청:

```text
POST /stt/process
{
  "recordingId": 2
}
```

## 수동 실행

특정 녹음 1건만 처리:

```powershell
python run_worker.py --recording-id 2
```

대기 중인 건 배치 1회 처리:

```powershell
python run_worker.py --once
```

폴링 루프만 계속 실행:

```powershell
python run_worker.py
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
