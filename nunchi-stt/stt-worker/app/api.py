from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from .worker import get_runtime, process_recording, start_polling_thread


def _warm_transcriber() -> None:
    try:
        print("[STT] background model warmup started")
        get_runtime()._get_transcriber()
        print("[STT] background model warmup finished")
    except Exception as error:
        print(f"[STT] background model warmup failed: {error}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warmup_thread = threading.Thread(
        target=_warm_transcriber,
        name="nunchi-stt-warmup",
        daemon=True,
    )
    warmup_thread.start()
    polling = start_polling_thread()
    try:
        yield
    finally:
        if polling is not None:
            _thread, stop_event = polling
            stop_event.set()


app = FastAPI(title="Nunchi STT Worker", lifespan=lifespan)


class SttProcessRequest(BaseModel):
    recordingId: int


@app.get("/stt/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/stt/process")
def process(request: SttProcessRequest, background_tasks: BackgroundTasks) -> dict[str, object]:
    background_tasks.add_task(process_recording, request.recordingId)
    return {
        "status": "accepted",
        "recordingId": request.recordingId,
    }
