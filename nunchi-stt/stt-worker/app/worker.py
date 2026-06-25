from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import load_settings
from .db import (
    claim_pending_recordings,
    claim_recording_by_id,
    mark_recording_failed,
    save_stt_result,
)
from .storage import download_audio_file
from .schedule_worker import process_schedule_for_recording
from .text_normalizer import TranscriptNormalizer
from .transcriber import Transcriber


class WorkerRuntime:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.transcriber: Transcriber | None = None
        self.normalizer: TranscriptNormalizer | None = None
        self._model_lock = threading.Lock()
        self._transcriber_init_lock = threading.Lock()

    def process_recording(self, recording_id: int) -> bool:
        if not self._ensure_transcriber_ready():
            return False
        row = claim_recording_by_id(self.settings, recording_id)
        if not row:
            print(f"[STT] recording_id={recording_id} is not in STT_PENDING state.")
            return False
        return self._process_row(row)

    def process_pending_batch(self) -> int:
        if not self._ensure_transcriber_ready():
            return 0
        claimed = claim_pending_recordings(self.settings, self.settings.worker_batch_size)
        if not claimed:
            return 0

        processed = 0
        for row in claimed:
            if self._process_row(row):
                processed += 1
        return processed

    def poll_forever(self, stop_event: threading.Event) -> None:
        print(
            f"[STT] polling loop started interval={self.settings.worker_poll_interval_sec}s "
            f"batch_size={self.settings.worker_batch_size}"
        )
        while not stop_event.is_set():
            processed = self.process_pending_batch()
            if processed == 0:
                stop_event.wait(self.settings.worker_poll_interval_sec)

    def _process_row(self, row: dict[str, object]) -> bool:
        recording_id = int(row["recording_id"])
        user_id = int(row["user_id"])
        object_key = str(row["object_key"])
        total_started_at = time.perf_counter()

        print(f"[STT] processing recording_id={recording_id} object_key={object_key}")

        try:
            with TemporaryDirectory(prefix="nunchi-stt-") as temp_dir:
                temp_path = Path(temp_dir) / Path(object_key).name
                download_started_at = time.perf_counter()
                download_audio_file(self.settings, object_key, temp_path)
                print(
                    f"[STT] downloaded recording_id={recording_id} "
                    f"elapsed={time.perf_counter() - download_started_at:.1f}s"
                )
                transcribe_started_at = time.perf_counter()
                with self._model_lock:
                    raw_text = self._get_transcriber().transcribe(temp_path)
                corrected_text = self._normalize_text(raw_text)
                print(
                    f"[STT] transcribed recording_id={recording_id} "
                    f"elapsed={time.perf_counter() - transcribe_started_at:.1f}s "
                    f"text_length={len(raw_text)} "
                    f"corrected_length={len(corrected_text) if corrected_text else 0}"
                )
                save_stt_result(self.settings, recording_id, user_id, raw_text, corrected_text)
                if self.settings.llm_schedule_enabled:
                    process_schedule_for_recording(
                        recording_id=recording_id,
                        user_id=user_id,
                        raw_text=raw_text,
                        settings=self.settings,
                    )
                print(
                    f"[STT] done recording_id={recording_id} "
                    f"total_elapsed={time.perf_counter() - total_started_at:.1f}s"
                )
                return True
        except Exception as error:
            message = str(error) or error.__class__.__name__
            mark_recording_failed(self.settings, recording_id, message)
            print(f"[STT] failed recording_id={recording_id}: {message}")
            return False

    def _get_transcriber(self) -> Transcriber:
        if self.transcriber is None:
            with self._transcriber_init_lock:
                if self.transcriber is None:
                    self.transcriber = Transcriber(self.settings)
        return self.transcriber

    def _normalize_text(self, raw_text: str) -> str | None:
        if not self.settings.text_normalization_enabled:
            return None

        if self.normalizer is None:
            self.normalizer = TranscriptNormalizer(self.settings.text_normalization_terms_path)

        result = self.normalizer.normalize(raw_text)
        if result.replacement_count:
            print(f"[STT] normalized text replacements={result.replacement_count}")
        return result.text

    def _ensure_transcriber_ready(self) -> bool:
        try:
            self._get_transcriber()
            return True
        except Exception as error:
            print(f"[STT] transcriber init failed: {error}")
            return False


_runtime: WorkerRuntime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> WorkerRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = WorkerRuntime()
    return _runtime


def process_recording(recording_id: int) -> bool:
    return get_runtime().process_recording(recording_id)


def process_batch(run_once: bool = False) -> None:
    runtime = get_runtime()
    if run_once:
        processed = runtime.process_pending_batch()
        if processed == 0:
            print("No STT_PENDING recordings found.")
        return

    stop_event = threading.Event()
    try:
        runtime.poll_forever(stop_event)
    finally:
        stop_event.set()


def _polling_runner(stop_event: threading.Event) -> None:
    runtime = get_runtime()
    runtime.poll_forever(stop_event)


def start_polling_thread() -> tuple[threading.Thread, threading.Event] | None:
    settings = load_settings()
    if not settings.worker_enable_polling:
        print("[STT] polling loop disabled by WORKER_ENABLE_POLLING=false")
        return None

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_polling_runner,
        args=(stop_event,),
        name="nunchi-stt-poller",
        daemon=True,
    )
    thread.start()
    return thread, stop_event


def main() -> None:
    parser = argparse.ArgumentParser(description="Nunchi STT worker")
    parser.add_argument("--once", action="store_true", help="process one batch and exit")
    parser.add_argument("--recording-id", type=int, help="process one specific recording and exit")
    args = parser.parse_args()
    if args.recording_id is not None:
        process_recording(args.recording_id)
        return
    process_batch(run_once=args.once)
