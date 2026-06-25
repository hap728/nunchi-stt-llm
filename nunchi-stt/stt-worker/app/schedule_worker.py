from __future__ import annotations

import argparse
import time

from .config import Settings, load_settings
from .db import (
    get_latest_stt_result,
    mark_schedule_status,
    save_schedule_result,
)
from .schedule_extractor import extract_schedule


def process_schedule_for_recording(
    recording_id: int,
    user_id: int | None = None,
    raw_text: str | None = None,
    settings: Settings | None = None,
) -> bool:
    settings = settings or load_settings()
    if not settings.llm_schedule_enabled:
        print(f"[LLM] schedule extraction disabled recording_id={recording_id}")
        return False

    try:
        if raw_text is None or user_id is None:
            stt_result = get_latest_stt_result(settings, recording_id)
            if not stt_result:
                raise RuntimeError(f"STT_RESULT_NOT_FOUND: {recording_id}")

            user_id = int(stt_result["user_id"])
            raw_text = stt_result.get("raw_text") or stt_result.get("corrected_text")

        if not raw_text:
            raise RuntimeError(f"STT_RAW_TEXT_EMPTY: {recording_id}")

        print(f"[LLM] processing recording_id={recording_id}")
        mark_schedule_status(settings, recording_id, "LLM_PROCESSING")

        started_at = time.perf_counter()
        schedule = extract_schedule(settings, raw_text)
        save_schedule_result(settings, recording_id, int(user_id), schedule)
        mark_schedule_status(settings, recording_id, "LLM_DONE")

        print(
            f"[LLM] done recording_id={recording_id} "
            f"calendar_events={schedule['calendar_event_count']} "
            f"review_events={schedule['review_event_count']} "
            f"elapsed={time.perf_counter() - started_at:.1f}s"
        )
        return True
    except Exception as error:
        message = str(error) or error.__class__.__name__
        try:
            mark_schedule_status(settings, recording_id, "LLM_FAILED", message)
        except Exception as status_error:
            print(f"[LLM] failed to update status recording_id={recording_id}: {status_error}")
        print(f"[LLM] failed recording_id={recording_id}: {message}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Nunchi schedule extraction worker")
    parser.add_argument("--recording-id", type=int, required=True, help="recording_id to process")
    args = parser.parse_args()
    process_schedule_for_recording(args.recording_id)


if __name__ == "__main__":
    main()
