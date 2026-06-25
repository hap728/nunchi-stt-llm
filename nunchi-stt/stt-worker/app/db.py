from __future__ import annotations

import json
import re
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from .config import Settings

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def get_connection(settings: Settings) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_username,
        password=settings.db_password,
        database=settings.db_name,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
    )


def claim_pending_recordings(settings: Settings, batch_size: int) -> list[dict[str, Any]]:
    claimed: list[dict[str, Any]] = []
    with get_connection(settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT recording_id, user_id, audio_url, object_key, status, retry_count
                FROM recordings
                WHERE status IN ('STT_PENDING', 'STT_REQUESTED', 'UPLOADED')
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (batch_size,),
            )
            candidates = cursor.fetchall()

            for row in candidates:
                updated = cursor.execute(
                    """
                    UPDATE recordings
                    SET status = 'STT_PROCESSING',
                        processing_started_at = NOW(),
                        stt_completed_at = NULL,
                        failure_reason = NULL
                    WHERE recording_id = %s
                      AND status IN ('STT_PENDING', 'STT_REQUESTED', 'UPLOADED')
                    """,
                    (row["recording_id"],),
                )
                if updated == 1:
                    claimed.append(row)

        connection.commit()
    return claimed


def claim_recording_by_id(settings: Settings, recording_id: int) -> dict[str, Any] | None:
    with get_connection(settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT recording_id, user_id, audio_url, object_key, status, retry_count
                FROM recordings
                WHERE recording_id = %s
                """,
                (recording_id,),
            )
            row = cursor.fetchone()
            if not row:
                connection.commit()
                return None

            updated = cursor.execute(
                """
                UPDATE recordings
                SET status = 'STT_PROCESSING',
                    processing_started_at = NOW(),
                    stt_completed_at = NULL,
                    failure_reason = NULL
                WHERE recording_id = %s
                  AND status IN ('STT_PENDING', 'STT_REQUESTED', 'UPLOADED')
                """,
                (recording_id,),
            )
            connection.commit()

            if updated != 1:
                return None

            return row


def save_stt_result(
    settings: Settings,
    recording_id: int,
    user_id: int,
    raw_text: str,
    corrected_text: str | None = None,
) -> None:
    with get_connection(settings) as connection:
        with connection.cursor() as cursor:
            if _has_column(cursor, "stt_results", "corrected_text"):
                cursor.execute(
                    """
                    INSERT INTO stt_results (
                        recording_id,
                        user_id,
                        raw_text,
                        corrected_text,
                        processed_at
                    )
                    VALUES (%s, %s, %s, %s, NOW())
                    """,
                    (recording_id, user_id, raw_text, corrected_text),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO stt_results (
                        recording_id,
                        user_id,
                        raw_text,
                        processed_at
                    )
                    VALUES (%s, %s, %s, NOW())
                    """,
                    (recording_id, user_id, raw_text),
                )
            cursor.execute(
                """
                UPDATE recordings
                SET status = 'STT_DONE',
                    stt_completed_at = NOW(),
                    failure_reason = NULL
                WHERE recording_id = %s
                """,
                (recording_id,),
            )
        connection.commit()


def get_latest_stt_result(settings: Settings, recording_id: int) -> dict[str, Any] | None:
    with get_connection(settings) as connection:
        with connection.cursor() as cursor:
            select_columns = ["recording_id", "user_id", "raw_text"]
            if _has_column(cursor, "stt_results", "corrected_text"):
                select_columns.append("corrected_text")

            order_by = ""
            if _has_column(cursor, "stt_results", "processed_at"):
                order_by = "ORDER BY processed_at DESC"
            elif _has_column(cursor, "stt_results", "stt_result_id"):
                order_by = "ORDER BY stt_result_id DESC"

            cursor.execute(
                f"""
                SELECT {", ".join(select_columns)}
                FROM stt_results
                WHERE recording_id = %s
                {order_by}
                LIMIT 1
                """,
                (recording_id,),
            )
            row = cursor.fetchone()
        connection.commit()
    return row


def save_schedule_result(
    settings: Settings,
    recording_id: int,
    user_id: int,
    schedule: dict[str, Any],
) -> None:
    table_name = settings.schedule_results_table
    result_json = json.dumps(schedule, ensure_ascii=False)
    calendar_events_json = json.dumps(schedule.get("calendar_events", schedule.get("events", [])), ensure_ascii=False)
    review_events_json = json.dumps(schedule.get("review_events", []), ensure_ascii=False)

    with get_connection(settings) as connection:
        with connection.cursor() as cursor:
            if not _has_table(cursor, table_name):
                raise RuntimeError(f"SCHEDULE_RESULTS_TABLE_NOT_FOUND: {table_name}")

            table_columns = _get_columns(cursor, table_name)
            columns: list[str] = []
            placeholders: list[str] = []
            params: list[Any] = []

            def add_value(column_name: str, value: Any) -> None:
                if column_name not in table_columns:
                    return
                columns.append(column_name)
                placeholders.append("%s")
                params.append(value)

            def add_now(column_name: str) -> None:
                if column_name not in table_columns:
                    return
                columns.append(column_name)
                placeholders.append("NOW()")

            add_value("recording_id", recording_id)
            add_value("user_id", user_id)
            add_value("model_name", schedule.get("model"))
            add_value("model", schedule.get("model"))
            add_value("reference_date", schedule.get("reference_date"))
            add_value("chunk_count", schedule.get("chunk_count"))
            add_value("raw_event_count", schedule.get("raw_event_count"))
            add_value("calendar_event_count", schedule.get("calendar_event_count"))
            add_value("review_event_count", schedule.get("review_event_count"))

            for json_column in (
                "result_json",
                "schedule_json",
                "extracted_json",
                "llm_result_json",
                "raw_json",
            ):
                add_value(json_column, result_json)

            for json_column in ("events_json", "calendar_events_json"):
                add_value(json_column, calendar_events_json)

            add_value("review_events_json", review_events_json)
            add_now("processed_at")
            add_now("created_at")

            json_columns = {
                "result_json",
                "schedule_json",
                "extracted_json",
                "llm_result_json",
                "raw_json",
                "events_json",
                "calendar_events_json",
                "review_events_json",
            }
            if not any(column in columns for column in json_columns):
                raise RuntimeError(
                    "SCHEDULE_RESULTS_TABLE_HAS_NO_SUPPORTED_JSON_COLUMN: "
                    f"{table_name}"
                )

            quoted_table = _quote_identifier(table_name)
            quoted_columns = ", ".join(_quote_identifier(column) for column in columns)
            cursor.execute(
                f"""
                INSERT INTO {quoted_table} ({quoted_columns})
                VALUES ({", ".join(placeholders)})
                """,
                tuple(params),
            )
        connection.commit()


def mark_schedule_status(
    settings: Settings,
    recording_id: int,
    status: str,
    failure_reason: str | None = None,
) -> None:
    with get_connection(settings) as connection:
        with connection.cursor() as cursor:
            recording_columns = _get_columns(cursor, "recordings")
            updates: list[str] = []
            params: list[Any] = []

            if "llm_status" in recording_columns:
                updates.append("llm_status = %s")
                params.append(status)
            elif "schedule_status" in recording_columns:
                updates.append("schedule_status = %s")
                params.append(status)

            if "llm_failure_reason" in recording_columns:
                updates.append("llm_failure_reason = %s")
                params.append(_trim_failure_reason(failure_reason or ""))
            elif "schedule_failure_reason" in recording_columns:
                updates.append("schedule_failure_reason = %s")
                params.append(_trim_failure_reason(failure_reason or ""))

            if status == "LLM_PROCESSING":
                if "llm_started_at" in recording_columns:
                    updates.append("llm_started_at = NOW()")
                elif "schedule_started_at" in recording_columns:
                    updates.append("schedule_started_at = NOW()")

            if status == "LLM_DONE":
                if "llm_completed_at" in recording_columns:
                    updates.append("llm_completed_at = NOW()")
                elif "schedule_completed_at" in recording_columns:
                    updates.append("schedule_completed_at = NOW()")

            if not updates:
                connection.commit()
                return

            params.append(recording_id)
            cursor.execute(
                f"""
                UPDATE recordings
                SET {", ".join(updates)}
                WHERE recording_id = %s
                """,
                tuple(params),
            )
        connection.commit()


def _has_column(cursor: DictCursor, table_name: str, column_name: str) -> bool:
    cursor.execute(f"SHOW COLUMNS FROM {_quote_identifier(table_name)} LIKE %s", (column_name,))
    return cursor.fetchone() is not None


def _has_table(cursor: DictCursor, table_name: str) -> bool:
    cursor.execute("SHOW TABLES LIKE %s", (table_name,))
    return cursor.fetchone() is not None


def _get_columns(cursor: DictCursor, table_name: str) -> set[str]:
    cursor.execute(f"SHOW COLUMNS FROM {_quote_identifier(table_name)}")
    return {str(row["Field"]) for row in cursor.fetchall()}


def _quote_identifier(identifier: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(f"INVALID_SQL_IDENTIFIER: {identifier}")
    return f"`{identifier}`"


def mark_recording_failed(settings: Settings, recording_id: int, failure_reason: str) -> None:
    with get_connection(settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE recordings
                SET status = 'STT_FAILED',
                    retry_count = retry_count + 1,
                    failure_reason = %s
                WHERE recording_id = %s
                """,
                (_trim_failure_reason(failure_reason), recording_id),
            )
        connection.commit()


def _trim_failure_reason(message: str) -> str:
    if not message:
        return "UNKNOWN_ERROR"
    return message[:255]
