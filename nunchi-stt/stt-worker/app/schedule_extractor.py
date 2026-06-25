from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

from .config import Settings


CHUNK_CHARS = 6000
OVERLAP_CHARS = 800
JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def extract_schedule(settings: Settings, raw_text: str) -> dict[str, Any]:
    chunks = split_text(raw_text)
    raw_events: list[dict[str, Any]] = []
    chunk_results: list[dict[str, Any]] = []

    print(f"[LLM] chunks={len(chunks)} model={settings.llm_model}")

    for index, chunk in enumerate(chunks, start=1):
        prompt = build_prompt(settings, chunk, index)
        raw_output, elapsed = call_llm(settings, prompt)
        events = parse_json(raw_output)

        for event in events:
            event["source_chunk"] = event.get("source_chunk") or index

        print(f"[LLM] chunk={index}/{len(chunks)} events={len(events)} elapsed={elapsed:.1f}s")

        raw_events.extend(events)
        chunk_results.append(
            {
                "chunk_index": index,
                "input_chars": len(chunk),
                "elapsed_sec": round(elapsed, 2),
                "raw_output": raw_output,
                "events": events,
            }
        )

    calendar_events, review_events = split_calendar_events(raw_events)

    return {
        "model": settings.llm_model,
        "reference_date": get_reference_date(settings),
        "chunk_count": len(chunks),
        "raw_event_count": len(raw_events),
        "calendar_event_count": len(calendar_events),
        "review_event_count": len(review_events),
        "events": calendar_events,
        "calendar_events": calendar_events,
        "review_events": review_events,
        "chunk_results": chunk_results,
    }


def split_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        chunk = text[start:end]

        if end < len(text):
            boundary = max(
                chunk.rfind("."),
                chunk.rfind("?"),
                chunk.rfind("!"),
                chunk.rfind("다."),
                chunk.rfind("요."),
            )
            if boundary > CHUNK_CHARS * 0.65:
                end = start + boundary + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = max(0, end - OVERLAP_CHARS)

    return [chunk for chunk in chunks if chunk]


def build_prompt(settings: Settings, chunk: str, chunk_index: int) -> str:
    reference_date = get_reference_date(settings)
    return f"""
너는 회의 STT 전사문에서 캘린더에 등록할 수 있는 일정 정보를 추출하는 일정 추출기다.

반드시 지켜야 할 규칙:
- 반드시 JSON 배열만 출력한다.
- 설명 문장, 마크다운, 코드블록을 출력하지 않는다.
- 일정 정보가 없으면 []만 출력한다.
- 단순 작업 항목, 일반 논의, 리스크, 품질 검토 내용은 제외한다.
- 회의, 리뷰, 리허설, 발표, 공유, 마감, 제출, 점검처럼 특정 날짜나 시간에 수행될 가능성이 있는 항목은 일정 후보로 추출한다.
- 확정되지 않은 값은 null로 둔다.
- date는 YYYY-MM-DD 형식으로 넣고, 날짜 계산이 어렵다면 null로 둔다.
- date_expr에는 원문 날짜 표현을 그대로 넣는다.
- time은 HH:MM 24시간 형식으로 넣고, 확신할 수 없으면 null로 둔다.
- time_expr에는 원문 시간 표현을 그대로 넣는다.
- 장소가 없으면 location은 null로 둔다.
- evidence에는 일정이라고 판단한 원문 근거 문장을 넣는다.
- source_chunk는 {chunk_index}로 넣는다.

기준 날짜: {reference_date}

출력 형식:
[
  {{
    "title": "일정 제목",
    "date": "YYYY-MM-DD 또는 null",
    "date_expr": "원문 날짜 표현 또는 null",
    "time": "HH:MM 또는 null",
    "time_expr": "원문 시간 표현 또는 null",
    "location": "장소 또는 null",
    "description": "일정 설명",
    "evidence": "근거 문장",
    "recurrence": "WEEKLY 또는 MONTHLY 또는 null",
    "source_chunk": {chunk_index}
  }}
]

STT 전사문:
{chunk}
""".strip()


def call_llm(settings: Settings, prompt: str) -> tuple[str, float]:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "너는 한국어 회의 전사문에서 캘린더 등록 가능한 일정 정보만 "
                    "JSON 배열로 추출하는 일정 추출기다. JSON만 출력한다."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "top_p": 0.1,
        },
    }

    started_at = time.perf_counter()
    response = requests.post(
        f"{settings.ollama_host}/api/chat",
        json=payload,
        timeout=settings.llm_request_timeout_sec,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started_at
    return response.json().get("message", {}).get("content", ""), elapsed


def parse_json(text: str) -> list[dict[str, Any]]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    try:
        data = json.loads(text)
    except Exception:
        match = JSON_BLOCK.search(text)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except Exception:
            return []

    if isinstance(data, dict):
        if isinstance(data.get("events"), list):
            data = data["events"]
        elif isinstance(data.get("schedules"), list):
            data = data["schedules"]
        elif isinstance(data.get("schedule"), list):
            data = data["schedule"]
        elif data:
            data = [data]
        else:
            return []

    if not isinstance(data, list):
        return []

    return [normalize_event(item) for item in data if isinstance(item, dict)]


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "title",
        "date",
        "date_expr",
        "time",
        "time_expr",
        "location",
        "description",
        "evidence",
        "recurrence",
        "source_chunk",
    )
    normalized = {field: event.get(field) for field in fields}

    if isinstance(normalized["title"], str):
        normalized["title"] = normalized["title"].strip() or None
    if isinstance(normalized["date"], str):
        normalized["date"] = normalize_date(normalized["date"])
    if isinstance(normalized["time"], str):
        normalized["time"] = normalize_time(normalized["time"])

    return normalized


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    return None


def normalize_time(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", value)
    if match:
        return f"{int(match.group(1)):02d}:{match.group(2)}"
    return None


def split_calendar_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calendar_events: list[dict[str, Any]] = []
    review_events: list[dict[str, Any]] = []

    for event in events:
        missing: list[str] = []
        if not event.get("title"):
            missing.append("title")
        if not event.get("date") and not event.get("date_expr"):
            missing.append("date")
        if not event.get("time") and not event.get("time_expr"):
            missing.append("time")

        if missing:
            event["review_reason"] = "missing_" + "_".join(missing)
            review_events.append(event)
        else:
            calendar_events.append(event)

    return calendar_events, review_events


def get_reference_date(settings: Settings) -> str:
    if settings.llm_reference_date:
        return settings.llm_reference_date
    return datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
