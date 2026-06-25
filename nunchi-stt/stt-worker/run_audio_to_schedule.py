from pathlib import Path
import argparse
import gc
import json
import os
import re
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from app.config import load_settings
from app.transcriber import Transcriber

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "exaone3.5:7.8b")
REFERENCE_DATE = os.getenv("REFERENCE_DATE") or datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()

CHUNK_CHARS = 6000
OVERLAP_CHARS = 800
JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)

WEEKDAY_TO_INDEX = {
    "월": 0,
    "월요일": 0,
    "화": 1,
    "화요일": 1,
    "수": 2,
    "수요일": 2,
    "목": 3,
    "목요일": 3,
    "금": 4,
    "금요일": 4,
    "토": 5,
    "토요일": 5,
    "일": 6,
    "일요일": 6,
}

ORDINAL_WEEK = {
    "첫": 1,
    "첫째": 1,
    "두": 2,
    "둘째": 2,
    "셋": 3,
    "셋째": 3,
    "세": 3,
    "넷": 4,
    "넷째": 4,
    "네": 4,
    "다섯": 5,
    "다섯째": 5,
}


def split_text(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
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


def build_prompt(chunk: str, chunk_index: int):
    return f"""
너는 회의 STT 전사문에서 캘린더에 등록할 수 있는 일정 후보를 추출하는 일정 추출기다.

반드시 지켜야 할 규칙:
- 반드시 JSON 배열만 출력한다.
- 설명 문장, 마크다운, 코드블록을 출력하지 않는다.
- {{"schedules": [...]}} 또는 {{"events": [...]}} 형태로 감싸지 않는다.
- 일정 후보가 없으면 []만 출력한다.
- 단순 작업 항목, 일반 논의, 리스크, 품질 검토 내용은 제외한다.
- 회의, 리뷰, 리허설, 발표, 공유, 마감, 제출, 점검처럼 특정 날짜/시간에 수행될 가능성이 있는 항목은 일정 후보로 추출한다.
- 날짜나 시간이 일부 부족해도 일정 후보라면 추출한다.
- 확정되지 않은 값은 null로 둔다.
- date는 항상 null로 둔다. 실제 날짜 계산은 시스템에서 처리한다.
- date_expr에는 원문 날짜 표현을 그대로 넣는다.
- time은 HH:MM 24시간 형식으로 넣는다.
- time을 확신할 수 없으면 null로 둔다.
- time_expr에는 원문 시간 표현을 그대로 넣는다.
- 반복 일정이면 recurrence에 WEEKLY, MONTHLY 같은 값을 넣는다.
- 장소가 없으면 location은 null로 둔다.
- evidence에는 일정이라고 판단한 원문 근거 문장을 넣는다.
- source_chunk는 {chunk_index}로 넣는다.

기준 날짜: {REFERENCE_DATE}

출력 형식:
[
  {{
    "title": "일정 제목",
    "date": null,
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


def call_llm(prompt: str):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "너는 한국어 회의 전사문에서 캘린더 등록 가능한 일정 후보를 "
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

    started = time.perf_counter()
    res = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=300)
    res.raise_for_status()
    elapsed = time.perf_counter() - started

    return res.json().get("message", {}).get("content", ""), elapsed


def parse_json(text: str):
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

    return flatten_events(data)


def flatten_events(data):
    if isinstance(data, dict):
        if isinstance(data.get("events"), list):
            data = data["events"]
        elif isinstance(data.get("schedules"), list):
            data = data["schedules"]
        elif isinstance(data.get("schedule"), list):
            data = data["schedule"]
        elif not data:
            return []
        else:
            data = [data]

    if not isinstance(data, list):
        return []

    flattened = []

    for item in data:
        if not isinstance(item, dict):
            continue

        if not item:
            continue

        if isinstance(item.get("events"), list):
            flattened.extend(flatten_events(item["events"]))
            continue

        if isinstance(item.get("schedules"), list):
            flattened.extend(flatten_events(item["schedules"]))
            continue

        if isinstance(item.get("schedule"), list):
            flattened.extend(flatten_events(item["schedule"]))
            continue

        flattened.append(item)

    return flattened


def clean_value(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ("null", "none", "undefined", "nan"):
            return None
        if value in ("", "미정", "없음", "보류", "확인 필요"):
            return None

    return value


def normalize_event(event, chunk_index=None):
    keys = [
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
    ]

    normalized = {key: clean_value(event.get(key)) for key in keys}
    normalized["date"] = resolve_date(normalized.get("date"), normalized.get("date_expr"))
    normalized["time"] = normalize_time(normalized.get("time"), normalized.get("time_expr"))
    normalized["recurrence"] = normalize_recurrence(normalized)

    if chunk_index is not None:
        normalized["source_chunk"] = chunk_index
    else:
        normalized["source_chunk"] = normalize_source_chunk(normalized.get("source_chunk"))

    return normalized


def normalize_source_chunk(value):
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def parse_reference_date():
    return datetime.strptime(REFERENCE_DATE, "%Y-%m-%d").date()


def resolve_date(date_value, date_expr):
    # Important: trust date_expr first. LLM often miscalculates ISO dates.
    if isinstance(date_expr, str) and date_expr.strip():
        resolved = resolve_date_expr(date_expr)
        if resolved:
            return resolved

    if isinstance(date_value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_value):
        parsed = datetime.strptime(date_value, "%Y-%m-%d").date()
        if parsed >= parse_reference_date():
            return date_value

    return None


def resolve_date_expr(date_expr):
    expr = re.sub(r"\s+", "", str(date_expr))
    base = parse_reference_date()

    if "오늘" in expr or "금일" in expr:
        return base.isoformat()

    if "내일모레" in expr:
        return (base + timedelta(days=2)).isoformat()

    if "내일" in expr or "명일" in expr:
        return (base + timedelta(days=1)).isoformat()

    if "모레" in expr:
        return (base + timedelta(days=2)).isoformat()

    if "글피" in expr:
        return (base + timedelta(days=3)).isoformat()

    explicit = parse_explicit_korean_date(expr, base)
    if explicit:
        return explicit.isoformat()

    ordinal_week = parse_ordinal_weekday(expr, base)
    if ordinal_week:
        return ordinal_week.isoformat()

    weekday = find_weekday(expr)
    if weekday is None:
        return None

    if "다다음주" in expr:
        return weekday_in_relative_week(base, weekday, 2).isoformat()

    if "다음주" in expr or "담주" in expr:
        return weekday_in_relative_week(base, weekday, 1).isoformat()

    if "이번주" in expr or "금주" in expr:
        return weekday_in_relative_week(base, weekday, 0).isoformat()

    if "매주" in expr or "반복" in expr:
        return next_weekday(base, weekday).isoformat()

    return next_weekday(base, weekday).isoformat()


def parse_explicit_korean_date(expr: str, base: date):
    match = re.search(r"(?:(\d{4})년)?(\d{1,2})월(\d{1,2})일", expr)
    if not match:
        return None

    year = int(match.group(1)) if match.group(1) else base.year
    month = int(match.group(2))
    day = int(match.group(3))

    try:
        parsed = date(year, month, day)
    except ValueError:
        return None

    if parsed < base and not match.group(1):
        try:
            parsed = date(year + 1, month, day)
        except ValueError:
            return None

    return parsed


def parse_ordinal_weekday(expr: str, base: date):
    match = re.search(
        r"(?:(\d{4})년)?(\d{1,2})월(첫째|첫|두|둘째|셋|셋째|세|넷|넷째|네|다섯|다섯째|마지막|말)(?:번째)?주?([월화수목금토일])요일?",
        expr,
    )
    if not match:
        return None

    year = int(match.group(1)) if match.group(1) else base.year
    month = int(match.group(2))
    ordinal = match.group(3)
    weekday = WEEKDAY_TO_INDEX[match.group(4)]

    if not match.group(1) and month < base.month:
        year += 1

    if ordinal in ("마지막", "말"):
        return last_weekday_of_month(year, month, weekday)

    return nth_weekday_of_month(year, month, weekday, ORDINAL_WEEK[ordinal])


def nth_weekday_of_month(year: int, month: int, weekday: int, nth: int):
    first_day = date(year, month, 1)
    days_until_weekday = (weekday - first_day.weekday()) % 7
    return first_day + timedelta(days=days_until_weekday + 7 * (nth - 1))


def last_weekday_of_month(year: int, month: int, weekday: int):
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    return last_day - timedelta(days=(last_day.weekday() - weekday) % 7)


def find_weekday(expr: str):
    for name in sorted(WEEKDAY_TO_INDEX.keys(), key=len, reverse=True):
        if name in expr:
            return WEEKDAY_TO_INDEX[name]
    return None


def weekday_in_relative_week(base: date, target_weekday: int, week_offset: int):
    monday = base - timedelta(days=base.weekday())
    return monday + timedelta(days=week_offset * 7 + target_weekday)


def next_weekday(base: date, target_weekday: int):
    days_ahead = target_weekday - base.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return base + timedelta(days=days_ahead)


def normalize_time(time_value, time_expr):
    if isinstance(time_value, str):
        match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", time_value.strip())
        if match:
            return f"{int(match.group(1)):02d}:{match.group(2)}"

    source = " ".join(str(value) for value in (time_value, time_expr) if value)
    if not source:
        return None

    source = source.replace(" ", "")
    source = source.replace("반", "30분")

    match = re.search(r"(오전|오후)?(\d{1,2})시(?:(\d{1,2})분)?", source)
    if not match:
        return None

    meridiem = match.group(1)
    hour = int(match.group(2))
    minute = int(match.group(3) or 0)

    if meridiem == "오후" and hour < 12:
        hour += 12
    if meridiem == "오전" and hour == 12:
        hour = 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    return f"{hour:02d}:{minute:02d}"


def normalize_recurrence(event):
    recurrence = clean_value(event.get("recurrence"))
    text = " ".join(
        str(value)
        for value in (
            recurrence,
            event.get("date_expr"),
            event.get("description"),
            event.get("evidence"),
        )
        if value
    )

    if "매주" in text or "반복" in text:
        return "WEEKLY"
    if "매월" in text:
        return "MONTHLY"

    if isinstance(recurrence, str):
        recurrence = recurrence.upper()
        if recurrence in ("WEEKLY", "MONTHLY", "DAILY", "YEARLY"):
            return recurrence

    return None


def is_calendar_ready(event):
    return bool(event.get("title") and event.get("date") and event.get("time"))


def split_calendar_events(events):
    calendar_events = []
    review_events = []
    seen = set()

    for raw_event in events:
        event = normalize_event(raw_event, raw_event.get("_source_chunk"))
        signature = (
            event.get("title"),
            event.get("date"),
            event.get("time"),
            event.get("recurrence"),
        )

        if signature in seen:
            continue
        seen.add(signature)

        if is_calendar_ready(event):
            calendar_events.append(event)
        else:
            missing = []
            if not event.get("title"):
                missing.append("title")
            if not event.get("date"):
                missing.append("date")
            if not event.get("time"):
                missing.append("time")
            event["review_reason"] = "missing_" + "_".join(missing)
            review_events.append(event)

    return calendar_events, review_events


def extract_schedule(raw_text: str):
    chunks = split_text(raw_text)
    raw_events = []
    chunk_results = []

    print(f"[LLM] chunks={len(chunks)} model={LLM_MODEL}")

    for idx, chunk in enumerate(chunks, start=1):
        prompt = build_prompt(chunk, idx)
        raw_output, elapsed = call_llm(prompt)
        events = parse_json(raw_output)

        for event in events:
            event["_source_chunk"] = idx

        print(f"[LLM] chunk={idx}/{len(chunks)} events={len(events)} elapsed={elapsed:.1f}s")

        raw_events.extend(events)
        chunk_results.append(
            {
                "chunk_index": idx,
                "input_chars": len(chunk),
                "elapsed_sec": round(elapsed, 2),
                "raw_output": raw_output,
                "events": events,
            }
        )

    calendar_events, review_events = split_calendar_events(raw_events)

    return {
        "model": LLM_MODEL,
        "reference_date": REFERENCE_DATE,
        "chunk_count": len(chunks),
        "raw_event_count": len(raw_events),
        "calendar_event_count": len(calendar_events),
        "review_event_count": len(review_events),
        "events": calendar_events,
        "calendar_events": calendar_events,
        "review_events": review_events,
        "chunk_results": chunk_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--out-dir", default="/workspace/nunchi-stt/pipeline_results")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = audio_path.stem
    stt_path = out_dir / f"result_{stem}.txt"
    json_path = out_dir / f"schedule_{stem}.json"

    print(f"[PIPELINE] audio={audio_path}")

    settings = load_settings()
    transcriber = Transcriber(settings)

    stt_started = time.perf_counter()
    raw_text = transcriber.transcribe(audio_path)
    stt_elapsed = time.perf_counter() - stt_started

    stt_path.write_text(raw_text, encoding="utf-8")
    print(f"[STT] saved={stt_path} elapsed={stt_elapsed:.1f}s chars={len(raw_text.replace(' ', ''))}")

    del transcriber
    gc.collect()

    llm_started = time.perf_counter()
    schedule = extract_schedule(raw_text)
    llm_elapsed = time.perf_counter() - llm_started

    result = {
        "audio_file": str(audio_path),
        "stt_file": str(stt_path),
        "stt_elapsed_sec": round(stt_elapsed, 2),
        "llm_elapsed_sec": round(llm_elapsed, 2),
        "total_elapsed_sec": round(stt_elapsed + llm_elapsed, 2),
        "raw_text": raw_text,
        "schedule": schedule,
    }

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] saved={json_path}")
    print(f"[DONE] calendar_events={schedule['calendar_event_count']} review_events={schedule['review_event_count']}")


if __name__ == "__main__":
    main()
