from pathlib import Path
import json
import re
import sys
import time
import requests

OLLAMA_HOST = "http://localhost:11434"
MODEL = "exaone3.5:7.8b"
REFERENCE_DATE = "2026-06-18"

CHUNK_CHARS = 6000
OVERLAP_CHARS = 800

JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


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
                chunk.rfind("다 "),
            )
            if boundary > CHUNK_CHARS * 0.65:
                end = start + boundary + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = max(0, end - OVERLAP_CHARS)

    return [c for c in chunks if c]


def build_prompt(chunk: str, chunk_index: int):
    return f"""
너는 일정 추출기다.
반드시 JSON 배열만 출력한다.
설명 문장, 마크다운, 코드블록을 절대 출력하지 않는다.

기준 날짜: {REFERENCE_DATE}

아래 STT 전사문에서 최종 확정된 일정만 추출해라.
제안되었지만 거절, 취소, 보류된 일정은 제외해라.
상대 날짜 표현은 기준 날짜를 기준으로 YYYY-MM-DD로 변환해라.
확실하지 않은 값은 null로 둔다.
일정이 없으면 []만 출력한다.

출력 형식:
[
  {{
    "title": "일정 제목 또는 null",
    "date": "YYYY-MM-DD 또는 null",
    "date_expr": "전사문에 나온 날짜 표현 또는 null",
    "time": "HH:MM 또는 null",
    "time_expr": "전사문에 나온 시간 표현 또는 null",
    "location": "장소 또는 null",
    "description": "일정 설명",
    "evidence": "근거 문장",
    "source_chunk": {chunk_index}
  }}
]

STT 전사문:
{chunk}
""".strip()


def call_llm(prompt: str):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "당신은 회의 전사문에서 일정만 추출하는 한국어 일정 추출기입니다. JSON만 출력합니다.",
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
    res = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=240)
    res.raise_for_status()
    elapsed = time.perf_counter() - started

    content = res.json().get("message", {}).get("content", "")
    return content, elapsed


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

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return []

    return [x for x in data if isinstance(x, dict)]


def main():
    if len(sys.argv) < 2:
        print("usage: python extract_schedule_from_txt.py result_large-v3_xxx.txt")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    text = input_path.read_text(encoding="utf-8")

    chunks = split_text(text)
    all_events = []
    chunk_results = []

    print("input:", input_path)
    print("chars:", len(text))
    print("chunks:", len(chunks))

    for idx, chunk in enumerate(chunks, start=1):
        prompt = build_prompt(chunk, idx)
        raw, elapsed = call_llm(prompt)
        events = parse_json(raw)

        print(f"[{idx}/{len(chunks)}] events={len(events)} elapsed={elapsed:.1f}s")

        all_events.extend(events)
        chunk_results.append({
            "chunk_index": idx,
            "input_chars": len(chunk),
            "elapsed_sec": round(elapsed, 2),
            "raw_output": raw,
            "events": events,
        })

    result = {
        "input_file": str(input_path),
        "model": MODEL,
        "reference_date": REFERENCE_DATE,
        "chunk_count": len(chunks),
        "events": all_events,
        "chunk_results": chunk_results,
    }

    out_path = input_path.with_name(input_path.stem.replace("result_", "schedule_") + ".json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved:", out_path)


if __name__ == "__main__":
    main()
