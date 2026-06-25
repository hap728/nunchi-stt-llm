from pathlib import Path
import difflib
import json
import re
import sys

def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1]

def cer(ref: str, hyp: str):
    r = ref.replace(" ", "")
    h = hyp.replace(" ", "")
    dist = edit_distance(r, h)
    return dist, round(dist / max(1, len(r)) * 100, 2)

def grouped_diffs(ref_tokens, hyp_tokens):
    matcher = difflib.SequenceMatcher(a=ref_tokens, b=hyp_tokens)
    rows = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        before_context = " ".join(ref_tokens[max(0, i1 - 5):i1])
        after_context = " ".join(ref_tokens[i2:i2 + 5])

        ref_part = " ".join(ref_tokens[i1:i2])
        hyp_part = " ".join(hyp_tokens[j1:j2])

        if tag == "replace":
            kind = "교체"
        elif tag == "delete":
            kind = "누락"
        elif tag == "insert":
            kind = "추가"
        else:
            kind = tag

        rows.append({
            "type": kind,
            "before_context": before_context,
            "original": ref_part,
            "whisper": hyp_part,
            "after_context": after_context,
        })

    return rows

def main():
    if len(sys.argv) < 3:
        print("usage: python compare_text_stt_readable.py original.txt whisper_result.txt")
        sys.exit(1)

    ref_path = Path(sys.argv[1])
    hyp_path = Path(sys.argv[2])

    ref = normalize(ref_path.read_text(encoding="utf-8"))
    hyp = normalize(hyp_path.read_text(encoding="utf-8"))

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    dist, cer_percent = cer(ref, hyp)
    rows = grouped_diffs(ref_tokens, hyp_tokens)

    out_dir = Path("/workspace/nunchi-stt/text_compare_results")
    out_dir.mkdir(exist_ok=True)

    stem = hyp_path.stem.replace("result_", "")
    readable_path = out_dir / f"readable_diff_{stem}.txt"
    json_path = out_dir / f"readable_diff_{stem}.json"

    lines = []
    lines.append(f"original_file: {ref_path}")
    lines.append(f"whisper_file: {hyp_path}")
    lines.append(f"original_chars: {len(ref.replace(' ', ''))}")
    lines.append(f"whisper_chars: {len(hyp.replace(' ', ''))}")
    lines.append(f"edit_distance: {dist}")
    lines.append(f"CER_percent: {cer_percent}%")
    lines.append(f"diff_groups: {len(rows)}")
    lines.append("")
    lines.append("=" * 100)
    lines.append("원본과 Whisper가 다르게 나온 부분")
    lines.append("=" * 100)

    for idx, row in enumerate(rows, start=1):
        lines.append("")
        lines.append(f"[{idx}] {row['type']}")
        if row["before_context"]:
            lines.append(f"앞문맥: {row['before_context']}")
        lines.append(f"원본  : {row['original']}")
        lines.append(f"Whisper: {row['whisper']}")
        if row["after_context"]:
            lines.append(f"뒷문맥: {row['after_context']}")

    readable_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps({
        "original_file": str(ref_path),
        "whisper_file": str(hyp_path),
        "original_chars": len(ref.replace(" ", "")),
        "whisper_chars": len(hyp.replace(" ", "")),
        "edit_distance": dist,
        "CER_percent": cer_percent,
        "diff_groups": len(rows),
        "diffs": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved:", readable_path)
    print("saved:", json_path)
    print("CER:", cer_percent, "%")
    print("diff_groups:", len(rows))

if __name__ == "__main__":
    main()
