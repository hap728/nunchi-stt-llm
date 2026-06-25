from pathlib import Path
import re
import sys
import difflib
import json

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

def main():
    if len(sys.argv) < 3:
        print("usage: python compare_text_stt.py original.txt stt_result.txt")
        sys.exit(1)

    ref_path = Path(sys.argv[1])
    hyp_path = Path(sys.argv[2])

    ref_raw = ref_path.read_text(encoding="utf-8")
    hyp_raw = hyp_path.read_text(encoding="utf-8")

    ref = normalize(ref_raw)
    hyp = normalize(hyp_raw)

    dist, cer_percent = cer(ref, hyp)

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    diff = "\n".join(difflib.unified_diff(
        ref_tokens,
        hyp_tokens,
        fromfile="original_text",
        tofile="whisper_text",
        lineterm="",
        n=3,
    ))

    out_dir = Path("/workspace/nunchi-stt/text_compare_results")
    out_dir.mkdir(exist_ok=True)

    stem = hyp_path.stem.replace("result_", "")
    summary_path = out_dir / f"compare_{stem}.json"
    diff_path = out_dir / f"diff_{stem}.txt"
    ref_norm_path = out_dir / f"original_normalized_{stem}.txt"
    hyp_norm_path = out_dir / f"whisper_normalized_{stem}.txt"

    summary = {
        "original_file": str(ref_path),
        "whisper_file": str(hyp_path),
        "original_chars": len(ref.replace(" ", "")),
        "whisper_chars": len(hyp.replace(" ", "")),
        "edit_distance": dist,
        "CER_percent": cer_percent,
        "diff_file": str(diff_path),
        "original_normalized_file": str(ref_norm_path),
        "whisper_normalized_file": str(hyp_norm_path),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    diff_path.write_text(diff, encoding="utf-8")
    ref_norm_path.write_text(ref, encoding="utf-8")
    hyp_norm_path.write_text(hyp, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
