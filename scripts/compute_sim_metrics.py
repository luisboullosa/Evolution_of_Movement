#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from statistics import mean, median


def compute(path) -> None:
    p = Path(path)
    if not p.exists():
        print(json.dumps({"error": f"MISSING:{p}"}))
        return
    per_frame_means = []
    per_frame_medians = []
    per_frame_maxes = []
    per_frame_props = []
    frame_count = 0
    first_frame_samples = []
    with p.open("r", encoding="utf-8") as f: sys.TextIOWrapper[_WrappedBuffer]:
        for i, line in enumerate(f):
            line: str = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e: Exception:
                print(json.dumps({"parse_error": str(e), "line_index": i}))
                continue
            frame_count += 1
            inds = obj.get("individuals", [])
            deltas: list[int] = [abs(int(x.get("last_delta", 0))) for x in inds]
            if not deltas:
                per_frame_means.append(0.0)
                per_frame_medians.append(0.0)
                per_frame_maxes.append(0)
                per_frame_props.append(0.0)
            else:
                per_frame_means.append(mean(deltas))
                per_frame_medians.append(median(deltas))
                per_frame_maxes.append(max(deltas))
                moved: int = sum(1 for x: int in deltas if x > 0)
                per_frame_props.append(moved / len(deltas))
            if i < 6:
                first_frame_samples.append(
                    {
                        "t": obj.get("t"),
                        "mean": per_frame_means[-1],
                        "median": per_frame_medians[-1],
                        "max": per_frame_maxes[-1],
                        "prop_moved": per_frame_props[-1],
                    }
                )
    overall = {
        "path": str(p),
        "frames": frame_count,
        "overall_mean_of_per_frame_means": mean(per_frame_means) if per_frame_means else 0.0,
        "overall_median_of_per_frame_medians": median(per_frame_medians) if per_frame_medians else 0.0,
        "overall_max_of_per_frame_maxes": max(per_frame_maxes) if per_frame_maxes else 0,
        "overall_mean_prop_moved": mean(per_frame_props) if per_frame_props else 0.0,
        "first_frame_samples": first_frame_samples,
    }
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        compute(sys.argv[1])
    else:
        compute(r"d:\Development\Movement\webviewer\sim0046\individuals.jsonl")
