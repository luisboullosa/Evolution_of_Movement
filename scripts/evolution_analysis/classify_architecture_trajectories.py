#!/usr/bin/env python3
"""Classify architecture trajectories from architecture_lineage.jsonl files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def classify_trajectory(events: list[dict]) -> str:
    if not events:
        return "empty"
    complexities = [e.get("child_complexity", 0) for e in events]
    kinds = [e.get("child_kind", "") for e in events]
    if len(set(kinds)) > 2:
        return "adaptive_radiation"
    if any(complexities[i] < complexities[i - 1] for i in range(1, len(complexities))):
        return "revert_to_simple"
    if len(complexities) >= 3 and complexities[-1] > complexities[0] * 2:
        jumps = sum(
            1
            for i in range(1, len(complexities))
            if complexities[i] - complexities[i - 1] > 2
        )
        if jumps >= 1 and len(events) > 20:
            return "punctuated"
    if all(complexities[i] <= complexities[i + 1] for i in range(len(complexities) - 1)):
        return "complexity_monotone"
    return "mixed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Simulation run directory")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    lineage_path = args.run_dir / "architecture_lineage.jsonl"
    if not lineage_path.exists():
        print(f"No lineage file at {lineage_path}")
        return 1

    events = load_events(lineage_path)
    label = classify_trajectory(events)
    result = {
        "run_dir": str(args.run_dir),
        "event_count": len(events),
        "trajectory_class": label,
        "final_kind": events[-1].get("child_kind") if events else None,
        "max_complexity": max((e.get("child_complexity", 0) for e in events), default=0),
    }
    out = args.out or (args.run_dir / "trajectory_class.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
