#!/usr/bin/env python3
"""Summarize architecture kind transitions over time from lineage JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lineage_jsonl", type=Path)
    args = parser.parse_args()

    transitions: Counter[tuple[str, str]] = Counter()
    kind_counts: Counter[str] = Counter()
    with args.lineage_jsonl.open(encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            pk = ev.get("parent_kind", "?")
            ck = ev.get("child_kind", "?")
            transitions[(pk, ck)] += 1
            kind_counts[ck] += 1

    report = {
        "transitions": {f"{a}->{b}": c for (a, b), c in transitions.items()},
        "child_kind_counts": dict(kind_counts),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
