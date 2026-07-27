#!/usr/bin/env python3
"""Compute phylogenetic metrics from cladogram.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def tree_balance_factor(children_counts: list[int]) -> float:
    if not children_counts:
        return 0.0
    return sum(children_counts) / len(children_counts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cladogram_json", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.cladogram_json.read_text(encoding="utf-8"))
    nodes = data.get("nodes", {})
    child_counts = [len(n.get("children", [])) for n in nodes.values()]
    branch_lengths = [
        n.get("genetic_distance_to_parent", 0.0)
        for n in nodes.values()
        if n.get("parent_id") is not None
    ]
    result = {
        "node_count": len(nodes),
        "mean_branching": tree_balance_factor(child_counts),
        "mean_branch_length": sum(branch_lengths) / max(len(branch_lengths), 1),
        "max_branch_length": max(branch_lengths) if branch_lengths else 0.0,
    }
    out = args.out or args.cladogram_json.with_name("phylo_metrics.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
