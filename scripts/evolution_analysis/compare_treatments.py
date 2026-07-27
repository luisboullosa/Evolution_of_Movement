#!/usr/bin/env python3
"""Compare treatment arms from an architecture comparison manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_final_population(run_dir: Path) -> int | None:
    for line in reversed((run_dir / "run_summary.json").read_text(encoding="utf-8").splitlines() if (run_dir / "run_summary.json").exists() else []):
        pass
    stdout_path = run_dir / "run_summary.json"
    if stdout_path.exists():
        summary = json.loads(stdout_path.read_text(encoding="utf-8"))
        tail = summary.get("stdout_tail", "")
        for part in tail.splitlines():
            part = part.strip()
            if part.startswith("{") and "final_population" in part:
                try:
                    return json.loads(part).get("final_population")
                except json.JSONDecodeError:
                    continue
    traj = run_dir / "trajectory_class.json"
    if traj.exists():
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="arch_compare manifest.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_policy: dict[str, list[dict]] = defaultdict(list)

    for job in manifest.get("jobs", []):
        run_dir = Path(job["output_dir"])
        policy = job["policy"]
        pop = read_final_population(run_dir)
        traj_path = run_dir / "trajectory_class.json"
        traj = json.loads(traj_path.read_text()) if traj_path.exists() else {}
        by_policy[policy].append(
            {
                "replicate": job["replicate"],
                "seed": job["seed"],
                "final_population": pop,
                "trajectory_class": traj.get("trajectory_class"),
                "max_complexity": traj.get("max_complexity"),
            }
        )

    summary = {}
    for policy, rows in by_policy.items():
        pops = [r["final_population"] for r in rows if r["final_population"] is not None]
        summary[policy] = {
            "replicates": len(rows),
            "mean_final_population": sum(pops) / len(pops) if pops else None,
            "trajectory_classes": [r.get("trajectory_class") for r in rows],
            "runs": rows,
        }

    out = args.out or args.manifest.with_name("treatment_comparison.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
