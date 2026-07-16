#!/usr/bin/env python3
"""Run architecture comparison experiments across treatment arms with replicates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

POLICIES = [
    "perceptron_only",
    "evolve_mlp",
    "evolve_spatial_graph",
    "evolve_goo",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_DIR = REPO_ROOT / "rust"


def build_cmd(
    policy: str,
    replicate: int,
    seed: int,
    steps: int,
    population: int,
    output_dir: Path,
    release: bool,
) -> list[str]:
    profile = "release" if release else "dev"
    cmd = [
        "cargo",
        "run",
        f"--{profile}",
        "--bin",
        "evo_movement_rust",
        "--",
        f"--architecture-policy={policy}",
        "--grid-topology=hex_2d",
        "--export-cladogram",
        "--write-snapshot",
        "--no-print-frames",
        f"--steps={steps}",
        f"--initial-population={population}",
        f"--seed={seed}",
        f"--output-run-dir={output_dir}",
        "--nn-nodes=1",
        "--nn-layers=1",
        "--nn-inputs=8",
    ]
    return cmd


def run_one(job: dict) -> dict:
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(
        job["policy"],
        job["replicate"],
        job["seed"],
        job["steps"],
        job["population"],
        output_dir,
        job["release"],
    )
    started = datetime.utcnow().isoformat()
    proc = subprocess.run(
        cmd,
        cwd=str(RUST_DIR),
        capture_output=True,
        text=True,
    )
    summary_path = output_dir / "run_summary.json"
    summary = {
        "policy": job["policy"],
        "replicate": job["replicate"],
        "seed": job["seed"],
        "started": started,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Architecture comparison runner")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-base", default="output/arch_compare")
    parser.add_argument("--release", action="store_true", default=True)
    parser.add_argument(
        "--policies",
        nargs="*",
        default=POLICIES,
        help="Subset of policies to run",
    )
    args = parser.parse_args()

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = REPO_ROOT / args.output_base / stamp
    base.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    for policy in args.policies:
        for rep in range(args.replicates):
            seed = args.base_seed + rep * 1000 + hash(policy) % 1000
            out = base / policy / f"rep{rep}_seed{seed}"
            jobs.append(
                {
                    "policy": policy,
                    "replicate": rep,
                    "seed": seed,
                    "steps": args.steps,
                    "population": args.population,
                    "output_dir": str(out),
                    "release": args.release,
                }
            )

    manifest = {"jobs": jobs, "created": stamp}
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                status = "ok" if res["returncode"] == 0 else "fail"
                print(f"[{status}] {job['policy']} rep={job['replicate']} seed={job['seed']}")
            except Exception as exc:
                print(f"[error] {job['policy']} rep={job['replicate']}: {exc}", file=sys.stderr)

    (base / "results_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Done. Output: {base}")
    return 0 if all(r.get("returncode") == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
