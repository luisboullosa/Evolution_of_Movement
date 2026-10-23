#!/usr/bin/env python3
"""
Expanded 6x6 Botero Grid Parameter Sweep:
Tests the outer boundaries of evolutionary strategies across:
- R in [0.0002, 0.001, 0.005, 0.02, 0.10, 0.40] (3 orders of magnitude in temporal volatility)
- P in [0.01, 0.03, 0.06, 0.12, 0.25, 0.50] (spanning continental macro-patches to cell-level spatial noise)

Comparing two mechanisms:
1. Heterosperma Plasticity (Lean -> Motile, Dev=10)
2. Constitutive Motility (Dev=0)
Total: 2 mechanisms x 36 cells x 2 replicates = 144 jobs (100,000 steps each).
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_BIN = REPO_ROOT / "rust" / "target" / "release" / "evo_movement_rust.exe"
if not RUST_BIN.exists():
    RUST_BIN = REPO_ROOT / "rust" / "target" / "release" / "evo_movement_rust"

def run_single_job(job: dict) -> dict:
    job_dir = Path(job["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = job_dir / "config.json"

    config = {
        "grid_topology": "hex2d",
        "hex_dish_radius": 35,
        "hex_dish_torus": True,
        "architecture_policy": "evolve_mlp",
        "architecture_mutation_rate": 0.0,
        "nn_nodes": 2,
        "nn_layers": 1,
        "nn_inputs": 8,
        "radiate": 5,
        "initial_population": 350,
        "max_population": 1930,
        "steps": job["steps"],
        "seed": job["seed"],
        "env_perlin_time_scale": job["R"],
        "env_perlin_spatial_scale": job["P"],
        "allow_concurrent_eat_and_move": True,
        "movement_cost_factor": 0.10,
        "run_duration": 3,
        "flagella_maintenance_cost": 0.02,
        "developmental_steps": job["dev_steps"],
        "dispersal_on_low_cue": job["dispersal_on_low_cue"],
        "default_morph_threshold": 0.25,
        "write_outputs": False,
        "show_progress": False,
    }

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    cmd = [str(RUST_BIN), f"--config-file={cfg_path}"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    t1 = time.perf_counter()

    res = {
        "job_id": job["job_id"],
        "mechanism": job["mechanism"],
        "dev_steps": job["dev_steps"],
        "dispersal_on_low_cue": job["dispersal_on_low_cue"],
        "R": job["R"],
        "P": job["P"],
        "seed": job["seed"],
        "replicate": job["replicate"],
        "steps": job["steps"],
        "duration_sec": round(t1 - t0, 3),
        "returncode": proc.returncode,
        "final_population": 0,
        "prop_motile": 0.0,
        "prop_moved": 0.0,
        "avg_morph_threshold": 0.25,
        "std_morph_threshold": 0.0,
        "prop_neural_move": 0.0,
        "avg_neural_drive": 0.0,
    }

    if proc.returncode == 0 and proc.stdout:
        lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = json.loads(line)
                    res["final_population"] = payload.get("final_population", 0)
                    res["prop_motile"] = payload.get("prop_motile", 0.0)
                    res["prop_moved"] = payload.get("prop_moved", 0.0)
                    res["avg_morph_threshold"] = payload.get("avg_morph_threshold", 0.25)
                    res["std_morph_threshold"] = payload.get("std_morph_threshold", 0.0)
                    res["prop_neural_move"] = payload.get("prop_neural_move", 0.0)
                    res["avg_neural_drive"] = payload.get("avg_neural_drive", 0.0)
                    break
                except Exception:
                    continue

    summary_file = job_dir / "result.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    return res

def main():
    parser = argparse.ArgumentParser(description="Expanded 6x6 Botero Grid Sweep")
    parser.add_argument("--steps", type=int, default=100000, help="Simulation steps per run")
    parser.add_argument("--workers", type=int, default=8, help="Worker threads")
    parser.add_argument("--replicates", type=int, default=2, help="Replicates per cell")
    parser.add_argument("--outdir", type=str, default="output/botero_expanded_grid", help="Output directory")
    args = parser.parse_args()

    out_base = Path(args.outdir)
    out_base.mkdir(parents=True, exist_ok=True)
    jobs_base = out_base / "jobs"
    jobs_base.mkdir(parents=True, exist_ok=True)

    r_vals = [0.0002, 0.001, 0.005, 0.02, 0.10, 0.40]
    p_vals = [0.01, 0.03, 0.06, 0.12, 0.25, 0.50]

    mechanisms = [
        {
            "name": "heterosperma",
            "label": "Heterosperma Plasticity (Lean->Motile)",
            "dev_steps": 10,
            "dispersal_on_low_cue": True,
        },
        {
            "name": "constitutive",
            "label": "Constitutive Motility (Dev=0)",
            "dev_steps": 0,
            "dispersal_on_low_cue": False,
        },
    ]

    base_seed = 9000
    jobs = []
    job_idx = 0
    for mech in mechanisms:
        m_name = mech["name"]
        for r in r_vals:
            for p in p_vals:
                for rep in range(args.replicates):
                    seed = base_seed + job_idx * 137
                    job_id = f"{m_name}_R{r:.4f}_P{p:.4f}_rep{rep}"
                    job_dir = jobs_base / m_name / f"R_{r}_P_{p}" / f"rep_{rep}"
                    jobs.append({
                        "job_id": job_id,
                        "job_dir": str(job_dir),
                        "mechanism": mech["label"],
                        "mech_name": m_name,
                        "dev_steps": mech["dev_steps"],
                        "dispersal_on_low_cue": mech["dispersal_on_low_cue"],
                        "R": r,
                        "P": p,
                        "seed": seed,
                        "replicate": rep,
                        "steps": args.steps,
                    })
                    job_idx += 1

    total_jobs = len(jobs)
    print("=" * 80)
    print("     EXPANDED 6x6 BOTERO GRID SWEEP: STRATEGY FRONTIER EXPLORATION     ")
    print(f"Total jobs: {total_jobs} (2 Mechanisms x {len(r_vals)} R x {len(p_vals)} P x {args.replicates} Reps)")
    print(f"R range: {r_vals}")
    print(f"P range: {p_vals}")
    print(f"Simulation steps: {args.steps} | Arena: Expansive Torus (R=35, N=3781)")
    print(f"Output: {out_base.resolve()}")
    print("=" * 80 + "\n")

    results = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_job, job): job for job in jobs}
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            print(
                f"[{completed:03d}/{total_jobs:03d}] {res['job_id']:<36} | "
                f"Pop: {res['final_population']:>4} | "
                f"Motile: {res['prop_motile']*100:>5.1f}% | "
                f"Moved: {res['prop_moved']*100:>5.1f}% | "
                f"G_move: {res['prop_neural_move']*100:>5.1f}% | "
                f"theta: {res['avg_morph_threshold']:.3f}±{res['std_morph_threshold']:.3f} | "
                f"{res['duration_sec']}s"
            )

    t_end = time.perf_counter()
    print(f"\nAll {total_jobs} jobs completed in {t_end - t_start:.1f}s.")

    df = pd.DataFrame(results)
    csv_path = out_base / "botero_expanded_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Complete results saved to {csv_path}")

    # Also save separate CSVs
    for mech in mechanisms:
        sub = df[df["mechanism"] == mech["label"]]
        sub_file = out_base / f"{mech['name']}_results.csv"
        sub.to_csv(sub_file, index=False)
        print(f"Saved {mech['name']} results to {sub_file}")

if __name__ == "__main__":
    main()
