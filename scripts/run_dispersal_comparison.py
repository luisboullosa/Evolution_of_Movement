#!/usr/bin/env python3
"""
Compares three developmental mechanisms on the Expansive Torus (R=35, N=3781, MLP-2):
1. Heterosperma Dispersal Plasticity: lean patch -> motile, rich patch -> sessile
2. Constitutive Flagella (developmental_steps = 0): 100% flagellated, brain decides movement
3. Classic Dispersal Plasticity: rich patch -> motile, lean patch -> sessile
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
import numpy as np
import matplotlib.pyplot as plt

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
        "steps": job["steps"],
        "duration_sec": round(t1 - t0, 3),
        "returncode": proc.returncode,
        "final_population": 0,
        "prop_motile": 0.0,
        "prop_moved": 0.0,
        "avg_morph_threshold": 0.25,
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
                    break
                except Exception:
                    continue

    summary_file = job_dir / "result.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    return res

def main():
    parser = argparse.ArgumentParser(description="Dispersal Plasticity Comparison")
    parser.add_argument("--steps", type=int, default=100000, help="Steps per run")
    parser.add_argument("--workers", type=int, default=8, help="Worker threads")
    parser.add_argument("--replicates", type=int, default=3, help="Replicates per cell")
    parser.add_argument("--outdir", type=str, default="output/dispersal_mechanism_comparison", help="Output directory")
    args = parser.parse_args()

    out_base = Path(args.outdir)
    out_base.mkdir(parents=True, exist_ok=True)
    jobs_base = out_base / "jobs"
    jobs_base.mkdir(parents=True, exist_ok=True)

    # Regimes to test
    regimes = [
        ("Slow_Broad", 0.005, 0.03),
        ("Med_Med", 0.020, 0.06),
        ("Fast_Fine", 0.100, 0.12),
        ("Slow_Fine", 0.001, 0.12),
    ]

    mechanisms = [
        {
            "name": "Heterosperma (Lean->Motile)",
            "dev_steps": 10,
            "dispersal_on_low_cue": True,
        },
        {
            "name": "Constitutive (Dev=0)",
            "dev_steps": 0,
            "dispersal_on_low_cue": False,
        },
        {
            "name": "Classic Plasticity (Rich->Motile)",
            "dev_steps": 10,
            "dispersal_on_low_cue": False,
        },
    ]

    base_seed = 1000
    jobs = []
    job_idx = 0
    for mech in mechanisms:
        m_tag = mech["name"].split()[0].lower()
        for r_name, r_val, p_val in regimes:
            for rep in range(args.replicates):
                seed = base_seed + job_idx * 137
                job_id = f"{m_tag}_{r_name}_rep{rep}"
                job_dir = jobs_base / m_tag / r_name / f"rep_{rep}"
                jobs.append({
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "mechanism": mech["name"],
                    "dev_steps": mech["dev_steps"],
                    "dispersal_on_low_cue": mech["dispersal_on_low_cue"],
                    "R": r_val,
                    "P": p_val,
                    "regime_name": r_name,
                    "seed": seed,
                    "steps": args.steps,
                })
                job_idx += 1

    total_jobs = len(jobs)
    print("=" * 80)
    print("       DEVELOPMENTAL DISPERSAL PLASTICITY COMPARISON (TORUS, MLP-2)")
    print(f"Total jobs: {total_jobs} (3 Mechanisms x {len(regimes)} Regimes x {args.replicates} Reps)")
    print(f"Simulation steps: {args.steps} | Arena: Expansive Torus (R=35, 3781 cells)")
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
                f"[{completed:02d}/{total_jobs:02d}] {res['job_id']:<32} | "
                f"Pop: {res['final_population']:>4} | "
                f"Motile: {res['prop_motile']*100:>5.1f}% | "
                f"Moved: {res['prop_moved']*100:>5.1f}% | "
                f"theta: {res['avg_morph_threshold']:.3f} | "
                f"{res['duration_sec']}s"
            )

    t_end = time.perf_counter()
    print(f"\nAll jobs completed in {t_end - t_start:.1f}s.")

    df = pd.DataFrame(results)
    csv_path = out_base / "dispersal_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Summary analysis
    print("\n" + "=" * 80)
    print("                      AGGREGATED COMPARISON TABLE                     ")
    print("=" * 80)
    summary = df.groupby(["mechanism"])[["prop_motile", "prop_moved", "final_population", "avg_morph_threshold"]].agg(["mean", "std"])
    print((summary * 100).round(2).to_string())

if __name__ == "__main__":
    main()
