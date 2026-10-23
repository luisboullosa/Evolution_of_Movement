#!/usr/bin/env python3
"""
Runs a 3-factor sweep to isolate the effects of:
1. Boundary Topology: Open Hex Dish (with wall effects) vs Toroidal Hex Dish (periodic wrapping).
2. Arena Size: Compact (R=15, 721 cells), Standard (R=25, 1,951 cells), Expansive (R=35, 3,781 cells).
3. Competition Density: Sparse (rho=0.20), Standard (rho=0.51), Crowded (rho=0.85).

Tuned for the peak volatility regime: R = 0.02, P = 0.06 using 2-Node Run-and-Tumble MLP.
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUST_BIN = REPO_ROOT / "rust" / "target" / "release" / "evo_movement_rust.exe"
if not RUST_BIN.exists():
    RUST_BIN = REPO_ROOT / "rust" / "target" / "release" / "evo_movement_rust"

def hex_cells_for_radius(r: int) -> int:
    return 1 + 3 * r * (r + 1)

def run_single_job(job: dict) -> dict:
    job_dir = Path(job["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = job_dir / "config.json"

    config = {
        "grid_topology": "hex2d",
        "hex_dish_radius": job["dish_radius"],
        "hex_dish_torus": job["torus"],
        "architecture_policy": "evolve_mlp",
        "architecture_mutation_rate": 0.0,
        "nn_nodes": 2,
        "nn_layers": 1,
        "nn_inputs": 8,
        "radiate": 5,
        "initial_population": job["initial_pop"],
        "max_population": job["max_pop"],
        "steps": job["steps"],
        "seed": job["seed"],
        "env_perlin_time_scale": 0.02,
        "env_perlin_spatial_scale": 0.06,
        "allow_concurrent_eat_and_move": True,
        "movement_cost_factor": 0.10,
        "run_duration": 3,
        "flagella_maintenance_cost": 0.02,
        "developmental_steps": 10,
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
        "boundary": "Torus" if job["torus"] else "Open",
        "torus": job["torus"],
        "dish_radius": job["dish_radius"],
        "arena_cells": job["cells"],
        "density_regime": job["density_regime"],
        "target_density": job["target_density"],
        "initial_pop": job["initial_pop"],
        "max_pop": job["max_pop"],
        "seed": job["seed"],
        "replicate": job["replicate"],
        "duration_sec": round(t1 - t0, 3),
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
    parser = argparse.ArgumentParser(description="Competition and Boundary Sweep")
    parser.add_argument("--workers", type=int, default=8, help="Worker threads")
    parser.add_argument("--steps", type=int, default=50000, help="Timesteps per simulation")
    parser.add_argument("--replicates", type=int, default=2, help="Replicates per condition")
    parser.add_argument("--outdir", type=str, default="output/ecology_competition_sweep", help="Output directory")
    args = parser.parse_args()

    out_base = Path(args.outdir)
    out_base.mkdir(parents=True, exist_ok=True)
    jobs_base = out_base / "jobs"
    jobs_base.mkdir(parents=True, exist_ok=True)

    dish_radii = [15, 25, 35]
    boundary_options = [False, True] # Open vs Torus
    density_levels = [
        ("Sparse", 0.20),
        ("Standard", 0.51),
        ("Crowded", 0.85),
    ]

    base_seed = 777
    jobs = []
    job_idx = 0

    for r in dish_radii:
        cells = hex_cells_for_radius(r)
        for torus in boundary_options:
            b_name = "Torus" if torus else "Open"
            for d_name, rho in density_levels:
                max_pop = max(50, int(round(cells * rho)))
                init_pop = max(20, int(round(max_pop * 0.25)))
                for rep in range(args.replicates):
                    seed = base_seed + rep * 1000 + job_idx
                    job_id = f"R{r}_{b_name}_{d_name}_rep{rep}"
                    job_dir = jobs_base / f"R{r}" / b_name / d_name / f"rep_{rep}"
                    jobs.append({
                        "job_id": job_id,
                        "job_dir": str(job_dir),
                        "dish_radius": r,
                        "cells": cells,
                        "torus": torus,
                        "density_regime": d_name,
                        "target_density": rho,
                        "initial_pop": init_pop,
                        "max_pop": max_pop,
                        "steps": args.steps,
                        "seed": seed,
                        "replicate": rep,
                    })
                    job_idx += 1

    total_jobs = len(jobs)
    print(f"=== Competition & Boundary Topology Sweep ===")
    print(f"Total jobs: {total_jobs} (3 Sizes x 2 Boundaries x 3 Densities x {args.replicates} Replicates)")
    print(f"Steps: {args.steps} | Output: {out_base.resolve()}\n")

    t_start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, res in enumerate(executor.map(run_single_job, jobs), 1):
            results.append(res)
            print(
                f"[{i:02d}/{total_jobs:02d}] {res['job_id']:<26} | "
                f"Pop: {res['final_population']:>4} | "
                f"Motile: {res['prop_motile']*100:>5.1f}% | "
                f"Moved: {res['prop_moved']*100:>5.1f}% | "
                f"theta: {res['avg_morph_threshold']:.3f} | {res['duration_sec']}s"
            )

    t_total = time.perf_counter() - t_start
    print(f"\nAll jobs completed in {t_total:.1f}s.")

    df = pd.DataFrame(results)
    csv_path = out_base / "ecology_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()
