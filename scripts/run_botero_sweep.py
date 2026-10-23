#!/usr/bin/env python3
"""
Orchestrates the 2D Botero parameter sweep across timescale (R) and spatial predictability (P)
for the Movement ALife simulation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_BIN = REPO_ROOT / "rust" / "target" / "release" / "evo_movement_rust.exe"
if not RUST_BIN.exists():
    RUST_BIN = REPO_ROOT / "rust" / "target" / "release" / "evo_movement_rust"


def run_single_job(job: dict) -> dict:
    job_dir = Path(job["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = job_dir / "config.json"

    # Assemble sim config
    nn_inputs = 8 if job["architecture"] in ("evolve_spatial_graph", "evolve_mlp") else 3
    nn_nodes = job.get("mlp_nodes", 2) if job["architecture"] == "evolve_mlp" else 1
    config = {
        "grid_topology": job["topology"],
        "hex_dish_radius": job["hex_dish_radius"],
        "hex_dish_torus": job.get("torus", False),
        "architecture_policy": job["architecture"],
        "architecture_mutation_rate": 0.0,
        "nn_nodes": nn_nodes,
        "nn_layers": 1,
        "nn_inputs": nn_inputs,
        "radiate": job.get("radiate", 0),
        "initial_population": job["initial_pop"],
        "max_population": job["max_pop"],
        "steps": job["steps"],
        "seed": job["seed"],
        "env_perlin_time_scale": job["R"],
        "env_perlin_spatial_scale": job["P"],
        "allow_concurrent_eat_and_move": True,
        "movement_cost_factor": job.get("movement_cost_factor", 0.10),
        "run_duration": job.get("run_duration", 0),
        "flagella_maintenance_cost": job["flagella_cost"],
        "developmental_steps": job["dev_steps"],
        "default_morph_threshold": 0.25,
        "write_outputs": False,
        "show_progress": False,
    }

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    cmd = [
        str(RUST_BIN),
        f"--config-file={cfg_path}",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    t1 = time.perf_counter()

    # Parse JSON output from last non-empty line of stdout
    res = {
        "job_id": job["job_id"],
        "architecture": job["architecture"],
        "topology": job["topology"],
        "torus": job.get("torus", False),
        "R": job["R"],
        "P": job["P"],
        "seed": job["seed"],
        "steps": job["steps"],
        "duration_sec": round(t1 - t0, 3),
        "returncode": proc.returncode,
        "final_population": 0,
        "prop_motile": 0.0,
        "prop_moved": 0.0,
        "avg_morph_threshold": 0.50,
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
                    res["avg_morph_threshold"] = payload.get("avg_morph_threshold", 0.50)
                    break
                except Exception:
                    continue

    summary_file = job_dir / "result.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    return res


def main():
    parser = argparse.ArgumentParser(description="Botero 2D Sweep Runner")
    parser.add_argument("--pilot", action="store_true", default=False, help="Run 3x3 pilot matrix")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads")
    parser.add_argument("--steps", type=int, default=100000, help="Timesteps per simulation")
    parser.add_argument("--replicates", type=int, default=3, help="Replicates per parameter cell")
    parser.add_argument("--outdir", type=str, default="output/botero_sweep", help="Output base directory")
    parser.add_argument("--dish-radius", type=int, default=25, help="Hex dish radius")
    parser.add_argument("--torus", action="store_true", default=False, help="Enable toroidal wrapping on hex dish")
    parser.add_argument("--pop-max", type=int, default=1000, help="Max population cap")
    parser.add_argument("--pop-init", type=int, default=None, help="Initial population (defaults to 200 for hex2d, 100 for ring1d)")
    parser.add_argument("--topologies", nargs="+", default=["hex2d"], help="Topologies to sweep: hex2d, ring1d")
    parser.add_argument("--architectures", nargs="+", default=["evolve_mlp"], help="Architectures to sweep: perceptron_only, evolve_mlp, evolve_spatial_graph")
    parser.add_argument("--mlp-nodes", type=int, default=2, help="Hidden nodes for MLP")
    parser.add_argument("--radiate", type=int, default=5, help="Founder weight radiation steps")
    parser.add_argument("--run-duration", type=int, default=3, help="Run-and-tumble persistence duration (steps)")
    parser.add_argument("--movement-cost-factor", type=float, default=0.10, help="Movement spend factor")
    args = parser.parse_args()

    if not RUST_BIN.exists():
        print(f"Error: Rust binary not found at {RUST_BIN}")
        print("Please build via: cargo build --release --bin evo_movement_rust")
        sys.exit(1)

    if args.pilot:
        r_vals = [0.001, 0.01, 0.10]
        p_vals = [0.03, 0.06, 0.10]
    else:
        r_vals = [0.001, 0.005, 0.02, 0.10]
        p_vals = [0.03, 0.06, 0.09, 0.12]

    architectures = args.architectures
    topologies = args.topologies
    base_seed = 42

    out_base = Path(args.outdir)
    out_base.mkdir(parents=True, exist_ok=True)
    jobs_base = out_base / "jobs"
    jobs_base.mkdir(parents=True, exist_ok=True)

    # Build job list
    jobs = []
    job_idx = 0
    for topo in topologies:
        dish_rad = args.dish_radius if topo == "hex2d" else 0
        init_pop = args.pop_init if args.pop_init is not None else (200 if topo == "hex2d" else 100)
        max_pop = args.pop_max if topo == "hex2d" else 500

        for arch in architectures:
            for r in r_vals:
                for p in p_vals:
                    for rep in range(args.replicates):
                        seed = base_seed + rep * 1000 + job_idx
                        torus_tag = "_torus" if (args.torus and topo == "hex2d") else ""
                        job_id = f"{topo}{torus_tag}_{arch}_R{r:.4f}_P{p:.4f}_rep{rep}"
                        job_dir = jobs_base / f"{topo}{torus_tag}" / f"{arch}" / f"R_{r}_P_{p}" / f"rep_{rep}"
                        jobs.append({
                            "job_id": job_id,
                            "job_dir": str(job_dir),
                            "architecture": arch,
                            "topology": topo,
                            "torus": args.torus if topo == "hex2d" else False,
                            "hex_dish_radius": dish_rad,
                            "initial_pop": init_pop,
                            "max_pop": max_pop,
                            "steps": args.steps,
                            "seed": seed,
                            "R": r,
                            "P": p,
                            "flagella_cost": 0.02,
                            "dev_steps": 10,
                            "mlp_nodes": args.mlp_nodes,
                            "radiate": args.radiate,
                            "movement_cost_factor": args.movement_cost_factor,
                            "run_duration": args.run_duration,
                        })
                        job_idx += 1

    total_jobs = len(jobs)
    print(f"=== Botero Parameter Sweep ===")
    print(f"Total jobs: {total_jobs} ({len(topologies)} Topo x {len(r_vals)} R x {len(p_vals)} P x {len(architectures)} Arch x {args.replicates} Reps)")
    print(f"Simulation steps: {args.steps} | Dish radius: {args.dish_radius} | Workers: {args.workers}")
    print(f"Output directory: {out_base.resolve()}\n")

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
                f"[{completed:02d}/{total_jobs:02d}] {res['job_id']:<38} | "
                f"Pop: {res['final_population']:>4} | "
                f"Motile: {res['prop_motile']*100:>5.1f}% | "
                f"Moved: {res['prop_moved']*100:>5.1f}% | "
                f"theta_morph: {res['avg_morph_threshold']:.3f} | "
                f"{res['duration_sec']}s"
            )

    t_end = time.perf_counter()
    total_duration = t_end - t_start
    print(f"\nAll jobs completed in {total_duration:.1f}s.")

    # Write aggregated CSV
    csv_path = out_base / "botero_sweep_results.csv"
    headers = [
        "job_id", "architecture", "topology", "torus", "R", "P", "seed", "steps",
        "final_population", "prop_motile", "prop_moved", "avg_morph_threshold", "duration_sec"
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in sorted(results, key=lambda x: (x["architecture"], x["R"], x["P"], x["seed"])):
            row = [
                str(r["job_id"]),
                str(r["architecture"]),
                str(r["topology"]),
                str(r.get("torus", False)),
                str(r["R"]),
                str(r["P"]),
                str(r["seed"]),
                str(r["steps"]),
                str(r["final_population"]),
                f"{r['prop_motile']:.4f}",
                f"{r['prop_moved']:.4f}",
                f"{r['avg_morph_threshold']:.4f}",
                str(r["duration_sec"]),
            ]
            f.write(",".join(row) + "\n")

    print(f"Results saved to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
