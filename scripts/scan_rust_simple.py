#!/usr/bin/env python3
"""
Simple parameter scan driver for the Rust simulator.
Creates per-job JSON config files and launches the Rust binary in parallel via worker_pool.

Usage:
  python scripts/scan_rust_simple.py --workers 4 --steps 2000 --initial-population 200

This script sweeps `movement_cost_factor` and `maintenance_cost_per_node` by default; edit ranges in the file as needed.
"""

import argparse
import itertools
import json
import os

from worker_pool import run_param_sweep

# detect Rust binary
RUST_BIN: str = os.path.normpath("rust/target/debug/evo_movement_rust.exe")
if not os.path.exists(RUST_BIN):
    RUST_BIN: str = os.path.normpath("rust/target/debug/evo_movement_rust")

if not os.path.exists(RUST_BIN):
    print("Rust binary not found at rust/target/debug/evo_movement_rust(.exe).")
    print("Please build it with: cd rust; cargo build")
    raise SystemExit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--initial-population", type=int, default=200)
parser.add_argument("--outdir", type=str, default="output/scan_rust_simple")
parser.add_argument("--pilot", action="store_true", help="Use a very small pilot grid")
args: argparse.Namespace = parser.parse_args()

# parameter ranges
if args.pilot:
    movement_vals: list[float] = [0.5, 1.0, 2.0]
    maintenance_vals: list[float] = [0.0015, 0.005]
else:
    movement_vals: list[float] = [0.5, 1.0, 2.0, 5.0]
    maintenance_vals: list[float] = [0.0005, 0.0015, 0.005, 0.01]

pairs = []
for mv, mc in itertools.product(movement_vals, maintenance_vals):
    pairs.append((mv, mc))

# prepare output dir and per-job configs
outdir = os.path.normpath(args.outdir)
os.makedirs(outdir, exist_ok=True)
params_csv: str = os.path.join(outdir, "params_grid.csv")
with open(params_csv, "w") as pf: os.TextIOWrapper[_WrappedBuffer]:
    pf.write("movement_cost_factor,maintenance_cost_per_node\n")
    for mv, mc in pairs:
        pf.write(f"{mv},{mc}\n")

run_dir: str = os.path.join(outdir, "jobs")
os.makedirs(run_dir, exist_ok=True)
run_cfg_paths = {}
for idx, (mv, mc) in enumerate(pairs):
    cfg = {
        "movement_cost_factor": mv,
        "maintenance_cost_per_node": mc,
        "steps": args.steps,
        "initial_population": args.initial_population,
        "show_progress": False,
        "write_frames": False,
        "webviewer_base": os.path.normpath(outdir),
        "sim_prefix": f"job{idx}_",
    }
    p: str = os.path.join(run_dir, f"job_{idx:04d}.json")
    with open(p, "w") as jf: os.TextIOWrapper[_WrappedBuffer]:
        json.dump(cfg, jf)
    run_cfg_paths[idx] = p

# build cmd for each job
EXE: str = RUST_BIN


def cmd_builder(job):
    idx = job[0]
    cfgpath = run_cfg_paths[idx]
    return [EXE, "--config", cfgpath]


# result row builder: parse extinction time from output using same parser logic as worker_pool
def result_row_builder(job, extinct, out_text):
    idx, (mv, mc) = job
    survived: int = 1 if extinct is None else 0
    ext_val = extinct if extinct is not None else args.steps
    # recorded_steps and movement metrics will be computed by downstream scripts; leave zeros now
    return [mv, mc, ext_val, survived]


header: list[str] = ["movement_cost_factor", "maintenance_cost_per_node", "extinction_time", "survived"]

pairs_idxed = list(enumerate(pairs))

print(f"Starting scan: {len(pairs_idxed)} jobs, workers={args.workers}")
run_param_sweep(
    pairs_idxed,
    cmd_builder,
    os.path.join(outdir, "results.csv"),
    header,
    workers=args.workers,
    timeout=3600,
    result_row_builder=result_row_builder,
)

print("Scan finished. Results in", outdir)
