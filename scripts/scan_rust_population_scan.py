#!/usr/bin/env python3
"""
Population-stability scan driver using the Rust simulator.
Performs a 2D sweep over `movement_cost_factor` and `maintenance_cost_per_node`,
runs short sims, records final population size, then runs deeper sims for promising
parameter sets (final_pop >= threshold).

Usage:
  python scripts/scan_rust_population_scan.py --workers 4 --steps 2000 --initial-population 500
"""
import os
import json
import argparse
import itertools
import time
import random
from worker_pool import run_param_sweep

# detect Rust binary
RUST_BIN = os.path.normpath("rust/target/debug/evo_movement_rust.exe")
if not os.path.exists(RUST_BIN):
    RUST_BIN = os.path.normpath("rust/target/debug/evo_movement_rust")

if not os.path.exists(RUST_BIN):
    print("Rust binary not found at rust/target/debug/evo_movement_rust(.exe). Build it first: cd rust; cargo build")
    raise SystemExit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--initial-population", type=int, default=500)
parser.add_argument("--outdir", type=str, default="output/scan_population")
parser.add_argument("--pilot", action="store_true")
parser.add_argument("--deep-steps", type=int, default=10000)
parser.add_argument("--deep-population", type=int, default=2000)
parser.add_argument("--threshold", type=int, default=1000, help="Final population threshold to consider promising")
parser.add_argument("--sample", type=int, default=0, help="If >0, randomly sample this many parameter combinations instead of full grid")
parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling mode")
args = parser.parse_args()

# expanded parameter grid: movement, maintenance, reproduction threshold/cost, env time speed, death risk
# also include grid size (`grid_width`) and zero-values where allowed
if args.pilot:
    # moderately larger pilot to sample broader behavior without exploding job count
    movement_vals = [0.0, 0.5, 1.0, 2.0, 5.0]
    maintenance_vals = [0.0, 0.0005, 0.0015, 0.005]
    repro_thresholds = [0.0, 10.0, 20.0]
    repro_costs = [0.0, 0.5, 1.0, 5.0]
    env_time_speeds = [0.0, 0.0001, 0.001]
    death_risks = [0.0, 1e-06, 1e-05]
    grid_vals = [50, 100]
else:
    # expanded full-grid ranges (much larger search space)
    movement_vals = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    maintenance_vals = [0.0, 1e-05, 5e-05, 1e-04, 5e-04, 0.001, 0.005, 0.01]
    repro_thresholds = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    repro_costs = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
    env_time_speeds = [0.0, 0.0001, 0.001, 0.01, 0.1]
    death_risks = [0.0, 1e-08, 1e-06, 1e-04, 1e-03]
    grid_vals = [25, 50, 100, 200]

# build a manageable set of combinations (cartesian product across dims)
# build product or sampled set
pairs = None
if args.sample and args.sample > 0:
    # sampling mode: draw values uniformly across the min/max of each list
    if args.seed is not None:
        random.seed(args.seed)
    def sample_range(values, log=False):
        lo = min(values)
        hi = max(values)
        if log:
            # sample log-uniform; avoid math domain errors when zeros or negatives are present
            pos = [v for v in values if v > 0]
            if pos:
                lo_p = min(pos)
                hi_p = max(pos)
                return 10 ** random.uniform(math.log10(lo_p), math.log10(hi_p))
            else:
                # no positive values to do log-sampling on; fallback to uniform
                return random.uniform(lo, hi)
        else:
            return random.uniform(lo, hi)

    import math
    sampled = []
    for i in range(args.sample):
        mv = sample_range(movement_vals, log=False)
        mc = sample_range(maintenance_vals, log=True)
        repro_thr = sample_range(repro_thresholds, log=False)
        repro_cost = sample_range(repro_costs, log=False)
        env_time_speed = sample_range(env_time_speeds, log=False)
        death_risk = sample_range(death_risks, log=True)
        # sample grid width discretely from grid_vals
        gw = random.choice(grid_vals)
        sampled.append((mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, gw))
    pairs = sampled
else:
    pairs = list(itertools.product(movement_vals, maintenance_vals, repro_thresholds, repro_costs, env_time_speeds, death_risks, grid_vals))

outdir = os.path.normpath(args.outdir)
os.makedirs(outdir, exist_ok=True)
jobs_dir = os.path.join(outdir, "jobs")
os.makedirs(jobs_dir, exist_ok=True)

run_cfg_paths = {}
for idx, tpl in enumerate(pairs):
    # tpl is now (mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, grid_width)
    mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, grid_width = tpl
    cfg = {
        "movement_cost_factor": mv,
        "maintenance_cost_per_node": mc,
        "reproduction_threshold": repro_thr,
        "reproduction_cost": repro_cost,
        "env_time_speed": env_time_speed,
        "death_risk": death_risk,
        "grid_width": grid_width,
        "steps": args.steps,
        "initial_population": args.initial_population,
        "show_progress": False,
        "write_frames": False,
        "webviewer_base": os.path.normpath(outdir),
        "sim_prefix": f"job{idx}_",
    }
    p = os.path.join(jobs_dir, f"job_{idx:04d}.json")
    with open(p, "w") as jf:
        json.dump(cfg, jf)
    run_cfg_paths[idx] = p

def cmd_builder(job):
    idx = job[0]
    cfgpath = run_cfg_paths[idx]
    return [RUST_BIN, "--config", cfgpath]

def find_sim_folder_for_prefix(base_dir, prefix):
    # look for directories in base_dir that start with prefix (simNNNN or job prefix)
    cand = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]
    if not cand:
        # fallback: any directory containing individuals.jsonl
        cand = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and os.path.exists(os.path.join(base_dir, d, 'individuals.jsonl'))]
    if not cand:
        return None
    # pick most recently modified
    cand_paths = [os.path.join(base_dir, d) for d in cand]
    cand_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand_paths[0]

def read_final_population(sim_path):
    jf = os.path.join(sim_path, 'individuals.jsonl')
    if not os.path.exists(jf):
        return 0
    last = None
    with open(jf, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            last = line
    if last is None:
        return 0
    try:
        obj = json.loads(last)
        inds = obj.get('individuals', [])
        return len(inds)
    except Exception:
        return 0

def result_row_builder(job, extinct, out_text):
    idx, tpl = job
    mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, grid_width = tpl
    prefix = f"job{idx}_"
    simf = find_sim_folder_for_prefix(outdir, prefix)
    final_pop = 0
    if simf:
        final_pop = read_final_population(simf)
    return [mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, grid_width, final_pop]

header = ["movement_cost_factor", "maintenance_cost_per_node", "reproduction_threshold", "reproduction_cost", "env_time_speed", "death_risk", "grid_width", "final_population"]
pairs_idxed = list(enumerate(pairs))

results_csv = os.path.join(outdir, 'results_short.csv')
print(f"Running short scan ({len(pairs)} jobs) -> {results_csv}")
run_param_sweep(pairs_idxed, cmd_builder, results_csv, header, workers=args.workers, timeout=1800, result_row_builder=result_row_builder)

# analyze results and pick promising candidates
promising = []
with open(results_csv, 'r', encoding='utf-8') as rf:
    import csv
    r = csv.DictReader(rf)
    for row in r:
        try:
            mv = float(row['movement_cost_factor'])
            mc = float(row['maintenance_cost_per_node'])
            repro_thr = float(row.get('reproduction_threshold', 0))
            repro_cost = float(row.get('reproduction_cost', 0))
            env_time_speed = float(row.get('env_time_speed', 0))
            death_risk = float(row.get('death_risk', 0))
            # grid_width may be present
            gw = int(row.get('grid_width', 0))
            fp = int(row['final_population'])
        except Exception:
            continue
        if fp >= args.threshold:
            promising.append((mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, gw, fp))

print(f"Found {len(promising)} promising parameter sets (final_pop >= {args.threshold})")
prom_csv = os.path.join(outdir, 'promising.csv')
with open(prom_csv, 'w', encoding='utf-8') as pf:
    pf.write('movement_cost_factor,maintenance_cost_per_node,grid_width,final_population\n')
    for mv, mc, gw, fp in promising:
        pf.write(f"{mv},{mc},{gw},{fp}\n")

if promising:
    # run deeper sims for promising candidates
    deep_jobs = []
    deep_run_dir = os.path.join(outdir, 'deep')
    os.makedirs(deep_run_dir, exist_ok=True)
    deep_cfg_paths = {}
    for i, entry in enumerate(promising):
        mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, gw, fp = entry
        cfg = {
            'movement_cost_factor': mv,
            'maintenance_cost_per_node': mc,
            'reproduction_threshold': repro_thr,
            'reproduction_cost': repro_cost,
            'env_time_speed': env_time_speed,
            'death_risk': death_risk,
            'grid_width': gw,
            'steps': args.deep_steps,
            'initial_population': args.deep_population,
            'show_progress': False,
            'write_frames': True,
            'webviewer_base': os.path.normpath(deep_run_dir),
            'sim_prefix': f'prom{i}_',
        }
        p = os.path.join(deep_run_dir, f'prom_{i:03d}.json')
        with open(p, 'w') as jf:
            json.dump(cfg, jf)
        deep_cfg_paths[i] = p
        deep_jobs.append((i, (mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, gw)))

    def deep_cmd(job):
        idx = job[0]
        return [RUST_BIN, '--config', deep_cfg_paths[idx]]

    deep_results_csv = os.path.join(outdir, 'deep_results.csv')
    print(f"Running {len(deep_jobs)} deep sims -> {deep_results_csv}")
    # reuse simple result builder that reads final population from deep_run_dir
    def deep_result_builder(job, extinct, out_text):
        idx, tpl = job
        mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, gw = tpl
        prefix = f'prom{idx}_'
        simf = find_sim_folder_for_prefix(deep_run_dir, prefix)
        final_pop = 0
        if simf:
            final_pop = read_final_population(simf)
        return [mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, gw, final_pop]

    # enumerate deep job parameter tuples
    deep_pairs_idxed = list(enumerate([(mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, gw) for (mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, gw, _) in promising]))
    run_param_sweep(deep_pairs_idxed, deep_cmd, deep_results_csv, header, workers=max(1, args.workers//2), timeout=7200, result_row_builder=deep_result_builder)
    print('Deep runs finished')

print('All done')
