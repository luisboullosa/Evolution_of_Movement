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
import tempfile

# detect Rust binary (prefer release build if present)
RUST_BIN = os.path.normpath("rust/target/release/evo_movement_rust.exe")
if not os.path.exists(RUST_BIN):
    RUST_BIN = os.path.normpath("rust/target/release/evo_movement_rust")
if not os.path.exists(RUST_BIN):
    RUST_BIN = os.path.normpath("rust/target/debug/evo_movement_rust.exe")
if not os.path.exists(RUST_BIN):
    RUST_BIN = os.path.normpath("rust/target/debug/evo_movement_rust")

if not os.path.exists(RUST_BIN):
    print("Rust binary not found. Build it first: cd rust; cargo build or cargo build --release")
    raise SystemExit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=6)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--initial-population", type=int, default=500)
parser.add_argument("--outdir", type=str, default="output/scan_population")
parser.add_argument("--pilot", action="store_true")
parser.add_argument("--deep-steps", type=int, default=10000)
parser.add_argument("--deep-population", type=int, default=10000)
parser.add_argument("--threshold", type=int, default=200, help="Final population threshold to consider promising")
parser.add_argument("--no-deep", action="store_true", help="Do not run deeper sims for promising candidates")
parser.add_argument("--sample", type=int, default=0, help="If >0, randomly sample this many parameter combinations instead of full grid")
parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling mode")
parser.add_argument("--max-population", type=int, default=1000, help="If set, override job max_population for short pilots")
parser.add_argument("--no-write-outputs", action="store_true", help="Pass to simulator to suppress per-step outputs (env.jsonl, individuals.jsonl, frames, nn_index)")
args = parser.parse_args()

# expanded parameter grid: movement, maintenance, reproduction threshold/cost, env time speed, death risk
# also include grid size (`grid_width`) and zero-values where allowed
if args.pilot:
    # moderately larger pilot to sample broader behavior without exploding job count
    movement_vals = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    maintenance_vals = [0.0, 1e-05, 5e-05, 1e-04, 5e-04, 0.001, 0.005, 0.01]
    repro_thresholds = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    repro_costs = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
    env_time_speeds = [0.0, 0.0001, 0.001, 0.01, 0.1]
    death_risks = [0.0, 1e-08, 1e-06, 1e-04, 1e-03]
    grid_vals = [25, 50, 100, 200]
    resource_scales = [10.0, 100.0]
else:
    # expanded full-grid ranges (much larger search space)
    movement_vals = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    maintenance_vals = [0.0, 1e-05, 5e-05, 1e-04, 5e-04, 0.001, 0.005, 0.01]
    repro_thresholds = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    repro_costs = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
    env_time_speeds = [0.0, 0.0001, 0.001, 0.01, 0.1]
    death_risks = [0.0, 1e-08, 1e-06, 1e-04, 1e-03]
    grid_vals = [25, 50, 100, 200]
    resource_scales = [10.0, 100.0]

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
        rscales = random.choice(resource_scales)
        sampled.append((mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, gw))
    pairs = sampled
else:
    pairs = list(itertools.product(movement_vals, maintenance_vals, repro_thresholds, repro_costs, env_time_speeds, death_risks, resource_scales, grid_vals))

outdir = os.path.normpath(args.outdir)
os.makedirs(os.path.dirname(outdir) or '.', exist_ok=True)
# create a single timestamped jobs folder to hold only the three summary files
jobs_dir = os.path.join(outdir, f"jobs{int(time.time())}")
os.makedirs(jobs_dir, exist_ok=True)

# create temp config files (in system temp dir) so the jobs folder doesn't
# accumulate per-job JSON files; these temp files will be removed later.
run_cfg_paths = {}
for idx, tpl in enumerate(pairs):
    mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, grid_width = tpl
    cfg = {
        "movement_cost_factor": mv,
        "maintenance_cost_per_node": mc,
        "reproduction_threshold": repro_thr,
        "reproduction_cost": repro_cost,
        "env_time_speed": env_time_speed,
        "env_time_amplitude": 1.0,
        "env_perlin_time_scale": 0.00001,
        "resource_scale": rscales,
        "death_risk": death_risk,
        "grid_width": grid_width,
        "steps": args.steps,
        "initial_population": args.initial_population,
        # disable interactive/per-step outputs
        "show_progress": False,
        "write_frames": False,
        "write_outputs": False,
        # set report interval larger than steps to avoid intermediate writes
        "report_interval": int(args.steps) + 1,
        "webviewer_base": os.path.normpath(jobs_dir),
        "sim_prefix": f"job{idx}_",
    }
    # For pilot runs, introduce phenotypic variation via ratiate multiplier
    if args.pilot:
        cfg['ratiate'] = 10
    tf = tempfile.NamedTemporaryFile(prefix=f"scan_job_{idx}_", suffix=".json", delete=False, mode="w", encoding="utf-8")
    json.dump(cfg, tf)
    tf.flush()
    tf.close()
    run_cfg_paths[idx] = tf.name

def cmd_builder(job):
    idx = job[0]
    cfgpath = run_cfg_paths[idx]
    cmd = [RUST_BIN, "--config-file", cfgpath]
    if args.no_write_outputs:
        cmd.append("--no-write-outputs")
    return cmd

# progress writer: append results to a JSON file for the web visualizer
def make_progress_writer(outdir):
    import threading
    import json

    lock = threading.Lock()
    entries = []
    prog_path = os.path.join(outdir, 'progress.json')
    last_write = 0.0
    write_interval = 30.0  # seconds between writes to reduce churn

    def atomic_write(path, data):
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(path)
                os.replace(tmp, path)
            except Exception:
                pass

    def writer(completed, total, row, job):
        nonlocal last_write
        # row expected to be list containing mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, resource_scale, grid_width, final_population, prop_moved
        mv = mc = repro_thr = repro_cost = env_time_speed = death_risk = resource_scale = grid_width = None
        final_pop = prop_moved = None
        try:
            if isinstance(row, (list, tuple)):
                # try index mapping when available
                if len(row) >= 10:
                    mv = float(row[0])
                    mc = float(row[1])
                    repro_thr = float(row[2])
                    repro_cost = float(row[3])
                    env_time_speed = float(row[4])
                    death_risk = float(row[5])
                    resource_scale = float(row[6])
                    grid_width = int(row[7])
                    final_pop = int(row[8])
                    prop_moved = float(row[9])
                else:
                    # unexpected short row: raise an informative exception so callers see diagnostics
                    raise ValueError(
                        f"Progress writer: unexpected short result row (length {len(row)}). "
                        f"Expected a list/tuple with 10 elements: [movement_cost_factor, maintenance_cost_per_node, "
                        f"reproduction_threshold, reproduction_cost, env_time_speed, death_risk, resource_scale, grid_width, "
                        f"final_population, prop_moved]. Row: {row!r}"
                    )
            elif isinstance(row, dict):
                mv = float(row.get('movement_cost_factor') or 0)
                mc = float(row.get('maintenance_cost_per_node') or 0)
                repro_thr = float(row.get('reproduction_threshold') or 0)
                repro_cost = float(row.get('reproduction_cost') or 0)
                env_time_speed = float(row.get('env_time_speed') or 0)
                death_risk = float(row.get('death_risk') or 0)
                resource_scale = float(row.get('resource_scale') or 0)
                grid_width = int(row.get('grid_width') or 0)
                final_pop = int(row.get('final_population') or 0)
                prop_moved = float(row.get('prop_moved') or 0)
        except Exception:
            mv = mc = repro_thr = repro_cost = env_time_speed = death_risk = resource_scale = grid_width = None
            final_pop = prop_moved = None

        # normalize job params to a dict mapping names -> values so the webviewer shows named axes
        param_names = ['movement_cost_factor', 'maintenance_cost_per_node', 'reproduction_threshold', 'reproduction_cost', 'env_time_speed', 'death_risk', 'resource_scale', 'grid_width']
        params = None
        try:
            if isinstance(job, (list, tuple)):
                tpl = job[1]
                if isinstance(tpl, (list, tuple)):
                    params = {name: tpl[i] for i, name in enumerate(param_names) if i < len(tpl)}
                elif isinstance(tpl, dict):
                    params = tpl
                else:
                    params = {'value': tpl}
            elif isinstance(job, dict):
                params = job
            else:
                params = {'value': job}
        except Exception:
            params = job[1] if isinstance(job, (list, tuple)) else job

        entry = {
            'completed': completed,
            'total': total,
            'movement_cost_factor': mv,
            'maintenance_cost_per_node': mc,
            'reproduction_threshold': repro_thr,
            'reproduction_cost': repro_cost,
            'env_time_speed': env_time_speed,
            'death_risk': death_risk,
            'resource_scale': resource_scale,
            'grid_width': grid_width,
            'final_population': final_pop,
            'prop_moved': prop_moved,
            'job': job[0] if isinstance(job, (list, tuple)) else None,
            'params': params,
        }
        now = time.time()
        do_write = False
        with lock:
            entries.append(entry)
            # Prefer to wait for the simulator-emitted summary metrics before
            # writing the very first progress.json entry to avoid placeholder
            # zeros when stdout parsing is still pending. If metrics are
            # present (non-zero) write immediately; otherwise fall back to
            # normal write_interval behavior or final completion.
            if len(entries) == 1:
                # if metrics present, write immediately
                if entry.get('final_population') and (entry.get('final_population') != 0 or entry.get('prop_moved')):
                    do_write = True
                    last_write = now
                else:
                    # delay first write until either interval passes or more entries accumulate
                    do_write = False
            # otherwise write if enough time passed or we're finished
            elif (now - last_write) >= write_interval or completed >= total:
                do_write = True
                last_write = now
        if do_write:
            try:
                atomic_write(prog_path, entries)
            except Exception:
                pass

    return writer

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
        return 0, 0.0
    last = None
    with open(jf, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            last = line
    if last is None:
        return 0, 0.0
    try:
        obj = json.loads(last)
        inds = obj.get('individuals', [])
        # compute proportion moved from last_delta if present
        moved = 0
        total = 0
        for ind in inds:
            total += 1
            try:
                if int(ind.get('last_delta', 0)) != 0:
                    moved += 1
            except Exception:
                pass
        prop = (moved / total) if total > 0 else 0.0
        return len(inds), prop
    except Exception:
        return 0, 0.0

def result_row_builder(job, extinct, out_text):
    idx, tpl = job
    # Expect the job parameter tuple to be the exact shape generated by
    # the driver: (mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, resource_scale, grid_width)
    # Fail fast and raise informative errors if the shape or keys are wrong so callers see useful diagnostics.
    if isinstance(tpl, (list, tuple)):
        if len(tpl) != 8:
            raise ValueError(f"job {idx}: expected parameter tuple length 8, got {len(tpl)}: {tpl!r}")
        mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, resource_scale, grid_width = tpl
    elif isinstance(tpl, dict):
        required = ['movement_cost_factor', 'maintenance_cost_per_node', 'reproduction_threshold', 'reproduction_cost', 'env_time_speed', 'death_risk', 'resource_scale', 'grid_width']
        missing = [k for k in required if k not in tpl]
        if missing:
            raise KeyError(f"job {idx}: missing required parameter keys: {missing}")
        mv = tpl['movement_cost_factor']
        mc = tpl['maintenance_cost_per_node']
        repro_thr = tpl['reproduction_threshold']
        repro_cost = tpl['reproduction_cost']
        env_time_speed = tpl['env_time_speed']
        death_risk = tpl['death_risk']
        resource_scale = tpl['resource_scale']
        grid_width = tpl['grid_width']
    else:
        raise TypeError(f"job {idx}: unsupported parameter tuple type: {type(tpl).__name__}: {tpl!r}")
    prefix = f"job{idx}_"
    simf = find_sim_folder_for_prefix(jobs_dir, prefix)
    final_pop = 0
    prop_moved = 0.0
    if simf:
        final_pop, prop_moved = read_final_population(simf)
    else:
        # If simulator didn't write per-sim files (write_outputs=False), the
        # simulator now emits a compact JSON summary to stdout. Try to parse
        # that from the captured `out_text` and use it as the final metrics.
        try:
            if out_text:
                for line in (out_text or "").splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    # Heuristic: JSON object containing final_population
                    if s.startswith('{') and 'final_population' in s:
                        try:
                            obj = json.loads(s)
                            if 'final_population' in obj:
                                final_pop = int(obj.get('final_population') or 0)
                            if 'prop_moved' in obj:
                                prop_moved = float(obj.get('prop_moved') or 0.0)
                            break
                        except Exception:
                            continue
        except Exception:
            pass
    return [mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, resource_scale, grid_width, final_pop, prop_moved]

# header for short results
# include resource_scale so progress.json exposes it as an axis
header = [
    'movement_cost_factor',
    'maintenance_cost_per_node',
    'reproduction_threshold',
    'reproduction_cost',
    'env_time_speed',
    'death_risk',
    'resource_scale',
    'grid_width',
    'final_population',
    'prop_moved',
]

# put summary files into the single jobs folder
results_csv = os.path.join(jobs_dir, 'results_short.csv')
print(f"Running short scan ({len(pairs)} jobs) -> {results_csv}")

# ensure temp config files enforce small max_population and no outputs
for p in list(run_cfg_paths.values()):
    try:
        j = json.load(open(p, 'r', encoding='utf-8'))
        if args.max_population is not None:
            j['max_population'] = int(args.max_population)
        else:
            j['max_population'] = 2000
        try:
            if 'initial_population' in j and j.get('max_population', 0) > 0:
                j['initial_population'] = min(int(j.get('initial_population', 0)), int(j.get('max_population', 0)))
        except Exception:
            pass
        j['write_outputs'] = False
        j['write_frames'] = False
        j['report_interval'] = int(j.get('steps', args.steps)) + 1
        j['webviewer_base'] = os.path.normpath(jobs_dir)
        json.dump(j, open(p, 'w', encoding='utf-8'))
    except Exception:
        pass
# disable local matplotlib visualizer (use webviewer + progress.json instead)
heat_cb = None
# create a progress writer so the web viewer can consume live updates
prog_writer = make_progress_writer(jobs_dir)

def _combined_progress(completed, total, row, job):
    try:
        if heat_cb:
            heat_cb(completed, total, row, job)
    except Exception:
        pass
    try:
        if prog_writer:
            prog_writer(completed, total, row, job)
    except Exception:
        pass

# run short sweep
run_param_sweep(list(pairs), cmd_builder, results_csv, header, workers=args.workers, timeout=1800, result_row_builder=result_row_builder, progress_callback=_combined_progress, stream_live_output=True)

# remove temporary config files created in the system temp dir
for p in list(run_cfg_paths.values()):
    try:
        os.remove(p)
    except Exception:
        pass

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
            rscales = float(row.get('resource_scale', 10.0))
            gw = int(row.get('grid_width', 0))
            # grid_width may be present
            gw = int(row.get('grid_width', 0))
            fp = int(row['final_population'])
        except Exception:
            continue
        if fp >= args.threshold:
            promising.append((mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, gw, fp))

print(f"Found {len(promising)} promising parameter sets (final_pop >= {args.threshold})")
prom_csv = os.path.join(jobs_dir, 'promising.csv')
with open(prom_csv, 'w', encoding='utf-8') as pf:
    pf.write('movement_cost_factor,maintenance_cost_per_node,resource_scale,grid_width,final_population\n')
    for entry in promising:
        mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, gw, fp = entry
        pf.write(f"{mv},{mc},{rscales},{gw},{fp}\n")

if promising:
    if args.no_deep:
        print(f"Found {len(promising)} promising parameter sets but --no-deep specified; skipping deep sims")
    else:
        # run deeper sims for promising candidates; keep outputs in the single jobs folder
        deep_jobs = []
        deep_run_dir = jobs_dir
        deep_results_csv = os.path.join(jobs_dir, 'deep_results.csv')
        print(f"Running {len(deep_jobs)} deep sims -> {deep_results_csv}")
        deep_cfg_paths = {}
        for i, entry in enumerate(promising):
            mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, gw, fp = entry
        cfg = {
            'movement_cost_factor': mv,
            'maintenance_cost_per_node': mc,
            'reproduction_threshold': repro_thr,
            'reproduction_cost': repro_cost,
            'env_time_speed': env_time_speed,
            'death_risk': death_risk,
            'grid_width': gw,
            'resource_scale': rscales,
            'steps': args.deep_steps,
            'initial_population': args.initial_population,
            'show_progress': False,
            'write_frames': False,
            'write_outputs': False,
            'report_interval': int(args.deep_steps) + 1,
            'webviewer_base': os.path.normpath(deep_run_dir),
            'sim_prefix': f'prom{i}_',
            'max_population': min(20000, int(args.deep_population)),
        }
        tf = tempfile.NamedTemporaryFile(prefix=f"deep_prom_{i}_", suffix=".json", delete=False, mode="w", encoding="utf-8")
        json.dump(cfg, tf)
        tf.flush()
        tf.close()
        deep_cfg_paths[i] = tf.name
        deep_jobs.append((i, (mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, gw)))

        def deep_cmd(job):
            idx = job[0]
            return [RUST_BIN, '--config-file', deep_cfg_paths[idx]]

        # reuse simple result builder that reads final population from deep_run_dir
        def deep_result_builder(job, extinct, out_text):
            idx, tpl = job
            mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, gw = tpl
            prefix = f'prom{idx}_'
            simf = find_sim_folder_for_prefix(deep_run_dir, prefix)
            final_pop = 0
            prop_moved = 0.0
            if simf:
                final_pop, prop_moved = read_final_population(simf)
            return [mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, gw, final_pop, prop_moved]

        # enumerate deep job parameter tuples (just parameter tuples; indices are assigned by runner)
        deep_pairs = [(mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, gw) for (mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, gw, _) in promising]
        # debug prints to help diagnose hangs in deep runs
        try:
            print("DEBUG: promising count:", len(promising))
            print("DEBUG: deep_cfg_paths keys:", sorted(list(deep_cfg_paths.keys())))
            print("DEBUG: deep_pairs sample (first 3):", deep_pairs[:3])
            print("DEBUG: deep_results_csv:", deep_results_csv)
            print("DEBUG: header:", header)
            # Use an explicit deep header matching deep_result_builder output
            deep_header = [
                "movement_cost_factor",
                "maintenance_cost_per_node",
                "reproduction_threshold",
                "reproduction_cost",
                "env_time_speed",
                "death_risk",
                "resource_scale",
                "grid_width",
                "final_population",
                "prop_moved",
            ]
            # reduce timeout and use a single worker for debugging
            run_param_sweep(deep_pairs, deep_cmd, deep_results_csv, deep_header, workers=1, timeout=600, result_row_builder=deep_result_builder)
        except Exception as e:
            print("ERROR running deep param sweep:", e)
        raise
        print('Deep runs finished')
        # cleanup deep temp config files
        try:
            for p in list(deep_cfg_paths.values()):
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception:
            pass

print('All done')
