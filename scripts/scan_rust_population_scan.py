#!/usr/bin/env python3
"""
Population-stability scan driver using the Rust simulator.
Performs a sweep over an expanded parameter grid. The scan,
runs short sims, records final population size, then runs deeper sims for promising
parameter sets (final_pop >= threshold).

Usage:
  python scripts/scan_rust_population_scan.py --workers 4 --steps 2000 --initial-population 500 --pilot --no-deep --no-write-outputs 
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
    # sampling mode (discrete): draw parameter values by randomly choosing
    # from the provided discrete lists so sampled values remain on the grid.
    if args.seed is not None:
        random.seed(args.seed)
    sampled = []
    for i in range(args.sample):
        mv = random.choice(movement_vals)
        mc = random.choice(maintenance_vals)
        repro_thr = random.choice(repro_thresholds)
        repro_cost = random.choice(repro_costs)
        env_time_speed = random.choice(env_time_speeds)
        death_risk = random.choice(death_risks)
        # sample grid width and resource scale discretely
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

# Write a small manifest describing the planned scan so a live Plotly viewer
# can initialize axes and know how many jobs to expect.
try:
        manifest = {
                'total_jobs': len(pairs),
                'movement_vals': movement_vals,
                'maintenance_vals': maintenance_vals,
                'resource_scales': resource_scales,
                'grid_widths': grid_vals,
                'param_names': ['movement_cost_factor', 'maintenance_cost_per_node', 'reproduction_threshold', 'reproduction_cost', 'env_time_speed', 'death_risk', 'resource_scale', 'grid_width']
        }
        manifest_path = os.path.join(jobs_dir, 'scan_manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as mf:
                json.dump(manifest, mf)
        # also write a tiny Plotly HTML viewer into the jobs dir that polls progress.json
        html_path = os.path.join(jobs_dir, 'plotly_live.html')
        html_content = r'''
<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Scan Live Viewer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body{font-family:Arial,Helvetica,sans-serif;margin:8px} #controls{margin-bottom:8px}</style>
</head>
<body>
    <h3>Scan Live Viewer</h3>
    <div id="controls">
        Response: <select id="resp"><option value="final_population">final_population</option><option value="prop_moved">prop_moved</option><option value="survived">survived</option></select>
        Resource scale: <select id="rs"></select>
        Grid width: <select id="gw"></select>
        Poll interval (s): <input id="interval" type="number" value="5" style="width:60px" />
        <button id="start">Start</button>
        <button id="stop">Stop</button>
    </div>
    <div id="plot" style="width:100%;height:700px"></div>

    <script>
    const manifestUrl = 'scan_manifest.json';
    const progressUrl = 'progress.json';
    let manifest = null;
    let pollTimer = null;

    async function loadManifest(){
        try{ manifest = await fetch(manifestUrl).then(r=>r.json()); }
        catch(e){ console.warn('Could not load manifest', e); manifest = null; }
        if(!manifest) return;
        const rsSel = document.getElementById('rs');
        const gwSel = document.getElementById('gw');
        rsSel.innerHTML = manifest.resource_scales.map(v=>`<option value="${v}">${v}</option>`).join('');
        gwSel.innerHTML = manifest.grid_widths.map(v=>`<option value="${v}">${v}</option>`).join('');
    }

    function makeEmptyGrid(movement_vals, maintenance_vals){
        const z = [];
        for(let i=0;i<maintenance_vals.length;i++){
            z.push(new Array(movement_vals.length).fill(null));
        }
        return z;
    }

    async function pollAndUpdate(){
        try{
            const data = await fetch(progressUrl).then(r=>r.json());
            // data is array of entries
            const resp = document.getElementById('resp').value;
            const rs = parseFloat(document.getElementById('rs').value);
            const gw = parseInt(document.getElementById('gw').value);
            const movement_vals = manifest.movement_vals;
            const maintenance_vals = manifest.maintenance_vals;
            // build mapping from (mv,mc,rs,gw) -> metric
            const metrics = {};
            for(const e of data){
                const key = `${e.movement_cost_factor}|${e.maintenance_cost_per_node}|${e.resource_scale}|${e.grid_width}`;
                metrics[key] = e;
            }
            const z = makeEmptyGrid(movement_vals, maintenance_vals);
            for(let i=0;i<maintenance_vals.length;i++){
                for(let j=0;j<movement_vals.length;j++){
                    const key = `${movement_vals[j]}|${maintenance_vals[i]}|${rs}|${gw}`;
                    const obj = metrics[key];
                    if(obj){
                        if(resp==='survived') z[i][j] = (obj.final_population>0)?1:0;
                        else z[i][j] = obj[resp]!=null?obj[resp]:null;
                    } else { z[i][j] = null; }
                }
            }
            const dataPlot = [{ z: z, x: movement_vals, y: maintenance_vals, type: 'heatmap', colorscale: resp==='survived'?[[0,'black'],[1,'yellow']]:undefined, hoverongaps:false }];
            const layout = { title: `${resp} (rs=${rs}, gw=${gw})`, xaxis:{title:'movement_cost_factor'}, yaxis:{title:'maintenance_cost_per_node'}};
            Plotly.react('plot', dataPlot, layout, {responsive:true});
        }catch(e){ console.warn('poll error', e); }
    }

    document.getElementById('start').addEventListener('click', ()=>{
        const iv = Math.max(1, parseInt(document.getElementById('interval').value)||5);
        if(pollTimer) clearInterval(pollTimer);
        pollAndUpdate();
        pollTimer = setInterval(pollAndUpdate, iv*1000);
    });
    document.getElementById('stop').addEventListener('click', ()=>{ if(pollTimer) clearInterval(pollTimer); pollTimer=null; });

    loadManifest();
    </script>
</body>
</html>
'''
        try:
                with open(html_path, 'w', encoding='utf-8') as hf:
                        hf.write(html_content)
        except Exception:
                pass
        print(f'Wrote scan manifest: {manifest_path} and live viewer: {html_path}')
except Exception:
        pass

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
    # For pilot runs, introduce phenotypic variation via radiate multiplier
    if args.pilot:
        cfg['radiate'] = 1000
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

# ---- Post-processing: matrix-of-matrices heatmap generator ----
def plot_matrix_of_matrices(results_csv, out_dir):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import colors
    except Exception:
        print('Matplotlib / numpy not available; skipping heatmap generation')
        return

    # read results into list of dicts
    rows = []
    with open(results_csv, 'r', encoding='utf-8') as rf:
        r = csv.DictReader(rf)
        for row in r:
            rows.append(row)

    if not rows:
        print('No results to plot')
        return

    # detect unique axes
    def unique_sorted(key, cast=float):
        vals = set()
        for row in rows:
            try:
                vals.add(cast(row.get(key, '') or 0))
            except Exception:
                pass
        return sorted(vals)

    movement_vals_r = unique_sorted('movement_cost_factor', float)
    maintenance_vals_r = unique_sorted('maintenance_cost_per_node', float)
    resource_scales_r = unique_sorted('resource_scale', float)
    grid_widths_r = unique_sorted('grid_width', int)

    if not movement_vals_r or not maintenance_vals_r:
        print('Insufficient parameter axes for plotting')
        return

    # build mapping from parameter tuple -> metrics
    metrics = {}
    for row in rows:
        try:
            mv = float(row.get('movement_cost_factor') or 0)
            mc = float(row.get('maintenance_cost_per_node') or 0)
            rs = float(row.get('resource_scale') or 0)
            gw = int(float(row.get('grid_width') or 0))
            fp = float(row.get('final_population') or 0)
            pm = float(row.get('prop_moved') or 0)
            metrics[(mv, mc, rs, gw)] = {'final_population': fp, 'prop_moved': pm}
        except Exception:
            continue

    # helper to build inner matrix for a given resource_scale & grid_width
    def inner_matrix(rs, gw, key):
        M = np.full((len(maintenance_vals_r), len(movement_vals_r)), np.nan, dtype=float)
        for i, mc in enumerate(maintenance_vals_r):
            for j, mv in enumerate(movement_vals_r):
                val = metrics.get((mv, mc, rs, gw))
                if val is not None:
                    if key == 'survived':
                        M[i, j] = 1.0 if float(val.get('final_population', 0)) > 0 else 0.0
                    else:
                        M[i, j] = float(val.get(key, np.nan))
        return M

    # responses to plot
    responses = {
        'final_population': dict(cmap='viridis', title='Final population'),
        'prop_moved': dict(cmap='plasma', title='Proportion moved'),
        'survived': dict(cmap=colors.ListedColormap(['black', 'yellow']), title='Survived (binary)')
    }

    # create outer grid (rows = resource_scales, cols = grid_widths)
    rows_out = len(resource_scales_r)
    cols_out = len(grid_widths_r)

    for resp_key, opts in responses.items():
        fig_w = max(6, cols_out * 2)
        fig_h = max(4, rows_out * 2)
        fig, axes = plt.subplots(rows_out, cols_out, figsize=(fig_w, fig_h), squeeze=False)

        # find global vmin/vmax for consistent color scaling (except for binary 'survived')
        if resp_key != 'survived':
            all_vals = []
            for rs in resource_scales_r:
                for gw in grid_widths_r:
                    M = inner_matrix(rs, gw, resp_key)
                    all_vals.append(M)
            all_concat = np.concatenate([m.flatten() for m in all_vals]) if all_vals else np.array([])
            finite = all_concat[np.isfinite(all_concat)]
            if finite.size:
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
                if vmin == vmax:
                    vmax = vmin + 1.0
            else:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0

        for i, rs in enumerate(resource_scales_r):
            for j, gw in enumerate(grid_widths_r):
                ax = axes[i][j]
                M = inner_matrix(rs, gw, resp_key)
                # mask NaNs to show as light gray
                M_masked = np.ma.masked_invalid(M)
                im = ax.imshow(M_masked, aspect='auto', interpolation='nearest', cmap=opts.get('cmap'), vmin=vmin, vmax=vmax)
                ax.set_xticks(range(len(movement_vals_r)))
                ax.set_xticklabels([str(v) for v in movement_vals_r], rotation=90, fontsize=6)
                ax.set_yticks(range(len(maintenance_vals_r)))
                ax.set_yticklabels([str(v) for v in maintenance_vals_r], fontsize=6)
                ax.set_title(f"rs={rs}, gw={gw}", fontsize=8)
                ax.tick_params(axis='both', which='both', length=0)
        # colorbar
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        fig.suptitle(opts.get('title', resp_key))
        out_path = os.path.join(out_dir, f'heatmap_{resp_key}.png')
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Wrote heatmap: {out_path}')

# generate heatmaps from the short-run results
try:
    plot_matrix_of_matrices(results_csv, jobs_dir)
except Exception as e:
    print('Error generating heatmaps:', e)
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
