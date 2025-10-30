#!/usr/bin/env python3
import os
import sys
import itertools
import json
from worker_pool import run_param_sweep

# ensure scripts/ is importable
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

EXE = (
    "./build/Release/evo-movement-sim.exe"
    if os.path.exists("./build/Release/evo-movement-sim.exe")
    else "./build/evo-movement-sim.exe"
)
EXE = os.path.normpath(EXE)
print("Using exe:", EXE)

# tiny grid for smoke test
movement_vals = [0.05, 5]
maintenance_vals = [0.01, 0.02]

burn_in = 5
steps = burn_in + 1

pairs = list(itertools.product(movement_vals, maintenance_vals))

# create deterministic scan run folder and per-run config so simulator writes into it
output_dir = os.path.join(os.path.normpath("."), "output", "Parameter scans")
os.makedirs(output_dir, exist_ok=True)
data_base = os.path.join(output_dir, "data")
os.makedirs(data_base, exist_ok=True)
existing = sorted([d for d in os.listdir(data_base) if d.startswith("scan")])
idx = len(existing) + 1
scan_dir = os.path.join(data_base, f"scan{idx:04d}")
os.makedirs(scan_dir, exist_ok=True)
run_cfg_path = os.path.normpath(os.path.join(scan_dir, "scan_config.json"))
with open(run_cfg_path, "w") as cf:
    json.dump({"webviewer_base": os.path.normpath(scan_dir)}, cf)

runs_out = os.path.join(scan_dir, "smoke_runs.csv")


def _safeval(v):
    s = f"{v:.6g}"
    return s.replace(".", "p")


def cmd_builder(job):
    mv, mc = job
    mv_s = _safeval(mv)
    mc_s = _safeval(mc)
    cfgname = f"config_mv_{mv_s}_mc_{mc_s}.json"
    # pass per-run config so simulator writes simNNNN under scan_dir
    return [
        EXE,
        f"--steps={steps}",
        f"--burn-in={burn_in}",
        "--report-interval=1",
        f"--movement-cost-factor={mv}",
        f"--maintenance-cost-per-node={mc}",
        "--no-progress",
        f"--config-out={cfgname}",
        f"--config-file={run_cfg_path}",
    ]


header = [
    "movement_cost_factor",
    "maintenance_cost_per_node",
    "extinction_time",
    "survived",
]

print("Running small sweep...")
run_param_sweep(pairs, cmd_builder, runs_out, header, workers=2, timeout=30)

print("Post-processing results...")
# prefer to look inside the scan run folder first
webviewer_base = os.path.join("webviewer")


def find_simdir_for_config(cfgname):
    try:
        if os.path.isdir(scan_dir):
            for nm in os.listdir(scan_dir):
                p = os.path.join(scan_dir, nm)
                if not os.path.isdir(p):
                    continue
                if os.path.exists(os.path.join(p, cfgname)):
                    return p
    except Exception:
        pass
    # fallback to legacy webviewer
    try:
        if not os.path.isdir(webviewer_base):
            return None
        for nm in os.listdir(webviewer_base):
            p = os.path.join(webviewer_base, nm)
            if not os.path.isdir(p):
                continue
            if os.path.exists(os.path.join(p, cfgname)):
                return p
    except Exception:
        return None
    return None


def latest_individuals_file(simdir):
    files = []
    for fn in os.listdir(simdir):
        if fn.startswith("individuals.tsv.") and fn.endswith(".tsv"):
            files.append(os.path.join(simdir, fn))
    if not files:
        for fn in os.listdir(simdir):
            if "individuals" in fn:
                files.append(os.path.join(simdir, fn))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def compute_median_delta_from_file(path):
    vals = []
    try:
        with open(path, "r", newline="") as f:
            _ = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                try:
                    ld = int(parts[-1])
                    vals.append(abs(ld) / 10.0)
                except Exception:
                    continue
    except Exception:
        return None
    if not vals:
        return None
    vals.sort()
    mid = vals[len(vals) // 2]
    return mid


for mv, mc in pairs:
    cfgname = f"config_mv_{_safeval(mv)}_mc_{_safeval(mc)}.json"
    simdir = find_simdir_for_config(cfgname)
    print("Pair", mv, mc, "simdir ->", simdir)
    if simdir:
        indf = latest_individuals_file(simdir)
        print("  individuals file ->", indf)
        if indf:
            med = compute_median_delta_from_file(indf)
            print("  median delta =", med)
        else:
            print("  no individuals file")

print("Smoke test complete.")
