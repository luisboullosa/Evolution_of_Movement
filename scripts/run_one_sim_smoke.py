#!/usr/bin/env python3
import os
import subprocess
import json

EXE = (
    "./build/Release/evo-movement-sim.exe"
    if os.path.exists("./build/Release/evo-movement-sim.exe")
    else "./build/evo-movement-sim.exe"
)
EXE = os.path.normpath(EXE)
print("Using exe:", EXE)

output_dir = os.path.join(os.path.normpath("."), "output", "Parameter scans")
os.makedirs(output_dir, exist_ok=True)
data_base = os.path.join(output_dir, "data")
os.makedirs(data_base, exist_ok=True)
existing = sorted([d for d in os.listdir(data_base) if d.startswith("scan")])
idx = len(existing) + 1
scan_dir = os.path.join(data_base, f"scan{idx:04d}")
os.makedirs(scan_dir, exist_ok=True)
run_cfg_path = os.path.join(scan_dir, "scan_config.json")
with open(run_cfg_path, "w") as f:
    json.dump({"webviewer_base": os.path.normpath(scan_dir)}, f)
print("Created run folder:", scan_dir)
print("Wrote config:", run_cfg_path)

# run a single short simulation to create simNNNN under scan_dir
cmd = [
    EXE,
    "--steps=2",
    "--burn-in=0",
    "--report-interval=1",
    "--no-progress",
    f"--config-file={run_cfg_path}",
]
print("Running:", cmd)
try:
    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30
    )
    print("Return code:", p.returncode)
    print("STDOUT:\n", p.stdout)
    print("STDERR:\n", p.stderr)
except Exception as e:
    print("Failed to run EXE:", e)

# list contents of scan_dir
for root, dirs, files in os.walk(scan_dir):
    level = root.replace(scan_dir, "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for f in files:
        print(f"{subindent}{f}")

# print any simNNNN directories if present
sims = [
    d
    for d in os.listdir(scan_dir)
    if d.startswith("sim") and os.path.isdir(os.path.join(scan_dir, d))
]
print("Sim dirs found under scan dir:", sims)
