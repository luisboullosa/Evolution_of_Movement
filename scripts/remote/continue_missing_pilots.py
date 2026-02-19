#!/usr/bin/env python3
import glob
import os
import subprocess

LOCAL_RUN: str = os.path.normpath("output/remote_scan/run20260114_010108")
REMOTE_HOST = "root@192.168.2.129"
REMOTE_RUN_ROOT = "/root/projects/scan_runs/run20260114_010108"
REMOTE_HELPER: str = REMOTE_RUN_ROOT + "/remote_helpers/remote_run_job.py"

configs: list[str] = sorted(glob.glob(os.path.join(LOCAL_RUN, "configs", "pilot", "pilot_*.json")))
if not configs:
    print("No pilot configs found; nothing to do")
    raise SystemExit(0)

os.makedirs(os.path.join(LOCAL_RUN, "logs", "pilot"), exist_ok=True)

for cfg: str in configs:
    name: str = os.path.basename(cfg)
    base: str = name[:-5]  # pilot_000001
    local_log: str = os.path.join(LOCAL_RUN, "logs", "pilot", f"{base}.stdout.txt")
    if os.path.exists(local_log):
        print(f"Skipping {base}: log exists")
        continue

    print(f"Processing missing pilot: {base}")
    # copy config to remote
    scp_cmd: list[str] = ["scp", cfg, f"{REMOTE_HOST}:{REMOTE_RUN_ROOT}/configs/pilot/{name}"]
    print("  scp ->", " ".join(scp_cmd))
    r: subprocess.CompletedProcess[str] = subprocess.run(scp_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print("  scp failed:", r.stdout)
        continue

    # run remote helper
    remote_cfg_path: str = f"{REMOTE_RUN_ROOT}/configs/pilot/{name}"
    job_dir: str = base.replace("pilot_", "job")
    remote_output_dir: str = f"{REMOTE_RUN_ROOT}/pilot/{job_dir}"
    ssh_cmd: list[str] = [
        "ssh",
        REMOTE_HOST,
        "python3",
        REMOTE_HELPER,
        "--repo-dir",
        "/root/projects/Movement",
        "--manifest-path",
        "rust/Cargo.toml",
        "--bin",
        "evo_movement_rust",
        "--config",
        remote_cfg_path,
        "--release",
        "--no-write-outputs",
        "--output-run-dir",
        remote_output_dir,
    ]
    print("  ssh ->", " ".join(ssh_cmd))
    try:
        cp: subprocess.CompletedProcess[str] = subprocess.run(ssh_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600 * 8)
        out: str = cp.stdout or ""
    except subprocess.TimeoutExpired:
        out = "[local orchestrator] ssh timed out"
    with open(local_log, "w", encoding="utf-8") as f: os.TextIOWrapper[_WrappedBuffer]:
        f.write(out)
    print(f"  wrote log: {local_log}")

print("All missing pilots processed")
