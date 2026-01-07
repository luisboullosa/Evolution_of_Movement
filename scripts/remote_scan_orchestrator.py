#!/usr/bin/env python3
"""Remote scan orchestrator for running Rust scan jobs on an Orange Pi.

What it does
- Generates pilot scan jobs (usually sampled), runs them remotely over SSH.
- Collects pilot results (stdout JSON summary emitted by the simulator).
- Selects promising jobs and runs deep scans remotely.
- Retrieves population snapshots (`population_snapshot.cbor`) from deep scans.

Assumptions
- The remote machine has:
  - `ssh` access from this machine (Windows OpenSSH works fine).
  - Rust + Cargo installed.
  - This repo checked out somewhere (or at least the `rust/` crate) so
    `cargo run --manifest-path <repo>/rust/Cargo.toml --bin evo_movement_rust` works.
  - Python 3 available (to run the remote helper).

This script deliberately avoids extra dependencies (no Paramiko); it shells out
to `ssh` and `scp`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import random
import sqlite3
import subprocess
import sys
import threading
import time
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable


def _posix_join(*parts: str) -> str:
    out = ""
    for p in parts:
        if not p:
            continue

        # First element: preserve absolute-path intent.
        if not out:
            if p.startswith("/"):
                out = "/" + p.strip("/")
            else:
                out = p.strip("/")
            continue

        out = out.rstrip("/")
        p = p.strip("/")
        if p:
            out = out + "/" + p
    return out


def _atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def _ssh_base_args(args) -> list[str]:
    base = ["ssh"]
    if args.ssh_port:
        base.extend(["-p", str(args.ssh_port)])
    if args.ssh_key:
        base.extend(["-i", args.ssh_key])
    if args.ssh_extra:
        base.extend(args.ssh_extra)
    return base


def _scp_base_args(args) -> list[str]:
    base = ["scp"]
    if args.ssh_port:
        base.extend(["-P", str(args.ssh_port)])
    if args.ssh_key:
        base.extend(["-i", args.ssh_key])
    if args.ssh_extra:
        base.extend(args.ssh_extra)
    return base


def _ssh(args, remote_cmd: str, *, check: bool = True) -> str:
    target = f"{args.ssh_user}@{args.remote_host}" if args.ssh_user else args.remote_host
    cmd = _ssh_base_args(args) + [target, remote_cmd]
    cp = _run(cmd, check=check)
    return cp.stdout or ""


def _scp_to_remote(args, local_path: str, remote_path: str) -> None:
    target = f"{args.ssh_user}@{args.remote_host}" if args.ssh_user else args.remote_host
    spec = f"{target}:{remote_path}"
    cmd = _scp_base_args(args) + [local_path, spec]
    _run(cmd, check=True)


def _scp_from_remote(args, remote_path: str, local_path: str) -> None:
    target = f"{args.ssh_user}@{args.remote_host}" if args.ssh_user else args.remote_host
    spec = f"{target}:{remote_path}"
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    cmd = _scp_base_args(args) + [spec, local_path]
    _run(cmd, check=True)


def _ensure_remote_dir(args, remote_dir: str) -> None:
    _ssh(args, f"mkdir -p {json.dumps(remote_dir)}")


def _remote_path_exists(args, path: str) -> bool:
    out = _ssh(args, f"test -e {json.dumps(path)} && echo OK || echo MISSING", check=True)
    return "OK" in out


def _remote_has_command(args, cmd_name: str) -> bool:
    # For cargo, also check if .cargo/env exists and can be sourced
    if cmd_name == "cargo":
        out = _ssh(args, f"test -f $HOME/.cargo/env && . $HOME/.cargo/env && command -v cargo >/dev/null 2>&1 && echo OK || echo MISSING", check=True)
    else:
        out = _ssh(args, f"command -v {json.dumps(cmd_name)} >/dev/null 2>&1 && echo OK || echo MISSING", check=True)
    return "OK" in out


def _should_exclude_repo_path(rel_posix: str) -> bool:
    # Avoid shipping bulky or irrelevant directories to the Pi.
    parts = [p for p in rel_posix.split("/") if p]
    if not parts:
        return False
    head = parts[0]
    if head in {".git", ".venv", "output", "build", "__pycache__"}:
        return True
    # common build dirs
    if rel_posix.startswith("rust/target/"):
        return True
    if rel_posix.startswith("rust/target"):
        return True
    if rel_posix.startswith("build/"):
        return True
    if rel_posix.endswith(".pyc"):
        return True
    return False


def _make_repo_tarball(local_repo_dir: str, tar_path: str, *, top_dir_name: str) -> None:
    local_repo_dir = os.path.abspath(local_repo_dir)
    with tarfile.open(tar_path, "w:gz") as tf:
        for root, dirs, files in os.walk(local_repo_dir):
            # prune excluded dirs in-place
            rel_root = os.path.relpath(root, local_repo_dir)
            rel_root_posix = "" if rel_root == "." else rel_root.replace("\\", "/")
            # prune children
            kept_dirs = []
            for d in dirs:
                rel = (rel_root_posix + "/" + d) if rel_root_posix else d
                if not _should_exclude_repo_path(rel):
                    kept_dirs.append(d)
            dirs[:] = kept_dirs

            for fn in files:
                rel = (rel_root_posix + "/" + fn) if rel_root_posix else fn
                if _should_exclude_repo_path(rel):
                    continue
                full = os.path.join(root, fn)
                arcname = (top_dir_name + "/" + rel) if rel else top_dir_name
                tf.add(full, arcname=arcname)


def _bootstrap_repo_if_missing(args, *, run_id: str) -> None:
    """Ensure the repo exists on the remote machine.

    Strategy:
    - If `remote_repo_dir` already exists and contains `rust/Cargo.toml`, do nothing.
    - Otherwise, upload a tar.gz of the local repo and extract it under /root/projects.
    """

    remote_repo_dir = args.remote_repo_dir
    if _remote_path_exists(args, _posix_join(remote_repo_dir, "rust", "Cargo.toml")):
        return

    if not args.bootstrap:
        raise RuntimeError(
            "Remote repo not found (missing rust/Cargo.toml). "
            "Re-run with --bootstrap to upload the repo to the remote host."
        )

    remote_parent = os.path.dirname(remote_repo_dir.rstrip("/")) or "/"
    _ensure_remote_dir(args, remote_parent)

    local_repo_dir = os.path.abspath(args.local_repo_dir)
    top_dir_name = os.path.basename(remote_repo_dir.rstrip("/")) or "Movement"

    remote_run_root = _posix_join(args.remote_run_base, run_id)
    _ensure_remote_dir(args, remote_run_root)
    remote_tar = _posix_join(remote_run_root, "repo_upload.tar.gz")

    with tempfile.TemporaryDirectory(prefix="movement_repo_upload_") as td:
        local_tar = os.path.join(td, "repo_upload.tar.gz")
        _make_repo_tarball(local_repo_dir, local_tar, top_dir_name=top_dir_name)
        _scp_to_remote(args, local_tar, remote_tar)

    # Extract into parent so it lands at remote_repo_dir.
    _ssh(args, f"tar -xzf {json.dumps(remote_tar)} -C {json.dumps(remote_parent)}")

    # Validate.
    if not _remote_path_exists(args, _posix_join(remote_repo_dir, "rust", "Cargo.toml")):
        raise RuntimeError("Bootstrap failed: remote rust/Cargo.toml still missing after upload.")


def _remote_cargo_build(args) -> str:
    manifest_abs = _posix_join(args.remote_repo_dir, args.remote_manifest_path)
    parts = [
        "set -e",
        "test -f $HOME/.cargo/env && . $HOME/.cargo/env",
        f"cd {json.dumps(args.remote_repo_dir)}",
        "cargo build",
        "--manifest-path",
        json.dumps(manifest_abs),
        "--bin",
        json.dumps(args.bin_name),
    ]
    if args.release:
        parts.append("--release")
    cmd = " && ".join(parts[:3]) + " && " + " ".join(parts[3:])
    return _ssh(args, cmd, check=False)


def _deploy_remote_helper(args, remote_helper_path: str) -> None:
    """Copy remote helper script if missing or if --force-deploy-helper."""

    local_helper = os.path.normpath(os.path.join("scripts", "remote", "remote_run_job.py"))
    if not os.path.exists(local_helper):
        raise FileNotFoundError(f"Missing local helper at {local_helper}")

    if args.force_deploy_helper:
        _ensure_remote_dir(args, os.path.dirname(remote_helper_path))
        _scp_to_remote(args, local_helper, remote_helper_path)
        return

    # Check existence; if missing, deploy.
    out = _ssh(args, f"test -f {json.dumps(remote_helper_path)} && echo OK || echo MISSING", check=True)
    if "OK" not in out:
        _ensure_remote_dir(args, os.path.dirname(remote_helper_path))
        _scp_to_remote(args, local_helper, remote_helper_path)


def _parse_remote_job_result(stdout: str) -> dict[str, Any] | None:
    # Remote helper prints: REMOTE_JOB_RESULT {json}
    for line in (stdout or "").splitlines()[::-1]:
        if line.startswith("REMOTE_JOB_RESULT "):
            payload = line[len("REMOTE_JOB_RESULT ") :].strip()
            try:
                obj = json.loads(payload)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


@dataclass(frozen=True)
class Params:
    movement_cost_factor: float
    maintenance_cost_per_node: float
    reproduction_threshold: float
    reproduction_cost: float
    env_time_speed: float
    death_risk: float
    resource_scale: float
    grid_width: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "movement_cost_factor": self.movement_cost_factor,
            "maintenance_cost_per_node": self.maintenance_cost_per_node,
            "reproduction_threshold": self.reproduction_threshold,
            "reproduction_cost": self.reproduction_cost,
            "env_time_speed": self.env_time_speed,
            "death_risk": self.death_risk,
            "resource_scale": self.resource_scale,
            "grid_width": self.grid_width,
        }


def _pilot_param_grid(pilot: bool) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[int]]:
    # Keep in sync with scripts/scan_rust_population_scan.py pilot ranges.
    movement_vals = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    maintenance_vals = [0.0, 1e-05, 5e-05, 1e-04, 5e-04, 0.001, 0.005, 0.01]
    repro_thresholds = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    repro_costs = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
    env_time_speeds = [0.0, 0.0001, 0.001, 0.01, 0.1]
    death_risks = [0.0, 1e-08, 1e-06, 1e-04, 1e-03]
    grid_vals = [25, 50, 100, 200]
    resource_scales = [10.0, 100.0]
    return movement_vals, maintenance_vals, repro_thresholds, repro_costs, env_time_speeds, death_risks, resource_scales, grid_vals


def _sample_params(sample_n: int, seed: int | None) -> list[Params]:
    if seed is not None:
        random.seed(seed)
    mv, mc, rt, rc, ets, dr, rs, gw = _pilot_param_grid(True)
    out: list[Params] = []
    for _ in range(sample_n):
        out.append(
            Params(
                movement_cost_factor=random.choice(mv),
                maintenance_cost_per_node=random.choice(mc),
                reproduction_threshold=random.choice(rt),
                reproduction_cost=random.choice(rc),
                env_time_speed=random.choice(ets),
                death_risk=random.choice(dr),
                resource_scale=random.choice(rs),
                grid_width=int(random.choice(gw)),
            )
        )
    return out


def _init_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              job_id TEXT PRIMARY KEY,
              phase TEXT NOT NULL,
              status TEXT NOT NULL,
              params_json TEXT NOT NULL,
              started_at REAL,
              ended_at REAL,
              final_population INTEGER,
              prop_moved REAL,
              remote_output_dir TEXT,
              local_snapshot_path TEXT,
              stdout_path TEXT
            )
            """
        )
        conn.commit()


def _db_upsert_job(db_path: str, *, job_id: str, phase: str, status: str, params: Params) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs(job_id, phase, status, params_json)
            VALUES(?,?,?,?)
            ON CONFLICT(job_id) DO UPDATE SET
              phase=excluded.phase,
              status=excluded.status,
              params_json=excluded.params_json
            """,
            (job_id, phase, status, json.dumps(params.as_dict())),
        )
        conn.commit()


def _db_mark_started(db_path: str, *, job_id: str, stdout_path: str | None) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status=?, started_at=?, stdout_path=? WHERE job_id=?",
            ("running", time.time(), stdout_path, job_id),
        )
        conn.commit()


def _db_mark_done(
    db_path: str,
    *,
    job_id: str,
    status: str,
    final_population: int | None,
    prop_moved: float | None,
    remote_output_dir: str | None,
    local_snapshot_path: str | None,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
              SET status=?, ended_at=?, final_population=?, prop_moved=?, remote_output_dir=?, local_snapshot_path=?
              WHERE job_id=?
            """,
            (status, time.time(), final_population, prop_moved, remote_output_dir, local_snapshot_path, job_id),
        )
        conn.commit()


def _write_csv_row(path: str, header: list[str], row: list[Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def _run_one_remote(
    args,
    *,
    run_id: str,
    phase: str,
    job_idx: int,
    params: Params,
    steps: int,
    initial_population: int,
    max_population: int,
    write_snapshot: bool,
    deep_output_dir_remote: str | None,
    local_run_dir: str,
    db_path: str,
    results_csv: str,
    results_header: list[str],
    progress_path: str,
    progress_lock: threading.Lock,
    progress_entries: list[dict[str, Any]],
) -> tuple[int | None, float | None]:
    job_id = f"{phase}_{job_idx:06d}"
    _db_upsert_job(db_path, job_id=job_id, phase=phase, status="pending", params=params)

    local_cfg_dir = os.path.join(local_run_dir, "configs", phase)
    os.makedirs(local_cfg_dir, exist_ok=True)
    local_cfg_path = os.path.join(local_cfg_dir, f"{job_id}.json")

    cfg: dict[str, Any] = {
        **params.as_dict(),
        "steps": int(steps),
        "initial_population": int(initial_population),
        "max_population": int(max_population),
        "show_progress": False,
        "write_frames": False,
        "write_outputs": False,
        "write_config": False,
        "write_snapshot": bool(write_snapshot),
        "report_interval": int(steps) + 1,
        "sim_prefix": f"{job_id}_",
        # output_dir_base is only relevant if write_snapshot/write_config/write_outputs are enabled.
        "output_dir_base": "",
    }
    with open(local_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)

    remote_run_root = _posix_join(args.remote_run_base, run_id)
    remote_cfg_dir = _posix_join(remote_run_root, "configs", phase)
    remote_cfg_path = _posix_join(remote_cfg_dir, os.path.basename(local_cfg_path))

    _ensure_remote_dir(args, remote_cfg_dir)
    _scp_to_remote(args, local_cfg_path, remote_cfg_path)

    remote_helper = _posix_join(remote_run_root, "remote_helpers", "remote_run_job.py")
    _deploy_remote_helper(args, remote_helper)

    stdout_path = os.path.join(local_run_dir, "logs", phase, f"{job_id}.stdout.txt")
    os.makedirs(os.path.dirname(stdout_path) or ".", exist_ok=True)

    _db_mark_started(db_path, job_id=job_id, stdout_path=stdout_path)

    remote_cmd_parts = [
        args.remote_python,
        json.dumps(remote_helper),
        "--repo-dir",
        json.dumps(args.remote_repo_dir),
        "--manifest-path",
        json.dumps(args.remote_manifest_path),
        "--bin",
        json.dumps(args.bin_name),
        "--config",
        json.dumps(remote_cfg_path),
    ]
    if args.release:
        remote_cmd_parts.append("--release")
    remote_cmd_parts.append("--no-write-outputs")
    if deep_output_dir_remote:
        _ensure_remote_dir(args, deep_output_dir_remote)
        remote_cmd_parts.extend(["--output-run-dir", json.dumps(deep_output_dir_remote)])

    remote_cmd = " ".join(remote_cmd_parts)

    out_text = ""
    ok = False
    try:
        out_text = _ssh(args, remote_cmd, check=False)
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(out_text)
        parsed = _parse_remote_job_result(out_text)
        ok = bool(parsed and parsed.get("ok"))
    except Exception as e:
        out_text = f"[orchestrator] SSH failure: {e}\n" + (out_text or "")
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(out_text)
        ok = False
        parsed = None

    final_pop: int | None = None
    prop_moved: float | None = None
    if parsed and isinstance(parsed.get("summary"), dict):
        s = parsed["summary"]
        try:
            final_pop = int(s.get("final_population") or 0)
        except Exception:
            final_pop = None
        try:
            prop_moved = float(s.get("prop_moved") or 0.0)
        except Exception:
            prop_moved = None

    local_snapshot_path: str | None = None
    if write_snapshot and deep_output_dir_remote:
        # pull snapshot and config.json from the known remote output dir
        snap_remote = _posix_join(deep_output_dir_remote, "population_snapshot.cbor")
        snap_local = os.path.join(local_run_dir, "deep_snapshots", f"{job_id}_population_snapshot.cbor")
        try:
            _scp_from_remote(args, snap_remote, snap_local)
            local_snapshot_path = snap_local
        except Exception:
            local_snapshot_path = None

    _db_mark_done(
        db_path,
        job_id=job_id,
        status="done" if ok else "failed",
        final_population=final_pop,
        prop_moved=prop_moved,
        remote_output_dir=deep_output_dir_remote,
        local_snapshot_path=local_snapshot_path,
    )

    _write_csv_row(
        results_csv,
        results_header,
        [
            job_id,
            phase,
            params.movement_cost_factor,
            params.maintenance_cost_per_node,
            params.reproduction_threshold,
            params.reproduction_cost,
            params.env_time_speed,
            params.death_risk,
            params.resource_scale,
            params.grid_width,
            final_pop if final_pop is not None else "",
            prop_moved if prop_moved is not None else "",
            "ok" if ok else "failed",
        ],
    )

    with progress_lock:
        progress_entries.append(
            {
                "job_id": job_id,
                "phase": phase,
                **params.as_dict(),
                "final_population": final_pop,
                "prop_moved": prop_moved,
                "ok": ok,
                "t": time.time(),
            }
        )
        _atomic_write_json(progress_path, progress_entries)

    return final_pop, prop_moved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-host", required=True, help="Remote host/IP (Orange Pi)")
    ap.add_argument("--ssh-user", default="", help="SSH user (optional)")
    ap.add_argument("--ssh-port", type=int, default=0, help="SSH port (optional)")
    ap.add_argument("--ssh-key", default="", help="Path to SSH private key (optional)")
    ap.add_argument("--ssh-extra", nargs="*", default=[], help="Extra args passed to ssh/scp (advanced)")

    ap.add_argument("--remote-repo-dir", default="/root/projects/Movement", help="Repo root on remote machine")
    ap.add_argument("--remote-run-base", default="/root/projects/scan_runs", help="Remote directory to store configs/outputs for this run")
    ap.add_argument("--remote-python", default="python3", help="Remote python executable")
    ap.add_argument("--remote-manifest-path", default="rust/Cargo.toml", help="Manifest path relative to repo root")

    ap.add_argument("--bootstrap", action="store_true", help="If remote repo is missing, upload this local repo to remote-repo-dir")
    ap.add_argument(
        "--bootstrap-build",
        action="store_true",
        help="After ensuring the repo exists remotely, run `cargo build` for the requested bin on the remote host",
    )
    ap.add_argument("--local-repo-dir", default=os.path.normpath("."), help="Local repo root to upload when using --bootstrap")

    ap.add_argument("--bin", dest="bin_name", default="evo_movement_rust", help="Cargo bin name")
    ap.add_argument("--release", action="store_true", help="Use --release (recommended)")
    ap.add_argument("--force-deploy-helper", action="store_true", help="Always re-copy the remote helper")

    ap.add_argument("--local-outdir", default="output/remote_scan", help="Local output base")
    ap.add_argument("--sample", type=int, default=200, help="Pilot sample count")
    ap.add_argument("--seed", type=int, default=None, help="Sampling seed")
    ap.add_argument("--workers", type=int, default=1, help="Concurrent remote jobs")

    ap.add_argument("--steps", type=int, default=200, help="Pilot steps")
    ap.add_argument("--initial-population", type=int, default=200, help="Pilot initial population")
    ap.add_argument("--max-population", type=int, default=1000, help="Pilot max population")
    ap.add_argument("--threshold", type=int, default=200, help="Pilot final-pop threshold")

    ap.add_argument("--deep", action="store_true", help="Run deep scans after pilot")
    ap.add_argument("--deep-steps", type=int, default=10000)
    ap.add_argument("--deep-max-population", type=int, default=20000)
    ap.add_argument("--max-deep", type=int, default=25, help="Cap number of deep runs")

    args = ap.parse_args()
    if not args.release:
        # default to release for remote execution
        args.release = True

    run_id = _dt.datetime.now().strftime("run%Y%m%d_%H%M%S")
    local_run_dir = os.path.normpath(os.path.join(args.local_outdir, run_id))
    os.makedirs(local_run_dir, exist_ok=True)

    # Ensure repo exists on the remote host (upload if requested).
    _bootstrap_repo_if_missing(args, run_id=run_id)

    # Preflight: we need cargo available for remote runs.
    if not _remote_has_command(args, "cargo"):
        raise RuntimeError(
            "Remote host is missing `cargo` in PATH. Install Rust/Cargo on the remote machine, then retry."
        )

    if args.bootstrap_build:
        build_out = _remote_cargo_build(args)
        os.makedirs(os.path.join(local_run_dir, "logs"), exist_ok=True)
        with open(os.path.join(local_run_dir, "logs", "bootstrap_build.stdout.txt"), "w", encoding="utf-8") as f:
            f.write(build_out or "")

    db_path = os.path.join(local_run_dir, "jobs.sqlite")
    _init_db(db_path)

    params_list = _sample_params(args.sample, args.seed)

    manifest = {
        "run_id": run_id,
        "remote_host": args.remote_host,
        "remote_repo_dir": args.remote_repo_dir,
        "remote_run_base": args.remote_run_base,
        "sample": args.sample,
        "seed": args.seed,
        "steps": args.steps,
        "initial_population": args.initial_population,
        "max_population": args.max_population,
        "threshold": args.threshold,
        "deep": bool(args.deep),
        "deep_steps": args.deep_steps,
        "deep_max_population": args.deep_max_population,
        "max_deep": args.max_deep,
    }
    _atomic_write_json(os.path.join(local_run_dir, "manifest.json"), manifest)

    results_header = [
        "job_id",
        "phase",
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
        "status",
    ]

    progress_path = os.path.join(local_run_dir, "progress.json")
    progress_entries: list[dict[str, Any]] = []
    progress_lock = threading.Lock()

    pilot_csv = os.path.join(local_run_dir, "results_pilot.csv")

    # Run pilot jobs.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    promising: list[tuple[int, Params, int]] = []

    remote_root = _posix_join(args.remote_run_base, run_id)
    _ensure_remote_dir(args, remote_root)

    def submit_pilot(idx: int, p: Params):
        return _run_one_remote(
            args,
            run_id=run_id,
            phase="pilot",
            job_idx=idx,
            params=p,
            steps=args.steps,
            initial_population=args.initial_population,
            max_population=args.max_population,
            write_snapshot=False,
            deep_output_dir_remote=None,
            local_run_dir=local_run_dir,
            db_path=db_path,
            results_csv=pilot_csv,
            results_header=results_header,
            progress_path=progress_path,
            progress_lock=progress_lock,
            progress_entries=progress_entries,
        )

    print(f"Remote pilot scan: {len(params_list)} jobs -> {pilot_csv}")
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {ex.submit(submit_pilot, i, p): (i, p) for i, p in enumerate(params_list)}
        completed = 0
        for fut in as_completed(futs):
            i, p = futs[fut]
            try:
                fp, _pm = fut.result()
            except Exception:
                fp = None
            completed += 1
            if fp is not None and fp >= args.threshold:
                promising.append((i, p, int(fp)))
            if completed % 10 == 0 or completed == len(params_list):
                print(f"Pilot progress: {completed}/{len(params_list)}")

    promising.sort(key=lambda x: x[2], reverse=True)
    prom_path = os.path.join(local_run_dir, "promising.json")
    _atomic_write_json(
        prom_path,
        [
            {"pilot_idx": i, "final_population": fp, "params": p.as_dict()}
            for (i, p, fp) in promising
        ],
    )
    print(f"Promising: {len(promising)} (threshold >= {args.threshold}) -> {prom_path}")

    if not args.deep:
        print("Done (pilot only).")
        return 0

    # Deep phase: run top-N promising with snapshots into deterministic remote output dirs.
    deep_csv = os.path.join(local_run_dir, "results_deep.csv")

    def submit_deep(rank: int, pilot_idx: int, p: Params):
        job_idx = rank
        deep_out_remote = _posix_join(remote_root, "deep", f"job{job_idx:06d}")
        return _run_one_remote(
            args,
            run_id=run_id,
            phase="deep",
            job_idx=job_idx,
            params=p,
            steps=args.deep_steps,
            initial_population=args.initial_population,
            max_population=args.deep_max_population,
            write_snapshot=True,
            deep_output_dir_remote=deep_out_remote,
            local_run_dir=local_run_dir,
            db_path=db_path,
            results_csv=deep_csv,
            results_header=results_header,
            progress_path=progress_path,
            progress_lock=progress_lock,
            progress_entries=progress_entries,
        )

    deep_targets = promising[: max(0, int(args.max_deep))]
    print(f"Remote deep scan: {len(deep_targets)} jobs -> {deep_csv}")
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {
            ex.submit(submit_deep, rank, pilot_idx, p): (rank, pilot_idx, p)
            for rank, (pilot_idx, p, _fp) in enumerate(deep_targets)
        }
        completed = 0
        for fut in as_completed(futs):
            completed += 1
            if completed % 5 == 0 or completed == len(deep_targets):
                print(f"Deep progress: {completed}/{len(deep_targets)}")

    print("Done (pilot + deep).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
