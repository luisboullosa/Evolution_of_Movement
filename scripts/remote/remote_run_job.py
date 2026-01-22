#!/usr/bin/env python3
"""Remote job runner helper.

This script is intended to run *on the remote machine* (e.g., Orange Pi) and
executes a single Rust simulator run, printing a final JSON line with the
parsed summary plus optional output directory.

It exists so the local orchestrator can run a stable command over SSH without
embedding large Python snippets.

Contract:
- It prints all simulator output as-is.
- It prints one final line that begins with "REMOTE_JOB_RESULT " followed by JSON.

The orchestrator parses that final line.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any


def _parse_last_summary_json(text: str) -> dict[str, Any] | None:
    last_obj: dict[str, Any] | None = None
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or not s.startswith("{"):
            continue
        if "final_population" not in s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict) and "final_population" in obj:
            last_obj = obj
    return last_obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True, help="Repo root on the remote machine")
    ap.add_argument("--bin", default="evo_movement_rust", help="Cargo bin name")
    ap.add_argument("--manifest-path", default="rust/Cargo.toml", help="Manifest path relative to repo-dir")
    ap.add_argument("--config", required=True, help="Config JSON path on remote machine")
    ap.add_argument("--release", action="store_true", help="Use cargo --release")
    ap.add_argument("--output-run-dir", default="", help="If set, pass --output-run-dir to simulator")
    ap.add_argument("--no-write-outputs", action="store_true", help="Pass --no-write-outputs to simulator")
    ap.add_argument("--replay-snapshot", default="", help="If set, pass --replay-snapshot to simulator")
    args = ap.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    manifest = args.manifest_path

    # Source cargo env if it exists (for non-interactive SSH sessions)
    cargo_env = os.path.expanduser("~/.cargo/env")
    shell_prefix = f". {cargo_env} && " if os.path.exists(cargo_env) else ""

    cmd = [
        "cargo",
        "run",
        "--quiet",
    ]
    if args.release:
        cmd.append("--release")
    cmd.extend(
        [
            "--manifest-path",
            os.path.join(repo_dir, manifest),
            "--bin",
            args.bin,
            "--",
            "--config-file",
            args.config,
        ]
    )
    if args.no_write_outputs:
        cmd.append("--no-write-outputs")
    if args.output_run_dir:
        cmd.extend(["--output-run-dir", args.output_run_dir])
    if args.replay_snapshot:
        cmd.extend(["--replay-snapshot", args.replay_snapshot])

    # Build shell command string to source cargo env
    import shlex

    cmd_str = shell_prefix + " ".join(shlex.quote(str(c)) for c in cmd)

    # Run and stream output, but also capture so we can parse summary at end.
    proc = subprocess.Popen(
        cmd_str,
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        sys.stdout.write(raw)
        sys.stdout.flush()
        captured.append(raw)
    rc = proc.wait()
    out_text = "".join(captured)
    summary = _parse_last_summary_json(out_text)

    result = {
        "ok": rc == 0,
        "return_code": rc,
        "summary": summary,
        "output_run_dir": args.output_run_dir or None,
    }
    sys.stdout.write("REMOTE_JOB_RESULT " + json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
