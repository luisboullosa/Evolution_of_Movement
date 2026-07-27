#!/usr/bin/env python3
"""Serve fitness landscape output with an API to rebuild probe timelines from the viewer."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_probe_timelines.ps1"
_rebuild_lock = threading.Lock()


def run_probe_timeline_build(out_dir: Path, sweep_output_modes: bool = True) -> dict:
    if not BUILD_SCRIPT.is_file():
        return {"ok": False, "error": f"Missing script: {BUILD_SCRIPT}"}

    cmd = [
        "pwsh",
        "-NoProfile",
        "-File",
        str(BUILD_SCRIPT),
        f"-OutDir={out_dir}",
    ]
    if sweep_output_modes:
        cmd.append("-SweepOutputModes")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Rebuild timed out after 15 minutes"}
    except FileNotFoundError:
        return {"ok": False, "error": "pwsh not found — install PowerShell 7+"}

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    combined = (stdout + "\n" + stderr).strip()
    tail = combined.splitlines()[-8:] if combined else []

    summary = None
    for line in reversed(combined.splitlines()):
        m = re.search(r"probe timelines: wrote (\d+) · skipped (\d+)", line)
        if m:
            summary = {"written": int(m.group(1)), "skipped": int(m.group(2))}
            break

    if proc.returncode != 0:
        return {
            "ok": False,
            "error": f"build_probe_timelines failed (exit {proc.returncode})",
            "tail": tail,
        }

    return {
        "ok": True,
        "summary": summary,
        "tail": tail,
    }


def make_handler(out_dir: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(out_dir), **kwargs)

        def log_message(self, fmt, *args):
            if str(args[0]).startswith("POST /api/"):
                super().log_message(fmt, *args)

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            super().end_headers()

        def do_GET(self):
            if self.path == "/api/health":
                self._json(200, {
                    "ok": True,
                    "rebuild_probe_timelines": True,
                    "out_dir": str(out_dir),
                })
                return
            super().do_GET()

        def do_POST(self):
            if self.path != "/api/rebuild-probe-timelines":
                self.send_error(404)
                return

            if not _rebuild_lock.acquire(blocking=False):
                self._json(409, {"ok": False, "error": "Rebuild already in progress"})
                return

            try:
                result = run_probe_timeline_build(out_dir, sweep_output_modes=True)
                status = 200 if result.get("ok") else 500
                self._json(status, result)
            finally:
                _rebuild_lock.release()

        def _json(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve fitness landscape HTML and expose probe timeline rebuild API.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(REPO_ROOT / "output" / "evolution_hunt_staged_all"),
        help="Output directory to serve (default: output/evolution_hunt_staged_all)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)

    out_dir = Path(args.directory).resolve()
    if not out_dir.is_dir():
        print(f"error: not a directory: {out_dir}", file=sys.stderr)
        return 1

    handler = make_handler(out_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {out_dir}")
    print(f"Open http://{args.host}:{args.port}/fitness_landscape_heatmap.html")
    print("POST /api/rebuild-probe-timelines — rebuild probe_timelines/ (same as build_probe_timelines.ps1 -SweepOutputModes)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
