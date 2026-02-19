#!/usr/bin/env python3
"""Create a self-contained webviewer bundle.

Usage:
  python webviewer/make_bundle.py --sim-dir <run_dir> --out <bundle_dir>

The script copies the `webviewer/` files (this folder) into the bundle and
copies the sim run files into `<bundle>/run/` so the bundle can be served
statically (e.g. via `python -m http.server`).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


def copytree(src: Path, dst: Path, ignore=None) -> None:
    """Robust copytree wrapper that retries on transient IO errors."""
    attempts = 3
    for attempt: int in range(1, attempts + 1):
        try:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=ignore)
            return
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(0.5 * attempt)


def main() -> None:
    p = argparse.ArgumentParser(description="Make a webviewer bundle")
    p.add_argument("--sim-dir", required=True, help="Path to simulation run folder")
    p.add_argument("--out", required=True, help="Output bundle directory")
    args: argparse.Namespace = p.parse_args()

    root: Path = Path(__file__).resolve().parent
    sim_dir: Path = Path(args.sim_dir).resolve()
    out: Path = Path(args.out).resolve()

    if not sim_dir.exists():
        print("Sim dir does not exist:", sim_dir)
        sys.exit(2)

    # Create destination
    out_run: Path = out / "run"
    print(f"Creating bundle at {out}")
    out.mkdir(parents=True, exist_ok=True)

    # Copy the webviewer folder (this folder) into the bundle as the viewer
    viewer_dst: Path = out / "viewer"
    print(f"Copying viewer files to {viewer_dst}")

    # ignore compiled caches and target/build dirs
    def _ignore(dir, names):
        ignore_names: set[str] = {"__pycache__", ".git", "node_modules", "target", "build"}
        return [n for n in names if n in ignore_names]

    copytree(root, viewer_dst, ignore=_ignore)

    # Copy the simulation run data into out/run
    print(f"Copying run files from {sim_dir} to {out_run}")
    copytree(sim_dir, out_run, ignore=_ignore)

    print("Bundle creation complete")
    print(f"Serve with: cd {out} && python -m http.server 8001")


if __name__ == "__main__":
    main()
