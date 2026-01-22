#!/usr/bin/env python3
"""Select a simulation or snapshot and launch a viewer.

Features:
- Lists simulations with env.jsonl/individuals.jsonl under search roots.
- Lists population snapshot CBOR files under the same roots.
- Optional build_id filter (reads build_id.txt or config/run_config when present).
- If a snapshot is selected, runs a 1000-step replay (configurable) with
  --write-outputs into a fresh run folder, then launches the viewer.

Examples:
  python /select_and_visualize.py --search output/visualize \
      --viewer both --steps 1000

  # Replay from a specific snapshot and open population viewer only
  python webviewer/select_and_visualize.py --select 3 --viewer population \
      --steps 1000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_SEARCH = [
    ROOT.parent / "output" / "visualize",
    ROOT.parent / "output" / "prod_runs",
    ROOT.parent / "output" / "remote_scan",
    ROOT.parent / "output" / "scan_population",
]


def read_build_id(run_dir: Path) -> Optional[str]:
    candidates = [
        "build_id.txt",
        "config.json",
        "run_config.json",
        "sim_config.json",
        "scan_config.json",
    ]
    for name in candidates:
        p = run_dir / name
        if not p.exists():
            continue
        if p.suffix == ".txt":
            try:
                return p.read_text().strip()
            except Exception:
                continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            for key in ("build_id", "BUILD_ID", "buildId"):
                v = data.get(key)
                if isinstance(v, str):
                    return v
    return None


def discover_runs(search_dirs: List[Path]) -> List[Dict[str, object]]:
    runs = []
    seen = set()
    for base in search_dirs:
        if not base.exists():
            continue
        for env_file in base.rglob("env.jsonl"):
            run_dir = env_file.parent
            if any(part in {"__pycache__", "target", ".git"} for part in run_dir.parts):
                continue
            if not (run_dir / "individuals.jsonl").exists():
                continue
            if run_dir in seen:
                continue
            seen.add(run_dir)
            runs.append(
                {
                    "kind": "run",
                    "path": run_dir,
                    "label": run_dir.name,
                    "build_id": read_build_id(run_dir),
                }
            )
    runs.sort(key=lambda r: str(r["path"]))
    return runs


def discover_snapshots(search_dirs: List[Path]) -> List[Dict[str, object]]:
    snaps = []
    seen = set()
    for base in search_dirs:
        if not base.exists():
            continue
        for snap in base.rglob("*snapshot*.cbor"):
            if any(part in {"__pycache__", "target", ".git"} for part in snap.parts):
                continue
            if snap in seen:
                continue
            seen.add(snap)
            snaps.append(
                {
                    "kind": "snapshot",
                    "path": snap,
                    "label": snap.stem,
                    "build_id": read_build_id(snap.parent),
                }
            )
    snaps.sort(key=lambda r: str(r["path"]))
    return snaps


def choose(entries: List[Dict[str, object]], auto: Optional[int]) -> Dict[str, object]:
    if not entries:
        print("No simulations or snapshots found under the search paths.")
        sys.exit(1)
    if auto is not None:
        if auto < 0 or auto >= len(entries):
            print(f"Invalid selection index {auto}; valid range 0..{len(entries)-1}")
            sys.exit(1)
        return entries[auto]

    print("Available entries:")
    for i, e in enumerate(entries):
        bid = e.get("build_id") or "?"
        print(f"[{i:02d}] {e['kind']:<9} build_id={bid:<8}  {e['path']}")
    choice = input("Select entry number: ").strip()
    try:
        idx = int(choice)
    except Exception:
        print("Invalid selection")
        sys.exit(1)
    if idx < 0 or idx >= len(entries):
        print("Selection out of range")
        sys.exit(1)
    return entries[idx]


def run_replay(
    snapshot: Path,
    out_dir: Path,
    steps: int,
    env_seed: Optional[int],
    profile: str,
    burn_in: int,
    randomize_start_positions: Optional[bool],
    output_budget_mb: Optional[float],
    output_budget_hard_stop: bool,
    output_budget_warn_fraction: Optional[float],
    write_individual_deltas: bool,
    delta_full_interval: Optional[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cargo",
        "run",
        f"--{profile}",
        "--bin",
        "evo_movement_rust",
        "--",
        "--replay-snapshot",
        str(snapshot),
        "--steps",
        str(steps),
        "--report-interval",
        "1",
        "--burn-in",
        str(burn_in),
        "--write-outputs",
        "--no-progress",
        "--output-run-dir",
        str(out_dir),
    ]
    if env_seed is not None:
        cmd += ["--replay-env-seed", str(env_seed)]
    if randomize_start_positions is True:
        cmd += ["--replay-randomize-start-positions"]
    elif randomize_start_positions is False:
        cmd += ["--replay-no-randomize-start-positions"]
    if output_budget_mb is not None:
        cmd += ["--output-budget-mb", str(output_budget_mb)]
    if output_budget_warn_fraction is not None:
        cmd += ["--output-budget-warn-fraction", str(output_budget_warn_fraction)]
    if output_budget_hard_stop:
        cmd += ["--output-budget-hard-stop"]
    if write_individual_deltas:
        cmd += ["--write-individual-deltas"]
        if delta_full_interval is not None:
            cmd += ["--delta-full-interval", str(delta_full_interval)]
    print("Running replay:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT.parent / "rust", check=True)


def launch_viewers(run_path: Path, viewer: str) -> None:
    targets = []
    if viewer in ("population", "both"):
        targets.append((ROOT / "visualize_population.py", "population"))
    if viewer in ("individual", "both"):
        targets.append((ROOT / "visualize_individual.py", "individual"))
    for script, label in targets:
        cmd = [sys.executable, str(script), str(run_path)]
        print(f"Launching {label} viewer -> {cmd}")
        subprocess.Popen(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select and visualize simulations")
    ap.add_argument(
        "--search",
        nargs="*",
        default=None,
        help="Directories to search for runs and snapshots",
    )
    ap.add_argument("--build-id", help="Filter entries by build_id", default=None)
    ap.add_argument("--select", type=int, default=None, help="Select entry index without prompt")
    ap.add_argument("--viewer", choices=["population", "individual", "both"], default="both")
    ap.add_argument("--steps", type=int, default=1000, help="Steps when replaying a snapshot")
    ap.add_argument("--env-seed", type=int, default=None, help="Override replay env seed")
    ap.add_argument("--burn-in", type=int, default=0, help="Burn-in steps for replay")
    ap.add_argument(
        "--randomize-start-positions",
        action="store_true",
        help="Shuffle start positions when initializing population (if supported)",
    )
    ap.add_argument(
        "--no-randomize-start-positions",
        action="store_true",
        help="Disable position randomization (if supported)",
    )
    ap.add_argument(
        "--output-budget-mb",
        type=float,
        default=600.0,
        help="Projected output budget in MiB (<=0 disables budget)",
    )
    ap.add_argument(
        "--output-budget-warn-fraction",
        type=float,
        default=0.8,
        help="Warn fraction of budget (0..1)",
    )
    ap.add_argument(
        "--write-individual-deltas",
        action="store_true",
        help="Write individuals.delta.jsonl instead of full individuals.jsonl",
    )
    ap.add_argument(
        "--no-write-individual-deltas",
        action="store_true",
        help="Disable delta writing (use full individuals.jsonl)",
    )
    ap.add_argument(
        "--delta-full-interval",
        type=int,
        default=25,
        help="Full resync frame every N steps in delta mode",
    )
    ap.add_argument(
        "--output-budget-hard-stop",
        action="store_true",
        default=True,
        help="Abort replay if projected outputs exceed budget",
    )
    ap.add_argument(
        "--no-output-budget-hard-stop",
        action="store_true",
        help="Allow budget warnings without aborting",
    )
    ap.add_argument(
        "--profile",
        choices=["release", "debug"],
        default="release",
        help="Cargo profile to use for replays",
    )
    ap.add_argument(
        "--out-base",
        default=None,
        help="Base directory for replay outputs (default: output/visualize)",
    )
    args = ap.parse_args()

    search_dirs = [Path(p).resolve() for p in args.search] if args.search else DEFAULT_SEARCH
    out_base = Path(args.out_base).resolve() if args.out_base else (ROOT.parent / "output" / "visualize")

    runs = discover_runs(search_dirs)
    snaps = discover_snapshots(search_dirs)
    entries = runs + snaps
    if args.build_id:
        entries = [e for e in entries if e.get("build_id") == args.build_id]
    selection = choose(entries, args.select)

    if selection["kind"] == "run":
        run_path: Path = selection["path"]  # type: ignore
        launch_viewers(run_path, args.viewer)
        return

    snapshot_path: Path = selection["path"]  # type: ignore
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / f"selector_{snapshot_path.stem}_{stamp}"
    try:
        rand_pos: Optional[bool]
        if args.randomize_start_positions and args.no_randomize_start_positions:
            rand_pos = None  # conflicting; let binary default
        elif args.randomize_start_positions:
            rand_pos = True
        elif args.no_randomize_start_positions:
            rand_pos = False
        else:
            rand_pos = None

        if args.no_write_individual_deltas:
            use_deltas = False
        elif args.write_individual_deltas:
            use_deltas = True
        else:
            use_deltas = True  # default on to reduce footprint

        run_replay(
            snapshot_path,
            out_dir,
            args.steps,
            args.env_seed,
            args.profile,
            args.burn_in,
            rand_pos,
            args.output_budget_mb,
            not args.no_output_budget_hard_stop,
            args.output_budget_warn_fraction,
            use_deltas,
            args.delta_full_interval,
        )
    except subprocess.CalledProcessError as e:
        print("Replay failed:", e)
        sys.exit(e.returncode)
    launch_viewers(out_dir, args.viewer)


if __name__ == "__main__":
    main()
