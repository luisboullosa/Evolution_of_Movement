#!/usr/bin/env python3
"""Build a static, upload-ready dashboard bundle for simple web hosting.

This script converts simulator replay files (`env.jsonl`, `individuals.jsonl`) into
a compact JSON payload and copies a lightweight HTML/JS dashboard template that
runs fully in the browser (no backend required).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _select_evenly_spaced_indices(total: int, target: int) -> List[int]:
    if total <= 0:
        return []
    if target <= 0 or total <= target:
        return list(range(total))
    if target == 1:
        return [0]
    return [round(i * (total - 1) / (target - 1)) for i in range(target)]


def _infer_grid_width(config_path: Path | None, env_rows: List[dict]) -> int:
    if config_path and config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            width = int(cfg.get("grid_width", 0))
            if width > 0:
                return width
        except Exception:
            pass

    for row in env_rows:
        resources = row.get("resources")
        if isinstance(resources, list) and resources:
            side = int(math.isqrt(len(resources)))
            if side > 0 and side * side == len(resources):
                return side

    return 50


def _quantize_resources(values: Iterable[float]) -> List[int]:
    out: List[int] = []
    for value in values:
        try:
            numeric = float(value)
        except Exception:
            numeric = 0.0
        if numeric < 0:
            numeric = 0.0
        if numeric > 65535.0:
            numeric = 65535.0
        out.append(int(round(numeric)))
    return out


def _compress_frame(individuals: List[dict], max_individuals: int) -> dict:
    if not isinstance(individuals, list):
        individuals = []

    ranked = sorted(
        individuals,
        key=lambda row: float(row.get("resources", 0.0)),
        reverse=True,
    )
    if max_individuals > 0:
        ranked = ranked[:max_individuals]

    positions: List[int] = []
    for row in ranked:
        try:
            positions.append(int(row.get("pos", 0)))
        except Exception:
            positions.append(0)

    pop = len(individuals)
    avg_resources = 0.0
    if pop > 0:
        avg_resources = sum(float(row.get("resources", 0.0)) for row in individuals) / pop

    return {
        "positions": positions,
        "population": pop,
        "avg_individual_resources": round(avg_resources, 4),
    }


def build_bundle(
    run_dir: Path,
    out_dir: Path,
    max_frames: int,
    max_individuals: int,
) -> Path:
    env_path = run_dir / "env.jsonl"
    individuals_path = run_dir / "individuals.jsonl"
    config_path = run_dir / "config.json"

    if not env_path.exists() or not individuals_path.exists():
        missing = [str(p.name) for p in (env_path, individuals_path) if not p.exists()]
        raise FileNotFoundError(f"Missing required input files in run dir: {', '.join(missing)}")

    env_rows = _read_jsonl(env_path)
    ind_rows = _read_jsonl(individuals_path)

    env_by_t: Dict[int, dict] = {}
    for row in env_rows:
        try:
            env_by_t[int(row.get("t", 0))] = row
        except Exception:
            continue

    ind_by_t: Dict[int, dict] = {}
    for row in ind_rows:
        try:
            ind_by_t[int(row.get("t", 0))] = row
        except Exception:
            continue

    all_ts = sorted(set(env_by_t.keys()) & set(ind_by_t.keys()))
    if not all_ts:
        raise RuntimeError("No shared timesteps found between env.jsonl and individuals.jsonl")

    selected_ts = [all_ts[i] for i in _select_evenly_spaced_indices(len(all_ts), max_frames)]
    grid_width = _infer_grid_width(config_path, env_rows)

    frames: List[dict] = []
    observed_max_resource = 0
    for t in selected_ts:
        env_row = env_by_t[t]
        ind_row = ind_by_t[t]

        resources = _quantize_resources(env_row.get("resources", []))
        if resources:
            observed_max_resource = max(observed_max_resource, max(resources))

        compressed = _compress_frame(ind_row.get("individuals", []), max_individuals)
        frames.append(
            {
                "t": t,
                "resources": resources,
                "positions": compressed["positions"],
                "population": compressed["population"],
                "avg_individual_resources": compressed["avg_individual_resources"],
            }
        )

    template_dir = Path(__file__).resolve().parents[1] / "webviewer" / "hostinger_bundle_template"
    if not template_dir.exists():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(template_dir, out_dir)

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "source_run": str(run_dir),
            "grid_width": grid_width,
            "frame_count": len(frames),
            "max_resource": max(1, observed_max_resource),
            "max_rendered_individuals_per_frame": max_individuals,
        },
        "frames": frames,
    }

    out_json = data_dir / "dashboard_data.json"
    out_json.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    return out_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package static movement dashboard for Hostinger")
    parser.add_argument("--run-dir", required=True, help="Path to a run folder containing env.jsonl and individuals.jsonl")
    parser.add_argument("--out", required=True, help="Output folder for the upload-ready bundle")
    parser.add_argument("--max-frames", type=int, default=180, help="Max frames to include after downsampling")
    parser.add_argument(
        "--max-individuals",
        type=int,
        default=3500,
        help="Max individuals rendered per frame (top by resources)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a .zip archive of the generated bundle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out).resolve()

    out_json = build_bundle(
        run_dir=run_dir,
        out_dir=out_dir,
        max_frames=max(1, args.max_frames),
        max_individuals=max(1, args.max_individuals),
    )

    print(f"Created static bundle at: {out_dir}")
    print(f"Data file: {out_json}")
    if args.zip:
        archive_base = str(out_dir)
        archive_path = shutil.make_archive(archive_base, "zip", root_dir=out_dir)
        print(f"Zip archive: {archive_path}")
    print("Upload the full bundle folder contents to your web host.")


if __name__ == "__main__":
    main()
