#!/usr/bin/env python3
"""
Run deep sims (long steps) for promising parameter sets produced by scan_rust_population_scan.
Reads promising.csv and results_short.csv from a jobs directory to reconstruct full parameter tuples,
then runs the Rust simulator for each tuple with a longer step count.
"""

import argparse
import csv
import json
import os
import tempfile

from worker_pool import run_param_sweep


def load_results(results_path):
    rows = []
    with open(results_path, "r", encoding="utf-8") as rf:
        rdr = csv.DictReader(rf)
        for row in rdr:
            rows.append(row)
    return rows


def build_index(rows):
    idx = {}
    for row in rows:
        try:
            key = (
                float(row.get("movement_cost_factor") or 0),
                float(row.get("maintenance_cost_per_node") or 0),
                float(row.get("resource_scale") or 0),
                int(float(row.get("grid_width") or 0)),
            )
            idx.setdefault(key, []).append(row)
        except Exception:
            continue
    return idx


def load_promising(prom_path, index):
    tuples = []
    # read and ignore header line
    with open(prom_path, "r", encoding="utf-8") as pf:
        pf.readline()
        for line in pf:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            mv, mc, rs, gw, fp = parts[:5]
            key = (float(mv), float(mc), float(rs), int(float(gw)))
            candidates = index.get(key, [])
            match = None
            for row in candidates:
                try:
                    if int(float(row.get("final_population") or 0)) == int(float(fp)):
                        match = row
                        break
                except Exception:
                    continue
            if match is None and candidates:
                match = candidates[0]
            if match is None:
                continue
            try:
                tpl = (
                    float(match.get("movement_cost_factor") or 0),
                    float(match.get("maintenance_cost_per_node") or 0),
                    float(match.get("reproduction_threshold") or 0),
                    float(match.get("reproduction_cost") or 0),
                    float(match.get("env_time_speed") or 0),
                    float(match.get("death_risk") or 0),
                    float(match.get("resource_scale") or 0),
                    int(float(match.get("grid_width") or 0)),
                )
                tuples.append(tpl)
            except Exception:
                continue
    return tuples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jobs-dir", required=True, help="Path to jobs folder containing promising.csv and results_short.csv"
    )
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--initial-population", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bin", dest="bin_name", default="evo_movement_rust")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--max-population", type=int, default=20000)
    ap.add_argument("--write-outputs", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200, help="Per-job timeout (seconds)")
    ap.add_argument("--results-name", default="deep_results.csv")
    args = ap.parse_args()

    if not args.release and not args.debug:
        args.release = True
    if args.debug:
        args.release = False

    jobs_dir = os.path.normpath(args.jobs_dir)
    results_path = os.path.join(jobs_dir, "results_short.csv")
    prom_path = os.path.join(jobs_dir, "promising.csv")
    if not os.path.exists(results_path) or not os.path.exists(prom_path):
        raise SystemExit("Missing results_short.csv or promising.csv in jobs dir")

    rows = load_results(results_path)
    index = build_index(rows)
    deep_pairs = load_promising(prom_path, index)
    if not deep_pairs:
        raise SystemExit("No deep pairs reconstructed from promising.csv")

    deep_run_dir = jobs_dir
    deep_results_csv = os.path.join(jobs_dir, args.results_name)
    deep_cfg_paths = {}

    for i, tpl in enumerate(deep_pairs):
        mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, gw = tpl
        cfg = {
            "movement_cost_factor": mv,
            "maintenance_cost_per_node": mc,
            "reproduction_threshold": repro_thr,
            "reproduction_cost": repro_cost,
            "env_time_speed": env_time_speed,
            "death_risk": death_risk,
            "grid_width": gw,
            "resource_scale": rscales,
            "steps": int(args.steps),
            "initial_population": int(args.initial_population),
            "show_progress": False,
            "write_frames": False,
            "write_outputs": bool(args.write_outputs),
            "write_config": False,
            "write_snapshot": False,
            "report_interval": int(args.steps) + 1,
            "output_dir_base": deep_run_dir,
            "sim_prefix": f"prom{i}_",
            "max_population": int(args.max_population),
        }
        tf = tempfile.NamedTemporaryFile(
            prefix=f"deep_prom_{i}_", suffix=".json", delete=False, mode="w", encoding="utf-8"
        )
        json.dump(cfg, tf)
        tf.flush()
        tf.close()
        deep_cfg_paths[i] = tf.name

    def deep_cmd(job):
        idx = job[0]
        cfgpath = deep_cfg_paths[idx]
        cmd = ["cargo", "run", "--quiet"]
        if args.release:
            cmd.append("--release")
        cmd.extend(
            [
                "--manifest-path",
                os.path.normpath("rust/Cargo.toml"),
                "--bin",
                args.bin_name,
                "--",
                "--config-file",
                cfgpath,
            ]
        )
        if not args.write_outputs:
            cmd.append("--no-write-outputs")
        return cmd

    def deep_result_builder(job, extinct, out_text):
        idx, tpl = job
        mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, gw = tpl
        final_pop = 0
        prop_moved = 0.0
        if out_text:
            for line in (out_text or "").splitlines():
                s = line.strip()
                if not s or not s.startswith("{"):
                    continue
                try:
                    obj = json.loads(s)
                    if "final_population" in obj:
                        final_pop = int(obj.get("final_population") or 0)
                    if "prop_moved" in obj:
                        prop_moved = float(obj.get("prop_moved") or 0.0)
                        break
                except Exception:
                    continue
        return [mv, mc, repro_thr, repro_cost, env_time_speed, death_risk, rscales, gw, final_pop, prop_moved]

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

    try:
        run_param_sweep(
            deep_pairs,
            deep_cmd,
            deep_results_csv,
            deep_header,
            workers=max(1, args.workers),
            timeout=int(args.timeout),
            result_row_builder=deep_result_builder,
            stream_live_output=True,
        )
    finally:
        for p in list(deep_cfg_paths.values()):
            try:
                os.remove(p)
            except Exception:
                pass


if __name__ == "__main__":
    main()
