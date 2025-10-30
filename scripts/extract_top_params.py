#!/usr/bin/env python3
"""Extract top-N parameter combinations by movement metric from a scan folder.

Usage: python scripts/extract_top_params.py --scan-dir <path> --metric max --top 10
If --scan-dir is not provided, the script finds the latest scanNNNN folder under
output/Parameter scans/data.
"""
import os
import argparse
import pandas as pd
from webviewer_helpers import iter_individuals_frames


def find_latest_scan(data_base):
    scans = [d for d in os.listdir(data_base) if d.startswith("scan") and os.path.isdir(os.path.join(data_base, d))]
    if not scans:
        return None
    scans_sorted = sorted(scans)
    return os.path.join(data_base, scans_sorted[-1])


def compute_metrics(scan_dir, results_csv):
    df = pd.read_csv(results_csv)
    sims = sorted([d for d in os.listdir(scan_dir) if os.path.isdir(os.path.join(scan_dir, d)) and d.startswith("sim")])

    mean_moves = []
    median_moves = []
    max_moves = []
    mean_norms = []

    for s in sims:
        sp = os.path.join(scan_dir, s)
        per_frame_means = []
        per_frame_norms = []
        all_abs = []
        for t, inds in iter_individuals_frames(sp):
            deltas = []
            for ind in inds:
                try:
                    ld = int(ind.get("last_delta", ind.get("last_delta", 0)))
                    deltas.append(abs(ld))
                    all_abs.append(abs(ld))
                except Exception:
                    pass
            if deltas:
                per_frame_means.append(float(sum(deltas)) / len(deltas))
                per_frame_norms.append((float(sum(deltas)) / len(deltas)) / float(len(deltas)))

        if per_frame_means:
            mean_moves.append(float(pd.Series(per_frame_means).mean()))
            median_moves.append(float(pd.Series(per_frame_means).median()))
            max_moves.append(float(pd.Series(all_abs).max()) if all_abs else 0.0)
            mean_norms.append(float(pd.Series(per_frame_norms).mean()) if per_frame_norms else 0.0)
        else:
            mean_moves.append(0.0)
            median_moves.append(0.0)
            max_moves.append(0.0)
            mean_norms.append(0.0)

    n = len(df)
    def pad(col, default=0.0):
        arr = list(col)
        if len(arr) < n:
            arr.extend([default] * (n - len(arr)))
        return arr[:n]

    df["mean_move"] = pad(mean_moves)
    df["median_move"] = pad(median_moves)
    df["max_move"] = pad(max_moves)
    df["mean_norm_by_pop"] = pad(mean_norms)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", help="Path to scan directory (e.g. output/Parameter scans/data/scan0015)")
    parser.add_argument("--metric", choices=["mean", "median", "max", "survived"], default="max")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    data_base = os.path.join(os.path.normpath("."), "output", "Parameter scans", "data")

    scan_dir = args.scan_dir
    if not scan_dir:
        scan_dir = find_latest_scan(data_base)
    if not scan_dir or not os.path.isdir(scan_dir):
        print("No scan directory found. Looked under", data_base)
        return

    results_csv = os.path.join(scan_dir, "scan_extended_results.csv")
    if not os.path.exists(results_csv):
        print("Results CSV not found in", scan_dir)
        return

    df = compute_metrics(scan_dir, results_csv)

    metric_col = "max_move" if args.metric == "max" else ("mean_move" if args.metric == "mean" else ("median_move" if args.metric == "median" else "survived"))

    topn = df.sort_values(by=metric_col, ascending=False).head(args.top)

    out_csv = os.path.join(scan_dir, f"top_{args.top}_{metric_col}.csv")
    topn.to_csv(out_csv, index=False)

    print(f"Wrote top {args.top} combos by {metric_col} to {out_csv}")
    print(topn[["maintenance_cost_per_node","reproduction_threshold","reproduction_cost","resource_scale","env_time_speed",metric_col]].to_string(index=False))


if __name__ == "__main__":
    main()
