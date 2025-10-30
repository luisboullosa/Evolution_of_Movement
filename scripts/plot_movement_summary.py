#!/usr/bin/env python3
"""
Plot movement summary CSV created by `scripts/scan_movement_analyze.py`.
Saves `movement_summary.png` into the same scan directory.
"""
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("scan_dir", help="scanNNNN directory containing movement_summary.csv")
args = parser.parse_args()
scan_dir = args.scan_dir
csvp = os.path.join(scan_dir, "movement_summary.csv")
if not os.path.exists(csvp):
    print("movement_summary.csv not found in", scan_dir)
    raise SystemExit(1)

sims = []
mean_frame_avgs = []
median_frame_avgs = []
mean_norms = []
surv = []
with open(csvp, "r", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        sims.append(row.get("sim"))
        try:
            mean_frame_avgs.append(float(row.get("mean_frame_avg_move", row.get("mean_frame_avg_move", row.get("avg_abs_move", "0")))))
        except Exception:
            mean_frame_avgs.append(0.0)
        try:
            median_frame_avgs.append(float(row.get("median_frame_avg_move", "0")))
        except Exception:
            median_frame_avgs.append(0.0)
        try:
            mean_norms.append(float(row.get("mean_norm_by_pop", "0")))
        except Exception:
            mean_norms.append(0.0)
        surv.append(int(row.get("survived", "0")))

# simple bar plot sorted by avg movement
order = np.argsort(mean_frame_avgs)[::-1]
ord_avgs = [mean_frame_avgs[i] for i in order]
ord_sims = [sims[i] for i in order]
ord_surv = [surv[i] for i in order]

plt.figure(figsize=(10, 4))
cols = ["tab:blue" if v else "tab:gray" for v in ord_surv]
plt.bar(range(len(ord_avgs)), ord_avgs, color=cols)
plt.xticks(range(len(ord_avgs)), ord_sims, rotation=45, fontsize=8)
plt.ylabel("mean abs move per frame")
plt.title("Movement summary (sorted)")
plt.tight_layout()
out = os.path.join(scan_dir, "movement_summary.png")
plt.savefig(out, dpi=150)
print("Wrote", out)
