import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description="Plot focused parameter sweep heatmap")
parser.add_argument("--in", dest="infile", default=None, help="Input CSV path")
parser.add_argument("--out", dest="outfile", default=None, help="Output image path")
args = parser.parse_args()

output_dir = os.path.join(os.path.normpath("."), "output", "Parameter scans")
os.makedirs(output_dir, exist_ok=True)

csv_default = os.path.join(output_dir, "scan_focused_results.csv")
if args.infile:
    csvpath = os.path.normpath(args.infile)
else:
    csvpath = csv_default
    if not os.path.exists(csvpath):
        csvpath = os.path.normpath("scan_focused_results.csv")
        if not os.path.exists(csvpath):
            print("scan_focused_results.csv not found at", os.path.relpath(csv_default))
            raise SystemExit(1)

out_default = os.path.join(output_dir, "scan_focused_heatmap.png")
out = os.path.normpath(args.outfile) if args.outfile else out_default

movement_vals = []
maintenance_vals = []
data = {}
with open(csvpath, "r", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        mv = float(row["movement_cost_factor"])
        mc = float(row["maintenance_cost_per_node"])
        try:
            et = int(row["extinction_time"])
        except Exception:
            et = int(float(row["extinction_time"]))
        surv = int(row.get("survived", "0"))
        movement_vals.append(mv)
        maintenance_vals.append(mc)
        data[(mv, mc)] = (et, surv)

movement_vals = sorted(set(movement_vals))
maintenance_vals = sorted(set(maintenance_vals))

# build matrix with NaN for missing cells
mat = np.full((len(maintenance_vals), len(movement_vals)), np.nan, dtype=float)
survivors = []
max_time = 0
count = 0
for i, mc in enumerate(maintenance_vals):
    for j, mv in enumerate(movement_vals):
        if (mv, mc) in data:
            et, surv = data[(mv, mc)]
            mat[i, j] = et
            count += 1
            if surv:
                survivors.append((j, i, mv, mc))
            if et > max_time:
                max_time = et

# Cap values for colormap
vmax = max_time if max_time > 0 else 1

plt.figure(figsize=(9, 5.5))
cmap = plt.get_cmap("magma_r").copy()
cmap.set_bad(color="#dddddd")
im = plt.imshow(
    mat,
    origin="lower",
    cmap=cmap,
    aspect="auto",
    vmin=0,
    vmax=vmax,
    interpolation="nearest",
)
cb = plt.colorbar(im)
cb.set_label("extinction time (timesteps)")

# ticks with formatted floats
plt.xticks(range(len(movement_vals)), [f"{v:.6g}" for v in movement_vals], rotation=45)
plt.yticks(range(len(maintenance_vals)), [f"{v:.6g}" for v in maintenance_vals])
plt.xlabel("movement_cost_factor")
plt.ylabel("maintenance_cost_per_node")
plt.title("Focused parameter sweep: extinction time (50k max)")

# overlay survivors
if survivors:
    xs = [s[0] for s in survivors]
    ys = [s[1] for s in survivors]
    plt.scatter(xs, ys, s=80, facecolors="none", edgecolors="white", linewidths=1.6)
    plt.scatter(xs, ys, s=18, facecolors="black")

plt.annotate(
    f"survivors: {len(survivors)} / points: {count}",
    xy=(0.98, 0.02),
    xycoords="axes fraction",
    ha="right",
    va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
)

plt.tight_layout()
plt.savefig(out, dpi=200)
print("Wrote", os.path.relpath(out))
