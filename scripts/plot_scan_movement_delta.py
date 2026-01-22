import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Plot median movement delta heatmap")
parser.add_argument(
    "--annotate",
    action="store_true",
    help="Annotate numeric median values on each cell",
)
parser.add_argument("--in", dest="infile", default=None, help="Input CSV path")
parser.add_argument("--out", dest="outfile", default=None, help="Output image path")
parser.add_argument("--burn-in", dest="burn_in", type=int, default=None, help="Burn-in steps used for the scan")
args = parser.parse_args()

# optional burn-in value to display in the plot title
parser.add_argument("--burn-in", dest="burn_in", type=int, default=None, help="Burn-in steps used for the scan")

output_dir = os.path.join(os.path.normpath("."), "output", "Parameter scans")
os.makedirs(output_dir, exist_ok=True)

csv_default = os.path.join(output_dir, "scan_movement_delta_results.csv")
csvpath = os.path.normpath(args.infile) if args.infile else csv_default
if not os.path.exists(csvpath):
    csvpath = os.path.normpath("scan_movement_delta_results.csv")
    if not os.path.exists(csvpath):
        print("scan_movement_delta_results.csv not found at", os.path.relpath(csv_default))
        raise SystemExit(1)

movement_vals = []
maintenance_vals = []
data = {}
with open(csvpath, "r", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            mv = float(row["movement_cost_factor"])
            mc = float(row["maintenance_cost_per_node"])
        except Exception:
            continue
        md = None
        note = row.get("note", "")
        if row.get("median_delta", "") != "":
            try:
                md = float(row["median_delta"])
            except Exception:
                md = None
        movement_vals.append(mv)
        maintenance_vals.append(mc)
        data[(mv, mc)] = (md, note)

movement_vals = sorted(set(movement_vals))
maintenance_vals = sorted(set(maintenance_vals))

# build matrix with NaN for missing cells
mat = np.full((len(maintenance_vals), len(movement_vals)), np.nan, dtype=float)
notes = {}
count = 0
for i, mc in enumerate(maintenance_vals):
    for j, mv in enumerate(movement_vals):
        if (mv, mc) in data:
            md, note = data[(mv, mc)]
            if md is not None:
                mat[i, j] = md
                count += 1
            else:
                notes[(j, i)] = note

vmax = np.nanmax(mat) if np.isfinite(np.nanmax(mat)) else 1.0

plt.figure(figsize=(9, 5.5))
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad(color="#dddddd")
im = plt.imshow(
    mat,
    origin="lower",
    cmap=cmap,
    aspect="auto",
    interpolation="nearest",
    vmin=0,
    vmax=vmax,
)
cb = plt.colorbar(im)

cb.set_label("median |delta| (cells)")

plt.xticks(range(len(movement_vals)), [f"{v:.6g}" for v in movement_vals], rotation=45)
plt.yticks(range(len(maintenance_vals)), [f"{v:.6g}" for v in maintenance_vals])
plt.xlabel("movement_cost_factor")
plt.ylabel("maintenance_cost_per_node")
title_burn = f" after burn-in {args.burn_in} steps" if args.burn_in is not None else ""
plt.title("Median absolute movement" + title_burn)

# annotate missing / notes
for (j, i), note in notes.items():
    if not note:
        lbl = "NA"
    else:
        lbl = note
    plt.text(j, i, "x", color="red", ha="center", va="center", fontsize=8)

plt.annotate(
    f"points: {count}",
    xy=(0.98, 0.02),
    xycoords="axes fraction",
    ha="right",
    va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
)

out = args.outfile if args.outfile else os.path.join(output_dir, "scan_movement_delta_heatmap.png")
# overlay numeric annotations if requested
if args.annotate:
    # choose text color based on luminance of the colormap at the cell value
    norm = plt.Normalize(vmin=0, vmax=vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                rgba = cmap(norm(val))
                # luminance approximation
                r, g, b, a = rgba
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                txtcol = "white" if lum < 0.5 else "black"
                plt.text(
                    j,
                    i,
                    f"{val:.2g}",
                    ha="center",
                    va="center",
                    color=txtcol,
                    fontsize=8,
                )

plt.tight_layout()
plt.savefig(out, dpi=200)
print("Wrote", out)
