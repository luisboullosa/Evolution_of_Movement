import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Plot extinction time heatmap")
parser.add_argument("--in", dest="infile", default=None, help="Input CSV path")
parser.add_argument("--out", dest="outfile", default=None, help="Output image path")
args: argparse.Namespace = parser.parse_args()

output_dir: str = os.path.join(os.path.normpath("."), "output", "Parameter scans")
os.makedirs(output_dir, exist_ok=True)

csv_default: str = os.path.join(output_dir, "scan_params_results.csv")
if args.infile:
    csvpath = os.path.normpath(args.infile)
else:
    csvpath: str = csv_default
    if not os.path.exists(csvpath):
        csvpath: str = os.path.normpath("scan_params_results.csv")
        if not os.path.exists(csvpath):
            print("scan_params_results.csv not found at", os.path.relpath(csv_default))
            raise SystemExit(1)

out_default: str = os.path.join(output_dir, "scan_params_heatmap.png")
out: argparse.Any | str = os.path.normpath(args.outfile) if args.outfile else out_default

data = {}
movement_vals = []
maintenance_vals = []
with open(csvpath, "r", newline="") as f: os.TextIOWrapper[_WrappedBuffer]:
    r: csv.DictReader[str] = csv.DictReader(f)
    fieldnames: argparse.Sequence[str] | list[str] = r.fieldnames or []
    # detect which columns are present and map to x/y keys
    if "movement_cost_factor" in fieldnames and "maintenance_cost_per_node" in fieldnames:
        x_key = "movement_cost_factor"
        y_key = "maintenance_cost_per_node"
        x_label = "movement_cost_factor"
        y_label = "maintenance_cost_per_node"
        plot_title = "Extinction time (50k max)"
    elif "reproduction_cost" in fieldnames and "grid_width" in fieldnames:
        x_key = "reproduction_cost"
        y_key = "grid_width"
        x_label = "reproduction_cost"
        y_label = "grid_width"
        plot_title = "Extinction time (quick scan)"
    else:
        # fallback: choose first two non-meta columns
        candidates: list[str] = [fn for fn: str in fieldnames if fn not in ("extinction_time", "survived")]
        if len(candidates) >= 2:
            x_key, y_key = candidates[0], candidates[1]
            x_label, y_label = x_key, y_key
            plot_title = "Extinction time"
        else:
            raise SystemExit("CSV missing expected parameter columns")

    for row: dict[str | Any, str | Any] in r:
        try:
            xv = float(row.get(x_key, "nan"))
        except Exception:
            try:
                xv = float(row.get(x_key, "nan").replace("_", ""))
            except Exception:
                xv = float("nan")
        try:
            yv = float(row.get(y_key, "nan"))
        except Exception:
            try:
                yv = float(row.get(y_key, "nan").replace("_", ""))
            except Exception:
                yv = float("nan")
        try:
            et = int(row["extinction_time"])
        except Exception:
            try:
                et = int(float(row.get("extinction_time", "nan")))
            except Exception:
                et = 0
        surv: int = int(row.get("survived", "0")) if row.get("survived") is not None else 0
        movement_vals.append(xv)
        maintenance_vals.append(yv)
        data[(xv, yv)] = (et, surv)

movement_vals = sorted(set(movement_vals))
maintenance_vals = sorted(set(maintenance_vals))

# build matrix with NaN for missing cells
mat: np.ndarray[tuple[int, int], np.dtype[argparse.Any]] = np.full((len(maintenance_vals), len(movement_vals)), np.nan, dtype=float)
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
cmap: plt.Colormap = plt.get_cmap("magma_r")
try:
    cmap: plt.Colormap = cmap.copy()
except Exception:
    # older matplotlib versions may not support copy(); proceed with the original
    pass
try:
    cmap.set_bad(color="#dddddd")
except Exception:
    pass
im: plt.AxesImage = plt.imshow(
    mat,
    origin="lower",
    cmap=cmap,
    aspect="auto",
    vmin=0,
    vmax=vmax,
    interpolation="nearest",
)
cb: plt.Colorbar = plt.colorbar(im)
cb.set_label("extinction time (timesteps)")

# ticks with formatted floats
plt.xticks(range(len(movement_vals)), [f"{v:.6g}" for v in movement_vals], rotation=45)
plt.yticks(range(len(maintenance_vals)), [f"{v:.6g}" for v in maintenance_vals])
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(plot_title)

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
