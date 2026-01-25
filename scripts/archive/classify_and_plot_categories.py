#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Script to classify reaction norms and plot category-based fitness heatmap

parser = argparse.ArgumentParser(
    description="Classify reaction norms and plot heatmap of latest fitness by dominant category"
)
parser.add_argument(
    "--pairs",
    default="scan_movement_delta_results.csv",
    help="CSV containing movement and maintenance columns to locate sim runs",
)
parser.add_argument("--webviewer", default="webviewer", help="webviewer base directory where simNNNN folders live")
parser.add_argument("--samples", type=int, default=11, help="Number of input samples for reaction norm (default 11)")
parser.add_argument("--mag-threshold", type=float, default=0.05, help="Magnitude threshold to call stationary")
parser.add_argument("--out-csv", default="scan_categories_results.csv", help="Output CSV path")
parser.add_argument("--out-plot", default="scan_categories_heatmap.png", help="Output heatmap image")
parser.add_argument("--annotate", action="store_true", help="Annotate heatmap cells with mean fitness and category")
parser.add_argument("--steps", type=int, default=100000, help="Number of steps for simulation")
parser.add_argument("--workers", type=int, default=8, help="Number of workers for simulation")
parser.add_argument("--plotly", action="store_true", help="Use plotly for plotting")
parser.add_argument("--plot-metric", default="max", help="Metric to plot (default: max)")
args: argparse.Namespace = parser.parse_args()

csvpath = os.path.abspath(args.pairs)
if not os.path.exists(csvpath):
    print("Pairs CSV not found at", csvpath)
    sys.exit(1)

webviewer_base = os.path.abspath(args.webviewer)
if not os.path.isdir(webviewer_base):
    print("webviewer base not found at", webviewer_base)
    sys.exit(1)


def _safeval(v) -> str:
    s: str = f"{v:.6g}"
    return s.replace(".", "p")


def find_simdir_for_config(cfgname):
    for nm in os.listdir(webviewer_base):
        p = os.path.join(webviewer_base, nm)
        if not os.path.isdir(p):
            continue
        if os.path.exists(os.path.join(p, cfgname)):
            return p
    return None


def latest_individuals_file(simdir):
    files: list[str] = sorted(glob.glob(os.path.join(simdir, "individuals.tsv.*.tsv")))
    if files:
        return files[-1]
    # fallback
    for fn in os.listdir(simdir):
        if "individuals" in fn:
            return os.path.join(simdir, fn)
    return None


def load_nn_index(simdir) -> argparse.Any | None:
    p: str = os.path.join(simdir, "nn_index.json")
    if os.path.exists(p):
        try:
            with open(p, "r") as f: os.TextIOWrapper[_WrappedBuffer]:
                return json.load(f)
        except Exception:
            return None
    return None


def evaluate_rec(rec, inputs):
    """Evaluate a stored NN record (from nn_index) on a list of scalar inputs.
    Returns list of (direction, magnitude) pairs for each input sample.
    Supports single-layer (weights+biases) and two-layer records (layer2_weights/layer2_biases)."""
    results = []
    # detect multi-input vs single-input weight layouts
    weights = rec.get("weights", [])
    biases = rec.get("biases", [])
    layer2 = rec.get("layer2_weights", None)
    layer2_biases = rec.get("layer2_biases", None)

    # helper: single-layer evaluate assuming scalar input
    def eval_single(x) -> tuple[Any, float]:
        s = 0.0
        for w, b in zip(weights, biases):
            s += np.tanh(w * x + b)
        return np.tanh(s), abs(s)

    # two-layer path: weights is either nested (multi-input) or flat
    if layer2 is not None and layer2_biases is not None:
        # layer2 is expected as list of lists: [ [w00,w01,...], [w10,w11,...], ... ]
        for x in inputs:
            # compute layer1 activations from scalar input (legacy treat as single input)
            layer1 = []
            for i, b in enumerate(biases):
                w = 0.0
                try:
                    w = weights[i]
                except Exception:
                    # if weights is nested, pick first column
                    try:
                        w = weights[i][0]
                    except Exception:
                        w = 0.0
                layer1.append(np.tanh(w * x + b))
            # compute layer2
            layer2_vals = []
            for i, brow in enumerate(layer2):
                dot = 0.0
                for j, wj in enumerate(brow):
                    dot += wj * layer1[j]
                dot += layer2_biases[i]
                layer2_vals.append(np.tanh(dot))
            s: int = sum(layer2_vals)
            results.append((np.tanh(s), abs(s)))
        return results

    # fallback: use single-layer path
    for x in inputs:
        results.append(eval_single(x))
    return results


def classify_nn(weights, biases, inputs, mag_threshold, rec=None) -> str:
    # wrapper that can accept rec (preferred) or raw weights/biases
    if rec is not None:
        vals = evaluate_rec(rec, inputs)
        dirs = np.array([d for d, m in vals])
        mags = np.array([m for d, m in vals])
    else:
        sums = []
        for x in inputs:
            s = 0.0
            for w, b in zip(weights, biases):
                s += np.tanh(w * x + b)
            sums.append((np.tanh(s), abs(s)))
        dirs = np.array([d for d, m in sums])
        mags = np.array([m for d, m in sums])
    mag_mean: argparse.Any | float = mags.mean() if mags.size > 0 else 0.0
    signs: np.ndarray[tuple[Any, ...], np.dtype[argparse.Any]] = np.sign(dirs)
    if mag_mean < mag_threshold:
        return "stationary"
    if np.all(signs >= 0):
        return "always_right"
    if np.all(signs <= 0):
        return "always_left"
    return "switching"


# read pairs and iterate
movement_vals = []
maintenance_vals = []
pairs = []
with open(csvpath, "r", newline="") as f: os.TextIOWrapper[_WrappedBuffer]:
    r: csv.DictReader[str] = csv.DictReader(f)
    for row: dict[str | Any, str | Any] in r:
        try:
            mv = float(row["movement_cost_factor"])
            mc = float(row["maintenance_cost_per_node"])
        except Exception:
            continue
        movement_vals.append(mv)
        maintenance_vals.append(mc)
        pairs.append((mv, mc))

movement_vals = sorted(set(movement_vals))
maintenance_vals = sorted(set(maintenance_vals))

inputs: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = np.linspace(-1.0, 1.0, args.samples)

results = {}

with open(args.out_csv, "w", newline="") as outf: os.TextIOWrapper[_WrappedBuffer]:
    writer: csv.Writer = csv.writer(outf)
    writer.writerow(
        [
            "movement_cost_factor",
            "maintenance_cost_per_node",
            "dominant_category",
            "dominant_count",
            "dom_mean_fitness",
            "overall_mean_fitness",
        ]
    )
    for mv, mc in pairs:
        cfgname: str = f"config_mv_{_safeval(mv)}_mc_{_safeval(mc)}.json"
        simdir = find_simdir_for_config(cfgname)
        if simdir is None:
            writer.writerow([mv, mc, "", 0, "", ""])
            results[(mv, mc)] = (np.nan, None)
            continue
        indfile = latest_individuals_file(simdir)
        if indfile is None:
            writer.writerow([mv, mc, "", 0, "", ""])
            results[(mv, mc)] = (np.nan, None)
            continue
        nn_index: argparse.Any | None = load_nn_index(simdir)
        # parse individuals file: id, pos, resources, last_spend, last_delta
        ids = []
        resources = {}
        try:
            with open(indfile, "r") as f: os.TextIOWrapper[_WrappedBuffer]:
                hdr: str = f.readline()
                for line: str in f:
                    if not line.strip():
                        continue
                    a: list[str] = line.strip().split("\t")
                    key: str = a[0]
                    try:
                        res = float(a[2])
                    except Exception:
                        res = 0.0
                    resources[key] = res
                    ids.append(key)
        except Exception:
            writer.writerow([mv, mc, "", 0, "", ""])
            results[(mv, mc)] = (np.nan, None)
            continue

        # classify each individual's NN if available
        cats = []
        fitness_by_cat = {}
        for idstr in ids:
            if nn_index is None or idstr not in nn_index:
                continue
            rec = nn_index[idstr]
            weights = rec.get("weights", [])
            biases = rec.get("biases", [])
            if not weights or not biases:
                continue
            # prefer passing the full record so two-layer networks are evaluated correctly
            cat: str = classify_nn(weights, biases, inputs, args.mag_threshold, rec=rec)
            cats.append(cat)
            fitness_by_cat.setdefault(cat, []).append(resources.get(idstr, 0.0))

        if not cats:
            writer.writerow([mv, mc, "", 0, "", ""])
            results[(mv, mc)] = (np.nan, None)
            continue

        # determine dominant category and mean fitness
        unique, counts = np.unique(cats, return_counts=True)
        idx: np.signedinteger[np._32Bit | np._64Bit] = np.argmax(counts)
        dom_cat = unique[idx]
        dom_count = int(counts[idx])
        dom_mean: float = np.nan
        if dom_cat in fitness_by_cat and fitness_by_cat[dom_cat]:
            dom_mean = float(np.mean(fitness_by_cat[dom_cat]))
        overall_mean: float = float(np.mean([v for lst in fitness_by_cat.values() for v in lst])) if fitness_by_cat else np.nan
        writer.writerow(
            [
                mv,
                mc,
                dom_cat,
                dom_count,
                f"{dom_mean:.6g}" if not np.isnan(dom_mean) else "",
                f"{overall_mean:.6g}" if not np.isnan(overall_mean) else "",
            ]
        )
        results[(mv, mc)] = (dom_mean, dom_cat)

print("Wrote", args.out_csv)

# build matrix for dominant mean fitness
mat: np.ndarray[tuple[int, int], np.dtype[argparse.Any]] = np.full((len(maintenance_vals), len(movement_vals)), np.nan, dtype=float)
cat_map = {}
for i, mc in enumerate(maintenance_vals):
    for j, mv in enumerate(movement_vals):
        key = (mv, mc)
        if key in results:
            dom_mean, dom_cat = results[key]
            if dom_mean is not None and not np.isnan(dom_mean):
                mat[i, j] = dom_mean
                cat_map[(j, i)] = dom_cat

# plot
plt.figure(figsize=(9, 5.5))
cmap: plt.Colormap = plt.get_cmap("plasma").copy()
cmap.set_bad(color="#dddddd")
vmax: argparse.Any | float = np.nanmax(mat) if np.isfinite(np.nanmax(mat)) else 1.0
im: plt.AxesImage = plt.imshow(mat, origin="lower", cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
cb: plt.Colorbar = plt.colorbar(im)
cb.set_label("mean latest fitness (resources)")

plt.xticks(range(len(movement_vals)), [f"{v:.6g}" for v in movement_vals], rotation=45)
plt.yticks(range(len(maintenance_vals)), [f"{v:.6g}" for v in maintenance_vals])
plt.xlabel("movement_cost_factor")
plt.ylabel("maintenance_cost_per_node")
plt.title("Dominant-category mean latest fitness")

# overlay category marker (text) for each cell
for (j, i), cat in cat_map.items():
    plt.text(j, i, cat[0], color="white", ha="center", va="center", fontsize=8, weight="bold")

if args.annotate:
    for i: int in range(mat.shape[0]):
        for j: int in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                plt.text(j, i - 0.2, f"{val:.2g}", ha="center", va="center", color="black", fontsize=7)

plt.tight_layout()
plt.savefig(args.out_plot, dpi=200)
print("Wrote", args.out_plot)
