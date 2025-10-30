#!/usr/bin/env python3
"""*************************************************************************************
Filename:       scripts/classify_strategies_global.py

Purpose:        Scan one or more simulation `webviewer/simNNNN` directories,
                classify neural-network reaction-norm strategies into canonical
                categories, and summarize abundance and normalized fitness distributions
                per category.

Author:         Luis Boullosa

Quick usage:    python scripts/classify_strategies_global.py [--sim sim0123]

Inputs:         - Sim directory folder (e.g. webviewer/sim0123).

Key outputs:    - CSV and PNG written to output/Classifiers/. Filenames include the
                  sim directory label (e.g. strategy_summary_sim0008.png).

Notes:
    - Fitness values are normalized to [0,1] across all individuals before stats.
*************************************************************************************"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Classify reaction-norms across sims and summarize abundance/fitness"
)
parser.add_argument(
    "--webviewer",
    default="webviewer",
    help="webviewer base directory containing simNNNN folders",
)
parser.add_argument(
    "--samples", type=int, default=11, help="Number of input samples for reaction norm"
)
parser.add_argument(
    "--mag-threshold",
    type=float,
    default=0.05,
    help="Magnitude threshold to call stationary",
)
parser.add_argument(
    "--out-csv", default="strategy_summary.csv", help="CSV summary output"
)
parser.add_argument(
    "--out-prefix", default="strategy_summary", help="Prefix for output plots"
)
parser.add_argument(
    "--sim",
    default=None,
    help="Specific sim directory name or glob pattern (e.g. sim0123 or sim*)",
)
parser.add_argument(
    "--bins",
    type=int,
    default=50,
    help="Number of histogram bins for fitness distribution",
)

args = parser.parse_args()

base = os.path.normpath(args.webviewer)
if not os.path.isdir(base):
    print("webviewer base not found at", os.path.relpath(base))
    sys.exit(1)


def classify_nn(weights, biases, inputs, mag_threshold):
    sums = []
    for x in inputs:
        s = 0.0
        for w, b in zip(weights, biases):
            s += np.tanh(w * x + b)
        dir_ = np.tanh(s)
        mag = abs(s)
        sums.append((dir_, mag))
    dirs = np.array([d for d, m in sums])
    mags = np.array([m for d, m in sums])
    mag_mean = mags.mean() if mags.size > 0 else 0.0
    signs = np.sign(dirs)
    if mag_mean < mag_threshold:
        return "stationary"
    elif np.all(signs >= 0):
        return "always_right"
    elif np.all(signs <= 0):
        return "always_left"
    else:
        return "switching"


def evaluate_rec(rec, inputs):
    # support optional layer2 weights/biases; returns list of (dir, mag)
    results = []
    weights = rec.get('weights', [])
    biases = rec.get('biases', [])
    layer2 = rec.get('layer2_weights', None)
    layer2_biases = rec.get('layer2_biases', None)
    if layer2 is not None and layer2_biases is not None:
        for x in inputs:
            layer1 = []
            for i, b in enumerate(biases):
                try:
                    w = weights[i]
                except Exception:
                    try:
                        w = weights[i][0]
                    except Exception:
                        w = 0.0
                layer1.append(np.tanh(w * x + b))
            layer2_vals = []
            for i, brow in enumerate(layer2):
                dot = 0.0
                for j, wj in enumerate(brow):
                    dot += wj * layer1[j]
                dot += layer2_biases[i]
                layer2_vals.append(np.tanh(dot))
            s = sum(layer2_vals)
            results.append((np.tanh(s), abs(s)))
        return results
    # fallback single-layer
    for x in inputs:
        s = 0.0
        for w,b in zip(weights, biases):
            s += np.tanh(w * x + b)
        results.append((np.tanh(s), abs(s)))
    return results


inputs = np.linspace(-1.0, 1.0, args.samples)

categories = {}

# canonical strategy definitions: key -> (symbol, description)
STRATEGIES = {
    "stationary": ("\u25cb", "stationary / small magnitude"),
    "always_left": ("\u2190", "always move left"),
    "always_right": ("\u2192", "always move right"),
    "switching": ("\u2194", "switch direction depending on input"),
}

all_sim_dirs = sorted(
    [
        os.path.join(base, d)
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.startswith("sim")
    ]
)
if not all_sim_dirs:
    print("No sim directories found under", os.path.relpath(base))
    sys.exit(0)

# if user provided --sim, try to resolve it (exact name or glob)
sim_dirs = []
if args.sim:
    # exact directory name
    candidate = os.path.join(base, args.sim)
    if os.path.isdir(candidate):
        sim_dirs = [candidate]
    else:
        # treat as glob pattern relative to base
        matches = sorted(glob.glob(os.path.join(base, args.sim)))
        sim_dirs = [m for m in matches if os.path.isdir(m)]
    if not sim_dirs:
        print("No sim directories matched pattern:", args.sim)
        sys.exit(1)
    rels = [os.path.relpath(s) for s in sim_dirs]
    print("Using sim directories:", rels)
else:
    # default: use only the last (most recent) simulation directory
    sim_dirs = [all_sim_dirs[-1]]
    print("Using last sim directory:", os.path.relpath(sim_dirs[0]))

for simdir in sim_dirs:
    from webviewer_helpers import iter_individuals_frames, iter_env_frames
    # find latest individuals frame via helper (supports jsonl or tsv)
    frames = list(iter_individuals_frames(simdir))
    if not frames:
        continue
    # pick last frame
    t_last, inds_last = frames[-1]
    nn_index_path = os.path.join(simdir, "nn_index.json")
    if not os.path.exists(nn_index_path):
        continue
    try:
        with open(nn_index_path, "r") as f:
            nn_index = json.load(f)
    except Exception:
        continue
    # resources for last frame are in inds_last
    ids = [str(ind.get("id")) for ind in inds_last]
    resources = {str(ind.get("id")): float(ind.get("resources", 0.0)) for ind in inds_last}

    for idstr in ids:
        if idstr not in nn_index:
            continue
        rec = nn_index[idstr]
        weights = rec.get("weights", [])
        biases = rec.get("biases", [])
        if not weights or not biases:
            continue
        # prefer evaluating full record (handles layer2 entries)
        try:
            rec = nn_index[idstr]
            vals = evaluate_rec(rec, inputs)
            dirs = np.array([d for d,m in vals])
            mags = np.array([m for d,m in vals])
            mag_mean = mags.mean() if mags.size>0 else 0.0
            signs = np.sign(dirs)
            if mag_mean < args.mag_threshold:
                cat = 'stationary'
            elif np.all(signs >= 0):
                cat = 'always_right'
            elif np.all(signs <= 0):
                cat = 'always_left'
            else:
                cat = 'switching'
        except Exception:
            cat = classify_nn(weights, biases, inputs, args.mag_threshold)
        fitness = resources.get(idstr, 0.0)
        categories.setdefault(cat, []).append(fitness)

# write summary CSV
rows = []
# Normalize fitness values to [0,1] across all individuals (global min/max)
all_vals = (
    np.concatenate([np.array(v) for v in categories.values()])
    if categories
    else np.array([])
)
if all_vals.size > 0:
    global_min = float(np.nanmin(all_vals))
    global_max = float(np.nanmax(all_vals))
else:
    global_min = 0.0
    global_max = 1.0


def normalize_array(arr, lo, hi):
    if arr.size == 0:
        return arr
    if hi <= lo:
        # constant case: map everything to 0.5
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


# Ensure standard strategy categories always appear (even with no data)
standard_cats = list(STRATEGIES.keys())
extra_cats = [c for c in categories.keys() if c not in standard_cats]
ordered_cats = standard_cats + sorted(extra_cats)

for cat in ordered_cats:
    vals = categories.get(cat, [])
    arr = np.array(vals, dtype=float)
    arrn = normalize_array(arr, global_min, global_max)
    rows.append(
        (
            cat,
            len(arrn),
            float(np.mean(arrn)) if arrn.size > 0 else float("nan"),
            float(np.median(arrn)) if arrn.size > 0 else float("nan"),
            float(np.std(arrn)) if arrn.size > 0 else float("nan"),
        )
    )

# write CSV and PNG to output/Classifiers/, overwrite existing files
classifier_dir = os.path.join(os.path.normpath("."), "output", "Classifiers")
os.makedirs(classifier_dir, exist_ok=True)
# include sim identifier in filenames
if len(sim_dirs) == 1:
    sim_label = os.path.basename(sim_dirs[0])
else:
    sim_label = "multiple_sims"
# sanitize sim_label to be filesystem-friendly
sim_label = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in sim_label])
csv_name = os.path.splitext(args.out_csv)[0] + f"_{sim_label}.csv"
csv_outp = os.path.join(classifier_dir, csv_name)
with open(csv_outp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category", "count", "mean_fitness", "median_fitness", "std_fitness"])
    for r in rows:
        # write raw formatted numbers or NaN where appropriate
        mean_s = f"{r[2]:.6g}" if not np.isnan(r[2]) else ""
        med_s = f"{r[3]:.6g}" if not np.isnan(r[3]) else ""
        std_s = f"{r[4]:.6g}" if not np.isnan(r[4]) else ""
        w.writerow([r[0], r[1], mean_s, med_s, std_s])

print("Wrote", os.path.relpath(csv_outp), "(overwritten)")

# plot: left column  = abundance (single tall axis)
#       right column = stacked histograms (one per category)
cats = [r[0] for r in rows]
counts = [r[1] for r in rows]
n_cats = len(cats)

if n_cats == 0:
    print("No categories to plot")
    sys.exit(0)

# prepare bins from pooled values if possible
# use normalized values for plotting
normalized_categories = {}
for c in cats:
    vals = np.array(categories.get(c, []), dtype=float)
    normalized_categories[c] = normalize_array(vals, global_min, global_max)

all_vals = (
    np.concatenate(
        [
            normalized_categories[c]
            for c in cats
            if c in normalized_categories and len(normalized_categories[c]) > 0
        ]
    )
    if cats
    else np.array([])
)
constant_hist = False
if all_vals.size > 0:
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    # if all normalized values are (nearly) equal, set a small epsilon range for bins
    if np.allclose(vmin, vmax):
        constant_hist = True
        center = float(vmin)
        eps = max(1e-3, 0.01 * (abs(center) + 1e-6))
        lo = max(0.0, center - eps)
        hi = min(1.0, center + eps)
        bins = np.linspace(lo, hi, args.bins)
    else:
        # use the normalized pooled min/max for the bins
        bins = np.linspace(vmin, vmax, args.bins)
else:
    bins = np.linspace(0.0, 1.0, args.bins)

# figure height proportional to number of hist panels
fig_height = max(3, 0.8 * n_cats + 2)
fig = plt.figure(figsize=(13, fig_height))
gs = plt.GridSpec(nrows=n_cats, ncols=2, width_ratios=[1, 3], wspace=0.3, hspace=0.4)

# colormap: 'YlGn' ecology palette
cmap = plt.get_cmap("YlGn")
colors = [cmap(i / max(1, n_cats - 1)) for i in range(n_cats)]

# left: abundance spanning all rows
ax_abund = fig.add_subplot(gs[:, 0])
bar_colors = colors
ax_abund.bar(range(n_cats), counts, color=bar_colors, edgecolor="k")
ax_abund.set_xticks(range(n_cats))
# map category names to symbols using STRATEGIES (fall back to name)
tick_labels = [STRATEGIES.get(c, (c, ""))[0] if c in STRATEGIES else c for c in cats]
ax_abund.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=14)
ax_abund.set_xlabel("Strategy", fontsize=12)
ax_abund.set_ylabel("Count", fontsize=14)

# right: stacked histogram panels, one per category
for i, cat in enumerate(cats):
    ax = fig.add_subplot(gs[i, 1])
    vals = np.array(normalized_categories.get(cat, []))
    col = colors[i]
    if vals.size > 0:
        # if global data are constant, plot counts instead of density to avoid
        # division-by-zero in histogram density calculation
        density_flag = False if constant_hist else True
        ax.hist(
            vals, bins=bins, density=density_flag, alpha=0.9, color=col, edgecolor="k"
        )
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Normalized fitness (0-1)", fontsize=12)
        # place the category title inside the plot at top-right for readability
        title_text = f"{cat} (n={len(vals)})  mean={np.mean(vals):.3g}"
        ax.text(
            0.98,
            0.92,
            title_text,
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.text(
            0.98,
            0.92,
            f"{cat} (n=0)",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

for ax in fig.axes:
    ax.label_outer()

# add a single shared Y-axis label for the right column (centered between columns)
if len(fig.axes) > 1:
    try:
        left_bbox = fig.axes[0].get_position()
        right_bbox = fig.axes[1].get_position()
        x = (left_bbox.x1 + right_bbox.x0) / 2.0
        fig.text(x, 0.5, "Frequency", va="center", rotation="vertical", fontsize=14)
    except Exception:
        # fallback: place label at a reasonable fixed x
        fig.text(0.07, 0.5, "Frequency", va="center", rotation="vertical", fontsize=14)

# manually adjust subplots spacing (avoid fig.tight_layout which can warn for some Axes)
# Use slightly larger left/bottom margins and a wider right column so
# histogram x-labels and abundance symbols don't overlap. Increase
# vertical spacing when there are many category rows.
calc_hspace = 0.4 if n_cats > 8 else 0.2
fig.subplots_adjust(
    left=0.10, right=0.96, top=0.94, bottom=0.12, hspace=calc_hspace, wspace=0.15
)

# write PNG with sim label included
png_name = args.out_prefix + f"_{sim_label}.png"
png_outp = os.path.join(classifier_dir, png_name)
fig.savefig(png_outp, dpi=200)
print("Wrote", os.path.relpath(png_outp), "(overwritten)")
