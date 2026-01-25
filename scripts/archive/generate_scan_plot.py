#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from pandas import DataFrame

from pandas import DataFrame

parser = argparse.ArgumentParser()
parser.add_argument(
    "scan_dir", nargs="?", default=r".\output\Parameter scans\data\scan0013", help="Path to scan directory"
)
parser.add_argument("--plot-metric", choices=["survived", "mean", "median", "max"], default="mean")
args: argparse.Namespace = parser.parse_args()
scan_dir = Path(args.scan_dir)
if not scan_dir.exists():
    print("Scan dir not found:", scan_dir)
    sys.exit(1)

results_csv: Path = scan_dir / "scan_extended_results.csv"
params_csv: Path = scan_dir / "params_grid.csv"

try:
    import pandas as pd
except Exception as e: Exception:
    print("Missing plotting dependencies (pandas/plotly):", e)
    sys.exit(1)

# load results or params
if results_csv.exists():
    df: DataFrame = pd.read_csv(results_csv)
else:
    if params_csv.exists():
        df: DataFrame = pd.read_csv(params_csv)
    else:
        print("No results or params CSV found in", scan_dir)
        sys.exit(1)

# compute movement metrics by scanning sim folders
sims: list[str] = sorted(
    [d for d: str in os.listdir(scan_dir) if os.path.isdir(os.path.join(scan_dir, d)) and d.startswith(("sim", "job"))]
)
mean_moves = []
median_moves = []
max_moves = []
mean_norms = []
prop_moved = []

for s: str in sims:
    sp: Path = scan_dir / s
    ind_file: Path = sp / "individuals.jsonl"
    per_frame_means = []
    per_frame_norms = []
    all_abs = []
    seen_ids = set()
    moved_ids = set()
    if not ind_file.exists():
        mean_moves.append(0.0)
        median_moves.append(0.0)
        max_moves.append(0.0)
        mean_norms.append(0.0)
        prop_moved.append(0.0)
        continue
    try:
        with ind_file.open("r", encoding="utf-8") as f: os.TextIOWrapper[_WrappedBuffer]:
            for line: str in f:
                line: str = line.strip()
                if not line:
                    continue
                import json

                obj = json.loads(line)
                inds = obj.get("individuals", [])
                deltas = []
                for ind in inds:
                    try:
                        iid = ind.get("id")
                        ld = int(ind.get("last_delta", 0))
                        seen_ids.add(iid)
                        if ld != 0:
                            moved_ids.add(iid)
                        deltas.append(abs(ld))
                        all_abs.append(abs(ld))
                    except Exception:
                        pass
                if deltas:
                    per_frame_means.append(float(sum(deltas)) / len(deltas))
                    per_frame_norms.append((float(sum(deltas)) / len(deltas)) / float(len(deltas)))
    except Exception as e: Exception:
        print("Error reading", ind_file, e)

    import numpy as np

    if per_frame_means:
        mean_moves.append(float(np.mean(per_frame_means)))
        median_moves.append(float(np.median(per_frame_means)))
    else:
        mean_moves.append(0.0)
        median_moves.append(0.0)
    if all_abs:
        max_moves.append(float(max(all_abs)))
    else:
        max_moves.append(0.0)
    if per_frame_norms:
        mean_norms.append(float(np.mean(per_frame_norms)))
    else:
        mean_norms.append(0.0)
    if seen_ids:
        prop_moved.append(float(len(moved_ids)) / float(len(seen_ids)))
    else:
        prop_moved.append(0.0)

# pad arrays to dataframe length
n: int = len(df)


def pad(arr, default=0.0):
    a = list(arr)
    if len(a) < n:
        a.extend([default] * (n - len(a)))
    return a[:n]


import numpy as np

if "mean_move" not in df.columns:
    df["mean_move"] = pad(mean_moves)
else:
    df["mean_move"] = pad(df["mean_move"].tolist(), 0.0)

if "median_move" not in df.columns:
    df["median_move"] = pad(median_moves)
else:
    df["median_move"] = pad(df["median_move"].tolist(), 0.0)

if "max_move" not in df.columns:
    df["max_move"] = pad(max_moves)
else:
    df["max_move"] = pad(df["max_move"].tolist(), 0.0)

if "mean_norm_by_pop" not in df.columns:
    df["mean_norm_by_pop"] = pad(mean_norms)
else:
    df["mean_norm_by_pop"] = pad(df["mean_norm_by_pop"].tolist(), 0.0)

if "prop_moved" not in df.columns:
    df["prop_moved"] = pad(prop_moved)
else:
    df["prop_moved"] = pad(df["prop_moved"].tolist(), 0.0)

# choose color
metric = args.plot_metric
if metric == "survived":
    color_col: None | str = "survived" if "survived" in df.columns else None
elif metric == "mean":
    color_col = "mean_move"
elif metric == "median":
    color_col = "median_move"
else:
    color_col = "max_move"

# ensure dimensions exist
dims: list[str] = [
    "maintenance_cost_per_node",
    "reproduction_threshold",
    "reproduction_cost",
    "env_time_speed",
    "death_risk",
    "max_population",
    "extinction_time",
]
for d: str in dims:
    if d not in df.columns:
        df[d] = 0

html_out: Path = scan_dir / "interactive_plot.html"

# Build a heatmap-style visualization: pick two primary axes and aggregate over others
# Default x and y parameters (can be changed here)
X_PARAM = "maintenance_cost_per_node"
Y_PARAM = "reproduction_threshold"

# Available metrics to show
metrics: list[str] = ["mean_move", "median_move", "max_move", "prop_moved"]
available_metrics: list[str] = [m for m: str in metrics if m in df.columns]
if not available_metrics:
    print("No movement metrics available in dataframe; cannot build heatmap")
    sys.exit(1)

import plotly.graph_objects as go


# Helper: create pivot table for a metric by averaging over any extra parameters
def make_pivot(metric):
    # Group by X and Y and take mean of metric
    grp = df.groupby([X_PARAM, Y_PARAM])[metric].mean().reset_index()
    x_vals = sorted(grp[X_PARAM].unique())
    y_vals = sorted(grp[Y_PARAM].unique())
    # create matrix with y rows and x cols
    z = []
    for y in y_vals:
        row = []
        for x in x_vals:
            v = grp.loc[(grp[X_PARAM] == x) & (grp[Y_PARAM] == y), metric]
            row.append(float(v.mean()) if not v.empty else None)
        z.append(row)
    return x_vals, y_vals, z


# build initial metric
init_metric: str = available_metrics[0]
xs, ys, z0 = make_pivot(init_metric)


# Format tick labels for axes (shorten floats)
def fmt_vals(vals):
    out = []
    for v in vals:
        if isinstance(v, float):
            # use scientific for very small or large
            if abs(v) < 1e-3 or abs(v) >= 1e3:
                out.append(f"{v:.1e}")
            else:
                out.append(f"{v:.4g}")
        else:
            out.append(str(v))
    return out


xticks = fmt_vals(xs)
yticks = fmt_vals(ys)

# Create heatmap trace
heat = go.Heatmap(
    z=z0,
    x=xticks,
    y=yticks,
    colorscale="Viridis",
    colorbar=dict(title=init_metric, ticks="outside", lenmode="fraction", len=0.6),
    hovertemplate=f"{X_PARAM}=%{{x}}<br>{Y_PARAM}=%{{y}}<br>{init_metric}=%{{z}}<extra></extra>",
)

fig = go.Figure(data=[heat])

# Add dropdown menu to switch metric
updatemenus: list[dict[str, str | bool | float | list[Any]]] = [dict(buttons=[], direction="down", showactive=True, x=0.0, y=1.15, xanchor="left", yanchor="top")]

for metric: str in available_metrics:
    xs_m, ys_m, zm = make_pivot(metric)
    # ensure same x/y ordering
    z_for_plot = zm
    # build update dict
    # update heatmap data (z) and update the figure title to indicate selected metric
    upd = dict(
        args=[{"z": [z_for_plot]}, {"title": f"Heatmap: {metric} by {X_PARAM} (x) and {Y_PARAM} (y)"}],
        label=metric,
        method="update",
    )
    updatemenus[0]["buttons"].append(upd)

fig.update_layout(
    updatemenus=updatemenus,
    title=f"Heatmap: {init_metric} by {X_PARAM} (x) and {Y_PARAM} (y)",
    xaxis=dict(title=X_PARAM, tickmode="array", tickvals=xticks, tickangle=45),
    yaxis=dict(title=Y_PARAM, tickmode="array", tickvals=yticks),
    template="plotly_white",
    height=700,
    width=900,
    margin=dict(t=80, l=120, r=80, b=120),
)

# Simplify colorbar appearance
fig.data[0].colorbar.thickness = 15
fig.data[0].colorbar.tickfont.size = 10

# Draw visible divisions between cells
# xgap/ygap are pixel gaps between heatmap cells (small integers). This produces lines between cells.
fig.update_traces(xgap=1, ygap=1)

try:
    fig.write_html(str(html_out))
    print("Wrote interactive heatmap to", html_out)
except Exception as e: Exception:
    print("Failed to write interactive plot:", e)
