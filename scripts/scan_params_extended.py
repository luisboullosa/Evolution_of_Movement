#!/usr/bin/env python3
"""
Extended parameter scan: sweep over maintenance_cost_per_node, reproduction_threshold,
reproduction_cost, resource_scale, and env_time_speed. Creates a parameters CSV and runs
simulations (default 100k steps). Also generates an interactive Plotly HTML to explore
results and parameters.
"""

import argparse
import itertools
import json
import os

from worker_pool import run_param_sweep

# Prefer relative exe paths
EXE = (
    "./build/Release/evo-movement-sim.exe"
    if os.path.exists("./build/Release/evo-movement-sim.exe")
    else "./build/evo-movement-sim.exe"
)
EXE = os.path.normpath(EXE)
print("Using exe:", EXE)

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=6)
parser.add_argument("--steps", type=int, default=100000, help="Number of timesteps per sim")
parser.add_argument("--pilot", action="store_true", help="Run a short pilot instead of full runs")
parser.add_argument("--mid", action="store_true", help="Run a mid-length pilot instead of full runs")
parser.add_argument("--timeout", type=int, default=3600, help="Per-sim timeout in seconds")
parser.add_argument("--plotly", action="store_true", help="Generate interactive Plotly HTML in scan folder")
parser.add_argument(
    "--initial-population",
    type=int,
    default=100,
    help="Initial population size passed to simulator (use larger to increase prop_moved resolution)",
)
# parse args later after all options are defined
parser.add_argument(
    "--plot-metric",
    choices=["survived", "mean", "median", "max"],
    default="survived",
    help="Which metric to color the interactive plot by: survival or movement stat",
)

# forward-through simulator options for convenience
parser.add_argument("--sim-death-risk", type=float, default=0.0001, help="Forwarded to simulator as --death-risk")
parser.add_argument(
    "--write-frames",
    action="store_true",
    help="Force simulator to write per-frame outputs (individuals/env TSVs) for inspection",
)

# allow plotting only existing scan without re-running via --plot-only
parser.add_argument(
    "--plot-only", action="store_true", help="Skip running sims and only generate the plot from existing sim folders"
)

# Optional CLI overrides for scanned ranges (comma-separated lists)
parser.add_argument(
    "--maintenance-vals",
    type=str,
    help="Comma-separated maintenance_cost_per_node values to sweep (overrides defaults)",
)
parser.add_argument(
    "--repro-thresholds", type=str, help="Comma-separated reproduction_threshold values to sweep (overrides defaults)"
)
parser.add_argument(
    "--repro-costs", type=str, help="Comma-separated reproduction_cost values to sweep (overrides defaults)"
)
parser.add_argument(
    "--env-time-speeds", type=str, help="Comma-separated env_time_speed values to sweep (overrides defaults)"
)
parser.add_argument("--death-risks", type=str, help="Comma-separated death_risk values to sweep (overrides defaults)")
parser.add_argument(
    "--max-populations", type=str, help="Comma-separated max_population values to sweep (overrides defaults)"
)

# reparse args to include the new options
args = parser.parse_args()

# if pilot requested, reduce steps and possibly sample sparser ranges
if args.pilot:
    steps = 2000
    burn_in = 0
    maintenance_vals = [0.0015, 0.01]
    repro_thresholds = [20.0]
    repro_costs = [1.0, 10.0]
    # keep resource scale and initial population fixed for the pilot
    env_time_speeds = [0.0, 0.15]
    death_risks = [0.00001]
    max_populations = [20000]
else:
    if args.mid:
        steps = 11000
        burn_in = steps - 1000
        maintenance_vals = [0.0015, 0.01]
        repro_thresholds = [20.0, 50.0]
        repro_costs = [5.0, 10.0]
        env_time_speeds = [0.0, 0.1]
        # Death risk/demographic sweep values
        death_risks = [0.0000001, 0.001]
        # Population cap sweep values (0 would mean no cap; here we choose a few caps)
        max_populations = [100000]
    else:
        # parameter ranges (choose a reasonable default grid - users can edit this file)
        steps = 101000
        burn_in = steps - 1000
        maintenance_vals = [0.0015, 0.005, 0.01]
        repro_thresholds = [20.0, 50.0, 100.0]
        repro_costs = [1.0, 5.0, 10.0]
        env_time_speeds = [0.0, 0.001, 0.01, 0.1]
        # Death risk/demographic sweep values
        death_risks = [0.0000001, 0.0001, 0.001]
        # Population cap sweep values (0 would mean no cap; here we choose a few caps)
        max_populations = [10000, 100000, 1000000]


# Apply command-line overrides (if provided)
def _parse_csv_floats(s):
    try:
        return [float(x) for x in s.split(",") if x.strip() != ""]
    except Exception:
        return None


if getattr(args, "maintenance_vals", None):
    vals = _parse_csv_floats(args.maintenance_vals)
    if vals:
        maintenance_vals = vals

if getattr(args, "repro_thresholds", None):
    vals = _parse_csv_floats(args.repro_thresholds)
    if vals:
        repro_thresholds = vals

if getattr(args, "repro_costs", None):
    vals = _parse_csv_floats(args.repro_costs)
    if vals:
        repro_costs = vals

if getattr(args, "env_time_speeds", None):
    vals = _parse_csv_floats(args.env_time_speeds)
    if vals:
        env_time_speeds = vals

if getattr(args, "death_risks", None):
    vals = _parse_csv_floats(args.death_risks)
    if vals:
        death_risks = vals

if getattr(args, "max_populations", None):
    vals = _parse_csv_floats(args.max_populations)
    if vals:
        max_populations = [int(x) for x in vals]

# create parameter combinations and write a CSV listing
# Sweep: maintenance, repro threshold, repro cost, env_time_speed, death_risk, max_population
raw_pairs = list(
    itertools.product(maintenance_vals, repro_thresholds, repro_costs, env_time_speeds, death_risks, max_populations)
)
# enumerate jobs so we can create per-job config files and reliably locate sim directories
pairs = list(enumerate(raw_pairs))

output_dir = os.path.join(os.path.normpath("."), "output", "Parameter scans")
os.makedirs(output_dir, exist_ok=True)
data_base = os.path.join(output_dir, "data")
os.makedirs(data_base, exist_ok=True)
existing = sorted([d for d in os.listdir(data_base) if d.startswith("scan")])

if args.plot_only:
    # In plot-only mode reuse the latest existing scan directory instead of creating a new one.
    if not existing:
        raise SystemExit("No existing scans found under data/; cannot run in --plot-only mode")
    latest = existing[-1]
    scan_dir = os.path.join(data_base, latest)
    print("Plot-only mode: using existing scan folder:", scan_dir)
    params_csv = os.path.join(scan_dir, "params_grid.csv")
    results_csv = os.path.join(scan_dir, "scan_extended_results.csv")
    # do not write any config files or params when plotting only
    run_cfg_paths = {}
else:
    idx = len(existing) + 1
    scan_dir = os.path.join(data_base, f"scan{idx:04d}")
    os.makedirs(scan_dir, exist_ok=True)
    params_csv = os.path.join(scan_dir, "params_grid.csv")
    results_csv = os.path.join(scan_dir, "scan_extended_results.csv")
    # create a small per-run config that forces simulator webviewer output into scan folder
    run_cfg_path = os.path.normpath(os.path.join(scan_dir, "scan_config.json"))
    with open(run_cfg_path, "w") as cf:
        json.dump({"webviewer_base": os.path.normpath(scan_dir)}, cf)

    # write the parameter listing
    with open(params_csv, "w", newline="") as pf:
        pf.write(
            "maintenance_cost_per_node,reproduction_threshold,reproduction_cost,env_time_speed,death_risk,max_population\n"
        )
        for m, rt, rc, ets, dr, mp in raw_pairs:
            pf.write(f"{m},{rt},{rc},{ets},{dr},{mp}\n")

    # create per-job config files so sim folders are predictable (prefix per job)
    run_cfg_paths = {}
    for idx, job in pairs:
        cfgp = os.path.join(scan_dir, f"scan_config_job_{idx}.json")
        cfg = {"webviewer_base": os.path.normpath(scan_dir), "sim_prefix": f"job{idx}_"}
        with open(cfgp, "w") as cfj:
            json.dump(cfg, cfj)
        run_cfg_paths[idx] = cfgp


# builder: job is a tuple (maintenance, repro_thresh, repro_cost, resource_scale, env_time_speed)
def cmd_builder(job):
    # job: (idx, (maintenance, repro_thresh, repro_cost, env_time_speed, death_risk, max_population))
    idx, packed = job
    m, rt, rc, ets, dr, mp = packed
    return [
        EXE,
        f"--steps={steps}",
        f"--burn-in={burn_in}",
        # pass through initial population from CLI so users can increase sampling resolution
        f"--initial-population={args.initial_population}",
        f"--max-population={mp}",
        f"--death-risk={dr}",
        f"--maintenance-cost-per-node={m}",
        f"--reproduction-threshold={rt}",
        f"--reproduction-cost={rc}",
        # resource_scale intentionally not varied here
        f"--env-time-speed={ets}",
        "--write-config",
        "--config-out=config.json",
        "--no-progress",
        # optionally request frames
        *(["--no-print-frames"] if not args.write_frames and not args.pilot else []),
        # when running pilot, force per-frame outputs and no burn-in so we can compute movement metrics
        # write per-frame outputs when running pilots or when explicitly requested; also enable for mid runs
        *(["--write-frames"] if args.pilot or args.write_frames or args.mid else []),
        f"--config-file={run_cfg_paths[idx]}",
    ]


# custom row builder: write a row including all parameter values and extinction/survived
def result_row_builder(job, extinct, out_text):
    # job is (idx, packed)
    idx, packed = job
    m, rt, rc, ets, dr, mp = packed
    survived = 1 if extinct is None else 0
    # default max if extinct none
    max_steps = steps
    ext_val = extinct if extinct is not None else max_steps

    # Try to locate the most-recent sim folder that contains individuals.jsonl and
    # stream it to compute movement metrics. This is more robust than relying on
    # simulator-created folder name prefixes (which may vary by platform or config).
    mean_move = 0.0
    median_move = 0.0
    max_move = 0.0
    prop_moved = 0.0
    recorded_steps = 0

    try:
        # list candidate sim dirs (prefer those starting with sim or job, but fall back
        # to any directory). Sort by modification time descending so the most-recent
        # sim folders are checked first.
        cand = [d for d in os.listdir(scan_dir) if os.path.isdir(os.path.join(scan_dir, d))]
        cand_paths = [os.path.join(scan_dir, d) for d in cand]
        cand_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        sim_found = None
        for p in cand_paths:
            if os.path.exists(os.path.join(p, "individuals.jsonl")):
                sim_found = p
                break

        if sim_found:
            moved_ids = set()
            seen_ids = set()
            per_frame_means = []
            all_abs = []
            # stream the large JSONL file line-by-line
            with open(os.path.join(sim_found, "individuals.jsonl"), "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    # count this as one recorded frame
                    recorded_steps += 1
                    frame_vals = []
                    for ind in obj.get("individuals", []):
                        try:
                            iid = ind.get("id")
                            if iid is None:
                                continue
                            seen_ids.add(iid)
                            ld = int(ind.get("last_delta", 0) or 0)
                            if ld != 0:
                                moved_ids.add(iid)
                            v = abs(ld)
                            frame_vals.append(v)
                            all_abs.append(v)
                        except Exception:
                            continue
                    if frame_vals:
                        per_frame_means.append(float(sum(frame_vals)) / float(len(frame_vals)))

            if per_frame_means:
                mean_move = float(sum(per_frame_means)) / float(len(per_frame_means))
                # median
                s = sorted(per_frame_means)
                L = len(s)
                if L % 2 == 1:
                    median_move = float(s[L // 2])
                else:
                    median_move = float((s[L // 2 - 1] + s[L // 2]) / 2.0)
            if all_abs:
                max_move = float(max(all_abs))
            if seen_ids:
                prop_moved = float(len(moved_ids)) / float(len(seen_ids))
    except Exception:
        # if anything goes wrong, fall back to zeros (we don't want to crash the sweep)
        mean_move = median_move = max_move = prop_moved = 0.0

    return [m, rt, rc, ets, dr, mp, ext_val, survived, recorded_steps, mean_move, median_move, max_move, prop_moved]


header = [
    "maintenance_cost_per_node",
    "reproduction_threshold",
    "reproduction_cost",
    "env_time_speed",
    "death_risk",
    "max_population",
    "extinction_time",
    "survived",
    # number of frames actually recorded in per-sim individuals.jsonl (may be < steps if extinct)
    "recorded_steps",
    # movement metrics
    "mean_move",
    "median_move",
    "max_move",
    "prop_moved",
]

if not args.plot_only:
    # run sweep
    run_param_sweep(
        pairs,
        cmd_builder,
        results_csv,
        header,
        workers=args.workers,
        timeout=args.timeout,
        result_row_builder=result_row_builder,
    )
else:
    print("Plot-only mode: skipping simulation run, will attempt to plot from existing sim folders in", scan_dir)

# optionally generate an interactive Plotly HTML
if args.plotly or args.plot_only:
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.colors import sequential
    except Exception:
        print("Plotly/pandas not available in the environment; skipping interactive plot")
    else:
        try:
            # load results or fall back to params
            if os.path.exists(results_csv):
                df = pd.read_csv(results_csv)
            else:
                df = pd.read_csv(params_csv)
                if "extinction_time" not in df.columns:
                    df["extinction_time"] = steps
                if "survived" not in df.columns:
                    df["survived"] = 0

            # compute movement metrics by scanning sim subfolders (sim directories may start with 'sim' or 'job')
            sims = sorted(
                [
                    d
                    for d in os.listdir(scan_dir)
                    if os.path.isdir(os.path.join(scan_dir, d)) and (d.startswith("sim") or d.startswith("job"))
                ]
            )
            mean_moves = []
            median_moves = []
            max_moves = []
            mean_norms = []
            prop_moved = []
            recorded_steps_list = []
            from webviewer_helpers import iter_individuals_frames

            for s in sims:
                sp = os.path.join(scan_dir, s)
                per_frame_means = []
                per_frame_norms = []
                all_abs = []
                seen_ids = set()
                moved_ids = set()
                recorded_steps = 0
                for t, inds in iter_individuals_frames(sp):
                    recorded_steps += 1
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
                if per_frame_means:
                    mean_moves.append(float(pd.Series(per_frame_means).mean()))
                    median_moves.append(float(pd.Series(per_frame_means).median()))
                    max_moves.append(float(pd.Series(all_abs).max()) if all_abs else 0.0)
                    mean_norms.append(float(pd.Series(per_frame_norms).mean()) if per_frame_norms else 0.0)
                    prop_moved.append(float(len(moved_ids)) / float(len(seen_ids)) if seen_ids else 0.0)
                    recorded_steps_list.append(recorded_steps)
                else:
                    mean_moves.append(0.0)
                    median_moves.append(0.0)
                    max_moves.append(0.0)
                    mean_norms.append(0.0)
                    prop_moved.append(0.0)
                    recorded_steps_list.append(0)

            # align lengths: df rows may equal number of pairs; we pad if sims shorter
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
            df["prop_moved"] = pad(prop_moved)
            # add recorded steps (frames) information
            df["recorded_steps"] = pad(recorded_steps_list, default=0)

            # Scanned dimensions (restore these so they're available in the plot/hover)
            dims = [
                "maintenance_cost_per_node",
                "reproduction_threshold",
                "reproduction_cost",
                "env_time_speed",
                "death_risk",
                "max_population",
                "extinction_time",
            ]

            # Build heatmap: choose two axes and aggregate metric by mean
            X_PARAM = "maintenance_cost_per_node"
            Y_PARAM = "reproduction_threshold"

            metrics = [c for c in ["mean_move", "median_move", "max_move", "prop_moved"] if c in df.columns]
            if not metrics:
                print("No movement metrics found to plot")
            else:

                def make_pivot(metric):
                    # group and compute metric aggregation
                    grp = df.groupby([X_PARAM, Y_PARAM]).agg({metric: "mean"})
                    # for each additional scanned dim compute mean for hover info
                    extra_dims = [d for d in dims if d not in (X_PARAM, Y_PARAM) and d in df.columns]
                    if extra_dims:
                        extras = df.groupby([X_PARAM, Y_PARAM])[extra_dims].mean()
                    else:
                        extras = None

                    grp = grp.reset_index()
                    x_vals = sorted(grp[X_PARAM].unique())
                    y_vals = sorted(grp[Y_PARAM].unique())
                    z = []
                    custom = []
                    for y in y_vals:
                        row = []
                        crow = []
                        for x in x_vals:
                            sel = grp.loc[(grp[X_PARAM] == x) & (grp[Y_PARAM] == y), metric]
                            val = float(sel.mean()) if not sel.empty else None
                            row.append(val)
                            # build customdata for this cell: include other scanned dims (mean)
                            if extras is not None:
                                try:
                                    e = extras.loc[(x, y)] if (x, y) in list(extras.index) else None
                                except Exception:
                                    e = None
                                if e is not None:
                                    parts = []
                                    for d in extra_dims:
                                        dv = (
                                            e.get(d, None)
                                            if hasattr(e, "get")
                                            else (float(e[d]) if d in e.index else None)
                                        )
                                        if dv is None:
                                            parts.append(f"{d}: -")
                                        else:
                                            parts.append(f"{d}: {dv:.4g}")
                                    crow.append("<br>".join(parts))
                                else:
                                    crow.append("")
                            else:
                                crow.append("")
                        z.append(row)
                        custom.append(crow)
                    return x_vals, y_vals, z, custom

                init_metric = metrics[0]
                xs, ys, z0, custom0 = make_pivot(init_metric)

                def fmt_vals(vals):
                    out = []
                    for v in vals:
                        if isinstance(v, float):
                            if abs(v) < 1e-3 or abs(v) >= 1e3:
                                out.append(f"{v:.1e}")
                            else:
                                out.append(f"{v:.4g}")
                        else:
                            out.append(str(v))
                    return out

                xticks = fmt_vals(xs)
                yticks = fmt_vals(ys)

                heat = go.Heatmap(
                    z=z0,
                    x=xticks,
                    y=yticks,
                    customdata=custom0,
                    colorscale=sequential.Viridis,
                    colorbar=dict(title=init_metric, ticks="outside", lenmode="fraction", len=0.6),
                    hovertemplate=f"{X_PARAM}=%{{x}}<br>{Y_PARAM}=%{{y}}<br>{init_metric}=%{{z}}<br>%{{customdata}}<extra></extra>",
                )

                fig = go.Figure(data=[heat])
                # add thin gaps between cells to show divisions
                fig.update_traces(xgap=1, ygap=1)

                # dropdown to switch metric
                buttons = []
                for metric in metrics:
                    xs_m, ys_m, zm, cm = make_pivot(metric)
                    # use same formatted ticks
                    buttons.append(
                        dict(
                            label=metric,
                            method="update",
                            args=[
                                {"z": [zm], "customdata": [cm]},
                                {"title": f"Heatmap: {metric} by {X_PARAM} (x) and {Y_PARAM} (y)"},
                            ],
                        )
                    )

                fig.update_layout(
                    updatemenus=[
                        dict(
                            buttons=buttons,
                            direction="down",
                            showactive=True,
                            x=0.0,
                            y=1.12,
                            xanchor="left",
                            yanchor="top",
                        )
                    ],
                    title=f"Heatmap: {init_metric} by {X_PARAM} (x) and {Y_PARAM} (y)",
                    xaxis=dict(title=X_PARAM, tickmode="array", tickvals=xticks, tickangle=45),
                    yaxis=dict(title=Y_PARAM, tickmode="array", tickvals=yticks),
                    template="plotly_white",
                    height=700,
                    width=900,
                    margin=dict(t=100, l=120, r=80, b=120),
                )

                # simplify colorbar
                fig.data[0].colorbar.thickness = 15
                fig.data[0].colorbar.tickfont.size = 10

                html_out = os.path.join(scan_dir, "interactive_plot.html")
                fig.write_html(html_out)
                print("Wrote interactive heatmap to", html_out)

                # Also produce an interactive scatter-matrix using the scanned dimensions
                try:
                    import plotly.express as px

                    matrix_dims = [d for d in dims if d in df.columns]
                    if matrix_dims:
                        matrix_metric = init_metric if init_metric in df.columns else matrix_dims[0]
                        matrix_fig = px.scatter_matrix(
                            df,
                            dimensions=matrix_dims,
                            color=matrix_metric,
                            title=f"Parameter matrix colored by {matrix_metric}",
                        )
                        matrix_fig.update_layout(template="plotly_white", height=900, width=1200)
                        matrix_out = os.path.join(scan_dir, "interactive_matrix.html")
                        matrix_fig.write_html(matrix_out)
                        print("Wrote interactive matrix to", matrix_out)
                except Exception as e:
                    print("Failed to create interactive matrix:", e)

        except Exception as e:
            print("Failed to create interactive plot:", e)
