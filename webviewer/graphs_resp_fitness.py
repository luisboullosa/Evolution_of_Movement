import numpy as np


def setup_resp_axis(fig, gs):
    ax_resp = fig.add_subplot(gs[2, 0])

    return ax_resp


def init_response(ax_resp, xs):
    # Create five scenario curves: peak, slope right, slope left, sink, flat
    (ln_peak,) = ax_resp.plot(xs, np.zeros_like(xs), label="peak")
    (ln_slope_r,) = ax_resp.plot(xs, np.zeros_like(xs), label="slope→")
    (ln_slope_l,) = ax_resp.plot(xs, np.zeros_like(xs), label="slope←")
    (ln_sink,) = ax_resp.plot(xs, np.zeros_like(xs), label="sink")
    (ln_flat,) = ax_resp.plot(xs, np.zeros_like(xs), label="flat")
    # Additional set of curves simulating heavy local occupancy (e.g., 1000 others at same position)
    (ln_peak_full,) = ax_resp.plot(xs, np.zeros_like(xs), label="peak_full", linestyle="--")
    (ln_slope_r_full,) = ax_resp.plot(xs, np.zeros_like(xs), label="slope→_full", linestyle="--")
    (ln_slope_l_full,) = ax_resp.plot(xs, np.zeros_like(xs), label="slope←_full", linestyle="--")
    (ln_sink_full,) = ax_resp.plot(xs, np.zeros_like(xs), label="sink_full", linestyle="--")
    (ln_flat_full,) = ax_resp.plot(xs, np.zeros_like(xs), label="flat_full", linestyle="--")

    # vertical marker and dot for current input
    ln_input_vline = ax_resp.axvline(xs[0], color="0.2", linestyle="--")
    (ln_input_dot,) = ax_resp.plot([xs[0]], [0.0], marker="o", color="0.0")

    # add a small legend identifying scenarios
    try:
        ax_resp.legend(loc="upper right", fontsize=8)
    except Exception:
        pass

    return (
        (
            (ln_peak, ln_slope_r, ln_slope_l, ln_sink, ln_flat),
            (ln_peak_full, ln_slope_r_full, ln_slope_l_full, ln_sink_full, ln_flat_full),
        ),
        ln_input_vline,
        ln_input_dot,
    )


def setup_fitness_axis(ax_fitness, bins, abs0, max_count):
    """Draw initial absolute-resources histogram and return patches + marker."""
    try:
        abs_counts, _ = np.histogram(abs0, bins=bins) if abs0.size > 0 else (np.zeros(len(bins) - 1, dtype=int), bins)
        bin_lefts = bins[:-1]
        bin_widths: np.ndarray[tuple[Any, ...], np.dtype[np.Any]] = np.diff(bins)
        abs_bar_container = ax_fitness.bar(
            bin_lefts,
            abs_counts,
            width=bin_widths,
            align="edge",
            color="0.4",
            edgecolor="k",
            alpha=0.8,
            label="resources",
        )
        try:
            abs_bar_patches = abs_bar_container.patches
        except Exception:
            abs_bar_patches = []
        ax_fitness.set_xlim(bins[0], bins[-1])
        ax_fitness.set_ylim(0, max_count * 1.1)
        # ax_fitness.legend(fontsize=8)
    except Exception:
        abs_bar_patches = []

    try:
        ln_marker = ax_fitness.axvline(0.0, color="0.0", linestyle="--", linewidth=2)
    except Exception:
        ln_marker = None

    return abs_bar_patches, ln_marker


def update_fitness_axis(
    ax_fitness,
    abs_bar_patches,
    ln_marker,
    new_abs_counts,
    resources,
    bins,
    max_count,
    times_i,
):
    # Update the histogram bar heights and the vertical marker for individual.
    try:
        for p, h in zip(abs_bar_patches, new_abs_counts):
            p.set_height(h)
    except Exception:
        try:
            if ln_marker is not None:
                ln_marker.remove()
        except Exception:
            pass
        ax_fitness.cla()
        if new_abs_counts.sum() > 0:
            abs_bar_container = ax_fitness.bar(
                bins[:-1],
                new_abs_counts,
                width=np.diff(bins),
                align="edge",
                color="0.4",
                edgecolor="k",
                alpha=0.8,
            )
            try:
                abs_bar_patches = abs_bar_container.patches
            except Exception:
                abs_bar_patches = []
        else:
            abs_bar_patches = []
        ax_fitness.set_xlim(bins[0], bins[-1])
        ax_fitness.set_ylim(0, max_count * 1.1)
        try:
            ln_marker = ax_fitness.axvline(resources, color="0.0", linestyle="--", linewidth=2)
        except Exception:
            ln_marker = None
        ax_fitness.set_title(f"Resources distribution (t={times_i})")

        return abs_bar_patches, ln_marker

    try:
        if ln_marker is None:
            ln_marker = ax_fitness.axvline(resources, color="0.0", linestyle="--", linewidth=2)
        else:
            ln_marker.set_xdata([resources, resources])
    except Exception:
        try:
            ln_marker.remove()
        except Exception:
            pass
        try:
            ln_marker = ax_fitness.axvline(resources, color="0.0", linestyle="--", linewidth=2)
        except Exception:
            ln_marker = None

    ax_fitness.set_title(f"Resources distribution (t={times_i})")

    return abs_bar_patches, ln_marker


def update_response_axis(
    ax_resp,
    ln_resp,
    ln_input_vline,
    ln_input_dot,
    xs,
    ys_display,
    curr_x,
    resp_abs_mode,
    n_nodes,
):
    # ln_resp may be a sequence of 5 lines (one per scenario) or a single line.
    try:
        # ln_resp may be nested ((standard_tuple), (full_tuple)) or a flat sequence
        if isinstance(ln_resp, (list, tuple)):
            # handle nested case
            if len(ln_resp) == 2 and all(isinstance(x, (list, tuple)) for x in ln_resp):
                stds, fulls = ln_resp
                # standard scenarios
                stds[0].set_ydata(ys_display.get("peak", np.zeros_like(xs)))
                stds[1].set_ydata(ys_display.get("slope_r", np.zeros_like(xs)))
                stds[2].set_ydata(ys_display.get("slope_l", np.zeros_like(xs)))
                stds[3].set_ydata(ys_display.get("sink", np.zeros_like(xs)))
                stds[4].set_ydata(ys_display.get("flat", np.zeros_like(xs)))
                # full-occupancy scenarios (keys with _full suffix)
                fulls[0].set_ydata(ys_display.get("peak_full", np.zeros_like(xs)))
                fulls[1].set_ydata(ys_display.get("slope_r_full", np.zeros_like(xs)))
                fulls[2].set_ydata(ys_display.get("slope_l_full", np.zeros_like(xs)))
                fulls[3].set_ydata(ys_display.get("sink_full", np.zeros_like(xs)))
                fulls[4].set_ydata(ys_display.get("flat_full", np.zeros_like(xs)))
            elif len(ln_resp) == 5:
                # backward-compatible: single set of 5 lines
                ln_resp[0].set_ydata(ys_display.get("peak", np.zeros_like(xs)))
                ln_resp[1].set_ydata(ys_display.get("slope_r", np.zeros_like(xs)))
                ln_resp[2].set_ydata(ys_display.get("slope_l", np.zeros_like(xs)))
                ln_resp[3].set_ydata(ys_display.get("sink", np.zeros_like(xs)))
                ln_resp[4].set_ydata(ys_display.get("flat", np.zeros_like(xs)))
            else:
                # unexpected shape, attempt best-effort
                try:
                    rep = ys_display.get("peak", np.zeros_like(xs))
                except Exception:
                    rep = np.zeros_like(xs)
                try:
                    ln_resp.set_ydata(rep)
                except Exception:
                    pass
        else:
            # legacy single-line usage: accept an array
            ln_resp.set_ydata(ys_display)
    except Exception:
        pass

    try:
        ln_input_vline.set_xdata([curr_x, curr_x])
        # pick a representative ys to interpolate the dot; prefer 'peak' if available
        try:
            rep = ys_display.get("peak", None) if isinstance(ys_display, dict) else ys_display
        except Exception:
            rep = ys_display
        ln_input_dot.set_data([curr_x], [np.interp(curr_x, xs, rep)])
    except Exception:
        pass
    y_margin = 0.1
    y_lim: float = max(1.0, n_nodes)
    if resp_abs_mode:
        ax_resp.set_ylim(0 - y_margin, y_lim + y_margin)
    else:
        ax_resp.set_ylim(-y_lim - y_margin, y_lim + y_margin)
    ax_resp.relim()
    ax_resp.autoscale_view(False)


def prepare_fitness_data(ind_list, nbins=31):
    """Compute fitness arrays and fixed histogram bins/limits for the whole simulation.

    Returns:
        Tuple containing:
        - all_abs_fitnesses
        - all_rel_fitnesses
        - abs_fitness_by_frame
        - rel_fitness_by_frame
        - bins
        - max_count
        - existence_map
    """
    all_abs_fitnesses = []
    all_rel_fitnesses = []
    existence_map = {}

    # collect absolute and relative fitness across frames and record existence ranges
    for fi, ind in enumerate(ind_list):
        res_arr = ind.get("resources", np.array([]))
        total: float = float(np.sum(res_arr)) if res_arr.size > 0 else 0.0
        for iid in ind.get("ids", []):
            if iid not in existence_map:
                existence_map[iid] = [fi, fi]
            else:
                existence_map[iid][1] = fi
        for r in res_arr:
            try:
                rval = float(r)
                all_abs_fitnesses.append(rval)
                all_rel_fitnesses.append(rval / total if total > 0.0 else 0.0)
            except Exception:
                all_abs_fitnesses.append(0.0)
                all_rel_fitnesses.append(0.0)

    # per-frame arrays
    abs_fitness_by_frame = []
    rel_fitness_by_frame = []
    for ind in ind_list:
        res_arr = ind.get("resources", np.array([]))
        total: float = float(np.sum(res_arr)) if res_arr.size > 0 else 0.0
        abs_fitness_by_frame.append(np.array([float(r) for r in res_arr]) if res_arr.size > 0 else np.array([]))
        if res_arr.size > 0:
            rel_fitness_by_frame.append(np.array([float(r) / total if total > 0.0 else 0.0 for r in res_arr]))
        else:
            rel_fitness_by_frame.append(np.array([]))

    # prepare fixed histogram bins and y-range
    overall_vals = []
    if len(all_rel_fitnesses) > 0:
        overall_vals.extend(all_rel_fitnesses)
    if len(all_abs_fitnesses) > 0:
        overall_vals.extend(all_abs_fitnesses)
    if len(overall_vals) == 0:
        hist_min, hist_max = 0.0, 1.0
    else:
        hist_min = float(np.min(overall_vals))
        hist_max = float(np.max(overall_vals))
        if hist_min == hist_max:
            hist_min -= 0.5
            hist_max += 0.5
    hist_pad: float = (hist_max - hist_min) * 0.05 if hist_max > hist_min else 0.5
    bins: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = np.linspace(hist_min - hist_pad, hist_max + hist_pad, nbins)

    # compute max bin count across frames
    max_bin = 1
    for rel_arr, abs_arr in zip(rel_fitness_by_frame, abs_fitness_by_frame):
        try:
            cnt_rel, _ = (
                np.histogram(rel_arr, bins=bins) if rel_arr.size > 0 else (np.zeros(len(bins) - 1, dtype=int), bins)
            )
            cnt_abs, _ = (
                np.histogram(abs_arr, bins=bins) if abs_arr.size > 0 else (np.zeros(len(bins) - 1, dtype=int), bins)
            )
            max_bin: int = max(
                max_bin,
                int(
                    max(
                        cnt_rel.max() if cnt_rel.size > 0 else 0,
                        cnt_abs.max() if cnt_abs.size > 0 else 0,
                    )
                ),
            )
        except Exception:
            pass
    max_count: int = max(1, max_bin)

    return (
        all_abs_fitnesses,
        all_rel_fitnesses,
        abs_fitness_by_frame,
        rel_fitness_by_frame,
        bins,
        max_count,
        existence_map,
    )
