#!/usr/bin/env python3
"""
Analyzes the 2D Botero parameter sweep results and generates summary tables and heatmaps
comparing architectures (Perceptron vs SpatialBlob) and topologies (Hex Dish vs Ring 1D).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Analyze Botero 2D Sweep Results")
    parser.add_argument("--csv", type=str, default="output/botero_sweep/botero_sweep_results.csv", help="Input CSV")
    parser.add_argument("--outdir", type=str, default="output/botero_sweep/figures", help="Output figures directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: Results CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("                      BOTERO (2014) 2D REGIME SWEEP ANALYSIS                    ")
    print("================================================================================")

    # Summarize across (topology, architecture)
    combos = df[["topology", "architecture"]].drop_duplicates().values.tolist()

    for topo, arch in combos:
        sub = df[(df["topology"] == topo) & (df["architecture"] == arch)]
        print(f"\n=======================================================")
        print(f"  Topology: {topo} | Architecture: {arch}")
        print(f"=======================================================")
        piv_motile = sub.pivot_table(index="P", columns="R", values="prop_motile", aggfunc="mean")
        piv_moved = sub.pivot_table(index="P", columns="R", values="prop_moved", aggfunc="mean")
        piv_thresh = sub.pivot_table(index="P", columns="R", values="avg_morph_threshold", aggfunc="mean")
        piv_pop = sub.pivot_table(index="P", columns="R", values="final_population", aggfunc="mean")

        print("\nMotility Ratio (Prop Motile, F = 1) [%]:")
        print((piv_motile * 100).round(1).to_string())

        print("\nActive Movement Rate (Prop Moved) [%]:")
        print((piv_moved * 100).round(1).to_string())

        print("\nEvolved Morph Threshold (theta_morph, Initial = 0.50):")
        print(piv_thresh.round(4).to_string())

        print("\nFinal Population:")
        print(piv_pop.round(0).to_string())

    # Generate heatmaps
    n_cols = len(combos)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 10), squeeze=False)

    for col, (topo, arch) in enumerate(combos):
        sub = df[(df["topology"] == topo) & (df["architecture"] == arch)]
        piv_motile = sub.pivot_table(index="P", columns="R", values="prop_motile", aggfunc="mean")
        piv_thresh = sub.pivot_table(index="P", columns="R", values="avg_morph_threshold", aggfunc="mean")

        # Upper row: Motility Ratio
        ax_m = axes[0, col]
        vmax_m = max(0.25, piv_motile.values.max() * 1.1)
        im_m = ax_m.imshow(piv_motile.values, cmap="viridis", aspect="auto", origin="lower", vmin=0.0, vmax=vmax_m)
        short_arch = "SpatialBlob" if "spatial" in arch else "Perceptron"
        ax_m.set_title(f"{topo.upper()} | {short_arch}\nMotility Ratio (prop_motile)", fontsize=11, fontweight="bold")
        ax_m.set_xticks(range(len(piv_motile.columns)))
        ax_m.set_xticklabels([f"{c}" for c in piv_motile.columns])
        ax_m.set_yticks(range(len(piv_motile.index)))
        ax_m.set_yticklabels([f"{r}" for r in piv_motile.index])
        ax_m.set_xlabel("Timescale of Variation R")
        if col == 0:
            ax_m.set_ylabel("Predictability P")
        fig.colorbar(im_m, ax=ax_m, label="Motile Fraction (F=1)")

        for i in range(len(piv_motile.index)):
            for j in range(len(piv_motile.columns)):
                val = piv_motile.values[i, j]
                ax_m.text(j, i, f"{val*100:.1f}%", ha="center", va="center", color="white" if val < (vmax_m * 0.6) else "black", fontsize=9, fontweight="bold")

        # Lower row: Evolved Morph Threshold
        ax_t = axes[1, col]
        im_t = ax_t.imshow(piv_thresh.values, cmap="magma", aspect="auto", origin="lower", vmin=0.48, vmax=0.52)
        ax_t.set_title(f"{topo.upper()} | {short_arch}\nEvolved Threshold (theta_morph)", fontsize=11, fontweight="bold")
        ax_t.set_xticks(range(len(piv_thresh.columns)))
        ax_t.set_xticklabels([f"{c}" for c in piv_thresh.columns])
        ax_t.set_yticks(range(len(piv_thresh.index)))
        ax_t.set_yticklabels([f"{r}" for r in piv_thresh.index])
        ax_t.set_xlabel("Timescale of Variation R")
        if col == 0:
            ax_t.set_ylabel("Predictability P")
        fig.colorbar(im_t, ax=ax_t, label="Average theta_morph")

        for i in range(len(piv_thresh.index)):
            for j in range(len(piv_thresh.columns)):
                val = piv_thresh.values[i, j]
                ax_t.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < 0.505 else "black", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plot_file = outdir / "botero_regime_matrix.png"
    plt.savefig(plot_file, dpi=180)
    plt.close()
    print(f"\nFigure saved to {plot_file.resolve()}")


if __name__ == "__main__":
    main()
