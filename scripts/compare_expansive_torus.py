#!/usr/bin/env python3
"""
Compares the Expansive Torus Botero Sweep (R=35, N=3781, MLP-2) against
the Open Dish baseline (R=25, N=1951, MLP-2) and previous architectures.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = Path(r"C:\Users\luisf\.gemini\antigravity-ide\brain\d9032cc4-f2ec-4bdb-8a09-a7a788854dfb")

    torus_csv = repo_root / "output" / "botero_sweep_mlp_expansive_torus" / "botero_sweep_results.csv"
    open_csv = repo_root / "output" / "botero_sweep_mlp_run_tumble" / "botero_sweep_results.csv"
    spatial_csv = repo_root / "output" / "botero_sweep_spatial" / "botero_sweep_results.csv"
    percep_csv = repo_root / "output" / "botero_sweep" / "botero_sweep_results.csv"

    print("=" * 80)
    print("      EXPANSIVE TORUS BOTERO SWEEP ANALYSIS (R=35, N=3781, MLP-2)     ")
    print("=" * 80)

    df_torus = pd.read_csv(torus_csv)
    df_open = pd.read_csv(open_csv)

    print("\n--- 1. Expansive Torus (R=35, Torus, MLP-2, N=3781) ---")
    piv_torus_motile = df_torus.pivot_table(index="P", columns="R", values="prop_motile", aggfunc="mean") * 100
    piv_torus_moved = df_torus.pivot_table(index="P", columns="R", values="prop_moved", aggfunc="mean") * 100
    piv_torus_pop = df_torus.pivot_table(index="P", columns="R", values="final_population", aggfunc="mean")

    print("\nMotility Ratio [%]:")
    print(piv_torus_motile.round(2))
    print(f"\nOverall Mean Motility: {df_torus['prop_motile'].mean()*100:.2f}% (std: {df_torus['prop_motile'].std()*100:.2f}%)")

    print("\nActive Movement Rate [%]:")
    print(piv_torus_moved.round(2))
    print(f"\nOverall Mean Moved: {df_torus['prop_moved'].mean()*100:.2f}% (std: {df_torus['prop_moved'].std()*100:.2f}%)")

    print("\nFinal Population:")
    print(piv_torus_pop.round(0))
    print(f"Overall Mean Pop: {df_torus['final_population'].mean():.1f} / 1930 cap ({df_torus['final_population'].mean() / 3781 * 100:.1f}% density)")

    print("\n" + "=" * 80)
    print("--- 2. Open Dish Baseline (R=25, Open, MLP-2, N=1951) ---")
    piv_open_motile = df_open.pivot_table(index="P", columns="R", values="prop_motile", aggfunc="mean") * 100
    piv_open_moved = df_open.pivot_table(index="P", columns="R", values="prop_moved", aggfunc="mean") * 100
    piv_open_pop = df_open.pivot_table(index="P", columns="R", values="final_population", aggfunc="mean")

    print("\nMotility Ratio [%]:")
    print(piv_open_motile.round(2))
    print(f"\nOverall Mean Motility: {df_open['prop_motile'].mean()*100:.2f}% (std: {df_open['prop_motile'].std()*100:.2f}%)")

    print("\nActive Movement Rate [%]:")
    print(piv_open_moved.round(2))
    print(f"\nOverall Mean Moved: {df_open['prop_moved'].mean()*100:.2f}% (std: {df_open['prop_moved'].std()*100:.2f}%)")

    print("\n" + "=" * 80)
    print("--- 3. Delta (Expansive Torus - Open Dish) ---")
    delta_motile = piv_torus_motile - piv_open_motile
    delta_moved = piv_torus_moved - piv_open_moved

    print("\nDelta Motility Ratio [% points]:")
    print(delta_motile.round(2))

    print("\nDelta Active Movement Rate [% points]:")
    print(delta_moved.round(2))

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    # Styling
    plt.rcParams["font.family"] = "sans-serif"

    # Heatmap 1: Open Dish Motility
    im0 = axes[0, 0].imshow(piv_open_motile.values, cmap="YlGnBu", aspect="auto", origin="lower", vmin=0, vmax=3.5)
    axes[0, 0].set_title("Open Dish (R=25, Open Wall)\nMotility Ratio (prop_motile %)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xticks(range(len(piv_open_motile.columns)))
    axes[0, 0].set_xticklabels([f"{c}" for c in piv_open_motile.columns])
    axes[0, 0].set_yticks(range(len(piv_open_motile.index)))
    axes[0, 0].set_yticklabels([f"{r}" for r in piv_open_motile.index])
    axes[0, 0].set_xlabel("Timescale of Variation R")
    axes[0, 0].set_ylabel("Predictability P")
    fig.colorbar(im0, ax=axes[0, 0], label="% Motile")
    for i in range(len(piv_open_motile.index)):
        for j in range(len(piv_open_motile.columns)):
            v = piv_open_motile.values[i, j]
            axes[0, 0].text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v > 2.0 else "black", fontsize=9, fontweight="bold")

    # Heatmap 2: Expansive Torus Motility
    im1 = axes[0, 1].imshow(piv_torus_motile.values, cmap="YlGnBu", aspect="auto", origin="lower", vmin=0, vmax=3.5)
    axes[0, 1].set_title("Expansive Torus (R=35, Periodic)\nMotility Ratio (prop_motile %)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xticks(range(len(piv_torus_motile.columns)))
    axes[0, 1].set_xticklabels([f"{c}" for c in piv_torus_motile.columns])
    axes[0, 1].set_yticks(range(len(piv_torus_motile.index)))
    axes[0, 1].set_yticklabels([f"{r}" for r in piv_torus_motile.index])
    axes[0, 1].set_xlabel("Timescale of Variation R")
    fig.colorbar(im1, ax=axes[0, 1], label="% Motile")
    for i in range(len(piv_torus_motile.index)):
        for j in range(len(piv_torus_motile.columns)):
            v = piv_torus_motile.values[i, j]
            axes[0, 1].text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v > 2.0 else "black", fontsize=9, fontweight="bold")

    # Heatmap 3: Delta Motility
    vmax_d = max(abs(delta_motile.values.min()), abs(delta_motile.values.max()), 1.0)
    im2 = axes[0, 2].imshow(delta_motile.values, cmap="coolwarm", aspect="auto", origin="lower", vmin=-vmax_d, vmax=vmax_d)
    axes[0, 2].set_title("Motility Difference\n(Torus - Open Dish)", fontsize=11, fontweight="bold")
    axes[0, 2].set_xticks(range(len(delta_motile.columns)))
    axes[0, 2].set_xticklabels([f"{c}" for c in delta_motile.columns])
    axes[0, 2].set_yticks(range(len(delta_motile.index)))
    axes[0, 2].set_yticklabels([f"{r}" for r in delta_motile.index])
    axes[0, 2].set_xlabel("Timescale of Variation R")
    fig.colorbar(im2, ax=axes[0, 2], label="Difference (% points)")
    for i in range(len(delta_motile.index)):
        for j in range(len(delta_motile.columns)):
            v = delta_motile.values[i, j]
            axes[0, 2].text(j, i, f"{v:+.1f}%", ha="center", va="center", color="white" if abs(v) > 0.8 else "black", fontsize=9, fontweight="bold")

    # Row 2: Active Movement Rate (Moved)
    # Heatmap 4: Open Dish Moved
    im3 = axes[1, 0].imshow(piv_open_moved.values, cmap="magma", aspect="auto", origin="lower", vmin=0, vmax=2.5)
    axes[1, 0].set_title("Open Dish (R=25, Open Wall)\nActive Movement (prop_moved %)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(range(len(piv_open_moved.columns)))
    axes[1, 0].set_xticklabels([f"{c}" for c in piv_open_moved.columns])
    axes[1, 0].set_yticks(range(len(piv_open_moved.index)))
    axes[1, 0].set_yticklabels([f"{r}" for r in piv_open_moved.index])
    axes[1, 0].set_xlabel("Timescale of Variation R")
    axes[1, 0].set_ylabel("Predictability P")
    fig.colorbar(im3, ax=axes[1, 0], label="% Moved")
    for i in range(len(piv_open_moved.index)):
        for j in range(len(piv_open_moved.columns)):
            v = piv_open_moved.values[i, j]
            axes[1, 0].text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v < 1.8 else "black", fontsize=9, fontweight="bold")

    # Heatmap 5: Expansive Torus Moved
    im4 = axes[1, 1].imshow(piv_torus_moved.values, cmap="magma", aspect="auto", origin="lower", vmin=0, vmax=2.5)
    axes[1, 1].set_title("Expansive Torus (R=35, Periodic)\nActive Movement (prop_moved %)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks(range(len(piv_torus_moved.columns)))
    axes[1, 1].set_xticklabels([f"{c}" for c in piv_torus_moved.columns])
    axes[1, 1].set_yticks(range(len(piv_torus_moved.index)))
    axes[1, 1].set_yticklabels([f"{r}" for r in piv_torus_moved.index])
    axes[1, 1].set_xlabel("Timescale of Variation R")
    fig.colorbar(im4, ax=axes[1, 1], label="% Moved")
    for i in range(len(piv_torus_moved.index)):
        for j in range(len(piv_torus_moved.columns)):
            v = piv_torus_moved.values[i, j]
            axes[1, 1].text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v < 1.8 else "black", fontsize=9, fontweight="bold")

    # Heatmap 6: Delta Moved
    vmax_dm = max(abs(delta_moved.values.min()), abs(delta_moved.values.max()), 1.0)
    im5 = axes[1, 2].imshow(delta_moved.values, cmap="coolwarm", aspect="auto", origin="lower", vmin=-vmax_dm, vmax=vmax_dm)
    axes[1, 2].set_title("Active Movement Difference\n(Torus - Open Dish)", fontsize=11, fontweight="bold")
    axes[1, 2].set_xticks(range(len(delta_moved.columns)))
    axes[1, 2].set_xticklabels([f"{c}" for c in delta_moved.columns])
    axes[1, 2].set_yticks(range(len(delta_moved.index)))
    axes[1, 2].set_yticklabels([f"{r}" for r in delta_moved.index])
    axes[1, 2].set_xlabel("Timescale of Variation R")
    fig.colorbar(im5, ax=axes[1, 2], label="Difference (% points)")
    for i in range(len(delta_moved.index)):
        for j in range(len(delta_moved.columns)):
            v = delta_moved.values[i, j]
            axes[1, 2].text(j, i, f"{v:+.1f}%", ha="center", va="center", color="white" if abs(v) > 0.6 else "black", fontsize=9, fontweight="bold")

    plot_path = artifact_dir / "expansive_torus_botero_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nFigure saved to {plot_path}")

if __name__ == "__main__":
    main()
