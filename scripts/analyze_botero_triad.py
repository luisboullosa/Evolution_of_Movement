#!/usr/bin/env python3
"""
Comprehensive 3-Way Botero (2014) Analysis:
Compares the full 4x4 R x P regime matrices across:
1. Classic Plasticity (Baseline, Rich -> Motile, Dev=10)
2. Heterosperma Plasticity (Lean -> Motile, Dev=10)
3. Constitutive Motility (Dev=0)
All on the Expansive Torus (R=35, N=3,781, MLP-2, 100k steps).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = Path(r"C:\Users\luisf\.gemini\antigravity-ide\brain\d9032cc4-f2ec-4bdb-8a09-a7a788854dfb")

    path_classic = repo_root / "output" / "botero_sweep_mlp_expansive_torus" / "botero_sweep_results.csv"
    path_hetero = repo_root / "output" / "botero_sweep_heterosperma" / "botero_sweep_results.csv"
    path_const = repo_root / "output" / "botero_sweep_constitutive" / "botero_sweep_results.csv"

    df_classic = pd.read_csv(path_classic)
    df_hetero = pd.read_csv(path_hetero)
    df_const = pd.read_csv(path_const)

    print("=" * 85)
    print("      FULL BOTERO 4x4 COMPARISON: CLASSIC vs HETEROSPERMA vs CONSTITUTIVE      ")
    print("=" * 85)

    triad = [
        ("Classic Plasticity (Rich->Motile)", df_classic),
        ("Heterosperma (Lean->Motile)", df_hetero),
        ("Constitutive Motility (Dev=0)", df_const),
    ]

    for name, df in triad:
        print(f"\n=======================================================")
        print(f"  {name}")
        print(f"=======================================================")
        piv_mot = df.pivot_table(index="P", columns="R", values="prop_motile", aggfunc="mean") * 100
        piv_mov = df.pivot_table(index="P", columns="R", values="prop_moved", aggfunc="mean") * 100

        print(f"Overall Motility: {df['prop_motile'].mean()*100:.2f}% (std: {df['prop_motile'].std()*100:.2f}%)")
        print(f"Overall Moved:    {df['prop_moved'].mean()*100:.2f}% (std: {df['prop_moved'].std()*100:.2f}%)")
        print(f"Peak Replicate:   {df['prop_moved'].max()*100:.2f}%")

        print("\nActive Movement Rate Matrix (% Moved):")
        print(piv_mov.round(2))

    # Generate 2 rows x 3 columns of heatmaps
    # Row 1: Motile Fraction (%)
    # Row 2: Active Movement Rate (% Moved)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    titles = [
        "Classic Plasticity\n(Rich -> Motile)",
        "Heterosperma Plasticity\n(Lean -> Motile)",
        "Constitutive Motility\n(Dev Steps = 0)"
    ]

    r_labels = [0.001, 0.005, 0.02, 0.10]
    p_labels = [0.03, 0.06, 0.09, 0.12]

    # Row 1: Motility Fraction
    for col, (name, df) in enumerate(triad):
        piv_mot = df.pivot_table(index="P", columns="R", values="prop_motile", aggfunc="mean") * 100
        ax = axes[0, col]
        vmax_m = 100.0 if col > 0 else 3.5
        cmap_m = "Blues" if col == 0 else "YlGnBu"
        im = ax.imshow(piv_mot.values, cmap=cmap_m, aspect="auto", origin="lower", vmin=0, vmax=vmax_m)
        ax.set_title(f"{titles[col]}\nMotile Fraction (F = 1 %)", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(r_labels)))
        ax.set_xticklabels([f"{c}" for c in r_labels])
        ax.set_yticks(range(len(p_labels)))
        ax.set_yticklabels([f"{r}" for r in p_labels])
        ax.set_xlabel("Timescale of Variation R")
        if col == 0:
            ax.set_ylabel("Predictability P")
        fig.colorbar(im, ax=ax, label="% Flagellated")
        for i in range(len(piv_mot.index)):
            for j in range(len(piv_mot.columns)):
                v = piv_mot.values[i, j]
                txt = f"{v:.1f}%"
                ax.text(j, i, txt, ha="center", va="center", color="white" if v > vmax_m * 0.6 else "black", fontsize=9, fontweight="bold")

    # Row 2: Active Movement Rate
    vmax_mov = max(df_classic["prop_moved"].max(), df_hetero["prop_moved"].max(), df_const["prop_moved"].max()) * 100
    vmax_mov = 7.0  # Shared scale for crisp comparison

    for col, (name, df) in enumerate(triad):
        piv_mov = df.pivot_table(index="P", columns="R", values="prop_moved", aggfunc="mean") * 100
        ax = axes[1, col]
        im = ax.imshow(piv_mov.values, cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=vmax_mov)
        ax.set_title(f"{titles[col]}\nActive Locomotion (% Moved)", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(r_labels)))
        ax.set_xticklabels([f"{c}" for c in r_labels])
        ax.set_yticks(range(len(p_labels)))
        ax.set_yticklabels([f"{r}" for r in p_labels])
        ax.set_xlabel("Timescale of Variation R")
        if col == 0:
            ax.set_ylabel("Predictability P")
        fig.colorbar(im, ax=ax, label="% Active Steps")
        for i in range(len(piv_mov.index)):
            for j in range(len(piv_mov.columns)):
                v = piv_mov.values[i, j]
                txt = f"{v:.2f}%"
                ax.text(j, i, txt, ha="center", va="center", color="white" if v < vmax_mov * 0.6 else "black", fontsize=9, fontweight="bold")

    plot_path = artifact_dir / "botero_3way_developmental_heatmaps.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nFigure saved to {plot_path}")

if __name__ == "__main__":
    main()
