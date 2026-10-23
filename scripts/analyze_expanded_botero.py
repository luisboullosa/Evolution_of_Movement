#!/usr/bin/env python3
"""
Analyzes the Expanded 6x6 Botero Parameter Sweep (144 runs):
- R in [0.0002, 0.001, 0.005, 0.02, 0.10, 0.40]
- P in [0.01, 0.03, 0.06, 0.12, 0.25, 0.50]
Comparing Heterosperma Plasticity vs Constitutive Motility.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = Path(r"C:\Users\luisf\.gemini\antigravity-ide\brain\d9032cc4-f2ec-4bdb-8a09-a7a788854dfb")

    csv_path = repo_root / "output" / "botero_expanded_grid" / "botero_expanded_results.csv"
    df = pd.read_csv(csv_path)

    print("=" * 85)
    print("      EXPANDED 6x6 BOTERO (2014) SWEEP: STRATEGY FRONTIER ANALYSIS      ")
    print("=" * 85)

    r_vals = [0.0002, 0.001, 0.005, 0.02, 0.10, 0.40]
    p_vals = [0.01, 0.03, 0.06, 0.12, 0.25, 0.50]

    mechanisms = [
        ("Heterosperma Plasticity (Lean->Motile)", "Heterosperma"),
        ("Constitutive Motility (Dev=0)", "Constitutive"),
    ]

    for label, short in mechanisms:
        sub = df[df["mechanism"] == label]
        print(f"\n=======================================================")
        print(f"  {label}")
        print(f"=======================================================")
        piv_mov = sub.pivot_table(index="P", columns="R", values="prop_moved", aggfunc="mean") * 100
        piv_mot = sub.pivot_table(index="P", columns="R", values="prop_motile", aggfunc="mean") * 100
        piv_theta = sub.pivot_table(index="P", columns="R", values="avg_morph_threshold", aggfunc="mean")

        print(f"Overall Mean Moved:    {sub['prop_moved'].mean()*100:.2f}% (std: {sub['prop_moved'].std()*100:.2f}%)")
        print(f"Peak Replicate Moved:  {sub['prop_moved'].max()*100:.2f}%")
        print(f"Overall Mean Motility: {sub['prop_motile'].mean()*100:.2f}%")

        print("\nActive Locomotion Matrix (% Moved):")
        print(piv_mov.round(2))

        if short == "Heterosperma":
            print("\nEvolved Morph Threshold (theta_morph):")
            print(piv_theta.round(4))

    # Plot 2x2 Heatmap Matrix:
    # Top row: Active Movement for Heterosperma vs Constitutive
    # Bottom row: Genetic Components (Evolved theta_morph for Heterosperma, Peak replicate for Constitutive)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

    # Panel 1: Heterosperma Active Locomotion
    sub_h = df[df["mechanism"] == "Heterosperma Plasticity (Lean->Motile)"]
    piv_h_mov = sub_h.pivot_table(index="P", columns="R", values="prop_moved", aggfunc="mean") * 100
    im0 = axes[0, 0].imshow(piv_h_mov.values, cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=10.0)
    axes[0, 0].set_title("Heterosperma Plasticity (Lean -> Motile)\nActive Locomotion (% Steps Moved)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xticks(range(len(r_vals)))
    axes[0, 0].set_xticklabels([f"{r}" for r in r_vals])
    axes[0, 0].set_yticks(range(len(p_vals)))
    axes[0, 0].set_yticklabels([f"{p}" for p in p_vals])
    axes[0, 0].set_xlabel("Timescale of Variation R (Fast ->)")
    axes[0, 0].set_ylabel("Spatial Predictability P (Fine Noise ->)")
    fig.colorbar(im0, ax=axes[0, 0], label="% Active Steps")
    for i in range(len(piv_h_mov.index)):
        for j in range(len(piv_h_mov.columns)):
            v = piv_h_mov.values[i, j]
            axes[0, 0].text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v < 6.0 else "black", fontsize=8, fontweight="bold")

    # Panel 2: Constitutive Active Locomotion
    sub_c = df[df["mechanism"] == "Constitutive Motility (Dev=0)"]
    piv_c_mov = sub_c.pivot_table(index="P", columns="R", values="prop_moved", aggfunc="mean") * 100
    im1 = axes[0, 1].imshow(piv_c_mov.values, cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=10.0)
    axes[0, 1].set_title("Constitutive Motility (Dev Steps = 0)\nActive Locomotion (% Steps Moved)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xticks(range(len(r_vals)))
    axes[0, 1].set_xticklabels([f"{r}" for r in r_vals])
    axes[0, 1].set_yticks(range(len(p_vals)))
    axes[0, 1].set_yticklabels([f"{p}" for p in p_vals])
    axes[0, 1].set_xlabel("Timescale of Variation R (Fast ->)")
    fig.colorbar(im1, ax=axes[0, 1], label="% Active Steps")
    for i in range(len(piv_c_mov.index)):
        for j in range(len(piv_c_mov.columns)):
            v = piv_c_mov.values[i, j]
            axes[0, 1].text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v < 6.0 else "black", fontsize=8, fontweight="bold")

    # Panel 3: Heterosperma Evolved Reaction Norm (theta_morph)
    piv_h_theta = sub_h.pivot_table(index="P", columns="R", values="avg_morph_threshold", aggfunc="mean")
    im2 = axes[1, 0].imshow(piv_h_theta.values, cmap="magma", aspect="auto", origin="lower", vmin=0.240, vmax=0.260)
    axes[1, 0].set_title("Heterosperma Genetic Reaction Norm\nEvolved Starvation Threshold (theta_morph)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(range(len(r_vals)))
    axes[1, 0].set_xticklabels([f"{r}" for r in r_vals])
    axes[1, 0].set_yticks(range(len(p_vals)))
    axes[1, 0].set_yticklabels([f"{p}" for p in p_vals])
    axes[1, 0].set_xlabel("Timescale of Variation R (Fast ->)")
    axes[1, 0].set_ylabel("Spatial Predictability P (Fine Noise ->)")
    fig.colorbar(im2, ax=axes[1, 0], label="Evolved theta_morph (init = 0.25)")
    for i in range(len(piv_h_theta.index)):
        for j in range(len(piv_h_theta.columns)):
            v = piv_h_theta.values[i, j]
            axes[1, 0].text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v < 0.255 else "black", fontsize=8, fontweight="bold")

    # Panel 4: Constitutive Peak Exploration Rate Across Replicates
    piv_c_max = sub_c.pivot_table(index="P", columns="R", values="prop_moved", aggfunc="max") * 100
    im3 = axes[1, 1].imshow(piv_c_max.values, cmap="plasma", aspect="auto", origin="lower", vmin=0, vmax=15.0)
    axes[1, 1].set_title("Constitutive Exploration Frontier\nPeak Replicate Movement Rate (%)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks(range(len(r_vals)))
    axes[1, 1].set_xticklabels([f"{r}" for r in r_vals])
    axes[1, 1].set_yticks(range(len(p_vals)))
    axes[1, 1].set_yticklabels([f"{p}" for p in p_vals])
    axes[1, 1].set_xlabel("Timescale of Variation R (Fast ->)")
    fig.colorbar(im3, ax=axes[1, 1], label="Max % Moved in Cell")
    for i in range(len(piv_c_max.index)):
        for j in range(len(piv_c_max.columns)):
            v = piv_c_max.values[i, j]
            axes[1, 1].text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v < 10.0 else "black", fontsize=8, fontweight="bold")

    plot_path = artifact_dir / "botero_expanded_6x6_heatmaps.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nExpanded 6x6 heatmaps saved to {plot_path}")

if __name__ == "__main__":
    main()
