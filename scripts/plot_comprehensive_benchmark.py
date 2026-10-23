#!/usr/bin/env python3
"""
Generates a comprehensive summary chart across all Hex 2D Botero sweep experiments.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = Path(r"C:\Users\luisf\.gemini\antigravity-ide\brain\d9032cc4-f2ec-4bdb-8a09-a7a788854dfb")

    configs = [
        ("Perceptron (Baseline)", repo_root / "output/botero_sweep/botero_sweep_results.csv", "perceptron_only", "hex2d"),
        ("SpatialBlob", repo_root / "output/botero_sweep/botero_sweep_results.csv", "evolve_spatial_graph", "hex2d"),
        ("MLP-8 (Unconstrained)", repo_root / "output/botero_sweep_mlp/botero_sweep_results.csv", "evolve_mlp", "hex2d"),
        ("MLP-2 (Bottleneck)", repo_root / "output/botero_sweep_mlp_2node/botero_sweep_results.csv", "evolve_mlp", "hex2d"),
        ("MLP-2 + Run&Tumble (Open R=25)", repo_root / "output/botero_sweep_mlp_run_tumble/botero_sweep_results.csv", "evolve_mlp", "hex2d"),
        ("MLP-2 + Expansive Torus (R=35)", repo_root / "output/botero_sweep_mlp_expansive_torus/botero_sweep_results.csv", "evolve_mlp", "hex2d"),
    ]

    names = []
    mot_means = []
    mot_stds = []
    mov_means = []
    mov_stds = []

    for label, path, arch, topo in configs:
        df = pd.read_csv(path)
        sub = df[(df["architecture"] == arch)]
        if "topology" in sub.columns:
            sub = sub[sub["topology"] == topo]
        names.append(label)
        mot_means.append(sub["prop_motile"].mean() * 100)
        mot_stds.append(sub["prop_motile"].std() * 100)
        mov_means.append(sub["prop_moved"].mean() * 100)
        mov_stds.append(sub["prop_moved"].std() * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    y_pos = np.arange(len(names))
    colors_mot = ["#4A90E2", "#50E3C2", "#B8E986", "#F5A623", "#D0021B", "#9013FE"]
    colors_mov = ["#357ABD", "#38B293", "#8EBE5A", "#C68213", "#A50013", "#6F0FA8"]

    # Motility Ratio (F = 1)
    bars1 = ax1.barh(y_pos, mot_means, xerr=mot_stds, color=colors_mot, alpha=0.85, edgecolor="#333", height=0.6, capsize=4)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean Motile Fraction (% of population with F = 1)", fontsize=10, fontweight="bold")
    ax1.set_title("Motility Ratio Across Model Configurations (Hex 2D)", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    for i, bar in enumerate(bars1):
        ax1.text(bar.get_width() + mot_stds[i] + 0.08, bar.get_y() + bar.get_height()/2, f"{mot_means[i]:.2f}%", va="center", fontsize=9, fontweight="bold")

    # Active Movement Rate (Moved)
    bars2 = ax2.barh(y_pos, mov_means, xerr=mov_stds, color=colors_mov, alpha=0.85, edgecolor="#333", height=0.6, capsize=4)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * len(names))
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Active Movement Rate (% steps taken)", fontsize=10, fontweight="bold")
    ax2.set_title("Active Locomotion Across Model Configurations (Hex 2D)", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    for i, bar in enumerate(bars2):
        ax2.text(bar.get_width() + mov_stds[i] + 0.05, bar.get_y() + bar.get_height()/2, f"{mov_means[i]:.2f}%", va="center", fontsize=9, fontweight="bold")

    plot_path = artifact_dir / "hex2d_6config_benchmark.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Summary figure saved to {plot_path}")

if __name__ == "__main__":
    main()
