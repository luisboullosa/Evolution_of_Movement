#!/usr/bin/env python3
"""
Analyzes the Dispersal Plasticity Comparison (Heterosperma vs Constitutive vs Classic)
and generates summary charts and regime breakdowns.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = Path(r"C:\Users\luisf\.gemini\antigravity-ide\brain\d9032cc4-f2ec-4bdb-8a09-a7a788854dfb")

    csv_path = repo_root / "output" / "dispersal_mechanism_comparison" / "dispersal_comparison_results.csv"
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("      DEVELOPMENTAL DISPERSAL PLASTICITY COMPARISON ANALYSIS")
    print("=" * 80)

    # 1. Overall Mechanism Comparison
    mech_summary = df.groupby("mechanism")[["prop_motile", "prop_moved", "final_population"]].agg(["mean", "std"])
    print("\n--- Overall Summary ---")
    print((mech_summary * 100).round(2))

    # 2. Breakdown by Environmental Regime
    # Identify regime name from job_id
    def get_regime(job_id):
        if "Slow_Broad" in job_id:
            return "Slow_Broad (R=0.005, P=0.03)"
        elif "Med_Med" in job_id:
            return "Med_Med (R=0.02, P=0.06)"
        elif "Fast_Fine" in job_id:
            return "Fast_Fine (R=0.10, P=0.12)"
        elif "Slow_Fine" in job_id:
            return "Slow_Fine (R=0.001, P=0.12)"
        return "Unknown"

    df["regime"] = df["job_id"].apply(get_regime)

    print("\n--- Active Movement Rate (% Moved) by Regime ---")
    piv_moved = df.pivot_table(index="regime", columns="mechanism", values="prop_moved", aggfunc="mean") * 100
    piv_moved_std = df.pivot_table(index="regime", columns="mechanism", values="prop_moved", aggfunc="std") * 100
    print(piv_moved.round(2))

    print("\n--- Motility Ratio (% Motile) by Regime ---")
    piv_motile = df.pivot_table(index="regime", columns="mechanism", values="prop_motile", aggfunc="mean") * 100
    print(piv_motile.round(2))

    # 3. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Plot 1: Overall Mean Active Movement Rate with std
    mechanisms = [
        "Classic Plasticity (Rich->Motile)",
        "Heterosperma (Lean->Motile)",
        "Constitutive (Dev=0)"
    ]
    colors = ["#4A90E2", "#E67E22", "#27AE60"]

    means_mov = [df[df["mechanism"] == m]["prop_moved"].mean() * 100 for m in mechanisms]
    stds_mov = [df[df["mechanism"] == m]["prop_moved"].std() * 100 for m in mechanisms]
    short_names = [
        "Classic Plasticity\n(Rich -> Motile)\n[Baseline]",
        "Heterosperma\n(Lean -> Motile)\n[Dispersal Morph]",
        "Constitutive\n(Dev Steps = 0)\n[Unconditional Flagella]"
    ]

    bars = ax1.bar(range(len(mechanisms)), means_mov, yerr=stds_mov, color=colors, alpha=0.85, edgecolor="#222", width=0.55, capsize=5)
    ax1.set_xticks(range(len(mechanisms)))
    ax1.set_xticklabels(short_names, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Active Locomotion (% steps taken)", fontsize=11, fontweight="bold")
    ax1.set_title("Active Movement Rate Across Developmental Mechanisms", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, m, s in zip(bars, means_mov, stds_mov):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.15, f"{m:.2f}%\n(±{s:.2f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Plot 2: Active Movement Grouped by Regime
    regimes_order = [
        "Slow_Broad (R=0.005, P=0.03)",
        "Med_Med (R=0.02, P=0.06)",
        "Slow_Fine (R=0.001, P=0.12)",
        "Fast_Fine (R=0.10, P=0.12)"
    ]
    short_regimes = ["Slow Broad\n(R=0.005, P=0.03)", "Med Med\n(R=0.02, P=0.06)", "Slow Fine\n(R=0.001, P=0.12)", "Fast Fine\n(R=0.10, P=0.12)"]
    x = np.arange(len(regimes_order))
    width = 0.25

    for i, (m, col) in enumerate(zip(mechanisms, colors)):
        y_vals = [piv_moved.loc[r, m] for r in regimes_order]
        y_errs = [piv_moved_std.loc[r, m] for r in regimes_order]
        label = m.split("(")[0].strip()
        ax2.bar(x + (i - 1) * width, y_vals, yerr=y_errs, width=width, label=label, color=col, alpha=0.85, edgecolor="#222", capsize=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_regimes, fontsize=9, fontweight="bold")
    ax2.set_ylabel("Active Locomotion (% steps taken)", fontsize=11, fontweight="bold")
    ax2.set_title("Active Movement by Environmental Regime", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, facecolor="#fafafa", edgecolor="#ddd", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plot_path = artifact_dir / "heterosperma_vs_constitutive_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nFigure saved to {plot_path}")

if __name__ == "__main__":
    main()
