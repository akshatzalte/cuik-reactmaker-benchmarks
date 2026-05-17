#!/usr/bin/env python
"""
Figure 2: Training time per epoch vs dataset size.

Reads: results/raw/training_timing.csv
Writes: results/figures/fig2_training_speedup.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

INPUT = "results/raw/training_timing.csv"
OUTPUT_DIR = "results/figures"

COLORS = {"baseline": "#d62728", "cuik": "#1f77b4"}
LABELS = {"baseline": "Python CGR (baseline)", "cuik": "C++ CGR (cuik-molmaker)"}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT)

    # Aggregate across seeds: median ± std per (path, n_reactions)
    agg = (
        df.groupby(["path", "n_reactions"])["time_per_epoch_s"]
        .agg(["median", "std"])
        .reset_index()
    )

    sizes = sorted(agg["n_reactions"].unique())
    x_labels = [f"{n // 1000}k" for n in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left: absolute time per epoch ---
    ax = axes[0]
    for path in ["baseline", "cuik"]:
        sub = agg[agg["path"] == path].sort_values("n_reactions")
        ax.errorbar(
            sub["n_reactions"], sub["median"],
            yerr=sub["std"],
            marker="o", linewidth=2, capsize=3,
            color=COLORS[path], label=LABELS[path],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size (reactions)", fontsize=12)
    ax.set_ylabel("Time per epoch (s)", fontsize=12)
    ax.set_title("Training time per epoch", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(sizes)
    ax.set_xticklabels(x_labels)

    # --- Right: speedup vs dataset size ---
    ax = axes[1]
    base_agg = agg[agg["path"] == "baseline"].set_index("n_reactions")["median"]
    cuik_agg = agg[agg["path"] == "cuik"].set_index("n_reactions")["median"]
    common = sorted(set(base_agg.index) & set(cuik_agg.index))

    speedups = [base_agg[n] / cuik_agg[n] for n in common]
    common_labels = [f"{n // 1000}k" for n in common]

    ax.plot(common, speedups, marker="o", linewidth=2, color="#2ca02c")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Dataset size (reactions)", fontsize=12)
    ax.set_ylabel("Training speedup (baseline / cuik)", fontsize=12)
    ax.set_title("End-to-end training speedup", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(common)
    ax.set_xticklabels(common_labels)
    ax.set_ylim(bottom=0)

    # Annotate each point
    for x, y in zip(common, speedups):
        ax.annotate(f"{y:.1f}×", xy=(x, y), xytext=(x, y + 0.05),
                    ha="center", fontsize=9, color="#2ca02c")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig2_training_speedup.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
