#!/usr/bin/env python
"""
Figure 2 (main): Training time per epoch vs dataset size.
Figure SI:       Training speedup vs dataset size (supplementary).

Reads: results/raw/training_timing.csv
Writes:
  results/figures/fig2_training_speedup.{pdf,png}
  results/figures/figSI_training_speedup_vs_size.{pdf,png}
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update(plt.rcParamsDefault)

INPUT = "results/raw/training_timing.csv"
OUTPUT_DIR = "results/figures"

COLORS = {"baseline": "#d62728", "cuik": "#1f77b4"}
LABELS = {"baseline": "Python CGR", "cuik": "cuik-reactmaker (C++)"}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT)

    agg = (
        df.groupby(["path", "n_reactions"])["time_per_epoch_s"]
        .agg(["median", "std"])
        .reset_index()
    )

    base_agg = agg[agg["path"] == "baseline"].set_index("n_reactions")["median"]
    cuik_agg = agg[agg["path"] == "cuik"].set_index("n_reactions")["median"]
    common = sorted(set(base_agg.index) & set(cuik_agg.index))
    speedups = [base_agg[n] / cuik_agg[n] for n in common]

    # ── Main figure: absolute time per epoch ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))

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
    ax.set_xlabel("Number of reactions")
    ax.set_ylabel("Training time per epoch (s)")
    ax.legend()
    ax.grid(False)

    # Annotate speedup: 80% from left, 10% from bottom (axes fraction coords)
    speedup_at_largest = speedups[-1]
    ax.text(
        0.95, 0.10, f"{speedup_at_largest:.1f}× speedup",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=13, fontweight="bold", color="dimgray",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
    )

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig2_training_speedup.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()

    # ── SI figure: speedup vs dataset size ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(common, speedups, marker="o", linewidth=2, color="#2ca02c")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of reactions")
    ax2.set_ylabel("Speedup by cuik-reactmaker")
    ax2.grid(False)
    ax2.xaxis.set_minor_locator(ticker.NullLocator())
    ax2.set_ylim(bottom=1.0, top=max(speedups) * 1.2)

    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "figSI_training_speedup_vs_size.pdf")
    plt.savefig(out2, bbox_inches="tight")
    plt.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close()


if __name__ == "__main__":
    main()
