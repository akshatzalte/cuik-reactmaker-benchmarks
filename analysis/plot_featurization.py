#!/usr/bin/env python
"""
Figure 1 (main): Total featurization time vs N reactions at batch_size=50.
Figure SI:       Speedup vs batch size (supplementary).

Reads:
  results/raw/featurization_total.csv
  results/raw/featurization_timing.csv
Writes:
  results/figures/fig1_featurization_speedup.{pdf,png}
  results/figures/figSI_featurization_speedup_vs_batchsize.{pdf,png}
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update(plt.rcParamsDefault)

INPUT_TOTAL = "results/raw/featurization_total.csv"
INPUT_PERRXN = "results/raw/featurization_timing.csv"
OUTPUT_DIR = "results/figures"

COLORS = {"python": "#d62728", "cuik": "#1f77b4"}
LABELS = {"python": "Python CGR", "cuik": "cuik-reactmaker (C++)"}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Main figure: total time vs N ──────────────────────────────────────────
    df_total = pd.read_csv(INPUT_TOTAL)
    agg_total = (
        df_total.groupby(["path", "n_reactions"])["total_time_s"]
        .agg(["median", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    for path in ["python", "cuik"]:
        sub = agg_total[agg_total["path"] == path].sort_values("n_reactions")
        ax.errorbar(
            sub["n_reactions"], sub["median"],
            yerr=sub["std"],
            marker="o", linewidth=2, capsize=3,
            color=COLORS[path], label=LABELS[path],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of reactions")
    ax.set_ylabel("Total featurization time (s)")
    ax.legend()
    ax.grid(False)

    # Annotate speedup: 80% from left, 10% from bottom (axes fraction coords)
    ax.text(
        0.95, 0.10, "8.4× speedup",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=13, fontweight="bold", color="dimgray",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
    )

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig1_featurization_speedup.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()

    # ── SI figure: speedup vs batch size ──────────────────────────────────────
    df_perrxn = pd.read_csv(INPUT_PERRXN)
    df_perrxn = df_perrxn[df_perrxn["batch_size"] >= 8]
    agg_bs = (
        df_perrxn.groupby(["path", "batch_size"])["time_per_rxn_us"]
        .median()
        .reset_index()
    )
    py_bs = agg_bs[agg_bs["path"] == "python"].set_index("batch_size")["time_per_rxn_us"]
    cu_bs = agg_bs[agg_bs["path"] == "cuik"].set_index("batch_size")["time_per_rxn_us"]
    common_bs = sorted(set(py_bs.index) & set(cu_bs.index))
    speedups = [py_bs[bs] / cu_bs[bs] for bs in common_bs]

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(common_bs, speedups, marker="o", linewidth=2, color="#2ca02c")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Speedup by cuik-reactmaker")
    ax2.grid(False)
    ax2.xaxis.set_minor_locator(ticker.NullLocator())

    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "figSI_featurization_speedup_vs_batchsize.pdf")
    plt.savefig(out2, bbox_inches="tight")
    plt.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close()


if __name__ == "__main__":
    main()
