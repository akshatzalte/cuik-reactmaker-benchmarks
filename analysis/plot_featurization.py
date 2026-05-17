#!/usr/bin/env python
"""
Figure 1: Featurization speedup vs batch size.

Reads: results/raw/featurization_timing.csv
Writes: results/figures/fig1_featurization_speedup.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

INPUT = "results/raw/featurization_timing.csv"
OUTPUT_DIR = "results/figures"

COLORS = {"python": "#d62728", "cuik": "#1f77b4"}
LABELS = {"python": "Python CGR", "cuik": "C++ CGR (cuik-molmaker)"}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT)

    # Aggregate: median and std per (path, batch_size)
    agg = (
        df.groupby(["path", "batch_size"])["time_per_rxn_us"]
        .agg(["median", "std"])
        .reset_index()
    )

    batch_sizes = sorted(agg["batch_size"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left panel: absolute timing ---
    ax = axes[0]
    for path in ["python", "cuik"]:
        sub = agg[agg["path"] == path].sort_values("batch_size")
        ax.errorbar(
            sub["batch_size"], sub["median"],
            yerr=sub["std"],
            marker="o", linewidth=2, capsize=3,
            color=COLORS[path], label=LABELS[path],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size", fontsize=12)
    ax.set_ylabel("Time per reaction (µs)", fontsize=12)
    ax.set_title("Featurization time per reaction", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(batch_sizes)

    # --- Right panel: speedup ---
    ax = axes[1]
    py_agg = agg[agg["path"] == "python"].set_index("batch_size")["median"]
    cuik_agg = agg[agg["path"] == "cuik"].set_index("batch_size")["median"]
    common = sorted(set(py_agg.index) & set(cuik_agg.index))

    speedups = [py_agg[bs] / cuik_agg[bs] for bs in common]
    ax.plot(common, speedups, marker="o", linewidth=2, color="#2ca02c")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Batch size", fontsize=12)
    ax.set_ylabel("Speedup (Python / C++)", fontsize=12)
    ax.set_title("Featurization speedup", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(common)

    # Annotate peak speedup
    peak_idx = int(np.argmax(speedups))
    ax.annotate(
        f"{speedups[peak_idx]:.1f}×",
        xy=(common[peak_idx], speedups[peak_idx]),
        xytext=(common[peak_idx], speedups[peak_idx] * 1.1),
        ha="center", fontsize=10, color="#2ca02c",
    )

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig1_featurization_speedup.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
