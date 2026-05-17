#!/usr/bin/env python
"""
Figure 1: Featurization benchmark — two panels.
  Left:  Total featurization time (s) vs N reactions at batch_size=50
  Right: Speedup (Python / C++) vs batch size (8–1024)

Reads:
  results/raw/featurization_total.csv   (from --mode total)
  results/raw/featurization_timing.csv  (from --mode per-rxn)
Writes:
  results/figures/fig1_featurization_speedup.{pdf,png}
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update(plt.rcParamsDefault)

INPUT_TOTAL = "results/raw/featurization_total.csv"
INPUT_PERRXN = "results/raw/featurization_timing.csv"
OUTPUT_DIR = "results/figures"

COLORS = {"python": "#d62728", "cuik": "#1f77b4"}
LABELS = {"python": "Python CGR", "cuik": "C++ CGR (cuik-molmaker)"}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Left panel data: total time vs N ──────────────────────────────────────
    df_total = pd.read_csv(INPUT_TOTAL)
    agg_total = (
        df_total.groupby(["path", "n_reactions"])["total_time_s"]
        .agg(["median", "std"])
        .reset_index()
    )
    n_vals = sorted(agg_total["n_reactions"].unique())

    # ── Right panel data: speedup vs batch size (8–1024 only) ─────────────────
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

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: total time vs N
    ax = axes[0]
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
    ax.set_title("Total featurization time vs. dataset size\n(batch size = 50)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(n_vals)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
    )

    # Right: speedup vs batch size
    ax = axes[1]
    ax.plot(common_bs, speedups, marker="o", linewidth=2, color="#2ca02c")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup (Python / C++)")
    ax.set_title("Featurization speedup vs. batch size")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(common_bs)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))
    ax.tick_params(axis="x", rotation=45)

    # Annotate peak speedup
    peak_idx = int(np.argmax(speedups))
    ax.annotate(
        f"{speedups[peak_idx]:.1f}×",
        xy=(common_bs[peak_idx], speedups[peak_idx]),
        xytext=(0, 10), textcoords="offset points",
        ha="center", fontsize=10, color="#2ca02c",
    )

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig1_featurization_speedup.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
