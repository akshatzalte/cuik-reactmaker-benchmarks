#!/usr/bin/env python
"""
Combined headline figure: Featurization | Training | Inference side by side.
Intended for the GitHub README.

Reads:
  results/raw/featurization_total.csv
  results/raw/training_timing.csv
  results/raw/inference_timing.csv
Writes:
  results/figures/fig_combined.{pdf,png}
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update(plt.rcParamsDefault)

OUTPUT_DIR = "results/figures"

COLORS = {"python": "#d62728", "baseline": "#d62728", "cuik": "#1f77b4"}
LABELS = {"python": "Python CGR", "baseline": "Python CGR", "cuik": "cuik-reactmaker (C++)"}


def plot_panel(ax, agg, path_col, y_col, ylabel, title, speedup_text):
    for path in [k for k in ["python", "baseline"] if k in agg[path_col].values] + ["cuik"]:
        sub = agg[agg[path_col] == path].sort_values("n_reactions")
        ax.errorbar(
            sub["n_reactions"], sub["median"],
            yerr=sub["std"],
            marker="o", linewidth=2, capsize=3,
            color=COLORS[path], label=LABELS[path],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of reactions")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(False)

    ax.text(
        0.95, 0.10, speedup_text,
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=12, fontweight="bold", color="dimgray",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load and aggregate each dataset ──────────────────────────────────────
    df_feat = pd.read_csv("results/raw/featurization_total.csv")
    agg_feat = (
        df_feat.groupby(["path", "n_reactions"])["total_time_s"]
        .agg(["median", "std"]).reset_index()
    )

    df_train = pd.read_csv("results/raw/training_timing.csv")
    agg_train = (
        df_train.groupby(["path", "n_reactions"])["time_per_epoch_s"]
        .agg(["median", "std"]).reset_index()
    )

    df_infer = pd.read_csv("results/raw/inference_timing.csv")
    agg_infer = (
        df_infer.groupby(["path", "n_reactions"])["total_time_s"]
        .agg(["median", "std"]).reset_index()
    )

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    plot_panel(
        axes[0], agg_feat, "path", "total_time_s",
        ylabel="Total featurization time (s)",
        title="Featurization",
        speedup_text="8.4× speedup",
    )
    plot_panel(
        axes[1], agg_train, "path", "time_per_epoch_s",
        ylabel="Training time per epoch (s)",
        title="Training",
        speedup_text="3.2× speedup",
    )
    plot_panel(
        axes[2], agg_infer, "path", "total_time_s",
        ylabel="Total inference time (s)",
        title="Inference",
        speedup_text="4.7× speedup",
    )

    # Shared legend centered below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=11)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig_combined.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
