#!/usr/bin/env python
"""
Generate summary tables for the JOSS paper.

Reads all three raw timing CSVs and produces:
  - results/tables/summary_table.csv   (main paper table)
  - results/tables/featurization_by_batch.csv
  - results/tables/training_by_size.csv
  - results/tables/inference_by_size.csv
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = "results/raw"
TABLE_DIR = "results/tables"


def fmt_speedup(val):
    return f"{val:.1f}×"


def fmt_time(val, unit="s"):
    if unit == "us":
        return f"{val:.1f} µs"
    return f"{val:.1f} s"


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Featurization table: µs/rxn at each batch size, with speedup
    # -----------------------------------------------------------------------
    feat = pd.read_csv(os.path.join(RAW_DIR, "featurization_timing.csv"))
    feat_agg = (
        feat.groupby(["path", "batch_size"])["time_per_rxn_us"]
        .median().reset_index()
        .pivot(index="batch_size", columns="path", values="time_per_rxn_us")
        .reset_index()
    )
    if "python" in feat_agg.columns and "cuik" in feat_agg.columns:
        feat_agg["speedup"] = feat_agg["python"] / feat_agg["cuik"]
        feat_agg.columns.name = None
        feat_agg = feat_agg.rename(columns={
            "batch_size": "Batch size",
            "python": "Python CGR (µs/rxn)",
            "cuik": "C++ CGR (µs/rxn)",
            "speedup": "Speedup",
        })
        feat_agg["Python CGR (µs/rxn)"] = feat_agg["Python CGR (µs/rxn)"].round(1)
        feat_agg["C++ CGR (µs/rxn)"] = feat_agg["C++ CGR (µs/rxn)"].round(1)
        feat_agg["Speedup"] = feat_agg["Speedup"].apply(fmt_speedup)
        out = os.path.join(TABLE_DIR, "featurization_by_batch.csv")
        feat_agg.to_csv(out, index=False)
        print(f"Saved {out}")
        print(feat_agg.to_string(index=False))
        print()

    # -----------------------------------------------------------------------
    # Training table: s/epoch at each dataset size, with speedup
    # -----------------------------------------------------------------------
    try:
        train = pd.read_csv(os.path.join(RAW_DIR, "training_timing.csv"))
        train_agg = (
            train.groupby(["path", "n_reactions"])["time_per_epoch_s"]
            .median().reset_index()
            .pivot(index="n_reactions", columns="path", values="time_per_epoch_s")
            .reset_index()
        )
        if "baseline" in train_agg.columns and "cuik" in train_agg.columns:
            train_agg["speedup"] = train_agg["baseline"] / train_agg["cuik"]
            train_agg["Dataset size"] = train_agg["n_reactions"].apply(
                lambda n: f"{n // 1000}k"
            )
            train_agg = train_agg[["Dataset size", "baseline", "cuik", "speedup"]]
            train_agg.columns = ["Dataset size", "Baseline (s/epoch)", "C++ CGR (s/epoch)", "Speedup"]
            train_agg["Baseline (s/epoch)"] = train_agg["Baseline (s/epoch)"].round(1)
            train_agg["C++ CGR (s/epoch)"] = train_agg["C++ CGR (s/epoch)"].round(1)
            train_agg["Speedup"] = train_agg["Speedup"].apply(fmt_speedup)
            out = os.path.join(TABLE_DIR, "training_by_size.csv")
            train_agg.to_csv(out, index=False)
            print(f"Saved {out}")
            print(train_agg.to_string(index=False))
            print()
    except FileNotFoundError:
        print("training_timing.csv not found — skipping training table")

    # -----------------------------------------------------------------------
    # Inference table: total s at each dataset size, with speedup
    # -----------------------------------------------------------------------
    try:
        infer = pd.read_csv(os.path.join(RAW_DIR, "inference_timing.csv"))
        infer_agg = (
            infer.groupby(["path", "n_reactions"])["total_time_s"]
            .median().reset_index()
            .pivot(index="n_reactions", columns="path", values="total_time_s")
            .reset_index()
        )
        if "baseline" in infer_agg.columns and "cuik" in infer_agg.columns:
            infer_agg["speedup"] = infer_agg["baseline"] / infer_agg["cuik"]
            infer_agg["Dataset size"] = infer_agg["n_reactions"].apply(
                lambda n: f"{n // 1000}k"
            )
            infer_agg = infer_agg[["Dataset size", "baseline", "cuik", "speedup"]]
            infer_agg.columns = ["Dataset size", "Baseline (s)", "C++ CGR (s)", "Speedup"]
            infer_agg["Baseline (s)"] = infer_agg["Baseline (s)"].round(2)
            infer_agg["C++ CGR (s)"] = infer_agg["C++ CGR (s)"].round(2)
            infer_agg["Speedup"] = infer_agg["Speedup"].apply(fmt_speedup)
            out = os.path.join(TABLE_DIR, "inference_by_size.csv")
            infer_agg.to_csv(out, index=False)
            print(f"Saved {out}")
            print(infer_agg.to_string(index=False))
            print()
    except FileNotFoundError:
        print("inference_timing.csv not found — skipping inference table")

    # -----------------------------------------------------------------------
    # Summary table (main paper table): one row per tier at representative config
    # -----------------------------------------------------------------------
    print("=== Summary table (representative config: batch_size=50, N=100k) ===")
    summary_rows = []

    # Featurization at batch_size=50
    if "python" in feat_agg.columns or "Python CGR (µs/rxn)" in feat_agg.columns:
        try:
            feat_raw = feat[feat["batch_size"] == 50].groupby("path")["time_per_rxn_us"].median()
            summary_rows.append({
                "Tier": "Featurization (batch=50)",
                "Baseline": f"{feat_raw.get('python', float('nan')):.1f} µs/rxn",
                "C++ CGR": f"{feat_raw.get('cuik', float('nan')):.1f} µs/rxn",
                "Speedup": fmt_speedup(feat_raw.get("python", 1) / feat_raw.get("cuik", 1)),
            })
        except Exception:
            pass

    # Training at N=100k
    try:
        train_100k = train[train["n_reactions"] == 100_000].groupby("path")["time_per_epoch_s"].median()
        summary_rows.append({
            "Tier": "Training (N=100k, epoch)",
            "Baseline": f"{train_100k.get('baseline', float('nan')):.1f} s",
            "C++ CGR": f"{train_100k.get('cuik', float('nan')):.1f} s",
            "Speedup": fmt_speedup(train_100k.get("baseline", 1) / train_100k.get("cuik", 1)),
        })
    except Exception:
        pass

    # Inference at N=100k
    try:
        infer_100k = infer[infer["n_reactions"] == 100_000].groupby("path")["total_time_s"].median()
        summary_rows.append({
            "Tier": "Inference (N=100k)",
            "Baseline": f"{infer_100k.get('baseline', float('nan')):.2f} s",
            "C++ CGR": f"{infer_100k.get('cuik', float('nan')):.2f} s",
            "Speedup": fmt_speedup(infer_100k.get("baseline", 1) / infer_100k.get("cuik", 1)),
        })
    except Exception:
        pass

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        out = os.path.join(TABLE_DIR, "summary_table.csv")
        summary.to_csv(out, index=False)
        print(summary.to_string(index=False))
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
