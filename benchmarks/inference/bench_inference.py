#!/usr/bin/env python
"""
Experiment 3: Inference throughput — Python CGR vs C++ CGR.

Trains a small reference model (once per path) then times `chemprop predict`
on test sets of varying size. Inference is featurization-dominated (no backward
pass), so the speedup is the most dramatic of the three tiers.

Usage:
    conda activate chemprop_cuik_rxn
    python benchmarks/inference/bench_inference.py \
        --data-dir data/ \
        --model-train-size 10k \
        --output results/raw/inference_timing.csv \
        --n-trials 3

Results CSV columns:
    path, n_reactions, trial, total_time_s, rxns_per_sec
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time

PREDICT_SIZES = [1, 5, 10, 50, 100]   # in thousands

COMMON_ARGS = [
    "--reaction-columns", "smiles",
    "--keep-h",
]


def find_model_pt(model_dir):
    """Find the .pt model file inside a chemprop output directory."""
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".pt"):
                return os.path.join(root, f)
    return None


def train_reference_model(data_path, output_dir, use_cuik, seed=0):
    """Train a small reference model used only for inference timing."""
    chemprop_bin = os.path.join(os.path.dirname(sys.executable), "chemprop")
    cmd = [
        chemprop_bin, "train",
        "--data-path", data_path,
        "--output-dir", output_dir,
        "--epochs", "3",
        "--batch-size", "50",
        "--data-seed", str(seed),
        "--pytorch-seed", str(seed),
        "--target-columns", "ea",
    ] + COMMON_ARGS

    if use_cuik:
        cmd.append("--use-cuikmolmaker-featurization")

    print(f"    Training reference model ({'cuik' if use_cuik else 'baseline'}) ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("    ERROR training reference model:")
        print(result.stderr[-2000:])
        return None

    model_pt = find_model_pt(output_dir)
    if model_pt is None:
        print("    ERROR: no .pt model file found in", output_dir)
    return model_pt


def run_chemprop_predict(test_path, model_path, output_path, use_cuik):
    chemprop_bin = os.path.join(os.path.dirname(sys.executable), "chemprop")
    cmd = [
        chemprop_bin, "predict",
        "--test-path", test_path,
        "--model-path", model_path,
        "--output", output_path,
    ] + COMMON_ARGS

    if use_cuik:
        cmd.append("--use-cuikmolmaker-featurization")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"    ERROR (returncode={result.returncode}):")
        print(result.stderr[-2000:])
        return None

    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--model-train-size", default="10k",
                        help="Dataset size (e.g. '10k') used to train the reference model")
    parser.add_argument("--output", default="results/raw/inference_timing.csv")
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--sizes", nargs="+", type=int, default=PREDICT_SIZES,
                        help="Test set sizes in thousands")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    rows = []
    paths = [("baseline", False), ("cuik", True)]

    # Train a single reference model (baseline path) reused for both timing runs.
    # Model weights don't affect timing — only the featurization flag differs.
    train_data = os.path.join(args.data_dir, f"rgd1_{args.model_train_size}.csv")
    if not os.path.exists(train_data):
        print(f"ERROR: {train_data} not found. Run prepare_subsets.py first.")
        sys.exit(1)

    model_cache = "results/models/ref_model"
    model_pt = find_model_pt(model_cache)
    if model_pt is None:
        os.makedirs(model_cache, exist_ok=True)
        model_pt = train_reference_model(train_data, model_cache, use_cuik=False)
        if model_pt is None:
            print("ERROR: reference model training failed")
            sys.exit(1)
        print(f"  Reference model saved at {model_pt}")
    else:
        print(f"  Reusing reference model at {model_pt}")

    for path_name, use_cuik in paths:
        for n_k in args.sizes:
            test_path = os.path.join(args.data_dir, f"rgd1_{n_k}k.csv")
            if not os.path.exists(test_path):
                print(f"  Skipping n={n_k}k: {test_path} not found")
                continue

            n_reactions = n_k * 1000

            for trial in range(args.n_trials):
                print(f"  path={path_name} | n={n_k}k | trial={trial+1}/{args.n_trials} ...",
                      flush=True)

                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    out_path = tmp.name

                elapsed = run_chemprop_predict(
                    test_path=test_path,
                    model_path=model_pt,
                    output_path=out_path,
                    use_cuik=use_cuik,
                )

                try:
                    os.unlink(out_path)
                except OSError:
                    pass

                if elapsed is None:
                    print(f"    FAILED — skipping trial")
                    continue

                rxns_per_sec = n_reactions / elapsed
                print(f"    elapsed={elapsed:.2f}s | {rxns_per_sec:.0f} rxns/s")

                rows.append({
                    "path": path_name,
                    "n_reactions": n_reactions,
                    "trial": trial,
                    "total_time_s": round(elapsed, 3),
                    "rxns_per_sec": round(rxns_per_sec, 1),
                })

                # Write incrementally
                with open(args.output, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
