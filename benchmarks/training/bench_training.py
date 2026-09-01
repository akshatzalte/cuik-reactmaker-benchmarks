#!/usr/bin/env python
"""
Experiment 2: End-to-end training timing — Python CGR vs C++ CGR.

Runs `chemprop train` via subprocess for each (dataset_size, path, seed) combination
and records wall-clock time per epoch. Both paths use the same conda env and chemprop
branch; only --use-cuikmolmaker-featurization differs.

Usage:
    conda activate chemprop_bench_v031
    python benchmarks/training/bench_training.py \
        --data-dir data/ \
        --output results/raw/training_timing.csv \
        --epochs 5 \
        --batch-size 50 \
        --seeds 0 1 2

Results CSV columns:
    path, n_reactions, batch_size, seed, total_time_s, time_per_epoch_s
"""

import argparse
import csv
import os
import subprocess
import sys
import time

DATASET_SIZES = [1, 5, 10, 50, 100, 300]   # in thousands — matches rgd1_{N}k.csv filenames

CHEMPROP_TRAIN_ARGS = [
    "--reaction-columns", "smiles",
    "--target-columns", "ea",
    "--keep-h",
]


def run_chemprop_train(data_path, output_dir, batch_size, epochs, seed, use_cuik):
    chemprop_bin = os.path.join(os.path.dirname(sys.executable), "chemprop")
    cmd = [
        chemprop_bin, "train",
        "--data-path", data_path,
        "--output-dir", output_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--data-seed", str(seed),
        "--pytorch-seed", str(seed),
    ] + CHEMPROP_TRAIN_ARGS

    if use_cuik:
        cmd.append("--use-cuikmolmaker-featurization")
    else:
        cmd.append("--no-cache")  # disable graph precomputation for fair on-the-fly comparison

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
    parser.add_argument("--data-dir", default="data/", help="Directory with rgd1_{N}k.csv files")
    parser.add_argument("--output", default="results/raw/training_timing.csv")
    parser.add_argument("--model-dir", default="results/models/training",
                        help="Root directory for saved model checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--sizes", nargs="+", type=int, default=DATASET_SIZES,
                        help="Dataset sizes in thousands (e.g. 1 5 10 50 100)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Load existing rows so partial reruns append rather than overwrite
    rows = []
    if os.path.exists(args.output):
        import csv as _csv
        with open(args.output, newline="") as f:
            rows = list(_csv.DictReader(f))
    paths = [("baseline", False), ("cuik", True)]

    total_runs = len(args.sizes) * len(paths) * len(args.seeds)
    run_idx = 0

    for n_k in args.sizes:
        data_path = os.path.join(args.data_dir, f"rgd1_{n_k}k.csv")
        if not os.path.exists(data_path):
            print(f"Skipping n={n_k}k: {data_path} not found")
            continue

        n_reactions = n_k * 1000

        for path_name, use_cuik in paths:
            for seed in args.seeds:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] n={n_k}k | path={path_name} | seed={seed} ...",
                      flush=True)

                output_dir = os.path.join(args.model_dir, f"{n_k}k_{path_name}_seed{seed}")
                os.makedirs(output_dir, exist_ok=True)

                elapsed = run_chemprop_train(
                    data_path=data_path,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    seed=seed,
                    use_cuik=use_cuik,
                )

                if elapsed is None:
                    print(f"    FAILED — skipping")
                    continue

                time_per_epoch = elapsed / args.epochs
                print(f"    total={elapsed:.1f}s | per epoch={time_per_epoch:.1f}s")

                rows.append({
                    "path": path_name,
                    "n_reactions": n_reactions,
                    "batch_size": args.batch_size,
                    "seed": seed,
                    "epochs": args.epochs,
                    "total_time_s": round(elapsed, 3),
                    "time_per_epoch_s": round(time_per_epoch, 3),
                })

                # Write incrementally so partial results survive interruption
                with open(args.output, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    print(f"\nDone. Results saved to {args.output}")
    print(f"Total runs completed: {len(rows)}/{total_runs}")


if __name__ == "__main__":
    main()
