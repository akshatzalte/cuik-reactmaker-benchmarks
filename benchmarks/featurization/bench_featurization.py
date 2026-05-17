#!/usr/bin/env python
"""
Experiment 1: Pure featurization timing — Python CGR vs C++ CGR.

Two sub-experiments:
  --mode per-rxn   : per-reaction time vs batch size (fixed pool, vary batch_size)
  --mode total     : total featurization time vs dataset size (fixed batch_size=50, vary N)

Usage:
    conda activate chemprop_cuik_rxn
    cd ~/projects/cuik-reactmaker-benchmarks

    # Per-reaction time vs batch size
    python benchmarks/featurization/bench_featurization.py \
        --mode per-rxn \
        --data-path /home/akshatz/bond_order_free/barriers_rgd1/dataset/rgd1_data.csv \
        --batch-sizes 8 16 32 64 128 256 512 1024 \
        --n-warmup 10 --n-trials 50 \
        --output results/raw/featurization_per_rxn.csv

    # Total featurization time vs N
    python benchmarks/featurization/bench_featurization.py \
        --mode total \
        --data-path /home/akshatz/bond_order_free/barriers_rgd1/dataset/rgd1_data.csv \
        --n-reactions 1000 5000 10000 50000 100000 \
        --batch-size 50 \
        --n-warmup 2 --n-trials 5 \
        --output results/raw/featurization_total.csv
"""

import argparse
import csv
import itertools
import os
import time

import numpy as np
import pandas as pd
from rdkit import Chem


# ---------------------------------------------------------------------------
# Featurizer setup
# ---------------------------------------------------------------------------

def setup_python_featurizer():
    from chemprop.featurizers.molgraph.reaction import (
        CondensedGraphOfReactionFeaturizer,
        RxnMode,
    )
    return CondensedGraphOfReactionFeaturizer(mode_=RxnMode.REAC_DIFF)


def setup_cuik_featurizer():
    import cuik_molmaker
    atom_onehot = cuik_molmaker.atom_onehot_feature_names_to_array([
        "atomic-number-common", "total-degree", "formal-charge",
        "chirality", "num-hydrogens", "hybridization-expanded",
    ])
    atom_float = cuik_molmaker.atom_float_feature_names_to_array(["aromatic", "mass"])
    bond_feats = cuik_molmaker.bond_feature_names_to_array(
        ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"]
    )
    mode_int = cuik_molmaker.reaction_mode_names_to_array(["REAC_DIFF"])[0]
    return cuik_molmaker, atom_onehot, atom_float, bond_feats, mode_int


# ---------------------------------------------------------------------------
# Timing functions
# ---------------------------------------------------------------------------

def time_python_batch(featurizer, batch_smiles):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    t0 = time.perf_counter()
    for rxn_smi in batch_smiles:
        r, _, p = rxn_smi.split(">")
        featurizer((Chem.MolFromSmiles(r, params), Chem.MolFromSmiles(p, params)))
    return time.perf_counter() - t0


def time_cuik_batch(cm, atom_onehot, atom_float, bond_feats, mode_int, batch_smiles):
    reac, prod = [], []
    for rxn_smi in batch_smiles:
        r, _, p = rxn_smi.split(">")
        reac.append(r); prod.append(p)
    t0 = time.perf_counter()
    cm.batch_reaction_featurizer(reac, prod, atom_onehot, atom_float, bond_feats,
                                  True, False, False, mode_int)
    return time.perf_counter() - t0


def make_batches(smiles_pool, batch_size, n_needed):
    """Return exactly n_needed batches, cycling through the pool as needed."""
    batches = [
        smiles_pool[i:i + batch_size]
        for i in range(0, len(smiles_pool) - batch_size + 1, batch_size)
    ]
    if len(batches) < n_needed:
        batches = list(itertools.islice(itertools.cycle(batches), n_needed))
    return batches[:n_needed]


# ---------------------------------------------------------------------------
# Sub-experiment A: per-reaction time vs batch size
# ---------------------------------------------------------------------------

def run_per_rxn(args, smiles_pool, py_feat, cuik_args):
    rows = []
    for batch_size in args.batch_sizes:
        print(f"  batch_size={batch_size} ...", flush=True)
        batches = make_batches(smiles_pool, batch_size, args.n_warmup + args.n_trials)

        for b in batches[:args.n_warmup]:
            time_python_batch(py_feat, b)
        for b in batches[:args.n_warmup]:
            time_cuik_batch(*cuik_args, b)

        py_times, cuik_times = [], []
        for b in batches[args.n_warmup:]:
            py_times.append(time_python_batch(py_feat, b) / len(b))
            cuik_times.append(time_cuik_batch(*cuik_args, b) / len(b))

        for i, (py_t, cuik_t) in enumerate(zip(py_times, cuik_times)):
            rows.append({"batch_size": batch_size, "path": "python",
                         "trial": i, "time_per_rxn_us": py_t * 1e6})
            rows.append({"batch_size": batch_size, "path": "cuik",
                         "trial": i, "time_per_rxn_us": cuik_t * 1e6})

        py_med = np.median(py_times) * 1e6
        cuik_med = np.median(cuik_times) * 1e6
        print(f"    Python: {py_med:.1f} µs/rxn | C++: {cuik_med:.1f} µs/rxn "
              f"| Speedup: {py_med/cuik_med:.1f}x")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_size", "path", "trial", "time_per_rxn_us"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved {args.output}")


# ---------------------------------------------------------------------------
# Sub-experiment B: total featurization time vs N (fixed batch_size)
# ---------------------------------------------------------------------------

def run_total(args, smiles_pool, py_feat, cuik_args):
    rows = []
    for n in args.n_reactions:
        pool = smiles_pool[:n]   # fixed slice so same reactions across trials
        n_batches = n // args.batch_size
        if n_batches == 0:
            continue
        print(f"  N={n:,} ({n_batches} batches of {args.batch_size}) ...", flush=True)

        batches = make_batches(pool, args.batch_size, args.n_warmup + args.n_trials)

        # Warmup
        for b in batches[:args.n_warmup]:
            time_python_batch(py_feat, b)
        for b in batches[:args.n_warmup]:
            time_cuik_batch(*cuik_args, b)

        # Timed: measure total time for n_batches consecutive batches
        # (restart from the beginning of the pool each trial for consistency)
        py_totals, cuik_totals = [], []
        trial_batches = make_batches(pool, args.batch_size, n_batches)

        for _ in range(args.n_trials):
            t0 = time.perf_counter()
            for b in trial_batches:
                time_python_batch(py_feat, b)
            py_totals.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            for b in trial_batches:
                time_cuik_batch(*cuik_args, b)
            cuik_totals.append(time.perf_counter() - t0)

        for i, (py_t, cuik_t) in enumerate(zip(py_totals, cuik_totals)):
            rows.append({"n_reactions": n, "batch_size": args.batch_size,
                         "path": "python", "trial": i, "total_time_s": py_t})
            rows.append({"n_reactions": n, "batch_size": args.batch_size,
                         "path": "cuik",   "trial": i, "total_time_s": cuik_t})

        py_med = np.median(py_totals)
        cuik_med = np.median(cuik_totals)
        print(f"    Python: {py_med:.2f}s | C++: {cuik_med:.2f}s "
              f"| Speedup: {py_med/cuik_med:.1f}x")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["n_reactions", "batch_size", "path", "trial", "total_time_s"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved {args.output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["per-rxn", "total"], default="per-rxn")
    parser.add_argument("--data-path", required=True)
    # per-rxn args
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[8, 16, 32, 64, 128, 256, 512, 1024])
    # total args
    parser.add_argument("--n-reactions", nargs="+", type=int,
                        default=[1000, 5000, 10000, 50000, 100000])
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Fixed batch size for --mode total")
    # shared
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Loading {args.data_path} ...")
    smiles_pool = pd.read_csv(args.data_path)["smiles"].tolist()
    print(f"  Pool size: {len(smiles_pool):,} reactions")

    print("Setting up featurizers ...")
    py_feat = setup_python_featurizer()
    cuik_args = setup_cuik_featurizer()

    if args.mode == "per-rxn":
        run_per_rxn(args, smiles_pool, py_feat, cuik_args)
    else:
        run_total(args, smiles_pool, py_feat, cuik_args)


if __name__ == "__main__":
    main()
