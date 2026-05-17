#!/usr/bin/env python
"""
Experiment 1: Pure featurization timing — Python CGR vs C++ CGR.

Measures wall-clock time to featurize a batch of reactions, sweeping batch size.
No model, no GPU, no training — isolates the featurization cost only.

Timing covers parsing + featurization for both paths (this matches the actual
training loop: Python path pre-parses molecules per epoch; C++ path parses +
featurizes together per batch).

Usage:
    conda activate chemprop_cuik_rxn
    python benchmarks/featurization/bench_featurization.py \
        --data-path data/rgd1_10k.csv \
        --batch-sizes 1 10 50 100 500 1000 \
        --n-warmup 5 --n-trials 20 \
        --output results/raw/featurization_timing.csv
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
# Setup helpers
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
        "atomic-number-common",
        "total-degree",
        "formal-charge",
        "chirality",
        "num-hydrogens",
        "hybridization-expanded",
    ])
    atom_float = cuik_molmaker.atom_float_feature_names_to_array(["aromatic", "mass"])
    bond_feats = cuik_molmaker.bond_feature_names_to_array([
        "is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"
    ])
    mode_int = cuik_molmaker.reaction_mode_names_to_array(["REAC_DIFF"])[0]
    return cuik_molmaker, atom_onehot, atom_float, bond_feats, mode_int


# ---------------------------------------------------------------------------
# Per-batch timing functions
# ---------------------------------------------------------------------------

def time_python_batch(featurizer, batch_smiles):
    """Parse + featurize one batch with the Python CGR path."""
    params = Chem.SmilesParserParams()
    params.removeHs = False  # keep_h=True

    t0 = time.perf_counter()
    for rxn_smi in batch_smiles:
        rct_smi, _, pdt_smi = rxn_smi.split(">")
        rct_mol = Chem.MolFromSmiles(rct_smi, params)
        pdt_mol = Chem.MolFromSmiles(pdt_smi, params)
        featurizer((rct_mol, pdt_mol))
    return time.perf_counter() - t0


def time_cuik_batch(cuik_molmaker, atom_onehot, atom_float, bond_feats, mode_int, batch_smiles):
    """Parse + featurize one batch with the C++ CGR path."""
    reac_smiles, prod_smiles = [], []
    for rxn_smi in batch_smiles:
        rct_smi, _, pdt_smi = rxn_smi.split(">")
        reac_smiles.append(rct_smi)
        prod_smiles.append(pdt_smi)

    t0 = time.perf_counter()
    cuik_molmaker.batch_reaction_featurizer(
        reac_smiles, prod_smiles,
        atom_onehot, atom_float, bond_feats,
        True, False, False, mode_int,
    )
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(smiles_pool, batch_size, n_warmup, n_trials, time_fn):
    """Returns array of per-reaction times (seconds) over n_trials timed batches."""
    # Cycle through the pool to get enough batches
    batches = [
        smiles_pool[i:i + batch_size]
        for i in range(0, len(smiles_pool) - batch_size + 1, batch_size)
    ]
    # Ensure we have at least n_warmup + n_trials batches
    if len(batches) < n_warmup + n_trials:
        batches = list(itertools.islice(itertools.cycle(batches), n_warmup + n_trials))
    else:
        batches = batches[:n_warmup + n_trials]

    for batch in batches[:n_warmup]:
        time_fn(batch)

    per_reaction_times = []
    for batch in batches[n_warmup:n_warmup + n_trials]:
        t = time_fn(batch)
        per_reaction_times.append(t / len(batch))

    return np.array(per_reaction_times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="CSV with 'smiles' column (R>>P)")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 10, 50, 100, 500, 1000])
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--output", default="results/raw/featurization_timing.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading {args.data_path} ...")
    smiles_pool = pd.read_csv(args.data_path)["smiles"].tolist()
    print(f"  Pool size: {len(smiles_pool):,} reactions")

    print("Setting up featurizers ...")
    py_featurizer = setup_python_featurizer()
    cuik_mol, atom_onehot, atom_float, bond_feats, mode_int = setup_cuik_featurizer()

    rows = []
    for batch_size in args.batch_sizes:
        if batch_size > len(smiles_pool):
            print(f"  Skipping batch_size={batch_size}: larger than pool")
            continue

        print(f"  batch_size={batch_size} ...", flush=True)

        # Python path
        py_times = run_benchmark(
            smiles_pool, batch_size, args.n_warmup, args.n_trials,
            lambda b: time_python_batch(py_featurizer, b),
        )
        # C++ path
        cuik_times = run_benchmark(
            smiles_pool, batch_size, args.n_warmup, args.n_trials,
            lambda b: time_cuik_batch(cuik_mol, atom_onehot, atom_float, bond_feats, mode_int, b),
        )

        for trial_idx, (py_t, cuik_t) in enumerate(zip(py_times, cuik_times)):
            rows.append({
                "batch_size": batch_size,
                "path": "python",
                "trial": trial_idx,
                "time_per_rxn_us": py_t * 1e6,
            })
            rows.append({
                "batch_size": batch_size,
                "path": "cuik",
                "trial": trial_idx,
                "time_per_rxn_us": cuik_t * 1e6,
            })

        py_median = np.median(py_times) * 1e6
        cuik_median = np.median(cuik_times) * 1e6
        speedup = py_median / cuik_median
        print(f"    Python: {py_median:.1f} µs/rxn | C++: {cuik_median:.1f} µs/rxn | Speedup: {speedup:.1f}x")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_size", "path", "trial", "time_per_rxn_us"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
