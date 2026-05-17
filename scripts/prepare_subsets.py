#!/usr/bin/env python
"""
Create fixed-size RGD1 subsets for benchmarking.

Usage:
    conda activate chemprop_cuik_rxn
    python scripts/prepare_subsets.py \
        --source /home/akshatz/bond_order_free/barriers_rgd1/dataset/rgd1_data.csv \
        --outdir data/
"""
import argparse
import os
import pandas as pd

SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to rgd1_data.csv")
    parser.add_argument("--outdir", default="data/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.source} ...")
    df = pd.read_csv(args.source)
    print(f"  Total reactions: {len(df):,}")

    for n in SIZES:
        if n > len(df):
            print(f"  Skipping {n}: larger than dataset")
            continue
        subset = df.sample(n=n, random_state=SEED)
        name = f"rgd1_{n // 1000}k.csv"
        out = os.path.join(args.outdir, name)
        subset.to_csv(out, index=False)
        print(f"  Saved {n:,} reactions → {out}")

    print("Done.")


if __name__ == "__main__":
    main()
