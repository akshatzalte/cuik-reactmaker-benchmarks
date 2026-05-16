#!/usr/bin/env bash
# Download the USPTO reaction dataset for benchmarking.
# Data is stored in data/ which is gitignored.
# Update this script with the actual dataset URL once confirmed.
set -euo pipefail

DATADIR="$(dirname "$0")/../data"
mkdir -p "$DATADIR"

echo "TODO: set dataset URL and format once dataset is confirmed."
echo "Data should be placed in: $DATADIR"
echo "Expected format: CSV with columns 'smiles' (atom-mapped reaction SMILES, e.g. 'R>>P') and target property column(s)."
