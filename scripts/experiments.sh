#!/usr/bin/env bash
# Run all cuik-reactmaker benchmarks.
# Usage: bash experiments.sh [GPU_ID]   (default GPU_ID=1)
#
# Runs:
#   Step 1 — data subsets (one-time, skipped if already exist)
#   Step 2 — featurization timing (CPU, ~10 min)
#   Step 3 — featurization total time vs N (CPU, ~5 min)
#   Step 4 — training benchmark (GPU, ~hours)
#   Step 5 — inference benchmark (GPU, ~30 min)
#   Step 6 — figures + tables
#
# All scripts must be run from the repo root with chemprop_cuik_rxn env active:
#   conda activate chemprop_cuik_rxn
#   cd ~/chemprop && git checkout cuik_reactmaker
#   cd ~/projects/cuik-reactmaker-benchmarks
#   bash experiments.sh

set -e

GPU=${1:-1}
DATA=/home/akshatz/bond_order_free/barriers_rgd1/dataset/rgd1_data.csv

echo "========================================"
echo " cuik-reactmaker benchmark suite"
echo " GPU: $GPU | Data: $DATA"
echo "========================================"

# --- Step 1: data subsets ---
if [ ! -f data/rgd1_100k.csv ]; then
    echo ""
    echo "[Step 1] Creating dataset subsets..."
    python scripts/prepare_subsets.py --source "$DATA" --outdir data/
else
    echo "[Step 1] Data subsets already exist — skipping."
fi

# --- Step 2: featurization per-reaction timing ---
echo ""
echo "[Step 2] Featurization timing (per-reaction vs batch size)..."
python benchmarks/featurization/bench_featurization.py \
    --mode per-rxn \
    --data-path "$DATA" \
    --batch-sizes 8 16 32 64 128 256 512 1024 \
    --n-warmup 5 --n-trials 50 \
    --output results/raw/featurization_timing.csv

# --- Step 3: featurization total time vs N ---
echo ""
echo "[Step 3] Featurization total time vs dataset size..."
python benchmarks/featurization/bench_featurization.py \
    --mode total \
    --data-path "$DATA" \
    --n-reactions 1000 5000 10000 50000 100000 \
    --batch-size 50 \
    --n-warmup 2 --n-trials 5 \
    --output results/raw/featurization_total.csv

# --- Step 4: training benchmark ---
echo ""
echo "[Step 4] Training benchmark (GPU $GPU)..."
CUDA_VISIBLE_DEVICES=$GPU python benchmarks/training/bench_training.py \
    --data-dir data/ \
    --output results/raw/training_timing.csv \
    --epochs 5 \
    --batch-size 50 \
    --seeds 0 1 2

# --- Step 5: inference benchmark ---
# Use the baseline 100k checkpoint from training so inference is timed with
# a realistically trained model (weights don't affect timing, but avoids
# a separate training step here).
INFERENCE_MODEL=$(find results/models/training/100k_baseline_seed0 -name "*.pt" 2>/dev/null | head -1)
echo ""
echo "[Step 5] Inference benchmark (GPU $GPU)..."
CUDA_VISIBLE_DEVICES=$GPU python benchmarks/inference/bench_inference.py \
    --data-dir data/ \
    --model-path "$INFERENCE_MODEL" \
    --output results/raw/inference_timing.csv \
    --n-trials 3

# --- Step 6: figures and tables ---
echo ""
echo "[Step 6] Generating figures and tables..."
python analysis/plot_featurization.py
python analysis/plot_training.py
python analysis/plot_inference.py
python analysis/make_tables.py

echo ""
echo "========================================"
echo " All benchmarks complete."
echo " Results: results/raw/"
echo " Figures: results/figures/"
echo "========================================"
