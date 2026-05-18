#!/usr/bin/env bash
set -e
cd /home/akshatz/projects/cuik-reactmaker-benchmarks

GPU=${1:-1}

echo "[$(date)] Step 4: Training benchmark (GPU $GPU)..."
CUDA_VISIBLE_DEVICES=$GPU python benchmarks/training/bench_training.py \
    --data-dir data/ \
    --output results/raw/training_timing.csv \
    --epochs 5 \
    --batch-size 50 \
    --seeds 0 1 2

echo "[$(date)] Step 5: Inference benchmark (GPU $GPU)..."
INFERENCE_MODEL=$(find results/models/training/100k_baseline_seed0 -name "*.pt" 2>/dev/null | head -1)
if [ -z "$INFERENCE_MODEL" ]; then
    echo "ERROR: No model checkpoint found in results/models/training/100k_baseline_seed0"
    exit 1
fi
echo "Using model: $INFERENCE_MODEL"
CUDA_VISIBLE_DEVICES=$GPU python benchmarks/inference/bench_inference.py \
    --data-dir data/ \
    --model-path "$INFERENCE_MODEL" \
    --output results/raw/inference_timing.csv \
    --n-trials 3

echo "[$(date)] Step 6: Figures and tables..."
python analysis/plot_featurization.py
python analysis/plot_training.py
python analysis/plot_inference.py
python analysis/make_tables.py

echo "[$(date)] Done."
