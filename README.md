# cuik-reactmaker-benchmarks

Rigorous timing benchmarks for C++ vs. Python reaction featurization in [Chemprop](https://github.com/chemprop/chemprop), powered by [cuik-molmaker](https://github.com/NVIDIA-Digital-Bio/cuik-molmaker).

Benchmarks the `--use-cuikmolmaker-featurization` flag (C++ `batch_reaction_featurizer`) against the default Python `CondensedGraphOfReactionFeaturizer` (CGR) path.

## What is benchmarked

| Tier | Metric | Method |
|------|--------|--------|
| Featurization only | Time to featurize N reactions | Direct Python vs. C++ call, sweep N and batch_size |
| End-to-end training | Wall-clock for 50-epoch training | `chemprop train` with/without `--use-cuikmolmaker-featurization` |
| Inference throughput | Predictions/second | `chemprop predict` with/without flag |

## Dataset

A single large public reaction dataset (~360k reactions) is used for all experiments.
Datasets are **not committed** — fetch them with:

```bash
bash scripts/download_data.sh
```

## Setup

```bash
conda env create -f environment.yml
conda activate cuik-benchmarks
```

## Running benchmarks

```bash
# 1. Download data
bash scripts/download_data.sh

# 2. Featurization microbenchmark (no training, no GPU needed)
python benchmarks/featurization/bench_reaction_featurizer.py

# 3. End-to-end training
bash benchmarks/training/run_python_cgr.sh
bash benchmarks/training/run_cuik_cgr.sh

# 4. Inference
bash benchmarks/inference/run_python_infer.sh
bash benchmarks/inference/run_cuik_infer.sh

# 5. Generate paper figures
python analysis/plot_results.py
```

## Results

Committed results (tables and figures) are in `results/`. Raw timing logs are gitignored.

## Reproducibility

- All experiments use a fixed random seed (`--seed 42`).
- Each configuration is run 5 times; we report median ± std.
- Batch sizes swept: 16, 32, 64, 128, 256.
- Dataset sizes swept: 1k, 10k, 50k, 360k reactions.
- Hardware details are logged alongside results.

## Repo structure

```
cuik-reactmaker-benchmarks/
├── environment.yml
├── .gitignore
├── data/                         # gitignored; filled by download_data.sh
├── scripts/
│   ├── download_data.sh
│   └── setup_env.sh
├── benchmarks/
│   ├── featurization/
│   │   └── bench_reaction_featurizer.py
│   ├── training/
│   │   ├── run_python_cgr.sh
│   │   └── run_cuik_cgr.sh
│   └── inference/
│       ├── run_python_infer.sh
│       └── run_cuik_infer.sh
├── results/                      # committed tables + plots
└── analysis/
    └── plot_results.py
```
