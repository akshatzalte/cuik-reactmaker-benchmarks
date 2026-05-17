# cuik-reactmaker-benchmarks

Rigorous timing benchmarks for C++ vs Python CGR reaction featurization in [Chemprop](https://github.com/chemprop/chemprop), powered by [cuik-molmaker](https://github.com/NVIDIA-Digital-Bio/cuik-molmaker).

Benchmarks the `--use-cuikmolmaker-featurization` flag (C++ `batch_reaction_featurizer`) against the default Python `CondensedGraphOfReactionFeaturizer` across three tiers:

| Tier | Metric | Script |
|------|--------|--------|
| Featurization only | µs/reaction, sweep batch size | `benchmarks/featurization/bench_featurization.py` |
| End-to-end training | s/epoch, sweep dataset size | `benchmarks/training/run_all_training.sh` |
| Inference throughput | total s, sweep dataset size | `benchmarks/inference/run_all_inference.sh` |

## Dataset

**RGD1** (Grambow et al.) — 353,984 atom-mapped reactions with activation energies.
- Citation: https://doi.org/10.5281/zenodo.10078142
- Local path (not committed): `/home/akshatz/bond_order_free/barriers_rgd1/dataset/rgd1_data.csv`

## Environment

All benchmarks run in the `chemprop_cuik_rxn` conda env with chemprop's `cuik_reactmaker` branch checked out. Both benchmark paths (baseline and cuik) use the **same env** — `--use-cuikmolmaker-featurization` is the only difference.

```bash
conda activate chemprop_cuik_rxn
cd ~/chemprop && git checkout cuik_reactmaker
```

## Quick start

```bash
# 1. Create data subsets (run once)
conda activate chemprop_cuik_rxn
python scripts/prepare_subsets.py \
    --source /home/akshatz/bond_order_free/barriers_rgd1/dataset/rgd1_data.csv \
    --outdir data/
# produces data/rgd1_1k.csv through data/rgd1_100k.csv

# 2. Featurization microbenchmark (~10 min, CPU only)
python benchmarks/featurization/bench_featurization.py \
    --data-path data/rgd1_10k.csv \
    --output results/raw/featurization_timing.csv

# 3. Training benchmarks (all sizes × both paths × 3 seeds)
bash benchmarks/training/run_all_training.sh

# 4. Inference benchmarks (requires trained models from step 3)
bash benchmarks/inference/run_all_inference.sh

# 5. Figures and tables
python analysis/plot_featurization.py
python analysis/plot_training.py
python analysis/plot_inference.py
python analysis/make_tables.py
```

## Results layout

```
results/
├── raw/                        # committed — raw timing CSVs
│   ├── featurization_timing.csv
│   ├── training_timing.csv
│   └── inference_timing.csv
├── figures/                    # committed — paper-ready plots
│   ├── fig1_featurization_speedup.pdf
│   ├── fig2_training_speedup.pdf
│   └── fig3_inference_speedup.pdf
├── tables/                     # committed — CSV/LaTeX summary tables
│   └── summary_table.csv
└── hardware.txt                # committed — hardware spec log
```

## Repo structure

```
cuik-reactmaker-benchmarks/
├── README.md
├── .gitignore
├── data/                              # gitignored; filled by prepare_subsets.py
├── scripts/
│   ├── prepare_subsets.py             # create rgd1_{N}.csv subsets
│   └── log_hardware.sh                # log GPU/CPU specs
├── benchmarks/
│   ├── featurization/
│   │   └── bench_featurization.py    # Exp 1: pure timing, sweeps batch size
│   ├── training/
│   │   ├── run_training.sh           # single training run (parametric)
│   │   └── run_all_training.sh       # sweeps all sizes × paths × seeds
│   └── inference/
│       ├── run_inference.sh          # single inference run (parametric)
│       └── run_all_inference.sh      # sweeps all sizes × paths
├── results/
└── analysis/
    ├── plot_featurization.py
    ├── plot_training.py
    ├── plot_inference.py
    └── make_tables.py
```

## Experimental design

- **Featurization**: batch sizes 1, 10, 50, 100, 500, 1000; pool of 10k reactions; 5 warmup + 20 timed trials; report median ± std µs/reaction
- **Training**: dataset sizes 1k, 5k, 10k, 50k, 100k; batch_size=50; 5 epochs; 3 seeds; metric = median s/epoch
- **Inference**: predict on test splits of size 1k, 5k, 10k, 50k, 100k using a model trained at 50k; 3 trials; metric = total wall-clock seconds
- All benchmarks use V2 featurizer mode, REAC_DIFF reaction mode (defaults)
- Both paths run in `chemprop_cuik_rxn` env; only `--use-cuikmolmaker-featurization` flag differs
