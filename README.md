# cuik-reactmaker-benchmarks

Timing benchmarks for **C++ vs. Python Condensed Graph of Reaction (CGR) featurization** in
[Chemprop](https://github.com/chemprop/chemprop), powered by
[cuik-molmaker](https://github.com/NVIDIA-Digital-Bio/cuik-molmaker).

Both featurization paths ship in the same released Chemprop (`>= 2.3.0`) and differ by a single
CLI flag, so every comparison here is one flag apart — same env, same data, same seeds:

| Path | Featurizer | How it is enabled |
|------|------------|-------------------|
| Baseline (Python) | `CondensedGraphOfReactionFeaturizer` — one reaction at a time | default |
| C++ (cuik-molmaker) | `batch_reaction_featurizer` — whole batch in one C++ call | `--use-cuikmolmaker-featurization` |

Three tiers are measured:

| Tier | Metric | Script |
|------|--------|--------|
| Featurization only | µs/reaction and total time; sweeps batch size and dataset size | `benchmarks/featurization/bench_featurization.py` |
| End-to-end training | s/epoch; sweeps dataset size | `benchmarks/training/bench_training.py` |
| Inference throughput | total seconds; sweeps dataset size | `benchmarks/inference/bench_inference.py` |

---

## Headline results

> RGD1 (353k reactions) · V2 featurizer · REAC_DIFF mode · `batch_size=50` · NVIDIA GeForce RTX 3090.

![Headline results](results/figures/fig_combined.png)

| Tier | Dataset | Baseline | C++ CGR | **Speedup** |
|------|---------|----------|---------|-------------|
| Featurization | 100k reactions | 71.4 s | 8.5 s | **8.4×** |
| Training (per epoch) | 100k reactions | 75.6 s | 23.6 s | **3.2×** |
| Inference | 100k reactions | 81.8 s | 17.5 s | **4.7×** |

<!-- FIGURE PLACEHOLDER: pipeline / overview schematic goes here once available.
     Add the image to results/figures/ (or docs/) and reference it as:
     ![Pipeline overview](results/figures/<filename>.png) -->

---

## Requirements

### Hardware

| Resource | Needed for | Notes |
|----------|-----------|-------|
| CPU | all tiers | Featurization timings are CPU-bound and the most sensitive to contention. |
| NVIDIA GPU | training, inference | ~8 GB is plenty at `batch_size=50`. Reference numbers used an RTX 3090. |
| Disk | dataset subsets + checkpoints | ~1 GB for `data/`, a few GB for `results/models/`. |
| RAM | 300k subset | ~8 GB. |

> **Run on an idle machine.** Wall-clock timings are meaningless under contention. Before starting,
> check `nvidia-smi` shows no other compute processes and the CPU is quiet (`uptime`).

### Software

| Package | Version used | Why pinned |
|---------|--------------|-----------|
| Python | 3.11 | Chemprop-supported (3.11 or 3.12). |
| `chemprop` | 2.3.1 | First release containing **both** CGR paths (PR [#1365](https://github.com/chemprop/chemprop/pull/1365), merged 2026-08-03). |
| `cuik-molmaker-pin` | 2026.3.5 | Mandatory Chemprop dependency; provides `cuik_molmaker` 0.3.1 (first release with `batch_reaction_featurizer`, cuik-molmaker PR [#4](https://github.com/NVIDIA-Digital-Bio/cuik-molmaker/pull/4)). |
| `rdkit` | 2026.03.5 | Locked by `cuik-molmaker-pin` — the pin exists to keep the RDKit ABI matched. |
| `torch` | 2.5.1+cu121 | Host-specific: this machine's driver is CUDA 12.0, and torch ≥ 2.12 requires > 12.0. On a newer driver, drop the pin and let pip resolve torch. |
| `pandas`, `matplotlib` | any recent | Analysis and plotting only. |

---

## Installation

This follows Chemprop's recommended install
([Option 1](https://chemprop.readthedocs.io/en/latest/installation.html)) — pip pulls
`cuik-molmaker-pin`, which brings a matched `cuik_molmaker` + `rdkit` pair:

```bash
conda create -n chemprop_bench_v031 -y python=3.11
conda activate chemprop_bench_v031

# Host-specific torch pin (CUDA 12.0 driver); skip on a newer driver.
pip install "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121

pip install "chemprop==2.3.1"
pip install pandas matplotlib pytest
```

Verify the stack before timing anything:

```bash
cd ~   # NOT a directory containing a `cuik_molmaker/` or `matplotlib/` folder — see pitfalls
python -c "
import chemprop, cuik_molmaker, rdkit, torch
print(chemprop.__version__, rdkit.__version__, torch.__version__, torch.cuda.is_available())
print('reaction API:', hasattr(cuik_molmaker, 'batch_reaction_featurizer'))
"
```

Expected: `2.3.1 2026.03.5 2.5.1+cu121 True` and `reaction API: True`.

### Pitfalls

<details>
<summary><b>Do not mix conda-forge RDKit with the pip pin</b></summary>

`cuik_molmaker` from conda-forge links against conda's RDKit shared libraries; the PyPI wheel
links against the hashed libraries inside the `rdkit` wheel. Installing one on top of the other
yields `ImportError: libRDKit*.so: cannot open shared object file` or an `undefined symbol`
error. Pick **one** channel for the whole `rdkit` + `cuik_molmaker` pair:

- all-pip: `pip install chemprop` (this README), or
- all-conda: `conda env create -f environment.yml` from the Chemprop repo, then `pip install --no-deps -e .`
</details>

<details>
<summary><b>Watch out for cwd shadowing</b></summary>

Python puts the working directory first on `sys.path`. Running from a directory that contains a
`cuik_molmaker/` folder (e.g. a local cuik-molmaker source checkout) or a `matplotlib/` folder
(e.g. `/tmp`, which often holds `MPLCONFIGDIR`) imports that folder instead of the installed
package. **Always run benchmarks from the repo root**, `~/projects/cuik-reactmaker-benchmarks`.
</details>

---

## Data

**RGD1** (Zhao et al.) — 353,984 atom-mapped reactions with activation energies in kcal/mol.

- Citation / download: <https://doi.org/10.5281/zenodo.10078142>
- Columns: `smiles` (`reactants>>products`, atom-mapped), target `ea`
- All reactions carry mapped explicit hydrogens, so `--keep-h` is required.
- Datasets are **never committed**. Local path on this machine:
  `/home/akshatz/bond_order_free/barriers_rgd1/dataset/rgd1_data.csv`

Create the fixed-size subsets (1k / 5k / 10k / 50k / 100k / 300k, `random_state=42`):

```bash
python scripts/prepare_subsets.py \
    --source /path/to/rgd1_data.csv \
    --outdir data/
```

---

## Reproducing the benchmarks

Always run from the repo root with the env active:

```bash
conda activate chemprop_bench_v031
cd ~/projects/cuik-reactmaker-benchmarks
```

### Step 0 — confirm correctness first

Timing a path that computes the wrong thing is worthless, so check equivalence before speed:

```bash
# Chemprop's own cuik featurizer tests (expect 96 passed).
# Needs a Chemprop source checkout; skip if you installed only the wheel.
pytest ~/chemprop/tests/unit/featurizers/test_cuikmolmaker.py \
       ~/chemprop/tests/unit/featurizers/test_cuikmolmaker_reaction.py -q --no-cov

# Same data, same seed, both paths -> identical test/mse
for FLAG in "--no-cache" "--use-cuikmolmaker-featurization"; do
  chemprop train --data-path data/rgd1_1k.csv --output-dir /tmp/parity_check \
      --epochs 3 --batch-size 50 --data-seed 0 --pytorch-seed 0 \
      --reaction-columns smiles --target-columns ea --keep-h $FLAG 2>&1 | grep "test/mse"
done
```

### Everything at once

```bash
bash scripts/experiments.sh 0     # argument = GPU id (default 1)
```

| Step | What | Cost |
|------|------|------|
| 1 | Dataset subsets (skipped if `data/rgd1_100k.csv` exists) | ~1 min, CPU |
| 2 | Featurization: µs/reaction vs. batch size | ~10 min, CPU |
| 3 | Featurization: total time vs. N | ~5 min, CPU |
| 4 | Training: s/epoch vs. N, 3 seeds | hours, GPU |
| 5 | Inference: total time vs. N, 3 trials | ~30 min, GPU |
| 6 | Figures + tables | seconds |

### Individual steps

```bash
# Featurization — per-reaction time vs batch size
python benchmarks/featurization/bench_featurization.py \
    --mode per-rxn \
    --data-path /path/to/rgd1_data.csv \
    --batch-sizes 8 16 32 64 128 256 512 1024 \
    --n-warmup 5 --n-trials 50 \
    --output results/raw/featurization_timing.csv

# Featurization — total time vs dataset size at fixed batch size
python benchmarks/featurization/bench_featurization.py \
    --mode total \
    --data-path /path/to/rgd1_data.csv \
    --n-reactions 1000 5000 10000 50000 100000 300000 \
    --batch-size 50 --n-warmup 2 --n-trials 5 \
    --output results/raw/featurization_total.csv

# Training
CUDA_VISIBLE_DEVICES=0 python benchmarks/training/bench_training.py \
    --data-dir data/ --output results/raw/training_timing.csv \
    --epochs 5 --batch-size 50 --seeds 0 1 2

# Inference (reuses a checkpoint from the training step; trains its own reference model if
# --model-path is omitted)
MODEL=$(find results/models/training/100k_baseline_seed0 -name "*.pt" | head -1)
CUDA_VISIBLE_DEVICES=0 python benchmarks/inference/bench_inference.py \
    --data-dir data/ --model-path "$MODEL" \
    --output results/raw/inference_timing.csv --n-trials 3

# Figures and tables
python analysis/plot_featurization.py
python analysis/plot_training.py
python analysis/plot_inference.py
python analysis/make_tables.py
```

The training and inference scripts write their CSV **incrementally**, so results collected before
an interruption survive. They do not skip already-measured configurations, though: rerunning the
same command appends a second set of rows. To redo a run cleanly, delete the output CSV first;
to extend one, pass a narrower `--sizes` / `--seeds` so only the missing configurations are run.

---

## Detailed results

### Featurization

![Featurization speedup](results/figures/fig1_featurization_speedup.png)

**8–8.4×** across all batch sizes (8–1024), so the gain comes from C++ computation rather than
from batching overhead alone:

| Batch size | Python CGR | C++ CGR | Speedup |
|---|---|---|---|
| 8 | ~707 µs/rxn | ~92 µs/rxn | 7.7× |
| 50 (default) | ~700 µs/rxn | ~84 µs/rxn | **8.3×** |
| 256 | ~701 µs/rxn | ~84 µs/rxn | 8.4× |
| 1024 | ~695 µs/rxn | ~92 µs/rxn | 7.5× |

### Training

![Training speedup](results/figures/fig2_training_speedup.png)

Speedup grows with dataset size and converges to **~3.2×** at 50k–100k reactions. Both paths use
on-the-fly featurization (`--no-cache` for the baseline) for a fair comparison:

| Dataset size | Baseline (s/epoch) | C++ CGR (s/epoch) | Speedup |
|---|---|---|---|
| 1k | 1.58 | 1.07 | 1.5× |
| 5k | 4.57 | 1.93 | 2.4× |
| 10k | 8.19 | 3.17 | 2.6× |
| 50k | 37.97 | 12.04 | 3.2× |
| 100k | 75.58 | 23.61 | **3.2×** |

### Inference

![Inference speedup](results/figures/fig3_inference_speedup.png)

Still growing at 100k (not yet converged), because fixed model-loading overhead amortizes as N
increases:

| Dataset size | Baseline (s) | C++ CGR (s) | Speedup |
|---|---|---|---|
| 1k | 4.46 | 3.78 | 1.2× |
| 5k | 7.58 | 4.30 | 1.8× |
| 10k | 11.80 | 5.06 | 2.3× |
| 50k | 43.57 | 10.52 | 4.1× |
| 100k | 81.84 | 17.45 | **4.7×** |

### Experimental design

- **Featurization**: batch sizes 8–1024; full RGD1 pool; 5 warmup + 50 timed trials; median µs/reaction. Also total time vs. N at fixed `batch_size=50`.
- **Training**: dataset sizes 1k–300k; `batch_size=50`; 5 epochs; 3 seeds; both paths featurize on the fly.
- **Inference**: held-out sets of 1k–100k; 3 trials; one shared reference model (100k, seed 0).
- All runs use the V2 featurizer and REAC_DIFF reaction mode (Chemprop defaults) with `--keep-h`.
- Both paths run in the same env; only `--use-cuikmolmaker-featurization` differs.

### Other runs

`results/provisional_2026-08-31_v0.3.1/` holds a validation pass on the released
chemprop 2.3.1 + cuik-molmaker 0.3.1 stack. It was collected on a **busy** machine and is kept
for reference only — see the `NOTES.md` in that directory. The numbers above remain the
reference results.

---

## Repository layout

```
cuik-reactmaker-benchmarks/
├── benchmarks/
│   ├── featurization/bench_featurization.py   # tier 1: featurization only
│   ├── training/bench_training.py             # tier 2: end-to-end training
│   └── inference/bench_inference.py           # tier 3: inference throughput
├── scripts/
│   ├── experiments.sh                         # full suite, steps 1-6
│   ├── prepare_subsets.py                     # rgd1_{N}k.csv subsets
│   └── test_fixtures/                         # reference-data generators, parity checks
├── analysis/
│   ├── plot_featurization.py
│   ├── plot_training.py
│   ├── plot_inference.py
│   └── make_tables.py
├── results/
│   ├── raw/                                   # committed — raw timing CSVs
│   │   ├── featurization_timing.csv           # per-reaction time vs batch size
│   │   ├── featurization_total.csv            # total time vs dataset size
│   │   ├── training_timing.csv                # s/epoch by dataset size and path
│   │   └── inference_timing.csv               # total inference time by dataset size
│   ├── figures/                               # committed — paper-ready PDF + PNG
│   ├── tables/                                # committed — summary CSVs
│   └── provisional_*/                         # non-reference validation runs
├── paper/                                     # JOSS manuscript sources
└── data/                                      # gitignored; created by prepare_subsets.py
```

## References

- cuik-molmaker: <https://github.com/NVIDIA-Digital-Bio/cuik-molmaker>
- Chemprop: <https://github.com/chemprop/chemprop>
- RGD1 dataset: <https://doi.org/10.5281/zenodo.10078142>
