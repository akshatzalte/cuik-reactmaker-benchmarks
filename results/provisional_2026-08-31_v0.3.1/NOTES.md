# Provisional benchmark run — 2026-08-31

**Status: VALIDATION ONLY. Do not cite these numbers.** The machine was not idle
(other GPU jobs were running), so wall-clock timings are indicative, not final.
A clean rerun on an idle machine is planned for the week of 2026-09-07.

## What this run validated

The published-package stack works end-to-end after both merges:

- cuik-molmaker `feat/cgr-reaction-featurization` merged upstream (PR #4) and
  released as **v0.3.0 / v0.3.1**.
- Chemprop CGR C++ path merged upstream (PR #1365, 2026-08-03), shipped in
  **chemprop 2.3.1**. No branch switching is needed any more — the Python and
  C++ paths differ only by `--use-cuikmolmaker-featurization`.

Environment `chemprop_bench_v031` (chemprop's recommended Option 1 install):

```bash
conda create -n chemprop_bench_v031 -y python=3.11
conda activate chemprop_bench_v031
pip install "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.0 driver on this host
pip install "chemprop==2.3.1"    # pulls cuik-molmaker-pin 2026.3.5 -> rdkit 2026.03.5, cuik_molmaker 0.3.1
```

`cuik-molmaker-pin` locks rdkit to the matching release (2026.03.5); accepted.
Do NOT mix conda-forge rdkit with the pip pin — the ABI does not match.

## Correctness checks (these ARE conclusive)

- cuik-molmaker CGR goldens: **24/24 pass** (`pytest tests/python/test_reaction_features.py`)
- chemprop cuik featurizer tests: **96/96 pass**
  (`tests/unit/featurizers/test_cuikmolmaker{,_reaction}.py`)
- Both training paths produce identical `test/mse = 440.0153503417969` on
  `data/rgd1_1k.csv` (3 epochs, seed 0).

## Provisional timings

Featurization (RGD1, V2 / REAC_DIFF), µs per reaction, median:

| batch | Python | C++ | speedup |
|-------|--------|-----|---------|
| 1     | 716.6  | 97.8| 7.3x    |
| 10    | 697.5  | 77.0| 9.1x    |
| 50    | 694.9  | 72.3| 9.6x    |
| 100   | 692.6  | 70.7| 9.8x    |
| 500   | 683.3  | 74.5| 9.2x    |
| 1000  | 687.9  | 80.7| 8.5x    |

End-to-end featurization of 100k reactions at batch 50: 67.6 s -> 7.2 s (9.4x).

Training (5 epochs, batch 50, 3 seeds), s/epoch, median — machine busy:

| N       | baseline | cuik  | speedup | (May run) |
|---------|----------|-------|---------|-----------|
| 1k      | 1.67     | 1.11  | 1.50x   | 1.48x     |
| 5k      | 4.86     | 1.92  | 2.53x   | 2.36x     |
| 10k     | 9.49     | 3.20  | 2.96x   | 2.59x     |
| 50k     | 39.93    | 11.81 | 3.38x   | 3.15x     |
| 100k    | 78.37    | 23.30 | 3.36x   | 3.20x     |

Inference: only a 1k / 1-trial smoke check was run (`inference_smoke_1k.csv`),
enough to confirm `chemprop predict` works on both paths under 2.3.1.

## To redo properly (idle machine)

```bash
conda activate chemprop_bench_v031
cd ~/projects/cuik-reactmaker-benchmarks
bash scripts/experiments.sh 0      # or run steps 1-4 individually per README
```

Confirm `nvidia-smi` shows no other compute processes and the CPU is idle first;
featurization timings are CPU-bound and are the most sensitive to contention.
