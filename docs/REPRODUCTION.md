# Reproducing the noise-peak investigation

This document covers the follow-up work on one question raised about Figure 2A:

> How can performance peak at a noise level of 0.5? Shouldn't performance always
> decrease with increasing noise?

It maps each investigation step to the exact code and commands needed to
reproduce it. Everything here is GB1 + additive unless stated otherwise.

**Short answer to the question**, for context while reading: the peak is not a
real effect. Two things produce it, and they compound:

1. **A metric floor.** NDCG, AUROC, AP@R, per-variant NDCG and mutation NDCG all
   sit in a narrow band (~0.86-0.93 for NDCG) across the *entire* range of model
   quality — and a model trained on shuffled labels, with no learnable signal at
   all, already scores inside that same band. There is very little dynamic range
   for a real trend to show up in. The main driver is base rate: 43 of GB1's 56
   positions are annotated as important, so almost any ranking scores well.
2. **Seed variance.** With only 5 seeds per noise level, even *sign accuracy* —
   the one metric that clearly separates a trained model from a random one —
   bounces non-monotonically across noise levels (0.575, 0.540, 0.540, 0.737,
   0.547, 0.685, ...). That is sampling noise, not signal.

The underlying relationship *is* monotonic: mean test Spearman falls cleanly from
0.908 at noise=0.0 to 0.440 at noise=1.0. The attribution metrics just cannot
track it at this sample size.

---

## 0. Prerequisites

```bash
cd /path/to/IGLOO
source IGLOO_env/bin/activate          # or your own venv
module load python/3.12 cuda/12.2      # cluster only
```

The Slurm scripts default to this project's paths and allocation. Override both
without editing files:

```bash
export IGLOO_ROOT=/your/path/to/IGLOO      # honoured by every job script
sbatch --account=your-alloc jobs/...       # CLI beats the #SBATCH directive
```

Steps that train models need a GPU. Steps that only read already-computed
integrated-gradients CSVs are CPU-only, but slow (see the runtime note below).

**Runtime warning.** The plotting scripts re-read every per-cell IG CSV
(~81k rows each; 60 sweep cells + 5 baseline cells per interpolation path).
A single-path figure takes roughly 10 minutes; all three paths take 15-20.
Run them with `nohup`/`&` or under Slurm, not interactively.

---

## 1. Regenerate the underlying runs

Results are **not** tracked in git (see `.gitignore`) because they are large and
regenerable — the reduced-model sweep alone is 3.5 GB. Generate what you need:

| What | Command | Output |
|---|---|---|
| Full noise sweep, 12 levels x 5 seeds x 3 paths | `bash jobs/submit_gb1_decile_fill.sh` | `results/synthetic/GB1_additive/noise_*/seed_*/` |
| Shuffled-label random baseline, 5 seeds | `for s in 0 1 2 3 4; do SEED=$s sbatch jobs/run_random_baseline_cell.sh; done` | `.../random_baseline/seed_*/` |
| Large-Gaussian-noise cell (sigma=20) | `SEED=1 NOISE_STD=20.0 sbatch jobs/run_high_gaussian_noise_cell.sh` | `.../gaussian_noise_std_20.0/seed_1/` |
| Reduced-model sweep (60 cells + 5 baseline) | `bash jobs/submit_gb1_reduced_sweep.sh` | `results/synthetic_reduced_model/` |

Each cell takes roughly 11-13 minutes on one L40S. The 65-job reduced-model
sweep finished in about 25 minutes wall-clock with the scheduler running many
in parallel.

---

## 2. Attempt-by-attempt reproduction

### Attempt 1 — Reframe the x-axis off noise level

*Hypothesis: "noise level" is a confounded knob; plotting against the model's own
measured performance, with seed variability shown, should reveal a cleaner trend.*

```bash
python scripts/plot_simple_figures.py --mode interp-panels --no-legend
```

Produces one panel per interpolation path. **Result:** the reframing did not
dissolve the peak — per-noise-level NDCG means still bump around noise 0.3-0.5.
The x-axis was the wrong thing to blame.

### Attempt 2 — Add a random baseline, expand to more metrics

*Hypothesis: a shuffled-label negative control will show a visible gap between
"real model" and "chance" in both model performance and attribution quality.*

```bash
python scripts/plot_simple_figures.py --mode metric-grid
```

Writes one panel per metric plus a combined 2x2 grid. **Result:** this is where
the floor problem became visible. Best model vs. shuffled-label baseline:

| Metric | Best model (ρ=0.934) | Random baseline (5 seeds) | Separated? |
|---|---|---|---|
| NDCG | 0.900 | 0.891 ± 0.045 | no |
| AUROC | 0.508 | 0.472 ± 0.082 | no (both ≈ chance) |
| AP@R | 0.991 | 1.030 ± 0.075 | no (baseline *higher*) |
| Per-variant NDCG | 0.881 | 0.882 ± 0.023 | no |
| Mutation NDCG | 0.911 | 0.897 ± 0.004 | barely |
| Category separation | 2.355 | 1.524 ± 0.486 | partially |
| **Sign accuracy** | **0.703** | **0.363 ± 0.078** | **yes** |

### Attempt 3 — Cross-check the floor with a second randomization

*Hypothesis: if the floor is real, an independent way of destroying the signal
should land in the same place.*

The sweep's noise is additive Gaussian label noise (`inject_label_noise()` in
`scripts/run_integrated_gradients.py`). Its maximum swept value, sigma=1.0, only
drops Spearman to 0.44 — nowhere near chance. Pushing to sigma=20:

```bash
SEED=1 NOISE_STD=20.0 sbatch jobs/run_high_gaussian_noise_cell.sh
python scripts/plot_simple_figures.py --mode gaussian
```

**Result:** sigma=20 gives Spearman = -0.112 and NDCG = 0.869 — inside the
shuffled-label baseline's own NDCG range (0.840-0.933). Two unrelated ways of
destroying the signal reach the same floor, so the floor is a property of the
metric, not of the shuffling.

One caveat worth stating: `model_test_spearman` for the Gaussian cell is measured
against the *noised* labels, so at sigma=20 the evaluation target is itself mostly
noise. That makes it a less clean negative control than label shuffling, which
destroys the sequence-fitness pairing exactly by construction.

### Attempt 4 — Unpack IG and every metric's intermediate steps

Two artifacts:

- `colab_standalone/IGLOO_playground.ipynb` — a self-contained notebook (11
  sections) that walks embeddings → attention pooling → integrated gradients →
  all 7 metrics, and compares the best model against the shuffled-label model at
  every stage. See `colab_standalone/README.md`. Its vendored checkpoints and IG
  CSVs are gitignored, as is the vendored copy of `src/igloo` inside it (tracking
  that would guarantee drift from the canonical package). Rebuild the bundle with:

  ```bash
  bash scripts/build_colab_bundle.sh --zip    # needs the two runs from section 1
  ```

  then upload `colab_standalone.zip` to Colab.
- `scripts/toy_igloo_walkthrough.py` — a 5-residue toy protein running the whole
  pipeline with numbers small enough to verify by hand:

  ```bash
  python scripts/toy_igloo_walkthrough.py     # seconds, CPU only
  ```

  Ground truth (BLOSUM penalties, fitness, leave-one-out deltas) comes from the
  real `src/igloo/synthetic_data.py` functions. The embeddings and predictor are
  deliberate stand-ins — 2 dims instead of 960, a fixed attention vector, and a
  single nonlinear head — chosen so path integration measurably differs from
  gradient x input (-0.0326 vs -0.0476 at the mutated position) and so the
  completeness axiom can be checked by hand (gap ~1e-17).

### Attempt 5 — Shrink the model

*Hypothesis: if attribution quality is capacity-driven, a much smaller model
should change the picture.*

`ESMAttentionLinear` (`src/igloo/models.py`) keeps the same attention pooling but
replaces the 4-layer MLP with a single linear layer: **700,290 → 3,842 parameters**
(182x smaller). Select it with `model.model_head: "linear"`
(`configs/synthetic/GB1_additive_reduced_config.yaml`).

```bash
bash jobs/submit_gb1_reduced_sweep.sh          # 65 GPU jobs
python scripts/plot_simple_figures.py --mode metric-grid \
    --results-root results/synthetic_reduced_model --label "reduced model"
```

**Result:** test Spearman drops modestly (0.934 → 0.877), but most attribution
metrics hold steady or separate *slightly better* from baseline. Only sign
accuracy collapses (gap over baseline 0.34 → 0.09). So these metrics mostly do
not reward the extra 696k parameters either.

### Attempt 6 — Magnitude-based instead of rank-based metrics

*Hypothesis: rank metrics discard magnitude information; a regression-style
metric might discriminate better.*

Added to `src/igloo/ig_metrics.py`: `pearson_r`, `mae_z`, `r2_z` (global) and
`mutation_pearson_vals`, `mutation_mae_z`, `mutation_r2_z` (mutation-level).

```bash
python scripts/plot_simple_figures.py --mode regression
```

**Results, and one negative result worth knowing before reusing this code:**

- Global Pearson r: best 0.227 vs baseline 0.232 ± 0.126 — no separation at the
  single-cell level, but it tracks model performance across the sweep better than
  NDCG does (raw r=0.402 vs 0.354).
- Mutation-level Pearson r: best 0.193 vs baseline 0.076 ± 0.038 — real separation.
- **`r2_z` is mathematically redundant.** Once both vectors are z-scored, R²
  against the identity line reduces exactly to `2 * pearson_r - 1`. Verified
  numerically: r=0.2267 → R²=-0.5465 (computed -0.546509); baseline mean r=0.2324
  → -0.5352 (computed -0.535117). It carries no information beyond correlation.
  `mae_z` is not forced to be redundant but is empirically very close.

  The deeper reason: IG (embedding-gradient units) and the ground-truth weights
  (an arbitrary 0.05-1.5 scale) share no physical unit, so any comparison must
  first normalize them onto common footing — and that normalization is exactly
  what destroys the absolute-magnitude information a true magnitude metric would
  need. For this comparison, a genuinely magnitude-sensitive metric is not
  achievable; every version collapses back to correlation.

---

## 3. Code added or changed for this work

| File | Change |
|---|---|
| `src/igloo/models.py` | `ESMAttentionLinear` — attention pooling + single linear layer |
| `src/igloo/utils.py` | `create_model()` dispatches on `model_head` ("mlp" default / "linear") |
| `src/igloo/train_and_eval.py` | Plumb `model_head` through the model-config dicts and the checkpoint-compatibility check |
| `src/igloo/ig_metrics.py` | Pearson / MAE / R² companions to the existing rank metrics, global and mutation-level |
| `scripts/run_integrated_gradients.py` | `training.noise.shuffle_labels` — full label permutation for an exact chance-level control |
| `scripts/plot_ndcg_vs_model_performance.py` | Metric-vs-model-performance plots, all 7 metrics + regression family, random-baseline overlay, attenuation fit |
| `scripts/plot_simple_figures.py` | Slide-ready versions of the above |
| `scripts/toy_igloo_walkthrough.py` | 5-residue hand-checkable pipeline |
| `configs/synthetic/GB1_additive_reduced_config.yaml` | Reduced-model config |
| `jobs/*.sh` | Slurm drivers for every sweep above |
| `colab/`, `colab_standalone/` | Interactive playground notebooks |
| `scripts/build_colab_bundle.sh` | Reassembles the self-contained Colab bundle + zip |

---

## 4. What is still open

The one experiment not yet run: **more seeds per noise level**. Everything above
uses 5, which is too few for even sign accuracy to produce a monotonic curve.
Re-running the sweep at 20-30 seeds per level, and plotting sign accuracy
specifically, would settle whether the apparent peak disappears entirely once
sampling noise is averaged out — or whether something structural remains.

```bash
# sketch: widen the seed loop in jobs/submit_gb1_decile_fill.sh
SEEDS=(0 1 2 ... 24)
```
