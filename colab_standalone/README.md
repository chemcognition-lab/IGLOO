# IGLOO Playground — Standalone / Colab Edition

A fully self-contained version of the IGLOO attribution playground. Unlike the in-repo
`IGLOO/colab/` version (which references the surrounding cluster repo's `src/`, `data/`,
and `results/`), everything this notebook needs is vendored inside this one folder —
zip it, upload it anywhere (including Google Colab), and run.

## What's in this folder

```
colab_standalone/
  IGLOO_playground.ipynb        <- the notebook, start here
  igloo_playground_utils.py     <- transparent wrapper functions (same as the in-repo
                                    version, except PROJECT_ROOT points at this folder)
  requirements.txt              <- packages beyond Colab's preinstalled set
  README.md                     <- this file
  src/igloo/                    <- the full IGLOO package, vendored verbatim
  scripts/
    plot_ndcg_vs_model_performance.py   <- recreates the metric-vs-model-performance plots
    run_integrated_gradients.py          <- the full training+IG pipeline, for generating
                                             additional (noise, seed) cells yourself
  data/synthetic/
    GB1_additive.csv                                  <- the synthetic dataset
    GB1_ground_truth_weights.json                      <- per-residue importance annotations
    GB1_additive_signal_families_vs_model_perf_raw_linear.csv   <- all 60 (noise,seed) x 7 metrics
    GB1_additive_random_baseline_linear.csv             <- the shuffled-label "chance" baseline (5 seeds)
  data/embeddings/               <- starts empty; fills up as you embed sequences (cache)
  results/synthetic/GB1_additive/
    noise_0.0/seed_1/                                   <- the "best model performance" run
      GB1_additive_training_model.pth
      linear/GB1_additive_integrated_gradients_results.csv  (81k rows)
      linear/GB1_additive_config_snapshot_*.yaml
    random_baseline/seed_0/                             <- the "complete random" negative control
      GB1_additive_training_model.pth                     (trained on fully shuffled labels)
      linear/GB1_additive_integrated_gradients_results.csv
      linear/GB1_additive_config_snapshot_*.yaml
```

**Why this specific structure matters:** `src/igloo/ig_metrics.py` computes its own
"project root" as *two directories above itself* and loads ground-truth annotations from
`<that root>/data/synthetic/`. `src/igloo/embedding.py`'s caching similarly walks up from
the *caller's* file location looking for a `data/` folder. Both resolve correctly as long
as this folder's layout stays intact — don't flatten or rename the `src/`, `data/`, or
`results/` subfolders relative to `igloo_playground_utils.py`.

## Running on Google Colab

1. Zip this folder and upload it to Colab (drag into the file browser, or upload to
   Google Drive and mount it).
2. Unzip and `%cd` into it:
   ```python
   !unzip -q igloo_colab_standalone.zip -d /content/
   %cd /content/colab_standalone
   ```
3. **Runtime → Change runtime type → GPU** (T4 is fine). Not strictly required, but the
   embedding and integrated-gradients cells are much faster with one.
4. Open `IGLOO_playground.ipynb` and run cells top to bottom. The first code cell
   auto-detects Colab and installs the extra packages from `requirements.txt`.

## Running locally / anywhere else

```bash
pip install torch numpy pandas matplotlib scipy scikit-learn -r requirements.txt
jupyter notebook IGLOO_playground.ipynb
```
(The Colab-detection `%pip install` cell is a no-op outside Colab — it checks for the
`google.colab` module before doing anything.)

## Things I could not verify without an actual Colab session

I built and tested this on the original HPC cluster (GPU job, fully isolated from the
source repo to confirm no hidden path dependency — see below), but couldn't run it on
literal Colab infrastructure. Two things worth double-checking there specifically:

- **`esm` package + Colab's preinstalled torch version.** If Colab's default torch build
  is incompatible with what `esm` expects, you may need to pin a torch version before
  installing `esm`.
- **`ESMC.from_pretrained('esmc_300m')` auth.** This downloads model weights on first use
  via the `esm` package. I don't know from here whether it requires a HuggingFace/
  EvolutionaryScale login/token on a fresh machine that's never used it before — if it
  prompts for one, you may need to `huggingface-cli login` or set `HF_TOKEN` first.

If either of these surfaces an error, the fix is almost certainly a version pin or an
auth token, not a bug in the notebook logic itself (the underlying pipeline logic was
verified end-to-end, including a bit-exact cross-check against the original production
IG computation — max abs diff ~1e-10, Pearson r=1.000000).

## Isolation test performed

Before calling this done, this exact folder was copied to a location with **zero
relationship to the source cluster repo** and executed end-to-end via `jupyter nbconvert
--execute` on a GPU node, confirming every import, path lookup, and cache resolves purely
from files inside this folder — nothing reaches back into the original `IGLOO/` checkout.

## What the notebook covers

1. Load the precomputed 7-metric x 60-cell summary table and pick a run (best model
   performance by default).
2. Load that run's trained checkpoint + config + official IG results CSV, cross-checked
   against a live recompute.
3. Embed a chosen sequence and inspect the raw `(1, 56, 960)` tensor.
4. Sparsity/PCA diagnostic on the embedding space.
5. Normalize, forward pass, and attention-pooling weights.
6. Step through the linear-path integrated-gradients computation, cross-checked bit-exact
   against the production CSV.
7. **All 7 metrics' intermediate steps** for the selected model — the actual per-position
   |IG| vs. ground truth, per-sequence NDCG distributions, category means, and a hands-on
   leave-one-out mutation-contribution demo — not just the final numbers.
8. **Best model vs. a "complete random" (shuffled-label) model** — same sequence, both
   models' predictions, attention weights, and full IG compared side by side.
9. **Floor diagnostic** — which of the 7 metrics can actually tell the best model apart
   from the random one, both for this one cell and across the full noise-level sweep.
10. **Prototype a new metric** — a z-score-vs-random-baseline calibration, with room to
    define and test your own.
11. Recreate the metric-vs-model-performance plots for all 7 metrics inline.

## What's NOT bundled (by design, to keep this small)

- The other 59 (noise, seed) cells' checkpoints/results, and random-baseline seeds 1-4 —
  only noise=0.0/seed_1 (best) and random_baseline/seed_0 ship. Section 1 shows which cell
  is "best" per metric; to explore another one, regenerate it (~15 min on one GPU) via
  `scripts/run_integrated_gradients.py`. (`configs/synthetic/GB1_additive_config.yaml`
  itself isn't bundled either; use a config snapshot under `results/.../linear/*.yaml` as
  your template.)
- The shared ESM-C embedding cache (~hundreds of MB on the cluster) — embeddings compute
  fresh here and cache locally as you go.
