# IGLOO Playground

An interactive, Colab-like notebook for tuning and investigating the IGLOO attribution
pipeline step by step — GB1 additive, Linear interpolation, focused on the
best-model-performance run by default.

## What's in this folder

```
colab/
  IGLOO_playground.ipynb        <- the notebook, start here
  igloo_playground_utils.py     <- thin, transparent wrappers around src/igloo that expose
                                    intermediate tensors (embeddings, attention weights,
                                    per-step IG gradients) instead of hiding them
  README.md                     <- this file
  data/
    GB1_additive.csv                                  <- the synthetic dataset
    GB1_ground_truth_weights.json                      <- per-residue importance annotations
    GB1_additive_config.yaml                            <- base training/IG config
    GB1_additive_signal_families_vs_model_perf_raw_linear.csv   <- all 60 (noise,seed) x 7 metrics
    GB1_additive_random_baseline_linear.csv             <- the shuffled-label "chance" baseline (5 seeds)
```

**Design choice:** this folder does *not* duplicate the multi-GB ESM embedding cache
(`../data/embeddings/`) or the full `results/synthetic/` tree — those are referenced by
relative path from the notebook, since `colab/` lives inside the same `IGLOO/` checkout.
Only the small, self-contained reference files above are copied in. If you move this
folder outside the repo, embedding lookups and model loading will stop hitting the cache
and fall back to slow recomputation (or fail for model loading).

## Setup (one-time)

Packages needed beyond the existing project environment: `jupyter`, `ipykernel`,
`nbconvert`. These were installed directly into the project's `IGLOO_env` venv (not a
separate environment — duplicating the ~GB PyTorch/ESM stack for a notebook-only env
wasn't worth it) and a kernel was registered:

```bash
cd /home/uoftshen/scratch/IGLOO
IGLOO_env/bin/pip install jupyter ipykernel nbconvert   # already done
IGLOO_env/bin/python3 -m ipykernel install --user --name igloo_env --display-name "IGLOO_env"   # already done
```

If you're setting this up fresh (new clone, new venv), just re-run those two commands.

## Launching

This is a compute cluster — **don't run the notebook's heavy cells on the login node**.
Two ways to get a GPU-backed Jupyter session:

### Option A — VSCode / Cursor remote (recommended)
1. Connect your editor to the cluster via Remote-SSH (or open an interactive terminal
   inside a `salloc` session).
2. Open `colab/IGLOO_playground.ipynb`.
3. Select kernel **IGLOO_env** (top-right kernel picker).
4. Run cells top to bottom.

### Option B — salloc + Jupyter + SSH port forward
```bash
# On the cluster:
salloc --account=aip-bensl --gres=gpu:l40s:1 --cpus-per-task=4 --mem=32G --time=2:00:00
module load python/3.12 cuda/12.2
cd /home/uoftshen/scratch/IGLOO
source IGLOO_env/bin/activate
jupyter notebook --no-browser --port=8888 --notebook-dir=colab

# On your laptop, in a new terminal:
ssh -L 8888:<compute-node-hostname>:8888 <cluster-login>
# then open the printed http://localhost:8888/?token=... URL locally
```

### GPU vs CPU
Most cells (loading the model, loading cached embeddings, reading CSVs, plotting) run
fine on CPU and are effectively instant since almost everything hits an existing disk
cache. Only the integrated-gradients cell (Section 6) benefits meaningfully from a GPU
— on CPU it'll just be slower (still fine for one sequence at `steps=500`), and embedding
a genuinely new sequence not already in `data/embeddings/` needs a GPU to be fast (the ESM-C
forward pass itself).

## What the notebook covers

1. Load the precomputed 7-metric x 60-cell summary table and pick a run (defaults to
   the best model-performance cell, currently noise=0.0/seed_1).
2. Load that run's trained checkpoint + config + official IG results CSV.
3. Embed a chosen sequence and inspect the raw `(1, 56, 960)` tensor.
4. Sparsity / PCA diagnostic on the embedding space — for hypotheses like "would a
   different pooling/reduction technique help given how the embedding variance is
   distributed."
5. Normalize, run the forward pass, and visualize attention-pooling weights per residue.
6. Step through the linear-path integrated-gradients computation batch by batch,
   printing every intermediate tensor shape, then cross-check the result against the
   official precomputed CSV values.
7. Recreate `GB1_additive_ndcg_vs_model_test_spearman_linear.png` (or any of the other
   6 metric families) inline, with the random-baseline overlay — no file written.

## Known gap

Only the newly-generated decile-fill noise levels (0.2, 0.4, 0.6, 0.7, 0.9, each x 5
seeds) have saved model checkpoints. The noise levels recovered from the original
`results_seed0_4.zip` backup (0.0, 0.1, 0.3, 0.5, 0.8, 1.0) have IG results but are
missing the `.pth` checkpoint. If you pick one of those cells in Section 1, Section 2's
`load_checkpoint` will raise `FileNotFoundError` with a pointer to
`jobs/run_sweep_cell_gb1_deciles.sh` for regenerating it (~15 min on one GPU). The
default focus cell (noise=0.0/seed_1) was regenerated for this notebook specifically so
it works out of the box.
