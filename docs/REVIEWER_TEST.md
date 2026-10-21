# IGLOO reviewer test

This page is for someone **new to IGLOO** who will try the pipeline from scratch and
write down how it went. Do Workflow A. Workflow B is optional. Then fill in the
[feedback template](#feedback-template) and send it back.

Please time each step and note **every** place you hesitated, guessed, or had to read
the code. Those notes are the point of the exercise.

| Workflow | Goal | Expected time |
|---|---|---|
| Setup | Install IGLOO and pass the health check | ~20–30 min* |
| **A. Reproduce the main result** | Show that some attribution methods still look plausible for a model that has learned nothing | ~30 min, of which ~10 min hands-on and ~20 min of GPU job (plus queue wait) |
| B. Bring your own dataset | Run the pipeline on a CSV you choose | ~20–40 min* |

\* Setup and Workflow B times are estimates and have not been timed yet. Workflow A was
measured on this branch from an empty embedding cache on an H100 MIG slice
(`h100_1g.10gb`): 18.6 min for the whole job, 3.0 GB peak memory. That job reused
already-downloaded ESM-C weights, so on a new account the first run also includes the
weight download (allow a few extra minutes).

---

## Setup

Follow **Installation** in [`README.rst`](../README.rst) (the Alliance-cluster or
workstation instructions). You only need the main venv, not the E1 one. Then:

```bash
python health_check.py
python scripts/pipeline/run_integrated_gradients.py configs/synthetic/GB1_additive_config.yaml --validate-only
```

✅ **Pass:** the health check lists no critical issues, and the second command prints
`Validation complete. Exiting.`

---

## Workflow A: reproduce the main result

**The claim.** IGLOO trains a model to predict protein fitness, then asks which residues
the model relied on (attribution). On a synthetic protein where we *know* which residues
matter, a faithful attribution should:

- match the true important residues when the model is good, and
- stop matching them when the model is useless (trained on random labels).

You will train **two model heads** on the synthetic GB1 landscape, each twice: once on
clean labels (noise 0.0) and once on fully randomized labels (noise 1.0).

- **MeanPool** (`esm_perres_mean`): attribution quality should **collapse** on the
  useless model.
- **Pool First** (`esm_attention`, the default head): attribution quality stays about the
  **same**. Its attributions look plausible even though the model has learned nothing,
  which is the failure IGLOO is built to detect.

### A1. Submit the job

Save this as `reviewer_test.sh` in the repository root. Edit `--account` and the `cd`
path. If the GPU request fails on your cluster, adjust `--gpus`.

```bash
#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:45:00
#SBATCH --output=reviewer_test_%j.out

cd /path/to/IGLOO || exit 1
module load python/3.11 gcc arrow cuda/12.2
source ~/IGLOO_env/bin/activate

for HEAD in esm_perres_mean esm_attention; do
  for NOISE in 0.0 1.0; do
    OUT=results/reviewer_test/${HEAD}/noise_${NOISE}
    python scripts/pipeline/run_integrated_gradients.py configs/synthetic/GB1_additive_config.yaml \
        model.pooling_method=$HEAD \
        integrated_gradients.interpolation_path=linear \
        training.noise.label_noise_mode=blend training.noise.label_noise_level=$NOISE \
        training.noise.label_noise_seed=1 training.seed=1 \
        model_dir=$OUT results_dir=$OUT/linear force_retrain=true
  done
done
```

```bash
sbatch reviewer_test.sh
```

Each of the 4 runs takes about 4 minutes. The first run takes about 6, because it also
embeds the ~1,450 GB1 variants and caches them in `data/embeddings/` (about 300 MB);
the other runs reuse that cache. The whole job takes about 20 minutes once it starts.
While you wait, read §2 "How a run works" in [`docs/README.md`](README.md).

### A2. Look at one run's output

Open `results/reviewer_test/esm_perres_mean/noise_0.0/linear/`. Every run writes the
same set of files (prefix `GB1_additive_`):

| File | What it tells you |
|---|---|
| `*_test_metrics.csv` | How well the model predicts held-out variants (`spearman_rho`, `pearson_r`, `r2`) |
| `*_predicted_vs_actual_test.png`, `*_cv_loss_curves.png` | Model fit and training curves |
| `*_position_importance_summary.csv` | **Per-residue importance**: one row per position (1-based), ranked by `importance_score` |
| `*_top_beneficial_positions.csv`, `*_top_detrimental_positions.csv` | Positions where mutations raise or lower predicted fitness |
| `*_amino_acid_effects_detailed.csv` | Attribution for every variant × position (the raw data) |
| `*_integrated_gradients_heatmap*.png`, `*_signed_gradient_effects.png` | Attribution plots |
| `*_integrated_gradients_summary.txt` | Short summary of the run |
| `*_timing_report.txt` | Time spent in each step (please copy the TOTAL line into your feedback) |
| `*_run_log_*.txt`, `*_config_snapshot_*.yaml` | Full log and the exact config used |

The trained model is saved one level up (`noise_0.0/GB1_additive_training_model.pth`).

### A3. Check the result

```bash
python scripts/evaluation/check_ground_truth.py results/reviewer_test/*/noise_*/linear
```

Expected output (measured on this branch; small differences between GPUs are normal):

```text
                                                   run      dataset  model_rho  attr_rho  top10_hits
  results/reviewer_test/esm_attention/noise_0.0/linear GB1_additive       0.93      0.23        0.70
  results/reviewer_test/esm_attention/noise_1.0/linear GB1_additive      -0.03      0.25        0.90
results/reviewer_test/esm_perres_mean/noise_0.0/linear GB1_additive       0.94      0.62        1.00
results/reviewer_test/esm_perres_mean/noise_1.0/linear GB1_additive      -0.01      0.04        0.70
```

✅ **Pass if all of these hold:**

| Check | Expected | Range seen over 3–10 seeds |
|---|---|---|
| Clean-label models predict well: `model_rho` at noise 0.0 | ≥ 0.85 | 0.88–0.95 |
| Random-label models predict nothing: `model_rho` at noise 1.0 | between −0.15 and 0.15 | −0.06–0.04 |
| MeanPool attribution matches the truth on a good model: `attr_rho`, `esm_perres_mean`, noise 0.0 | ≥ 0.5 | 0.62–0.71 |
| MeanPool attribution collapses on a useless model: `attr_rho`, `esm_perres_mean`, noise 1.0 | ≤ 0.2 | 0.02–0.08 |

Then look at the **Pool First** (`esm_attention`) rows. Its `attr_rho` on the useless
model (seen range 0.04–0.44) is usually about the same as on the good model
(0.17–0.55). Can you explain from the docs alone why that matters? Please note whether
the docs made this clear.

---

## Workflow B (optional): bring your own dataset

Use any CSV with a protein sequence column and a numeric fitness/activity column. If you
don't have one, use the bundled FLIP2 file
`data/FLIP2_processed/alpha_amylase__mutation__by_mutation_combined.csv`.

1. Following [`docs/README.md`](README.md) §3.5 and [`configs/README.md`](../configs/README.md),
   write a config for your CSV. Don't ask the authors; we want to know where the docs
   fall short.
2. Validate it with `--validate-only`.
3. Run it as a GPU job. Use the A1 script with the `for` loops replaced by a single
   command. To keep it short, attribute only a few representative variants:

   ```bash
   python scripts/pipeline/run_integrated_gradients.py configs/my_dataset.yaml \
       analysis.sequence_selection.include_all=false results_dir=results/reviewer_test/my_dataset
   ```

   For reference: with `include_all=true`, all 3,707 alpha-amylase variants took
   about 30 minutes.

✅ **Pass:** the run finishes, `*_test_metrics.csv` and `*_position_importance_summary.csv`
exist, and you can say which positions the model considers most important.

---

## Feedback template

Copy this into a doc and fill it in.

```markdown
# IGLOO reviewer test: <your name>, <date>

Background: <e.g. ML / protein / HPC experience; used Slurm before? y/n>
Machine: <cluster or workstation, GPU type>
Branch / commit: <git rev-parse --short HEAD>

## Timings
| Step | Expected | Actual (hands-on) | Actual (waiting) |
|---|---|---|---|
| Setup | 20–30 min | | |
| A: submit + inspect + check | ~10 min hands-on, ~20 min job | | |
| B: own dataset (optional) | 20–40 min | | |
TOTAL lines from the 4 *_timing_report.txt files:

## Results
- Paste the check_ground_truth.py output:
- Pass / fail for each check in A3:
- In your own words, what does the Pool First vs MeanPool comparison show?

## Friction log (most important)
For each problem: what you were trying to do, what happened, how you got past it, how long it cost.
1.
2.

## Docs
- What was missing, wrong or confusing?
- What did you have to read the source code for?

## Overall
- Could you use IGLOO on your own data next week without help? (1–5):
- Single most important fix:
```
