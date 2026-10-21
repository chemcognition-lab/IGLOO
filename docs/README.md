# IGLOO guide

This is the detailed guide to the IGLOO core pipeline. For the overview and install
commands, see the top-level [`README.rst`](../README.rst).

- [1. Setup notes](#1-setup-notes)
- [2. How a run works](#2-how-a-run-works)
- [3. Workflows](#3-workflows)
- [4. Running on Slurm](#4-running-on-slurm)
- [5. Script reference](#5-script-reference)
- [6. Library reference (`src/igloo/`)](#6-library-reference-srcigloo)

New to IGLOO? Start with [`REVIEWER_TEST.md`](REVIEWER_TEST.md), a ~30-minute
end-to-end run with expected outputs and pass criteria.

Run every command **from the repository root**. Paths in configs (`data.path`,
`results_dir`, `model_dir`) are relative to it.

---

## 1. Setup notes

| Topic | Details |
|---|---|
| **Python** | Python 3.11 or 3.12. Profluent E1 needs Python 3.12 and its own venv (see `README.rst`). Everything else shares one venv. |
| **GPU** | Needed for training, IG, ISM and cascading randomization. Evaluation and figure scripts run on CPU. |
| **`ankh`** | Not in the Alliance wheelhouse, so install it from PyPI. Cluster compute nodes usually have no internet, so install from a login node. |
| **Model weights** | ESM-C, Ankh and E1 weights download on first use and are cached by Hugging Face / the ESM package. Fetch them once before you submit many jobs. |
| **Embedding cache** | Every sequence's embedding is cached at `data/embeddings/by_seq/<model>/L<layer>/<hh>/<sha1>.safetensors` and reused across runs, seeds and heads. Before a large sweep, warm the cache with one run per backbone/layer so parallel jobs don't all recompute it. |
| **Outputs are not committed** | `results/`, `data/embeddings/`, `logs/` and `*.out` Slurm logs in the repository root are git-ignored. Delete them freely, and copy results you want to keep somewhere backed up (on Alliance clusters: `/project`, not `/scratch`). |

---

## 2. How a run works

`scripts/pipeline/run_integrated_gradients.py` is the main entry point:

```text
run_integrated_gradients.py CONFIG [--validate-only] [--skip-ig] [key.subkey=value ...]
```

For one config it:

1. **Loads data** (`data.*`): reads the CSV, picks the target column and normalization,
   and defines the wild type (from a variant column, a row position, or an explicit
   sequence). If `training.noise.*` is set, it adds label noise.
2. **Embeds** every sequence with `model.esm_model_name` at layer `model.esm_layer`,
   using the cache.
3. **Trains** the head given by `model.pooling_method` with K-fold CV and early stopping,
   then saves it to `model_dir`. An existing model is reused unless `force_retrain=true`.
4. **Attributes** the variants chosen by `analysis.sequence_selection` with the method set
   in `integrated_gradients.*`.
5. **Writes** metrics, `*_amino_acid_effects_detailed.csv` (per-residue attributions),
   plots, a timing report and a run log to `results_dir`.

Overrides use dot paths into the YAML and are applied before validation. One base config
plus overrides is the recommended way to sweep.

### Regression heads

| Name | `model.pooling_method` | Idea |
|---|---|---|
| Pool First | `esm_attention` | Attention-pool the residue embeddings, then MLP (the IG baseline) |
| PreMLP | `esm_attention_premlp` | Per-residue MLP, then attention pool |
| MeanPool | `esm_perres_mean` | Per-residue scores, then mean |
| AttnPool | `esm_perres_attn` | Per-residue scores, then attention-weighted sum |
| OneHot | any head + `model.esm_model_name=onehot` | No pLM: one-hot amino acids as input |

Experimental heads that are also accepted: `esm_attention_channel`, `esm_mutation_pool`,
`esm_perres_flatten`, `esm_crossattn_perres`.

---

## 3. Workflows

### 3.1 Synthetic benchmark: is the attribution faithful?

The synthetic landscapes (`data/synthetic/`) store the true per-position weights, so
attributions can be scored directly against them. The main experiment trains the same
model with increasing label noise (so the models get worse) and checks whether
attribution quality falls with model quality.

```bash
# (optional) regenerate the landscapes
python scripts/data/generate_synthetic_datasets.py

# one cell of the noise × seed sweep; repeat over N in {0.0 … 1.0}, S in {1,2,3}
P=GB1; L=additive; N=0.35; S=1
OUT=results/synthetic/seedsweep/${P}_${L}/noise_${N}/seed_${S}
python scripts/pipeline/run_integrated_gradients.py configs/synthetic/${P}_${L}_config.yaml \
    integrated_gradients.interpolation_path=linear \
    training.noise.label_noise_mode=blend training.noise.label_noise_level=$N \
    training.noise.label_noise_seed=$S training.seed=$S \
    model_dir=$OUT results_dir=$OUT/linear force_retrain=true

# score everything under results/synthetic (writes results/synthetic/metric_suite/)
python scripts/evaluation/metric_suite.py --root results/synthetic
```

`metric_suite.py` finds runs by this directory layout, so keep to it:

| Folder under `results/synthetic/` | Contents |
|---|---|
| `seedsweep/<P>_<L>/noise_<N>/seed_<S>/linear/` | Pool First (`esm_attention`) on `esmc_300m` |
| `premlp/<P>_<L>/noise_<N>/seed_<S>/linear/` | PreMLP on `esmc_300m` |
| `newarch_perres_mean/…`, `newarch_perres_attn/…`, `onehot/…` (same sub-layout) | MeanPool, AttnPool, OneHot |
| `<pooling_method>_<backbone>/<P>_<L>/noise_<N>/seed_<S>/linear/` | Heads on other backbones (see 3.3) |
| `ism_attribution/ism_position_*.csv` | ISM outputs (see 3.2) |

It reports per-metric scores, chance-normalised `skill_*` scores, a TOPSIS composite, and
how strongly each metric tracks model test performance. Metric definitions are in
[`attribution_methods_and_metrics.md`](attribution_methods_and_metrics.md).

`scripts/pipeline/run_synthetic_ig_analyses.py` is a simpler batch runner: it covers every
protein × landscape × interpolation path at seed 42, then runs
`scripts/evaluation/evaluate_synthetic_ig.py`. Use it to compare interpolation paths.

### 3.2 Baselines and sanity checks on a trained run

All of these reuse a model trained in 3.1 (`--model-dir`) and its results (`--run-dir`).

```bash
RUN=results/synthetic/seedsweep/GB1_additive/noise_0.0/seed_1

# in-silico mutagenesis (model-dependent baseline)
python scripts/pipeline/ism_attribution.py --run-dir $RUN/linear --model-dir $RUN

# cascading parameter randomization (Adebayo et al.): attributions should degrade
python scripts/sanity_checks/cascading_randomization.py --run-dir $RUN/linear --model-dir $RUN
python scripts/sanity_checks/cascading_randomization_summary.py
```

Gradient×input and input-difference baselines: `scripts/pipeline/run_baseline_attributions.py`
(run with `--help` for options).

### 3.3 Comparing backbones, heads or IG paths

Keep the config and change one override, giving each variant its own output folder. A
trained model is reused across IG paths as long as `model_dir` is the same and
`force_retrain` is false:

```bash
CFG=configs/synthetic/T4Lysozyme_epistatic_config.yaml
for bb in esmc_600m ankh_base; do            # e1_* backbones must run in the E1 venv
  OUT=results/synthetic/esm_perres_mean_$bb/T4Lysozyme_epistatic/noise_0.0/seed_1
  python scripts/pipeline/run_integrated_gradients.py $CFG \
      model.esm_model_name=$bb model.pooling_method=esm_perres_mean training.seed=1 \
      model_dir=$OUT results_dir=$OUT/linear
done
```

For Ankh, `model.esm_layer` selects an intermediate encoder layer.

### 3.4 Real proteins

```bash
python scripts/pipeline/run_integrated_gradients.py configs/FLIP2/alpha_amylase__mutation__by_mutation.yaml
python scripts/pipeline/run_integrated_gradients.py configs/ProteinGym/BLAT_ECOLX_Firnberg_2014_config.yaml
```

- FLIP2 configs use the benchmark's own train/test split (`training.split_method: column`).
- `scripts/data/generate_proteingym_configs.py` regenerates `configs/ProteinGym/` from
  `data/ProteinGym_DMS_Assays_Sub_processed/`.
- To get a per-position consensus with a reliability score, train several seeds
  (`training.seed=1..4`, each with its own `results_dir`), then run
  `scripts/evaluation/ensemble_real_protein_attributions.py`.

### 3.5 Your own dataset

1. Put a CSV in `data/` with a sequence column, a numeric target column, and a way to
   identify the wild type.
2. Copy `configs/FLIP2/alpha_amylase__mutation__by_mutation.yaml`, then set
   `data.name`, `data.path`, `data.target_column`, `data.wildtype` and `results_dir`.
   For a random split instead of a predefined one, remove `training.split_method` and
   `training.split_column`.
3. Check it with `--validate-only`, then run it (on a GPU node).

---

## 4. Running on Slurm

Run anything beyond `--validate-only` inside a job. Below is a minimal array script for
the noise × seed sweep in 3.1. Adjust the account, GPU type, time and paths, and run
`mkdir -p logs` before submitting.

```bash
#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=1-27
#SBATCH --output=logs/igloo_%A_%a.out

cd /path/to/IGLOO || exit 1
module load python/3.11 gcc arrow cuda/12.2
source ~/IGLOO_env/bin/activate

NOISE=(0.0 0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0); SEEDS=(1 2 3)
i=$((SLURM_ARRAY_TASK_ID - 1))
N=${NOISE[$((i / 3))]}; S=${SEEDS[$((i % 3))]}
P=GB1; L=additive
OUT=results/synthetic/seedsweep/${P}_${L}/noise_${N}/seed_${S}

python scripts/pipeline/run_integrated_gradients.py configs/synthetic/${P}_${L}_config.yaml \
    integrated_gradients.interpolation_path=linear \
    training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
    training.noise.label_noise_seed="$S" training.seed="$S" \
    model_dir="$OUT" results_dir="$OUT/linear" force_retrain=true
```

Tips:

- Warm the embedding cache with one short job before you submit an array.
- Request only what a single run needs. Small proteins (GB1, 56 aa) need far less memory
  and time than p53 (393 aa) with `include_all` sequence selection.
- Run `metric_suite.py` after the array finishes. It is CPU-only, so submit it as a CPU
  job with `--dependency=afterok:<array_job_id>`.

---

## 5. Script reference

### `scripts/pipeline/` (GPU)

| Script | Purpose |
|---|---|
| `run_integrated_gradients.py` | **Main entry point.** Config-driven train + evaluate + attribute, with CLI overrides. |
| `run_train_and_eval.py` | Same pipeline without attributions. |
| `run_ig_dataset.py` | Variant of the main runner that loads data via `igloo.dataset.DatasetLoader`. |
| `run_synthetic_ig_analyses.py` | Batch runner: every synthetic protein × landscape × interpolation path, then evaluation. |
| `run_baseline_attributions.py` | Gradient×input and input-difference baselines. |
| `ism_attribution.py` | In-silico mutagenesis attribution for a trained model. |

### `scripts/data/`

| Script | Purpose |
|---|---|
| `generate_synthetic_datasets.py` | Builds the synthetic landscapes and ground-truth weight files in `data/synthetic/`. |
| `generate_proteingym_configs.py` | Writes one config per ProteinGym assay. |

### `scripts/evaluation/` (CPU)

| Script | Purpose |
|---|---|
| `check_ground_truth.py` | Quick per-run check: attribution vs ground truth, next to model test performance. |
| `metric_suite.py` | **Main scorer.** Attribution-quality metrics, skill scores and TOPSIS for IG and ISM across the noise sweep. |
| `evaluate_synthetic_ig.py` | NDCG / AUROC evaluation used by `run_synthetic_ig_analyses.py`. |
| `ensemble_real_protein_attributions.py` | Multi-seed consensus attribution with a cross-seed reliability ratio. |

### `scripts/sanity_checks/`

| Script | Purpose |
|---|---|
| `cascading_randomization.py` | Randomizes layers from the output end toward the input and recomputes IG (GPU). |
| `cascading_randomization_summary.py` | Combines the per-run CSVs into one figure (CPU). |

### `scripts/figures/`

| Script | Purpose |
|---|---|
| `plot_synthetic_overview.py` | Fitness distributions and position-tier overview of the synthetic datasets. |

---

## 6. Library reference (`src/igloo/`)

| Module | Responsibility |
|---|---|
| `config_utils.py` | YAML loading, validation, dot-path CLI overrides |
| `data_processing.py`, `dataset.py` | Loading, normalization, wild-type detection, label noise, splits |
| `esm_adapter.py` | One loader for the ESM-C, ESM-2, Ankh and E1 backends |
| `embedding.py` | Embedding extraction and the `by_seq` cache |
| `models.py` | All regression heads (`pooling_method`) |
| `train_and_eval.py`, `loss.py` | K-fold training, early stopping, evaluation |
| `integrated_gradients.py`, `sequence_ig.py`, `advanced_ig.py` | IG and interpolation paths |
| `completeness_validation.py` | Completeness-axiom checks |
| `mutagenesis.py` | In-silico mutagenesis |
| `synthetic_data.py` | Synthetic landscape generation and ground truth |
| `ig_metrics.py`, `method_selection.py` | Attribution metrics and method ranking |
| `plot.py` | Plotting helpers |
| `igloo.py` | High-level `IGLOO` class |
