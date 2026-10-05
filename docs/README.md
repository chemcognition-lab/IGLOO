# IGLOO documentation

This is the detailed guide to the IGLOO codebase. For a short overview and the install
commands, see the top-level [`README.rst`](../README.rst).

- [1. Setup notes](#1-setup-notes)
- [2. How a run works](#2-how-a-run-works)
- [3. Workflows](#3-workflows)
- [4. Script catalogue (`scripts/`)](#4-script-catalogue-scripts)
- [5. Slurm job catalogue (`jobs/`)](#5-slurm-job-catalogue-jobs)
- [6. Other documents in `docs/`](#6-other-documents-in-docs)

All commands are run **from the repository root**: scripts locate the project root from
their own path, but configs and `results_dir` values are relative paths.

---

## 1. Setup notes

The install commands are in [`README.rst`](../README.rst#installation). Some details that
are easy to miss:

| Topic | Details |
|---|---|
| **Two virtualenvs** | `IGLOO_env` (Python 3.11) covers ESM-C, ESM-2, Ankh and one-hot. `IGLOO_env_e1` (Python 3.12) is needed only for the Profluent E1 backbones and has `vendor/E1` installed in editable mode. Job scripts pick the right one (see `jobs/backbones/run_plm_backbone_train_array.sh`). |
| **Venv location** | Keep the venvs in `$HOME` and symlink them into the repository. `/project` has a 500K inode quota, and a venv alone can use a large share of it (see `jobs/maintenance/move_venvs_to_home.sh`). |
| **Modules** | Job scripts run `module load python/3.12 gcc arrow cuda/12.2` before activating the venv. |
| **`ankh`** | Not in the Alliance wheelhouse. Install it from PyPI on a login node (compute nodes have no internet). |
| **Model weights** | ESM-C, Ankh and E1 weights are downloaded on first use. Download them once on a login node, or with a `jobs/smoketests/` job on a node with internet, before you submit large arrays. |
| **Embedding cache** | Embeddings are cached per sequence at `data/embeddings/by_seq/<model>/L<layer>/<hh>/<sha1>.safetensors`. Training arrays check that this cache is warm and abort if it is not. Warm it with the matching `jobs/cache/` script first. |
| **Optional dependencies** | `graphfla` is needed only for `scripts/evaluation/fla_correlation_analysis.py`. `openpyxl` is needed only for `scripts/utils/update_tracker.py`. |

Quick check (on a login node; it only validates the config):

```bash
python scripts/pipeline/run_integrated_gradients.py configs/synthetic/GB1_additive_config.yaml --validate-only
```

---

## 2. How a run works

`scripts/pipeline/run_integrated_gradients.py` is the main entry point. Given one YAML
config it:

1. loads the CSV and defines the wild type (`data.*`), then applies label noise if
   `training.noise.*` is set;
2. embeds every sequence with the chosen backbone (`model.esm_model_name`,
   `model.esm_layer`), reading and writing the `by_seq` cache;
3. trains the regression head (`model.pooling_method`) with K-fold CV and early stopping,
   then saves the model to `model_dir` (an existing model is reused unless
   `force_retrain=true`);
4. computes per-residue attributions (`integrated_gradients.*`) for the variants chosen by
   `analysis.sequence_selection`;
5. writes metrics, `*_amino_acid_effects_detailed.csv`, plots, a timing report and a run
   log to `results_dir`.

Command-line flags:

```text
run_integrated_gradients.py CONFIG [--validate-only] [--skip-ig] [key.subkey=value ...]
```

Overrides use dot paths into the YAML, e.g. `model.pooling_method=esm_perres_mean`,
`training.seed=3`, `results_dir=results/x`. Every sweep in `jobs/` is the same config with
different overrides. The full schema is in [`configs/README.md`](../configs/README.md).

Architecture names used in figures and job scripts:

| Name in figures | `pooling_method` | Results folder (esmc_300m) |
|---|---|---|
| Pool First (IG baseline) | `esm_attention` | `results/synthetic/seedsweep/` |
| PreMLP | `esm_attention_premlp` | `results/synthetic/premlp/` |
| MeanPool | `esm_perres_mean` | `results/synthetic/newarch_perres_mean/` |
| AttnPool | `esm_perres_attn` | `results/synthetic/newarch_perres_attn/` |
| OneHot | any; set `model.esm_model_name=onehot` | `results/synthetic/onehot/` |

---

## 3. Workflows

### 3.1 Synthetic benchmark (the main experiment)

```text
generate data ─▶ warm cache ─▶ train + attribute (noise × seed arrays) ─▶ ISM / sanity checks ─▶ metric_suite ─▶ figures
```

| Step | Run locally / interactively | Slurm job(s) |
|---|---|---|
| 1. Landscapes (already committed in `data/synthetic/`) | `python scripts/data/generate_synthetic_datasets.py` | none |
| 2. Warm the embedding cache | none | `jobs/cache/rebuild_warm_cache_v3.sh` (esmc_300m) |
| 3. Pool First noise × seed sweep | `python scripts/pipeline/run_integrated_gradients.py configs/synthetic/<P>_<L>_config.yaml training.noise.label_noise_mode=blend training.noise.label_noise_level=<N> training.seed=<S> ...` | `jobs/synthetic/run_train_array_v2.sh`, `jobs/synthetic/submit_seedsweep.sh` |
| 4. Other heads (PreMLP / MeanPool / AttnPool / OneHot) | same command plus a `model.pooling_method=` override | `jobs/architectures/run_newarch_train_array_v3.sh`, `run_onehot_train_array.sh`, `run_p53_esmc300m_methods_array.sh` |
| 5. ISM, mechanism and cascading randomization | `scripts/pipeline/ism_attribution.py`, `scripts/sanity_checks/cascading_randomization.py --run-dir … --model-dir …` | `jobs/attribution/run_attr_array_v2.sh`, `run_cascading_*.sh` |
| 6. Score everything | `python scripts/evaluation/metric_suite.py --root results/synthetic` | `jobs/analysis/run_analysis.sh` (runs the whole CPU analysis suite) |
| 7. Figures | `python scripts/figures/make_*.py`, then `export_final_figures_nature_methods.py` | none (CPU, fast) |

### 3.2 Comparing pLM backbones and layers

1. Warm the cache: `jobs/cache/rebuild_warm_cache_newbackbones.sh` (Ankh / E1 / ESM-C 600M),
   `rebuild_warm_cache_ankh_layersweep.sh` (Ankh intermediate layers), and the
   `rebuild_warm_cache_p53_*` scripts for p53.
2. Smoke-test that the backbone loads: `jobs/smoketests/smoketest_new_backbones.sh`,
   `smoketest_ankh_layers.sh`.
3. Train: `jobs/backbones/run_plm_backbone_train_array.sh` (every backbone × head × noise),
   `run_ankh_layersweep_train_array.sh`, `run_p53_backbone_array.sh`,
   `run_p53_layersweep_train_array.sh`.
4. Score with `metric_suite.py`, then plot with `make_backbone_*` and
   `make_layersweep_topsis_by_landscape_figure.py`.

### 3.3 Ensembles across seeds

- More seeds: `jobs/synthetic/run_gb1_seedsweep_extra_seeds.sh`,
  `jobs/architectures/run_gb1_other_archs_extra_seeds.sh`.
- Scoring: `scripts/evaluation/ensemble_by_K.py`.
- Figures: `make_poolfirst_ensemble_*`, `make_architecture_ensemble_variance_figure.py`.

### 3.4 Real proteins (FLIP2 alpha-amylase, ProteinGym beta-lactamase)

1. Optional: `python scripts/data/generate_proteingym_configs.py` regenerates
   `configs/ProteinGym/`.
2. Warm the cache: `jobs/cache/rebuild_warm_cache_real_proteins.sh` or
   `rebuild_warm_cache_flip2.sh`.
3. Run: `jobs/real_proteins/run_real_proteins_array.sh`, then
   `run_real_proteins_ensemble_array.sh` (5 seeds). For all four FLIP2 splits, use
   `run_flip2_array.sh`.
4. Seed consensus: `python scripts/evaluation/ensemble_real_protein_attributions.py`.
5. Structure view: `python scripts/figures/generate_ig_colored_pdbs.py` writes IG values
   into PDB B-factors (structures are in `data/PDBs/`).

### 3.5 Your own dataset

1. Put a CSV with a sequence column and a target column in `data/`.
2. Copy a config (e.g. `configs/FLIP2/*.yaml`) and set `data.path`, `data.target_column`
   and `data.wildtype`, plus `results_dir`.
3. `--validate-only` on a login node, then run it inside `salloc` or `sbatch`. Use
   `jobs/real_proteins/run_flip2_IG.sh` as a template.

---

## 4. Script catalogue (`scripts/`)

`scripts/run_integrated_gradients.py` (top level) is a **temporary forwarding shim** for
Slurm jobs submitted before the reorganisation. It can be deleted once those jobs finish.
Always use `scripts/pipeline/run_integrated_gradients.py`.

### `pipeline/`: train models and compute attributions (GPU)

| Script | Purpose |
|---|---|
| `run_integrated_gradients.py` | **Main entry point.** Config-driven train + evaluate + IG, with CLI overrides. |
| `run_train_and_eval.py` | Same pipeline without IG (train and evaluate only). |
| `run_ig_dataset.py` | Variant of the main runner that uses the `DatasetLoader` data path. |
| `run_synthetic_ig_analyses.py` | Batch runner: every protein × landscape × interpolation path, then evaluation. |
| `run_baseline_attributions.py` | Gradient×input and input-difference baseline attributions. |
| `ism_attribution.py` | In-silico mutagenesis attribution for an existing trained model. |

### `data/`: inputs and configs

| Script | Purpose |
|---|---|
| `generate_synthetic_datasets.py` | Builds the GB1 / T4Lysozyme / p53 additive, epistatic and nonlinear landscapes plus ground-truth weights. |
| `generate_proteingym_configs.py` | Writes one config per ProteinGym DMS assay. |
| `zip_amino_acid_effects.py` | Archives the large per-run `*_amino_acid_effects_detailed.csv` files. |

### `evaluation/`: score attribution quality (CPU)

| Script | Purpose |
|---|---|
| `metric_suite.py` | **Main scorer.** All metrics plus TOPSIS for IG, ISM and every architecture or backbone. Other scripts import its constants and functions. |
| `metric_suite_focused.py` | Reduced figure set for the main comparison (IG, PreMLP, MeanPool, AttnPool, ISM). |
| `ensemble_by_K.py` | Scores ensembles of K = 1..5 seeds. |
| `ensemble_real_protein_attributions.py` | Per-position seed consensus and reliability for real proteins. |
| `analyze_attrq_vs_modelperf_noise.py` | Attribution quality vs model test performance, coloured by label noise. |
| `ism_attribution_analysis.py` | Whether ISM quality tracks model performance. |
| `evaluate_synthetic_ig.py` / `_robust.py` | Older NDCG/AUROC evaluation of the interpolation-path sweep (the robust version corrects survivorship bias). |
| `analyze_method_scenario.py`, `compare_methods.py` | Interpolation method × scenario comparison (older sweep). |
| `compare_attribution_coverage.py` | Attribution compared with mutation coverage in the dataset. |
| `analyze_embedding_distribution.py` | Checks interpolation paths against the embedding distribution. |
| `fla_correlation_analysis.py` | Correlates GraphFLA landscape features with attribution quality (needs `graphfla`). |

### `sanity_checks/`: does the attribution reflect the model? (GPU, except summaries)

| Script | Purpose |
|---|---|
| `cascading_randomization.py` | Adebayo-style test: randomizes layers from the output end and recomputes IG. |
| `cascading_randomization_summary.py` | Combines the cascading CSVs into one figure. |
| `attribution_mechanism.py` | Checks that attribution is driven by attention weights in the attention-pool head. |
| `factor_ablation_analysis.py` | Separates the gradient and displacement factors of IG. |
| `k_predictor_agreement.py` | Agreement between seeds on predictions vs on attributions. |
| `ism_seed_agreement.py` | The same k-predictor test for ISM vs IG. |
| `token_level_ig_esmc.py` | Pilot: token-level IG through the full ESM-C model. |

### `figures/`: plots (CPU, read from `results/`)

| Group | Scripts |
|---|---|
| Pool First (IG) | `make_poolfirst_topsis_figure.py`, `make_poolfirst_cascading_figure.py`, `make_poolfirst_cascading_noisebands_figure.py`, `make_poolfirst_summary_composite_figure.py` |
| Ensembles | `make_poolfirst_ensemble_{boxplot,violin,variance,variance_noref}_figure.py`, `make_architecture_ensemble_variance_figure.py` |
| Architectures | `make_architecture_comparison_scatter.py`, `make_architecture_comparison_scatter_allnoise.py`, `make_architecture_metric_heatmap.py`, `make_architecture_summary_composite_figure.py`, `make_cascading_by_landscape_figure.py`, `make_cascading_by_landscape_figure_print.py` |
| Backbones / layers | `make_backbone_size_scaling_figure.py`, `make_backbone_topsis_strip.py`, `make_backbone_summary_composite_figure.py`, `make_layersweep_topsis_by_landscape_figure.py`, `make_meanpool_layersweep_ensemble_composite_figure.py` |
| Datasets / diagnostics | `plot_synthetic_overview.py`, `plot_ig_quality.py`, `generate_ig_heatmaps.py`, `generate_mutation_frequency_heatmap.py`, `generate_ig_colored_pdbs.py` |
| Export | `export_final_figures_nature_methods.py` copies the curated figures to `results/synthetic/final_figures_nature_methods/` at 183 mm / 300 dpi. |

Figure scripts import shared helpers from `evaluation/metric_suite.py` and from each
other. Each script adds its own folder and `evaluation/` to `sys.path`, so run them with
`python scripts/figures/<name>.py`.

### `e15_crossplm/`: E15 cross-pLM experiment (historical)

These are the scripts for the earlier E15 cross-pLM study (`run_E15_phase1/2.py`,
`recompute_E15_metrics.py`, `analyze_E15_*`, `generate_E15_*`). The backbone comparison in
§3.2 replaces it. Some of these scripts still read `config/E15_crosspLM_config.yaml` and
`results/revision/E15_crosspLM/`, which have since been removed.

### `utils/` and `archive/`

- `utils/update_tracker.py`: appends a completed run to the experiment tracker.
- `utils/test_config_validation.py`, `utils/test_unified_runner.py`: validate every
  config.
- `archive/`: superseded scripts kept for reference only (they may not run as they are).

---

## 5. Slurm job catalogue (`jobs/`)

Every job script `cd`s to an **absolute** repository path, activates `IGLOO_env`, and
writes to `jobs/logs/`. Submit from the repository root. Before you reuse a script, check
`--account`, the `cd` path (older scripts point at `/project/aip-bensl/...` or
`/scratch/...`), the GPU type, and `--array`. Most arrays list their tasks in a bash
`TASKS` array at the top; edit that list to change the scope.

| Folder | Contents |
|---|---|
| `cache/` | `rebuild_warm_cache*.sh`: precompute `by_seq` embeddings for a backbone, layer or protein set. Run before any training array. |
| `synthetic/` | Main synthetic sweeps: per-landscape runs (`run_GB1_*`, `run_T4Lysozyme_*`, `run_p53_<landscape>`), interpolation-path × noise sweeps (`submit_sweep*.sh` → `run_sweep_job.sh`), seed sweeps (`submit_seedsweep.sh` → `run_seedsweep_job.sh`, `run_gb1_seedsweep_extra_seeds.sh`), and the rebuild training arrays (`run_train_array*.sh`, `run_noise_sweep_job.sh`, `run_shuffle_job.sh`). |
| `architectures/` | Alternative heads: PreMLP (`run_premlp_job.sh`, `run_fillin_premlp_*`), MeanPool / AttnPool / others (`run_newarch_train_array*.sh`), OneHot, mutation-pool, extra GB1 seeds, and the full p53 method comparison (`run_p53_esmc300m_methods_array.sh`). |
| `backbones/` | pLM backbone and layer sweeps: `run_plm_backbone_train_array.sh`, the Ankh and p53 layer sweeps, `run_p53_backbone_array.sh`, ESM-C 600M / E1 600M fill-ins, the pLM × path sweep (`submit_sweep_pLM.sh` → `run_sweep_pLM_job.sh`). |
| `attribution/` | Attribution and diagnostic jobs on trained models: ISM (`run_ism_*`), cascading randomization (`run_cascading_*`), mechanism, factor ablation, PreMLP diagnostics, gradient×input / input-difference baselines (`submit_baseline_*.sh` → `run_baseline_job.sh`), and the combined arrays `run_attr_array*.sh`. |
| `analysis/` | CPU analysis: `run_analysis.sh` (whole suite), `run_metric_suite_fresh.sh`, `run_fla_analysis.sh`. |
| `real_proteins/` | FLIP2 / ProteinGym runs and the 5-seed real-protein ensemble. |
| `smoketests/` | Short pilots and smoke tests (new backbones, Ankh layers, p53 pilot, venv check). Run one before a large new array. |
| `maintenance/` | One-off housekeeping: moving venvs, wiping stale outputs, zipping results, the disabled rebuild orchestrator. **Read these before running them: some delete data.** |
| `e15_crossplm/` | Historical E15 submission scripts. |

`submit_*.sh` scripts are wrappers run with `bash` (most support `--dry-run`). They call
`sbatch` on the matching `run_*_job.sh` once per cell. Everything else is submitted
directly with `sbatch`.

---

## 6. Other documents in `docs/`

| File | Status |
|---|---|
| [`attribution_methods_and_metrics.md`](attribution_methods_and_metrics.md), [`attribution_formulas.md`](attribution_formulas.md) | Current. Method and metric definitions. |
| [`path2_evidence_report.md`](path2_evidence_report.md) | Current. Write-up of the sanity-check ("Path-2") results. |
| [`results_guide.md`](results_guide.md) | How to read result files. |
| [`configuration/`](configuration/) | Config guide and option summary (see also `configs/README.md`). |
| [`paper_outline.md`](paper_outline.md), [`skills/IGLOO_experiments_brief.md`](skills/IGLOO_experiments_brief.md) | Manuscript planning and experiment brief. |
| `setup/`, `quickstart.rst`, `usage.rst`, `installation.rst`, `development/Makefile` | **Outdated.** These describe the original conda-based setup and mention scripts that no longer exist (`run_all_ig_analyses.py`, `*_linear_IG.py`). Use `README.rst` and this page instead. |
| `config/` | Packaging files (`pyproject.toml` is symlinked to the repository root, plus `requirements*.txt` and conda `environment*.yml`). |
| `index.rst`, `make.bat`, `*.rst` | Sphinx sources (`cd docs && make html`). |
