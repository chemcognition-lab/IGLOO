🧊 IGLOO
========

**Integrated Gradients for LOcational sensitivity in prOteins**

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
        :target: https://www.python.org/downloads/
        :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
        :target: https://github.com/ranaabarghout/IGLOO/blob/main/LICENSE
        :alt: License

A machine learning pipeline for evaluating Integrated Gradients (IG) attribution quality on protein fitness landscapes. IGLOO trains ESM-2-based regression models, computes IG attributions along configurable interpolation paths, and benchmarks the faithfulness of those attributions against known ground-truth mutation effects using synthetic datasets, FLIP2, and ProteinGym DMS assays.

**Key Features**
------------------

**Model Training & Evaluation**
   * Pluggable protein language model backbone with attention-pooling regression head
   * Supported pLMs: **ESM-C** (300M; EvolutionaryScale), **Ankh** (base/large; T5-encoder),
     and **Profluent E1** (150M / 300M / 600M)
   * Embedding-level Integrated Gradients
   * Train/validation/test splits with early stopping; Pearson r and Spearman ρ monitoring
   * Support for real datasets: FLIP2, ProteinGym DMS benchmarks, and user-provided CSVs

**Interpolation Path Analysis**
   * Three interpolation strategies for IG baseline paths and two non-IG baselines:

     - ``linear`` — straight-line interpolation in embedding space
     - ``nearest_neighbor`` — path through nearest-neighbor embeddings
     - ``slerp`` — spherical linear interpolation
     - ``gradient_x_input`` — gradient × input (no path integration)
     - ``input_difference`` — direct input-difference attribution

   * Completeness axiom validation for all path-based methods

**Synthetic Benchmarking**
   * Controlled ground-truth fitness landscapes for GB1, p53, and T4 Lysozyme
   * Three landscape types: *additive*, *epistatic*, and *nonlinear*
   * Quantitative IG quality metrics: global NDCG, per-category AUROC/AUPRC/AP-graded,
     sign accuracy, category separation, mutation-tier NDCG
   * Sweep across all (interpolation method × landscape type × protein) combinations
   * Publication-ready standalone panel figures with separate legend exports



**Quick Start**
-----------------

.. code-block:: bash

    git clone https://github.com/ranaabarghout/IGLOO.git
    cd IGLOO
    conda create -n IGLOO_env python=3.10
    conda activate IGLOO_env
    pip install -e .

Verify your environment:

.. code-block:: bash

    ./health_check.py


**System Requirements**
--------------------------

* Python 3.9+ (3.10 recommended)
* NVIDIA GPU with ≥ 16 GB VRAM (L40S / H100 used in development)
* 64 GB+ RAM recommended for large embedding sweeps
* CUDA 11.8+ or 12.1+


**Configuration**
-------------------

IGLOO uses YAML configuration files in ``configs/``.

Each config specifies dataset path, wild-type identification method, ESM layer, model architecture, and IG parameters.  See `configs/README.md <configs/README.md>`_ for the full schema.

**Synthetic configs** (``configs/synthetic/*.yaml``):

.. code-block:: bash

    configs/synthetic/
    ├── GB1_additive_config.yaml
    ├── GB1_epistatic_config.yaml
    ├── GB1_nonlinear_config.yaml
    ├── p53_additive_config.yaml   ...
    └── T4Lysozyme_nonlinear_config.yaml ...

**FLIP2 / ProteinGym configs** (``configs/FLIP2/``, ``configs/ProteinGym/``).


**Usage**
-----------

**Train & run IG on a real protein dataset**

.. code-block:: bash

    python scripts/run_integrated_gradients.py --config configs/pafa_config.yaml
    python scripts/run_integrated_gradients.py --config configs/betalactamase_config_embedding-signed.yaml

    # Validate config without running
    python scripts/run_integrated_gradients.py --config configs/synthetic/GB1_additive_config.yaml --validate-only

**Generate synthetic benchmarking datasets**

.. code-block:: bash

    python scripts/generate_synthetic_datasets.py

**Run the full synthetic IG sweep (SLURM)**

.. code-block:: bash


**Evaluate and plot synthetic results**

.. code-block:: bash

    # Compute IG quality metrics and generate plots for one protein
    python scripts/evaluate_synthetic_ig.py \
        --results-dir results/synthetic/GB1_nonlinear \
        --output-dir results/synthetic/evaluation_plots/GB1_results

    # Regenerate plots only (skip recomputation)
    python scripts/plot_ig_quality.py --plot-only \
        --output-dir results/synthetic/evaluation_plots/GB1_results

    # Dataset overview figures (mutation tiers + fitness distributions)
    python scripts/plot_synthetic_overview.py
    python scripts/plot_synthetic_overview.py --transparent   # transparent background

**Analyze attribution method quality across scenarios**

.. code-block:: bash

    python scripts/analyze_method_scenario.py --plot-only \
        --output-dir results/synthetic/method_analysis

**Compare attribution methods (composite score)**

.. code-block:: bash

    python scripts/compare_methods.py

**Baseline attributions**

.. code-block:: bash

    python scripts/run_baseline_attributions.py --config configs/pafa_config.yaml

**IG-colored PDB structures**

.. code-block:: bash

    python scripts/generate_ig_colored_pdbs.py \
        --ig-csv results/PafA_embedding_magnitude/attributions.csv \
        --pdb data/PDBs/PafA.pdb \
        --output results/colored_pdbs/


**Key Scripts**
-----------------

+---------------------------------------------+----------------------------------------------------------+
| Script                                      | Purpose                                                  |
+=============================================+==========================================================+
| ``run_integrated_gradients.py``             | Train model and compute IG on real datasets              |
+---------------------------------------------+----------------------------------------------------------+
| ``generate_synthetic_datasets.py``          | Build GB1 / p53 / T4 synthetic fitness landscapes        |
+---------------------------------------------+----------------------------------------------------------+
| ``run_synthetic_ig_analyses.py``            | Run IG sweep over all synthetic configs                  |
+---------------------------------------------+----------------------------------------------------------+
| ``evaluate_synthetic_ig.py``               | Compute IG quality metrics from sweep results            |
+---------------------------------------------+----------------------------------------------------------+
| ``evaluate_synthetic_ig_robust.py``        | Robust version with collapse detection                   |
+---------------------------------------------+----------------------------------------------------------+
| ``plot_ig_quality.py``                      | Generate all evaluation panel figures                    |
+---------------------------------------------+----------------------------------------------------------+
| ``plot_synthetic_overview.py``              | Dataset overview (tiers, fitness distributions)          |
+---------------------------------------------+----------------------------------------------------------+
| ``analyze_method_scenario.py``              | Cross-scenario interpolation method comparison           |
+---------------------------------------------+----------------------------------------------------------+
| ``compare_methods.py``                      | Composite method score figure                            |
+---------------------------------------------+----------------------------------------------------------+
| ``run_baseline_attributions.py``            | Gradient×input / input-difference baselines              |
+---------------------------------------------+----------------------------------------------------------+
| ``generate_ig_colored_pdbs.py``             | Map IG attributions onto PDB B-factors                   |
+---------------------------------------------+----------------------------------------------------------+
| ``run_train_and_eval.py``                   | Standalone train + eval without IG                       |
+---------------------------------------------+----------------------------------------------------------+

**Core Library Modules** (``src/igloo/``)
-------------------------------------------

* ``igloo.py`` — top-level IGLOO class; orchestrates training and IG computation
* ``integrated_gradients.py`` — path-integral IG with all five interpolation methods
* ``advanced_ig.py`` — slerp, nearest-neighbor, and gradient-based path implementations
* ``completeness_validation.py`` — axiom checking for IG paths
* ``ig_metrics.py`` — NDCG, AUROC, AUPRC, sign-accuracy, category-separation metrics
* ``method_selection.py`` — aggregate metric loading and ranking across scenarios
* ``synthetic_data.py`` — synthetic fitness landscape generation (additive / epistatic / nonlinear)
* ``models.py`` — pLM attention-pooling regression head (shared across ESM-C / Ankh / E1)
* ``esm_adapter.py`` — unified backend loader for ESM-C, Ankh, and Profluent E1
* ``train_and_eval.py`` — training loop, early stopping, Pearson/Spearman evaluation
* ``plot.py`` — all figure generation (standalone panels, breakdowns, overviews)
* ``data_processing.py`` — dataset loading, normalization, train/val/test splitting
* ``embedding.py`` — embedding extraction and caching for all supported pLMs
* ``config_utils.py`` — YAML config validation and loading


**Results Structure**
-----------------------

.. code-block:: text

    results/
    ├── synthetic/
    │   ├── GB1_nonlinear/          # Raw IG outputs per interpolation method
    │   ├── p53_additive/
    │   ├── T4Lysozyme_epistatic/
    │   ├── evaluation_plots/       # Per-protein panel figures + CSVs
    │   │   ├── GB1_results/
    │   │   ├── p53_results/
    │   │   └── T4_results/
    │   └── method_analysis/        # Cross-scenario composite figures
    ├── FLIP2/
    └── ProteinGym/


**Environment Validation**
----------------------------

.. code-block:: bash

    ./health_check.py

Checks Python version, GPU/CUDA availability, installed packages, and data file integrity.


**Documentation**
-------------------

* `Configuration Guide <configs/README.md>`_
* `Configuration Summary <configs/CONFIGURATION_SUMMARY.md>`_
* `Results Guide <docs/results_guide.md>`_


**License**
-------------

MIT License — see `LICENSE <LICENSE>`_.

**Acknowledgments**
---------------------

* ESM-C protein language models by EvolutionaryScale
* Ankh protein language models (Elnaggar et al.)
* Profluent E1 protein language models by Profluent AI
* PyTorch and Captum for attribution method infrastructure
* FLIP2 and ProteinGym benchmark datasets

**Support**
-------------

* `Report Issues <https://github.com/ranaabarghout/IGLOO/issues>`_
* `Discussions <https://github.com/ranaabarghout/IGLOO/discussions>`_
* `Contact <mailto:ranamoneim@gmail.com>`_

---

**Citation**: If you use IGLOO in your research, please cite our work (citation details coming soon).

**Development Status**: Active development — contributions welcome!
