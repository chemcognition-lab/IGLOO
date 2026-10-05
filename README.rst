🧊 IGLOO
========

**Integrated Gradients for LOcational sensitivity in prOteins**

.. image:: https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg
        :target: https://www.python.org/downloads/
        :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
        :target: https://github.com/ranaabarghout/IGLOO/blob/main/docs/LICENSE
        :alt: License

IGLOO trains regression models on protein language model (pLM) embeddings and computes
per-residue attributions that explain their predictions. It also measures how faithful
those attributions are by checking them against synthetic fitness landscapes with known
ground truth.

.. contents:: **Contents**
   :local:
   :depth: 1


What it does
------------

* **Models**: a frozen pLM backbone (**ESM-C** 300M/600M, **ESM-2**, **Ankh** base/large,
  **Profluent E1** 150M/300M/600M, or a **one-hot** baseline) with a trainable regression
  head (attention pooling, per-residue MLP, mean pooling or per-residue attention).
  Training uses K-fold CV with early stopping, and label noise can be added on purpose.
* **Attributions**: embedding-level Integrated Gradients with ``linear``, ``slerp``,
  ``nearest_neighbor`` or ``guided_ig`` paths and completeness checks.
  Baselines: gradient×input, input-difference and in-silico mutagenesis (ISM).
* **Evaluation**: synthetic GB1, T4 Lysozyme and p53 landscapes (additive, epistatic,
  nonlinear) with ground-truth position weights. A metric suite (Spearman, AUROC,
  precision@k, sign accuracy, …) is combined into a TOPSIS score. Cascading parameter
  randomization tests whether the attributions actually depend on the model.
* **Real data**: ready-to-run configs for FLIP2 alpha-amylase and ProteinGym DMS assays.


Repository layout
-----------------

.. code-block:: text

    IGLOO/
    ├── src/igloo/              # Core library
    ├── scripts/
    │   ├── pipeline/           # Train + attribute (main entry point: run_integrated_gradients.py)
    │   ├── data/               # Generate synthetic datasets / ProteinGym configs
    │   ├── evaluation/         # Score attributions (metric_suite.py, …)
    │   ├── sanity_checks/      # Cascading parameter randomization
    │   └── figures/            # Dataset overview plot
    ├── configs/                # YAML run configs: synthetic/, FLIP2/, ProteinGym/
    ├── data/                   # Datasets; embedding cache is written to data/embeddings/
    ├── docs/                   # Detailed guide (docs/README.md), method/metric definitions
    ├── tests/
    └── health_check.py         # Environment check


Installation
------------

Requires Python 3.11 or 3.12. Training and attribution need a CUDA GPU. Evaluation runs on CPU.

**On a Digital Research Alliance of Canada cluster**, use modules and the Alliance
wheelhouse. Every dependency except ``ankh`` is available there:

.. code-block:: bash

    module load python/3.11 gcc arrow cuda/12.2
    virtualenv --no-download ~/IGLOO_env          # keep venvs in $HOME (inode quota on /project)
    source ~/IGLOO_env/bin/activate
    pip install --no-index --upgrade pip
    pip install --no-index -e .
    pip install ankh                              # from PyPI; run on a login node (needs internet)

**On a workstation:**

.. code-block:: bash

    git clone -b pipeline/core https://github.com/ranaabarghout/IGLOO.git
    cd IGLOO
    python3.11 -m venv IGLOO_env && source IGLOO_env/bin/activate
    pip install -e . ankh

**Profluent E1 (optional)** needs Python 3.12 and a separate environment:

.. code-block:: bash

    python3.12 -m venv IGLOO_env_e1 && source IGLOO_env_e1/bin/activate
    pip install -e .
    git clone https://github.com/Profluent-AI/E1.git vendor/E1 && pip install -e vendor/E1

**Check the install:**

.. code-block:: bash

    python health_check.py
    python scripts/pipeline/run_integrated_gradients.py configs/synthetic/GB1_additive_config.yaml --validate-only

pLM weights are downloaded on first use. On clusters where compute nodes have no internet
access, run one small job, or download the weights on a login node, before you submit
large runs.


Quick start
-----------

Run all commands from the repository root.

**1. Train a model and compute attributions** (GPU):

.. code-block:: bash

    python scripts/pipeline/run_integrated_gradients.py configs/synthetic/GB1_additive_config.yaml

This writes the trained model, test-set metrics, per-residue attributions
(``*_amino_acid_effects_detailed.csv``), plots and a run log to the config's ``results_dir``.

**2. Change anything from the command line** with ``key.subkey=value`` overrides:

.. code-block:: bash

    python scripts/pipeline/run_integrated_gradients.py configs/synthetic/GB1_additive_config.yaml \
        model.esm_model_name=ankh_base \
        model.pooling_method=esm_perres_mean \
        integrated_gradients.interpolation_path=slerp \
        training.noise.label_noise_mode=blend training.noise.label_noise_level=0.35 \
        training.seed=1 \
        model_dir=results/demo/model results_dir=results/demo/slerp

Add ``--skip-ig`` to train and evaluate only.

**3. Score the attributions against ground truth** (CPU):

.. code-block:: bash

    python scripts/evaluation/metric_suite.py --root results/synthetic

**4. Run it on real data:**

.. code-block:: bash

    python scripts/pipeline/run_integrated_gradients.py configs/FLIP2/alpha_amylase__mutation__by_mutation.yaml
    python scripts/pipeline/run_integrated_gradients.py configs/ProteinGym/BLAT_ECOLX_Firnberg_2014_config.yaml

`docs/README.md <docs/README.md>`_ covers each workflow in detail: noise and seed sweeps,
ISM and baselines, sanity checks, ensembles, using your own dataset, and running on Slurm.


Configuration at a glance
-------------------------

=============================================  ==============================================================
Key                                            Values
=============================================  ==============================================================
``model.esm_model_name``                       ``esmc_300m`` (default), ``esmc_600m``, ``esm2_*``,
                                               ``ankh_base``, ``ankh_large``, ``e1_150m``, ``e1_300m``,
                                               ``e1_600m``, ``onehot``
``model.pooling_method``                       ``esm_attention`` (Pool First), ``esm_attention_premlp``
                                               (PreMLP), ``esm_perres_mean`` (MeanPool),
                                               ``esm_perres_attn`` (AttnPool)
``integrated_gradients.interpolation_path``    ``linear``, ``slerp``, ``nearest_neighbor``, ``guided_ig``
``training.noise.*``                           ``label_noise_mode``, ``label_noise_level``,
                                               ``label_noise_seed``
``results_dir`` / ``model_dir``                where outputs and the trained model go
                                               (``force_retrain`` to retrain)
=============================================  ==============================================================

The full schema is in `configs/README.md <configs/README.md>`_.


Documentation
-------------

* `docs/README.md <docs/README.md>`_: setup notes, workflows, script reference, Slurm usage
* `docs/REVIEWER_TEST.md <docs/REVIEWER_TEST.md>`_: a ~45-minute end-to-end test with expected
  outputs and pass criteria (start here if you are new)
* `configs/README.md <configs/README.md>`_: configuration schema
* `docs/attribution_methods_and_metrics.md <docs/attribution_methods_and_metrics.md>`_
  and `docs/attribution_formulas.md <docs/attribution_formulas.md>`_: method and metric definitions


Tests
-----

.. code-block:: bash

    pytest tests/


License and acknowledgments
---------------------------

MIT License (see `docs/LICENSE <docs/LICENSE>`_).

This work builds on ESM-C (EvolutionaryScale), Ankh (Elnaggar et al.), Profluent E1
(Profluent Bio), PyTorch, Captum, and the FLIP2 and ProteinGym benchmarks.

Issues and questions: https://github.com/ranaabarghout/IGLOO/issues ·
`Contact <mailto:ranamoneim@gmail.com>`_

**Citation**: if you use IGLOO in your research, please cite our work (citation details coming soon).
