🧊 IGLOO
========

**Integrated Gradients for LOcational sensitivity in prOteins**

.. image:: https://img.shields.io/pypi/v/igloo.svg
        :target: https://pypi.python.org/pypi/igloo
        :alt: PyPI Version

.. image:: https://img.shields.io/github/workflow/status/ranaabarghout/IGLOO/Tests
        :target: https://github.com/ranaabarghout/IGLOO/actions
        :alt: Build Status

.. image:: https://readthedocs.org/projects/igloo/badge/?version=latest
        :target: https://igloo.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
        :target: https://www.python.org/downloads/
        :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
        :target: https://github.com/ranaabarghout/IGLOO/blob/main/LICENSE
        :alt: License

A comprehensive machine learning pipeline for analyzing structural contributions to performance in protein variants using Integrated Gradients and advanced attribution methods.

**Key Features**
------------------

**Protein Analysis**
   * ESM-2 protein language model integration
   * Token-level and embedding-level attribution analysis
   * Support for multiple protein datasets (PafA, Beta-lactamase, ERBC)
   * Mutation effect analysis with beneficial/detrimental classification

**Attribution Methods**
   * Integrated Gradients with magnitude and signed analysis
   * Comprehensive statistical analysis of amino acid contributions
   * Position-specific and amino acid type analysis

**Visualization & Results**
   * 3D protein structure visualization with PyMOL and py3Dmol
   * Comprehensive CSV exports for further analysis
   * Publication-ready plots and figures


**Quick Start**
-----------------

**One-Command Setup (Recommended)**

.. code-block:: bash

    # Clone and setup in one go
    git clone https://github.com/ranaabarghout/IGLOO.git && cd IGLOO && ./docs/setup/quick_setup.sh

See `Quick Start Guide <docs/quickstart.rst>`_ for detailed setup instructions and alternative installation methods.

**Manual Setup**

.. code-block:: bash

    # Create conda environment
    conda env create -f docs/config/environment.yml
    conda activate igloo

    # Or use pip with requirements
    pip install -r docs/config/requirements.txt

**Configuration Files:**
   * **Environment**: `docs/config/environment.yml <docs/config/environment.yml>`_
   * **Requirements**: `docs/config/requirements.txt <docs/config/requirements.txt>`_
   * **Development**: `docs/config/requirements_dev.txt <docs/config/requirements_dev.txt>`_
   * **Project Config**: `docs/config/pyproject.toml <docs/config/pyproject.toml>`_
   * **Docker**: `docs/config/docker-compose.yml <docs/config/docker-compose.yml>`_


**System Requirements**
--------------------------

**Minimum Requirements:**
   * Python 3.9+ (Python 3.10 recommended)
   * 8GB RAM (16GB+ recommended)
   * 5GB free disk space

**Recommended for Optimal Performance:**
   * NVIDIA GPU with 8GB+ VRAM
   * 32GB+ RAM for large protein datasets
   * CUDA 11.8+ or 12.1+
   * SSD storage

**Usage Examples**
--------------------

**Configuration-Driven Analysis (Recommended)**

.. code-block:: bash

    # Run analysis using the main configuration-driven script
    python scripts/run_integrated_gradients.py configs/pafa_config.yaml
    python scripts/run_integrated_gradients.py configs/betalactamase_config.yaml
    python scripts/run_integrated_gradients.py configs/erbc_config.yaml

    # Run with specific attribution methods
    python scripts/run_integrated_gradients.py configs/pafa_config_token-magnitude.yaml
    python scripts/run_integrated_gradients.py configs/betalactamase_config_embedding-signed.yaml

**Run Complete Analysis Pipeline**

.. code-block:: bash

    # Run all analyses across datasets
    make analysis

    # Or run specific dataset analysis
    python scripts/PafA_linear_IG.py
    python scripts/betalactamase_linear_IG.py
    python scripts/ERBC_linear_IG.py

**Interactive Analysis with Jupyter**

.. code-block:: bash

    make jupyter
    # Access at http://localhost:8888

**3D Protein Structure Visualization**

.. code-block:: python

    from igloo.plot import plot_attributions_on_3d_structure

    # Map attributions onto 3D protein structure
    plot_attributions_on_3d_structure(
        pdb_file="data/PafA_structure.pdb",
        attributions=attribution_data,
        output_file="results/3d_visualization.html"
    )

**Custom Analysis**

.. code-block:: python

    from igloo.utils import create_visualizations_and_exports

    # Run integrated gradients analysis
    results = create_visualizations_and_exports(
        model=your_model,
        data=your_data,
        method="integrated_gradients",
        output_dir="results/"
    )

🛠️ **Available Commands**
-------------------------

.. code-block:: bash

    # Environment setup and validation
    make -f docs/development/Makefile dev-setup    # Complete development setup
    ./health_check.py                              # Comprehensive environment validation

    # Analysis and experiments
    make -f docs/development/Makefile analysis     # Run full analysis pipeline
    make -f docs/development/Makefile analysis-pafa # Run PafA analysis only
    make -f docs/development/Makefile demo         # Run demo analysis

    # Testing and quality
    make -f docs/development/Makefile test         # Run full test suite
    make -f docs/development/Makefile test-fast    # Quick tests only
    make -f docs/development/Makefile lint         # Code linting
    make -f docs/development/Makefile format       # Code formatting

    # Docker operations (using docker-compose in docs/config/)
    docker-compose -f docs/config/docker-compose.yml up dev    # Development container
    docker-compose -f docs/config/docker-compose.yml up prod   # Production container

    # Documentation and utilities
    make -f docs/development/Makefile docs         # Build documentation
    make -f docs/development/Makefile jupyter      # Start Jupyter Lab
    make -f docs/development/Makefile clean-all    # Deep cleanup

**Environment Validation**
----------------------------

Verify your installation with our comprehensive health check:

.. code-block:: bash

    ./health_check.py

This will check:
   * ✅ Python version compatibility
   * ✅ Package dependencies
   * ✅ GPU/CUDA availability
   * ✅ System resources
   * ✅ Data file integrity
   * ✅ Basic functionality tests

**Documentation**
-------------------

* **Quick Start Guide**: `docs/quickstart.rst <docs/quickstart.rst>`_
* **Installation Guide**: `docs/setup/installation.md <docs/setup/installation.md>`_
* **Environment Setup**: `docs/setup/environment_setup.md <docs/setup/environment_setup.md>`_
* **Setup Files Overview**: `docs/setup/setup_guide.md <docs/setup/setup_guide.md>`_
* **Configuration Guide**: `docs/configuration/configuration_guide.md <docs/configuration/configuration_guide.md>`_
* **Results Guide**: `docs/results_guide.md <docs/results_guide.md>`_
* **API Documentation**: Coming soon
* **Examples**: ``examples/`` directory

**Contributing**
------------------

We welcome contributions! To get started:

.. code-block:: bash

    # Development setup
    make dev-setup
    make install-pre-commit

    # Run tests
    make test

    # Check code quality
    make check-all

See `docs/development/contributing.rst <docs/development/contributing.rst>`_ for detailed guidelines.

**Troubleshooting**
---------------------

**Common Issues:**

* **CUDA Issues**: Run ``nvidia-smi`` to check drivers, reinstall PyTorch with correct CUDA version
* **Memory Issues**: Reduce batch sizes in configuration, enable memory optimization
* **Import Errors**: Check ``PYTHONPATH``, reinstall with ``pip install -e .``
* **Package Conflicts**: Clean install with ``make clean-all`` and re-setup

**Platform-Specific:**

* **macOS**: Install Xcode command line tools, use native ARM conda for M1/M2
* **Linux**: Install build essentials, check CUDA paths
* **Windows**: Install Visual C++ Build Tools, consider WSL2

**License**
-------------

This project is licensed under the MIT License - see the `LICENSE <LICENSE>`_ file for details.

**Acknowledgments**
---------------------

* ESM-2 protein language models by Meta AI
* PyTorch and Captum teams for attribution methods
* The broader computational biology community

**Support**
-------------

* `Report Issues <https://github.com/ranaabarghout/IGLOO/issues>`_
* `Ask Questions <https://github.com/ranaabarghout/IGLOO/discussions>`_
* `Contact <mailto:ranamoneim@gmail.com>`_
* `Documentation <https://igloo.readthedocs.io>`_

---

**Citation**: If you use IGLOO in your research, please cite our work (citation details coming soon).

**Development Status**: Active development - contributions welcome!
