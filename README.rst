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

✨ **Key Features**
------------------

🧬 **Protein Analysis**
   * ESM-2 protein language model integration
   * Token-level and embedding-level attribution analysis
   * Support for multiple protein datasets (PafA, Beta-lactamase, ERBC)
   * Mutation effect analysis with beneficial/detrimental classification

🎯 **Attribution Methods**
   * Integrated Gradients with magnitude and signed analysis
   * XGBoost-based variant regression
   * Comprehensive statistical analysis of amino acid contributions
   * Position-specific and amino acid type analysis

📊 **Visualization & Results**
   * 3D protein structure visualization with PyMOL and py3Dmol
   * Interactive HTML visualizations
   * Comprehensive CSV exports for further analysis
   * Publication-ready plots and figures

⚡ **High Performance**
   * GPU acceleration with CUDA support
   * Batch processing for large datasets
   * Memory-optimized operations
   * Multi-platform support (Linux, macOS, Windows)

🚀 **Quick Start**
-----------------

**Option 1: One-Command Setup (Recommended)**

.. code-block:: bash

    # Clone and setup in one go
    git clone https://github.com/ranaabarghout/IGLOO.git && cd IGLOO && ./quick_setup.sh

**Option 2: Platform-Specific Setup**

.. code-block:: bash

    # macOS (with Apple Silicon optimizations)
    ./setup_macos.sh

    # Linux (with CUDA support)
    ./setup_linux.sh

    # Windows
    setup_windows.bat

**Option 3: Using Make**

.. code-block:: bash

    make dev-setup    # Complete development setup
    make prod-setup   # Production setup
    make docker-dev   # Docker development environment

**Option 4: Manual Setup**

.. code-block:: bash

    # Conda (recommended)
    conda env create -f environment.yml
    conda activate igloo
    pip install -e .

    # Or pip virtual environment
    python3 -m venv venv_igloo
    source venv_igloo/bin/activate
    pip install -r requirements.txt
    pip install -e .

📋 **System Requirements**
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

**Platform Support:**
   * ✅ Linux (Ubuntu 18.04+, CentOS, RHEL)
   * ✅ macOS (10.15+, native Apple Silicon support)
   * ✅ Windows 10+ (WSL2 recommended)

🔬 **Usage Examples**
--------------------

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
    make dev-setup           # Complete development setup
    make health-check        # Comprehensive environment validation
    make setup-env          # Create .env configuration

    # Analysis and experiments
    make analysis           # Run full analysis pipeline
    make analysis-pafa      # Run PafA analysis only
    make demo               # Run demo analysis

    # Testing and quality
    make test               # Run full test suite
    make test-fast          # Quick tests only
    make lint               # Code linting
    make format             # Code formatting

    # Docker operations
    make docker-dev         # Development container
    make docker-prod        # Production container
    make docker-test        # Testing container

    # Documentation and utilities
    make docs               # Build documentation
    make jupyter            # Start Jupyter Lab
    make clean-all          # Deep cleanup

🔍 **Environment Validation**
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

📚 **Documentation**
-------------------

* **Installation Guide**: `INSTALLATION.md <INSTALLATION.md>`_
* **Environment Setup**: `ENVIRONMENT_SETUP.md <ENVIRONMENT_SETUP.md>`_
* **Setup Files Overview**: `SETUP_README.md <SETUP_README.md>`_
* **API Documentation**: Coming soon
* **Examples**: ``examples/`` directory

🤝 **Contributing**
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

See `CONTRIBUTING.rst <CONTRIBUTING.rst>`_ for detailed guidelines.

🐛 **Troubleshooting**
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

📄 **License**
-------------

This project is licensed under the MIT License - see the `LICENSE <LICENSE>`_ file for details.

🙏 **Acknowledgments**
---------------------

* ESM-2 protein language models by Meta AI
* PyTorch and Captum teams for attribution methods
* The broader computational biology community

📞 **Support**
-------------

* 🐛 `Report Issues <https://github.com/ranaabarghout/IGLOO/issues>`_
* 💬 `Ask Questions <https://github.com/ranaabarghout/IGLOO/discussions>`_
* 📧 `Contact <mailto:ranamoneim@gmail.com>`_
* 📖 `Documentation <https://igloo.readthedocs.io>`_

---

**Citation**: If you use IGLOO in your research, please cite our work (citation details coming soon).

**Development Status**: Active development - contributions welcome!
