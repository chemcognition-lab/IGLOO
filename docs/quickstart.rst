**Quick Start**
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