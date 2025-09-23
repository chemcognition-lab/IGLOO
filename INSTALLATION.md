# Environment Setup Guide

This guide provides multiple options for setting up the ERBC Analysis environment.

## Quick Setup (Recommended)

### Using the Setup Script

The easiest way to set up the environment is using the provided setup script:

```bash
# Make the script executable (if not already)
chmod +x setup_environment.sh

# Run the setup script
./setup_environment.sh
```

This script will automatically detect whether you have conda or pip available and set up the appropriate environment.

## Manual Setup Options

### Option 1: Conda Environment (Recommended for Scientific Computing)

If you have conda installed:

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate erbc_analysis

# Install the package in development mode
pip install -e .
```

### Option 2: Python Virtual Environment with pip

If you prefer using pip:

```bash
# Create virtual environment
python3 -m venv venv_erbc_analysis

# Activate virtual environment
source venv_erbc_analysis/bin/activate  # On Linux/Mac
# OR
venv_erbc_analysis\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 3: Direct pip install

For a simple installation:

```bash
pip install -r requirements.txt
```

## Key Dependencies

The main dependencies included are:

- **Scientific Computing**: numpy, pandas, scipy
- **Machine Learning**: scikit-learn, xgboost
- **Deep Learning**: PyTorch (torch, torchvision)
- **Protein Language Models**: fair-esm
- **Visualization**: matplotlib, seaborn
- **Data Storage**: safetensors
- **Utilities**: tqdm (progress bars)

## GPU Support

For GPU acceleration with PyTorch:

### CUDA 11.8
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### CPU Only
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Verification

To verify your installation works:

```bash
# Test import of main modules
python -c "import torch; import esm; import numpy; import pandas; print('All imports successful!')"

# Run a simple analysis script
python scripts/demo_analysis_methods.py
```

## Troubleshooting

### Common Issues

1. **ESM Installation Issues**: If `fair-esm` fails to install, try:
   ```bash
   pip install fair-esm --no-deps
   pip install biotite
   ```

2. **PyTorch CUDA Issues**: Make sure your CUDA version matches the PyTorch installation.

3. **Memory Issues**: The ESM models require significant GPU memory. For CPU-only usage, expect slower performance.

### Environment File Updates

If you need to update the environment:

```bash
# For conda
conda env update -f environment.yml

# For pip
pip install -r requirements.txt --upgrade
```

## Development Setup

For development work:

```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Running Analysis Scripts

Once the environment is set up, you can run the analysis scripts:

```bash
# Beta-lactamase analysis
python scripts/betalactamase_linear_IG.py

# PafA analysis
python scripts/PafA_linear_IG.py

# XGBoost analysis
python scripts/PafA_xgboost_variant_regression.py
```

Make sure your data files are in the `data/` directory as expected by the scripts.
