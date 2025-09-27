#!/bin/bash

# IGLOO Environment Setup Script
# This script helps set up the environment for the IGLOO project

echo "Setting up IGLOO environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found. Creating conda environment..."

    # Create conda environment from environment.yml
    conda env create -f environment.yml

    echo "Environment created successfully!"
    echo "To activate the environment, run:"
    echo "  conda activate igloo"

elif command -v python3 &> /dev/null && command -v pip &> /dev/null; then
    echo "Conda not found. Using pip to install dependencies..."

    # Create virtual environment
    python3 -m venv venv_erbc_analysis

    # Activate virtual environment
    source venv_erbc_analysis/bin/activate

    # Install dependencies
    pip install -r requirements.txt

    # Install the package in development mode
    pip install -e .

    echo "Environment created successfully!"
    echo "To activate the environment, run:"
    echo "  source venv_erbc_analysis/bin/activate"

else
    echo "Neither conda nor python3/pip found. Please install conda or python3 with pip."
    exit 1
fi

echo ""
echo "Setup complete! You can now run the analysis scripts."
echo "Example usage:"
echo "  python scripts/betalactamase_linear_IG.py"
echo "  python scripts/PafA_linear_IG.py"
