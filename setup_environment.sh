#!/bin/bash

# IGLOO Environment Setup Script
# This script helps set up the environment for the IGLOO project

set -e  # Exit on any error

echo "🧊 Setting up IGLOO environment..."
echo "================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    local python_cmd=$1
    local version=$($python_cmd --version 2>&1 | awk '{print $2}')
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)

    if [ "$major" -eq 3 ] && [ "$minor" -ge 9 ]; then
        echo "✅ Python $version found (compatible)"
        return 0
    else
        echo "❌ Python $version found (requires Python 3.9+)"
        return 1
    fi
}

# Parse command line arguments
SETUP_TYPE=""
DEV_MODE=false
FORCE_REINSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --conda)
            SETUP_TYPE="conda"
            shift
            ;;
        --pip)
            SETUP_TYPE="pip"
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --force)
            FORCE_REINSTALL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --conda           Force use of conda"
            echo "  --pip             Force use of pip"
            echo "  --dev             Install development dependencies"
            echo "  --force           Force reinstall even if environment exists"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-detect setup type if not specified
if [ -z "$SETUP_TYPE" ]; then
    if command_exists conda; then
        SETUP_TYPE="conda"
        echo "🔍 Auto-detected: Using conda"
    elif command_exists python3 && command_exists pip; then
        SETUP_TYPE="pip"
        echo "🔍 Auto-detected: Using pip"
    else
        echo "❌ Neither conda nor python3/pip found."
        echo "Please install conda (recommended) or Python 3.9+ with pip."
        exit 1
    fi
fi

# Setup with conda
if [ "$SETUP_TYPE" = "conda" ]; then
    echo "📦 Setting up with conda..."

    # Check if environment already exists
    if conda env list | grep -q "igloo" && [ "$FORCE_REINSTALL" = false ]; then
        echo "⚠️  Environment 'igloo' already exists."
        echo "To reinstall, run: $0 --conda --force"
        echo "To activate existing environment: conda activate igloo"
        exit 0
    fi

    if [ "$FORCE_REINSTALL" = true ]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n igloo -y 2>/dev/null || true
    fi

    # Create conda environment
    echo "🏗️  Creating conda environment from environment.yml..."
    conda env create -f environment.yml

    if [ "$DEV_MODE" = true ]; then
        echo "🛠️  Installing development dependencies..."
        conda activate igloo
        pip install -r requirements_dev.txt
    fi

    echo "✅ Conda environment created successfully!"
    echo ""
    echo "To activate the environment:"
    echo "  conda activate igloo"

# Setup with pip
elif [ "$SETUP_TYPE" = "pip" ]; then
    echo "🐍 Setting up with pip..."

    # Check Python version
    if ! check_python_version python3; then
        echo "❌ Please install Python 3.9 or later"
        exit 1
    fi

    # Setup virtual environment
    VENV_DIR="venv_igloo"

    if [ -d "$VENV_DIR" ] && [ "$FORCE_REINSTALL" = false ]; then
        echo "⚠️  Virtual environment '$VENV_DIR' already exists."
        echo "To reinstall, run: $0 --pip --force"
        echo "To activate existing environment: source $VENV_DIR/bin/activate"
        exit 0
    fi

    if [ "$FORCE_REINSTALL" = true ]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi

    # Create virtual environment
    echo "🏗️  Creating virtual environment..."
    python3 -m venv "$VENV_DIR"

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip setuptools wheel

    # Install dependencies
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt

    if [ "$DEV_MODE" = true ]; then
        echo "🛠️  Installing development dependencies..."
        pip install -r requirements_dev.txt
    fi

    # Install the package in development mode
    echo "🔧 Installing IGLOO package in development mode..."
    pip install -e .

    echo "✅ Virtual environment created successfully!"
    echo ""
    echo "To activate the environment:"
    echo "  source $VENV_DIR/bin/activate"
fi

echo ""
echo "🎉 Setup complete! You can now run the analysis scripts."
echo ""
echo "📚 Example usage:"
echo "  python scripts/betalactamase_linear_IG.py"
echo "  python scripts/PafA_linear_IG.py"
echo "  python scripts/run_all_ig_analyses.py"
echo ""
echo "🧪 To run tests:"
echo "  pytest tests/"
echo ""
echo "📊 For interactive analysis:"
echo "  jupyter lab"
