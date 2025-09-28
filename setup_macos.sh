#!/bin/bash

# macOS-specific setup for IGLOO
# This script handles macOS-specific dependencies and optimizations

set -e

echo "🍎 Setting up IGLOO on macOS..."
echo "==============================="

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only!"
    exit 1
fi

# Check for Xcode command line tools
if ! xcode-select -p &>/dev/null; then
    echo "📱 Installing Xcode command line tools..."
    xcode-select --install
    echo "Please complete the Xcode installation and run this script again."
    exit 1
fi

# Check for Homebrew
if ! command -v brew &>/dev/null; then
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH for M1/M2 Macs
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
brew update
brew install python@3.10 git cmake pkg-config libomp

# For M1/M2 Macs, set up OpenMP
if [[ $(uname -m) == "arm64" ]]; then
    echo "🔧 Setting up OpenMP for Apple Silicon..."
    export LDFLAGS="-L/opt/homebrew/lib"
    export CPPFLAGS="-I/opt/homebrew/include"
    export CC="clang"
    export CXX="clang++"
fi

# Install conda if not present
if ! command -v conda &>/dev/null; then
    echo "🐍 Installing Miniconda..."
    if [[ $(uname -m) == "arm64" ]]; then
        curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh
    else
        curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o miniconda.sh
    fi

    bash miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init zsh
    rm miniconda.sh

    echo "🔄 Please restart your terminal and run this script again."
    exit 0
fi

# Set up conda-forge and pytorch channels
conda config --add channels conda-forge
conda config --add channels pytorch

# Create environment
echo "🧊 Creating IGLOO conda environment..."
if conda env list | grep -q "igloo"; then
    echo "⚠️  Environment 'igloo' already exists. Removing..."
    conda env remove -n igloo -y
fi

# Create optimized environment for macOS
cat > environment_macos.yml << EOF
name: igloo
channels:
  - conda-forge
  - pytorch
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - seaborn
  - plotly
  - jupyter
  - jupyterlab
  - pytest
  - tqdm
  - pyyaml
  - requests
  - biopython
  - pytorch::pytorch
  - pytorch::torchvision
  - pip
  - pip:
    - fair-esm
    - transformers
    - safetensors
    - captum
    - py3Dmol
    - xgboost
EOF

conda env create -f environment_macos.yml

# Activate environment and install package
conda activate igloo
pip install -e .

# Set up PyMOL (optional)
read -p "Install PyMOL for 3D visualization? (y/N): " install_pymol
if [[ $install_pymol =~ ^[Yy]$ ]]; then
    echo "🧬 Installing PyMOL..."
    brew install pymol
fi

# macOS-specific optimizations
echo "⚡ Setting up macOS optimizations..."

# Create macOS-specific environment variables
cat >> ~/.zshrc << 'EOF'

# IGLOO macOS optimizations
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8

# PyTorch optimizations for macOS
export PYTORCH_ENABLE_MPS_FALLBACK=1

# IGLOO paths
export IGLOO_ROOT="$(pwd)"
export PYTHONPATH="$IGLOO_ROOT/src:$PYTHONPATH"
EOF

echo "✅ macOS setup complete!"
echo ""
echo "🎉 Next steps:"
echo "1. Restart your terminal or run: source ~/.zshrc"
echo "2. Activate environment: conda activate igloo"
echo "3. Copy environment template: cp .env.template .env"
echo "4. Run tests: make test"
echo "5. Start analysis: make analysis"
echo ""
echo "💡 macOS-specific notes:"
echo "- PyTorch will use Metal Performance Shaders (MPS) if available"
echo "- OpenMP is configured for optimal performance"
echo "- Use 'make docker-dev' for containerized development"

# Clean up
rm -f environment_macos.yml
