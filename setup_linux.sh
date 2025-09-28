#!/bin/bash

# Linux-specific setup for IGLOO
# Optimized for Ubuntu/Debian and CentOS/RHEL systems

set -e

echo "🐧 Setting up IGLOO on Linux..."
echo "==============================="

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
    VERSION=$VERSION_ID
else
    echo "❌ Cannot detect Linux distribution"
    exit 1
fi

echo "📋 Detected: $PRETTY_NAME"

# Function to install system dependencies
install_system_deps() {
    case $DISTRO in
        ubuntu|debian)
            echo "📦 Installing system dependencies (Ubuntu/Debian)..."
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                python3.10 \
                python3.10-dev \
                python3.10-venv \
                python3-pip \
                git \
                curl \
                wget \
                cmake \
                pkg-config \
                libblas-dev \
                liblapack-dev \
                libhdf5-dev \
                libssl-dev \
                libffi-dev \
                nodejs \
                npm
            ;;
        centos|rhel|fedora)
            echo "📦 Installing system dependencies (CentOS/RHEL/Fedora)..."
            if command -v dnf &>/dev/null; then
                PKG_MGR="dnf"
            else
                PKG_MGR="yum"
            fi

            sudo $PKG_MGR update -y
            sudo $PKG_MGR install -y \
                gcc \
                gcc-c++ \
                python3.10 \
                python3.10-devel \
                python3-pip \
                git \
                curl \
                wget \
                cmake \
                pkgconfig \
                blas-devel \
                lapack-devel \
                hdf5-devel \
                openssl-devel \
                libffi-devel \
                nodejs \
                npm
            ;;
        *)
            echo "⚠️  Unsupported distribution: $DISTRO"
            echo "Please install the following manually:"
            echo "- Python 3.10+"
            echo "- build-essential/development tools"
            echo "- cmake, git, curl"
            echo "- BLAS/LAPACK libraries"
            ;;
    esac
}

# Install CUDA (optional)
install_cuda() {
    read -p "Install NVIDIA CUDA for GPU acceleration? (y/N): " install_cuda_opt
    if [[ $install_cuda_opt =~ ^[Yy]$ ]]; then
        echo "🚀 Installing CUDA..."

        # Check if GPU is present
        if ! command -v nvidia-smi &>/dev/null; then
            echo "⚠️  No NVIDIA GPU detected or drivers not installed"
            echo "Please install NVIDIA drivers first"
            return
        fi

        case $DISTRO in
            ubuntu)
                # Add NVIDIA repository
                wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
                sudo dpkg -i cuda-keyring_1.0-1_all.deb
                sudo apt-get update
                sudo apt-get install -y cuda-toolkit-12-1
                rm cuda-keyring_1.0-1_all.deb
                ;;
            centos|rhel)
                # Add NVIDIA repository
                sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
                sudo yum install -y cuda-toolkit-12-1
                ;;
            *)
                echo "Please install CUDA manually for your distribution"
                ;;
        esac

        # Add CUDA to PATH
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi
}

# Install system dependencies
install_system_deps

# Install CUDA if requested
install_cuda

# Install conda if not present
if ! command -v conda &>/dev/null; then
    echo "🐍 Installing Miniconda..."

    # Detect architecture
    ARCH=$(uname -m)
    if [[ $ARCH == "x86_64" ]]; then
        CONDA_ARCH="x86_64"
    elif [[ $ARCH == "aarch64" ]]; then
        CONDA_ARCH="aarch64"
    else
        echo "❌ Unsupported architecture: $ARCH"
        exit 1
    fi

    curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${CONDA_ARCH}.sh -o miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    rm miniconda.sh

    echo "🔄 Please restart your terminal and run this script again."
    exit 0
fi

# Create optimized environment for Linux
echo "🧊 Creating IGLOO conda environment..."
if conda env list | grep -q "igloo"; then
    echo "⚠️  Environment 'igloo' already exists. Removing..."
    conda env remove -n igloo -y
fi

# Check for CUDA availability
HAS_CUDA=false
if command -v nvidia-smi &>/dev/null; then
    HAS_CUDA=true
    echo "🚀 CUDA detected - installing GPU-enabled packages"
else
    echo "💻 No CUDA detected - installing CPU-only packages"
fi

# Create environment file
if [ "$HAS_CUDA" = true ]; then
    cat > environment_linux_gpu.yml << EOF
name: igloo
channels:
  - conda-forge
  - pytorch
  - nvidia
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
  - pytorch::pytorch-cuda=12.1
  - pip
  - pip:
    - fair-esm
    - transformers
    - safetensors
    - captum
    - py3Dmol
    - xgboost
EOF
    conda env create -f environment_linux_gpu.yml
    rm environment_linux_gpu.yml
else
    cat > environment_linux_cpu.yml << EOF
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
  - pytorch::cpuonly
  - pip
  - pip:
    - fair-esm
    - transformers
    - safetensors
    - captum
    - py3Dmol
    - xgboost
EOF
    conda env create -f environment_linux_cpu.yml
    rm environment_linux_cpu.yml
fi

# Activate environment and install package
conda activate igloo
pip install -e .

# Install PyMOL (optional)
read -p "Install PyMOL for 3D visualization? (y/N): " install_pymol
if [[ $install_pymol =~ ^[Yy]$ ]]; then
    echo "🧬 Installing PyMOL..."
    case $DISTRO in
        ubuntu|debian)
            sudo apt-get install -y pymol
            ;;
        *)
            pip install pymol-open-source
            ;;
    esac
fi

# Linux-specific optimizations
echo "⚡ Setting up Linux optimizations..."

# Detect CPU cores
CPU_CORES=$(nproc)
echo "💻 Detected $CPU_CORES CPU cores"

# Set up environment variables
cat >> ~/.bashrc << EOF

# IGLOO Linux optimizations
export OMP_NUM_THREADS=$CPU_CORES
export MKL_NUM_THREADS=$CPU_CORES
export NUMEXPR_NUM_THREADS=$CPU_CORES

# Memory optimizations
export MALLOC_ARENA_MAX=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# IGLOO paths
export IGLOO_ROOT="$(pwd)"
export PYTHONPATH="\$IGLOO_ROOT/src:\$PYTHONPATH"
EOF

# Set up systemd service for Jupyter (optional)
read -p "Set up Jupyter as a system service? (y/N): " setup_service
if [[ $setup_service =~ ^[Yy]$ ]]; then
    echo "🚀 Setting up Jupyter service..."

    sudo tee /etc/systemd/system/igloo-jupyter.service > /dev/null << EOF
[Unit]
Description=IGLOO Jupyter Lab
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$HOME/miniconda3/envs/igloo/bin:$PATH
ExecStart=$HOME/miniconda3/envs/igloo/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
Restart=always

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable igloo-jupyter

    echo "✅ Jupyter service created. Start with: sudo systemctl start igloo-jupyter"
fi

echo "✅ Linux setup complete!"
echo ""
echo "🎉 Next steps:"
echo "1. Restart your terminal or run: source ~/.bashrc"
echo "2. Activate environment: conda activate igloo"
echo "3. Copy environment template: cp .env.template .env"
echo "4. Run tests: make test"
echo "5. Start analysis: make analysis"
echo ""
if [ "$HAS_CUDA" = true ]; then
    echo "🚀 GPU acceleration enabled!"
    echo "- PyTorch will use CUDA for faster computation"
    echo "- Monitor GPU usage with: nvidia-smi"
fi
echo ""
echo "💡 Linux-specific features:"
echo "- Optimized for multi-core CPU usage"
echo "- Memory settings tuned for large datasets"
echo "- Optional Jupyter systemd service available"
