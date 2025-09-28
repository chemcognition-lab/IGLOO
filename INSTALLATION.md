# 🧊 IGLOO Installation Guide

**IGLOO (Integrated Gradients for LOcational sensitivity in prOteins)** - A comprehensive ML pipeline for analyzing structural contributions to performance in protein variants.

## 🚀 Quick Start

### Option 1: One-Command Setup (Recommended)
```bash
# Clone and setup in one go
git clone https://github.com/ranaabarghout/IGLOO.git && cd IGLOO && ./quick_setup.sh
```

### Option 2: Platform-Specific Setup
```bash
# macOS
./setup_macos.sh

# Linux
./setup_linux.sh

# Windows
setup_windows.bat
```

### Option 3: Using Make (After cloning)
```bash
make dev-setup    # Complete development setup
make prod-setup   # Production setup
```

---

## 📋 System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.9+ (Python 3.10 recommended)
- **Memory**: 8GB RAM minimum, 16GB+ recommended
- **Storage**: 5GB free space (10GB+ for datasets and results)
- **CPU**: Multi-core processor (4+ cores recommended)

### Recommended Requirements
- **Memory**: 32GB+ RAM for large protein datasets
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080, A100, etc.)
- **CUDA**: 11.8+ or 12.1+ for GPU acceleration
- **Storage**: SSD with 20GB+ free space

### GPU Support
- **NVIDIA GPUs**: Full support with CUDA 11.8/12.1
- **Apple Silicon (M1/M2)**: Native support with Metal Performance Shaders
- **AMD GPUs**: Limited support via ROCm (experimental)
- **Intel GPUs**: CPU fallback mode

---

## 🛠️ Installation Methods

### Method 1: Conda (Recommended)

#### Prerequisites
```bash
# Install Miniconda (if not already installed)
# Linux/macOS:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# macOS Apple Silicon:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

#### Installation
```bash
# Clone repository
git clone https://github.com/ranaabarghout/IGLOO.git
cd IGLOO

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate igloo

# Install development dependencies (optional)
pip install -r requirements_dev.txt

# Install IGLOO package
pip install -e .
```

### Method 2: Python Virtual Environment

#### Prerequisites
```bash
# Ensure Python 3.10 is installed
python3.10 --version

# Install venv if not available
sudo apt-get install python3.10-venv  # Ubuntu/Debian
```

#### Installation
```bash
# Clone repository
git clone https://github.com/ranaabarghout/IGLOO.git
cd IGLOO

# Create virtual environment
python3.10 -m venv venv_igloo

# Activate virtual environment
source venv_igloo/bin/activate  # Linux/macOS
# venv_igloo\Scripts\activate.bat  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements_dev.txt

# Install IGLOO package
pip install -e .
```

### Method 3: Docker

#### Prerequisites
```bash
# Install Docker
# Linux: https://docs.docker.com/engine/install/
# macOS: https://docs.docker.com/desktop/mac/install/
# Windows: https://docs.docker.com/desktop/windows/install/

# Install Docker Compose
pip install docker-compose
```

#### Installation
```bash
# Clone repository
git clone https://github.com/ranaabarghout/IGLOO.git
cd IGLOO

# Build and start development environment
docker-compose up igloo-dev

# Or build custom image
docker build -t igloo:latest .
docker run -it --gpus all -p 8888:8888 igloo:latest
```

---

## ⚙️ Configuration

### Environment Setup
```bash
# Copy environment template
cp .env.template .env

# Edit configuration (required)
nano .env  # or vim .env, code .env
```

### Key Configuration Options
```bash
# Data directories
IGLOO_DATA_DIR=/path/to/your/data
IGLOO_RESULTS_DIR=/path/to/results

# Compute settings
CUDA_VISIBLE_DEVICES=0,1,2,3  # GPU devices
OMP_NUM_THREADS=8             # CPU threads

# Model settings
ESM_MODEL_NAME=esm2_t33_650M_UR50D
DEFAULT_BATCH_SIZE=32
```

---

## 🧪 Verification

### Basic Tests
```bash
# Activate environment
conda activate igloo  # or source venv_igloo/bin/activate

# Run basic tests
make test-fast

# Run full test suite
make test

# Check installation
make health-check
```

### Manual Verification
```python
# Test core functionality
python -c "
import igloo
from igloo.utils import create_visualizations_and_exports
from igloo.plot import plot_attributions_on_3d_structure
print('✅ IGLOO installed successfully!')
print(f'Version: {igloo.__version__}')
"
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 2. Memory Issues
```bash
# Reduce batch size in configuration
export DEFAULT_BATCH_SIZE=16
export DEFAULT_INTERNAL_BATCH_SIZE=4
```

#### 3. Import Errors
```bash
# Check Python path
export PYTHONPATH="/path/to/IGLOO/src:$PYTHONPATH"

# Reinstall in development mode
pip install -e .
```

---

## 📚 Next Steps

### After Installation
1. **Verify Installation**: `make health-check`
2. **Run Demo**: `make demo`
3. **Start Analysis**: `make analysis`
4. **Open Jupyter**: `make jupyter`

---

**Need help?** Check our troubleshooting guide or open an issue on GitHub.
