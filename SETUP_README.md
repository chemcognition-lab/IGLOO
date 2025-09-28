# 🧊 IGLOO Environment Setup Files

This directory contains comprehensive environment setup files and utilities for the IGLOO project. Choose the method that best fits your needs and platform.

## 📁 Files Overview

### Setup Scripts
- **`quick_setup.sh`** - Interactive setup wizard (RECOMMENDED)
- **`setup_environment.sh`** - Original automated setup script
- **`setup_macos.sh`** - macOS-specific setup with optimizations
- **`setup_linux.sh`** - Linux-specific setup with GPU support
- **`setup_windows.bat`** - Windows setup script

### Configuration Files
- **`.env.template`** - Environment configuration template
- **`environment.yml`** - Conda environment specification
- **`requirements.txt`** - Python package dependencies
- **`requirements_dev.txt`** - Development dependencies
- **`pyproject.toml`** - Project configuration and dependencies

### Docker Files
- **`Dockerfile`** - Multi-stage Docker image
- **`docker-compose.yml`** - Docker Compose orchestration

### Utilities
- **`health_check.py`** - Comprehensive environment health check
- **`Makefile`** - Task automation and workflow management
- **`INSTALLATION.md`** - Detailed installation guide

---

## 🚀 Quick Start Guide

### 1. Clone Repository
```bash
git clone https://github.com/ranaabarghout/IGLOO.git
cd IGLOO
```

### 2. Choose Your Setup Method

#### Option A: Interactive Setup (Recommended)
```bash
./quick_setup.sh
```
This wizard will:
- Detect your platform automatically
- Guide you through setup options
- Handle platform-specific optimizations
- Verify the installation

#### Option B: Platform-Specific Setup
```bash
# macOS
./setup_macos.sh

# Linux (Ubuntu/CentOS/RHEL)
./setup_linux.sh

# Windows
setup_windows.bat
```

#### Option C: Make-based Setup
```bash
# Development setup
make dev-setup

# Production setup
make prod-setup

# Docker setup
make docker-dev
```

#### Option D: Manual Setup
```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate igloo
pip install -e .

# Or pip virtual environment
python3 -m venv venv_igloo
source venv_igloo/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 3. Configure Environment
```bash
# Copy and edit configuration
cp .env.template .env
# Edit .env with your settings
```

### 4. Verify Installation
```bash
# Quick health check
./health_check.py

# Or using make
make health-check

# Run tests
make test
```

---

## 🔧 Platform-Specific Features

### macOS Setup (`setup_macos.sh`)
- ✅ Apple Silicon (M1/M2) optimizations
- ✅ Metal Performance Shaders support
- ✅ Homebrew dependency management
- ✅ OpenMP configuration
- ✅ PyMOL integration

### Linux Setup (`setup_linux.sh`)
- ✅ Multi-distribution support (Ubuntu/CentOS/RHEL)
- ✅ CUDA detection and installation
- ✅ GPU optimization
- ✅ Multi-core CPU utilization
- ✅ Optional Jupyter systemd service

### Windows Setup (`setup_windows.bat`)
- ✅ Visual C++ Build Tools detection
- ✅ Conda environment creation
- ✅ Batch file shortcuts
- ✅ WSL compatibility notes

---

## 🐳 Docker Support

### Development Container
```bash
# Start development environment with Jupyter
docker-compose up igloo-dev

# Access Jupyter at http://localhost:8888
```

### Production Container
```bash
# Run production workload
docker-compose --profile production up igloo-prod
```

### Testing Container
```bash
# Run tests in isolated environment
docker-compose run igloo-test
```

---

## 🔍 Health Check System

The `health_check.py` script provides comprehensive environment validation:

```bash
./health_check.py
```

**Checks performed:**
- ✅ Python version compatibility
- ✅ Package dependencies
- ✅ GPU/CUDA availability
- ✅ IGLOO package installation
- ✅ Data file availability
- ✅ Environment configuration
- ✅ System resources
- ✅ Basic functionality tests

**Sample output:**
```
🧊 IGLOO Environment Health Check
==================================

Python Environment
==================
✅ Python version is compatible (3.10+)

Package Dependencies
===================
✅ PyTorch: 2.1.0
✅ Transformers: 4.21.0
...

Overall Health Score: 95.0% (19/20)
✅ Environment is ready for analysis!
```

---

## 📋 Make Commands

The enhanced Makefile provides 50+ commands for common tasks:

```bash
# Setup and installation
make dev-setup          # Complete development setup
make install-conda       # Setup conda environment
make setup-env          # Create .env from template

# Testing and quality
make test               # Run full test suite
make test-fast          # Run fast tests only
make lint               # Code linting
make format             # Code formatting

# Analysis and experiments
make analysis           # Run full analysis pipeline
make analysis-pafa      # Run PafA analysis
make jupyter           # Start Jupyter Lab

# Docker operations
make docker-dev         # Development container
make docker-prod        # Production container

# Documentation and build
make docs              # Build documentation
make build             # Build package
make clean-all         # Deep clean

# Utilities
make health-check      # Environment validation
make help              # Show all commands
```

---

## ⚙️ Environment Configuration

### Required Settings (`.env`)
```bash
# Data directories
IGLOO_DATA_DIR=/path/to/data
IGLOO_RESULTS_DIR=/path/to/results

# Compute settings
CUDA_VISIBLE_DEVICES=0,1,2,3
OMP_NUM_THREADS=8

# Model configuration
ESM_MODEL_NAME=esm2_t33_650M_UR50D
DEFAULT_BATCH_SIZE=32
```

### Optional Settings
```bash
# Development mode
IGLOO_DEBUG=true
IGLOO_VERBOSE=true

# Jupyter configuration
JUPYTER_PORT=8888
JUPYTER_TOKEN=your-token

# Cloud integration
AWS_S3_BUCKET=your-bucket
WANDB_PROJECT=igloo-experiments
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Permission Errors
```bash
chmod +x *.sh
chmod +x health_check.py
```

#### 2. Python Version Issues
```bash
# Check version
python --version

# Use specific version
python3.10 -m venv venv_igloo
```

#### 3. CUDA Issues
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Memory Issues
```bash
# Reduce batch size
export DEFAULT_BATCH_SIZE=16
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 5. Package Conflicts
```bash
# Clean reinstall
make clean-all
conda env remove -n igloo
./quick_setup.sh
```

### Platform-Specific Issues

#### macOS
- Install Xcode command line tools: `xcode-select --install`
- For M1/M2: Use native ARM conda installation
- OpenMP issues: `brew install libomp`

#### Linux
- Missing build tools: `sudo apt install build-essential`
- CUDA path issues: Check `LD_LIBRARY_PATH`
- Permission issues: Use `sudo` for system packages

#### Windows
- Install Visual C++ Build Tools
- Use PowerShell as Administrator
- Consider WSL2 for better compatibility

---

## 📚 Additional Resources

- **Installation Guide**: `INSTALLATION.md`
- **API Documentation**: `docs/`
- **Examples**: `examples/`
- **GitHub Issues**: Report problems
- **Discussions**: Ask questions

---

## 🤝 Contributing

To set up a development environment:

```bash
# Clone and setup
git clone https://github.com/ranaabarghout/IGLOO.git
cd IGLOO

# Development setup
make dev-setup

# Install pre-commit hooks
make install-pre-commit

# Run tests
make test

# Check code quality
make check-all
```

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

**Need help?**
- 📖 Check the [Installation Guide](INSTALLATION.md)
- 🐛 [Report Issues](https://github.com/ranaabarghout/IGLOO/issues)
- 💬 [Ask Questions](https://github.com/ranaabarghout/IGLOO/discussions)
- 📧 [Contact](mailto:ranamoneim@gmail.com)
