# IGLOO Environment Setup Guide

This guide helps you set up the IGLOO (Integrated Gradients for LOcational sensitivity in prOteins) environment using various methods.

## Quick Start

### Option 1: Automatic Setup (Recommended)

```bash
# Make the setup script executable
chmod +x setup_environment.sh

# Run the automatic setup
./setup_environment.sh

# For development setup
./setup_environment.sh --dev

# For CPU-only setup
./setup_environment.sh --pip
```

### Option 2: Manual Setup with Conda (Recommended for GPU)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate igloo

# For development dependencies
pip install -r requirements_dev.txt
```

### Option 3: Manual Setup with pip

```bash
# Create virtual environment
python3 -m venv venv_igloo
source venv_igloo/bin/activate  # On Windows: venv_igloo\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Environment Options

### GPU vs CPU

- **GPU Environment** (recommended if you have CUDA): Use `environment.yml`
- **CPU-Only Environment**: Use `environment-cpu.yml` or `requirements-cpu.txt`

```bash
# For CPU-only conda setup
conda env create -f environment-cpu.yml

# For CPU-only pip setup
pip install -r requirements-cpu.txt
```

### Development vs Production

- **Development**: Includes testing, linting, and documentation tools
- **Production**: Only runtime dependencies

```bash
# Development setup
./setup_environment.sh --dev

# Or manually add dev dependencies
pip install -r requirements_dev.txt
```

## Docker Setup

### Development with Docker

```bash
# Build and start development environment
docker-compose up igloo-dev

# Access Jupyter Lab at http://localhost:8888
```

### Production with Docker

```bash
# Build and run production environment
docker-compose --profile production up igloo-prod
```

### Testing with Docker

```bash
# Run tests in Docker
docker-compose --profile testing run igloo-test
```

## Verification

After setup, verify your installation:

```bash
# Test basic imports
python -c "import igloo; print(igloo.__version__)"

# Run tests
pytest tests/

# Test GPU availability (if using GPU environment)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test ESM model loading
python -c "import esm; print('ESM model can be loaded')"
```

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvidia-smi

   # Use CPU-only environment if GPU issues persist
   conda env create -f environment-cpu.yml
   ```

2. **Memory Issues with ESM Models**
   ```bash
   # Use smaller models or increase swap space
   # ESM-2 models require significant RAM/VRAM
   ```

3. **Package Conflicts**
   ```bash
   # Clean install
   conda env remove -n igloo
   ./setup_environment.sh --force
   ```

### System Requirements

- **Minimum**: Python 3.9+, 8GB RAM
- **Recommended**: Python 3.10+, 16GB RAM, CUDA-compatible GPU
- **Development**: 32GB RAM recommended for large protein models

### Platform-Specific Notes

#### Linux
- Full support for all features
- CUDA support available

#### macOS
- CPU-only PyTorch (MPS support in development)
- Some molecular visualization tools may have limitations

#### Windows
- WSL2 recommended for best compatibility
- Native Windows support available but less tested

## Package Structure

```
requirements.txt          # Core runtime dependencies
requirements-cpu.txt       # CPU-only dependencies
requirements_dev.txt       # Development dependencies
environment.yml           # Conda environment (GPU)
environment-cpu.yml       # Conda environment (CPU)
setup_environment.sh      # Automated setup script
pyproject.toml            # Package configuration
Dockerfile                # Docker container definition
docker-compose.yml        # Docker orchestration
```

## Next Steps

After successful setup:

1. **Explore Examples**
   ```bash
   python scripts/demo_analysis_methods.py
   ```

2. **Run Analysis**
   ```bash
   python scripts/PafA_linear_IG.py
   python scripts/betalactamase_linear_IG.py
   ```

3. **Interactive Analysis**
   ```bash
   jupyter lab
   # Open notebooks in the repository
   ```

4. **3D Visualization**
   ```bash
   python scripts/demo_3d_structure_visualization.py
   ```

## Support

If you encounter issues:

1. Check this guide first
2. Run `./setup_environment.sh --help` for options
3. Check the main repository README
4. Open an issue on GitHub with your system details and error messages

## Contributing

For development setup:

```bash
./setup_environment.sh --dev
pre-commit install  # Install git hooks
pytest tests/       # Run tests
```
