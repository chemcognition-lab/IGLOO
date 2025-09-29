#!/bin/bash

# IGLOO Quick Setup Script
# This script provides a streamlined setup for the IGLOO project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_system() {
    print_status "Checking system requirements..."

    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_success "Linux detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_success "macOS detected"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        print_success "Windows detected"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi

    # Check available memory
    if command -v free >/dev/null 2>&1; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -lt 8 ]; then
            print_warning "Low memory detected (${MEMORY_GB}GB). IGLOO requires at least 8GB RAM."
        else
            print_success "Memory check passed (${MEMORY_GB}GB available)"
        fi
    fi

    # Check for CUDA
    if command -v nvidia-smi >/dev/null 2>&1; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    else
        print_warning "No NVIDIA GPU detected. Will use CPU-only mode."
    fi
}

# Setup function selector
main() {
    echo "🧊 IGLOO Quick Setup"
    echo "===================="

    check_system

    echo ""
    echo "Choose your setup method:"
    echo "1) Conda (Recommended)"
    echo "2) Python Virtual Environment"
    echo "3) Docker"
    echo "4) System-wide installation (not recommended)"
    echo "5) Check existing installation"

    read -p "Enter your choice (1-5): " choice

    case $choice in
        1)
            setup_conda
            ;;
        2)
            setup_venv
            ;;
        3)
            setup_docker
            ;;
        4)
            setup_system
            ;;
        5)
            check_installation
            ;;
        *)
            print_error "Invalid choice. Please run the script again."
            exit 1
            ;;
    esac
}

# Conda setup
setup_conda() {
    print_status "Setting up with Conda..."

    if ! command -v conda >/dev/null 2>&1; then
        print_error "Conda not found. Please install Miniconda or Anaconda first."
        echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    # Ask for development mode
    read -p "Install development dependencies? (y/N): " dev_mode

    # Run the main setup script
    if [[ $dev_mode =~ ^[Yy]$ ]]; then
        ./setup_environment.sh --conda --dev
    else
        ./setup_environment.sh --conda
    fi

    print_success "Conda setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the environment: conda activate igloo"
    echo "2. Copy and configure environment: cp .env.template .env"
    echo "3. Run tests: pytest tests/"
    echo "4. Start analysis: python scripts/run_all_ig_analyses.py"
}

# Virtual environment setup
setup_venv() {
    print_status "Setting up Python virtual environment..."

    if ! command -v python3 >/dev/null 2>&1; then
        print_error "Python 3 not found. Please install Python 3.9+ first."
        exit 1
    fi

    # Check Python version
    python_version=$(python3 --version | awk '{print $2}')
    python_major=$(echo $python_version | cut -d. -f1)
    python_minor=$(echo $python_version | cut -d. -f2)

    if [ "$python_major" -eq 3 ] && [ "$python_minor" -ge 9 ]; then
        print_success "Python $python_version is compatible"
    else
        print_error "Python $python_version found. Requires Python 3.9+"
        exit 1
    fi

    # Ask for development mode
    read -p "Install development dependencies? (y/N): " dev_mode

    # Run the main setup script
    if [[ $dev_mode =~ ^[Yy]$ ]]; then
        ./setup_environment.sh --pip --dev
    else
        ./setup_environment.sh --pip
    fi

    print_success "Virtual environment setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the environment: source venv_igloo/bin/activate"
    echo "2. Copy and configure environment: cp .env.template .env"
    echo "3. Run tests: pytest tests/"
    echo "4. Start analysis: python scripts/run_all_ig_analyses.py"
}

# Docker setup
setup_docker() {
    print_status "Setting up with Docker..."

    if ! command -v docker >/dev/null 2>&1; then
        print_error "Docker not found. Please install Docker first."
        echo "Download from: https://www.docker.com/get-started"
        exit 1
    fi

    if ! command -v docker-compose >/dev/null 2>&1; then
        print_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi

    print_status "Building Docker images..."
    docker-compose build igloo-dev

    print_status "Setting up volumes..."
    docker-compose up -d --no-start

    print_success "Docker setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Start development environment: docker-compose up igloo-dev"
    echo "2. Access Jupyter at: http://localhost:8888"
    echo "3. Run tests: docker-compose run igloo-test"
    echo "4. For production: docker-compose --profile production up igloo-prod"
}

# System-wide installation
setup_system() {
    print_warning "System-wide installation is not recommended!"
    read -p "Are you sure you want to continue? (y/N): " confirm

    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        print_status "Aborting system-wide installation."
        exit 0
    fi

    print_status "Installing system-wide..."

    # Install base requirements
    pip install --user -r requirements.txt

    # Ask for development mode
    read -p "Install development dependencies? (y/N): " dev_mode
    if [[ $dev_mode =~ ^[Yy]$ ]]; then
        pip install --user -r requirements_dev.txt
    fi

    # Install package
    pip install --user -e .

    print_success "System-wide installation complete!"
    print_warning "Make sure ~/.local/bin is in your PATH"
}

# Check existing installation
check_installation() {
    print_status "Checking existing IGLOO installation..."

    # Check if igloo package is installed
    if python -c "import igloo; print(f'IGLOO version: {igloo.__version__}')" 2>/dev/null; then
        print_success "IGLOO package is installed"
    else
        print_error "IGLOO package not found"
        return
    fi

    # Check key dependencies
    dependencies=("numpy" "pandas" "torch" "transformers" "sklearn" "matplotlib" "seaborn")

    for dep in "${dependencies[@]}"; do
        if python -c "import $dep" 2>/dev/null; then
            version=$(python -c "import $dep; print(getattr($dep, '__version__', 'unknown'))" 2>/dev/null)
            print_success "$dep ($version)"
        else
            print_error "$dep - not installed"
        fi
    done

    # Check CUDA availability
    if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
        cuda_version=$(python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null)
        print_success "$cuda_version"
    else
        print_warning "CUDA not available or PyTorch not installed"
    fi

    # Check data directory
    if [ -d "data" ]; then
        data_files=$(find data -name "*.csv" -o -name "*.safetensors" | wc -l)
        print_success "Data directory found with $data_files files"
    else
        print_warning "Data directory not found"
    fi

    # Run basic tests
    print_status "Running basic functionality tests..."
    if python -c "
import igloo
from igloo.utils import create_visualizations_and_exports
from igloo.plot import plot_attributions_on_3d_structure
print('✅ All core functions importable')
" 2>/dev/null; then
        print_success "Core functionality test passed"
    else
        print_error "Core functionality test failed"
    fi
}

# Run main function
main "$@"
