#!/usr/bin/env python3
"""
IGLOO Environment Health Check Script
=====================================

This script performs a comprehensive health check of the IGLOO environment,
checking dependencies, GPU availability, data integrity, and more.
"""

import importlib
import os
import platform
import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text: str) -> None:
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text: str) -> None:
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text: str) -> None:
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text: str) -> None:
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text: str) -> None:
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

def check_python_version() -> bool:
    """Check Python version compatibility"""
    print_header("Python Environment")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_info(f"Python version: {version_str}")
    print_info(f"Python executable: {sys.executable}")
    print_info(f"Platform: {platform.platform()}")

    if version.major == 3 and version.minor >= 9:
        print_success("Python version is compatible (3.9+)")
        return True
    else:
        print_error(f"Python {version_str} is not supported. Requires Python 3.9+")
        return False

def check_package_installation(packages: dict[str, str]) -> dict[str, bool]:
    """Check if required packages are installed"""
    print_header("Package Dependencies")

    results = {}
    for package, description in packages.items():
        try:
            if package == "torch":
                import torch
                version = torch.__version__
                print_success(f"{description}: {version}")
                results[package] = True
            elif package == "transformers":
                import transformers
                version = transformers.__version__
                print_success(f"{description}: {version}")
                results[package] = True
            elif package == "sklearn":
                import sklearn
                version = sklearn.__version__
                print_success(f"{description}: {version}")
                results[package] = True
            elif package == "esm.models.esmc":
                import esm.models.esmc  # noqa: F401 – missing in esm<3.1
                from importlib.metadata import version as dist_version
                version = dist_version("esm")
                print_success(f"{description}: {version}")
                results[package] = True
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                print_success(f"{description}: {version}")
                results[package] = True
        except ImportError:
            print_error(f"{description}: Not installed")
            results[package] = False
        except Exception as e:
            print_warning(f"{description}: Error - {str(e)}")
            results[package] = False

    return results

def check_gpu_availability() -> dict[str, any]:
    """Check GPU and CUDA availability"""
    print_header("GPU & Compute Environment")

    gpu_info = {
        'cuda_available': False,
        'mps_available': False,
        'gpu_count': 0,
        'gpu_names': []
    }

    try:
        import torch

        # Check CUDA
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['gpu_count'] = torch.cuda.device_count()

            print_success(f"CUDA available: {torch.version.cuda}")
            print_success(f"GPU count: {gpu_info['gpu_count']}")

            for i in range(gpu_info['gpu_count']):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_info['gpu_names'].append(gpu_name)

                # Get GPU memory info
                gpu_props = torch.cuda.get_device_properties(i)
                total_memory = gpu_props.total_memory / (1024**3)  # GB
                print_success(f"GPU {i}: {gpu_name} ({total_memory:.1f} GB)")

        # Check Metal Performance Shaders (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info['mps_available'] = True
            print_success("Metal Performance Shaders (MPS) available")

        if not gpu_info['cuda_available'] and not gpu_info['mps_available']:
            print_warning("No GPU acceleration available - will use CPU")

    except ImportError:
        print_error("PyTorch not installed - cannot check GPU availability")

    return gpu_info

def check_igloo_installation() -> bool:
    """Check IGLOO package installation"""
    print_header("IGLOO Package")

    try:
        import igloo
        version = getattr(igloo, '__version__', 'development')
        print_success(f"IGLOO package installed: {version}")

        # Check core modules
        core_modules = [
            ('igloo.utils', 'Core utilities'),
            ('igloo.plot', 'Plotting functions'),
        ]

        all_modules_ok = True
        for module_name, description in core_modules:
            try:
                importlib.import_module(module_name)
                print_success(f"{description} module available")
            except ImportError as e:
                print_error(f"{description} module failed: {str(e)}")
                all_modules_ok = False

        return all_modules_ok

    except ImportError:
        print_error("IGLOO package not installed")
        print_info("Install with: pip install -e .")
        return False

def check_data_availability() -> dict[str, bool]:
    """Check for data files"""
    print_header("Data Files")

    data_dir = Path("data")
    results = {}

    if not data_dir.exists():
        print_error("Data directory not found")
        return {'data_dir': False}

    # Check for key datasets
    datasets = {
        'synthetic/GB1_additive.csv': 'Synthetic GB1 landscape',
        'synthetic/T4Lysozyme_additive.csv': 'Synthetic T4 Lysozyme landscape',
        'synthetic/p53_additive.csv': 'Synthetic p53 landscape',
        'FLIP2_processed/alpha_amylase__mutation__by_mutation_combined.csv': 'FLIP2 alpha-amylase'
    }

    for filename, description in datasets.items():
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024*1024)
            print_success(f"{description}: {size_mb:.1f} MB")
            results[filename] = True
        else:
            print_warning(f"{description}: Not found")
            results[filename] = False

    # Check embeddings directory
    embeddings_dir = data_dir / "embeddings"
    if embeddings_dir.exists():
        embedding_files = list(embeddings_dir.glob("*.safetensors"))
        print_success(f"Embedding files: {len(embedding_files)} found")
        results['embeddings'] = True
    else:
        print_warning("Embeddings directory not found")
        results['embeddings'] = False

    return results

def check_environment_config() -> dict[str, bool]:
    """Check environment configuration"""
    print_header("Environment Configuration")

    results = {}

    # Check for .env file
    if Path(".env").exists():
        print_success("Environment file (.env) found")
        results['env_file'] = True
    else:
        print_warning("No .env file found")
        print_info("Copy .env.template to .env and configure")
        results['env_file'] = False

    # Check key environment variables
    env_vars = {
        'PYTHONPATH': 'Python module search path',
        'CUDA_VISIBLE_DEVICES': 'GPU device selection',
        'OMP_NUM_THREADS': 'OpenMP thread count'
    }

    for var_name, description in env_vars.items():
        value = os.environ.get(var_name)
        if value:
            print_success(f"{description}: {value}")
            results[var_name] = True
        else:
            print_info(f"{description}: Not set (using defaults)")
            results[var_name] = False

    return results

def run_basic_functionality_test() -> bool:
    """Test basic IGLOO functionality"""
    print_header("Functionality Test")

    try:
        # Test imports
        print_info("Testing core imports...")
        from igloo.plot import plot_attributions_on_3d_structure
        from igloo.utils import create_visualizations_and_exports
        print_success("Core function imports successful")

        # Test basic PyTorch operations
        print_info("Testing PyTorch functionality...")
        import torch
        x = torch.randn(10, 10)
        y = torch.mm(x, x.t())
        print_success("PyTorch operations working")

        # Test ESM model availability (without loading)
        print_info("Testing model availability...")
        from transformers import EsmTokenizer
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        print_success("ESM model components accessible")

        return True

    except Exception as e:
        print_error(f"Functionality test failed: {str(e)}")
        return False

def check_system_resources() -> dict[str, any]:
    """Check system resources"""
    print_header("System Resources")

    resources = {}

    try:
        import psutil

        # Memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)

        print_info(f"Total RAM: {memory_gb:.1f} GB")
        print_info(f"Available RAM: {memory_available_gb:.1f} GB")

        if memory_gb >= 16:
            print_success("Sufficient memory for large datasets")
        elif memory_gb >= 8:
            print_warning("Limited memory - may need to reduce batch sizes")
        else:
            print_error("Low memory - may encounter issues with large models")

        resources['memory_gb'] = memory_gb

        # CPU
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        print_info(f"CPU cores: {cpu_count}")
        if cpu_freq:
            print_info(f"CPU frequency: {cpu_freq.current:.0f} MHz")

        resources['cpu_count'] = cpu_count

        # Disk space
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)

        print_info(f"Free disk space: {disk_free_gb:.1f} GB")

        if disk_free_gb >= 20:
            print_success("Sufficient disk space")
        elif disk_free_gb >= 10:
            print_warning("Limited disk space")
        else:
            print_error("Low disk space - may cause issues")

        resources['disk_free_gb'] = disk_free_gb

    except ImportError:
        print_warning("psutil not available - cannot check system resources")
        print_info("Install with: pip install psutil")

    return resources

def generate_report(results: dict) -> None:
    """Generate summary report"""
    print_header("Health Check Summary")

    # Count successes and failures
    total_checks = 0
    passed_checks = 0
    critical_failures = []

    # Core requirements
    if results.get('python_ok', False):
        passed_checks += 1
    else:
        critical_failures.append("Python version")
    total_checks += 1

    if results.get('igloo_ok', False):
        passed_checks += 1
    else:
        critical_failures.append("IGLOO package")
    total_checks += 1

    if results.get('functionality_ok', False):
        passed_checks += 1
    else:
        critical_failures.append("Basic functionality")
    total_checks += 1

    # Package dependencies
    package_results = results.get('packages', {})
    for package, status in package_results.items():
        total_checks += 1
        if status:
            passed_checks += 1
        else:
            critical_failures.append(f"Package: {package}")

    # Calculate score
    score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0

    print(f"{Colors.BOLD}Overall Health Score: {score:.1f}% ({passed_checks}/{total_checks}){Colors.END}")

    if score >= 80:
        print_success("Environment is ready for analysis!")
    elif score >= 60:
        print_warning("Environment has some issues but may work")
    else:
        print_error("Environment needs attention before use")

    if critical_failures:
        print(f"\n{Colors.BOLD}Critical Issues:{Colors.END}")
        for failure in critical_failures:
            print_error(failure)

    # Recommendations
    print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")

    if not results.get('gpu_info', {}).get('cuda_available', False) and not results.get('gpu_info', {}).get('mps_available', False):
        print_info("Consider using GPU acceleration for faster processing")

    if not results.get('env_config', {}).get('env_file', False):
        print_info("Set up environment configuration: cp .env.template .env")

    memory_gb = results.get('resources', {}).get('memory_gb', 0)
    if memory_gb < 16:
        print_info("Consider reducing batch sizes for memory-constrained systems")

    if score < 80:
        print_info("Run installation guide: see Installation in README.rst")

def main():
    """Main health check routine"""
    print(f"{Colors.BOLD}{Colors.MAGENTA}")
    print("🧊 IGLOO Environment Health Check")
    print("==================================")
    print(f"{Colors.END}")

    results = {}

    # Core checks
    results['python_ok'] = check_python_version()

    # Package dependencies
    required_packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'TQDM',
        'Bio': 'BioPython',
        'httpx': 'HTTPX',
        'esm.models.esmc': 'ESM (with ESM-C)'
    }

    results['packages'] = check_package_installation(required_packages)
    results['gpu_info'] = check_gpu_availability()
    results['igloo_ok'] = check_igloo_installation()
    results['data'] = check_data_availability()
    results['env_config'] = check_environment_config()
    results['resources'] = check_system_resources()
    results['functionality_ok'] = run_basic_functionality_test()

    # Generate summary
    generate_report(results)

    print(f"\n{Colors.BOLD}Need help?{Colors.END}")
    print("📖 Check the installation guide: Installation in README.rst")
    print("🐛 Report issues: https://github.com/ranaabarghout/IGLOO/issues")
    print("💬 Ask questions: https://github.com/ranaabarghout/IGLOO/discussions")

if __name__ == "__main__":
    main()
