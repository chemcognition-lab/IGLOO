# =============================================================================
# IGLOO Project Makefile
# Advanced workflow automation for protein analysis
# =============================================================================

.PHONY: help install install-dev install-conda clean clean-all test test-fast test-cov lint format type-check docs build release docker run jupyter notebooks setup-env check-env analysis

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
PROJECT_NAME := igloo
VENV_NAME := venv_igloo
CONDA_ENV := igloo
DOCKER_IMAGE := igloo:latest

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Helper function to print colored output
define print_success
	@echo "$(GREEN)[SUCCESS]$(NC) $(1)"
endef

define print_info
	@echo "$(BLUE)[INFO]$(NC) $(1)"
endef

define print_warning
	@echo "$(YELLOW)[WARNING]$(NC) $(1)"
endef

define print_error
	@echo "$(RED)[ERROR]$(NC) $(1)"
endef

# =============================================================================
# HELP & INFORMATION
# =============================================================================

help: ## Show this help message
	@echo "🧊 IGLOO Project Makefile"
	@echo "========================="
	@echo ""
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make install-dev    # Install for development"
	@echo "  make test           # Run tests"
	@echo "  make analysis       # Run full analysis pipeline"
	@echo "  make docker         # Build and run Docker container"
	@echo ""

check-env: ## Check if the environment is properly set up
	@$(call print_info,"Checking environment setup...")
	@if [ -f .env ]; then \
		$(call print_success,"Environment file found"); \
	else \
		$(call print_warning,"No .env file found. Copy .env.template to .env"); \
	fi
	@if $(PYTHON) -c "import $(PROJECT_NAME)" 2>/dev/null; then \
		$(call print_success,"$(PROJECT_NAME) package is installed"); \
	else \
		$(call print_error,"$(PROJECT_NAME) package not found. Run 'make install'"); \
	fi

# =============================================================================
# INSTALLATION & SETUP
# =============================================================================

install: ## Install the package for production use
	@$(call print_info,"Installing $(PROJECT_NAME) for production...")
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install .
	@$(call print_success,"Installation complete!")

install-dev: ## Install the package for development
	@$(call print_info,"Installing $(PROJECT_NAME) for development...")
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements_dev.txt
	$(PIP) install -e .
	@$(call print_success,"Development installation complete!")

install-conda: ## Install using conda environment
	@$(call print_info,"Setting up conda environment...")
	@if conda env list | grep -q "$(CONDA_ENV)"; then \
		$(call print_warning,"Conda environment '$(CONDA_ENV)' already exists"); \
	else \
		conda env create -f environment.yml; \
		$(call print_success,"Conda environment created"); \
	fi

setup-env: ## Set up environment from template
	@$(call print_info,"Setting up environment configuration...")
	@if [ ! -f .env ]; then \
		cp .env.template .env; \
		$(call print_success,"Created .env from template"); \
		$(call print_warning,"Please edit .env with your configuration"); \
	else \
		$(call print_info,".env file already exists"); \
	fi

# =============================================================================
# TESTING
# =============================================================================

test: ## Run full test suite
	@$(call print_info,"Running full test suite...")
	$(PYTEST) tests/ -v --tb=short
	@$(call print_success,"All tests completed!")

test-fast: ## Run fast tests only (skip slow integration tests)
	@$(call print_info,"Running fast tests...")
	$(PYTEST) tests/ -v -m "not slow" --tb=short
	@$(call print_success,"Fast tests completed!")

test-cov: ## Run tests with coverage report
	@$(call print_info,"Running tests with coverage...")
	$(PYTEST) tests/ --cov=$(PROJECT_NAME) --cov-report=html --cov-report=term
	@$(call print_success,"Coverage report generated in htmlcov/")

test-parallel: ## Run tests in parallel
	@$(call print_info,"Running tests in parallel...")
	$(PYTEST) tests/ -n auto -v
	@$(call print_success,"Parallel tests completed!")

# =============================================================================
# CODE QUALITY
# =============================================================================

lint: ## Run linting checks
	@$(call print_info,"Running linting checks...")
	ruff check src/ tests/ scripts/
	@$(call print_success,"Linting completed!")

format: ## Format code with black and isort
	@$(call print_info,"Formatting code...")
	black src/ tests/ scripts/
	isort src/ tests/ scripts/
	@$(call print_success,"Code formatting completed!")

type-check: ## Run type checking with mypy
	@$(call print_info,"Running type checks...")
	mypy src/$(PROJECT_NAME)/
	@$(call print_success,"Type checking completed!")

check-all: lint type-check test-fast ## Run all code quality checks

# =============================================================================
# ANALYSIS & EXPERIMENTS
# =============================================================================

analysis: ## Run the full integrated gradients analysis pipeline
	@$(call print_info,"Starting full analysis pipeline...")
	$(PYTHON) scripts/run_all_ig_analyses.py
	@$(call print_success,"Analysis pipeline completed!")

analysis-pafa: ## Run PafA analysis only
	@$(call print_info,"Running PafA analysis...")
	$(PYTHON) scripts/PafA_linear_IG.py
	@$(call print_success,"PafA analysis completed!")

analysis-betalactamase: ## Run betalactamase analysis only
	@$(call print_info,"Running betalactamase analysis...")
	$(PYTHON) scripts/betalactamase_linear_IG.py
	@$(call print_success,"Betalactamase analysis completed!")

analysis-erbc: ## Run ERBC analysis only
	@$(call print_info,"Running ERBC analysis...")
	$(PYTHON) scripts/ERBC_linear_IG.py
	@$(call print_success,"ERBC analysis completed!")

demo: ## Run demo analysis with visualizations
	@$(call print_info,"Running demo analysis...")
	$(PYTHON) scripts/demo_analysis_methods.py
	@$(call print_success,"Demo completed!")

# =============================================================================
# JUPYTER & NOTEBOOKS
# =============================================================================

jupyter: ## Start Jupyter Lab server
	@$(call print_info,"Starting Jupyter Lab...")
	@$(call print_warning,"Access at: http://localhost:8888")
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

notebooks: ## Validate and test all notebooks
	@$(call print_info,"Testing notebooks...")
	$(PYTEST) --nbval notebooks/ -v
	@$(call print_success,"Notebook validation completed!")

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

docker-build: ## Build Docker image
	@$(call print_info,"Building Docker image...")
	docker build -t $(DOCKER_IMAGE) .
	@$(call print_success,"Docker image built!")

docker-dev: ## Run development Docker container
	@$(call print_info,"Starting development container...")
	docker-compose up igloo-dev

docker-prod: ## Run production Docker container
	@$(call print_info,"Starting production container...")
	docker-compose --profile production up igloo-prod

docker-test: ## Run tests in Docker container
	@$(call print_info,"Running tests in Docker...")
	docker-compose run igloo-test

docker-clean: ## Clean Docker images and containers
	@$(call print_info,"Cleaning Docker resources...")
	docker-compose down --volumes --rmi all
	@$(call print_success,"Docker cleanup completed!")

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs: ## Build documentation
	@$(call print_info,"Building documentation...")
	$(MAKE) -C docs html
	@$(call print_success,"Documentation built in docs/_build/html/")

docs-serve: ## Serve documentation locally
	@$(call print_info,"Serving documentation at http://localhost:8000")
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	@$(call print_info,"Cleaning documentation...")
	$(MAKE) -C docs clean

# =============================================================================
# BUILD & RELEASE
# =============================================================================

build: clean ## Build source and wheel package
	@$(call print_info,"Building package...")
	$(PYTHON) -m build
	@$(call print_success,"Package built in dist/")

release: clean build ## Package and upload a release
	@$(call print_info,"Uploading release...")
	twine upload dist/*
	@$(call print_success,"Release uploaded!")

release-test: clean build ## Upload to test PyPI
	@$(call print_info,"Uploading to test PyPI...")
	twine upload --repository testpypi dist/*
	@$(call print_success,"Test release uploaded!")

# =============================================================================
# CLEANING
# =============================================================================

clean: ## Remove build artifacts
	@$(call print_info,"Cleaning build artifacts...")
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove Python file artifacts
	@$(call print_info,"Cleaning Python artifacts...")
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test and coverage artifacts
	@$(call print_info,"Cleaning test artifacts...")
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache/
	rm -fr .cache/

clean-all: clean clean-pyc clean-test docs-clean ## Remove all artifacts
	@$(call print_info,"Cleaning all artifacts...")
	rm -fr .mypy_cache/
	rm -fr .ruff_cache/
	rm -fr logs/
	rm -fr results/temp/
	@$(call print_success,"Deep clean completed!")

# =============================================================================
# DEVELOPMENT HELPERS
# =============================================================================

install-pre-commit: ## Install pre-commit hooks
	@$(call print_info,"Installing pre-commit hooks...")
	pre-commit install
	@$(call print_success,"Pre-commit hooks installed!")

update-deps: ## Update all dependencies
	@$(call print_info,"Updating dependencies...")
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r requirements_dev.txt
	@$(call print_success,"Dependencies updated!")

freeze-deps: ## Freeze current dependencies
	@$(call print_info,"Freezing dependencies...")
	$(PIP) freeze > requirements_frozen.txt
	@$(call print_success,"Dependencies frozen to requirements_frozen.txt")

profile: ## Run profiling on sample analysis
	@$(call print_info,"Running performance profiling...")
	$(PYTHON) -m cProfile -o profile.stats scripts/demo_analysis_methods.py
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('tottime').print_stats(20)"
	@$(call print_success,"Profiling completed!")

# =============================================================================
# QUICK WORKFLOWS
# =============================================================================

dev-setup: install-dev setup-env install-pre-commit ## Complete development setup
	@$(call print_success,"Development environment ready!")
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env with your configuration"
	@echo "2. Run 'make test' to verify installation"
	@echo "3. Run 'make analysis' to start protein analysis"

prod-setup: install setup-env ## Production setup
	@$(call print_success,"Production environment ready!")

ci-test: check-all test-cov ## Run all CI checks
	@$(call print_success,"All CI checks passed!")

# =============================================================================
# HEALTH CHECKS
# =============================================================================

health-check: ## Run comprehensive health check
	@$(call print_info,"Running health check...")
	@echo "System Information:"
	@echo "==================="
	@$(PYTHON) --version
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
	@if [ -d data ]; then echo "Data directory: ✅"; else echo "Data directory: ❌"; fi
	@if [ -f .env ]; then echo "Environment file: ✅"; else echo "Environment file: ❌"; fi
	@$(call print_success,"Health check completed!")

benchmark: ## Run performance benchmarks
	@$(call print_info,"Running benchmarks...")
	$(PYTHON) -m pytest benchmarks/ -v --benchmark-only
	@$(call print_success,"Benchmarks completed!")

# =============================================================================
# SPECIAL TARGETS
# =============================================================================

requirements: ## Generate requirements from pyproject.toml
	@$(call print_info,"Generating requirements.txt from pyproject.toml...")
	pip-compile pyproject.toml --output-file requirements.txt
	@$(call print_success,"Requirements generated!")

version: ## Show current version
	@$(PYTHON) -c "import $(PROJECT_NAME); print(f'$(PROJECT_NAME) version: {$(PROJECT_NAME).__version__}')"

# Safety check to prevent accidental execution
ifeq ($(MAKECMDGOALS),)
$(error Please specify a target. Run 'make help' to see available targets.)
endif
