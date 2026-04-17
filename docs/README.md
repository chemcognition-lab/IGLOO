# IGLOO Documentation

This directory contains comprehensive documentation for the IGLOO project.

## Structure

### Getting Started
- **[quickstart.rst](quickstart.rst)** - Quick start guide with setup options
- **[setup/](setup/)** - Installation and environment setup guides
  - `installation.md` - Detailed installation instructions
  - `environment_setup.md` - Environment configuration guide
  - `setup_guide.md` - Overview of all setup files and methods
  - Setup scripts: `quick_setup.sh`, `setup_*.sh`, `setup_windows.bat`

### User Guides
- **[usage.rst](usage.rst)** - Basic usage documentation
- **[configuration/](configuration/)** - Configuration and analysis guides
  - `configuration_guide.md` - How to create and use config files
  - `configuration_summary.md` - Complete overview of all configuration options
- **[results_guide.md](results_guide.md)** - Understanding analysis results

### Development
- **[development/](development/)** - Development and contribution guides
  - `contributing.rst` - How to contribute to the project
  - `modularization_summary.md` - Architecture and modularization overview
  - `authors.rst` - Project authors and contributors
  - `history.rst` - Project history and changelog

### Reference
- **[readme.rst](readme.rst)** - Main project README
- **[modules](modules)** - API documentation (auto-generated)

## Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Online Documentation

The documentation is also available online at: [https://igloo.readthedocs.io](https://igloo.readthedocs.io)
