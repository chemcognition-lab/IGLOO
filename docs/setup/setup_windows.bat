@echo off
REM Windows setup script for IGLOO
REM This script handles Windows-specific dependencies and setup

echo 🪟 Setting up IGLOO on Windows...
echo ================================

REM Check Windows version
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j
echo 📋 Windows version: %VERSION%

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% found

REM Check for Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git not found. Please install Git from https://git-scm.com
    pause
    exit /b 1
)

REM Check for conda
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 🐍 Conda not found. Installing Miniconda...
    echo Please download and install Miniconda from:
    echo https://docs.conda.io/en/latest/miniconda.html
    echo Then restart this script.
    pause
    exit /b 1
)

REM Install Microsoft Visual C++ Build Tools if needed
echo 🔧 Checking for Visual C++ Build Tools...
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Visual C++ Build Tools not found
    echo Please install "Microsoft C++ Build Tools" or Visual Studio
    echo Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
)

REM Create conda environment
echo 🧊 Creating IGLOO conda environment...
conda env list | findstr "igloo" >nul
if %errorlevel% equ 0 (
    echo ⚠️  Environment 'igloo' already exists. Removing...
    conda env remove -n igloo -y
)

REM Create Windows-specific environment file
echo name: igloo > environment_windows.yml
echo channels: >> environment_windows.yml
echo   - conda-forge >> environment_windows.yml
echo   - pytorch >> environment_windows.yml
echo   - defaults >> environment_windows.yml
echo dependencies: >> environment_windows.yml
echo   - python=3.10 >> environment_windows.yml
echo   - numpy >> environment_windows.yml
echo   - pandas >> environment_windows.yml
echo   - scipy >> environment_windows.yml
echo   - scikit-learn >> environment_windows.yml
echo   - matplotlib >> environment_windows.yml
echo   - seaborn >> environment_windows.yml
echo   - plotly >> environment_windows.yml
echo   - jupyter >> environment_windows.yml
echo   - jupyterlab >> environment_windows.yml
echo   - pytest >> environment_windows.yml
echo   - tqdm >> environment_windows.yml
echo   - pyyaml >> environment_windows.yml
echo   - requests >> environment_windows.yml
echo   - biopython >> environment_windows.yml
echo   - pytorch::pytorch >> environment_windows.yml
echo   - pytorch::torchvision >> environment_windows.yml
echo   - pip >> environment_windows.yml
echo   - pip: >> environment_windows.yml
echo     - fair-esm >> environment_windows.yml
echo     - transformers >> environment_windows.yml
echo     - safetensors >> environment_windows.yml
echo     - captum >> environment_windows.yml
echo     - py3Dmol >> environment_windows.yml
echo     - xgboost >> environment_windows.yml

conda env create -f environment_windows.yml

REM Activate environment and install package
call conda activate igloo
pip install -e .

REM Set up environment variables
echo ⚡ Setting up Windows optimizations...
setx IGLOO_ROOT "%CD%"
setx PYTHONPATH "%CD%\src;%PYTHONPATH%"
setx OMP_NUM_THREADS "4"
setx MKL_NUM_THREADS "4"
setx NUMEXPR_NUM_THREADS "4"

REM Create batch files for easy access
echo @echo off > start_igloo.bat
echo call conda activate igloo >> start_igloo.bat
echo cd /d "%CD%" >> start_igloo.bat
echo cmd /k >> start_igloo.bat

echo @echo off > start_jupyter.bat
echo call conda activate igloo >> start_jupyter.bat
echo cd /d "%CD%" >> start_jupyter.bat
echo jupyter lab --ip=0.0.0.0 --port=8888 --no-browser >> start_jupyter.bat

REM Clean up
del environment_windows.yml

echo ✅ Windows setup complete!
echo.
echo 🎉 Next steps:
echo 1. Close and reopen Command Prompt/PowerShell
echo 2. Run: conda activate igloo
echo 3. Copy environment template: copy .env.template .env
echo 4. Run tests: python -m pytest tests/
echo 5. Start analysis: python scripts/run_all_ig_analyses.py
echo.
echo 💡 Windows-specific files created:
echo - start_igloo.bat: Activate environment and open command prompt
echo - start_jupyter.bat: Start Jupyter Lab server
echo.
echo ⚠️  Note: Some features may require WSL for full compatibility
echo Consider using Docker for the most consistent experience

pause
