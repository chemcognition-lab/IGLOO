"""Backward-compatible entry point.

The real script now lives in scripts/pipeline/run_integrated_gradients.py.
This shim only exists so already-submitted Slurm jobs that call
``python scripts/run_integrated_gradients.py`` keep working; new code and job
scripts should call the pipeline/ path directly.
"""
import runpy
import sys
from pathlib import Path

_target = Path(__file__).resolve().parent / "pipeline" / "run_integrated_gradients.py"
sys.path.insert(0, str(_target.parent))
sys.argv[0] = str(_target)
runpy.run_path(str(_target), run_name="__main__")
