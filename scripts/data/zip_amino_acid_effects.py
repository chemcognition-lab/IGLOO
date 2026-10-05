#!/usr/bin/env python3
"""
Zip amino_acid_effects_detailed CSVs by fitness model.

For each model type (additive, epistatic, nonlinear) this script:
  1. Scans  results/synthetic/GB1_{model}/noise_{level}/{interp}/
             GB1_{model}_amino_acid_effects_detailed.csv
  2. Renames each file inside the archive to:
             GB1_{model}_noise_{level}_{interp}_amino_acid_effects_detailed.csv
  3. Writes  results/synthetic/GB1_{model}_amino_acid_effects_detailed.zip

Usage
-----
    source IGLOO_env/bin/activate
    python scripts/data/zip_amino_acid_effects.py

    # Custom output dir:
    python scripts/data/zip_amino_acid_effects.py --output-dir /some/other/path

    # Only specific models:
    python scripts/data/zip_amino_acid_effects.py --models additive epistatic
"""

import argparse
import re
import zipfile
from pathlib import Path

SYNTHETIC_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "synthetic"
MODELS = ["additive", "epistatic", "nonlinear"]
PROTEIN = "GB1"


def zip_model(model: str, output_dir: Path) -> None:
    """Collect and zip all amino_acid_effects_detailed.csv files for one model."""
    model_root = SYNTHETIC_RESULTS_DIR / f"{PROTEIN}_{model}"
    if not model_root.is_dir():
        print(f"  [skip] Directory not found: {model_root}")
        return

    # Find all matching CSVs
    csv_files = sorted(model_root.rglob("*_amino_acid_effects_detailed.csv"))
    if not csv_files:
        print(f"  [skip] No amino_acid_effects_detailed.csv files found under {model_root}")
        return

    zip_path = output_dir / f"{PROTEIN}_{model}_amino_acid_effects_detailed.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_path in csv_files:
            # Path relative to model root: noise_{level}/{interp}/GB1_{model}_amino_acid_effects_detailed.csv
            # We parse noise level and interp from the parent dirs
            try:
                rel = csv_path.relative_to(model_root)
                parts = rel.parts  # e.g. ('noise_0.0', 'linear', 'GB1_additive_amino_acid_effects_detailed.csv')
            except ValueError:
                parts = csv_path.parts[-3:]  # fallback

            # Extract noise level
            noise_part = next(
                (p for p in parts if re.match(r"noise[_\-][0-9]", p, re.IGNORECASE)),
                None,
            )
            noise_str = ""
            if noise_part:
                m = re.search(r"([0-9]*\.?[0-9]+)", noise_part)
                noise_str = m.group(1) if m else noise_part

            # Extract interpolation method (the directory directly containing the CSV)
            interp_str = csv_path.parent.name  # e.g. 'linear', 'slerp', 'nearest_neighbor'

            # Build archive name
            archive_name = (
                f"{PROTEIN}_{model}_noise_{noise_str}_{interp_str}"
                f"_amino_acid_effects_detailed.csv"
            )

            zf.write(csv_path, archive_name)
            print(f"    + {archive_name}")

    print(f"  Wrote {zip_path}  ({len(csv_files)} files)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zip amino_acid_effects_detailed CSVs by fitness model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help=f"Fitness models to process (default: {MODELS}).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SYNTHETIC_RESULTS_DIR),
        help="Directory to write the zip files (default: results/synthetic/).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        print(f"\nProcessing {PROTEIN}_{model} ...")
        zip_model(model, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
