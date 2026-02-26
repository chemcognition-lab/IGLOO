#!/usr/bin/env python
"""
generate_synthetic_datasets.py
-------------------------------
Generate synthetic protein fitness datasets for IGLOO validation.

Usage examples
--------------
# Generate all proteins, all models, all noise levels (default):
    uv run scripts/generate_synthetic_datasets.py

# Generate only GB1 with 500 variants at noise 0.0 and 0.1:
    uv run scripts/generate_synthetic_datasets.py \
        --proteins GB1 \
        --noise-levels 0.0 0.1 \
        --n-variants 500

# Generate T4Lysozyme and p53DBD, additive + epistatic only:
    uv run scripts/generate_synthetic_datasets.py \
        --proteins T4Lysozyme p53DBD \
        --models additive epistatic

# List proteins and exit:
    uv run scripts/generate_synthetic_datasets.py --list

Output directory: data/synthetic/  (customisable with --output-dir)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── resolve project root so src/igloo is importable without editable install ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from igloo.synthetic_data import (   # noqa: E402
    PROTEINS,
    SyntheticDatasetGenerator,
    generate_all_proteins,
    _NOISE_LEVELS,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate synthetic protein fitness datasets for IGLOO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--proteins",
        nargs="+",
        choices=list(PROTEINS.keys()),
        default=list(PROTEINS.keys()),
        metavar="PROTEIN",
        help=(
            f"Proteins to generate datasets for. "
            f"Choices: {list(PROTEINS.keys())}. "
            f"Default: all."
        ),
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=["additive", "epistatic", "nonlinear"],
        default=["additive", "epistatic", "nonlinear"],
        metavar="MODEL",
        help="Fitness models to generate. Default: all three.",
    )
    p.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=_NOISE_LEVELS,
        metavar="FLOAT",
        help=f"Gaussian noise std values. Default: {_NOISE_LEVELS}",
    )
    p.add_argument(
        "--n-variants",
        type=int,
        default=2000,
        metavar="N",
        help="Number of variant sequences per noise level. Default: 2000.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        metavar="DIR",
        help="Output directory for CSVs and ground-truth JSON files. Default: data/synthetic.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for reproducibility. Default: 42.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available proteins and exit.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing any files.",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def list_proteins() -> None:
    print("Available proteins:")
    print()
    for name, prot in PROTEINS.items():
        n_annotated = len(prot.annotations)
        categories  = sorted(set(prot.annotations.values()))
        print(f"  {name:15s}  UniProt={prot.uniprot_id}  "
              f"len={len(prot.sequence)} aa  "
              f"annotated={n_annotated} positions")
        print(f"  {'':15s}  categories: {', '.join(categories)}")
        print()


def dry_run_report(
    proteins: list,
    models: list,
    noise_levels: list,
    n_variants: int,
    output_dir: str,
) -> None:
    print("\n[DRY RUN — no files will be written]\n")
    total_files = 0
    total_rows  = 0
    for protein in proteins:
        for model in models:
            for noise in noise_levels:
                fname = (
                    Path(output_dir) /
                    f"{protein}_{model}_noise{noise:.1f}.csv"
                )
                n_rows = n_variants + 1  # +1 for WT
                print(f"  Would write  {fname}  ({n_rows} rows)")
                total_files += 1
                total_rows  += n_rows
        gt_path = Path(output_dir) / f"{protein}_ground_truth_weights.json"
        print(f"  Would write  {gt_path}")
        total_files += 1
    print()
    print(f"Total: {total_files} files, ~{total_rows:,} data rows")


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    # Resolve output_dir relative to project root so that
    # `uv run scripts/...` from any cwd puts files in the right place
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir

    if args.list:
        list_proteins()
        return 0

    if args.dry_run:
        dry_run_report(
            args.proteins,
            args.models,
            args.noise_levels,
            args.n_variants,
            str(out_dir),
        )
        return 0

    # ── Generate ──
    t0 = time.perf_counter()

    all_saved: dict[str, list[str]] = {}
    for protein_name in args.proteins:
        print(f"\n{'='*60}")
        print(f"  {protein_name}")
        print(f"{'='*60}")
        gen = SyntheticDatasetGenerator(protein_name)

        # We need to selectively generate only the requested models.
        # Call generate_all internals per-model to honour --models flag.
        import pandas as pd
        from igloo.synthetic_data import _FITNESS_MODELS

        saved = []
        for model in args.models:
            if model not in _FITNESS_MODELS:
                print(f"  [WARN] Unknown model '{model}', skipping.")
                continue
            frames = []
            for noise in args.noise_levels:
                df = gen._generate_one(
                    model=model,
                    n_variants=args.n_variants,
                    noise=noise,
                    seed=args.seed,
                )
                frames.append(df)
                fname = out_dir / f"{protein_name}_{model}_noise{noise:.1f}.csv"
                out_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(fname, index=False)
                print(f"  Saved {fname}  ({len(df):,} rows)")
                saved.append(str(fname))

        # Ground-truth importance vector
        from igloo.synthetic_data import _p53_pos  # noqa
        gt_path = out_dir / f"{protein_name}_ground_truth_weights.json"
        gen._save_ground_truth(gt_path)
        print(f"  Saved {gt_path}")
        saved.append(str(gt_path))

        all_saved[protein_name] = saved

    elapsed = time.perf_counter() - t0
    total = sum(len(v) for v in all_saved.values())
    print(f"\nDone. {total} files written in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
