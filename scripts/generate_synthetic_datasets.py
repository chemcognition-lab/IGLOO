#!/usr/bin/env python
"""
generate_synthetic_datasets.py
-------------------------------
Generate synthetic protein fitness datasets for IGLOO validation.

Fitness values are stored noiseless; apply noise at model-training time.

Usage examples
--------------
# Generate all proteins, all models (per-protein default n_variants):
    uv run scripts/generate_synthetic_datasets.py

# Generate T4Lysozyme and p53, additive + epistatic only, 2000 variants each:
    uv run scripts/generate_synthetic_datasets.py \\
        --proteins T4Lysozyme p53 \\
        --models additive epistatic \\
        --n-variants 2000

# List proteins and exit:
    uv run scripts/generate_synthetic_datasets.py --list

Output directory: data/synthetic/  (customisable with --output-dir)
Output files:     {protein}_{model}.csv
                  {protein}_ground_truth_weights.json
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

from igloo.synthetic_data import (  # noqa: E402
    _FITNESS_MODELS,
    PROTEINS,
    SyntheticDatasetGenerator,
    split_train_test,
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
        choices=list(_FITNESS_MODELS.keys()),
        default=list(_FITNESS_MODELS.keys()),
        metavar="MODEL",
        help=f"Fitness models to generate. Choices: {list(_FITNESS_MODELS.keys())}. Default: all.",
    )
    p.add_argument(
        "--n-variants",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of variant sequences to generate. "
            "Default: per-protein value (GB1=1500, T4Lysozyme=4000, p53=10000)."
        ),
    )
    p.add_argument(
        "--split-test",
        action="store_true",
        help=(
            "After generating each dataset, write two additional files: "
            "{protein}_{model}_trainval.csv (80%%) and "
            "{protein}_{model}_test.csv (20%%, stratified by variant type). "
            "The split fraction can be changed with --test-fraction."
        ),
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        metavar="F",
        help="Fraction of non-wildtype variants to hold out as test set (default: 0.20).",
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
              f"annotated={n_annotated} positions  "
              f"default_n_variants={prot.default_n_variants}")
        print(f"  {'':15s}  categories: {', '.join(categories)}")
        print()


def dry_run_report(
    proteins: list,
    models: list,
    n_variants_override: int | None,
    output_dir: str,
    split_test: bool = False,
    test_fraction: float = 0.20,
) -> None:
    print("\n[DRY RUN — no files will be written]\n")
    total_files = 0
    total_rows  = 0
    for protein_name in proteins:
        prot = PROTEINS[protein_name]
        nv   = n_variants_override if n_variants_override is not None else prot.default_n_variants
        n_rows = nv + 1  # +1 for WT
        for model in models:
            fname = Path(output_dir) / f"{protein_name}_{model}.csv"
            print(f"  Would write  {fname}  (~{n_rows} rows)")
            total_files += 1
            total_rows  += n_rows
            if split_test:
                n_test     = max(1, round(nv * test_fraction))
                n_trainval = nv + 1 - n_test
                print(f"  Would write  {Path(output_dir) / f'{protein_name}_{model}_trainval.csv'}"
                      f"  (~{n_trainval} rows)")
                print(f"  Would write  {Path(output_dir) / f'{protein_name}_{model}_test.csv'}"
                      f"  (~{n_test} rows)")
                total_files += 2
        gt_path = Path(output_dir) / f"{protein_name}_ground_truth_weights.json"
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
            args.n_variants,
            str(out_dir),
            split_test=args.split_test,
            test_fraction=args.test_fraction,
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
        nv  = args.n_variants if args.n_variants is not None else gen.protein.default_n_variants

        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        for model in args.models:
            if model not in _FITNESS_MODELS:
                print(f"  [WARN] Unknown model '{model}', skipping.")
                continue
            df = gen.generate(
                model=model,
                n_variants=nv,
                seed=args.seed,
            )
            fname = out_dir / f"{protein_name}_{model}.csv"
            df.to_csv(fname, index=False)
            print(f"  Saved {fname}  ({len(df):,} rows)")
            saved.append(str(fname))

            if args.split_test:
                trainval_df, test_df = split_train_test(
                    df, test_fraction=args.test_fraction, seed=args.seed
                )
                tv_fname   = out_dir / f"{protein_name}_{model}_trainval.csv"
                test_fname = out_dir / f"{protein_name}_{model}_test.csv"
                trainval_df.to_csv(tv_fname,   index=False)
                test_df.to_csv(test_fname, index=False)
                print(f"  Saved {tv_fname}  ({len(trainval_df):,} rows, train+val)")
                print(f"  Saved {test_fname}  ({len(test_df):,} rows, test)")
                saved.extend([str(tv_fname), str(test_fname)])

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
