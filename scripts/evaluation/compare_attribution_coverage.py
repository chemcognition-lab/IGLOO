#!/usr/bin/env python3
"""
compare_attribution_coverage.py
--------------------------------
Compare per-position attribution importance from one or more results directories
against the mutational frequency observed in a dataset CSV.

Checks:
  1. How do attribution scores align with positions that actually carry mutations
     in the training data?
  2. Are there positions with high attribution but *zero* mutational coverage —
     i.e., the model attributes importance to positions it was never trained on?

Outputs
-------
  <output-dir>/attribution_vs_coverage.png
      One row per results directory — normalised attribution line overlaid on
      normalised mutation-frequency fill.  Uncovered hot-spots marked with red
      stars.  Ground-truth annotations shaded when --ground-truth is supplied.

  <output-dir>/uncovered_hotspots.csv
      Table of (label, position, mean_abs_gradient, mutation_count=0) for every
      flagged position.

Usage examples
--------------
# Single method directory
python scripts/evaluation/compare_attribution_coverage.py \\
    --results-dirs results/synthetic/GB1_additive/noise_0.0/linear \\
    --dataset data/synthetic/GB1_additive.csv

# Compare multiple methods / noise levels with ground-truth overlay
python scripts/evaluation/compare_attribution_coverage.py \\
    --results-dirs \\
        results/synthetic/GB1_additive/noise_0.0/linear \\
        results/synthetic/GB1_additive/noise_0.0/slerp \\
        results/synthetic/GB1_additive/noise_0.5/linear \\
    --dataset data/synthetic/GB1_additive.csv \\
    --ground-truth data/synthetic/GB1_ground_truth_weights.json \\
    --output-dir results/synthetic/GB1_additive/coverage_analysis

# Sweep an entire dataset results tree (all noise levels × all interp methods)
python scripts/evaluation/compare_attribution_coverage.py \\
    --results-dir-tree results/synthetic/GB1_additive \\
    --dataset data/synthetic/GB1_additive.csv \\
    --ground-truth data/synthetic/GB1_ground_truth_weights.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from igloo.plot import plot_attribution_vs_mutation_coverage  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_importance_csv(results_dir: Path) -> Path | None:
    """Return the *_position_importance_summary.csv inside results_dir, or None."""
    hits = sorted(results_dir.glob("*_position_importance_summary.csv"))
    if hits:
        return hits[0]
    return None


def _load_results_dirs(paths: list[Path]) -> dict[str, pd.DataFrame]:
    """
    Load position_importance_summary CSVs from a list of leaf directories.
    Label is  ``<parent_dir>/<dir_name>`` (noise level / interp method).
    """
    loaded: dict[str, pd.DataFrame] = {}
    for p in paths:
        csv = _find_importance_csv(p)
        if csv is None:
            print(f"  [warn] No position_importance_summary.csv found in {p} — skipping.")
            continue
        # Build a concise label from the two innermost path components
        label = "/".join(p.parts[-2:]) if len(p.parts) >= 2 else p.name
        loaded[label] = pd.read_csv(csv)
        print(f"  Loaded: {label}  ({len(loaded[label])} positions)")
    return loaded


def _discover_leaf_dirs(tree_root: Path) -> list[Path]:
    """
    Walk tree_root and return every directory that contains a
    *_position_importance_summary.csv file.
    """
    return sorted(
        p.parent
        for p in tree_root.rglob("*_position_importance_summary.csv")
    )


def _mutation_frequency(
    df: pd.DataFrame,
    exclude_wt: bool = True,
    mutation_col: str = "mutation_string",
) -> pd.Series:
    """
    Count how many variants in *df* carry a mutation at each 1-based position.

    Parses *mutation_col* (format: ``A1B;C2D;…`` or ``;``-separated).
    Returns a Series indexed by position (int).
    """
    if mutation_col not in df.columns:
        # Try common aliases
        for alias in ("mutation_string", "mutant", "mutations", "variant"):
            if alias in df.columns:
                mutation_col = alias
                break
        else:
            raise ValueError(
                f"Could not find a mutation column in the dataset. "
                f"Available columns: {list(df.columns)}. "
                f"Use --mutation-col to specify the correct column name."
            )

    if exclude_wt and "is_wildtype" in df.columns:
        df = df[~df["is_wildtype"].astype(bool)]

    # Detect WT rows by empty / sentinel mutation strings when is_wildtype absent
    wt_sentinels = {"", "wt", "wildtype", "wild_type", "wild-type"}

    pos_counts: dict[int, int] = {}
    for ms in df[mutation_col].dropna():
        if str(ms).strip().lower() in wt_sentinels or str(ms).strip().upper() == "WT":
            continue
        for mut in ms.split(";"):
            mut = mut.strip()
            if not mut:
                continue
            # Extract all consecutive digits from the string (position)
            digits = "".join(c for c in mut if c.isdigit())
            if digits:
                pos = int(digits)
                pos_counts[pos] = pos_counts.get(pos, 0) + 1

    return pd.Series(pos_counts, dtype=int)


def _infer_seq_length(df: pd.DataFrame, gt: dict | None) -> int:
    """Infer sequence length from ground-truth JSON or from the WT sequence."""
    if gt and "sequence_length" in gt:
        return int(gt["sequence_length"])
    wt_rows = df[df.get("is_wildtype", pd.Series(False, index=df.index)).astype(bool)]
    seq_col = next((c for c in ["SEQ", "sequence", "Sequence"] if c in df.columns), None)
    if seq_col and not wt_rows.empty:
        return len(wt_rows.iloc[0][seq_col])
    if seq_col:
        return df[seq_col].str.len().max()
    raise ValueError(
        "Cannot infer sequence length — supply --ground-truth or ensure the "
        "dataset has a SEQ/sequence column."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-position attributions against dataset mutational "
            "frequency and flag uncovered hot-spots."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage examples")[1] if "Usage examples" in __doc__ else "",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--results-dirs",
        nargs="+",
        type=Path,
        metavar="DIR",
        help=(
            "One or more leaf result directories, each containing a "
            "*_position_importance_summary.csv file "
            "(e.g. results/synthetic/GB1_additive/noise_0.0/linear)."
        ),
    )
    src.add_argument(
        "--results-dir-tree",
        type=Path,
        metavar="ROOT",
        help=(
            "Root directory to scan recursively for all "
            "*_position_importance_summary.csv files."
        ),
    )

    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        metavar="CSV",
        help="Synthetic dataset CSV (must contain mutation_string column).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        metavar="JSON",
        help="Optional ground-truth weights JSON for annotation shading.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory for output files.  Defaults to "
            "<first-results-dir>/coverage_analysis or "
            "<results-dir-tree>/coverage_analysis."
        ),
    )
    parser.add_argument(
        "--uncovered-threshold",
        type=float,
        default=10.0,
        metavar="PCT",
        help=(
            "Positions whose normalised attribution is in the top PCT%% "
            "AND have zero mutation count are flagged (default: 10)."
        ),
    )
    parser.add_argument(
        "--mutation-col",
        default="mutation_string",
        metavar="COL",
        help=(
            "Name of the column in the dataset CSV that contains mutation strings "
            "(semicolon-separated, e.g. 'A1B;C2D'). "
            "Defaults to 'mutation_string'. FLIP2 datasets use 'mutant'."
        ),
    )
    parser.add_argument(
        "--include-wt",
        action="store_true",
        help="Include the WT row when computing mutation frequency (default: excluded).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution for saved figures.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ── Collect results directories ──────────────────────────────────────
    if args.results_dirs:
        dirs = [Path(d) for d in args.results_dirs]
        default_out_parent = dirs[0]
    else:
        print(f"Scanning {args.results_dir_tree} for result directories …")
        dirs = _discover_leaf_dirs(args.results_dir_tree)
        default_out_parent = args.results_dir_tree
        print(f"  Found {len(dirs)} directories.")

    if not dirs:
        print("No result directories found — nothing to do.")
        sys.exit(0)

    output_dir = args.output_dir or (default_out_parent / "coverage_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load attributions ────────────────────────────────────────────────
    print("Loading attribution summaries …")
    results = _load_results_dirs(dirs)
    if not results:
        print("No valid attribution files loaded — aborting.")
        sys.exit(1)

    # ── Load dataset ─────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset} …")
    dataset_df = pd.read_csv(args.dataset)
    mut_freq = _mutation_frequency(dataset_df, exclude_wt=not args.include_wt,
                                    mutation_col=args.mutation_col)
    print(f"  {len(dataset_df):,} variants | "
          f"{int(mut_freq.sum()):,} total mutations across "
          f"{len(mut_freq)} positions")

    # ── Load ground truth ─────────────────────────────────────────────────
    gt = None
    if args.ground_truth:
        with open(args.ground_truth) as fh:
            gt = json.load(fh)
        print(f"  Ground-truth loaded: {args.ground_truth.name}")

    # ── Infer sequence length ─────────────────────────────────────────────
    seq_len = _infer_seq_length(dataset_df, gt)
    print(f"  Sequence length: {seq_len}")

    # ── Plot ──────────────────────────────────────────────────────────────
    png_path = output_dir / "attribution_vs_coverage.png"
    uncovered_df = plot_attribution_vs_mutation_coverage(
        results=results,
        mut_freq=mut_freq,
        seq_length=seq_len,
        ground_truth=gt,
        uncovered_threshold_pct=args.uncovered_threshold,
        save_path=str(png_path),
        dpi=args.dpi,
    )

    # ── Save uncovered hot-spots table ────────────────────────────────────
    csv_path = output_dir / "uncovered_hotspots.csv"
    uncovered_df.to_csv(csv_path, index=False)

    if uncovered_df.empty:
        print("No uncovered hot-spots detected.")
    else:
        print(f"\nUncovered hot-spots (top {args.uncovered_threshold:.0f}% attribution, "
              f"zero mutation coverage):")
        print(uncovered_df.to_string(index=False))
        print(f"\nSaved → {csv_path}")

    print(f"\nDone.  Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
