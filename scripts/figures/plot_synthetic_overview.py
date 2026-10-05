#!/usr/bin/env python3
"""
plot_synthetic_overview.py
--------------------------
Generate distribution and fitness overview plots for the synthetic datasets.

Produces two figures:

  results/synthetic/dataset_overview.png
      One row per dataset (9 rows: 3 proteins × 3 fitness models).
      Left panel  – mutation-tier bar chart (WT / single / double / triple).
      Right panel – per-tier fitness KDE.

  results/synthetic/dataset_overview_deep.png
      Same layout for the deep (high-order mutation) variants in data/synthetic/deep/.
      Only GB1 datasets are present in the deep split.

Usage
-----
    python scripts/figures/plot_synthetic_overview.py               # all proteins & models
    python scripts/figures/plot_synthetic_overview.py --proteins GB1 p53
    python scripts/figures/plot_synthetic_overview.py --models additive nonlinear
    python scripts/figures/plot_synthetic_overview.py --no-deep     # skip deep variants
    python scripts/figures/plot_synthetic_overview.py --output-dir  results/figures/synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── resolve project root so src/igloo is importable without an editable install ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from igloo.plot import load_synthetic_datasets, plot_synthetic_dataset_overview  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot distribution and fitness overview for synthetic datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=str(_PROJECT_ROOT / "data" / "synthetic"),
        help="Root directory that contains the synthetic CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_PROJECT_ROOT / "results" / "synthetic"),
        help="Directory where the output PNG files will be written.",
    )
    parser.add_argument(
        "--proteins",
        nargs="+",
        default=None,
        metavar="PROTEIN",
        help="Restrict to these proteins (case-insensitive). E.g. GB1 p53 T4Lysozyme.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Restrict to these fitness models. E.g. additive epistatic nonlinear.",
    )
    parser.add_argument(
        "--no-deep",
        action="store_true",
        help="Skip the deep-variant overview plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution (dots per inch) for saved figures.",
    )
    parser.add_argument(
        "--white-bg",
        action="store_true",
        help="Use a white background instead of the default dark theme.",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Save with a transparent background.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Standard datasets ────────────────────────────────────────────────
    print("Loading standard synthetic datasets …")
    datasets = load_synthetic_datasets(
        data_dir,
        proteins=args.proteins,
        models=args.models,
    )

    if not datasets:
        print("No datasets matched the given filters — nothing to plot.")
        sys.exit(0)

    print(f"  Found {len(datasets)} dataset(s): {', '.join(datasets.keys())}")

    _facecolor = "white" if args.white_bg else None
    _fc_kwargs  = {} if _facecolor is None else {"facecolor": _facecolor}
    if args.transparent:
        _fc_kwargs["transparent"] = True

    out_standard = output_dir / "dataset_overview.png"
    plot_synthetic_dataset_overview(
        datasets,
        save_path=str(out_standard),
        dpi=args.dpi,
        **_fc_kwargs,
    )

    # ── Deep datasets ────────────────────────────────────────────────────
    if not args.no_deep:
        deep_dir = data_dir / "deep"
        if deep_dir.is_dir():
            print("Loading deep-variant synthetic datasets …")
            deep_datasets = load_synthetic_datasets(
                data_dir,
                proteins=args.proteins,
                models=args.models,
                deep=True,
            )

            if deep_datasets:
                print(f"  Found {len(deep_datasets)} deep dataset(s): {', '.join(deep_datasets.keys())}")
                out_deep = output_dir / "dataset_overview_deep.png"
                plot_synthetic_dataset_overview(
                    deep_datasets,
                    save_path=str(out_deep),
                    dpi=args.dpi,
                    deep=True,
                    **_fc_kwargs,
                )
            else:
                print("  No deep datasets matched the given filters — skipping.")
        else:
            print(f"  No deep/ sub-directory found at {deep_dir} — skipping.")

    print("Done.")


if __name__ == "__main__":
    main()
