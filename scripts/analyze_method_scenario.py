#!/usr/bin/env python3
"""
Method-scenario interaction analysis — CLI entry point.

All analysis logic lives in ``src/igloo/method_selection``.
This script just wires command-line arguments to those functions.

Usage
-----
    python scripts/analyze_method_scenario.py

    python scripts/analyze_method_scenario.py \\
        --results-dir results/synthetic \\
        --output-dir  results/synthetic/method_analysis \\
        --proteins GB1 T4Lysozyme p53 \\
        --fitness-models additive epistatic nonlinear

Flag reference
--------------
    --results-dir PATH           Root of synthetic results (default: results/synthetic/)
    --output-dir  PATH           Output directory          (default: results/synthetic/method_analysis/)
    --proteins [...]             Restrict proteins         (default: all)
    --fitness-models [...]       Restrict fitness models   (default: all)
    --concordance-threshold F    Kendall W cutoff: above → universal, below → conditional (default: 0.6)
    --min-output-diff F          Collapse threshold        (default: 0.01)
    --min-count INT              Min sequences per position (default: 5)
    --alpha F                    Significance level for Friedman / Nemenyi (default: 0.05)
    --dpi INT                    Figure DPI                (default: 150)
    --split {trainval,test}      Restrict to a data split  (default: all)
    --no-decay                   Skip exponential-decay fitting
"""

from __future__ import annotations

import argparse
import sys
import pandas as pd
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "results" / "synthetic"
DEFAULT_OUTPUT_DIR  = _PROJECT_ROOT / "results" / "synthetic" / "method_analysis"

from src.igloo.ig_metrics import _KNOWN_PROTEINS, NEAR_ZERO_THRESHOLD  # noqa: E402
from src.igloo.method_selection import (                                 # noqa: E402
    METRIC_KEYS,
    METRIC_LABELS,
    load_all_metrics,
    add_composite_score,
    build_auc_matrix,
    run_friedman,
    kendalls_w,
    nemenyi_posthoc,
    fit_noise_decay,
    run_decision_tree,
    plot_critical_difference,
    plot_composite_noise_curves,
    plot_auc_heatmap,
    plot_peak_vs_decay,
    plot_metric_breakdown,
    print_summary,
    save_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Method-scenario interaction analysis for IGLOO IG attributions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results-dir",  default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir",   default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--proteins",     nargs="+", default=None, choices=_KNOWN_PROTEINS)
    parser.add_argument("--fitness-models", nargs="+", default=None,
                        choices=["additive", "epistatic", "nonlinear"])
    parser.add_argument("--concordance-threshold", type=float, default=0.6)
    parser.add_argument("--min-output-diff", type=float, default=NEAR_ZERO_THRESHOLD)
    parser.add_argument("--min-count",    type=int, default=5)
    parser.add_argument("--alpha",        type=float, default=0.05)
    parser.add_argument("--dpi",          type=int, default=150)
    parser.add_argument("--split",        default=None, choices=["trainval", "test"])
    parser.add_argument("--no-decay",     action="store_true", default=False)
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_root.exists():
        print(f"[ERROR] Results directory not found: {results_root}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("IGLOO Method-Scenario Interaction Analysis")
    print(f"  Metrics : {', '.join(METRIC_LABELS[k] for k in METRIC_KEYS)}")
    print(f"  Results : {results_root}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*60}\n")

    print("Step 1: Loading metrics...")
    df = load_all_metrics(
        results_root, proteins=args.proteins, fitness_models=args.fitness_models,
        min_output_diff=args.min_output_diff, min_count=args.min_count,
        split_filter=args.split,
    )

    print("\nStep 2: Computing composite scores...")
    df = add_composite_score(df)

    print("\nStep 3: AUC over noise...")
    auc_matrix = build_auc_matrix(df)
    print(auc_matrix.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nStep 4: Friedman test...")
    friedman_res = run_friedman(auc_matrix, alpha=args.alpha)
    print(f"  χ²={friedman_res['stat']:.3f}  p={friedman_res['p']:.4f}  "
          f"({'SIGNIFICANT' if friedman_res['significant'] else 'not significant'})")

    print("\nStep 5: Kendall's W...")
    W = kendalls_w(auc_matrix)
    print(f"  W = {W:.3f}  (threshold = {args.concordance_threshold})")
    stable = W >= args.concordance_threshold

    nemenyi_pvals = None
    tree_rules    = None

    if stable or not friedman_res["significant"]:
        print("  → Rankings consistent → Nemenyi post-hoc + CD diagram")
        nemenyi_pvals = nemenyi_posthoc(auc_matrix)
        if nemenyi_pvals is not None:
            print("  Nemenyi p-values:")
            print(nemenyi_pvals.to_string(float_format=lambda v: f"{v:.3f}"))
        plot_critical_difference(
            auc_matrix, nemenyi_pvals,
            output_dir / "critical_difference_diagram.png",
            alpha=args.alpha, dpi=args.dpi,
        )
    else:
        print("  → Rankings unstable → decision tree on scenario features")
        tree_rules = run_decision_tree(
            auc_matrix, output_dir / "decision_tree_importances.png", dpi=args.dpi,
        )
        if tree_rules:
            print("\n  Decision tree rules:")
            print(tree_rules)

    decay_df = pd.DataFrame()
    if not args.no_decay:
        print("\nStep 6: Noise-decay characterisation...")
        decay_df = fit_noise_decay(df)
        print(
            decay_df[["method", "scenario", "peak", "decay_rate", "floor", "fit_r2"]]
            .sort_values(["scenario", "peak"], ascending=[True, False])
            .to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )

    print("\nGenerating figures...")
    plot_composite_noise_curves(df,         output_dir / "composite_noise_curves.png", args.dpi)
    plot_auc_heatmap(auc_matrix,            output_dir / "auc_heatmap.png",            args.dpi)
    plot_metric_breakdown(df,               output_dir / "metric_breakdown.png",        args.dpi)
    if not decay_df.empty:
        plot_peak_vs_decay(decay_df,        output_dir / "peak_vs_decay.png",          args.dpi)

    print_summary(auc_matrix, friedman_res, W, decay_df, args.concordance_threshold)
    save_report(
        output_dir, auc_matrix, friedman_res, W, decay_df,
        nemenyi_pvals, tree_rules, args.concordance_threshold,
        raw_df=df,
    )
    print(f"\nDone. All outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
