#!/usr/bin/env python3
"""
IGLOO Attribution Quality Plotting Script
==========================================
Generates noise-vs-metric figures for IG attribution quality across
interpolation methods, noise levels, proteins, and fitness models.

Mirrors the directory structure expected by evaluate_synthetic_ig_robust.py:

    results/synthetic/
        {Protein}_{fitness_model}/          e.g. GB1_additive/
            noise_{level}/                  e.g. noise_0.0/
                {interp_method}/            e.g. linear/
                    *_integrated_gradients_results.csv

One figure is produced per (protein × fitness_model) combination showing:
  Row 0  – Headline metrics: Global Spearman ρ, AUROC, AUPRC, Precision@K
  Row 1  – Per-variant Spearman ρ (mean±std), Sign accuracy,
             Category separation ratio, Model collapse fraction
  Row 2  – Category mean |IG| per interp method (3 subplots) +
             Δ Spearman vs linear
  Row 3  – Per-variant ρ violin plots per interp method + metric heatmap

Usage
-----
    # Auto-discover all results under results/synthetic/
    python scripts/plot_ig_quality.py

    # Filter by protein and/or fitness model
    python scripts/plot_ig_quality.py --proteins GB1 --fitness-models additive epistatic

    # Specify a custom results root and output directory
    python scripts/plot_ig_quality.py \\
        --results-dir /path/to/results/synthetic \\
        --output-dir  /path/to/output/figures

    # Combine filters
    python scripts/plot_ig_quality.py \\
        --proteins GB1 T4Lysozyme \\
        --fitness-models additive nonlinear \\
        --output-dir results/synthetic/figures

Flag reference
--------------
    --results-dir PATH      Root of synthetic results (default: results/synthetic/)
    --output-dir  PATH      Where to write PNG figures   (default: results/synthetic/figures/)
    --proteins    [...]     Restrict to specific proteins  (GB1, T4Lysozyme, p53)
    --fitness-models [...]  Restrict to specific models    (additive, epistatic, nonlinear)
    --min-output-diff FLOAT Threshold for collapsed sequences (default: 0.01)
    --min-count INT         Min sequences per position for reliability mask (default: 5)
    --dpi INT               Figure DPI (default: 150)
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "results" / "synthetic"
DEFAULT_OUTPUT_DIR  = _PROJECT_ROOT / "results" / "synthetic" / "figures"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Metric + plotting functions from library modules
# ---------------------------------------------------------------------------
from src.igloo.ig_metrics import (  # noqa: E402
    BACKGROUND_IMPORTANCE,
    NEAR_ZERO_THRESHOLD,
    _KNOWN_PROTEINS,
    _PROTEIN_DEFS,
    _FIT_FNS,
    _load_gt_json,
    _build_importance_vec,
    _wt_sequence,
    discover_result_dirs,
    discover_result_dirs_multiseed,
    _load_ig_csv,
    _is_wt,
    _ndcg_1d,
    _ap_graded,
    compute_metrics_for_dir,
    _agg_metrics,
)
from src.igloo.plot import (  # noqa: E402
    FIG_BG,
    INTERP_COLORS,
    INTERP_LABELS,
    INTERP_MARKERS,
    CAT_COLORS,
    CAT_LABELS,
    _style_ax,
    _plot_metric_line,
    _interp_legend,
    make_figure,
)


# ---------------------------------------------------------------------------
# Parallel helper (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _compute_one(task: tuple):
    """Called by ProcessPoolExecutor; returns (key_buf, metrics) or (key_buf, None)."""
    protein, fm, noise, seed, interp, path, split, min_output_diff, min_count = task
    metrics = compute_metrics_for_dir(
        path, protein,
        min_output_diff=min_output_diff,
        min_count=min_count,
        fitness_model=fm,
        split_filter=split,
    )
    return (protein, fm, noise, seed, interp, split), metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate IGLOO attribution quality figures across noise levels "
            "and interpolation methods."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Root of synthetic results tree (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for output PNG figures (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--proteins",
        nargs="+",
        default=None,
        choices=_KNOWN_PROTEINS,
        help="Restrict to specific proteins (default: all with annotations defined).",
    )
    parser.add_argument(
        "--fitness-models",
        nargs="+",
        default=None,
        choices=["additive", "epistatic", "nonlinear"],
        help="Restrict to specific fitness models (default: all).",
    )
    parser.add_argument(
        "--min-output-diff",
        type=float,
        default=NEAR_ZERO_THRESHOLD,
        help=f"Threshold for collapsed sequences (default: {NEAR_ZERO_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Min sequences per position for reliability mask (default: 5).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output figure DPI (default: 150).",
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        default=False,
        help=(
            "Treat results-dir as containing seed_N sub-directories. "
            "Metrics are averaged across seeds and ±1σ error bands are drawn. "
            "Expected layout: {root}/{Protein}_{fm}/noise_{level}/{interp}/"
        ),
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    output_dir   = Path(args.output_dir)

    if not results_root.exists():
        print(f"[ERROR] Results directory does not exist: {results_root}")
        sys.exit(1)

    print(f"Scanning: {results_root}")

    # ── Multi-seed path ──────────────────────────────────────────────────────
    if args.multi_seed:
        ms_entries = discover_result_dirs_multiseed(
            results_root,
            proteins=args.proteins,
            fitness_models=args.fitness_models,
        )
        if not ms_entries:
            print("No multi-seed result directories found. Check your --results-dir and filters.")
            sys.exit(0)
        print(f"Found {len(ms_entries)} leaf directories across seeds.\n")

        SPLITS: list[str | None] = [None, "trainval", "test"]
        # groups[split][(protein,fm)][interp][noise] = mean metrics dict
        groups: dict[str | None, dict[tuple[str,str], dict[str, dict[float, dict]]]] = {s: {} for s in SPLITS}
        # errors[split][(protein,fm)][interp][noise] = std metrics dict
        seed_errors: dict[str | None, dict[tuple[str,str], dict[str, dict[float, dict]]]] = {s: {} for s in SPLITS}
        # buffer: (protein,fm,split,interp,noise) -> list[metrics]
        _buf: dict[tuple, list[dict]] = {}

        # Build task list for parallel execution
        tasks = [
            (protein, fm, noise, seed, interp, path, split,
             args.min_output_diff, args.min_count)
            for protein, fm, noise, seed, interp, path in ms_entries
            for split in SPLITS
        ]
        n_workers = min(os.cpu_count() or 4, 16)
        print(f"  [parallel: {len(tasks)} tasks, {n_workers} workers]")
        _seen: dict[tuple, bool] = {}  # (protein,fm,noise,seed,interp) -> any_ok
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_compute_one, t): t for t in tasks}
            for fut in as_completed(futs):
                (protein, fm, noise, seed, interp, split), metrics = fut.result()
                entry_key = (protein, fm, noise, seed, interp)
                if metrics is not None:
                    key_buf = (protein, fm, split, interp, noise)
                    _buf.setdefault(key_buf, []).append(metrics)
                    _seen[entry_key] = True
                else:
                    _seen.setdefault(entry_key, False)
        for (protein, fm, noise, seed, interp), ok in sorted(_seen.items()):
            status = "ok" if ok else "SKIP (no data)"
            print(f"  {protein}/{fm}/noise={noise}/{seed}/{interp} ... {status}")

        # Aggregate across seeds
        print("\nAggregating across seeds...")
        for (protein, fm, split, interp, noise), mlist in _buf.items():
            key = (protein, fm)
            groups[split].setdefault(key, {}).setdefault(interp, {})
            seed_errors[split].setdefault(key, {}).setdefault(interp, {})
            mean_m, std_m = _agg_metrics(mlist)
            groups[split][key][interp][noise]      = mean_m
            seed_errors[split][key][interp][noise] = std_m
            if split is None:
                n_seeds = len(mlist)
                sp = mean_m.get("spearman", float("nan"))
                sp_std = std_m.get("spearman", float("nan"))
                print(f"  {protein}/{fm}/noise={noise}/{interp}: "
                      f"n_seeds={n_seeds}  sp={sp:+.3f}±{sp_std:.3f}")

    # ── Single-seed path ─────────────────────────────────────────────────────
    else:
        seed_errors = {s: {} for s in [None, "trainval", "test"]}
        entries = discover_result_dirs(
            results_root,
            proteins=args.proteins,
            fitness_models=args.fitness_models,
        )
        if not entries:
            print("No result directories found. Check your --results-dir and filters.")
            sys.exit(0)

        print(f"Found {len(entries)} result directories.\n")

        SPLITS: list[str | None] = [None, "trainval", "test"]
        groups: dict[
            str | None,
            dict[tuple[str, str], dict[str, dict[float, dict]]]
        ] = {s: {} for s in SPLITS}

        for protein, fm, noise, interp, path in entries:
            key = (protein, fm)
            for split in SPLITS:
                groups[split].setdefault(key, {}).setdefault(interp, {})

            print(f"  Computing: {protein}/{fm}/noise={noise}/{interp} ... ", end="", flush=True)
            _resolved_fn = _FIT_FNS.get(fm)
            _fn_name = getattr(_resolved_fn, "__name__", "additive-approx(fallback)")
            if _resolved_fn is None:
                _fn_name = "additive-approx(fallback — fitness_model not in _FIT_FNS)"
            print(f"[LOO fn: {_fn_name}] ", end="", flush=True)
            any_computed = False
            for split in SPLITS:
                metrics = compute_metrics_for_dir(
                    path, protein,
                    min_output_diff=args.min_output_diff,
                    min_count=args.min_count,
                    fitness_model=fm,
                    split_filter=split,
                )
                if metrics is not None:
                    groups[split][key][interp][noise] = metrics
                    any_computed = True
            if any_computed:
                m = groups[None].get(key, {}).get(interp, {}).get(noise)
                if m:
                    print(
                        f"sp={m['spearman']:+.3f}  "
                        f"auroc={m['auroc']:.3f}  "
                        f"pv={m['pv_mean']:+.3f}  "
                        f"sign={m['sign_acc']:.3f}  "
                        f"sep={m['cat_sep']:.2f}  "
                        f"collapse={m['collapse_frac']:.3f}  "
                        f"n_reliable={m.get('n_reliable', float('nan'))}"
                    )
                else:
                    print("computed (split-only)")
            else:
                print("SKIP (no data)")

    if not any(groups[s] for s in SPLITS):
        print("\nNo valid metrics computed.")
        sys.exit(0)

    # Generate figures — one per (protein, fitness_model, split)
    print(f"\nGenerating figures -> {output_dir}")
    split_labels: dict[str | None, str] = {None: "all", "trainval": "trainval", "test": "test"}
    for split in SPLITS:
        split_groups = groups[split]
        if not split_groups:
            continue
        for (protein, fm), interp_data in sorted(split_groups.items()):
            # Skip split-specific figures if no sequences carry that label
            if not any(
                nd for nd in interp_data.values() if nd
            ):
                continue
            lbl = split_labels[split]
            print(f"\n  Figure [{lbl}]: {protein} / {fm}")
            _errors_for_pfm = seed_errors.get(split, {}).get((protein, fm)) if args.multi_seed else None
            for _mfam in ("spearman", "ndcg"):
                make_figure(
                    protein=protein,
                    fitness_model=fm,
                    data=interp_data,
                    output_dir=output_dir,
                    dpi=args.dpi,
                    split_label=split,
                    errors=_errors_for_pfm,
                    metric_family=_mfam,
                )

    # ── Save collapse stats CSV ──────────────────────────────────────────────
    import pandas as _pd
    collapse_rows = []
    for split in SPLITS:
        for (protein, fm), interp_data in groups[split].items():
            for interp, noise_data in interp_data.items():
                for noise, m in noise_data.items():
                    if m and "collapse_frac" in m:
                        collapse_rows.append({
                            "split":        split if split is not None else "all",
                            "protein":      protein,
                            "fitness_model": fm,
                            "noise":        noise,
                            "method":       interp,
                            "collapse_frac": m["collapse_frac"],
                            "n_reliable":   m.get("n_reliable", float("nan")),
                        })
    if collapse_rows:
        cstats = _pd.DataFrame(collapse_rows).drop_duplicates()
        cstats = cstats.sort_values(["split", "protein", "fitness_model", "noise", "method"])
        csv_path = output_dir / "collapse_stats.csv"
        cstats.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    print(f"\nDone. Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()