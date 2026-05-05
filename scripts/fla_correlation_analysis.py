"""
FLA Correlation Analysis
========================
Computes GraphFLA landscape features for the 9 synthetic protein datasets
(GB1/T4Lysozyme/p53 × additive/epistatic/nonlinear) and correlates them with
the AUC-over-noise composite scores from the heatmap.

Output:
  results/synthetic/fla_analysis/
    fla_features.csv          – FLA metrics per scenario
    fla_auc_correlation.csv   – Spearman rho per (method, FLA metric)
    fla_correlation_heatmap.png
    fla_category_correlation_heatmap.png
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

import sys
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.igloo.plot import plot_fla_correlation_heatmap, plot_fla_category_heatmap  # noqa: E402

# ── GraphFLA imports ──────────────────────────────────────────────────────────
from graphfla.landscape import ProteinLandscape
from graphfla.analysis import (
    lo_ratio,
    r_s_ratio,
    autocorrelation,
    gradient_intensity,
    neighbor_fit_corr,
    neutrality,
    fitness_distance_corr,
    fitness_distribution,
    classify_epistasis,
    higher_order_epistasis,
    diminishing_returns_index,
    increasing_costs_index,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO   = Path(__file__).resolve().parents[1]
DATA   = REPO / "data" / "synthetic"
AUC_CSV = REPO / "results" / "synthetic" / "method_analysis" / "auc_matrix.csv"
OUT    = REPO / "results" / "synthetic" / "fla_analysis"
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "GB1_additive", "GB1_epistatic", "GB1_nonlinear",
    "T4Lysozyme_additive", "T4Lysozyme_epistatic", "T4Lysozyme_nonlinear",
    "p53_additive", "p53_epistatic", "p53_nonlinear",
]

METHODS = ["gradient_x_input", "input_difference", "linear", "nearest_neighbor", "slerp"]

METHOD_LABELS = {
    "gradient_x_input":  "Gradient × input",
    "input_difference":  "Input difference",
    "linear":            "Linear",
    "nearest_neighbor":  "Nearest neighbor",
    "slerp":             "Slerp",
}

# FLA metric categories for grouping
METRIC_CATEGORIES = {
    "lo_ratio":                  "Ruggedness",
    "r_s_ratio":                 "Ruggedness",
    "autocorrelation":           "Ruggedness",
    "gradient_intensity":        "Ruggedness",
    "neighbor_fit_corr":         "Ruggedness",
    "neutrality":                "Neutrality",
    "fitness_distance_corr":     "Navigability",
    "fitness_skewness":          "Fitness Distribution",
    "fitness_kurtosis":          "Fitness Distribution",
    "fitness_cv":                "Fitness Distribution",
    "sign_epistasis":            "Epistasis",
    "reciprocal_sign_epistasis": "Epistasis",
    "higher_order_epistasis":    "Epistasis",
    "diminishing_returns":       "Epistasis",
    "increasing_costs":          "Epistasis",
}


# ── Helper: compute FLA metrics for one scenario ──────────────────────────────
def compute_fla_metrics(scenario: str) -> dict:
    csv_path = DATA / f"{scenario}.csv"
    df = pd.read_csv(csv_path)

    X = df["SEQ"]
    f = df["fitness"]

    print(f"  Building landscape for {scenario} ({len(df)} variants)...")
    landscape = ProteinLandscape(maximize=True)
    landscape.build_from_data(X, f, verbose=False)

    metrics: dict[str, float] = {"scenario": scenario}

    print(f"  Computing metrics...")

    # Ruggedness
    try:
        metrics["lo_ratio"] = float(lo_ratio(landscape))
    except Exception as e:
        print(f"    lo_ratio failed: {e}")
        metrics["lo_ratio"] = np.nan

    try:
        metrics["r_s_ratio"] = float(r_s_ratio(landscape))
    except Exception as e:
        print(f"    r_s_ratio failed: {e}")
        metrics["r_s_ratio"] = np.nan

    try:
        metrics["autocorrelation"] = float(autocorrelation(landscape))
    except Exception as e:
        print(f"    autocorrelation failed: {e}")
        metrics["autocorrelation"] = np.nan

    try:
        metrics["gradient_intensity"] = float(gradient_intensity(landscape))
    except Exception as e:
        print(f"    gradient_intensity failed: {e}")
        metrics["gradient_intensity"] = np.nan

    try:
        metrics["neighbor_fit_corr"] = float(neighbor_fit_corr(landscape))
    except Exception as e:
        print(f"    neighbor_fit_corr failed: {e}")
        metrics["neighbor_fit_corr"] = np.nan

    # Neutrality
    try:
        metrics["neutrality"] = float(neutrality(landscape))
    except Exception as e:
        print(f"    neutrality failed: {e}")
        metrics["neutrality"] = np.nan

    # Navigability
    try:
        metrics["fitness_distance_corr"] = float(fitness_distance_corr(landscape))
    except Exception as e:
        print(f"    fitness_distance_corr failed: {e}")
        metrics["fitness_distance_corr"] = np.nan

    # Fitness distribution
    try:
        fd = fitness_distribution(landscape)
        metrics["fitness_skewness"] = float(fd.get("skewness", np.nan))
        metrics["fitness_kurtosis"] = float(fd.get("kurtosis", np.nan))
        metrics["fitness_cv"]       = float(fd.get("cv", np.nan))
    except Exception as e:
        print(f"    fitness_distribution failed: {e}")
        metrics["fitness_skewness"] = np.nan
        metrics["fitness_kurtosis"] = np.nan
        metrics["fitness_cv"]       = np.nan

    # Epistasis
    try:
        epi = classify_epistasis(landscape)
        metrics["sign_epistasis"]            = float(epi.get("sign epistasis", np.nan))
        metrics["reciprocal_sign_epistasis"] = float(epi.get("reciprocal sign epistasis", np.nan))
    except Exception as e:
        print(f"    classify_epistasis failed: {e}")
        metrics["sign_epistasis"]            = np.nan
        metrics["reciprocal_sign_epistasis"] = np.nan

    try:
        metrics["higher_order_epistasis"] = float(higher_order_epistasis(landscape))
    except Exception as e:
        print(f"    higher_order_epistasis failed: {e}")
        metrics["higher_order_epistasis"] = np.nan

    try:
        metrics["diminishing_returns"] = float(diminishing_returns_index(landscape))
    except Exception as e:
        print(f"    diminishing_returns failed: {e}")
        metrics["diminishing_returns"] = np.nan

    try:
        metrics["increasing_costs"] = float(increasing_costs_index(landscape))
    except Exception as e:
        print(f"    increasing_costs failed: {e}")
        metrics["increasing_costs"] = np.nan

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", type=str, default=None,
                        help="Only process scenarios for this protein (e.g. GB1, T4Lysozyme, p53)")
    args = parser.parse_args()

    scenarios = (
        [s for s in SCENARIOS if s.startswith(args.protein + "_")]
        if args.protein else SCENARIOS
    )

    # 1. Compute FLA metrics with per-scenario caching
    fla_cache = OUT / "fla_features.csv"
    scenario_cache_dir = OUT / "scenario_cache"
    scenario_cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if full cache exists and all scenarios are present
    if fla_cache.exists():
        _existing = pd.read_csv(fla_cache, index_col="scenario")
        if set(scenarios).issubset(set(_existing.index)):
            print(f"Loading cached FLA features from {fla_cache}")
            fla_df = _existing
        else:
            # Partial cache — reuse per-scenario files
            print("Full cache incomplete — reusing per-scenario cache files...")
            fla_df = None
    else:
        fla_df = None

    if fla_df is None:
        rows = []
        for scenario in scenarios:
            per_cache = scenario_cache_dir / f"{scenario}.json"
            if per_cache.exists():
                print(f"  [{scenario}] loading from cache")
                with open(per_cache) as _f:
                    row = json.load(_f)
            else:
                print(f"\n[{scenario}]")
                row = compute_fla_metrics(scenario)
                with open(per_cache, "w") as _f:
                    json.dump({k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                                for k, v in row.items()}, _f, indent=2)
                print(f"  Saved per-scenario cache -> {per_cache}")
            rows.append(row)
        fla_df = pd.DataFrame(rows).set_index("scenario")
        # Merge with any existing cached rows from other proteins
        if fla_cache.exists():
            _existing = pd.read_csv(fla_cache, index_col="scenario")
            fla_df = pd.concat([_existing[~_existing.index.isin(fla_df.index)], fla_df])
        if set(SCENARIOS).issubset(set(fla_df.index)):
            fla_df.to_csv(fla_cache)
            print(f"\nSaved full FLA features to {fla_cache}")
        else:
            print(f"\nPer-scenario caches saved; full CSV will be written when all proteins complete.")

    print("\nFLA features:")
    print(fla_df.to_string())

    # 2. Load AUC matrix
    auc_df = pd.read_csv(AUC_CSV, index_col="scenario")
    print(f"\nAUC matrix ({auc_df.shape[0]} scenarios × {auc_df.shape[1]} methods)")

    # 3. Compute Spearman correlations: for each method, correlate AUC with each FLA metric
    # Filter to scenarios present in both dataframes
    shared_scenarios = sorted(set(fla_df.index) & set(auc_df.index))
    print(f"\nShared scenarios: {shared_scenarios}")

    fla_metrics = [c for c in fla_df.columns if c in METRIC_CATEGORIES]
    corr_rows = []
    pval_rows = []

    for method in METHODS:
        if method not in auc_df.columns:
            continue
        corr_row = {}
        pval_row = {}
        for metric in fla_metrics:
            # Get AUC and FLA values for scenarios where both exist
            pairs = [
                (auc_df.loc[s, method], fla_df.loc[s, metric])
                for s in shared_scenarios
                if s in auc_df.index and s in fla_df.index
                   and not pd.isna(auc_df.loc[s, method])
                   and not pd.isna(fla_df.loc[s, metric])
            ]
            if len(pairs) < 4:
                corr_row[metric] = np.nan
                pval_row[metric] = np.nan
            else:
                aucs, flas = zip(*pairs)
                rho, pval = stats.spearmanr(aucs, flas)
                corr_row[metric] = rho
                pval_row[metric] = pval
        corr_rows.append({"method": method, **corr_row})
        pval_rows.append({"method": method, **pval_row})

    corr_df = pd.DataFrame(corr_rows).set_index("method")
    pval_df = pd.DataFrame(pval_rows).set_index("method")

    corr_df.to_csv(OUT / "fla_auc_correlation.csv")
    pval_df.to_csv(OUT / "fla_auc_pvalues.csv")
    print("\nSpearman correlations (ρ) — FLA metrics vs AUC over noise:")
    print(corr_df.to_string(float_format=lambda x: f"{x:.3f}"))

    # 4. Plots
    print("\nGenerating plots...")
    plot_fla_correlation_heatmap(
        corr_df,
        "Spearman ρ: FLA metric vs AUC-over-noise (per method)",
        OUT / "fla_correlation_heatmap.png",
        method_labels=METHOD_LABELS,
    )

    plot_fla_category_heatmap(
        corr_df,
        "Spearman ρ: FLA category (avg) vs AUC-over-noise (per method)",
        OUT / "fla_category_correlation_heatmap.png",
        metric_categories=METRIC_CATEGORIES,
        method_labels=METHOD_LABELS,
    )

    # 5. Print summary: top correlated FLA metrics overall
    mean_abs_rho = corr_df.abs().mean(axis=0).sort_values(ascending=False)
    print("\nTop FLA metrics by mean |ρ| across all methods:")
    print(mean_abs_rho.to_string(float_format=lambda x: f"{x:.3f}"))

    # Save summary JSON
    summary = {
        "top_metrics": mean_abs_rho.head(5).to_dict(),
        "per_method_top_metric": {
            m: corr_df.loc[m].abs().idxmax()
            for m in corr_df.index
            if not corr_df.loc[m].isna().all()
        },
    }
    with open(OUT / "fla_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
