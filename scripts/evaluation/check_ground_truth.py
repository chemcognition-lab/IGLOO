#!/usr/bin/env python3
"""
Quick ground-truth check for one synthetic run.

Compares a run's per-position importance (``*_position_importance_summary.csv``)
with the true position weights in ``data/synthetic/<protein>_ground_truth_weights.json``
and prints it next to the model's test-set Spearman. Intended as a fast,
human-readable sanity check (and the pass/fail step in docs/REVIEWER_TEST.md);
use metric_suite.py for the full analysis.

Usage:
    python scripts/evaluation/check_ground_truth.py RESULTS_DIR [RESULTS_DIR ...]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GT_DIR = PROJECT_ROOT / "data" / "synthetic"


def check(results_dir: Path) -> dict:
    summary = next(results_dir.glob("*_position_importance_summary.csv"), None)
    metrics = next(results_dir.glob("*_test_metrics.csv"), None)
    if summary is None or metrics is None:
        raise FileNotFoundError(f"{results_dir}: missing position_importance_summary or test_metrics CSV")

    dataset = summary.name.removesuffix("_position_importance_summary.csv")
    protein = dataset.split("_")[0]
    gt_file = GT_DIR / f"{protein}_ground_truth_weights.json"
    if not gt_file.exists():
        raise FileNotFoundError(f"{results_dir}: no ground truth for protein '{protein}' ({gt_file})")
    weights = pd.Series(json.loads(gt_file.read_text())["importance_vector"])

    imp = pd.read_csv(summary)
    # position column is 1-based; the ground-truth vector is 0-based
    attr_rho = spearmanr(imp["importance_score"], weights[imp["position"] - 1].values).correlation
    top10 = imp.nlargest(10, "importance_score")["position"] - 1
    top10_hits = float((weights[top10] >= 1.0).mean())
    model_rho = float(pd.read_csv(metrics)["spearman_rho"].iloc[0])
    return {"run": str(results_dir), "dataset": dataset, "model_rho": model_rho,
            "attr_rho": attr_rho, "top10_hits": top10_hits}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dirs", nargs="+", type=Path,
                    help="results_dir of one or more synthetic runs")
    args = ap.parse_args()

    rows = [check(d) for d in args.results_dirs]
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nmodel_rho  = Spearman(prediction, true fitness) on the held-out test set")
    print("attr_rho   = Spearman(per-position importance, ground-truth position weight)")
    print("top10_hits = fraction of the 10 most-attributed positions that are truly important (weight >= 1)")


if __name__ == "__main__":
    main()
