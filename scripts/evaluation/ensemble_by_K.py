#!/usr/bin/env python3
"""
For each (protein, landscape, noise) in results/synthetic/seedsweep/ (5 seeds,
GB1 + T4Lysozyme), score every K=1..5 subset of seeds' ensemble-mean
attribution using the exact same metric code as metric_suite.py
(all_metrics()): average importance_score (magnitude) and mean_gradient
(signed) across the subset's seeds, then compute all 8 base metrics + their
skill scores + sign_acc/skill_sign_acc.

K=1 includes every individual seed (5 subsets) so single-model spread is
visible; K=2..5 use all C(5,K) subsets, not just cumulative ones.

Output: long format, one row per (protein, landscape, noise, K, subset_id, metric).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_suite import all_metrics, PROTEINS, LANDSCAPES  # noqa: E402

ROOT = Path("results/synthetic/seedsweep")
SEEDS = [1, 2, 3, 4, 5]


def load_seed_vectors(protein, landscape, noise, seed):
    leaf = ROOT / f"{protein}_{landscape}" / f"noise_{noise}" / f"seed_{seed}" / "linear"
    files = list(leaf.glob("*_position_importance_summary.csv"))
    if not files:
        return None
    d = pd.read_csv(files[0]).sort_values("position")
    mag = d["importance_score"].to_numpy()
    sgn = d["mean_gradient"].to_numpy() if "mean_gradient" in d else mag
    return mag, sgn


def main():
    rows = []
    proteins = [p for p in PROTEINS if p != "p53"]  # seedsweep is GB1/T4Lysozyme only
    for protein in proteins:
        for landscape in LANDSCAPES:
            for noise_dir in sorted((ROOT / f"{protein}_{landscape}").glob("noise_*")):
                noise = float(noise_dir.name.split("_", 1)[1])
                seed_vecs = {}
                for s in SEEDS:
                    v = load_seed_vectors(protein, landscape, noise, s)
                    if v is not None:
                        seed_vecs[s] = v
                if len(seed_vecs) < 1:
                    continue
                available = sorted(seed_vecs.keys())

                for K in range(1, len(available) + 1):
                    for subset in itertools.combinations(available, K):
                        mags = np.stack([seed_vecs[s][0] for s in subset])
                        sgns = np.stack([seed_vecs[s][1] for s in subset])
                        ens_mag = mags.mean(axis=0)
                        ens_sgn = sgns.mean(axis=0)
                        metrics = all_metrics(ens_mag, protein, signed=ens_sgn)
                        subset_id = "-".join(str(s) for s in subset)
                        for metric, value in metrics.items():
                            rows.append(dict(protein=protein, landscape=landscape,
                                             noise=noise, K=K, subset_id=subset_id,
                                             metric=metric, value=value))
                print(f"done {protein} {landscape} noise={noise} ({len(available)} seeds available)")

    out = pd.DataFrame(rows)
    out_path = "results/synthetic/metric_suite/ensemble_by_K.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
