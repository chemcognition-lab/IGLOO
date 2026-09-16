#!/usr/bin/env python3
"""
Recompute E15 metrics from saved attribution vectors.
Much faster than rerunning the full pipeline - just loads attributions and computes stats.

Usage:
  python scripts/recompute_E15_metrics.py
"""

import pickle
import csv
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
import numpy as np

results_dir = Path("results/revision/E15_crosspLM")
out_file = results_dir / "metrics_long.csv"

# Find all saved attribution files
attr_files = list(results_dir.glob("attributions_*.pkl"))
print(f"Found {len(attr_files)} attribution files to process")

results = []

for attr_file in sorted(attr_files):
    print(f"Processing: {attr_file.name}")

    with open(attr_file, 'rb') as f:
        data = pickle.load(f)

    all_attributions_residue = data['all_attributions_residue']  # Dict: method -> (N, L) array
    ground_truth_w = data['ground_truth_w']  # (L,) array
    methods = data['methods']
    noise_levels = data['noise_levels']

    # Parse filename: attributions_{protein}_{landscape}_{plm}_seed{seed}.pkl
    parts = attr_file.stem.replace('attributions_', '').split('_')
    protein = parts[0]
    landscape = parts[1]
    plm = parts[2]
    seed = int(parts[3].replace('seed', ''))

    # Compute metrics per method and noise level
    for method in methods:
        if method not in all_attributions_residue:
            print(f"  Skipping {method} (not in attributions)")
            continue

        method_attrs = all_attributions_residue[method]  # (N, L) - per sequence

        # Compute metrics per test sequence, then average
        for noise_level in noise_levels:
            seq_metrics = []

            for seq_idx in range(len(method_attrs)):
                attr_trimmed = method_attrs[seq_idx, :len(ground_truth_w)]

                # Spearman correlation
                spearman_r, _ = spearmanr(attr_trimmed, ground_truth_w)

                # Kendall tau
                kendall_tau, _ = kendalltau(attr_trimmed, ground_truth_w)

                seq_metrics.append({
                    'spearman_corr': float(spearman_r) if not np.isnan(spearman_r) else 0.0,
                    'kendall_tau': float(kendall_tau) if not np.isnan(kendall_tau) else 0.0,
                })

            # Average across sequences
            for metric_name in ['spearman_corr', 'kendall_tau']:
                values = [m[metric_name] for m in seq_metrics if metric_name in m]
                if values:
                    metric_value = np.mean(values)
                else:
                    metric_value = None

                if metric_value is not None:
                    results.append({
                        'protein': protein,
                        'landscape': landscape,
                        'plm': plm,
                        'method': method,
                        'metric': metric_name,
                        'noise_level': noise_level,
                        'seed': seed,
                        'value': metric_value,
                    })

print(f"\nGenerated {len(results)} metric rows")

# Write CSV
with open(out_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['protein', 'landscape', 'plm', 'method', 'metric', 'noise_level', 'seed', 'value'])
    writer.writeheader()
    writer.writerows(results)

print(f"✓ Metrics saved to: {out_file}")
print(f"\nSummary:")
print(f"  Proteins: {set(r['protein'] for r in results)}")
print(f"  Landscapes: {set(r['landscape'] for r in results)}")
print(f"  Methods: {set(r['method'] for r in results)}")
print(f"  Metrics: {set(r['metric'] for r in results)}")
print(f"  Noise levels: {sorted(set(r['noise_level'] for r in results))}")
