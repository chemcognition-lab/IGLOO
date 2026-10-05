#!/usr/bin/env python3
"""
Simple analysis of E15 results using only standard library.
Generates summary statistics and an HTML report.
"""

import csv
from pathlib import Path
from collections import defaultdict
import statistics

results_dir = Path("results/revision/E15_crosspLM")
metrics_file = results_dir / "metrics_long.csv"
output_html = results_dir / "E15_analysis_report.html"

# Load data
print("Loading metrics...")
data = []
with open(metrics_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            row['value'] = float(row['value'])
            row['noise_level'] = float(row['noise_level'])
            row['seed'] = float(row['seed'])
            data.append(row)
        except (ValueError, KeyError):
            continue

print(f"Loaded {len(data)} metric rows")

# Extract unique values
methods = sorted(set(r['method'] for r in data))
metrics = sorted(set(r['metric'] for r in data))
landscapes = sorted(set(r['landscape'] for r in data))
noise_levels = sorted(set(r['noise_level'] for r in data))

print("Methods:", methods)
print("Landscapes:", landscapes)
print("Noise levels:", noise_levels)
print("Metrics:", len(metrics), "total")

# Key metrics for summary
key_metrics = ['ndcg', 'ap_r', 'auroc', 'sign_accuracy', 'spearman_corr']

# Compute statistics
print("\nComputing statistics...")

# Method performance at zero noise
method_perf = {}
for method in methods:
    method_perf[method] = {}
    for metric in key_metrics:
        values = [r['value'] for r in data
                 if r['method'] == method and r['metric'] == metric and r['noise_level'] == 0.0]
        if values:
            method_perf[method][metric] = {
                'mean': statistics.mean(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0
            }

# Noise robustness
robustness = {}
for method in methods:
    robustness[method] = {}
    for nl in [0.0, 1.0]:
        values = [r['value'] for r in data
                 if r['method'] == method and r['metric'] == 'spearman_corr' and r['noise_level'] == nl]
        if values:
            robustness[method][nl] = statistics.mean(values)

# Landscape comparison
landscape_perf = {}
for landscape in landscapes:
    values = [r['value'] for r in data
             if r['landscape'] == landscape and r['metric'] == 'ndcg']
    if values:
        landscape_perf[landscape] = statistics.mean(values)

print("Generating HTML report...")

# Build HTML as string parts to avoid formatting issues
html_parts = [
    "<!DOCTYPE html>",
    "<html>",
    "<head>",
    "<title>E15 Phase 1 Analysis Report</title>",
    "<style>",
    "body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }",
    "h1 { color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }",
    "h2 { color: #0066cc; margin-top: 30px; }",
    "table { border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
    "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
    "th { background-color: #0066cc; color: white; font-weight: bold; }",
    "tr:hover { background-color: #f9f9f9; }",
    ".summary { background-color: #f0f0f0; padding: 15px; border-left: 4px solid #0066cc; margin: 20px 0; }",
    "</style>",
    "</head>",
    "<body>",
    "<h1>E15 Phase 1: Cross-pLM Comparison Analysis</h1>",
    "<div class='summary'>",
    f"<strong>Dataset Overview:</strong><br/>",
    f"Total metric rows: {len(data)}<br/>",
    f"Methods: {len(methods)}<br/>",
    f"Landscapes: {', '.join(landscapes)}<br/>",
    f"Noise levels: {', '.join(f'{x:.1f}' for x in noise_levels)}<br/>",
    f"Total metrics: {len(metrics)}",
    "</div>",
    "",
    "<h2>Method Performance at Zero Noise</h2>",
    "<table>",
    "<tr><th>Method</th><th>NDCG</th><th>AP@R</th><th>AUROC</th><th>Sign Accuracy</th><th>Spearman</th></tr>"
]

for method in methods:
    row_parts = [f"<tr><td><strong>{method}</strong></td>"]
    for metric in key_metrics:
        if metric in method_perf.get(method, {}):
            m = method_perf[method][metric]
            row_parts.append(f"<td>{m['mean']:.4f} +/- {m['stdev']:.4f}</td>")
        else:
            row_parts.append("<td>N/A</td>")
    row_parts.append("</tr>")
    html_parts.append("".join(row_parts))

html_parts.extend([
    "</table>",
    "",
    "<h2>Noise Robustness (Spearman Correlation)</h2>",
    "<table>",
    "<tr><th>Method</th><th>Zero Noise</th><th>High Noise (1.0)</th><th>Drop</th></tr>"
])

for method in methods:
    if method in robustness:
        zero = robustness[method].get(0.0, 0)
        high = robustness[method].get(1.0, 0)
        drop = zero - high
        html_parts.append(f"<tr><td><strong>{method}</strong></td><td>{zero:.4f}</td><td>{high:.4f}</td><td>{drop:.4f}</td></tr>")

html_parts.extend([
    "</table>",
    "",
    "<h2>Landscape Comparison (Mean NDCG)</h2>",
    "<table>",
    "<tr><th>Landscape</th><th>Mean NDCG</th></tr>"
])

for landscape in landscapes:
    if landscape in landscape_perf:
        html_parts.append(f"<tr><td><strong>{landscape.capitalize()}</strong></td><td>{landscape_perf[landscape]:.4f}</td></tr>")

html_parts.extend([
    "</table>",
    "<hr/>",
    f"<p style='color: #666; font-size: 12px;'>Report generated from {metrics_file}</p>",
    "</body>",
    "</html>"
])

html = "\n".join(html_parts)

# Write HTML report
with open(output_html, 'w') as f:
    f.write(html)

print(f"✓ Report saved to: {output_html}")

# Print summary
print("\n" + "=" * 70)
print("E15 PHASE 1 RESULTS SUMMARY")
print("=" * 70)
print(f"{'Method':<30} {'NDCG (0.0)':<20} {'NDCG (1.0)':<20}")
print("-" * 70)

for method in methods:
    ndcg_zero = [r['value'] for r in data
                if r['method'] == method and r['metric'] == 'ndcg' and r['noise_level'] == 0.0]
    ndcg_high = [r['value'] for r in data
                if r['method'] == method and r['metric'] == 'ndcg' and r['noise_level'] == 1.0]

    z = f"{statistics.mean(ndcg_zero):.4f}" if ndcg_zero else "N/A"
    h = f"{statistics.mean(ndcg_high):.4f}" if ndcg_high else "N/A"

    print(f"{method:<30} {z:<20} {h:<20}")

print("=" * 70)
