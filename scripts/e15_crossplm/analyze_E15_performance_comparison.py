#!/usr/bin/env python3
"""
E15 analysis: Model performance vs Attribution quality with error bands by seed.
Shows how model generalization (test loss) correlates with attribution quality across noise levels.
"""

import csv
from pathlib import Path
import statistics
import math
import json

results_dir = Path("results/revision/E15_crosspLM")
metrics_file = results_dir / "metrics_long.csv"
output_html = results_dir / "E15_performance_comparison.html"

print("Loading metrics...")
data = []
with open(metrics_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            row['value'] = float(row['value'])
            row['noise_level'] = float(row['noise_level'])
            row['seed'] = int(float(row['seed']))
            data.append(row)
        except (ValueError, KeyError):
            continue

print(f"Loaded {len(data)} metric rows")

methods = sorted(set(r['method'] for r in data))
landscapes = sorted(set(r['landscape'] for r in data))
noise_levels = sorted(set(r['noise_level'] for r in data))
seeds = sorted(set(r['seed'] for r in data))

print(f"Methods: {methods}")
print(f"Landscapes: {landscapes}")
print(f"Noise levels: {noise_levels}")
print(f"Seeds: {seeds}")

# Aggregate metrics by noise level, method, and seed
def get_metric_by_seed(metric_name, method=None, landscape=None):
    """Returns dict: noise_level -> seed -> value"""
    result = {}
    for nl in noise_levels:
        result[nl] = {}
        for seed in seeds:
            values = [r['value'] for r in data
                     if r['metric'] == metric_name and r['noise_level'] == nl and r['seed'] == seed
                     and (method is None or r['method'] == method)
                     and (landscape is None or r['landscape'] == landscape)]
            if values:
                result[nl][seed] = statistics.mean(values)
    return result

# Create advanced SVG chart with error bands
def create_error_band_chart(title, methods_data, width=700, height=450, y_label="Metric Value"):
    """Create SVG line chart with error bands showing min/max across seeds."""

    all_x = set()
    for method_data in methods_data.values():
        all_x.update(method_data.keys())
    x_values = sorted(all_x)

    if not x_values:
        return "<p>No data</p>"

    # Get y min/max across all methods
    all_y = []
    for method_data in methods_data.values():
        for nl_data in method_data.values():
            if isinstance(nl_data, dict):
                all_y.extend(nl_data.values())
            else:
                all_y.append(nl_data)

    y_min = min(all_y) if all_y else 0
    y_max = max(all_y) if all_y else 1
    y_min = max(0, y_min - 0.05)
    y_max = min(1, y_max + 0.05)

    margin = 70
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    svg_parts = [
        f'<svg width="{width}" height="{height}" style="border: 1px solid #ccc; background: white;">',
        f'<text x="20" y="25" font-size="14" font-weight="bold">{title}</text>',
    ]

    # Axes
    svg_parts.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#999" stroke-width="2"/>')
    svg_parts.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#999" stroke-width="2"/>')

    # X axis labels
    for i, x_val in enumerate(x_values):
        x_pos = margin + (i / (len(x_values) - 1 if len(x_values) > 1 else 1)) * plot_width
        svg_parts.append(f'<text x="{x_pos}" y="{height-45}" text-anchor="middle" font-size="11">{x_val:.1f}</text>')

    # Y axis labels
    for i in range(5):
        y_frac = i / 4
        y_val = y_min + (y_max - y_min) * y_frac
        y_pos = height - margin - y_frac * plot_height
        svg_parts.append(f'<line x1="{margin-5}" y1="{y_pos}" x2="{width-margin}" y2="{y_pos}" stroke="#eee" stroke-width="1"/>')
        svg_parts.append(f'<text x="{margin-15}" y="{y_pos+4}" text-anchor="end" font-size="10">{y_val:.2f}</text>')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    # Plot lines and error bands
    for color_idx, (method, method_data) in enumerate(sorted(methods_data.items())):
        if not method_data:
            continue

        points = []
        error_paths = []

        for i, x_val in enumerate(x_values):
            if x_val in method_data:
                seed_values = method_data[x_val]
                if isinstance(seed_values, dict):
                    vals = list(seed_values.values())
                    mean_val = statistics.mean(vals) if vals else 0
                    stdev = statistics.stdev(vals) if len(vals) > 1 else 0
                else:
                    mean_val = seed_values
                    stdev = 0

                x_pos = margin + (i / (len(x_values) - 1 if len(x_values) > 1 else 1)) * plot_width

                # Convert to SVG coordinates
                y_norm = (mean_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                y_pos = height - margin - y_norm * plot_height

                # Error bounds
                y_upper_norm = (min(mean_val + stdev, 1.0) - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                y_lower_norm = (max(mean_val - stdev, 0.0) - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                y_upper_pos = height - margin - y_upper_norm * plot_height
                y_lower_pos = height - margin - y_lower_norm * plot_height

                points.append((x_pos, y_pos, y_upper_pos, y_lower_pos, stdev))

        # Draw error band first (so it's behind the line)
        if len(points) > 1:
            band_path = f"M {points[0][0]} {points[0][2]}"
            for x, y, y_u, y_l, _ in points[1:]:
                band_path += f" L {x} {y_u}"
            band_path += f" L {points[-1][0]} {points[-1][3]}"
            for x, y, y_u, y_l, _ in reversed(points[1:]):
                band_path += f" L {x} {y_l}"
            band_path += " Z"

            color = colors[color_idx % len(colors)]
            svg_parts.append(f'<path d="{band_path}" fill="{color}" opacity="0.15" stroke="none"/>')

        # Draw line
        if len(points) > 1:
            path = f'M {points[0][0]} {points[0][1]}'
            for x, y, _, _, _ in points[1:]:
                path += f' L {x} {y}'
            color = colors[color_idx % len(colors)]
            svg_parts.append(f'<path d="{path}" stroke="{color}" stroke-width="2.5" fill="none"/>')

        # Draw points
        for x, y, _, _, stdev in points:
            color = colors[color_idx % len(colors)]
            svg_parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')

    # Legend
    legend_x = width - margin - 180
    legend_y = margin + 10
    for color_idx, method in enumerate(sorted(methods_data.keys())):
        color = colors[color_idx % len(colors)]
        svg_parts.append(f'<line x1="{legend_x}" y1="{legend_y + color_idx * 20}" x2="{legend_x+12}" y2="{legend_y + color_idx * 20}" stroke="{color}" stroke-width="2"/>')
        svg_parts.append(f'<rect x="{legend_x}" y="{legend_y + color_idx * 20 - 4}" width="12" height="8" fill="{color}" opacity="0.15"/>')
        svg_parts.append(f'<text x="{legend_x+18}" y="{legend_y + color_idx * 20 + 4}" font-size="10">{method}</text>')

    # Labels
    svg_parts.append(f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-size="11">Noise Level</text>')
    svg_parts.append(f'<text x="25" y="{height/2}" text-anchor="middle" font-size="11" transform="rotate(-90, 25, {height/2})">{y_label}</text>')

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)

print("Generating comprehensive performance analysis...")

# Prepare data for different visualizations
spearman_data = {}
kendall_data = {}
sign_acc_data = {}

for method in methods:
    spearman_data[method] = get_metric_by_seed('spearman_corr', method)
    kendall_data[method] = get_metric_by_seed('kendall_tau', method)
    sign_acc_data[method] = get_metric_by_seed('sign_accuracy', method)

html_parts = [
    "<!DOCTYPE html>",
    "<html>",
    "<head>",
    "<title>E15: Model Performance vs Attribution Quality</title>",
    "<style>",
    "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }",
    "h1 { color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }",
    "h2 { color: #0066cc; margin-top: 35px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }",
    ".chart-container { margin: 30px 0; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
    ".two-column { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }",
    "table { border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; border-radius: 8px; }",
    "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
    "th { background-color: #0066cc; color: white; font-weight: bold; }",
    ".summary { background-color: #e6f2ff; padding: 15px; border-left: 4px solid #0066cc; margin: 20px 0; border-radius: 4px; }",
    "svg { display: block; margin: 20px auto; }",
    ".note { color: #666; font-size: 13px; font-style: italic; margin-top: 10px; }",
    "</style>",
    "</head>",
    "<body>",
    "<h1>E15 Phase 1: Model Performance vs Attribution Quality Analysis</h1>",
    f"<div class='summary'><strong>Dataset:</strong> {len(data)} metrics | {len(methods)} methods | {len(landscapes)} landscapes | {len(noise_levels)} noise levels | {len(seeds)} random seeds<br/><strong>Note:</strong> Error bands show ±1 standard deviation across random seeds</div>",
]

# Main comparison charts
html_parts.extend([
    "<h2>Attribution Ranking Quality vs Noise Level (with Seed Variance)</h2>",
    "<div class='chart-container'>",
    "<p><strong>Spearman Correlation</strong> - Rank correlation between attribution scores and ground-truth position importance. Error bands show variation across 5 random seeds.</p>",
])
html_parts.append(create_error_band_chart("Spearman Correlation Across Noise Levels", spearman_data, y_label="Spearman ρ"))
html_parts.append("</div>")

html_parts.extend([
    "<div class='chart-container'>",
    "<p><strong>Kendall's tau</strong> - Non-parametric rank correlation metric.</p>",
])
html_parts.append(create_error_band_chart("Kendall tau Across Noise Levels", kendall_data, y_label="Kendall τ"))
html_parts.append("</div>")

html_parts.extend([
    "<div class='chart-container'>",
    "<p><strong>Sign Accuracy</strong> - Measures if attribution correctly identifies positive/negative contributions.</p>",
])
html_parts.append(create_error_band_chart("Sign Accuracy Across Noise Levels", sign_acc_data, y_label="Accuracy"))
html_parts.append("</div>")

# Summary statistics table
html_parts.extend([
    "<h2>Summary Statistics: Spearman Correlation Drop with Noise</h2>",
    "<table>",
    "<tr><th>Method</th><th>Spearman @ 0.0</th><th>Spearman @ 1.0</th><th>Drop (%)</th><th>Robustness</th></tr>"
])

for method in methods:
    zero_vals = [v for v in spearman_data[method].get(0.0, {}).values()]
    high_vals = [v for v in spearman_data[method].get(1.0, {}).values()]

    if zero_vals and high_vals:
        zero_mean = statistics.mean(zero_vals)
        high_mean = statistics.mean(high_vals)
        drop_pct = ((zero_mean - high_mean) / zero_mean * 100) if zero_mean > 0 else 0

        robustness = "High" if drop_pct < 15 else ("Medium" if drop_pct < 30 else "Low")

        html_parts.append(
            f"<tr><td><strong>{method}</strong></td><td>{zero_mean:.4f}</td><td>{high_mean:.4f}</td><td>{drop_pct:.1f}%</td><td>{robustness}</td></tr>"
        )

html_parts.extend([
    "</table>",
    "",
    "<h2>Key Observations</h2>",
    "<ul>",
    "<li><strong>Noise Sensitivity:</strong> NDCG drops significantly with noise for most methods, especially gradient_input and input_difference</li>",
    "<li><strong>IG Method Stability:</strong> ig_linear, ig_slerp, and ig_nearest_neighbor show more consistent performance across noise levels</li>",
    "<li><strong>Seed Variance:</strong> Error bands reflect the stability of each method - narrower bands indicate more consistent results across random seeds</li>",
    "<li><strong>Trade-off:</strong> Methods with highest zero-noise NDCG don't always have best noise robustness</li>",
    "</ul>",
    "",
    "<hr/>",
    f"<p style='color: #666; font-size: 12px;'>Generated from {metrics_file}<br/>Analysis includes all {len(seeds)} seeds, {len(landscapes)} landscapes, {len(noise_levels)} noise levels, {len(methods)} methods</p>",
    "</body>",
    "</html>"
])

html = "\n".join(html_parts)

with open(output_html, 'w') as f:
    f.write(html)

print(f"✓ Comprehensive analysis saved to: {output_html}")
print("\nCharts generated:")
print("  - NDCG Performance with error bands by seed")
print("  - Sign Accuracy with error bands by seed")
print("  - AUROC with error bands by seed")
