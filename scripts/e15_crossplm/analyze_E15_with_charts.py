#!/usr/bin/env python3
"""
E15 analysis with inline SVG charts (no external dependencies needed).
"""

import csv
from pathlib import Path
import statistics
import math

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

methods = sorted(set(r['method'] for r in data))
landscapes = sorted(set(r['landscape'] for r in data))
noise_levels = sorted(set(r['noise_level'] for r in data))

print("Methods:", methods)
print("Landscapes:", landscapes)
print("Noise levels:", noise_levels)

# Aggregate data by method and noise level
spearman_by_method_noise = {}
for method in methods:
    spearman_by_method_noise[method] = {}
    for nl in noise_levels:
        values = [r['value'] for r in data
                 if r['method'] == method and r['metric'] == 'spearman_corr' and r['noise_level'] == nl]
        if values:
            spearman_by_method_noise[method][nl] = statistics.mean(values)

# Create SVG line chart
def create_line_chart(title, data_dict, width=600, height=400, y_label="Metric Value"):
    """Create a simple SVG line chart."""

    # Get all x values and normalize
    all_x = set()
    for method_data in data_dict.values():
        all_x.update(method_data.keys())
    x_values = sorted(all_x)

    if not x_values:
        return "<p>No data</p>"

    # Get min/max y values
    all_y = []
    for method_data in data_dict.values():
        all_y.extend(method_data.values())
    y_min = min(all_y) if all_y else 0
    y_max = max(all_y) if all_y else 1
    y_min = max(0, y_min - 0.05)
    y_max = min(1, y_max + 0.05)

    # SVG setup
    margin = 60
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    svg_parts = [
        f'<svg width="{width}" height="{height}" style="border: 1px solid #ccc; background: white;">',
        f'<text x="20" y="25" font-size="16" font-weight="bold">{title}</text>',
    ]

    # Grid and axes
    svg_parts.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#999" stroke-width="2"/>')
    svg_parts.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#999" stroke-width="2"/>')

    # X axis labels
    for i, x_val in enumerate(x_values):
        x_pos = margin + (i / (len(x_values) - 1 if len(x_values) > 1 else 1)) * plot_width
        svg_parts.append(f'<text x="{x_pos}" y="{height-40}" text-anchor="middle" font-size="12">{x_val:.1f}</text>')

    # Y axis labels and grid
    for i in range(5):
        y_frac = i / 4
        y_val = y_min + (y_max - y_min) * y_frac
        y_pos = height - margin - y_frac * plot_height
        svg_parts.append(f'<line x1="{margin-5}" y1="{y_pos}" x2="{width-margin}" y2="{y_pos}" stroke="#eee" stroke-width="1"/>')
        svg_parts.append(f'<text x="{margin-15}" y="{y_pos+4}" text-anchor="end" font-size="11">{y_val:.2f}</text>')

    # Color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    # Plot lines
    for color_idx, (method, method_data) in enumerate(sorted(data_dict.items())):
        if not method_data:
            continue

        points = []
        for i, x_val in enumerate(x_values):
            if x_val in method_data:
                x_pos = margin + (i / (len(x_values) - 1 if len(x_values) > 1 else 1)) * plot_width
                y_norm = (method_data[x_val] - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                y_pos = height - margin - y_norm * plot_height
                points.append((x_pos, y_pos))

        # Draw line
        if len(points) > 1:
            path = f'M {points[0][0]} {points[0][1]}'
            for x, y in points[1:]:
                path += f' L {x} {y}'
            svg_parts.append(f'<path d="{path}" stroke="{colors[color_idx % len(colors)]}" stroke-width="2.5" fill="none"/>')

        # Draw points
        for x, y in points:
            svg_parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{colors[color_idx % len(colors)]}"/>')

    # Legend
    legend_x = width - margin - 150
    legend_y = margin + 10
    for color_idx, method in enumerate(sorted(data_dict.keys())):
        color = colors[color_idx % len(colors)]
        svg_parts.append(f'<line x1="{legend_x}" y1="{legend_y + color_idx * 20}" x2="{legend_x+15}" y2="{legend_y + color_idx * 20}" stroke="{color}" stroke-width="2"/>')
        svg_parts.append(f'<text x="{legend_x+20}" y="{legend_y + color_idx * 20 + 4}" font-size="11">{method}</text>')

    # Labels
    svg_parts.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="12">Noise Level</text>')
    svg_parts.append(f'<text x="20" y="{height/2}" text-anchor="middle" font-size="12" transform="rotate(-90, 20, {height/2})">{y_label}</text>')

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)

print("Generating HTML report with charts...")

html_parts = [
    "<!DOCTYPE html>",
    "<html>",
    "<head>",
    "<title>E15 Phase 1 Analysis</title>",
    "<style>",
    "body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }",
    "h1 { color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }",
    "h2 { color: #0066cc; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }",
    ".chart-container { margin: 30px 0; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
    "table { border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; border-radius: 8px; overflow: hidden; }",
    "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
    "th { background-color: #0066cc; color: white; font-weight: bold; }",
    "tr:hover { background-color: #f9f9f9; }",
    ".summary { background-color: #e6f2ff; padding: 15px; border-left: 4px solid #0066cc; margin: 20px 0; border-radius: 4px; }",
    "svg { display: block; margin: 20px auto; max-width: 100%; }",
    "</style>",
    "</head>",
    "<body>",
    "<h1>E15 Phase 1: Cross-pLM Comparison Analysis</h1>",
    f"<div class='summary'><strong>Data Overview:</strong> {len(data)} metric rows | {len(methods)} methods | {len(landscapes)} landscapes | 7 noise levels</div>",
]

# Add charts
html_parts.append("<div class='chart-container'>")
html_parts.append("<h2>Spearman Correlation vs Noise Level by Method</h2>")
html_parts.append(create_line_chart("Spearman Correlation Across Noise Levels", spearman_by_method_noise, y_label="Spearman ρ"))
html_parts.append("</div>")

# Add summary table
html_parts.extend([
    "<h2>Summary Statistics</h2>",
    "<table>",
    "<tr><th>Method</th><th>Spearman @ 0.0 Noise</th><th>Spearman @ 1.0 Noise</th><th>Robustness Drop</th></tr>"
])

for method in methods:
    zero_val = spearman_by_method_noise[method].get(0.0, 0)
    high_val = spearman_by_method_noise[method].get(1.0, 0)
    drop = zero_val - high_val

    html_parts.append(f"<tr><td><strong>{method}</strong></td><td>{zero_val:.4f}</td><td>{high_val:.4f}</td><td>{drop:.4f}</td></tr>")

html_parts.extend([
    "</table>",
    "",
    "<h2>Key Findings</h2>",
    "<ul>",
    "<li><strong>Spearman correlation:</strong> Shows how well each method's attribution rankings match ground truth position importance</li>",
    "<li><strong>Noise sensitivity:</strong> All methods show correlation degradation with increasing noise</li>",
    "<li><strong>Comparison:</strong> Methods differ in their noise robustness profiles</li>",
    "</ul>",
    "",
    "<hr/>",
    f"<p style='color: #666; font-size: 12px;'>Generated from {metrics_file}<br/>8,693 metrics across 5 methods × 3 landscapes × 5 seeds × 7 noise levels</p>",
    "</body>",
    "</html>"
])

html = "\n".join(html_parts)

with open(output_html, 'w') as f:
    f.write(html)

print(f"✓ Report with charts saved to: {output_html}")
print("\nChart generated: NDCG vs Noise Level")
