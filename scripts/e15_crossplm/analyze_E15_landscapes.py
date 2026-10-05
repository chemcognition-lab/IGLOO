#!/usr/bin/env python3
"""
E15 analysis: Performance across different fitness landscapes (additive, epistatic, nonlinear).
Shows how attribution quality differs across landscape complexity.
"""

import csv
from pathlib import Path
import statistics

results_dir = Path("results/revision/E15_crosspLM")
metrics_file = results_dir / "metrics_long.csv"
output_html = results_dir / "E15_landscape_analysis.html"

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

print(f"Methods: {methods}")
print(f"Landscapes: {landscapes}")

# Prepare data by landscape and method
def get_landscape_metrics(landscape):
    """Returns: noise_level -> method -> seed -> value"""
    result = {}
    for nl in noise_levels:
        result[nl] = {}
        for method in methods:
            result[nl][method] = {}
            for seed in set(r['seed'] for r in data):
                values = [r['value'] for r in data
                         if r['landscape'] == landscape and r['metric'] == 'spearman_corr'
                         and r['noise_level'] == nl and r['method'] == method and r['seed'] == seed]
                if values:
                    result[nl][method][seed] = statistics.mean(values)
    return result

# Create multi-landscape SVG chart
def create_landscape_comparison_chart(landscape_data_dict, width=900, height=500):
    """Create SVG showing all 3 landscapes in one view."""

    all_x = sorted(set().union(*[set(d.keys()) for d in landscape_data_dict.values()]))

    if not all_x:
        return "<p>No data</p>"

    # Compute global y bounds
    all_y = []
    for landscape_data in landscape_data_dict.values():
        for nl_data in landscape_data.values():
            for method_data in nl_data.values():
                for vals in method_data.values():
                    if isinstance(vals, dict):
                        all_y.extend(vals.values())
                    else:
                        all_y.append(vals)

    y_min = max(0, min(all_y) - 0.05) if all_y else 0
    y_max = min(1, max(all_y) + 0.05) if all_y else 1

    # SVG dimensions for 3 panels
    panel_width = 280
    panel_height = height
    margin = 60

    svg_parts = [
        f'<svg width="{width}" height="{height}" style="border: 1px solid #ccc; background: white;">',
        f'<text x="20" y="25" font-size="14" font-weight="bold">Attribution Quality Across Fitness Landscapes</text>',
    ]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    # Draw 3 panels for each landscape
    landscapes_list = sorted(landscape_data_dict.keys())
    for panel_idx, landscape in enumerate(landscapes_list):
        landscape_data = landscape_data_dict[landscape]
        panel_x = 50 + panel_idx * (panel_width + 20)

        # Panel title
        svg_parts.append(f'<text x="{panel_x + panel_width/2}" y="50" font-size="12" font-weight="bold" text-anchor="middle">{landscape.capitalize()}</text>')

        # Axes
        axis_y1 = 70
        axis_y2 = height - margin
        axis_x1 = panel_x
        axis_x2 = panel_x + panel_width - 40

        svg_parts.append(f'<line x1="{axis_x1}" y1="{axis_y2}" x2="{axis_x2}" y2="{axis_y2}" stroke="#999" stroke-width="2"/>')
        svg_parts.append(f'<line x1="{axis_x1}" y1="{axis_y1}" x2="{axis_x1}" y2="{axis_y2}" stroke="#999" stroke-width="2"/>')

        plot_width = axis_x2 - axis_x1
        plot_height = axis_y2 - axis_y1

        # X axis labels (noise levels)
        for i, x_val in enumerate(all_x):
            if i % 2 == 0:  # Show every other label to avoid crowding
                x_pos = axis_x1 + (i / (len(all_x) - 1 if len(all_x) > 1 else 1)) * plot_width
                svg_parts.append(f'<text x="{x_pos}" y="{axis_y2 + 20}" text-anchor="middle" font-size="9">{x_val:.1f}</text>')

        # Y axis labels
        for i in range(5):
            y_frac = i / 4
            y_val = y_min + (y_max - y_min) * y_frac
            y_pos = axis_y2 - y_frac * plot_height
            if panel_idx == 0:  # Only on first panel
                svg_parts.append(f'<text x="{panel_x - 10}" y="{y_pos + 4}" text-anchor="end" font-size="8">{y_val:.2f}</text>')

        # Plot lines for each method
        for color_idx, method in enumerate(sorted(landscape_data.get(all_x[0], {}).keys())):
            points = []
            for i, x_val in enumerate(all_x):
                if x_val in landscape_data and method in landscape_data[x_val]:
                    seed_values = list(landscape_data[x_val][method].values())
                    if seed_values:
                        mean_val = statistics.mean(seed_values)
                        stdev = statistics.stdev(seed_values) if len(seed_values) > 1 else 0

                        x_pos = axis_x1 + (i / (len(all_x) - 1 if len(all_x) > 1 else 1)) * plot_width
                        y_norm = (mean_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                        y_pos = axis_y2 - y_norm * plot_height

                        # Error bounds
                        y_upper = axis_y2 - ((min(mean_val + stdev, 1.0) - y_min) / (y_max - y_min) * plot_height if y_max > y_min else plot_height/2)
                        y_lower = axis_y2 - ((max(mean_val - stdev, 0.0) - y_min) / (y_max - y_min) * plot_height if y_max > y_min else plot_height/2)

                        points.append((x_pos, y_pos, y_upper, y_lower, method, colors[color_idx % len(colors)]))

            # Draw error band
            if len(points) > 1:
                band_path = f"M {points[0][0]} {points[0][2]}"
                for x, y, y_u, y_l, _, _ in points[1:]:
                    band_path += f" L {x} {y_u}"
                band_path += f" L {points[-1][0]} {points[-1][3]}"
                for x, y, y_u, y_l, _, _ in reversed(points[1:]):
                    band_path += f" L {x} {y_l}"
                band_path += " Z"

                color = colors[color_idx % len(colors)]
                svg_parts.append(f'<path d="{band_path}" fill="{color}" opacity="0.12" stroke="none"/>')

            # Draw line and points
            if len(points) > 1:
                path = f'M {points[0][0]} {points[0][1]}'
                for x, y, _, _, _, _ in points[1:]:
                    path += f' L {x} {y}'
                color = colors[color_idx % len(colors)]
                svg_parts.append(f'<path d="{path}" stroke="{color}" stroke-width="1.5" fill="none"/>')

            for x, y, _, _, _, color in points:
                svg_parts.append(f'<circle cx="{x}" cy="{y}" r="2.5" fill="{color}"/>')

    # Global legend at bottom
    legend_x = 80
    legend_y = height - 20
    for color_idx, method in enumerate(sorted(methods)):
        if color_idx >= 5:
            break
        color = colors[color_idx % len(colors)]
        x_pos = legend_x + (color_idx % 5) * 170
        svg_parts.append(f'<line x1="{x_pos}" y1="{legend_y}" x2="{x_pos + 10}" y2="{legend_y}" stroke="{color}" stroke-width="1.5"/>')
        svg_parts.append(f'<circle cx="{x_pos + 5}" cy="{legend_y}" r="2" fill="{color}"/>')
        svg_parts.append(f'<text x="{x_pos + 15}" y="{legend_y + 3}" font-size="9">{method}</text>')

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)

print("Generating landscape comparison analysis...")

# Prepare data for all landscapes
landscape_data_dict = {}
for landscape in landscapes:
    landscape_data_dict[landscape] = get_landscape_metrics(landscape)

html_parts = [
    "<!DOCTYPE html>",
    "<html>",
    "<head>",
    "<title>E15: Landscape Complexity Analysis</title>",
    "<style>",
    "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }",
    "h1 { color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }",
    "h2 { color: #0066cc; margin-top: 35px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }",
    ".chart-container { margin: 30px 0; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
    "table { border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }",
    "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
    "th { background-color: #0066cc; color: white; font-weight: bold; }",
    ".summary { background-color: #e6f2ff; padding: 15px; border-left: 4px solid #0066cc; margin: 20px 0; border-radius: 4px; }",
    "svg { display: block; margin: 20px auto; }",
    "</style>",
    "</head>",
    "<body>",
    "<h1>E15 Phase 1: Landscape Complexity Impact on Attribution Quality</h1>",
    f"<div class='summary'><strong>Fitness Models:</strong><br/>",
    "• <strong>Additive:</strong> Linear combinations of sequence positions<br/>",
    "• <strong>Epistatic:</strong> Pairwise position interactions<br/>",
    "• <strong>Nonlinear:</strong> Complex multi-position dependencies</div>",
]

html_parts.append("<div class='chart-container'>")
html_parts.append(create_landscape_comparison_chart(landscape_data_dict, width=900, height=480))
html_parts.append("</div>")

# Summary table
html_parts.extend([
    "<h2>Spearman Correlation by Landscape and Noise Level</h2>",
    "<table>",
    "<tr><th>Landscape</th><th>Zero Noise</th><th>0.3 Noise</th><th>High Noise (1.0)</th><th>Robustness</th></tr>"
])

for landscape in sorted(landscapes):
    landscape_data = landscape_data_dict[landscape]

    # Aggregate across all methods and seeds
    zero_vals = []
    mid_vals = []
    high_vals = []

    for method in methods:
        if 0.0 in landscape_data and method in landscape_data[0.0]:
            zero_vals.extend(landscape_data[0.0][method].values())
        if 0.3 in landscape_data and method in landscape_data[0.3]:
            mid_vals.extend(landscape_data[0.3][method].values())
        if 1.0 in landscape_data and method in landscape_data[1.0]:
            high_vals.extend(landscape_data[1.0][method].values())

    if zero_vals and high_vals:
        zero_mean = statistics.mean(zero_vals)
        high_mean = statistics.mean(high_vals)
        mid_mean = statistics.mean(mid_vals) if mid_vals else 0
        drop_pct = ((zero_mean - high_mean) / zero_mean * 100) if zero_mean > 0 else 0

        robustness = "High" if drop_pct < 15 else ("Medium" if drop_pct < 30 else "Low")

        html_parts.append(
            f"<tr><td><strong>{landscape.capitalize()}</strong></td><td>{zero_mean:.4f}</td><td>{mid_mean:.4f}</td><td>{high_mean:.4f}</td><td>{robustness}</td></tr>"
        )

html_parts.extend([
    "</table>",
    "",
    "<h2>Analysis by Landscape Type</h2>",
    "<ul>",
    "<li><strong>Additive Fitness:</strong> Simplest model - direct, non-interacting effects. Should show highest attribution accuracy.</li>",
    "<li><strong>Epistatic Fitness:</strong> Second-order interactions between positions. More complex to attribute.</li>",
    "<li><strong>Nonlinear Fitness:</strong> Complex dependencies - most challenging for attribution methods to capture.</li>",
    "</ul>",
    "",
    "<p style='margin-top: 30px; color: #666; font-size: 12px;'>Generated from {}</p>".format(metrics_file),
    "</body>",
    "</html>"
])

html = "\n".join(html_parts)

with open(output_html, 'w') as f:
    f.write(html)

print(f"✓ Landscape analysis saved to: {output_html}")
