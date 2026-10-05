#!/usr/bin/env python3
"""
Generate E15 analysis figures from metrics CSV and save as SVG and PNG files.
"""

import csv
from pathlib import Path
import statistics
import subprocess
import sys

results_dir = Path("results/revision/E15_crosspLM")
metrics_file = results_dir / "metrics_long.csv"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# Helper to convert SVG to PNG
def svg_to_png(svg_path, png_path, width=700, height=450):
    """Convert SVG to PNG using ImageMagick or cairosvg."""
    try:
        # Try ImageMagick first (convert command)
        subprocess.run(
            ['convert', '-density', '150', str(svg_path), str(png_path)],
            check=True,
            capture_output=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        # Try cairosvg
        import cairosvg
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
        return True
    except (ImportError, Exception):
        pass

    return False

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
metrics_list = sorted(set(r['metric'] for r in data))

print(f"Methods: {methods}")
print(f"Landscapes: {landscapes}")
print(f"Metrics: {metrics_list[:5]}... ({len(metrics_list)} total)")

# Color scheme: IG methods vs direct methods
def get_method_color(method_name):
    """Return color based on method type."""
    ig_colors = {
        'ig_linear': '#1f77b4',  # Blue
        'ig_slerp': '#ff7f0e',   # Orange
        'ig_nearest_neighbor': '#2ca02c',  # Green
    }
    direct_colors = {
        'gradient_input': '#d62728',  # Red
        'input_difference': '#9467bd',  # Purple
    }
    return ig_colors.get(method_name, direct_colors.get(method_name, '#7f7f7f'))

# Create SVG line chart function with error bands
def create_line_chart(title, data_dict, data_by_seed=None, width=700, height=450, y_label="Metric Value", filename=None):
    """Create SVG line chart with error bands and optionally save to file.

    Args:
        data_dict: Dict of method -> noise_level -> mean_value
        data_by_seed: Dict of method -> noise_level -> seed -> value (for error bands)
    """

    all_x = set()
    for method_data in data_dict.values():
        all_x.update(method_data.keys())
    x_values = sorted(all_x)

    if not x_values:
        return "<p>No data</p>"

    all_y = []
    for method_data in data_dict.values():
        all_y.extend(method_data.values())
    y_min = min(all_y) if all_y else 0
    y_max = max(all_y) if all_y else 1
    y_min = max(0, y_min - 0.05)
    y_max = min(1, y_max + 0.05)

    margin = 70
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    svg_parts = [
        f'<svg width="{width}" height="{height}" style="border: 1px solid #ccc; background: white; font-family: Roboto, sans-serif;">',
        f'<text x="20" y="25" font-size="16" font-weight="bold" font-family="Roboto">{title}</text>',
    ]

    # Axes
    svg_parts.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333" stroke-width="2"/>')
    svg_parts.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333" stroke-width="2"/>')

    # X axis labels
    for i, x_val in enumerate(x_values):
        x_pos = margin + (i / (len(x_values) - 1 if len(x_values) > 1 else 1)) * plot_width
        svg_parts.append(f'<text x="{x_pos}" y="{height-45}" text-anchor="middle" font-size="11" font-family="Roboto">{x_val:.1f}</text>')

    # Y axis labels and grid
    for i in range(5):
        y_frac = i / 4
        y_val = y_min + (y_max - y_min) * y_frac
        y_pos = height - margin - y_frac * plot_height
        svg_parts.append(f'<line x1="{margin-5}" y1="{y_pos}" x2="{width-margin}" y2="{y_pos}" stroke="#ddd" stroke-width="1"/>')
        svg_parts.append(f'<text x="{margin-15}" y="{y_pos+4}" text-anchor="end" font-size="10" font-family="Roboto">{y_val:.2f}</text>')

    # Plot lines and error bands
    for method in sorted(data_dict.keys()):
        method_data = data_dict[method]
        if not method_data:
            continue

        color = get_method_color(method)
        points = []
        error_points = []

        for i, x_val in enumerate(x_values):
            if x_val in method_data:
                x_pos = margin + (i / (len(x_values) - 1 if len(x_values) > 1 else 1)) * plot_width
                mean_val = method_data[x_val]
                y_norm = (mean_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                y_pos = height - margin - y_norm * plot_height
                points.append((x_pos, y_pos))

                # Compute error band from seed data if available
                if data_by_seed and method in data_by_seed and x_val in data_by_seed[method]:
                    seed_vals = list(data_by_seed[method][x_val].values())
                    if len(seed_vals) > 1:
                        std_dev = statistics.stdev(seed_vals)
                        upper_val = min(mean_val + std_dev, y_max)
                        lower_val = max(mean_val - std_dev, y_min)
                        y_upper_norm = (upper_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                        y_lower_norm = (lower_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                        y_upper_pos = height - margin - y_upper_norm * plot_height
                        y_lower_pos = height - margin - y_lower_norm * plot_height
                        error_points.append((x_pos, y_upper_pos, y_lower_pos))

        # Draw error band
        if error_points and len(error_points) > 1:
            # Create polygon path: upper line then lower line reversed
            band_path = f"M {error_points[0][0]} {error_points[0][1]}"
            # Draw upper edge
            for x, y_u, y_l in error_points[1:]:
                band_path += f" L {x} {y_u}"
            # Draw lower edge (reverse)
            for x, y_u, y_l in reversed(error_points):
                band_path += f" L {x} {y_l}"
            band_path += " Z"
            svg_parts.append(f'<path d="{band_path}" fill="{color}" opacity="0.2" stroke="none"/>')

        # Draw line
        if len(points) > 1:
            path = f'M {points[0][0]} {points[0][1]}'
            for x, y in points[1:]:
                path += f' L {x} {y}'
            svg_parts.append(f'<path d="{path}" stroke="{color}" stroke-width="2.5" fill="none"/>')

        # Draw points
        for x, y in points:
            svg_parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}" stroke="white" stroke-width="1"/>')

    # Legend
    legend_x = width - margin - 180
    legend_y = margin + 15
    for idx, method in enumerate(sorted(data_dict.keys())):
        color = get_method_color(method)
        is_ig = method.startswith('ig_')
        label = method.replace('_', ' ').title()

        svg_parts.append(f'<line x1="{legend_x}" y1="{legend_y + idx * 22}" x2="{legend_x+15}" y2="{legend_y + idx * 22}" stroke="{color}" stroke-width="2.5"/>')
        svg_parts.append(f'<circle cx="{legend_x+7.5}" cy="{legend_y + idx * 22}" r="3" fill="{color}" stroke="white" stroke-width="1"/>')
        svg_parts.append(f'<text x="{legend_x+20}" y="{legend_y + idx * 22 + 4}" font-size="11" font-family="Roboto">{label}</text>')

        # Add IG/Direct indicator
        if is_ig:
            svg_parts.append(f'<text x="{legend_x+150}" y="{legend_y + idx * 22 + 4}" font-size="9" font-family="Roboto" fill="#666">(IG)</text>')

    # Labels
    svg_parts.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="12" font-family="Roboto">Noise Level</text>')
    svg_parts.append(f'<text x="25" y="{height/2}" text-anchor="middle" font-size="12" font-family="Roboto" transform="rotate(-90, 25, {height/2})">{y_label}</text>')

    svg_parts.append('</svg>')
    svg_str = '\n'.join(svg_parts)

    if filename:
        svg_path = figures_dir / filename
        with open(svg_path, 'w') as f:
            f.write(svg_str)
        print(f"✓ Saved SVG: {svg_path}")

        # Convert to PNG
        png_filename = filename.replace('.svg', '.png')
        png_path = figures_dir / png_filename
        if svg_to_png(svg_path, png_path, width=width, height=height):
            print(f"✓ Saved PNG: {png_path}")
        else:
            print(f"⚠ Could not convert to PNG (ImageMagick or cairosvg not available)")

    return svg_str

# Generate figures for each metric
print("\nGenerating figures...")

# 1. NDCG vs Noise
ndcg_data = {}
ndcg_by_seed = {}
for method in methods:
    ndcg_data[method] = {}
    ndcg_by_seed[method] = {}
    for nl in noise_levels:
        values = [r['value'] for r in data
                 if r['method'] == method and r['metric'] == 'ndcg' and r['noise_level'] == nl]
        if values:
            ndcg_data[method][nl] = statistics.mean(values)

        # Collect by seed for error bands
        ndcg_by_seed[method][nl] = {}
        for seed in seeds:
            seed_vals = [r['value'] for r in data
                        if r['method'] == method and r['metric'] == 'ndcg' and r['noise_level'] == nl and r['seed'] == seed]
            if seed_vals:
                ndcg_by_seed[method][nl][seed] = seed_vals[0]

create_line_chart("NDCG vs Noise Level", ndcg_data, ndcg_by_seed, y_label="NDCG", filename="01_ndcg_vs_noise.svg")

# 2. AUROC vs Noise
auroc_data = {}
auroc_by_seed = {}
for method in methods:
    auroc_data[method] = {}
    auroc_by_seed[method] = {}
    for nl in noise_levels:
        values = [r['value'] for r in data
                 if r['method'] == method and r['metric'] == 'auroc' and r['noise_level'] == nl]
        if values:
            auroc_data[method][nl] = statistics.mean(values)

        auroc_by_seed[method][nl] = {}
        for seed in seeds:
            seed_vals = [r['value'] for r in data
                        if r['method'] == method and r['metric'] == 'auroc' and r['noise_level'] == nl and r['seed'] == seed]
            if seed_vals:
                auroc_by_seed[method][nl][seed] = seed_vals[0]

if auroc_data and any(auroc_data.values()):
    create_line_chart("AUROC vs Noise Level", auroc_data, auroc_by_seed, y_label="AUROC", filename="02_auroc_vs_noise.svg")

# 3. Sign Accuracy vs Noise
sign_acc_data = {}
sign_acc_by_seed = {}
for method in methods:
    sign_acc_data[method] = {}
    sign_acc_by_seed[method] = {}
    for nl in noise_levels:
        values = [r['value'] for r in data
                 if r['method'] == method and r['metric'] == 'sign_accuracy' and r['noise_level'] == nl]
        if values:
            sign_acc_data[method][nl] = statistics.mean(values)

        sign_acc_by_seed[method][nl] = {}
        for seed in seeds:
            seed_vals = [r['value'] for r in data
                        if r['method'] == method and r['metric'] == 'sign_accuracy' and r['noise_level'] == nl and r['seed'] == seed]
            if seed_vals:
                sign_acc_by_seed[method][nl][seed] = seed_vals[0]

if sign_acc_data and any(sign_acc_data.values()):
    create_line_chart("Sign Accuracy vs Noise Level", sign_acc_data, sign_acc_by_seed, y_label="Accuracy", filename="03_sign_accuracy_vs_noise.svg")

# 4. AP@R vs Noise
ap_r_data = {}
ap_r_by_seed = {}
for method in methods:
    ap_r_data[method] = {}
    ap_r_by_seed[method] = {}
    for nl in noise_levels:
        values = [r['value'] for r in data
                 if r['method'] == method and r['metric'] == 'ap_r' and r['noise_level'] == nl]
        if values:
            ap_r_data[method][nl] = statistics.mean(values)

        ap_r_by_seed[method][nl] = {}
        for seed in seeds:
            seed_vals = [r['value'] for r in data
                        if r['method'] == method and r['metric'] == 'ap_r' and r['noise_level'] == nl and r['seed'] == seed]
            if seed_vals:
                ap_r_by_seed[method][nl][seed] = seed_vals[0]

if ap_r_data and any(ap_r_data.values()):
    create_line_chart("AP@R vs Noise Level", ap_r_data, ap_r_by_seed, y_label="AP@R", filename="04_ap_r_vs_noise.svg")

svg_count = len(list(figures_dir.glob('*.svg')))
png_count = len(list(figures_dir.glob('*.png')))
print(f"\n✓ Generated {svg_count} SVG and {png_count} PNG figures in {figures_dir}")
print(f"\nFiles:")
for f in sorted(figures_dir.glob('*')):
    print(f"  {f.name}")
