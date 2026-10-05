#!/usr/bin/env python3
"""
Generate E15 box plot figures with random baselines.
Shows distribution of metric values across seeds for each method at each noise level.
"""

import csv
from pathlib import Path
import statistics
import subprocess

results_dir = Path("results/revision/E15_crosspLM")
metrics_file = results_dir / "metrics_long.csv"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

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
noise_levels = sorted(set(r['noise_level'] for r in data))

print(f"Methods: {methods}")
print(f"Noise levels: {noise_levels}")

# Color scheme for methods
def get_method_color(method_name):
    """Return color based on method type."""
    ig_colors = {
        'ig_linear': '#1f77b4',      # Blue
        'ig_slerp': '#ff7f0e',       # Orange
        'ig_nearest_neighbor': '#2ca02c',  # Green
    }
    direct_colors = {
        'gradient_input': '#d62728',  # Red
        'input_difference': '#9467bd',  # Purple
    }
    return ig_colors.get(method_name, direct_colors.get(method_name, '#7f7f7f'))

# Random baselines for metrics
RANDOM_BASELINES = {
    'ndcg': 0.2,  # Random ranking performs poorly
    'auroc': 0.5,  # Random classifier
    'sign_accuracy': 0.5,  # Random guess
    'ap_r': 0.3,  # Random retrieval
    'spearman_corr': 0.0,  # No correlation
    'kendall_tau': 0.0,  # No correlation
}

def svg_to_png(svg_path, png_path):
    """Convert SVG to PNG using ImageMagick."""
    try:
        subprocess.run(
            ['convert', '-density', '150', str(svg_path), str(png_path)],
            check=True,
            capture_output=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            import cairosvg
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
            return True
        except:
            return False

def create_boxplot(title, metric_name, data_by_method_noise, width=1000, height=500, y_label="Metric Value", filename=None):
    """Create SVG box plot showing distribution across seeds."""

    margin = 70
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    # Get y bounds
    all_vals = []
    for method_data in data_by_method_noise.values():
        for noise_vals in method_data.values():
            all_vals.extend(noise_vals)

    if not all_vals:
        return None

    y_min = min(all_vals) - 0.05
    y_max = max(all_vals) + 0.05
    random_baseline = RANDOM_BASELINES.get(metric_name, 0.0)

    svg_parts = [
        f'<svg width="{width}" height="{height}" style="border: 1px solid #ccc; background: white; font-family: Roboto, sans-serif;">',
        f'<text x="20" y="25" font-size="16" font-weight="bold" font-family="Roboto">{title}</text>',
    ]

    # Title with random baseline info
    svg_parts.append(f'<text x="{width-margin-200}" y="25" font-size="11" font-family="Roboto" fill="#666">Random baseline: {random_baseline:.2f}</text>')

    # Axes
    svg_parts.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333" stroke-width="2"/>')
    svg_parts.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333" stroke-width="2"/>')

    # Random baseline line
    if 0 <= random_baseline <= 1:
        y_norm = (random_baseline - y_min) / (y_max - y_min)
        y_pos = height - margin - y_norm * plot_height
        svg_parts.append(f'<line x1="{margin}" y1="{y_pos}" x2="{width-margin}" y2="{y_pos}" stroke="#999" stroke-width="2" stroke-dasharray="5,5"/>')
        svg_parts.append(f'<text x="{margin+5}" y="{y_pos-5}" font-size="10" font-family="Roboto" fill="#999">random</text>')

    # Y axis labels and grid
    for i in range(5):
        y_frac = i / 4
        y_val = y_min + (y_max - y_min) * y_frac
        y_pos = height - margin - y_frac * plot_height
        svg_parts.append(f'<line x1="{margin-5}" y1="{y_pos}" x2="{width-margin}" y2="{y_pos}" stroke="#ddd" stroke-width="1"/>')
        svg_parts.append(f'<text x="{margin-15}" y="{y_pos+4}" text-anchor="end" font-size="10" font-family="Roboto">{y_val:.2f}</text>')

    # Calculate box plot positions
    box_width = plot_width / len(noise_levels) / (len(methods) + 1)

    for noise_idx, noise_level in enumerate(noise_levels):
        noise_x = margin + (noise_idx + 0.5) * (plot_width / len(noise_levels))

        for method_idx, method in enumerate(sorted(methods)):
            if method not in data_by_method_noise or noise_level not in data_by_method_noise[method]:
                continue

            values = sorted(data_by_method_noise[method][noise_level])
            if not values:
                continue

            # Calculate quartiles
            q1 = values[len(values)//4] if len(values) >= 4 else values[0]
            median = statistics.median(values)
            q3 = values[3*len(values)//4] if len(values) >= 4 else values[-1]
            whisker_low = min(values)
            whisker_high = max(values)

            # Convert to y positions
            def val_to_y(val):
                norm = (val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                return height - margin - norm * plot_height

            y_q1 = val_to_y(q1)
            y_median = val_to_y(median)
            y_q3 = val_to_y(q3)
            y_low = val_to_y(whisker_low)
            y_high = val_to_y(whisker_high)

            # X position for this box
            x_center = noise_x + (method_idx - len(methods)/2) * box_width * 0.8
            box_half_width = box_width * 0.35

            color = get_method_color(method)

            # Draw whiskers
            svg_parts.append(f'<line x1="{x_center}" y1="{y_low}" x2="{x_center}" y2="{y_high}" stroke="{color}" stroke-width="1"/>')
            svg_parts.append(f'<line x1="{x_center-box_half_width*0.5}" y1="{y_low}" x2="{x_center+box_half_width*0.5}" y2="{y_low}" stroke="{color}" stroke-width="1"/>')
            svg_parts.append(f'<line x1="{x_center-box_half_width*0.5}" y1="{y_high}" x2="{x_center+box_half_width*0.5}" y2="{y_high}" stroke="{color}" stroke-width="1"/>')

            # Draw box
            box_height = y_q3 - y_q1
            svg_parts.append(f'<rect x="{x_center-box_half_width}" y="{min(y_q1, y_q3)}" width="{box_half_width*2}" height="{abs(box_height)}" fill="{color}" opacity="0.3" stroke="{color}" stroke-width="1.5"/>')

            # Draw median line
            svg_parts.append(f'<line x1="{x_center-box_half_width}" y1="{y_median}" x2="{x_center+box_half_width}" y2="{y_median}" stroke="{color}" stroke-width="2.5"/>')

            # Draw individual points
            for val in values:
                y_val = val_to_y(val)
                # Jitter x position slightly
                x_jitter = x_center + (hash(str(val)) % 20 - 10) / 30
                svg_parts.append(f'<circle cx="{x_jitter}" cy="{y_val}" r="2" fill="{color}" opacity="0.4"/>')

    # X axis labels
    for noise_idx, noise_level in enumerate(noise_levels):
        x_pos = margin + (noise_idx + 0.5) * (plot_width / len(noise_levels))
        svg_parts.append(f'<text x="{x_pos}" y="{height-45}" text-anchor="middle" font-size="11" font-family="Roboto">{noise_level:.1f}</text>')

    # Legend
    legend_x = width - margin - 200
    legend_y = margin + 20
    for idx, method in enumerate(sorted(methods)):
        color = get_method_color(method)
        label = method.replace('_', ' ').title()
        is_ig = method.startswith('ig_')

        svg_parts.append(f'<rect x="{legend_x}" y="{legend_y + idx*22-8}" width="12" height="12" fill="{color}" opacity="0.3" stroke="{color}"/>')
        svg_parts.append(f'<text x="{legend_x+18}" y="{legend_y + idx*22}" font-size="11" font-family="Roboto">{label}</text>')
        if is_ig:
            svg_parts.append(f'<text x="{legend_x+140}" y="{legend_y + idx*22}" font-size="9" font-family="Roboto" fill="#666">(IG)</text>')

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

        png_filename = filename.replace('.svg', '.png')
        png_path = figures_dir / png_filename
        if svg_to_png(svg_path, png_path):
            print(f"✓ Saved PNG: {png_path}")
        else:
            print(f"⚠ Could not convert to PNG")

    return svg_str

# Generate box plots for each metric
print("\nGenerating box plots...")

metrics_to_plot = ['ndcg', 'auroc', 'sign_accuracy', 'ap_r']

for metric in metrics_to_plot:
    # Collect data by method and noise level
    data_by_method_noise = {}
    for method in methods:
        data_by_method_noise[method] = {}
        for nl in noise_levels:
            values = [r['value'] for r in data
                     if r['method'] == method and r['metric'] == metric and r['noise_level'] == nl]
            if values:
                data_by_method_noise[method][nl] = values

    if not any(data_by_method_noise.values()):
        print(f"  Skipping {metric} (no data)")
        continue

    metric_label = metric.replace('_', ' ').title()
    filename = f"{metrics_to_plot.index(metric)+1:02d}_{metric}_boxplot.svg"

    create_boxplot(
        f"{metric_label} Distribution Across Seeds",
        metric,
        data_by_method_noise,
        y_label=metric_label,
        filename=filename
    )

print(f"\n✓ Generated box plot figures in {figures_dir}")
