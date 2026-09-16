#!/usr/bin/env python3
"""
Generate E15 plots with model performance (val_r2) on x-axis.
Shows how attribution quality (y-axis) depends on model generalization performance,
not just noise level. Same structure as noise plots but x-axis is model performance.
"""

import csv
from pathlib import Path
import statistics
import subprocess

results_dir = Path("results/revision/E15_crosspLM")
metrics_file = results_dir / "metrics_long.csv"
model_perf_file = results_dir / "model_performance.csv"
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

# Load model performance data
print("Loading model performance...")
model_perf_data = {}
try:
    with open(model_perf_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (row['protein'], row['landscape'], row['plm'], int(row['seed']))
                model_perf_data[key] = {
                    'val_spearman': float(row['val_spearman']),
                    'val_loss': float(row['val_loss']),
                }
            except (ValueError, KeyError):
                continue
    print(f"Loaded {len(model_perf_data)} model performance entries")
except FileNotFoundError:
    print(f"Warning: {model_perf_file} not found. Model performance plots will skip.")
    model_perf_data = {}

methods = sorted(set(r['method'] for r in data))
noise_levels = sorted(set(r['noise_level'] for r in data))

print(f"Methods: {methods}")
print(f"Noise levels: {noise_levels}")

def get_method_color(method_name):
    ig_colors = {
        'ig_linear': '#1f77b4',
        'ig_slerp': '#ff7f0e',
        'ig_nearest_neighbor': '#2ca02c',
    }
    direct_colors = {
        'gradient_input': '#d62728',
        'input_difference': '#9467bd',
    }
    return ig_colors.get(method_name, direct_colors.get(method_name, '#7f7f7f'))

def svg_to_png(svg_path, png_path):
    try:
        subprocess.run(
            ['convert', '-density', '150', str(svg_path), str(png_path)],
            check=True,
            capture_output=True
        )
        return True
    except:
        try:
            import cairosvg
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
            return True
        except:
            return False

def create_line_chart(title, data_dict, width=700, height=450, y_label="Metric Value", filename=None):
    """Create line chart with model performance (Spearman) on x-axis."""

    all_x = set()
    for method_data in data_dict.values():
        all_x.update(method_data.keys())
    x_values = sorted(all_x)

    if not x_values:
        return None

    all_y = []
    for method_data in data_dict.values():
        all_y.extend(method_data.values())

    if not all_y:
        return None

    y_min = min(all_y) - 0.05
    y_max = max(all_y) + 0.05
    x_min = min(x_values) - 0.02
    x_max = max(x_values) + 0.02

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
        svg_parts.append(f'<text x="{x_pos}" y="{height-45}" text-anchor="middle" font-size="11" font-family="Roboto">{x_val:.2f}</text>')

    # Y axis labels and grid
    for i in range(5):
        y_frac = i / 4
        y_val = y_min + (y_max - y_min) * y_frac
        y_pos = height - margin - y_frac * plot_height
        svg_parts.append(f'<line x1="{margin-5}" y1="{y_pos}" x2="{width-margin}" y2="{y_pos}" stroke="#ddd" stroke-width="1"/>')
        svg_parts.append(f'<text x="{margin-15}" y="{y_pos+4}" text-anchor="end" font-size="10" font-family="Roboto">{y_val:.2f}</text>')

    # Plot lines and points
    for method in sorted(data_dict.keys()):
        method_data = data_dict[method]
        if not method_data:
            continue

        color = get_method_color(method)
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
            svg_parts.append(f'<path d="{path}" stroke="{color}" stroke-width="2.5" fill="none"/>')

        # Draw points
        for x, y in points:
            svg_parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}" stroke="white" stroke-width="1"/>')

    # Legend
    legend_x = width - margin - 180
    legend_y = margin + 15
    for idx, method in enumerate(sorted(methods)):
        color = get_method_color(method)
        is_ig = method.startswith('ig_')
        label = method.replace('_', ' ').title()

        svg_parts.append(f'<line x1="{legend_x}" y1="{legend_y + idx * 22}" x2="{legend_x+15}" y2="{legend_y + idx * 22}" stroke="{color}" stroke-width="2.5"/>')
        svg_parts.append(f'<circle cx="{legend_x+7.5}" cy="{legend_y + idx * 22}" r="3" fill="{color}" stroke="white" stroke-width="1"/>')
        svg_parts.append(f'<text x="{legend_x+20}" y="{legend_y + idx * 22 + 4}" font-size="11" font-family="Roboto">{label}</text>')
        if is_ig:
            svg_parts.append(f'<text x="{legend_x+150}" y="{legend_y + idx * 22 + 4}" font-size="9" font-family="Roboto" fill="#666">(IG)</text>')

    # Labels
    svg_parts.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="12" font-family="Roboto">Model Performance (Spearman ρ)</text>')
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

    return svg_str

print("\nGenerating plots with model performance on x-axis...")

# For each metric, plot against Spearman correlation (model performance)
metrics_to_plot = [
    ('ndcg', 'NDCG'),
    ('auroc', 'AUROC'),
    ('sign_accuracy', 'Sign Accuracy'),
    ('ap_r', 'AP@R'),
]

if not model_perf_data:
    print("Skipping model performance plots: model_performance.csv not found")
else:
    for metric_name, metric_label in metrics_to_plot:
        # Aggregate by method and model performance (val_spearman)
        data_dict = {}

        # For each metric data point, map to model performance
        for row in data:
            if row['metric'] != metric_name:
                continue

            protein = row['protein']
            landscape = row['landscape']
            plm = row['plm']
            seed = int(row['seed'])
            method = row['method']

            key = (protein, landscape, plm, seed)
            if key not in model_perf_data:
                continue

            model_perf = model_perf_data[key]['val_spearman']
            attr_val = row['value']

            if method not in data_dict:
                data_dict[method] = {}

            # Aggregate by rounding model performance to 3 decimals for binning
            model_perf_bin = round(model_perf, 3)
            if model_perf_bin not in data_dict[method]:
                data_dict[method][model_perf_bin] = []
            data_dict[method][model_perf_bin].append(attr_val)

        # Average within each bin
        averaged_data_dict = {}
        for method, perf_dict in data_dict.items():
            averaged_data_dict[method] = {}
            for perf_bin, values in perf_dict.items():
                averaged_data_dict[method][perf_bin] = statistics.mean(values)

        if not any(averaged_data_dict.values()):
            print(f"  Skipping {metric_name} (no data)")
            continue

        filename = f"perf_{metric_name}.svg"
        create_line_chart(
            f"{metric_label} vs Model Performance",
            averaged_data_dict,
            y_label=metric_label,
            filename=filename
        )

print(f"\n✓ Generated {len(metrics_to_plot)} plots in {figures_dir}")
print("\nThese plots show attribution quality as a function of model generalization performance.")
print("If lines cluster, attribution quality depends more on model performance than noise level.")
