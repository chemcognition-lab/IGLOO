#!/usr/bin/env python3
"""
Generate E15 performance correlation plots.
Shows attribution quality (y-axis) vs model performance (Spearman correlation, x-axis)
at different noise levels, revealing if attribution quality depends on model generalization.
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

def get_method_color(method_name):
    """Return color based on method type."""
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
    """Convert SVG to PNG using ImageMagick."""
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

def create_scatter_plot(title, x_label, y_label, points, width=700, height=500, filename=None):
    """Create SVG scatter plot of attribution quality vs model performance."""

    if not points:
        print(f"  Skipping {filename} (no data)")
        return None

    margin = 70
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    # Get bounds
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    x_min = min(x_vals) - 0.05
    x_max = max(x_vals) + 0.05
    y_min = min(y_vals) - 0.05
    y_max = max(y_vals) + 0.05

    svg_parts = [
        f'<svg width="{width}" height="{height}" style="border: 1px solid #ccc; background: white; font-family: Roboto, sans-serif;">',
        f'<text x="20" y="25" font-size="16" font-weight="bold" font-family="Roboto">{title}</text>',
    ]

    # Axes
    svg_parts.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333" stroke-width="2"/>')
    svg_parts.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333" stroke-width="2"/>')

    # Grid
    for i in range(5):
        # X grid
        x_frac = i / 4
        x_pos = margin + x_frac * plot_width
        svg_parts.append(f'<line x1="{x_pos}" y1="{margin}" x2="{x_pos}" y2="{height-margin}" stroke="#ddd" stroke-width="1"/>')

        # Y grid
        y_frac = i / 4
        y_pos = height - margin - y_frac * plot_height
        svg_parts.append(f'<line x1="{margin}" y1="{y_pos}" x2="{width-margin}" y2="{y_pos}" stroke="#ddd" stroke-width="1"/>')

    # X axis labels
    for i in range(5):
        x_frac = i / 4
        x_val = x_min + (x_max - x_min) * x_frac
        x_pos = margin + x_frac * plot_width
        svg_parts.append(f'<text x="{x_pos}" y="{height-45}" text-anchor="middle" font-size="11" font-family="Roboto">{x_val:.2f}</text>')

    # Y axis labels
    for i in range(5):
        y_frac = i / 4
        y_val = y_min + (y_max - y_min) * y_frac
        y_pos = height - margin - y_frac * plot_height
        svg_parts.append(f'<text x="{margin-15}" y="{y_pos+4}" text-anchor="end" font-size="11" font-family="Roboto">{y_val:.2f}</text>')

    # Plot points (colored by method)
    for x, y, method in points:
        x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.5
        x_pos = margin + x_norm * plot_width
        y_pos = height - margin - y_norm * plot_height

        color = get_method_color(method)
        svg_parts.append(f'<circle cx="{x_pos}" cy="{y_pos}" r="4" fill="{color}" stroke="white" stroke-width="1.5" opacity="0.7"/>')

    # Legend
    legend_x = width - margin - 160
    legend_y = margin + 15
    for idx, method in enumerate(sorted(methods)):
        color = get_method_color(method)
        label = method.replace('_', ' ').title()
        is_ig = method.startswith('ig_')

        svg_parts.append(f'<circle cx="{legend_x}" cy="{legend_y + idx*22}" r="4" fill="{color}" opacity="0.7"/>')
        svg_parts.append(f'<text x="{legend_x+12}" y="{legend_y + idx*22 + 4}" font-size="11" font-family="Roboto">{label}</text>')
        if is_ig:
            svg_parts.append(f'<text x="{legend_x+130}" y="{legend_y + idx*22 + 4}" font-size="9" font-family="Roboto" fill="#666">(IG)</text>')

    # Labels
    svg_parts.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="12" font-family="Roboto">{x_label}</text>')
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

print("\nGenerating model performance correlation plots...")

# For each noise level, create a plot showing attribution quality vs model performance
for noise_idx, noise_level in enumerate(noise_levels):
    points_ndcg = []
    points_auroc = []

    # Collect data for this noise level
    for method in methods:
        # Model performance: use Spearman correlation as proxy
        model_perf_vals = [r['value'] for r in data
                          if r['method'] == method and r['metric'] == 'spearman_corr' and r['noise_level'] == noise_level]

        ndcg_vals = [r['value'] for r in data
                    if r['method'] == method and r['metric'] == 'ndcg' and r['noise_level'] == noise_level]

        auroc_vals = [r['value'] for r in data
                     if r['method'] == method and r['metric'] == 'auroc' and r['noise_level'] == noise_level]

        if model_perf_vals and ndcg_vals:
            model_perf = statistics.mean(model_perf_vals)
            ndcg = statistics.mean(ndcg_vals)
            points_ndcg.append((model_perf, ndcg, method))

        if model_perf_vals and auroc_vals:
            model_perf = statistics.mean(model_perf_vals)
            auroc = statistics.mean(auroc_vals)
            points_auroc.append((model_perf, auroc, method))

    # Create plots
    title_noise = f"Noise Level {noise_level}"

    if points_ndcg:
        filename = f"perf_ndcg_noise{noise_level:.1f}".replace('.', '_') + ".svg"
        create_scatter_plot(
            f"NDCG vs Model Performance ({title_noise})",
            "Model Performance (Spearman ρ)",
            "NDCG Score",
            points_ndcg,
            filename=filename
        )

    if points_auroc:
        filename = f"perf_auroc_noise{noise_level:.1f}".replace('.', '_') + ".svg"
        create_scatter_plot(
            f"AUROC vs Model Performance ({title_noise})",
            "Model Performance (Spearman ρ)",
            "AUROC",
            points_auroc,
            filename=filename
        )

print(f"\n✓ Generated performance correlation plots in {figures_dir}")
print(f"\nThese plots show whether attribution quality depends on model generalization performance.")
