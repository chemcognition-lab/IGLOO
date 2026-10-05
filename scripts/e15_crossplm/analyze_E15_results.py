#!/usr/bin/env python3
"""
Analyze E15 cross-pLM comparison results.
Generates figures showing metric performance across methods, noise levels, and landscapes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
sns.set_style("whitegrid")
results_dir = Path("results/revision/E15_crosspLM")
metrics_file = results_dir / "metrics_long.csv"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# Load data
print("Loading metrics...")
df = pd.read_csv(metrics_file)

# Drop rows with no method/metric/landscape (blank trailing rows in the CSV)
df = df.dropna(subset=['method', 'metric', 'landscape'])
df = df[(df['method'] != '') & (df['metric'] != '') & (df['landscape'] != '')]

# Convert numeric columns
df['noise_level'] = pd.to_numeric(df['noise_level'], errors='coerce')
df['seed'] = pd.to_numeric(df['seed'], errors='coerce')
df['value'] = pd.to_numeric(df['value'], errors='coerce')

print(f"Loaded {len(df)} metric rows")
print(f"Methods: {df['method'].unique()}")
print(f"Metrics: {df['metric'].unique()[:10]}... ({df['metric'].nunique()} total)")
print(f"Noise levels: {sorted(df['noise_level'].dropna().unique())}")

# Key metrics for analysis
key_metrics = ['ndcg', 'ap_r', 'auroc', 'sign_accuracy', 'spearman_corr']

# ============================================================================
# Figure 1: Metric performance vs noise level (by method)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('E15 Phase 1: Attribution Quality vs Noise Level', fontsize=16, fontweight='bold')

for idx, metric in enumerate(key_metrics):
    ax = axes.flat[idx]

    metric_data = df[df['metric'] == metric].copy()
    if len(metric_data) == 0:
        print(f"  Metric {metric} not found, skipping...")
        continue

    # Group by method and noise level
    grouped = metric_data.groupby(['method', 'noise_level'])['value'].agg(['mean', 'std']).reset_index()

    for method in grouped['method'].unique():
        method_data = grouped[grouped['method'] == method].sort_values('noise_level')
        ax.errorbar(method_data['noise_level'], method_data['mean'],
                   yerr=method_data['std'], marker='o', label=method, alpha=0.7)

    ax.set_xlabel('Noise Level')
    ax.set_ylabel(metric.upper())
    ax.set_title(metric.upper())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Remove extra subplot
axes.flat[5].axis('off')
plt.tight_layout()
plt.savefig(figures_dir / "01_metrics_vs_noise.png", dpi=150, bbox_inches='tight')
print("✓ Saved: 01_metrics_vs_noise.png")
plt.close()

# ============================================================================
# Figure 2: Method comparison (heatmap at zero noise)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('E15 Phase 1: Method Comparison at Zero Noise', fontsize=14, fontweight='bold')

landscapes = df['landscape'].unique()
for plot_idx, landscape in enumerate(['additive', 'epistatic', 'nonlinear']):
    ax = axes[plot_idx]

    landscape_data = df[(df['landscape'] == landscape) & (df['noise_level'] == 0.0)]

    # Create pivot table: methods x metrics
    methods = landscape_data['method'].unique()
    metrics = [m for m in key_metrics if m in landscape_data['metric'].unique()]

    heatmap_data = []
    for method in sorted(methods):
        row = []
        for metric in metrics:
            val = landscape_data[(landscape_data['method'] == method) &
                                (landscape_data['metric'] == metric)]['value'].mean()
            row.append(val if not pd.isna(val) else 0)
        heatmap_data.append(row)

    heatmap_array = np.array(heatmap_data)

    sns.heatmap(heatmap_array, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=metrics, yticklabels=sorted(methods), ax=ax,
                cbar_kws={'label': 'Metric Value'})
    ax.set_title(f'{landscape.capitalize()} Landscape')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Method')

plt.tight_layout()
plt.savefig(figures_dir / "02_method_comparison_zero_noise.png", dpi=150, bbox_inches='tight')
print("✓ Saved: 02_method_comparison_zero_noise.png")
plt.close()

# ============================================================================
# Figure 3: Noise robustness (Spearman correlation at different noise levels)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('E15 Phase 1: Noise Robustness (Spearman Correlation)', fontsize=14, fontweight='bold')

for plot_idx, landscape in enumerate(['additive', 'epistatic', 'nonlinear']):
    ax = axes[plot_idx]

    landscape_data = df[(df['landscape'] == landscape) & (df['metric'] == 'spearman_corr')]
    grouped = landscape_data.groupby(['method', 'noise_level'])['value'].mean().reset_index()

    for method in grouped['method'].unique():
        method_data = grouped[grouped['method'] == method].sort_values('noise_level')
        ax.plot(method_data['noise_level'], method_data['value'],
               marker='o', label=method, linewidth=2, markersize=6)

    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title(f'{landscape.capitalize()} Landscape')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.0])

plt.tight_layout()
plt.savefig(figures_dir / "03_noise_robustness.png", dpi=150, bbox_inches='tight')
print("✓ Saved: 03_noise_robustness.png")
plt.close()

# ============================================================================
# Figure 4: Landscape comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('E15 Phase 1: Landscape Comparison', fontsize=14, fontweight='bold')

# Panel 1: Mean metrics by landscape
ax = axes[0]
landscape_means = []
landscapes = sorted(df['landscape'].unique())
metrics_subset = ['ndcg', 'sign_accuracy']

for landscape in landscapes:
    row = []
    for metric in metrics_subset:
        val = df[(df['landscape'] == landscape) & (df['metric'] == metric)]['value'].mean()
        row.append(val if not pd.isna(val) else 0)
    landscape_means.append(row)

landscape_array = np.array(landscape_means)
sns.heatmap(landscape_array, annot=True, fmt='.3f', cmap='viridis',
            xticklabels=metrics_subset, yticklabels=landscapes, ax=ax,
            cbar_kws={'label': 'Mean Metric Value'})
ax.set_title('Mean Metric Performance by Landscape')
ax.set_ylabel('Landscape')

# Panel 2: Noise sensitivity by landscape
ax = axes[1]
for landscape in landscapes:
    landscape_data = df[(df['landscape'] == landscape) & (df['metric'] == 'ndcg')]
    grouped = landscape_data.groupby('noise_level')['value'].mean().sort_index()
    ax.plot(grouped.index, grouped.values, marker='o', label=landscape, linewidth=2, markersize=6)

ax.set_xlabel('Noise Level')
ax.set_ylabel('NDCG Score')
ax.set_title('NDCG Performance Across Noise Levels')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "04_landscape_comparison.png", dpi=150, bbox_inches='tight')
print("✓ Saved: 04_landscape_comparison.png")
plt.close()

# ============================================================================
# Figure 5: Statistical summary
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('E15 Phase 1: Statistical Summary', fontsize=14, fontweight='bold')

# Panel 1: Distribution of NDCG by method
ax = axes[0, 0]
ndcg_data = df[df['metric'] == 'ndcg']
sns.boxplot(data=ndcg_data, x='method', y='value', ax=ax)
ax.set_title('NDCG Distribution by Method')
ax.set_ylabel('NDCG Score')
ax.tick_params(axis='x', rotation=45)

# Panel 2: Sign accuracy by method
ax = axes[0, 1]
sign_data = df[df['metric'] == 'sign_accuracy']
sns.boxplot(data=sign_data, x='method', y='value', ax=ax)
ax.set_title('Sign Accuracy Distribution by Method')
ax.set_ylabel('Sign Accuracy')
ax.tick_params(axis='x', rotation=45)

# Panel 3: NDCG by noise level
ax = axes[1, 0]
ndcg_by_noise = df[df['metric'] == 'ndcg'].groupby('noise_level')['value'].apply(list)
bp = ax.boxplot([ndcg_by_noise[nl] for nl in sorted(ndcg_by_noise.index)],
                 labels=sorted(ndcg_by_noise.index))
ax.set_xlabel('Noise Level')
ax.set_ylabel('NDCG Score')
ax.set_title('NDCG by Noise Level')
ax.grid(True, alpha=0.3, axis='y')

# Panel 4: Metric count
ax = axes[1, 1]
metric_counts = df['metric'].value_counts().head(10)
metric_counts.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 10 Metrics by Frequency')
ax.set_xlabel('Count')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(figures_dir / "05_statistical_summary.png", dpi=150, bbox_inches='tight')
print("✓ Saved: 05_statistical_summary.png")
plt.close()

# ============================================================================
# Print summary statistics
# ============================================================================
print("\n" + "="*70)
print("E15 PHASE 1 RESULTS SUMMARY")
print("="*70)

print("\nData Overview:")
print(f"  Total metric rows: {len(df)}")
print(f"  Protein: {df['protein'].unique()}")
print(f"  Methods: {sorted(df['method'].unique())}")
print(f"  Landscapes: {sorted(df['landscape'].unique())}")
print(f"  Noise levels: {sorted(df['noise_level'].dropna().unique())}")
print(f"  Seeds: {sorted(df['seed'].dropna().unique())}")

print("\nMethod Performance (NDCG at zero noise):")
ndcg_zero = df[(df['metric'] == 'ndcg') & (df['noise_level'] == 0.0)]
for method in sorted(ndcg_zero['method'].unique()):
    mean_val = ndcg_zero[ndcg_zero['method'] == method]['value'].mean()
    print(f"  {method:25s}: {mean_val:.4f}")

print("\nNoise Robustness (Spearman correlation):")
print("  Method                    | Zero Noise | High Noise (1.0)")
print("  " + "-" * 50)
for method in sorted(df['method'].unique()):
    zero_noise = df[(df['method'] == method) & (df['metric'] == 'spearman_corr') &
                    (df['noise_level'] == 0.0)]['value'].mean()
    high_noise = df[(df['method'] == method) & (df['metric'] == 'spearman_corr') &
                    (df['noise_level'] == 1.0)]['value'].mean()
    print(f"  {method:25s} | {zero_noise:10.4f} | {high_noise:10.4f}")

print("\nLandscape Comparison (Mean NDCG):")
for landscape in sorted(df['landscape'].unique()):
    mean_ndcg = df[(df['landscape'] == landscape) & (df['metric'] == 'ndcg')]['value'].mean()
    print(f"  {landscape:25s}: {mean_ndcg:.4f}")

print("\n" + "="*70)
print(f"Figures saved to: {figures_dir}")
print("="*70)
