"""
Plotting utilities for IGLOO analysis.

This module contains all plotting functions used across the IGLOO analysis scripts,
providing a centralized location for visualization code.
"""

import os
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from src.igloo.synthetic_data import PROTEINS as _PROTEIN_DEFS  # type: ignore
except ImportError:
    _PROTEIN_DEFS = {}

# ---------------------------------------------------------------------------
# Global style for IG-quality figures
# ---------------------------------------------------------------------------
FIG_BG = "#F7F8FA"
import matplotlib
matplotlib.rcParams.update({
    "figure.facecolor":    FIG_BG,
    "axes.facecolor":      FIG_BG,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.titlesize":      14,
    "axes.titleweight":    "bold",
    "axes.labelsize":      12,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "legend.fontsize":     11,
    "lines.linewidth":     2.0,
    "lines.markersize":    7,
    "savefig.transparent": True,
})


def _safe_filename(name: str) -> str:
    """Convert a sequence name to a safe filename by replacing non-alphanumeric chars."""
    import re
    return re.sub(r'[^\w\-]', '_', name)


def plot_integrated_gradients(results_dict: dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot integrated gradients for multiple sequences.

    Each sequence is saved as its own PNG inside a ``_per_sequence/`` subdirectory
    derived from *save_path*.  This avoids generating a single enormous stacked
    figure (which could be hundreds of MB for large proteins / many sequences).

    Args:
        results_dict: Dictionary containing analysis results for each sequence
        save_path: Base path for output.  Individual files are written to
            ``<save_path without .png>_per_sequence/<sequence_name>.png``.
    """
    setup_matplotlib_style()
    # Filter out special validation results keys that don't have sequence analysis data
    analysis_results = {name: result for name, result in results_dict.items()
                       if not name.startswith('_') and 'positions' in result}

    n_sequences = len(analysis_results)
    if n_sequences == 0:
        print("No valid sequences found for plotting.")
        return

    # Derive per-sequence output directory from save_path
    if save_path:
        base = save_path[:-4] if save_path.endswith('.png') else save_path
        save_dir = base + '_per_sequence'
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    saved_paths = []
    for name, result in analysis_results.items():
        positions = result['positions']
        gradients = result['position_gradients']
        amino_acids = result['amino_acids']

        # Figure width scales with sequence length; height is fixed
        n_pos = len(positions)
        fig_width = max(12, min(n_pos * 0.15, 40))

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, 3.5))

        # Bar plot: blue = positive contribution, red = negative
        colors = ['#2166ac' if g >= 0 else '#d6604d' for g in gradients]
        ax.bar(positions, gradients, color=colors, alpha=0.8, width=0.8)

        # Label the top-10% magnitude positions
        threshold = np.percentile(np.abs(gradients), 90)
        for pos, aa, grad in zip(positions, amino_acids, gradients):
            if abs(grad) >= threshold:
                offset = np.sign(grad) * 0.015 * max(np.abs(gradients), default=1)
                ax.text(pos, grad + offset, aa,
                        ha='center', va='bottom' if grad >= 0 else 'top',
                        fontweight='bold', fontsize=7)

        ax.set_xlabel('Position', fontsize=9)
        ax.set_ylabel('Integrated Gradient', fontsize=9)
        ax.set_title(
            f'{name}   |   Predicted: {result["predicted_activity"]:.3f}'
            f'   |   Effect vs Baseline: {result["effect_vs_baseline"]:.3f}',
            fontsize=9,
        )
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.6)
        ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()

        if save_dir:
            out_path = os.path.join(save_dir, f'{_safe_filename(name)}.png')
            plt.savefig(out_path, dpi=120, bbox_inches='tight', transparent=True)
            saved_paths.append(out_path)
        plt.close(fig)

    if saved_paths:
        print(f"Integrated gradients plots saved: {len(saved_paths)} files in {save_dir}")


def plot_gradients_heatmap(results_df: pd.DataFrame, title: str, save_path: Optional[str] = None) -> None:
    """
    Create two separate figures from the gradient data:

    Figure 1 — Main heatmap (saved to *save_path*):
        One row per sequence, diverging RdBu colormap.

    Figure 2 — Summary heatmap (saved to *<stem>_summary.<ext>*):
        Two stacked rows: mean gradient per position (RdBu) and SD per
        position (YlOrRd), sharing the same x-axis.

    Y-axis labels in the main heatmap are suppressed when there are more than
    50 sequences. Figure height is capped so the file stays manageable.

    Args:
        results_df: DataFrame with columns ['sequence_name', 'position', 'integrated_gradient']
        title: Title for the main heatmap
        save_path: Optional path to save the main heatmap figure
    """
    import os

    pivot_data = results_df.pivot(index='sequence_name', columns='position', values='integrated_gradient')

    num_positions = len(pivot_data.columns)
    num_sequences = len(pivot_data)

    # Summary statistics (shape: 1 × positions)
    mean_row = pivot_data.mean(axis=0).values.reshape(1, -1)
    sd_row   = pivot_data.std(axis=0).values.reshape(1, -1)

    width = max(20, num_positions * 0.12)
    main_height = min(max(6, num_sequences * 0.12), 30)
    summary_row_height = max(0.8, width * 0.04)

    show_ylabels = num_sequences <= 50
    y_fontsize = max(4, min(9, 400 // num_sequences)) if show_ylabels else 0

    x_tick_frequency = max(1, num_positions // 60)
    x_tick_labels = [str(pos) if i % x_tick_frequency == 0 else ''
                     for i, pos in enumerate(pivot_data.columns)]

    # Shared diverging scale for main heatmap and mean row
    finite_vals = pivot_data.values[np.isfinite(pivot_data.values)]
    vabs = max(abs(finite_vals.min()), abs(finite_vals.max())) if len(finite_vals) else 1.0
    vabs = vabs if vabs > 0 else 1.0

    # ── Figure 1: Main heatmap ───────────────────────────────────────────────
    fig1, ax_main = plt.subplots(figsize=(width, main_height))
    sns.heatmap(
        pivot_data,
        ax=ax_main,
        cmap='RdBu',
        center=0,
        vmin=-vabs,
        vmax=vabs,
        cbar_kws={'label': 'Integrated Gradient'},
        xticklabels=x_tick_labels,
        yticklabels=show_ylabels,
    )
    ax_main.set_title(title)
    ax_main.set_xlabel('Position')
    ax_main.set_ylabel('Sequence')
    ax_main.tick_params(axis='x', labelsize=8, rotation=90)
    if show_ylabels:
        ax_main.tick_params(axis='y', labelsize=y_fontsize)
    else:
        ax_main.set_yticks([])

    fig1.tight_layout()
    if save_path:
        fig1.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Gradient heatmap saved to {save_path}")
    plt.close(fig1)

    # ── Figure 2: Mean + SD summary ──────────────────────────────────────────
    fig2, (ax_mean, ax_sd) = plt.subplots(
        2, 1,
        figsize=(width, 2 * summary_row_height + 0.5),
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.5},
    )

    sns.heatmap(
        mean_row,
        ax=ax_mean,
        cmap='RdBu',
        center=0,
        vmin=-vabs,
        vmax=vabs,
        cbar_kws={'label': 'Mean IG'},
        xticklabels=x_tick_labels,
        yticklabels=['Mean'],
    )
    ax_mean.set_title('Mean Integrated Gradient per Position', fontsize=10)
    ax_mean.set_xlabel('')
    ax_mean.tick_params(axis='x', labelsize=7, rotation=90)
    ax_mean.tick_params(axis='y', labelsize=8)

    sd_max = float(np.nanmax(sd_row)) if np.nanmax(sd_row) > 0 else 1.0
    sns.heatmap(
        sd_row,
        ax=ax_sd,
        cmap='YlOrRd',
        vmin=0,
        vmax=sd_max,
        cbar_kws={'label': 'SD'},
        xticklabels=x_tick_labels,
        yticklabels=['SD'],
    )
    ax_sd.set_title('Standard Deviation of Integrated Gradient per Position', fontsize=10)
    ax_sd.set_xlabel('Position')
    ax_sd.tick_params(axis='x', labelsize=7, rotation=90)
    ax_sd.tick_params(axis='y', labelsize=8)

    fig2.tight_layout()
    if save_path:
        base, ext = os.path.splitext(save_path)
        summary_path = f"{base}_summary{ext}"
        fig2.savefig(summary_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Gradient summary heatmap saved to {summary_path}")
    plt.close(fig2)


def plot_variants_vs_wt_gradients(ig_results: dict[str, Any], save_path: Optional[str] = None,
                                   max_variants: int = 9) -> None:
    """
    Plot gradient differences between a representative set of variants and wild-type.

    When many sequences are present (e.g. include_all=True), selects up to
    ``max_variants`` informative variants: the lowest-activity, highest-activity,
    and median-activity sequences, plus those with the largest L2 gradient
    deviation from WT.  This keeps the figure legible regardless of dataset size.

    Args:
        ig_results: Dictionary containing integrated gradients results
        save_path: Optional path to save the plot
        max_variants: Maximum number of variant subplots to show (default 9)
    """
    setup_matplotlib_style()
    # Filter out special/validation keys
    analysis_results = {name: result for name, result in ig_results.items()
                       if not name.startswith('_') and 'positions' in result}

    if len(analysis_results) == 0:
        print("No valid sequences found for variant comparison plotting.")
        return

    # Find wildtype result
    wt_key = None
    for key in ['WT', 'Wildtype', 'wildtype', 'wt']:
        if key in analysis_results:
            wt_key = key
            break
    if wt_key is None:
        wt_key = list(analysis_results.keys())[0]

    wt_result = analysis_results[wt_key]
    wt_gradients = wt_result['position_gradients']

    variants = {name: res for name, res in analysis_results.items() if name != wt_key}

    if len(variants) == 0:
        print("Only wildtype sequence found, skipping variant comparison plot.")
        return

    # --- Select a representative subset ---
    if len(variants) <= max_variants:
        selected = list(variants.keys())
    else:
        activities = {name: res['predicted_activity'] for name, res in variants.items()}
        sorted_by_act = sorted(activities, key=activities.get)

        n_extreme = max(1, max_variants // 3)
        # low / high activity extremes
        bottom = sorted_by_act[:n_extreme]
        top = sorted_by_act[-n_extreme:]
        # median-ish
        mid_idx = len(sorted_by_act) // 2
        mid_half = n_extreme // 2
        median = sorted_by_act[max(0, mid_idx - mid_half): mid_idx + mid_half + 1]

        # largest gradient deviation from WT
        l2_dev = {name: float(np.linalg.norm(res['position_gradients'] - wt_gradients))
                  for name, res in variants.items()}
        sorted_by_dev = sorted(l2_dev, key=l2_dev.get, reverse=True)
        top_dev = sorted_by_dev[:n_extreme]

        seen = set()
        selected = []
        for name in bottom + top + median + top_dev:
            if name not in seen:
                seen.add(name)
                selected.append(name)
            if len(selected) >= max_variants:
                break

    n_plots = len(selected)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, name in enumerate(selected):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        result = variants[name]
        positions = result['positions']
        gradient_diff = result['position_gradients'] - wt_gradients

        colors = ['red' if g < 0 else 'steelblue' for g in gradient_diff]
        ax.bar(positions, gradient_diff, color=colors, alpha=0.75, width=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)
        ax.set_xlabel('Position', fontsize=9)
        ax.set_ylabel('ΔGradient (variant − WT)', fontsize=9)
        act_v = result['predicted_activity']
        act_wt = wt_result['predicted_activity']
        ax.set_title(f'{name}\nact={act_v:.2f}  WT={act_wt:.2f}', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for idx in range(n_plots, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle('Gradient Difference from WT\n(representative variants)', fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Variants vs WT gradients plot saved to {save_path}")
    plt.close()


def plot_target_distribution(y_values: np.ndarray, title: str, xlabel: str, save_path: Optional[str] = None) -> None:
    """
    Plot distribution of target values.

    Args:
        y_values: Array of target values
        title: Title for the plot
        xlabel: Label for x-axis
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(y_values, bins=30, kde=True)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str,
                           r2: float, pearson_r: float, save_path: Optional[str] = None) -> None:
    """
    Create predicted vs actual scatter plot with performance metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Title for the plot
        r2: R-squared score
        pearson_r: Pearson correlation coefficient
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50)

    # Add diagonal line for perfect prediction
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{title}\nR² = {r2:.4f}, r = {pearson_r:.4f}')
    plt.grid(True, alpha=0.3)

    # Make plot square
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def plot_mutation_landscape(mutation_effects: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot mutation effects landscape.

    Args:
        mutation_effects: DataFrame with mutation effects data
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Create pivot table for heatmap
    pivot_data = mutation_effects.pivot(index='mutation', columns='position', values='effect')

    sns.heatmap(
        pivot_data,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Mutation Effect'},
        xticklabels=True,
        yticklabels=True
    )

    plt.title('Mutation Effects Landscape')
    plt.xlabel('Position')
    plt.ylabel('Mutation')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Mutation landscape plot saved to {save_path}")
    plt.close()


def plot_sequence_alignment_with_hotspots(sequences: dict[str, str], hotspots: list,
                                        save_path: Optional[str] = None) -> None:
    """
    Plot sequence alignment highlighting hotspot positions.

    Args:
        sequences: Dictionary of sequence names to sequences
        hotspots: List of hotspot positions
        save_path: Optional path to save the plot
    """
    if not sequences:
        return

    seq_names = list(sequences.keys())
    seq_length = len(list(sequences.values())[0])

    plt.figure(figsize=(max(15, seq_length * 0.1), len(seq_names) * 0.5))

    # Create color map for amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    colors = plt.cm.Set3(np.linspace(0, 1, len(amino_acids)))
    aa_color_map = {aa: colors[i] for i, aa in enumerate(amino_acids)}

    for i, (_name, seq) in enumerate(sequences.items()):
        for j, aa in enumerate(seq):
            color = aa_color_map.get(aa, 'white')
            alpha = 1.0 if (j + 1) in hotspots else 0.3
            plt.scatter(j + 1, i, c=[color], s=100, alpha=alpha, edgecolors='black', linewidth=0.5)

    plt.yticks(range(len(seq_names)), seq_names)
    plt.xlabel('Position')
    plt.title('Sequence Alignment with Hotspots Highlighted')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Sequence alignment plot saved to {save_path}")
    plt.close()


def setup_matplotlib_style() -> None:
    """Set up consistent matplotlib style for all plots with Roboto font and color theme."""
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

    # Color theme: professional palette
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.color'] = '#e5e5e5'
    plt.rcParams['grid.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['legend.framealpha'] = 0.95
    plt.rcParams['legend.edgecolor'] = '#cccccc'

    # Color cycle: modern palette (primary blue, secondary orange, supporting colors)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


def plot_training_history(
    history: dict,
    title: str = "Training History",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        history: Training history dictionary
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    setup_matplotlib_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss plot
    axes[0].plot(history['epoch'], history['train_loss'], label='Training Loss', marker='o')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(history['epoch'], history['val_loss'], label='Validation Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Log scale loss plot
    axes[1].semilogy(history['epoch'], history['train_loss'], label='Training Loss', marker='o')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[1].semilogy(history['epoch'], history['val_loss'], label='Validation Loss', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Training and Validation Loss (Log Scale)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Training history plot saved to {save_path}")

    plt.show()


def plot_predicted_vs_actual_batch(
    predicted_values: list,
    actual_values: list,
    sequence_names: Optional[list] = None,
    title: str = "Predicted vs Actual Activities",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> None:
    """
    Plot predicted vs actual values for multiple sequences.

    Args:
        predicted_values: List of predicted activities
        actual_values: List of actual/experimental activities
        sequence_names: Optional list of sequence names for labeling
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size
    """

    if len(predicted_values) != len(actual_values):
        raise ValueError("predicted_values and actual_values must have the same length")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot points
    ax.scatter(actual_values, predicted_values, alpha=0.7, s=60)

    # Add sequence names if provided
    if sequence_names and len(sequence_names) == len(predicted_values):
        for i, name in enumerate(sequence_names):
            ax.annotate(name, (actual_values[i], predicted_values[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    # Add perfect prediction line
    all_values = predicted_values + actual_values
    min_val, max_val = min(all_values), max(all_values)
    margin = (max_val - min_val) * 0.1
    line_min, line_max = min_val - margin, max_val + margin

    ax.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.5,
            label='Perfect Prediction', linewidth=2)

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(actual_values, predicted_values)
    line_y = slope * np.array([line_min, line_max]) + intercept
    ax.plot([line_min, line_max], line_y, 'red', alpha=0.7,
            label=f'Regression (R²={r_value**2:.3f})', linewidth=2)

    # Calculate and display metrics
    r2 = r2_score(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)

    ax.set_xlabel('Actual Activity')
    ax.set_ylabel('Predicted Activity')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text with metrics
    metrics_text = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nn = {len(predicted_values)}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Batch predicted vs actual plot saved to {save_path}")

    plt.show()


def plot_predicted_vs_actual_single(
    predicted_value: float,
    actual_value: float,
    baseline_predicted: float,
    baseline_actual: float,
    sequence_name: str = "Sequence",
    baseline_name: str = "Baseline",
    title: str = "Predicted vs Actual Activity",
    save_path: Optional[str] = None
) -> None:
    """
    Plot predicted vs actual values for single sequences with baseline comparison.

    Args:
        predicted_value: Predicted activity for the sequence
        actual_value: Actual/experimental activity for the sequence
        baseline_predicted: Predicted activity for baseline
        baseline_actual: Actual activity for baseline
        sequence_name: Name for the sequence point
        baseline_name: Name for the baseline point
        title: Plot title
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot points
    ax.scatter(actual_value, predicted_value, color='red', s=100, alpha=0.7,
               label=f'{sequence_name}', zorder=5)
    ax.scatter(baseline_actual, baseline_predicted, color='blue', s=100, alpha=0.7,
               label=f'{baseline_name}', zorder=5)

    # Add perfect prediction line
    all_values = [predicted_value, actual_value, baseline_predicted, baseline_actual]
    min_val, max_val = min(all_values), max(all_values)
    margin = (max_val - min_val) * 0.1
    line_min, line_max = min_val - margin, max_val + margin

    ax.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.5,
            label='Perfect Prediction')

    # Add connecting line between points
    ax.plot([baseline_actual, actual_value], [baseline_predicted, predicted_value],
            'gray', alpha=0.5, linestyle='-', linewidth=1)

    # Calculate and display metrics
    sequence_error = abs(predicted_value - actual_value)
    baseline_error = abs(baseline_predicted - baseline_actual)

    ax.set_xlabel('Actual Activity')
    ax.set_ylabel('Predicted Activity')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text with error information
    text_info = f'Sequence Error: {sequence_error:.3f}\nBaseline Error: {baseline_error:.3f}'
    ax.text(0.05, 0.95, text_info, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Predicted vs actual plot saved to {save_path}")

    plt.show()


def plot_signed_gradient_effects(
    results_dict: dict[str, Any],
    save_path: Optional[str] = None,
    top_n: int = 10,
    figsize: tuple[int, int] = (15, 8)
) -> None:
    """
    Create a scatter plot showing signed gradient effects by position.

    - Red/orange points: beneficial effects (positive gradients)
    - Blue points: detrimental effects (negative gradients)
    - Small position-space jitter for visibility; scales with data range
    - Top N most significant position-level effects labelled with short arrows

    Args:
        results_dict: Dictionary containing analysis results for each sequence
        save_path: Optional path to save the plot
        top_n: Number of top effects to label (default: 10)
        figsize: Figure size as tuple (width, height)
    """
    setup_matplotlib_style()
    analysis_results = {name: result for name, result in results_dict.items()
                        if not name.startswith('_') and 'positions' in result}

    if len(analysis_results) == 0:
        print("No valid sequences found for signed gradient effects plotting.")
        return

    all_effects = []
    for name, result in analysis_results.items():
        for pos, grad, aa in zip(result['positions'],
                                  result['position_gradients'],
                                  result['amino_acids']):
            all_effects.append({
                'sequence_name': name,
                'position': int(pos),
                'amino_acid': aa,
                'effect': float(grad),
                'abs_effect': abs(float(grad))
            })

    effects_df = pd.DataFrame(all_effects)

    y_min, y_max = effects_df['effect'].min(), effects_df['effect'].max()
    y_range = max(y_max - y_min, 1e-6)

    # Jitter scaled to ~1 % of the data range so points stay within the plot
    jitter_scale = 0.01 * y_range
    n_points = len(effects_df)

    beneficial = effects_df[effects_df['effect'] > 0].copy()
    detrimental = effects_df[effects_df['effect'] < 0].copy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    np.random.seed(42)

    # For large datasets, downsample to avoid a saturated solid-colour blob
    _SCATTER_MAX = 10_000
    if n_points > _SCATTER_MAX:
        sample_frac = _SCATTER_MAX / n_points
        ben_plot = beneficial.sample(frac=sample_frac, random_state=42) if len(beneficial) > 0 else beneficial
        det_plot = detrimental.sample(frac=sample_frac, random_state=42) if len(detrimental) > 0 else detrimental
    else:
        ben_plot = beneficial
        det_plot = detrimental

    alpha = 0.3
    point_size = 8

    if len(ben_plot) > 0:
        jitter = np.random.normal(0, jitter_scale, len(ben_plot))
        ax.scatter(ben_plot['position'], ben_plot['effect'] + jitter,
                   c='steelblue', alpha=alpha, s=point_size, linewidths=0,
                   label=f'Beneficial ({len(beneficial):,} pos-sequence pairs)')

    if len(det_plot) > 0:
        jitter = np.random.normal(0, jitter_scale, len(det_plot))
        ax.scatter(det_plot['position'], det_plot['effect'] + jitter,
                   c='coral', alpha=alpha, s=point_size, linewidths=0,
                   label=f'Detrimental ({len(detrimental):,} pos-sequence pairs)')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Label the top-N most extreme position-level mean effects
    # (average across sequences to get one representative point per position)
    pos_mean = (effects_df.groupby('position')
                .agg(mean_effect=('effect', 'mean'),
                     abs_mean=('abs_effect', 'mean'),
                     amino_acid=('amino_acid', 'first'))
                .reset_index()
                .nlargest(top_n, 'abs_mean'))

    for _, row in pos_mean.iterrows():
        sign = 1 if row['mean_effect'] >= 0 else -1
        ax.annotate(
            f"{row['amino_acid']}{int(row['position'])}",
            xy=(row['position'], row['mean_effect']),
            xytext=(0, sign * 18),          # 18 pt offset in display space
            textcoords='offset points',
            fontsize=8,
            ha='center',
            va='bottom' if sign > 0 else 'top',
            annotation_clip=True,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.85, edgecolor='none'),
            arrowprops=dict(arrowstyle='->', color='black', alpha=0.5, lw=0.5)
        )

    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Signed Gradient Effect', fontsize=12)
    ax.set_title('Signed Gradient Effects by Position', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.08 * y_range)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Signed gradient effects plot saved to {save_path}")
    plt.close()


def plot_averaged_gradient_effects_comparison(
    results_dict: dict[str, Any],
    save_path: Optional[str] = None,
    top_n: int = 5,
    bottom_n: int = 5,
    figsize: tuple[int, int] = (16, 10)
) -> None:
    """
    Create a comparison plot showing averaged gradient effects for top N vs bottom N sequences.

    This creates a figure similar to the mutation effects plot, but comparing:
    - Top N sequences (highest predicted activity)
    - Bottom N sequences (lowest predicted activity)

    Args:
        results_dict: Dictionary containing analysis results for each sequence
        save_path: Optional path to save the plot
        top_n: Number of top sequences to average (default: 5)
        bottom_n: Number of bottom sequences to average (default: 5)
        figsize: Figure size as tuple (width, height)
    """
    setup_matplotlib_style()
    # Filter out special validation results keys that don't have sequence analysis data
    analysis_results = {name: result for name, result in results_dict.items()
                       if not name.startswith('_') and 'positions' in result}

    if len(analysis_results) == 0:
        print("No valid sequences found for top vs bottom comparison plotting.")
        return

    # Extract predicted activities and sort sequences
    sequence_activities = []
    for name, result in analysis_results.items():
        sequence_activities.append({
            'name': name,
            'predicted_activity': result['predicted_activity'],
            'result': result
        })

    # Sort by predicted activity
    sequence_activities.sort(key=lambda x: x['predicted_activity'])

    # Get top N and bottom N sequences
    bottom_sequences = sequence_activities[:bottom_n]
    top_sequences = sequence_activities[-top_n:]

    def average_gradients_by_position(sequences):
        """Average gradients across sequences by position"""
        position_effects = {}

        for seq_data in sequences:
            result = seq_data['result']
            positions = result['positions']
            gradients = result['position_gradients']
            amino_acids = result['amino_acids']

            for pos, grad, aa in zip(positions, gradients, amino_acids):
                if pos not in position_effects:
                    position_effects[pos] = {'effects': [], 'amino_acids': []}
                position_effects[pos]['effects'].append(grad)
                position_effects[pos]['amino_acids'].append(aa)

        # Calculate averages
        averaged_effects = []
        for pos, data in position_effects.items():
            avg_effect = np.mean(data['effects'])
            # Use the most common amino acid at this position
            most_common_aa = max(set(data['amino_acids']), key=data['amino_acids'].count)
            averaged_effects.append({
                'position': pos,
                'effect': avg_effect,
                'amino_acid': most_common_aa,
                'abs_effect': abs(avg_effect)
            })

        return pd.DataFrame(averaged_effects)

    # Calculate averaged effects
    top_effects = average_gradients_by_position(top_sequences)
    bottom_effects = average_gradients_by_position(bottom_sequences)

    # Create subplot figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                   gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})

    def plot_effects(ax, effects_df, title, color_map):
        """Plot effects for one group"""
        # Separate beneficial and detrimental
        beneficial = effects_df[effects_df['effect'] > 0].copy()
        detrimental = effects_df[effects_df['effect'] < 0].copy()

        # Add jitter
        np.random.seed(42)

        # Plot beneficial effects
        if len(beneficial) > 0:
            beneficial_jitter = np.random.normal(0, 3, len(beneficial))
            ax.scatter(
                beneficial['position'],
                beneficial['effect'] + beneficial_jitter,
                c=color_map['beneficial'],
                alpha=0.7,
                s=40,
                label=f'Beneficial ({len(beneficial)})'
            )

        # Plot detrimental effects
        if len(detrimental) > 0:
            detrimental_jitter = np.random.normal(0, 3, len(detrimental))
            ax.scatter(
                detrimental['position'],
                detrimental['effect'] + detrimental_jitter,
                c=color_map['detrimental'],
                alpha=0.7,
                s=40,
                label=f'Detrimental ({len(detrimental)})'
            )

        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)

        # Label top effects
        top_pos_effects = effects_df.nlargest(5, 'effect')
        top_neg_effects = effects_df.nsmallest(5, 'effect')

        for _, row in pd.concat([top_pos_effects, top_neg_effects]).iterrows():
            ax.annotate(
                f"{row['amino_acid']}{int(row['position'])}",
                xy=(row['position'], row['effect']),
                xytext=(row['position'], row['effect'] + np.sign(row['effect']) * 0.05 *
                       (effects_df['effect'].max() - effects_df['effect'].min())),
                fontsize=8,
                ha='center',
                va='bottom' if row['effect'] > 0 else 'top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5, lw=0.5)
            )

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Position', fontsize=11)
        ax.set_ylabel('Averaged Signed Gradient', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Set consistent y-limits
        y_min, y_max = effects_df['effect'].min(), effects_df['effect'].max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

    # Plot top sequences (use warmer colors)
    top_avg_activity = np.mean([seq['predicted_activity'] for seq in top_sequences])
    plot_effects(
        ax1, top_effects,
        f'Top {top_n} Sequences (Avg Activity: {top_avg_activity:.2f})',
        {'beneficial': 'steelblue', 'detrimental': 'orangered'}
    )

    # Plot bottom sequences (use cooler colors)
    bottom_avg_activity = np.mean([seq['predicted_activity'] for seq in bottom_sequences])
    plot_effects(
        ax2, bottom_effects,
        f'Bottom {bottom_n} Sequences (Avg Activity: {bottom_avg_activity:.2f})',
        {'beneficial': 'steelblue', 'detrimental': 'coral'}
    )

    plt.suptitle('Signed Gradient Effects: High vs Low Activity Sequences',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Averaged gradient effects comparison plot saved to {save_path}")

    plt.close()


def plot_attributions_on_3d_structure(
    attribution_data: pd.DataFrame,
    pdb_path: str,
    save_path: Optional[str] = None,
    attribution_column: str = 'integrated_gradient',
    aggregation_method: str = 'mean',
    color_scheme: str = 'RdBu_r',
    html_output: bool = True,
    pymol_output: bool = False,
    figsize: tuple[int, int] = (800, 600)
) -> None:
    """
    Visualize integrated gradient attributions mapped onto 3D protein structure.

    This function creates interactive 3D visualizations showing how integrated gradients
    vary across the protein structure, providing spatial context for the attributions.

    Args:
        attribution_data: DataFrame containing position and attribution data
                         Expected columns: 'position', attribution_column, optionally 'sequence_name'
        pdb_path: Path to the PDB file of the protein structure
        save_path: Optional path to save the visualization (HTML or PyMOL script)
        attribution_column: Column name containing attribution values (default: 'integrated_gradient')
        aggregation_method: How to aggregate multiple values per position ('mean', 'max', 'min')
        color_scheme: Color scheme for mapping attributions ('RdBu_r', 'coolwarm', 'viridis')
        html_output: Whether to create interactive HTML output using py3Dmol
        pymol_output: Whether to create PyMOL script output
        figsize: Figure size for HTML output (width, height)
    """
    try:
        import py3Dmol
        html_available = True
    except ImportError:
        html_available = False
        if html_output:
            print("py3Dmol not available. Install with: pip install py3Dmol")
            html_output = False

    # Check if PDB file exists
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # Aggregate attribution data by position
    if 'sequence_name' in attribution_data.columns and len(attribution_data['sequence_name'].unique()) > 1:
        print(f"Aggregating attributions across sequences using {aggregation_method} method...")
        if aggregation_method == 'mean':
            position_attributions = attribution_data.groupby('position')[attribution_column].mean().reset_index()
        elif aggregation_method == 'max':
            position_attributions = attribution_data.groupby('position')[attribution_column].max().reset_index()
        elif aggregation_method == 'min':
            position_attributions = attribution_data.groupby('position')[attribution_column].min().reset_index()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    else:
        position_attributions = attribution_data[['position', attribution_column]].copy()

    # Normalize attributions for color mapping
    attr_values = position_attributions[attribution_column].values
    attr_min, attr_max = np.min(attr_values), np.max(attr_values)
    attr_range = attr_max - attr_min

    if attr_range == 0:
        print("Warning: All attribution values are the same. Using default coloring.")
        normalized_attrs = np.zeros_like(attr_values)
    else:
        normalized_attrs = (attr_values - attr_min) / attr_range

    # Create color mapping
    if color_scheme == 'RdBu_r':
        import matplotlib.cm as cm
        colormap = cm.RdBu_r
    elif color_scheme == 'coolwarm':
        import matplotlib.cm as cm
        colormap = cm.coolwarm
    elif color_scheme == 'viridis':
        import matplotlib.cm as cm
        colormap = cm.viridis
    else:
        import matplotlib.cm as cm
        colormap = cm.RdBu_r
        print(f"Unknown color scheme '{color_scheme}', using 'RdBu_r'")

    # Generate colors for each position
    colors = colormap(normalized_attrs)

    # Create position to color mapping
    position_colors = {}
    for i, (_, row) in enumerate(position_attributions.iterrows()):
        pos = int(row['position'])
        attr_val = row[attribution_column]
        color_rgba = colors[i]
        position_colors[pos] = {
            'attribution': attr_val,
            'color_rgba': color_rgba,
            'color_hex': f'#{int(color_rgba[0] * 255):02x}{int(color_rgba[1] * 255):02x}{int(color_rgba[2] * 255):02x}'
        }

    # HTML output using py3Dmol
    if html_output and html_available:
        print("Creating interactive 3D visualization...")

        # Read PDB file
        with open(pdb_path) as f:
            pdb_content = f.read()

        # Create py3Dmol view
        view = py3Dmol.view(width=figsize[0], height=figsize[1])
        view.addModel(pdb_content, 'pdb')

        # Set initial style (cartoon representation)
        view.setStyle({'cartoon': {'color': 'lightgray', 'opacity': 0.7}})

        # Color residues based on attributions
        for pos, color_data in position_colors.items():
            # Color the residue
            view.addStyle(
                {'resi': pos},
                {
                    'cartoon': {'color': color_data['color_hex'], 'opacity': 0.9},
                    'stick': {'color': color_data['color_hex'], 'radius': 0.3}
                }
            )

        # Add labels for top attributions
        top_attributions = position_attributions.nlargest(5, attribution_column)
        bottom_attributions = position_attributions.nsmallest(5, attribution_column)

        for _, row in pd.concat([top_attributions, bottom_attributions]).iterrows():
            pos = int(row['position'])
            attr_val = row[attribution_column]
            view.addLabel(
                f"Pos {pos}\n{attr_val:.4f}",
                {'resi': pos},
                {'fontSize': 12, 'fontColor': 'black', 'backgroundColor': 'white', 'opacity': 0.8}
            )

        view.zoomTo()
        view.spin(True)

        if save_path and save_path.endswith('.html'):
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PafA Protein Structure with Integrated Gradient Attributions</title>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .legend {{ display: flex; align-items: center; margin: 20px 0; }}
        .colorbar {{ width: 300px; height: 20px; background: linear-gradient(to right,
                     rgb(255,0,0), rgb(255,255,255), rgb(0,0,255)); margin-right: 20px; }}
        .legend-labels {{ display: flex; justify-content: space-between; width: 300px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PafA Protein Structure with Integrated Gradient Attributions</h1>
        <p>Interactive 3D visualization showing integrated gradient attributions mapped onto protein structure.</p>

        <div class="legend">
            <div class="colorbar"></div>
            <div>
                <div class="legend-labels">
                    <span>Detrimental ({attr_min:.4f})</span>
                    <span>Neutral</span>
                    <span>Beneficial ({attr_max:.4f})</span>
                </div>
                <div>Attribution Scale</div>
            </div>
        </div>

        <div id="molviewer" style="width: {figsize[0]}px; height: {figsize[1]}px; border: 1px solid #ccc;"></div>

        <div style="margin-top: 20px;">
            <h3>Summary Statistics:</h3>
            <ul>
                <li>Total positions colored: {len(position_colors)}</li>
                <li>Attribution range: {attr_min:.4f} to {attr_max:.4f}</li>
                <li>Aggregation method: {aggregation_method}</li>
                <li>Color scheme: {color_scheme}</li>
            </ul>
        </div>
    </div>

    <script>
        {view._make_html().split('<script>')[1].split('</script>')[0]}
    </script>
</body>
</html>"""

            with open(save_path, 'w') as f:
                f.write(html_content)
            print(f"Interactive 3D visualization saved to: {save_path}")
        else:
            view.show()

    # PyMOL script output
    if pymol_output:
        print("Creating PyMOL script...")

        pymol_script_path = save_path.replace('.html', '.pml') if save_path else 'protein_attributions.pml'

        with open(pymol_script_path, 'w') as f:
            f.write("# PyMOL script for visualizing integrated gradient attributions on PafA structure\n")
            f.write("# Generated by IGLOO analysis\n\n")

            f.write(f"load {pdb_path}\n")
            f.write("hide everything\n")
            f.write("show cartoon\n")
            f.write("color lightgray\n\n")

            f.write("# Color residues based on attributions\n")
            for pos, color_data in position_colors.items():
                attr_val = color_data['attribution']
                f.write(f"# Position {pos}: attribution = {attr_val:.4f}\n")
                f.write(f"set_color pos{pos}_color, [{color_data['color_rgba'][0]:.3f}, "
                       f"{color_data['color_rgba'][1]:.3f}, {color_data['color_rgba'][2]:.3f}]\n")
                f.write(f"color pos{pos}_color, resi {pos}\n")
                f.write(f"show sticks, resi {pos}\n\n")

            f.write("# Set view and rendering\n")
            f.write("orient\n")
            f.write("zoom\n")
            f.write("ray\n\n")

            f.write("# Attribution statistics:\n")
            f.write(f"# Min attribution: {attr_min:.4f}\n")
            f.write(f"# Max attribution: {attr_max:.4f}\n")
            f.write(f"# Positions colored: {len(position_colors)}\n")

        print(f"PyMOL script saved to: {pymol_script_path}")

    # Create a separate colorbar legend
    if save_path:
        legend_path = save_path.replace('.html', '_legend.png')
        create_attribution_colorbar(attr_min, attr_max, color_scheme, legend_path)


def create_attribution_colorbar(
    attr_min: float,
    attr_max: float,
    color_scheme: str = 'RdBu_r',
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (8, 2)
) -> None:
    """
    Create a colorbar legend for attribution values.

    Args:
        attr_min: Minimum attribution value
        attr_max: Maximum attribution value
        color_scheme: Color scheme used
        save_path: Path to save the colorbar
        figsize: Figure size (width, height)
    """
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Set up colormap
    if color_scheme == 'RdBu_r':
        cmap = cm.RdBu_r
    elif color_scheme == 'coolwarm':
        cmap = cm.coolwarm
    elif color_scheme == 'viridis':
        cmap = cm.viridis
    else:
        cmap = cm.RdBu_r

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create colorbar
    norm = Normalize(vmin=attr_min, vmax=attr_max)
    cb = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation='horizontal',
        fraction=0.8,
        aspect=30
    )

    cb.set_label('Integrated Gradient Attribution', fontsize=12, fontweight='bold')

    # Add labels
    ax.set_title('Attribution Color Scale', fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    # Add descriptive text
    ax.text(0.1, 0.3, 'Detrimental\n(Decreases Activity)',
            transform=ax.transAxes, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax.text(0.9, 0.3, 'Beneficial\n(Increases Activity)',
            transform=ax.transAxes, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Attribution colorbar saved to: {save_path}")

    plt.close()


def plot_attribution_heatmap(
    data,
    x_labels=None,
    y_labels=None,
    title="Attribution Heatmap",
    value_label="Attribution",
    vmin=None,
    vmax=None,
    center=0.0,
    figsize=(8, 4),
    output_path="attribution_heatmap.png",
    font_scale=1.0
):
    """
    Plot attribution heatmap with red-white-blue colormap.

    Args:
        data: 2D array-like data for the heatmap
        x_labels: Labels for x-axis (columns)
        y_labels: Labels for y-axis (rows)
        title: Title for the plot
        value_label: Label for the colorbar
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        center: Center value for diverging colormap
        figsize: Figure size as (width, height)
        output_path: Path to save the figure
        font_scale: Scale factor for font sizes (1.0 = normal, 1.5 = 50% larger)

    Returns:
        str: Path to the saved figure
    """
    # Set Roboto font with scalable size
    base_font_size = 14
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Roboto', 'DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': int(base_font_size * font_scale),
        'axes.titlesize': int(16 * font_scale),
        'axes.labelsize': int(14 * font_scale),
        'xtick.labelsize': int(13 * font_scale),
        'ytick.labelsize': int(13 * font_scale),
        'legend.fontsize': int(13 * font_scale)
    })

    data = np.asarray(data, dtype=float)

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    # normalization around zero
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    # red → white → blue colormap
    rwb = LinearSegmentedColormap.from_list("rwb", ["red", "white", "blue"], N=256)

    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    im = ax.imshow(data, cmap=rwb, norm=norm, aspect="auto", interpolation="nearest")

    if x_labels is not None:
        # Set ticks for every position
        ax.set_xticks(np.arange(len(x_labels)))

        # Create labels - only show every 10th position number
        tick_labels = []
        for i in range(len(x_labels)):
            if (i + 1) % 10 == 0:  # Every 10th position
                tick_labels.append(str(i + 1))
            else:
                tick_labels.append("")

        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Position")
    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    ax.set_title(title, pad=10)

    # Create colorbar with reduced spacing and custom ticks
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(value_label)

    # Set specific ticks for min, zero, and max values
    tick_values = [vmin, center, vmax]
    tick_labels = [f'{vmin:.3f}', '0.000', f'{vmax:.3f}']
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(tick_labels)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output_path


def plot_mutation_frequency_heatmap(
    data,
    wildtype_sequence,
    title="Mutation Frequency Heatmap",
    figsize=(24, 8),
    output_path="mutation_frequency_heatmap.png",
    wildtype_marker='o',
    wildtype_marker_size=50,
    cmap='Greens'
):
    """
    Plot mutation frequency heatmap with amino acids on y-axis and positions on x-axis.

    Args:
        data: DataFrame with columns ['position', 'amino_acid', 'frequency'] or 2D array
        wildtype_sequence: String of wildtype sequence to mark on heatmap
        title: Title for the plot
        figsize: Figure size as (width, height)
        output_path: Path to save the figure
        wildtype_marker: Marker style for wildtype positions ('o', 'x', 's', etc.)
        wildtype_marker_size: Size of wildtype markers
        cmap: Colormap for frequency values

    Returns:
        str: Path to the saved figure
    """

    # Set font parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Roboto', 'DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    })

    # Standard amino acids
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to pivot table
        pivot_data = data.pivot(index='amino_acid', columns='position', values='frequency')

        # Reindex to ensure all amino acids are present
        pivot_data = pivot_data.reindex(amino_acids, fill_value=0)

        # Get positions
        positions = sorted(pivot_data.columns)

    else:
        # Assume data is already a 2D array
        pivot_data = np.asarray(data)
        positions = list(range(1, pivot_data.shape[1] + 1))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Create custom white-to-dark green colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['white', 'darkgreen']
    custom_cmap = LinearSegmentedColormap.from_list('white_to_darkgreen', colors, N=256)

    # Create heatmap
    im = ax.imshow(pivot_data.values, cmap=custom_cmap, aspect='auto', interpolation='nearest')

    # Set up x-axis ticks and labels
    all_positions = range(len(positions))
    ax.set_xticks(all_positions)  # Set ticks for all positions

    # Create labels - only show every 15th position
    position_labels = []
    for i, pos in enumerate(positions):
        if i % 15 == 0:  # Every 15th position
            position_labels.append(str(pos))
        else:
            position_labels.append('')  # Empty label but tick remains

    ax.set_xticklabels(position_labels)
    ax.set_yticks(range(len(amino_acids)))
    ax.set_yticklabels(amino_acids)

    # Set labels
    ax.set_xlabel('Position', fontweight='bold')
    ax.set_ylabel('Amino Acid', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)

    # Add colorbar with only min and max labels
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Frequency', fontweight='bold')

    # Set colorbar to show only min and max values
    vmin, vmax = pivot_data.values.min(), pivot_data.values.max()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])

    # Mark wildtype positions
    if wildtype_sequence:
        wildtype_positions = []
        wildtype_amino_acids = []

        for pos_idx, aa in enumerate(wildtype_sequence):
            if aa in amino_acids and pos_idx < len(positions):
                aa_idx = amino_acids.index(aa)
                wildtype_positions.append(pos_idx)
                wildtype_amino_acids.append(aa_idx)

        # Plot wildtype markers
        if wildtype_positions:
            ax.scatter(wildtype_positions, wildtype_amino_acids,
                      marker=wildtype_marker, s=wildtype_marker_size,
                      c='red', alpha=0.8, edgecolors='white', linewidth=1.5,
                      zorder=10)

    # Grid for better readability
    ax.set_xticks(np.arange(-0.5, len(positions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(amino_acids), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150, transparent=True)
    plt.close()

    return output_path


def plot_interpolation_path_umap(
    emb_trajectory: list,
    alphas: np.ndarray,
    sequence_name: str = "Sequence",
    baseline_name: str = "Baseline",
    interpolation_method: str = 'linear',
    save_path: Optional[str] = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
) -> None:
    """Plot a 2-D UMAP projection of the interpolation path through embedding space.

    Each point along the path (one per IG alpha step) is mean-pooled to a
    single vector and then projected with UMAP for visualisation.

    Parameters
    ----------
    emb_trajectory:
        List of ``(embed_dim,)`` numpy arrays, one per alpha step.
    alphas:
        1-D array of alpha values corresponding to each trajectory point.
    sequence_name:
        Label for the target sequence (alpha=1 endpoint).
    baseline_name:
        Label for the baseline (alpha=0 endpoint).
    interpolation_method:
        Name of the interpolation method (shown in the plot title).
    save_path:
        If provided, the figure is saved to this path (PNG/PDF/…).
    umap_n_neighbors:
        ``n_neighbors`` passed to UMAP.
    umap_min_dist:
        ``min_dist`` passed to UMAP.
    random_state:
        Random seed for UMAP reproducibility.
    """
    setup_matplotlib_style()
    try:
        import umap  # type: ignore
    except ImportError:
        print("umap-learn is not installed. Run `pip install umap-learn`. Skipping UMAP plot.")
        return

    X = np.stack(emb_trajectory, axis=0).astype(np.float32)
    n_steps = X.shape[0]

    if n_steps < 4:
        print(f"Too few interpolation steps ({n_steps}) for UMAP (need ≥ 4). Skipping.")
        return

    # Detect degenerate case: sequence == baseline → all points are identical.
    # UMAP has no structure to preserve and will scatter them randomly, which is
    # misleading. Render a single-point figure instead.
    total_variance = float(np.var(X))
    if total_variance < 1e-10:
        print(f"Interpolation path is trivial (sequence == baseline). Skipping UMAP, saving single-point figure.")
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter([0], [0], s=300, marker='s', color='royalblue', zorder=3,
                   edgecolors='white', linewidth=1.5, label=f'α=0  ({baseline_name})')
        ax.scatter([0], [0], s=300, marker='*', color='crimson', zorder=3,
                   edgecolors='white', linewidth=1.5, label=f'α=1  ({sequence_name})')
        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.set_title(
            f'Interpolation path in embedding space\n'
            f'{baseline_name} → {sequence_name}   [{interpolation_method}]\n'
            f'(trivial path: sequence = baseline)',
            fontsize=12,
        )
        ax.legend(fontsize=9, loc='best')
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
            print(f"Interpolation UMAP saved to {save_path}")
        plt.close()
        return

    n_neighbors = min(umap_n_neighbors, n_steps - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(X)  # (n_steps, 2)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Connecting line
    ax.plot(coords[:, 0], coords[:, 1], color='grey', linewidth=1.0, alpha=0.45, zorder=1)

    # Scatter coloured by alpha
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=alphas, cmap='viridis',
        s=40, edgecolors='none', alpha=0.85, zorder=2,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('α  (interpolation step)', fontsize=10)

    # Endpoint markers
    ax.scatter(
        coords[0, 0], coords[0, 1],
        s=200, marker='s', color='royalblue', zorder=3,
        edgecolors='white', linewidth=1.5,
        label=f'α=0  ({baseline_name})',
    )
    ax.scatter(
        coords[-1, 0], coords[-1, 1],
        s=200, marker='*', color='crimson', zorder=3,
        edgecolors='white', linewidth=1.5,
        label=f'α=1  ({sequence_name})',
    )

    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)
    ax.set_title(
        f'Interpolation path in embedding space\n'
        f'{baseline_name} → {sequence_name}   [{interpolation_method}]',
        fontsize=12,
    )
    ax.legend(fontsize=9, loc='best')
    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Interpolation UMAP saved to {save_path}")

    plt.show()
    plt.close()


def plot_interpolation_path(
    emb_trajectory: list,
    alphas: np.ndarray,
    baseline_emb: np.ndarray,
    seq_emb: np.ndarray,
    sequence_name: str = "Sequence",
    baseline_name: str = "Baseline",
    interpolation_method: str = 'linear',
    predictions: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """Three-panel diagnostic plot for an IG interpolation path.

    Panels
    ------
    1. **Chord-plane projection** — projects every trajectory point onto the
       2-D subspace spanned by the baseline→sequence chord.  The x-axis is
       the fraction of the chord already traversed; the y-axis is the
       perpendicular deviation normalised by the chord length.
       * ``linear``          → straight diagonal line  (y = 0 everywhere)
       * ``slerp``           → smooth arc above the chord  (y > 0)
       * ``nearest_neighbor``→ staircase in x, near-zero y

    2. **Embedding norm vs α** — mean-pooled L2 norm at each step.
       Flat for slerp, midpoint-dip for linear, discrete steps for nn.

    3. **Prediction vs α** (shown when *predictions* is provided) — model
       output along the path; reveals whether the traversal is smooth or
       has unexpected jumps.

    Parameters
    ----------
    emb_trajectory:
        List of ``(D,)`` numpy arrays (mean-pooled embedding at each step).
    alphas:
        1-D array of α values for each step, length = len(emb_trajectory).
    baseline_emb:
        Mean-pooled baseline embedding ``(D,)``.
    seq_emb:
        Mean-pooled sequence embedding ``(D,)``.
    sequence_name, baseline_name:
        Labels for the α=1 and α=0 endpoints.
    interpolation_method:
        Name shown in the title.
    predictions:
        Optional 1-D array of model outputs (rescaled) at each step.
    save_path:
        If given, writes the figure to this path.
    """
    setup_matplotlib_style()
    X = np.stack(emb_trajectory, axis=0).astype(np.float64)  # (n_steps, D)
    n_steps = X.shape[0]

    # ── Degenerate case: sequence == baseline ───────────────────────────
    total_variance = float(np.var(X))
    if total_variance < 1e-10:
        n_panels = 2 if predictions is None else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        for ax in (axes if n_panels > 1 else [axes]):
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, f'Trivial path\n{baseline_name} = {sequence_name}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=13,
                    color='grey', style='italic')
        fig.text(0.5, 0.98, f'Interpolation path  [{interpolation_method}]',
                 ha='center', va='top', fontsize=14, fontweight='bold')
        fig.text(0.5, 0.92, f'{baseline_name} → {sequence_name}',
                 ha='center', va='top', fontsize=12)
        fig.subplots_adjust(top=0.83, bottom=0.12, left=0.08, right=0.96)
        if save_path:
            plt.savefig(save_path, dpi=150, transparent=True)
            print(f"Interpolation path figure saved to {save_path}")
        plt.close()
        return

    # ── Chord-plane decomposition ────────────────────────────────────────
    e0 = baseline_emb.astype(np.float64)
    e1 = seq_emb.astype(np.float64)
    chord = e1 - e0
    chord_norm = float(np.linalg.norm(chord))
    chord_unit = chord / chord_norm

    deltas = X - e0[None, :]                              # (n_steps, D)
    along = deltas @ chord_unit                           # (n_steps,) — along chord
    perp_vec = deltas - along[:, None] * chord_unit       # (n_steps, D)
    perp = np.linalg.norm(perp_vec, axis=1)               # (n_steps,) — perp deviation

    # Normalise by chord length → dimensionless coordinates in [0,1]
    along_norm = along / chord_norm
    perp_norm = perp / chord_norm

    # ── Embedding norms ──────────────────────────────────────────────────
    norms = np.linalg.norm(X, axis=1)  # (n_steps,)

    # ── Layout ──────────────────────────────────────────────────────────
    with_pred = predictions is not None
    n_panels = 3 if with_pred else 2
    fig = plt.figure(figsize=(5 * n_panels + 1, 6.2))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, n_panels, figure=fig)

    cmap = plt.get_cmap('viridis')
    colors = cmap(alphas)

    # ── Panel 1: chord-plane projection ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    # If max perpendicular deviation is < 1e-5 it is numerical noise (e.g. linear
    # interpolation in float32 leaving ~1e-7 residuals).  Show a clean "on chord" view.
    _on_chord = float(np.max(perp_norm)) < 1e-5
    if _on_chord:
        sc = ax1.scatter(along_norm, np.zeros_like(along_norm), c=alphas, cmap='viridis',
                         s=30, alpha=0.85, edgecolors='none', zorder=2)
        ax1.plot(along_norm, np.zeros_like(along_norm),
                 color='grey', linewidth=0.8, alpha=0.4, zorder=1)
    else:
        sc = ax1.scatter(along_norm, perp_norm, c=alphas, cmap='viridis',
                         s=30, alpha=0.85, edgecolors='none', zorder=2)
        ax1.plot(along_norm, perp_norm, color='grey', linewidth=0.8, alpha=0.4, zorder=1)
    # Endpoint markers
    _y0 = 0 if _on_chord else perp_norm[0]
    _y1 = 0 if _on_chord else perp_norm[-1]
    ax1.scatter(along_norm[0], _y0, s=180, marker='s', color='royalblue',
                edgecolors='white', linewidth=1.2, zorder=3, label=f'α=0  ({baseline_name})')
    ax1.scatter(along_norm[-1], _y1, s=180, marker='*', color='crimson',
                edgecolors='white', linewidth=1.2, zorder=3, label=f'α=1  ({sequence_name})')
    # Reference line: where linear interpolation lies (y=0, x in [0,1])
    ax1.axhline(0, color='steelblue', linewidth=1.2, linestyle='--', alpha=0.5,
                label='linear  (y=0)')
    cbar1 = plt.colorbar(sc, ax=ax1, pad=0.02)
    cbar1.set_label('α', fontsize=11)
    ax1.set_xlabel('Along chord  (fraction of ‖e₁−e₀‖)', fontsize=11)
    ax1.set_ylabel('Perp. deviation  (fraction of ‖e₁−e₀‖)', fontsize=11)
    ax1.set_title('Chord-plane projection', fontsize=12, fontweight='bold', pad=8)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.tick_params(labelsize=10)
    sns.despine(ax=ax1)

    # ── Panel 2: embedding norm vs α ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(alphas, norms, c=alphas, cmap='viridis', s=25, alpha=0.85,
                edgecolors='none', zorder=2)
    ax2.plot(alphas, norms, color='grey', linewidth=0.8, alpha=0.4, zorder=1)
    ax2.axhline(norms[0], color='royalblue', linewidth=1, linestyle=':', alpha=0.6,
                label=f'‖e₀‖ = {norms[0]:.3f}')
    ax2.axhline(norms[-1], color='crimson', linewidth=1, linestyle=':', alpha=0.6,
                label=f'‖e₁‖ = {norms[-1]:.3f}')
    ax2.set_xlabel('α  (interpolation step)', fontsize=11)
    ax2.set_ylabel('Mean-pooled embedding norm', fontsize=11)
    ax2.set_title('Embedding norm vs α', fontsize=12, fontweight='bold', pad=8)
    ax2.legend(fontsize=9)
    ax2.tick_params(labelsize=10)
    sns.despine(ax=ax2)

    # ── Panel 3: prediction vs α ─────────────────────────────────────────
    if with_pred:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(alphas, predictions, c=alphas, cmap='viridis', s=25,
                    alpha=0.85, edgecolors='none', zorder=2)
        ax3.plot(alphas, predictions, color='grey', linewidth=0.8, alpha=0.4, zorder=1)
        ax3.axhline(predictions[0], color='royalblue', linewidth=1, linestyle=':',
                    alpha=0.6, label=f'{baseline_name}  ({predictions[0]:.3f})')
        ax3.axhline(predictions[-1], color='crimson', linewidth=1, linestyle=':',
                    alpha=0.6, label=f'{sequence_name}  ({predictions[-1]:.3f})')
        ax3.set_xlabel('α  (interpolation step)', fontsize=11)
        ax3.set_ylabel('Model prediction (original scale)', fontsize=11)
        ax3.set_title('Prediction vs α', fontsize=12, fontweight='bold', pad=8)
        ax3.legend(fontsize=9)
        ax3.tick_params(labelsize=10)
        sns.despine(ax=ax3)

    fig.text(0.5, 0.98, f'Interpolation path  [{interpolation_method}]',
             ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.92, f'{baseline_name} → {sequence_name}',
             ha='center', va='top', fontsize=12)
    fig.subplots_adjust(top=0.83, bottom=0.12, left=0.08, right=0.96, wspace=0.45)

    if save_path:
        plt.savefig(save_path, dpi=150, transparent=True)
        print(f"Interpolation path figure saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Synthetic-evaluation plotting utilities
# (moved from scripts/evaluate_synthetic_ig.py)
# ---------------------------------------------------------------------------

_FITNESS_MODEL_CMAPS: dict[str, any] = {
    "additive":  plt.cm.Blues,
    "epistatic": plt.cm.Oranges,
    "nonlinear": plt.cm.Greens,
}
_DEFAULT_CMAP = plt.cm.Purples

# Shade intensity for each interpolation path within a model-family colormap.
_INTERP_SHADES: dict[str, float] = {
    "linear":              0.40,
    "linearinterpolation": 0.40,
    "slerp":               0.58,
    "nearest_neighbor":    0.72,
    "guided_ig":           0.88,
}

# Distinct colors for each interpolation path
_INTERP_COLORS: dict[str, str] = {
    "linear":            "#2196F3",  # blue
    "slerp":             "#FF5722",  # deep orange
    "nearest_neighbor":  "#4CAF50",  # green
    "guided_ig":         "#9C27B0",  # purple
    "gradient_x_input":  "#F44336",  # red
    "input_difference":  "#FF9800",  # amber/orange
    "attention_weights": "#009688",  # teal
}
_INTERP_MARKERS: dict[str, str] = {
    "linear":            "o",
    "slerp":             "s",
    "nearest_neighbor":  "^",
    "guided_ig":         "D",
    "gradient_x_input":  "v",
    "input_difference":  "P",
    "attention_weights": "X",
}


def _assign_colors(df: pd.DataFrame) -> list:
    """Return one RGBA colour per row.

    Hue = interpolation path (consistent with noise-vs-interp line plots).
    Lightness = noise level: low noise → light tint, high noise → full saturation.
    Falls back to fitness-model colourmap when no noise_level column is present.
    """
    import matplotlib.colors as mcolors

    has_noise = (
        "noise_level" in df.columns
        and df["noise_level"].notna().any()
    )
    noise_vals = df["noise_level"].values if has_noise else None
    noise_min  = float(np.nanmin(noise_vals)) if has_noise else 0.0
    noise_max  = float(np.nanmax(noise_vals)) if has_noise else 1.0
    noise_range = max(noise_max - noise_min, 1e-6)

    colours = []
    for i, (_, row) in enumerate(df.iterrows()):
        interp = str(row.get("interp", "unknown"))
        base_hex = _INTERP_COLORS.get(interp, "#607D8B")
        base_rgb = np.array(mcolors.to_rgb(base_hex))

        if has_noise and noise_vals is not None:
            nl = noise_vals[i]
            if np.isfinite(nl):
                t = 1.0 - (float(nl) - noise_min) / noise_range
                t = 0.20 + 0.80 * t
                colour = tuple(1.0 - t * (1.0 - c) for c in base_rgb) + (1.0,)
                colours.append(colour)
                continue
        model = str(row.get("fitness_model", "unknown"))
        cmap  = _FITNESS_MODEL_CMAPS.get(model, _DEFAULT_CMAP)
        shade = _INTERP_SHADES.get(interp, 0.55)
        colours.append(cmap(shade))
    return colours


def _config_legend_handles(df: pd.DataFrame) -> list:
    """Legend patches: one per unique interp path (full-saturation colour)."""
    import matplotlib.colors as mcolors

    has_noise = (
        "noise_level" in df.columns
        and df["noise_level"].notna().any()
        and df["noise_level"].nunique() > 1
    )

    seen: dict[str, mpatches.Patch] = {}
    for _, row in df.iterrows():
        interp = str(row.get("interp", "unknown"))
        if interp not in seen:
            base_hex = _INTERP_COLORS.get(interp, "#607D8B")
            seen[interp] = mpatches.Patch(color=base_hex, label=f"interp: {interp}")
    handles = list(seen.values())

    if has_noise:
        handles.append(mpatches.Patch(color="#444444", label="← noise=low (dark)"))
        handles.append(mpatches.Patch(color="#dddddd", label="→ noise=high (light)"))

    return handles


def _sort_df_and_get_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int, int]]]:
    """Sort *df* by fitness_model → interp → noise_level and return group boundaries.

    Returns
    -------
    sorted_df   : reset-indexed copy sorted for grouped display
    fm_groups   : list of (fitness_model_name, start_idx, end_idx_inclusive)
    """
    _fm_order = {"additive": 0, "epistatic": 1, "nonlinear": 2}
    df = df.copy()
    df["_fm_sort"] = df["fitness_model"].map(lambda x: _fm_order.get(str(x).lower(), 99))
    sort_cols = ["_fm_sort", "interp"]
    if "noise_level" in df.columns:
        df["noise_level"] = pd.to_numeric(df["noise_level"], errors="coerce")
        sort_cols.append("noise_level")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df.drop(columns=["_fm_sort"], inplace=True)

    fm_vals = df["fitness_model"].values
    n = len(fm_vals)
    groups: list[tuple[str, int, int]] = []
    i_start = 0
    for k in range(1, n):
        if fm_vals[k] != fm_vals[k - 1]:
            groups.append((str(fm_vals[i_start]), i_start, k - 1))
            i_start = k
    groups.append((str(fm_vals[i_start]), i_start, n - 1))
    return df, groups


def _make_x_positions(
    n: int,
    fm_groups: list[tuple[str, int, int]],
    gap: float = 2.0,
) -> np.ndarray:
    """Return x-axis positions for *n* bars with extra *gap* between fitness-model groups."""
    x = np.zeros(n)
    cursor = 0.0
    for g_idx, (_, g_start, g_end) in enumerate(fm_groups):
        if g_idx > 0:
            cursor += gap
        for i in range(g_start, g_end + 1):
            x[i] = cursor
            cursor += 1.0
    return x


def _apply_group_decorations(
    ax,
    fm_groups: list[tuple[str, int, int]],
    x_pos: np.ndarray | None = None,
) -> None:
    """Draw vertical separators and fitness-model header labels on *ax*."""
    y_top = ax.get_ylim()[1]
    for g_idx, (fm_name, g_start, g_end) in enumerate(fm_groups):
        if x_pos is not None:
            mid = (x_pos[g_start] + x_pos[g_end]) / 2
            sep_x = (x_pos[g_start] - 1.0) if g_start > 0 else None
            if g_start > 0 and x_pos is not None:
                sep_x = (x_pos[g_start - 1 if g_start > 0 else g_start] + x_pos[g_start]) / 2
        else:
            mid = (g_start + g_end) / 2
            sep_x = g_start - 0.5 if g_start > 0 else None
        ax.text(mid, y_top * 0.98, fm_name,
                ha="center", va="top", fontsize=9, fontweight="bold", color="dimgray",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="lightgray", alpha=0.7))
        if g_start > 0 and sep_x is not None:
            ax.axvline(sep_x, color="lightgray", lw=1.2, linestyle="-", zorder=0)


def plot_sign_accuracy_bars(
    sign_rows: list[dict],
    output_dir: Path,
    protein_prefix: str | None = None,
) -> None:
    """Bar chart of sign accuracy for each results directory.

    sign_rows : list of dicts, each with 'label', 'sign_acc_active',
                'sign_acc_wt', 'sign_acc_overall', and optionally
                'noise_level', 'interp', 'fitness_model'.
    """
    if not sign_rows:
        return

    import pandas as _pd_sa
    _df = _pd_sa.DataFrame(sign_rows)
    if "fitness_model" not in _df.columns:
        _df["fitness_model"] = _df.get("label", "unknown")
    if "interp" not in _df.columns:
        _df["interp"] = _df.get("label", "unknown")
    _df, fm_groups = _sort_df_and_get_groups(_df)
    sign_rows = _df.to_dict("records")

    acc_act  = [r["sign_acc_active"]  for r in sign_rows]
    acc_wt   = [r["sign_acc_wt"]      for r in sign_rows]
    acc_all  = [r["sign_acc_overall"] for r in sign_rows]

    n     = len(sign_rows)
    width = 0.25
    x_pos = _make_x_positions(n, fm_groups)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), 5), constrained_layout=True)
    b1 = ax.bar(x_pos - width, acc_act, width, label="Active-site (mutated+annotated)", color="steelblue",  alpha=0.85, edgecolor="k", linewidth=0.4)
    b2 = ax.bar(x_pos,         acc_wt,  width, label="WT residues (unchanged)",         color="darkorange", alpha=0.85, edgecolor="k", linewidth=0.4)
    b3 = ax.bar(x_pos + width, acc_all, width, label="Overall",                          color="seagreen",   alpha=0.85, edgecolor="k", linewidth=0.4)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                # Place label inside bar when tall, to avoid overflowing ylim=1
                if h > 0.85:
                    ax.text(bar.get_x() + bar.get_width() / 2, h - 0.02,
                            f"{h:.2f}", ha="center", va="top", fontsize=6, rotation=90)
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                            f"{h:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.axhline(0.5, color="grey", lw=1.0, linestyle="--", label="chance (0.5)")
    ax.set_xticks([])
    ax.set_ylabel("Sign accuracy (fraction correct)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    title_prot = f" [{protein_prefix}]" if protein_prefix else ""
    ax.set_title(f"Sign Correctness{title_prot}: should mutations at critical sites produce negative IG?")
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    _apply_group_decorations(ax, fm_groups, x_pos)

    fname = f"{protein_prefix}_ig_sign_accuracy.png" if protein_prefix else "synthetic_ig_sign_accuracy.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_cliff_damage(
    cliff_data: list[dict],
    output_dir: Path,
    label: str = "",
) -> None:
    """Scatter of mean |IG| at cliff-critical positions vs cumulative damage."""
    if not cliff_data:
        return

    damage  = [r["cumulative_damage"]   for r in cliff_data]
    mean_ig = [r["mean_absig_critical"] for r in cliff_data]
    above   = [r["above_cliff"]         for r in cliff_data]
    names   = [r["seq_name"]            for r in cliff_data]
    threshold = cliff_data[0].get("cliff_threshold", 1.5) if cliff_data else 1.5

    colours = ["firebrick" if a else "steelblue" for a in above]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(damage, mean_ig, c=colours, s=80, edgecolors="k", linewidths=0.5, zorder=3)
    for d, ig, n in zip(damage, mean_ig, names):
        if np.isfinite(ig):
            ax.annotate(n, (d, ig), textcoords="offset points", xytext=(4, 3), fontsize=9)
    ax.axvline(threshold, color="grey", linestyle="--", lw=1.2, label=f"cliff threshold = {threshold}")
    ax.set_xlabel("Cumulative critical damage (importance × BLOSUM penalty)")
    ax.set_ylabel("Mean |IG| at cliff-critical positions")
    ax.set_title(f"{label}\nCliff model: |IG| vs cumulative damage at critical sites")

    legend_patches = [
        mpatches.Patch(color="firebrick",  label="above cliff"),
        mpatches.Patch(color="steelblue",  label="below cliff"),
    ]
    ax.legend(handles=legend_patches + [plt.Line2D([0], [0], color="grey", lw=1.2, linestyle="--",
                                                    label=f"cliff threshold={threshold}")],
              fontsize=9)

    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    out_path = output_dir / f"{safe_label}_cliff_damage.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_metric_comparison(
    summary_df: pd.DataFrame,
    output_dir: Path,
    protein_prefix: str | None = None,
) -> None:
    """Grouped bar charts — one panel per metric, one bar per configuration."""
    metrics = ["spearman_r", "kendall_tau", "auroc", "auprc", "precision_at_k"]
    metric_labels = {
        "spearman_r":      "Spearman ρ",
        "kendall_tau":     "Kendall τ",
        "auroc":           "AUROC",
        "auprc":           "AUPRC",
        "precision_at_k":  "Precision@K",
    }

    summary_df = summary_df.copy()
    summary_df, fm_groups = _sort_df_and_get_groups(summary_df)

    n_configs = len(summary_df)
    fig, axes = plt.subplots(
        2, 3,
        figsize=(max(15, n_configs * 1.0), 10),
        constrained_layout=True,
    )
    axes = axes.flatten()
    axes[5].set_visible(False)

    colours = _assign_colors(summary_df)
    prot_handles = _config_legend_handles(summary_df)
    x_pos = _make_x_positions(n_configs, fm_groups)

    if "n_annotated" in summary_df.columns and "n_positions" in summary_df.columns:
        auprc_baseline = float(
            (summary_df["n_annotated"] / summary_df["n_positions"].replace(0, np.nan)).mean()
        )
    else:
        auprc_baseline = float("nan")
    if "precision_at_k_random" in summary_df.columns:
        pak_random_mean = float(summary_df["precision_at_k_random"].mean())
    else:
        pak_random_mean = float("nan")
    _gt_baselines = {
        "spearman_r":     (0.0,            "random (0)"),
        "kendall_tau":    (0.0,            "random (0)"),
        "auroc":          (0.5,            "random (0.5)"),
        "auprc":          (auprc_baseline, f"random ({auprc_baseline:.2f})"),
        "precision_at_k": (pak_random_mean, f"random ({pak_random_mean:.2f})"),
    }

    for panel_i, (ax, metric) in enumerate(zip(axes, metrics)):
        vals = summary_df[metric].values
        bars = ax.bar(
            x_pos, vals,
            color=colours, edgecolor="black", linewidth=0.5, alpha=0.85,
        )
        base_val, base_label = _gt_baselines.get(metric, (float("nan"), ""))
        extra_handles = []
        if np.isfinite(base_val):
            ax.axhline(base_val, color="grey", lw=1.2, linestyle="--")
            extra_handles.append(
                plt.Line2D([0], [0], color="grey", lw=1.2, linestyle="--", label=base_label)
            )
        if panel_i == 0:
            ax.legend(handles=prot_handles + extra_handles, fontsize=9, loc="upper right")
        elif extra_handles:
            ax.legend(handles=extra_handles, fontsize=9)
        _BOUNDED_01 = {"auroc", "auprc", "precision_at_k", "norm_auroc", "norm_auprc"}
        y_min = min(0, float(np.nanmin(vals)) - 0.05)
        y_max = float(np.nanmax(vals))
        y_range = max(y_max - y_min, 0.1)
        y_top = y_max + y_range * 0.28
        if metric in _BOUNDED_01:
            y_top = min(y_top, 1.0)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                # Place label inside bar when it would overflow the top of the axis
                if metric in _BOUNDED_01 and v > y_top - 0.18:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() - 0.02,
                        f"{v:.3f}",
                        ha="center", va="top", fontsize=6, rotation=90,
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6, rotation=90,
                    )
        ax.set_xticks([])
        ax.set_ylabel(metric_labels[metric], fontsize=9)
        ax.set_title(metric_labels[metric], fontsize=10, fontweight="bold")
        ax.set_ylim(bottom=y_min, top=y_top)
        ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        _apply_group_decorations(ax, fm_groups, x_pos)

    title_prot = f" [{protein_prefix}]" if protein_prefix else ""
    fig.suptitle(
        f"Synthetic IG Validation{title_prot} — Alignment with Ground-Truth Position Importance\n"
        "(Spearman/Kendall: rank correlation |IG| vs importance; "
        "AUROC/AUPRC: annotated vs unannotated positions)",
        fontsize=9,
    )

    fname = f"{protein_prefix}_ig_validation_metrics.png" if protein_prefix else "synthetic_ig_validation_metrics.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_fitness_metric_comparison(
    summary_df: pd.DataFrame,
    output_dir: Path,
    protein_prefix: str | None = None,
) -> None:
    """Grouped bar charts comparing |IG| to observed fitness effects."""
    fitness_metrics = ["fitness_spearman_r", "fitness_kendall_tau", "fitness_auroc", "fitness_auprc", "fitness_precision_at_k"]
    available = [m for m in fitness_metrics if m in summary_df.columns and summary_df[m].notna().any()]
    if not available:
        print("  [skip] No fitness-based metrics available to plot.")
        return

    metric_labels = {
        "fitness_spearman_r":   "Spearman ρ (fitness)",
        "fitness_kendall_tau":  "Kendall τ (fitness)",
        "fitness_auroc":        "AUROC (fitness)",
        "fitness_auprc":        "AUPRC (fitness)",
        "fitness_precision_at_k": "Precision@K (fitness)",
    }

    summary_df = summary_df.copy()
    summary_df, fm_groups = _sort_df_and_get_groups(summary_df)

    n_configs = len(summary_df)
    n_metrics = len(available)
    ncols = 2
    nrows = (n_metrics + 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(10, n_configs * 0.7), 5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    colours = _assign_colors(summary_df)
    prot_handles = _config_legend_handles(summary_df)
    x_pos = _make_x_positions(n_configs, fm_groups)

    if "fitness_precision_at_k_random" in summary_df.columns:
        fit_pak_random_mean = float(summary_df["fitness_precision_at_k_random"].mean())
    else:
        fit_pak_random_mean = float("nan")
    _fit_baselines = {
        "fitness_spearman_r":      (0.0, "random (0)"),
        "fitness_kendall_tau":     (0.0, "random (0)"),
        "fitness_auroc":           (0.5, "random (0.5)"),
        "fitness_auprc":           (0.5, "random (0.5)"),
        "fitness_precision_at_k":  (fit_pak_random_mean, f"random ({fit_pak_random_mean:.2f})"),
    }

    for i, (ax, metric) in enumerate(zip(axes_flat, available)):
        vals = summary_df[metric].values
        bars = ax.bar(
            x_pos, vals,
            color=colours, edgecolor="black", linewidth=0.5, alpha=0.85,
        )
        base_val, base_label = _fit_baselines.get(metric, (float("nan"), ""))
        extra_handles = []
        if np.isfinite(base_val):
            ax.axhline(base_val, color="grey", lw=1.2, linestyle="--")
            extra_handles.append(
                plt.Line2D([0], [0], color="grey", lw=1.2, linestyle="--", label=base_label)
            )
        if i == 0:
            ax.legend(handles=prot_handles + extra_handles, fontsize=9, loc="upper right")
        elif extra_handles:
            ax.legend(handles=extra_handles, fontsize=9)
        _BOUNDED_01 = {"fitness_auroc", "fitness_auprc", "fitness_precision_at_k"}
        y_min = min(0, float(np.nanmin(vals)) - 0.05)
        y_max = float(np.nanmax(vals))
        y_range = max(y_max - y_min, 0.1)
        y_top = y_max + y_range * 0.28
        if metric in _BOUNDED_01:
            y_top = min(y_top, 1.0)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                # Place label inside bar when it would overflow the top of the axis
                if metric in _BOUNDED_01 and v > y_top - 0.18:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() - 0.02,
                        f"{v:.3f}",
                        ha="center", va="top", fontsize=6, rotation=90,
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6, rotation=90,
                    )
        ax.set_xticks([])
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=9)
        ax.set_title(metric_labels.get(metric, metric), fontsize=10, fontweight="bold")
        ax.set_ylim(bottom=y_min, top=y_top)
        ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        _apply_group_decorations(ax, fm_groups, x_pos)

    for ax in axes_flat[len(available):]:
        ax.set_visible(False)

    title_prot = f" [{protein_prefix}]" if protein_prefix else ""
    fig.suptitle(
        f"Synthetic IG Validation{title_prot} — Alignment with Observed Fitness Effects\n"
        "(Spearman/Kendall: rank correlation |IG| vs mean fitness drop per position; "
        "AUROC/AUPRC: top-50% most deleterious positions as positives)",
        fontsize=9,
    )

    fname = f"{protein_prefix}_ig_fitness_metrics.png" if protein_prefix else "synthetic_ig_fitness_metrics.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_completeness_summary(
    summary_df: pd.DataFrame,
    output_dir: Path,
    protein_prefix: str | None = None,
) -> None:
    """Bar chart of completeness axiom pass-rates and normalized-IG rank metrics."""
    comp_col = "completeness_frac_pass_5pct"
    has_completeness = comp_col in summary_df.columns and summary_df[comp_col].notna().any()
    has_norm = "norm_spearman_r" in summary_df.columns and summary_df["norm_spearman_r"].notna().any()

    if not has_completeness and not has_norm:
        print("  [skip] No completeness / normalized-IG data to plot.")
        return

    df, fm_groups = _sort_df_and_get_groups(summary_df)
    n = len(df)
    colours = _assign_colors(df)
    config_handles = _config_legend_handles(df)
    x_pos = _make_x_positions(n, fm_groups)

    n_panels = int(has_completeness) + int(has_norm)
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(8, n * 0.9), 5 * n_panels),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    panel = 0
    if has_completeness:
        ax = axes[panel]; panel += 1
        tols = [("completeness_frac_pass_1pct",  "ε<1%",  0.55),
                (comp_col,                        "ε<5%",  0.68),
                ("completeness_frac_pass_10pct",  "ε<10%", 0.82)]
        w = 0.22
        offsets = [-w, 0, w]
        bar_colors = [plt.cm.Blues(s) for _, _, s in tols]
        for k, (col, lbl, _) in enumerate(tols):
            vals = df[col].values if col in df.columns else np.full(n, np.nan)
            brs = ax.bar(x_pos + offsets[k], vals, w, color=bar_colors[k],
                         edgecolor="k", linewidth=0.4, alpha=0.85, label=lbl)
            for bar, v in zip(brs, vals):
                if np.isfinite(v):
                    # Place label inside bar when tall, to avoid overflowing ylim=1
                    if v > 0.85:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.02,
                                f"{v:.2f}", ha="center", va="top", fontsize=5, rotation=90)
                    else:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                f"{v:.2f}", ha="center", va="bottom", fontsize=5, rotation=90)
        ax.set_xticks([])
        ax.set_ylabel("Fraction of sequences passing")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
        title_prot = f" [{protein_prefix}]" if protein_prefix else ""
        ax.set_title(f"Completeness Axiom Pass-Rate{title_prot}\n(Σ IG ≈ F(x)−F(x′) within tolerance)")
        ax.legend(handles=config_handles + [mpatches.Patch(color=c, label=l) for c, l in
                  [(plt.cm.Blues(s), lb) for _, lb, s in tols]], fontsize=9, loc="lower right")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        _apply_group_decorations(ax, fm_groups, x_pos)

        if "n_near_zero_sequences" in df.columns:
            for idx, (xp, (_, row)) in enumerate(zip(x_pos, df.iterrows())):
                nz = int(row.get("n_near_zero_sequences", 0))
                if nz > 0:
                    ax.text(xp, -0.06, f"⚠{nz}", ha="center", va="top",
                            fontsize=6, color="firebrick", transform=ax.transData)

    if has_norm:
        ax = axes[panel]
        raw_vals  = df["spearman_r"].values  if "spearman_r"  in df.columns else np.full(n, np.nan)
        norm_vals = df["norm_spearman_r"].values
        w = 0.35
        ax.bar(x_pos - w/2, raw_vals,  w, color=colours, edgecolor="k",
               linewidth=0.4, alpha=0.85, label="raw |IG|")
        ax.bar(x_pos + w/2, norm_vals, w, color=colours, edgecolor="k",
               linewidth=0.4, alpha=0.55, label="|IG| / |ΔF| (normalised)")
        ax.axhline(0, color="grey", lw=0.8, linestyle="--")
        ax.set_xticks([])
        ax.set_ylabel("Spearman ρ vs ground-truth importance")
        y_min = min(0, float(np.nanmin(np.concatenate([raw_vals, norm_vals]))) - 0.05)
        y_max = float(np.nanmax(np.concatenate([raw_vals, norm_vals])))
        y_range = max(y_max - y_min, 0.1)
        ax.set_ylim(bottom=y_min, top=y_max + y_range * 0.28)
        ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
        title_prot = f" [{protein_prefix}]" if protein_prefix else ""
        ax.set_title(f"Raw vs Normalised |IG| Rank Correlation{title_prot}\n"
                     "(normalise by |F(x)−F(x′)| to compare across sequences)")
        ax.legend(handles=config_handles, fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        _apply_group_decorations(ax, fm_groups, x_pos)

    fname = f"{protein_prefix}_completeness.png" if protein_prefix else "synthetic_completeness.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_precision_at_k(
    summary_df: pd.DataFrame,
    output_dir: Path,
    protein_prefix: str | None = None,
) -> None:
    """Bar chart of Precision@K per run, with per-run random baseline markers."""
    col = "precision_at_k"
    if col not in summary_df.columns or summary_df[col].isna().all():
        print("  [skip] No precision_at_k data to plot.")
        return

    df, fm_groups = _sort_df_and_get_groups(summary_df)
    n = len(df)

    colours = _assign_colors(df)
    config_handles = _config_legend_handles(df)
    x_pos = _make_x_positions(n, fm_groups)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), 5), constrained_layout=True)
    vals = df[col].values
    bars = ax.bar(x_pos, vals, color=colours, edgecolor="black", linewidth=0.5, alpha=0.85)

    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    if "precision_at_k_random" in df.columns:
        rand_vals = df["precision_at_k_random"].values
        for xp, rv in zip(x_pos, rand_vals):
            if np.isfinite(rv):
                ax.plot(xp, rv, marker="_", color="grey", markersize=12, markeredgewidth=1.5)
        rand_handle = plt.Line2D(
            [0], [0], color="grey", marker="_", linestyle="none",
            markersize=10, markeredgewidth=1.5, label="random (K/L)"
        )
        config_handles = config_handles + [rand_handle]

    ax.set_xticks([])
    ax.set_ylabel("Precision@K  (K = n_annotated)")
    y_max = float(np.nanmax(vals))
    ax.set_ylim(0, min(1.0, max(1.0, y_max)) * 1.28)
    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    title_prot = f" [{protein_prefix}]" if protein_prefix else ""
    ax.set_title(
        f"Precision@K{title_prot}\n"
        "Fraction of top-K |IG| positions that are annotated; K = n_annotated (grey tick = random baseline)"
    )
    ax.legend(handles=config_handles, fontsize=9, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    _apply_group_decorations(ax, fm_groups, x_pos)

    fname = f"{protein_prefix}_precision_at_k.png" if protein_prefix else "synthetic_precision_at_k.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_noise_vs_interp(
    summary_df: pd.DataFrame,
    output_dir: Path,
    protein: str,
    fitness_model: str,
    protein_prefix: str | None = None,
    split: str = "all",
) -> None:
    """Line plots of key metrics vs noise level, coloured by interpolation path."""
    if "noise_level" not in summary_df.columns:
        return

    df = summary_df[
        (summary_df["protein"] == protein)
        & (summary_df["fitness_model"] == fitness_model)
    ].copy()
    if df.empty:
        return

    df = df[df["noise_level"].notna()]
    if df.empty:
        return

    df["noise_level"] = df["noise_level"].astype(float)

    metrics = [
        ("spearman_r",                "Spearman ρ (GT)",          0.0),
        ("auroc",                     "AUROC (GT)",                0.5),
        ("auprc",                     "AUPRC (GT)",                None),
        ("pv_gt_spearman_mean",       "Spearman ρ (per-var GT)",  0.0),
        ("pv_gt_spearman_median",     "Median ρ (per-var GT)",    0.0),
        ("fitness_spearman_r",        "Spearman ρ (fitness)",     0.0),
        ("fitness_auroc",             "AUROC (fitness)",           0.5),
        ("pvfc_pooled_spearman",      "ρ (pooled fit-contrib)",   0.0),
        ("pvfc_per_seq_spearman_mean","ρ (per-seq fit-contrib)",  0.0),
        ("sign_acc_active",           "Sign acc (active-site)",   0.5),
        ("completeness_frac_pass_5pct", "Completeness (5% pass)", None),
        ("precision_at_k",            "Precision@K",              None),
    ]
    metrics = [(col, lbl, base) for col, lbl, base in metrics if col in df.columns]
    if not metrics:
        return

    _has_near_zero = (
        "n_near_zero_sequences" in df.columns
        and "n_completeness_sequences" in df.columns
    )

    n_panels = len(metrics)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 3.5),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()

    interps = sorted(df["interp"].unique())
    noise_levels = sorted(df["noise_level"].unique())

    # Mapping: main metric column → companion PolyAlanine column (per-sequence only)
    _POLYALA_COMPANION: dict[str, str] = {
        "pv_gt_spearman_mean":       "polyala_pv_gt_spearman",
        "pvfc_per_seq_spearman_mean": "polyala_pvfc_per_seq_spearman",
    }

    for ax_idx, (col, label, baseline) in enumerate(metrics):
        ax = axes_flat[ax_idx]
        for interp in interps:
            sub = df[df["interp"] == interp].sort_values("noise_level")
            if sub.empty:
                continue
            color  = _INTERP_COLORS.get(interp, "#607D8B")
            marker = _INTERP_MARKERS.get(interp, "o")
            vals = sub[col].values.astype(float)
            ax.plot(
                sub["noise_level"].values, vals,
                color=color, marker=marker, linewidth=1.8, markersize=6,
                label=interp, alpha=0.88,
            )

        # ── PolyAlanine reference lines (one dashed line per interp) ──────────
        poly_col = _POLYALA_COMPANION.get(col)
        if poly_col is not None and poly_col in df.columns:
            _added_poly_legend = False
            for interp in interps:
                sub = df[df["interp"] == interp].sort_values("noise_level")
                if sub.empty or sub[poly_col].isna().all():
                    continue
                color  = _INTERP_COLORS.get(interp, "#607D8B")
                marker = _INTERP_MARKERS.get(interp, "o")
                lbl = "PolyAlanine" if not _added_poly_legend else "_nolegend_"
                _added_poly_legend = True
                ax.plot(
                    sub["noise_level"].values,
                    sub[poly_col].values.astype(float),
                    color=color, marker=marker, linewidth=1.0, markersize=4,
                    linestyle="--", alpha=0.55, label=lbl,
                )

        if baseline is not None:
            ax.axhline(baseline, color="grey", lw=1.0, linestyle="--", alpha=0.7)
        ax.set_xlabel("Noise level", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(noise_levels)
        ax.set_xticklabels([str(n) for n in noise_levels], fontsize=9, rotation=45)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

        if _has_near_zero:
            ax2 = ax.twinx()
            for interp in interps:
                sub = df[df["interp"] == interp].sort_values("noise_level")
                if sub.empty:
                    continue
                total = sub["n_near_zero_sequences"] + sub["n_completeness_sequences"]
                frac = np.where(
                    total > 0,
                    sub["n_near_zero_sequences"].values / total.values,
                    0.0,
                )
                color = _INTERP_COLORS.get(interp, "#607D8B")
                ax2.plot(
                    sub["noise_level"].values, frac,
                    color=color, linewidth=1.0, linestyle=":",
                    alpha=0.5, marker=None,
                )
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("near-zero frac ↑ = unreliable", fontsize=6, color="dimgrey")
            ax2.tick_params(axis="y", labelsize=6, colors="dimgrey")
            ax2.spines["right"].set_color("dimgrey")
            ax2.spines["right"].set_linestyle(":")
            mean_frac = (
                df.groupby("noise_level").apply(
                    lambda g: (g["n_near_zero_sequences"].sum()
                               / max((g["n_near_zero_sequences"] + g["n_completeness_sequences"]).sum(), 1))
                ).reindex(noise_levels).values
            )
            for ni, (nl, mf) in enumerate(zip(noise_levels, mean_frac)):
                if mf > 0.3:
                    x_lo = nl - (noise_levels[1] - noise_levels[0]) * 0.4 if ni > 0 else nl - 0.02
                    x_hi = nl + (noise_levels[min(ni + 1, len(noise_levels) - 1)] - nl) * 0.4
                    ax.axvspan(x_lo, x_hi, color="red", alpha=0.06, zorder=0)

        if ax_idx == 0:
            leg = ax.legend(title="Interp path", fontsize=9, title_fontsize=9)
            if _has_near_zero:
                leg.get_frame().set_alpha(0.9)
                ax.plot([], [], color="dimgrey", linewidth=1.0, linestyle=":",
                        alpha=0.7, label="near-zero frac (right axis)")
                ax.legend(title="Interp path", fontsize=9, title_fontsize=9)

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    prot_label = protein_prefix or f"{protein}_{fitness_model}"
    split_label = f" [{split}]" if split and split != "all" else ""
    fig.suptitle(
        f"IG quality vs noise level — {protein} / {fitness_model}{split_label}\n"
        f"(lines = interpolation paths; dashed grey = random baseline; "
        f"dashed thin = PolyAlanine reference; "
        f"dotted right axis = near-zero fraction; red shading = >30% unreliable)",
        fontsize=9,
    )
    fname = f"{prot_label}_noise_vs_interp.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_noise_vs_interp_boxplots(
    summary_df: pd.DataFrame,
    output_dir: Path,
    protein: str,
    fitness_model: str,
    protein_prefix: str | None = None,
    split: str = "all",
) -> None:
    """Grouped distribution plots of key metrics vs noise level, coloured by interp path.

    For metrics that are means over per-sequence values (e.g. pv_gt_spearman_mean),
    the stored ``_std`` column is used to show mean ± 1 SD — giving a meaningful
    distribution even with a single seed.  For all other metrics, raw values per
    run are shown as scatter dots (box when multiple seeds are available).
    Companion to :func:`plot_noise_vs_interp` line plots.
    """
    if "noise_level" not in summary_df.columns:
        return

    df = summary_df[
        (summary_df["protein"] == protein)
        & (summary_df["fitness_model"] == fitness_model)
    ].copy()
    if df.empty:
        return

    df = df[df["noise_level"].notna()]
    if df.empty:
        return

    df["noise_level"] = df["noise_level"].astype(float)

    metrics = [
        ("spearman_r",                 "Spearman ρ (GT)",           0.0),
        ("auroc",                      "AUROC (GT)",                 0.5),
        ("auprc",                      "AUPRC (GT)",                 None),
        ("pv_gt_spearman_mean",        "Spearman ρ (per-var GT)\nmean ± 1 SD over seqs",   0.0),
        ("pv_gt_spearman_median",      "Median ρ (per-var GT)",     0.0),
        ("fitness_spearman_r",         "Spearman ρ (fitness)",      0.0),
        ("fitness_auroc",              "AUROC (fitness)",            0.5),
        ("pvfc_pooled_spearman",       "ρ (pooled fit-contrib)",    0.0),
        ("pvfc_per_seq_spearman_mean", "ρ (per-seq fit-contrib)\nmean ± 1 SD over seqs",   0.0),
        ("sign_acc_active",            "Sign acc (active-site)",    0.5),
        ("completeness_frac_pass_5pct","Completeness (5% pass)",    None),
        ("precision_at_k",             "Precision@K",               None),
    ]
    metrics = [(col, lbl, base) for col, lbl, base in metrics if col in df.columns]
    if not metrics:
        return

    interps = sorted(df["interp"].unique())
    noise_levels = sorted(df["noise_level"].unique())
    n_interps = len(interps)
    n_noise = len(noise_levels)

    # Mapping: main metric column → companion PolyAlanine column (per-sequence only)
    _POLYALA_COMPANION: dict[str, str] = {
        "pv_gt_spearman_mean":        "polyala_pv_gt_spearman",
        "pvfc_per_seq_spearman_mean":  "polyala_pvfc_per_seq_spearman",
    }

    n_panels = len(metrics)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 3.8),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()

    if n_interps > 0:
        width = 0.8 / n_interps
        half_span = 0.4 - width / 2
        offsets = np.linspace(-half_span, half_span, n_interps)
    else:
        width = 0.4
        offsets = np.array([0.0])

    legend_handles: list = []

    for ax_idx, (col, label, baseline) in enumerate(metrics):
        ax = axes_flat[ax_idx]

        # Detect whether this metric has a stored std (per-sequence distribution)
        std_col = col.replace("_mean", "_std") if col.endswith("_mean") else None
        has_std = std_col is not None and std_col in df.columns

        for i_interp, interp in enumerate(interps):
            color = _INTERP_COLORS.get(interp, "#607D8B")
            sub = df[df["interp"] == interp]

            means, stds, scat_vals, positions = [], [], [], []
            box_data, box_positions = [], []

            for j_noise, nl in enumerate(noise_levels):
                grp = sub[sub["noise_level"] == nl]
                vals = grp[col].dropna().values.astype(float)
                if len(vals) == 0:
                    continue
                pos = j_noise + offsets[i_interp]
                if has_std:
                    # Use stored mean ± std to represent per-sequence distribution
                    sd_vals = grp[std_col].dropna().values.astype(float)
                    for v, s in zip(vals, sd_vals):
                        means.append(v)
                        stds.append(s)
                        positions.append(pos)
                else:
                    if len(vals) > 1:
                        box_data.append(vals)
                        box_positions.append(pos)
                    else:
                        scat_vals.append(vals[0])
                        positions.append(pos)

            if has_std and means:
                ax.errorbar(
                    positions, means,
                    yerr=stds,
                    fmt="o",
                    color=color,
                    ecolor=color,
                    alpha=0.8,
                    capsize=3,
                    markersize=4,
                    linewidth=0,
                    elinewidth=1.2,
                    capthick=1.2,
                )
            if box_data:
                ax.boxplot(
                    box_data,
                    positions=box_positions,
                    widths=width * 0.85,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.6),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color=color, alpha=0.8),
                    capprops=dict(color=color, alpha=0.8),
                    flierprops=dict(marker="x", color=color, alpha=0.5, markersize=3),
                    manage_ticks=False,
                )
            if scat_vals:
                ax.scatter(
                    positions, scat_vals,
                    color=color, s=20, alpha=0.8, zorder=3,
                )

            if ax_idx == 0:
                legend_handles.append(
                    plt.matplotlib.patches.Patch(
                        facecolor=color, alpha=0.7, label=interp
                    )
                )

        # ── PolyAlanine reference markers (star per interp per noise level) ──
        poly_col = _POLYALA_COMPANION.get(col)
        if poly_col is not None and poly_col in df.columns:
            for i_interp, interp in enumerate(interps):
                color = _INTERP_COLORS.get(interp, "#607D8B")
                sub = df[df["interp"] == interp]
                px, py = [], []
                for j_noise, nl in enumerate(noise_levels):
                    grp = sub[sub["noise_level"] == nl]
                    pvals = grp[poly_col].dropna().values.astype(float)
                    if len(pvals) == 0:
                        continue
                    pos = j_noise + offsets[i_interp]
                    px.append(pos)
                    py.append(float(np.nanmean(pvals)))
                if px:
                    ax.scatter(px, py, color=color, marker="*", s=60, alpha=0.7,
                               zorder=4, edgecolors="none")

        if baseline is not None:
            ax.axhline(baseline, color="grey", lw=1.0, linestyle="--", alpha=0.7)

        ax.set_xlabel("Noise level", fontsize=10)
        ax.set_ylabel(label.split("\n")[0], fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(range(n_noise))
        ax.set_xticklabels([str(nl) for nl in noise_levels], fontsize=9, rotation=45)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

        if ax_idx == 0 and legend_handles:
            ax.legend(
                handles=legend_handles,
                title="Interp path",
                fontsize=9,
                title_fontsize=9,
            )

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    prot_label = protein_prefix or f"{protein}_{fitness_model}"
    split_label = f" [{split}]" if split and split != "all" else ""
    fig.suptitle(
        f"IG quality vs noise level — {protein} / {fitness_model}{split_label}\n"
        f"(dots = runs; error bars on per-seq metrics = ±1 SD over sequences; "
        f"★ = PolyAlanine reference; dashed grey = random baseline)",
        fontsize=11,
    )
    fname = f"{prot_label}_noise_vs_interp_boxplots.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_position_comparison(
    results_dir: Path,
    protein: str,
    gt: dict,
    ig_vec: np.ndarray,
    output_dir: Path,
    label: str = "",
) -> None:
    """Bar chart of |IG| vs ground-truth importance per position for one run."""
    from matplotlib.patches import Patch

    L = gt["sequence_length"]
    positions = np.arange(1, L + 1)
    gt_vec = gt["importance_vector"]
    mask = gt["annotated_mask"]

    fig, axes = plt.subplots(2, 1, figsize=(max(14, L // 3), 6),
                             sharex=True, constrained_layout=True)

    colours = np.where(mask, "steelblue", "lightgrey")
    axes[0].bar(positions, ig_vec, color=colours, edgecolor="none", width=0.8)
    axes[0].set_ylabel("|IG| (mean abs gradient)", fontsize=8)
    axes[0].set_title(f"{label} — |IG| per position", fontsize=9)
    axes[0].legend(
        handles=[Patch(color="steelblue", label="annotated"),
                 Patch(color="lightgrey", label="unannotated")],
        fontsize=7,
    )

    gt_colours = np.where(mask, "darkorange", "lightgrey")
    axes[1].bar(positions, gt_vec, color=gt_colours, edgecolor="none", width=0.8)
    axes[1].set_ylabel("Ground-truth importance", fontsize=8)
    axes[1].set_title("Ground-truth importance per position", fontsize=9)
    axes[1].set_xlabel("Position (1-indexed)", fontsize=8)
    axes[1].legend(
        handles=[Patch(color="darkorange", label="annotated"),
                 Patch(color="lightgrey", label="unannotated")],
        fontsize=7,
    )

    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    out_path = output_dir / f"{safe_label}_per_position.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"  Saved: {out_path}")


# ===========================================================================
# IG Attribution Quality Figures
# ===========================================================================

INTERP_COLORS  = {
    "linear":           "#1D3557",
    "nearest_neighbor": "#E63946",
    "slerp":            "#2A9D8F",
    "gradient_x_input": "#F4A261",
    "input_difference": "#A8DADC",
}
INTERP_LABELS  = {
    "linear":           "Linear",
    "nearest_neighbor": "Nearest neighbor",
    "slerp":            "Slerp",
    "gradient_x_input": "Gradient × input",
    "input_difference": "Input difference",
}
INTERP_MARKERS = {
    "linear":           "o",
    "nearest_neighbor": "s",
    "slerp":            "^",
    "gradient_x_input": "D",
    "input_difference": "P",
}
CAT_COLORS = {
    "active_site":  "#E63946",
    "metal_binding":"#F4A261",
    "signalling":   "#8338EC",
    "binding_site": "#06D6A0",
    "structural":   "#457B9D",
    "folding":      "#FFBE0B",
    "background":   "#ADB5BD",
}
CAT_LABELS = {
    "active_site":  "Active site",
    "metal_binding":"Metal binding",
    "signalling":   "Signalling (NLS)",
    "binding_site": "Binding site",
    "structural":   "Structural",
    "folding":      "Folding",
    "background":   "Background",
}


def _style_ax(ax: "plt.Axes", title: str, xlabel: str = "Noise level",
              ylabel: str = "", nn_slerp_cutoff: "float | None" = None) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(FIG_BG)
    if nn_slerp_cutoff is not None:
        ax.axvline(nn_slerp_cutoff, color="#888", ls=":", lw=1.2, alpha=0.5)


def _plot_metric_line(
    ax: "plt.Axes",
    data: dict,
    metric: str,
    hline: "float | None" = None,
    nn_slerp_cutoff: "float | None" = None,
    errors: "dict | None" = None,
) -> None:
    """Plot one metric line per interpolation method with optional ±1σ seed bands."""
    for interp, noise_dict in sorted(data.items()):
        xs  = sorted(noise_dict)
        ys  = [noise_dict[n].get(metric, float("nan")) for n in xs]
        col = INTERP_COLORS.get(interp, "#888")
        mk  = INTERP_MARKERS.get(interp, "o")
        lbl = INTERP_LABELS.get(interp, interp)
        ax.plot(xs, ys, color=col, marker=mk, lw=2.0, ms=7, alpha=0.9, label=lbl)
        if errors is not None:
            yerr  = [errors.get(interp, {}).get(n, {}).get(metric, 0.0) for n in xs]
            ys_lo = [y - e for y, e in zip(ys, yerr)]
            ys_hi = [y + e for y, e in zip(ys, yerr)]
            ax.fill_between(xs, ys_lo, ys_hi, color=col, alpha=0.18, linewidth=0)
    if hline is not None:
        ax.axhline(hline, color="grey", ls="--", lw=1, alpha=0.5)
    if nn_slerp_cutoff is not None:
        ax.axvline(nn_slerp_cutoff, color="#888", ls=":", lw=1.2, alpha=0.5)


def _interp_legend(interps: list) -> list:
    return [
        Line2D([0], [0],
               color=INTERP_COLORS.get(i, "#888"),
               marker=INTERP_MARKERS.get(i, "o"),
               lw=2, ms=7, label=INTERP_LABELS.get(i, i))
        for i in interps
    ]


def make_figure(
    protein: str,
    fitness_model: str,
    data: dict,
    output_dir: "Path",
    dpi: int = 150,
    split_label: "str | None" = None,
    errors: "dict | None" = None,
    metric_family: str = "spearman",
) -> None:
    """Generate and save the mixed attribution quality figure for one
    (protein × fitness_model) combination.

    Parameters
    ----------
    protein : Protein name string.
    fitness_model : One of 'additive', 'epistatic', 'nonlinear'.
    data : interp -> noise_level -> metrics_dict.
    output_dir : Directory where the PNG will be saved.
    dpi : Figure DPI.
    split_label : If 'trainval' or 'test', appended to title/filename.
    errors : interp -> noise -> metric -> std  (multi-seed error bands).
    metric_family : 'spearman' or 'ndcg'.
    """
    from pathlib import Path as _Path

    sfx   = ""
    track = "|IG|"
    sfx_s   = "_signed"
    track_s = "Signed IG"
    interps = sorted(data.keys())

    _is_ndcg       = metric_family == "ndcg"
    _global_key    = "ndcg"                if _is_ndcg else "spearman"
    _global_key_s  = "ndcg_signed"         if _is_ndcg else "spearman_signed"
    _pv_mean_key   = "pv_ndcg_mean"        if _is_ndcg else "pv_mean"
    _pv_std_key    = "pv_ndcg_std"         if _is_ndcg else "pv_std"
    _pv_vals_key   = "pv_ndcg_vals"        if _is_ndcg else "pv_vals"
    _pv_mean_key_s = "pv_ndcg_mean_signed" if _is_ndcg else "pv_mean_signed"
    _pv_std_key_s  = "pv_ndcg_std_signed"  if _is_ndcg else "pv_std_signed"
    _pv_vals_key_s = "pv_ndcg_vals_signed" if _is_ndcg else "pv_vals_signed"
    _mut_sp_key    = "mutation_ndcg"            if _is_ndcg else "mutation_spearman"
    _mut_rho_key   = "mutation_ndcg_vals"       if _is_ndcg else "mutation_rho_vals"
    _fitpv_key_s   = "pv_fitness_ndcg_signed"   if _is_ndcg else "pv_fitness_spearman_signed"
    _prec_k_key    = "ap_graded"                if _is_ndcg else "prec_k"
    _prec_k_lbl    = "AP@R (graded relevance)"  if _is_ndcg else "Precision@K"
    _prec_k_title  = (
        "AP@R (graded relevance)\n(weights background=0, ann. positions by importance)"
        if _is_ndcg else "Precision@K\n(K = n annotated positions)"
    )
    _metric_label   = "NDCG"  if _is_ndcg else "Spearman ρ"
    _sp_hline       = None    if _is_ndcg else 0.0
    _fname_mfx      = "_ndcg" if _is_ndcg else ""
    _model_test_key = "model_test_ndcg"     if _is_ndcg else "model_test_spearman"
    _model_val_key  = "model_val_ndcg"      if _is_ndcg else "model_val_spearman"
    _model_test_lbl = "Model test NDCG"     if _is_ndcg else "Model test ρ"
    _model_val_lbl  = "Model val NDCG"      if _is_ndcg else "Model val ρ (CV mean)"

    non_linear_noises: list = []
    for interp, noise_dict in data.items():
        if interp != "linear":
            non_linear_noises.extend(noise_dict.keys())
    nn_cutoff = max(non_linear_noises) if non_linear_noises else None

    if protein in _PROTEIN_DEFS:
        cat_weights = _PROTEIN_DEFS[protein].category_weights
        ann_cats = sorted(cat_weights, key=cat_weights.__getitem__)
        CATS     = ["background"] + ann_cats
        top_cat  = ann_cats[-1]
    else:
        sample_nd = next(iter(data.values()), {})
        sample_cm = next(iter(sample_nd.values()), {}).get("cat_means_signed", {})
        ann_cats  = sorted(k for k in sample_cm if k != "background")
        CATS      = ["background"] + ann_cats
        top_cat   = ann_cats[-1] if ann_cats else "binding_site"

    fig = plt.figure(figsize=(22, 31), facecolor=FIG_BG)
    gs  = gridspec.GridSpec(
        6, 4, figure=fig,
        hspace=0.54, wspace=0.38,
        left=0.06, right=0.97, top=0.95, bottom=0.03,
    )
    _split_tag = f" [{split_label.upper()} split]" if split_label else ""
    fig.suptitle(
        f"{protein} {fitness_model.capitalize()}{_split_tag} — IG Attribution Quality "
        f"[{metric_family.upper()}] "
        f"(|IG| rows 0/1/3/4 · Signed IG rows 2/5) by Interpolation Method Across Noise Levels",
        fontsize=13, fontweight="bold", color="#1D3557", y=0.97,
    )

    kw = dict(nn_slerp_cutoff=nn_cutoff, errors=errors)

    # ── Row 0: headline metrics ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _plot_metric_line(ax, data, _global_key, hline=_sp_hline, **kw)
    _style_ax(ax, f"Global {_metric_label}\n({track} vs GT importance)",
              ylabel=f"{_metric_label} (↑ better)")
    leg = ax.legend(handles=_interp_legend(interps), fontsize=11)
    leg.get_frame().set_alpha(0.5)
    leg.set_zorder(20)

    ax = fig.add_subplot(gs[0, 1])
    _plot_metric_line(ax, data, f"auroc{sfx}", hline=0.5, **kw)
    _style_ax(ax, "AUROC\n(annotated vs background)", ylabel="AUROC (↑ better)")
    ax.legend(handles=_interp_legend(interps), fontsize=11)

    ax = fig.add_subplot(gs[0, 2])
    _plot_metric_line(ax, data, f"auprc{sfx}", **kw)
    _style_ax(ax, "AUPRC\n(annotated vs background)", ylabel="AUPRC (↑ better)")
    ax.legend(handles=_interp_legend(interps), fontsize=11)

    ax = fig.add_subplot(gs[0, 3])
    _plot_metric_line(ax, data, f"{_prec_k_key}{sfx}", **kw)
    _style_ax(ax, _prec_k_title, ylabel=f"{_prec_k_lbl} (↑ better)")
    ax.legend(handles=_interp_legend(interps), fontsize=11)

    # ── Row 1: secondary metrics ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0:2])
    for interp, noise_dict in sorted(data.items()):
        xs   = sorted(noise_dict)
        ys_m = [noise_dict[n].get(_pv_mean_key, float("nan")) for n in xs]
        ys_s = [noise_dict[n].get(_pv_std_key,  float("nan")) for n in xs]
        col  = INTERP_COLORS.get(interp, "#888")
        mk   = INTERP_MARKERS.get(interp, "o")
        ax.plot(xs, ys_m, color=col, marker=mk, lw=2.0, ms=7, alpha=0.9,
                label=INTERP_LABELS.get(interp, interp))
        ys_lo = [m - s for m, s in zip(ys_m, ys_s)]
        ys_hi = [m + s for m, s in zip(ys_m, ys_s)]
        ax.fill_between(xs, ys_lo, ys_hi, alpha=0.12, color=col)
        if errors is not None:
            seed_err = [errors.get(interp, {}).get(n, {}).get(_pv_mean_key, 0.0) for n in xs]
            ax.fill_between(xs,
                            [m - e for m, e in zip(ys_m, seed_err)],
                            [m + e for m, e in zip(ys_m, seed_err)],
                            alpha=0.25, color=col, linewidth=0)
    if not _is_ndcg:
        ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.5)
    if nn_cutoff:
        ax.axvline(nn_cutoff, color="#888", ls=":", lw=1.2, alpha=0.5)
    _style_ax(ax, f"Per-variant {_metric_label} ({track})\n(mean ± std across sequences)",
              ylabel=f"{_metric_label} (↑ better)")
    ax.legend(fontsize=11)

    ax = fig.add_subplot(gs[1, 2:4])
    _plot_metric_line(ax, data, "sign_acc", hline=0.5, **kw)
    _style_ax(ax, "Sign accuracy\n(mutated annotated positions)",
              ylabel="Fraction correct (↑ better)")
    ax.set_ylim(0, 1)
    ax.legend(handles=_interp_legend(interps), fontsize=11)

    # ── Row 2: category means (signed IG) ────────────────────────────────────
    cat_means_key = f"cat_means{sfx_s}"
    interps_r2    = [i for i in ("linear", "nearest_neighbor", "slerp") if i in interps]
    _short = {"background": "bg", "active_site": "active", "metal_binding": "metal",
              "signalling": "NLS", "binding_site": "binding", "structural": "struct",
              "folding": "fold"}
    _expected_order = " < ".join(["bg"] + [_short.get(c, c) for c in ann_cats])
    _top_label      = _short.get(top_cat, top_cat)
    for col_idx, interp in enumerate(interps_r2[:3]):
        ax = fig.add_subplot(gs[2, col_idx])
        noise_dict = data[interp]
        xs = sorted(noise_dict)
        for cat in CATS:
            ys = [noise_dict[n].get(cat_means_key, {}).get(cat, float("nan")) for n in xs]
            ax.plot(xs, ys, color=CAT_COLORS[cat], marker=INTERP_MARKERS.get(interp, "o"),
                    lw=2.0, ms=7, alpha=0.9, label=CAT_LABELS[cat])
        _style_ax(
            ax,
            f"Category mean {track_s} — {INTERP_LABELS.get(interp, interp)}\n"
            f"(expected: {_expected_order})",
            ylabel=f"Mean {track_s} ({_top_label} ↑, bg ↓)",
        )
        ax.legend(fontsize=11)

    _interps_alt = [i for i in ("gradient_x_input", "input_difference") if i in interps]
    _gs_alt = gridspec.GridSpecFromSubplotSpec(
        len(_interps_alt) if _interps_alt else 1, 1,
        subplot_spec=gs[2, 3], hspace=0.75,
    )
    _cat_legend_handles = [
        Line2D([0], [0], color=CAT_COLORS[cat], lw=2, label=CAT_LABELS[cat])
        for cat in CATS
    ]
    for _alt_idx, interp_alt in enumerate(_interps_alt):
        ax = fig.add_subplot(_gs_alt[_alt_idx, 0])
        noise_dict = data[interp_alt]
        xs = sorted(noise_dict)
        for cat in CATS:
            ys = [noise_dict[n].get(cat_means_key, {}).get(cat, float("nan")) for n in xs]
            ax.plot(xs, ys, color=CAT_COLORS[cat],
                    marker=INTERP_MARKERS.get(interp_alt, "o"),
                    lw=2.0, ms=7, alpha=0.9, label=CAT_LABELS[cat])
        _style_ax(
            ax,
            f"Cat. mean {track_s} — {INTERP_LABELS.get(interp_alt, interp_alt)}\n"
            f"(expected: {_expected_order})",
            ylabel=f"Mean {track_s}",
        )
        ax.legend(handles=_cat_legend_handles, fontsize=10)

    # ── Row 5: category separation ratio (|IG|) ───────────────────────────────
    _ANN_CATS_SEP = [c for c in CATS if c != "background"]
    _sep_abs_key  = "cat_means"

    def _sep_ratios_for(interp: str):
        nd  = data.get(interp, {})
        xs  = sorted(nd)
        out = {c: [] for c in _ANN_CATS_SEP}
        for n in xs:
            cm = nd[n].get(_sep_abs_key, {})
            bg = cm.get("background", float("nan"))
            for c in _ANN_CATS_SEP:
                cv = cm.get(c, float("nan"))
                out[c].append(
                    cv / bg if (np.isfinite(cv) and np.isfinite(bg) and bg > 0)
                    else float("nan")
                )
        return xs, out

    for col_idx, interp in enumerate(interps_r2[:3]):
        ax = fig.add_subplot(gs[5, col_idx])
        xs, ratios = _sep_ratios_for(interp)
        for cat in _ANN_CATS_SEP:
            ax.plot(xs, ratios[cat], color=CAT_COLORS[cat],
                    marker=INTERP_MARKERS.get(interp, "o"),
                    lw=2.0, ms=7, alpha=0.9, label=CAT_LABELS[cat])
        ax.axhline(1.0, color="#ADB5BD", ls="--", lw=1, alpha=0.7)
        _style_ax(
            ax,
            f"Category separation — {INTERP_LABELS.get(interp, interp)}\n"
            f"(mean |IG| / background mean; >1 = above bg)",
            ylabel="Separation ratio (↑ better)",
        )
        ax.legend(fontsize=11)

    _gs_r5c3 = gridspec.GridSpecFromSubplotSpec(
        max(len(_interps_alt), 1), 1,
        subplot_spec=gs[5, 3], hspace=0.75,
    )
    for _sub_idx, interp_alt in enumerate(_interps_alt):
        ax = fig.add_subplot(_gs_r5c3[_sub_idx, 0])
        xs, ratios = _sep_ratios_for(interp_alt)
        for cat in _ANN_CATS_SEP:
            ax.plot(xs, ratios[cat], color=CAT_COLORS[cat],
                    marker=INTERP_MARKERS.get(interp_alt, "o"),
                    lw=2.0, ms=7, alpha=0.9, label=CAT_LABELS[cat])
        ax.axhline(1.0, color="#ADB5BD", ls="--", lw=1, alpha=0.7)
        _style_ax(
            ax,
            f"{INTERP_LABELS.get(interp_alt, interp_alt)}\n(sep ratio; >1 = above bg)",
            ylabel="Separation ratio",
        )
        ax.legend(fontsize=10)

    # ── Row 3: per-variant violin plots + metric heatmap ────────────────────
    cmap_noise     = plt.cm.RdYlBu_r
    all_noises_g   = sorted({n for nd in data.values() for n in nd})
    noise_norm     = plt.Normalize(
        min(all_noises_g) if all_noises_g else 0,
        max(all_noises_g) if all_noises_g else 1,
    )
    for col_idx, interp in enumerate(interps_r2[:3]):
        ax = fig.add_subplot(gs[3, col_idx])
        noise_dict    = data[interp]
        sorted_noises = sorted(noise_dict)
        pv_data = [noise_dict[n].get(_pv_vals_key, [0.0]) for n in sorted_noises]
        colors  = [cmap_noise(noise_norm(n)) for n in sorted_noises]
        vp = ax.violinplot(pv_data, positions=range(len(sorted_noises)),
                           showmedians=True, showextrema=True)
        for body, col_c in zip(vp["bodies"], colors):
            body.set_facecolor(col_c); body.set_alpha(0.7)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
        vp["cbars"].set_color("black")
        vp["cmaxes"].set_color("black")
        vp["cmins"].set_color("black")
        means = [float(np.mean(d)) if d else float("nan") for d in pv_data]
        ax.scatter(range(len(sorted_noises)), means, color="black", zorder=5, s=35)
        if not _is_ndcg:
            ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.5)
        ax.set_xticks(range(len(sorted_noises)))
        ax.set_xticklabels([f"{n:.2g}" for n in sorted_noises], fontsize=10)
        _style_ax(
            ax,
            f"Per-variant {_metric_label} dist. vs GT ({track})\n{INTERP_LABELS.get(interp, interp)}",
            xlabel="Noise level",
            ylabel=f"Per-variant {_metric_label} (↑ better)",
        )
        if col_idx == 2:
            sm = plt.cm.ScalarMappable(cmap=cmap_noise, norm=noise_norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02).set_label("Noise level", fontsize=10)

    # Metric heatmap (row 3 col 3)
    ax = fig.add_subplot(gs[3, 3])
    metric_keys = [_global_key, f"auroc{sfx}", _pv_mean_key, "sign_acc",
                   f"cat_sep{sfx}", _fitpv_key_s, _mut_sp_key]
    _ml_abbrev  = "NDCG" if _is_ndcg else "Spear."
    metric_lbls = [_metric_label, "AUROC", f"PV {_ml_abbrev}", "Sign acc", "Cat sep",
                   "Fit-PV", f"Mut {_ml_abbrev}"]
    all_keys_hm = sorted(
        [(interp, n) for interp, nd in data.items() for n in nd],
        key=lambda k: (k[0], k[1]),
    )
    row_lbls_hm = [f"{INTERP_LABELS.get(k[0], k[0])[:4]} n={k[1]:.2g}" for k in all_keys_hm]
    hm_data = np.array([
        [data[k[0]][k[1]].get(m, float("nan")) for m in metric_keys]
        for k in all_keys_hm
    ])
    col_min = np.nanmin(hm_data, axis=0)
    col_max = np.nanmax(hm_data, axis=0)
    hm_norm = (hm_data - col_min) / np.where(col_max - col_min > 0, col_max - col_min, 1)
    im = ax.imshow(hm_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(metric_lbls, fontsize=11, rotation=35, ha="right")
    ax.set_yticks(range(len(row_lbls_hm)))
    ax.set_yticklabels(row_lbls_hm, fontsize=11)
    ax.set_title("Metric heatmap (column-normalised)\ngreen = best, red = worst",
                 fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # ── Row 4: per-sequence mutation ρ violins + summary line ────────────────
    for col_idx, interp in enumerate(interps_r2[:3]):
        ax = fig.add_subplot(gs[4, col_idx])
        noise_dict    = data[interp]
        sorted_noises = sorted(noise_dict)
        rho_data = [noise_dict[n].get(_mut_rho_key, [0.0]) for n in sorted_noises]
        colors   = [cmap_noise(noise_norm(n)) for n in sorted_noises]
        valid    = [d for d in rho_data if len(d) >= 2]
        if valid:
            vp = ax.violinplot(rho_data, positions=range(len(sorted_noises)),
                               showmedians=True, showextrema=True)
            for body, col_c in zip(vp["bodies"], colors):
                body.set_facecolor(col_c); body.set_alpha(0.7)
            vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
            vp["cbars"].set_color("black")
            vp["cmaxes"].set_color("black")
            vp["cmins"].set_color("black")
        means = [float(np.mean(d)) if d else float("nan") for d in rho_data]
        ax.scatter(range(len(sorted_noises)), means, color="black", zorder=5, s=35, label="mean")
        if not _is_ndcg:
            ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.5)
        ax.set_xticks(range(len(sorted_noises)))
        ax.set_xticklabels([f"{n:.2g}" for n in sorted_noises], fontsize=10)
        _style_ax(
            ax,
            f"Per-sequence mutation {_metric_label} distribution\n{INTERP_LABELS.get(interp, interp)}",
            xlabel="Noise level",
            ylabel=f"{_metric_label} — LOO damage vs {track} (↑ better)",
        )
        ax.legend(fontsize=11)
        if col_idx == 2:
            sm = plt.cm.ScalarMappable(cmap=cmap_noise, norm=noise_norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02).set_label("Noise level", fontsize=10)

    ax = fig.add_subplot(gs[4, 3])
    _plot_metric_line(ax, data, _mut_sp_key, hline=None if _is_ndcg else 0.0, **kw)
    _style_ax(
        ax,
        f"Mean per-sequence mutation {_metric_label}\n(LOO damage vs {track})",
        ylabel=f"{_metric_label} (↑ better)",
    )
    ax.legend(handles=_interp_legend(interps), fontsize=11)
    if _is_ndcg:
        # Tight autoscale around the actual values — NDCG is always in [0, 1]
        # but typically clusters near 1; let matplotlib pick the range then add padding.
        _all_mut_vals = [
            noise_dict[n].get(_mut_sp_key, float("nan"))
            for noise_dict in data.values()
            for n in noise_dict
        ]
        _finite = [v for v in _all_mut_vals if np.isfinite(v)]
        if _finite:
            _lo, _hi = min(_finite), max(_finite)
            _pad = max((_hi - _lo) * 0.25, 0.02)
            ax.set_ylim(max(0.0, _lo - _pad), min(1.0, _hi + _pad))

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = _Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _split_suffix = f"_{split_label}" if split_label else ""
    fname    = f"{protein}_{fitness_model}_ig_quality_mixed{_split_suffix}{_fname_mfx}.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Standalone single-panel plots (same aspect ratio as Row-1 subplots)
# ---------------------------------------------------------------------------
# The main quality figure is 22×31 with a 6×4 GridSpec.
# Each 2-column-wide row-1 subplot occupies ~10.5 × 5.0 inches of figure space,
# so standalone figures use figsize=(11, 5) to match.

def plot_all_standalone(
    protein: str,
    fitness_model: str,
    data: dict,
    output_dir: "Path",
    dpi: int = 150,
    split_label: "str | None" = None,
    errors: "dict | None" = None,
    metric_family: str = "ndcg",
) -> None:
    """Generate every subplot from make_figure as an individual PNG.

    Each panel uses figsize=(5.5, 5) to match the single-subplot aspect ratio
    of the compiled 22×31 figure (22 / 4 columns = 5.5 per panel).
    """
    from pathlib import Path as _Path

    _is_ndcg       = metric_family == "ndcg"
    _global_key    = "ndcg"                if _is_ndcg else "spearman"
    _global_key_s  = "ndcg_signed"         if _is_ndcg else "spearman_signed"
    _pv_mean_key   = "pv_ndcg_mean"        if _is_ndcg else "pv_mean"
    _pv_std_key    = "pv_ndcg_std"         if _is_ndcg else "pv_std"
    _pv_vals_key   = "pv_ndcg_vals"        if _is_ndcg else "pv_vals"
    _mut_sp_key    = "mutation_ndcg"        if _is_ndcg else "mutation_spearman"
    _mut_rho_key   = "mutation_ndcg_vals"   if _is_ndcg else "mutation_rho_vals"
    _fitpv_key_s   = "pv_fitness_ndcg_signed" if _is_ndcg else "pv_fitness_spearman_signed"
    _prec_k_key    = "ap_graded"            if _is_ndcg else "prec_k"
    _prec_k_lbl    = "AP@R (graded relevance)" if _is_ndcg else "Precision@K"
    _metric_label  = "NDCG"                if _is_ndcg else "Spearman ρ"
    _sp_hline      = None                  if _is_ndcg else 0.0
    _fname_mfx     = "_ndcg"              if _is_ndcg else ""
    _ml_abbrev     = "NDCG"               if _is_ndcg else "Spear."
    _model_test_key = "model_test_ndcg"    if _is_ndcg else "model_test_spearman"
    _model_val_key  = "model_val_ndcg"     if _is_ndcg else "model_val_spearman"
    _model_test_lbl = "Model test NDCG"    if _is_ndcg else "Model test ρ"
    _model_val_lbl  = "Model val NDCG"     if _is_ndcg else "Model val ρ"
    track   = "|IG|"
    track_s = "Signed IG"
    sfx     = ""
    sfx_s   = "_signed"

    out_dir = _Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _sp = f"_{split_label}" if split_label else ""
    _pfx = f"{protein}_{fitness_model}"
    _sx = f"{_sp}{_fname_mfx}"

    non_linear_noises: list = []
    for interp, noise_dict in data.items():
        if interp != "linear":
            non_linear_noises.extend(noise_dict.keys())
    nn_cutoff = max(non_linear_noises) if non_linear_noises else None
    kw = dict(nn_slerp_cutoff=nn_cutoff, errors=errors)

    interps     = sorted(data.keys())
    interps_r2  = [i for i in ("linear", "nearest_neighbor", "slerp") if i in interps]
    interps_alt = [i for i in ("gradient_x_input", "input_difference") if i in interps]

    if protein in _PROTEIN_DEFS:
        cat_weights = _PROTEIN_DEFS[protein].category_weights
        ann_cats = sorted(cat_weights, key=cat_weights.__getitem__)
    else:
        sample_nd = next(iter(data.values()), {})
        sample_cm = next(iter(sample_nd.values()), {}).get("cat_means_signed", {})
        ann_cats  = sorted(k for k in sample_cm if k != "background")
    CATS    = ["background"] + ann_cats
    top_cat = ann_cats[-1] if ann_cats else "binding_site"
    _short  = {"background": "bg", "active_site": "active", "metal_binding": "metal",
               "signalling": "NLS", "binding_site": "binding", "structural": "struct",
               "folding": "fold"}
    _expected_order = " < ".join(["bg"] + [_short.get(c, c) for c in ann_cats])
    _top_label      = _short.get(top_cat, top_cat)
    _ANN_CATS_SEP   = [c for c in CATS if c != "background"]

    all_noises_g = sorted({n for nd in data.values() for n in nd})
    cmap_noise   = plt.cm.RdYlBu_r
    noise_norm   = plt.Normalize(
        min(all_noises_g) if all_noises_g else 0,
        max(all_noises_g) if all_noises_g else 1,
    )

    _cat_legend_handles = [
        Line2D([0], [0], color=CAT_COLORS[cat], lw=2, label=CAT_LABELS[cat])
        for cat in CATS
    ]

    def _save(fig, name):
        p = out_dir / f"{_pfx}_{name}{_sx}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"  Saved: {p}")

    def _new():
        return plt.subplots(figsize=(5.5, 5), facecolor=FIG_BG)

    # ── Export legends as standalone PNGs ────────────────────────────────────
    # Interp-method legend
    _leg_fig = plt.figure(figsize=(3.5, 0.45 * (len(interps) + 0.5)), facecolor=FIG_BG)
    _leg_ax  = _leg_fig.add_subplot(111)
    _leg_ax.axis("off")
    _leg_ax.legend(handles=_interp_legend(interps), fontsize=11,
                   loc="center", framealpha=0.5, title="Interpolation method",
                   title_fontsize=12)
    _leg_fig.tight_layout()
    _leg_path = out_dir / f"{_pfx}_legend_methods{_sp}.png"
    _leg_fig.savefig(_leg_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(_leg_fig)
    print(f"  Saved: {_leg_path}")

    # Category legend
    _cat_leg_fig = plt.figure(figsize=(3.5, 0.45 * (len(CATS) + 0.5)), facecolor=FIG_BG)
    _cat_leg_ax  = _cat_leg_fig.add_subplot(111)
    _cat_leg_ax.axis("off")
    _cat_leg_ax.legend(handles=_cat_legend_handles, fontsize=11,
                       loc="center", framealpha=0.5, title="Residue category",
                       title_fontsize=12)
    _cat_leg_fig.tight_layout()
    _cat_leg_path = out_dir / f"{_pfx}_legend_categories{_sp}.png"
    _cat_leg_fig.savefig(_cat_leg_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(_cat_leg_fig)
    print(f"  Saved: {_cat_leg_path}")

    def _sep_ratios_for(interp):
        nd, xs = data.get(interp, {}), sorted(data.get(interp, {}))
        out = {c: [] for c in _ANN_CATS_SEP}
        for n in xs:
            cm = nd[n].get("cat_means", {})
            bg = cm.get("background", float("nan"))
            for c in _ANN_CATS_SEP:
                cv = cm.get(c, float("nan"))
                out[c].append(cv / bg if (np.isfinite(cv) and np.isfinite(bg) and bg > 0) else float("nan"))
        return xs, out

    # ── Row 0: headline metrics ───────────────────────────────────────────────
    # 0a: Global NDCG / Spearman
    fig, ax = _new()
    _plot_metric_line(ax, data, _global_key, hline=_sp_hline, **kw)
    _style_ax(ax, f"Global {_metric_label}\n({track} vs GT importance)", ylabel=f"{_metric_label} (↑ better)")
    fig.tight_layout()
    _save(fig, "r0_global_ndcg")

    # 0b: AUROC
    fig, ax = _new()
    _plot_metric_line(ax, data, f"auroc{sfx}", hline=0.5, **kw)
    _style_ax(ax, "AUROC\n(annotated vs background)", ylabel="AUROC (↑ better)")
    fig.tight_layout()
    _save(fig, "r0_auroc")

    # 0c: AUPRC
    fig, ax = _new()
    _plot_metric_line(ax, data, f"auprc{sfx}", **kw)
    _style_ax(ax, "AUPRC\n(annotated vs background)", ylabel="AUPRC (↑ better)")
    fig.tight_layout()
    _save(fig, "r0_auprc")

    # 0d: AP@R graded
    fig, ax = _new()
    _plot_metric_line(ax, data, f"{_prec_k_key}{sfx}", **kw)
    _style_ax(ax, f"{_prec_k_lbl}\n(graded relevance)", ylabel=f"{_prec_k_lbl} (↑ better)")
    fig.tight_layout()
    _save(fig, "r0_ap_graded")

    # ── Row 1: per-variant NDCG (mean±std) and sign accuracy ─────────────────
    # 1a: PV NDCG
    fig, ax = _new()
    for interp, noise_dict in sorted(data.items()):
        xs   = sorted(noise_dict)
        ys_m = [noise_dict[n].get(_pv_mean_key, float("nan")) for n in xs]
        ys_s = [noise_dict[n].get(_pv_std_key,  float("nan")) for n in xs]
        col  = INTERP_COLORS.get(interp, "#888")
        mk   = INTERP_MARKERS.get(interp, "o")
        ax.plot(xs, ys_m, color=col, marker=mk, lw=2.0, ms=7, alpha=0.9,
                label=INTERP_LABELS.get(interp, interp))
        ys_lo = [m - s for m, s in zip(ys_m, ys_s)]
        ys_hi = [m + s for m, s in zip(ys_m, ys_s)]
        ax.fill_between(xs, ys_lo, ys_hi, alpha=0.12, color=col)
        if errors is not None:
            seed_err = [errors.get(interp, {}).get(n, {}).get(_pv_mean_key, 0.0) for n in xs]
            ax.fill_between(xs, [m - e for m, e in zip(ys_m, seed_err)],
                            [m + e for m, e in zip(ys_m, seed_err)], alpha=0.25, color=col, linewidth=0)
    if _sp_hline is not None:
        ax.axhline(_sp_hline, color="grey", ls="--", lw=1, alpha=0.5)
    if nn_cutoff:
        ax.axvline(nn_cutoff, color="#888", ls=":", lw=1.2, alpha=0.5)
    _style_ax(ax, f"Per-variant {_metric_label} ({track})\n(mean ± std across sequences)",
              ylabel=f"{_metric_label} (↑ better)")
    fig.tight_layout()
    _save(fig, "r1_pv_ndcg")

    # 1b: Sign accuracy
    fig, ax = _new()
    _plot_metric_line(ax, data, "sign_acc", hline=0.5, **kw)
    _style_ax(ax, "Sign accuracy\n(mutated annotated positions)", ylabel="Fraction correct (↑ better)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, "r1_sign_accuracy")

    # ── Row 2: category means (signed IG) per interpolation ──────────────────
    for interp in interps_r2 + interps_alt:
        noise_dict = data.get(interp, {})
        xs = sorted(noise_dict)
        fig, ax = _new()
        for cat in CATS:
            ys = [noise_dict[n].get(f"cat_means{sfx_s}", {}).get(cat, float("nan")) for n in xs]
            ax.plot(xs, ys, color=CAT_COLORS[cat], marker=INTERP_MARKERS.get(interp, "o"),
                    lw=2.0, ms=7, alpha=0.9, label=CAT_LABELS[cat])
        _style_ax(ax, f"Category mean {track_s} — {INTERP_LABELS.get(interp, interp)}\n"
                  f"(expected: {_expected_order})",
                  ylabel=f"Mean {track_s} ({_top_label} ↑, bg ↓)")
        fig.tight_layout()
        _save(fig, f"r2_cat_means_{interp}")

    # ── Row 3: per-variant NDCG violin plots ─────────────────────────────────
    for interp in interps_r2:
        noise_dict    = data.get(interp, {})
        sorted_noises = sorted(noise_dict)
        pv_data = [noise_dict[n].get(_pv_vals_key, [0.0]) for n in sorted_noises]
        colors  = [cmap_noise(noise_norm(n)) for n in sorted_noises]
        fig, ax = _new()
        vp = ax.violinplot(pv_data, positions=range(len(sorted_noises)),
                           showmedians=True, showextrema=True)
        for body, col_c in zip(vp["bodies"], colors):
            body.set_facecolor(col_c); body.set_alpha(0.7)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
        vp["cbars"].set_color("black"); vp["cmaxes"].set_color("black"); vp["cmins"].set_color("black")
        means = [float(np.mean(d)) if d else float("nan") for d in pv_data]
        ax.scatter(range(len(sorted_noises)), means, color="black", zorder=5, s=35)
        if not _is_ndcg:
            ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.5)
        ax.set_xticks(range(len(sorted_noises)))
        ax.set_xticklabels([f"{n:.2g}" for n in sorted_noises], fontsize=11)
        sm = plt.cm.ScalarMappable(cmap=cmap_noise, norm=noise_norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02).set_label("Noise level", fontsize=11)
        _style_ax(ax, f"Per-variant {_metric_label} dist. ({track})\n{INTERP_LABELS.get(interp, interp)}",
                  xlabel="Noise level", ylabel=f"Per-variant {_metric_label} (↑ better)")
        fig.tight_layout()
        _save(fig, f"r3_pv_violin_{interp}")

    # 3d: Metric heatmap
    metric_keys = [_global_key, f"auroc{sfx}", _pv_mean_key, "sign_acc",
                   f"cat_sep{sfx}", _fitpv_key_s, _mut_sp_key]
    metric_lbls = [_metric_label, "AUROC", f"PV {_ml_abbrev}", "Sign acc", "Cat sep",
                   "Fit-PV", f"Mut {_ml_abbrev}"]
    all_keys_hm = sorted([(interp, n) for interp, nd in data.items() for n in nd],
                         key=lambda k: (k[0], k[1]))
    row_lbls_hm = [f"{INTERP_LABELS.get(k[0], k[0])[:4]} n={k[1]:.2g}" for k in all_keys_hm]
    hm_data = np.array([[data[k[0]][k[1]].get(m, float("nan")) for m in metric_keys]
                        for k in all_keys_hm])
    col_min = np.nanmin(hm_data, axis=0)
    col_max = np.nanmax(hm_data, axis=0)
    hm_norm = (hm_data - col_min) / np.where(col_max - col_min > 0, col_max - col_min, 1)
    n_rows_hm = len(row_lbls_hm)
    fig, ax = plt.subplots(figsize=(5.5, max(5, n_rows_hm * 0.35 + 1.5)), facecolor=FIG_BG)
    im = ax.imshow(hm_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(metric_lbls, fontsize=11, rotation=35, ha="right")
    ax.set_yticks(range(n_rows_hm))
    ax.set_yticklabels(row_lbls_hm, fontsize=11)
    ax.set_title("Metric heatmap (column-normalised)\ngreen = best, red = worst",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    _save(fig, "r3_metric_heatmap")

    # ── Row 4: per-sequence mutation NDCG violin plots ───────────────────────
    for interp in interps_r2:
        noise_dict    = data.get(interp, {})
        sorted_noises = sorted(noise_dict)
        rho_data = [noise_dict[n].get(_mut_rho_key, [0.0]) for n in sorted_noises]
        colors   = [cmap_noise(noise_norm(n)) for n in sorted_noises]
        fig, ax  = _new()
        valid = [d for d in rho_data if len(d) >= 2]
        if valid:
            vp = ax.violinplot(rho_data, positions=range(len(sorted_noises)),
                               showmedians=True, showextrema=True)
            for body, col_c in zip(vp["bodies"], colors):
                body.set_facecolor(col_c); body.set_alpha(0.7)
            vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
            vp["cbars"].set_color("black"); vp["cmaxes"].set_color("black"); vp["cmins"].set_color("black")
        means = [float(np.mean(d)) if d else float("nan") for d in rho_data]
        ax.scatter(range(len(sorted_noises)), means, color="black", zorder=5, s=35, label="mean")
        if not _is_ndcg:
            ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.5)
        ax.set_xticks(range(len(sorted_noises)))
        ax.set_xticklabels([f"{n:.2g}" for n in sorted_noises], fontsize=11)
        sm = plt.cm.ScalarMappable(cmap=cmap_noise, norm=noise_norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02).set_label("Noise level", fontsize=11)
        _style_ax(ax, f"Per-sequence mutation {_metric_label} dist.\n{INTERP_LABELS.get(interp, interp)}",
                  xlabel="Noise level", ylabel=f"{_metric_label} — LOO damage vs {track} (↑ better)")
        fig.tight_layout()
        _save(fig, f"r4_mut_violin_{interp}")

    # 4d: Mean per-sequence mutation NDCG (line)
    fig, ax = _new()
    _plot_metric_line(ax, data, _mut_sp_key, hline=None if _is_ndcg else 0.0, **kw)
    _style_ax(ax, f"Mean per-sequence mutation {_metric_label}\n(LOO damage vs {track})",
              ylabel=f"{_metric_label} (↑ better)")
    if _is_ndcg:
        _all_mv = [nd[n].get(_mut_sp_key, float("nan")) for nd in data.values() for n in nd]
        _finite = [v for v in _all_mv if np.isfinite(v)]
        if _finite:
            _lo, _hi = min(_finite), max(_finite)
            _pad = max((_hi - _lo) * 0.25, 0.02)
            ax.set_ylim(max(0.0, _lo - _pad), min(1.0, _hi + _pad))
    fig.tight_layout()
    _save(fig, "r4_mean_mut_ndcg")

    # ── Row 5: category separation ratio ─────────────────────────────────────
    for interp in interps_r2 + interps_alt:
        xs, ratios = _sep_ratios_for(interp)
        fig, ax = _new()
        for cat in _ANN_CATS_SEP:
            ax.plot(xs, ratios[cat], color=CAT_COLORS[cat],
                    marker=INTERP_MARKERS.get(interp, "o"), lw=2.0, ms=7, alpha=0.9,
                    label=CAT_LABELS[cat])
        ax.axhline(1.0, color="#ADB5BD", ls="--", lw=1, alpha=0.7)
        _style_ax(ax, f"Category separation — {INTERP_LABELS.get(interp, interp)}\n"
                  f"(mean |IG| / background; >1 = above bg)",
                  ylabel="Separation ratio (↑ better)")
        fig.tight_layout()
        _save(fig, f"r5_cat_sep_{interp}")


# Keep backward-compatible aliases
def plot_standalone_pv_ndcg(protein, fitness_model, data, output_dir,
                             dpi=150, split_label=None, errors=None, metric_family="ndcg"):
    """Deprecated: use plot_all_standalone instead."""
    plot_all_standalone(protein, fitness_model, data, output_dir, dpi, split_label, errors, metric_family)


def plot_standalone_sign_accuracy(protein, fitness_model, data, output_dir,
                                   dpi=150, split_label=None, errors=None, metric_family="ndcg"):
    """Deprecated: use plot_all_standalone instead."""
    pass  # already called by plot_standalone_pv_ndcg via plot_all_standalone


# ---------------------------------------------------------------------------
# Method-selection analysis plots
# ---------------------------------------------------------------------------

from collections import defaultdict as _defaultdict
from scipy.stats import rankdata as _rankdata

_FALLBACK_COLORS: list[str] = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def _ms_method_color(method: str) -> str:
    return INTERP_COLORS.get(method, _FALLBACK_COLORS[hash(method) % len(_FALLBACK_COLORS)])


def _ms_method_label(method: str) -> str:
    return INTERP_LABELS.get(method, method)


_HEATMAP_ROW_ORDER: list[str] = [
    "input_difference",
    "gradient_x_input",
    "linear",
    "slerp",
    "nearest_neighbor",
]


def plot_critical_difference(
    auc_matrix: pd.DataFrame,
    nemenyi_pvals: "pd.DataFrame | None",
    output_path: "Path | str",
    alpha: float = 0.05,
    dpi: int = 150,
) -> None:
    """Critical-difference diagram (Demšar 2006) for the AUC matrix.

    Methods are plotted by average rank (1 = best, left).
    Horizontal bars connect groups whose pairwise Nemenyi p > *alpha*
    (i.e. not significantly different from each other).
    """
    from pathlib import Path as _Path

    methods = list(auc_matrix.columns)
    mask    = ~np.any(np.isnan(auc_matrix.values), axis=1)
    clean   = auc_matrix.loc[mask].values
    n, k    = clean.shape

    ranks_per_scenario = np.apply_along_axis(
        lambda r: _rankdata(-r), axis=1, arr=clean
    )
    avg_ranks = ranks_per_scenario.mean(axis=0)
    order     = np.argsort(avg_ranks)    # best → worst

    Q_ALPHA = {0.10: 2.291, 0.05: 2.569, 0.01: 3.143}
    q       = Q_ALPHA.get(alpha, 2.569)
    cd      = q * np.sqrt(k * (k + 1) / (6 * n))

    fig, ax = plt.subplots(figsize=(max(8, k * 1.5), 3.5))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1.5, 1.0)
    ax.set_xlabel("Average rank across scenarios (1 = best)", fontsize=11)
    ax.set_yticks([])
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))

    for i, idx in enumerate(order):
        r   = avg_ranks[idx]
        lbl = _ms_method_label(methods[idx])
        col = _ms_method_color(methods[idx])
        ax.plot(r, 0, "o", color=col, ms=8, zorder=3)
        ax.text(r, 0.15 + 0.25 * (i % 2), lbl,
                ha="center", va="bottom", fontsize=11, color=col)

    ax.annotate("", xy=(1 + cd, -0.9), xytext=(1, -0.9),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1 + cd / 2, -1.15, f"CD={cd:.2f}", ha="center", fontsize=8)

    if nemenyi_pvals is not None:
        parent = list(range(k))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            parent[_find(a)] = _find(b)

        for i in range(k):
            for j in range(i + 1, k):
                m_i = methods[order[i]]
                m_j = methods[order[j]]
                if nemenyi_pvals.loc[m_i, m_j] >= alpha:
                    _union(i, j)

        components: dict[int, list[int]] = _defaultdict(list)
        for i in range(k):
            components[_find(i)].append(i)

        drawn_count = 0
        for comp in components.values():
            if len(comp) > 1:
                xs    = sorted(avg_ranks[order[g]] for g in comp)
                y_bar = -0.5 - 0.12 * drawn_count
                ax.plot([xs[0], xs[-1]], [y_bar, y_bar],
                        lw=3, color="black", alpha=0.6, solid_capstyle="butt")
                drawn_count += 1

    ax.set_title(
        f"Critical-difference diagram  (α={alpha}, n={n} scenarios)",
        fontsize=12, pad=10,
    )
    fig.tight_layout()
    out = _Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_composite_noise_curves(
    df: pd.DataFrame,
    output_path: "Path | str",
    dpi: int = 150,
) -> None:
    """One subplot per scenario: composite score vs noise level, one line per method."""
    from pathlib import Path as _Path

    scenarios = sorted(df["scenario"].unique())
    methods   = sorted(df["method"].unique())
    ncols     = min(3, len(scenarios))
    nrows     = int(np.ceil(len(scenarios) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_visible(False)

    _has_model_perf = "model_test_spearman" in df.columns

    for i, scenario in enumerate(scenarios):
        ax  = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        sub = df[df["scenario"] == scenario]
        for method in methods:
            m_sub = sub[sub["method"] == method].sort_values("noise")
            if m_sub.empty:
                continue
            ax.plot(m_sub["noise"], m_sub["composite"],
                    "o-", color=_ms_method_color(method),
                    label=_ms_method_label(method), lw=1.8, ms=5)
        ax.set_title(scenario.replace("_", " / "), fontsize=10)
        ax.set_xlabel("Noise level", fontsize=9)
        ax.set_ylabel("Composite score", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # ── Right y-axis: model test-set Spearman ρ ───────────────────────
        if _has_model_perf:
            # Average across methods (model perf is per scenario×noise, not per method)
            perf = (
                sub.groupby("noise")["model_test_spearman"]
                .mean()
                .reset_index()
                .sort_values("noise")
                .dropna(subset=["model_test_spearman"])
            )
            if not perf.empty:
                ax2 = ax.twinx()
                ax2.plot(
                    perf["noise"], perf["model_test_spearman"],
                    "s--", color="#555555", lw=1.5, ms=5,
                    alpha=0.7, label="Model test ρ", zorder=2,
                )
                ax2.set_ylabel("Model test Spearman ρ", fontsize=8, color="#555555")
                ax2.tick_params(axis="y", labelcolor="#555555", labelsize=7)
                _rho_vals = perf["model_test_spearman"].values
                _finite   = _rho_vals[np.isfinite(_rho_vals)]
                if len(_finite):
                    _pad = max(0.05, (float(_finite.max()) - float(_finite.min())) * 0.2)
                    ax2.set_ylim(float(_finite.min()) - _pad, float(_finite.max()) + _pad)
                ax2.spines["right"].set_color("#555555")

    handles = [
        mpatches.Patch(color=_ms_method_color(m), label=_ms_method_label(m))
        for m in methods
    ]
    if _has_model_perf:
        handles.append(
            Line2D([0], [0], color="#555555", lw=1.5, ls="--", marker="s",
                   ms=5, alpha=0.7, label="Model test ρ (right axis)")
        )
    fig.legend(
        handles=handles, fontsize=8, loc="lower center",
        ncol=min(len(handles), 6),
        bbox_to_anchor=(0.5, -0.02), framealpha=0,
    )
    fig.suptitle(
        "Composite score (NDCG + AP@R + AUROC + pv-NDCG + sign-acc + cat-sep) vs noise",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out = _Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_auc_heatmap(
    auc_matrix: pd.DataFrame,
    output_path: "Path | str",
    dpi: int = 150,
) -> None:
    """Heatmap of AUC-over-noise composite scores (methods as rows, scenarios as columns)."""
    from pathlib import Path as _Path

    data = auc_matrix.T    # methods × scenarios

    ordered  = [m for m in _HEATMAP_ROW_ORDER if m in data.index]
    ordered += [m for m in data.index if m not in ordered]
    data = data.loc[ordered]

    n_cols = len(data.columns)
    n_rows = len(data.index)
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.5), max(4, n_rows * 0.85)))

    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="AUC-over-noise composite score")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [s.replace("_", "\n") for s in data.columns],
        fontsize=9, ha="center",
    )
    ax.tick_params(axis="x", which="major", pad=8)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([_ms_method_label(m) for m in data.index], fontsize=10)

    for r in range(n_rows):
        for c in range(n_cols):
            val = data.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    ax.set_title("AUC over noise — composite score (higher = better)",
                 fontsize=11, pad=10)
    fig.tight_layout()
    out = _Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_peak_vs_decay(
    decay_df: pd.DataFrame,
    output_path: "Path | str",
    dpi: int = 150,
) -> None:
    """Scatter: noise decay rate (x) vs peak performance (y), one subplot per scenario."""
    from pathlib import Path as _Path

    scenarios = sorted(decay_df["scenario"].unique())
    ncols     = min(3, len(scenarios))
    nrows     = int(np.ceil(len(scenarios) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_visible(False)

    for i, scenario in enumerate(scenarios):
        ax  = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        sub = decay_df[decay_df["scenario"] == scenario]
        for _, row in sub.iterrows():
            if np.isnan(row["peak"]) or np.isnan(row["decay_rate"]):
                continue
            col = _ms_method_color(row["method"])
            ax.scatter(row["decay_rate"], row["peak"], color=col, s=80, zorder=3)
            ax.text(row["decay_rate"], row["peak"] + 0.02,
                    _ms_method_label(row["method"]),
                    ha="center", va="bottom", fontsize=7, color=col)
        ax.set_title(scenario.replace("_", " / "), fontsize=10)
        ax.set_xlabel("Decay rate b  (higher = less robust)", fontsize=9)
        ax.set_ylabel("Peak performance (noise = 0)", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Peak performance vs noise-decay rate\n"
        "(top-left = best: high peak AND slow decay)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out = _Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_metric_breakdown(
    df: pd.DataFrame,
    output_path: "Path | str",
    composite_metrics: "list[tuple[str, str]]",
    dpi: int = 150,
    normalize: bool = True,
) -> None:
    """Grouped bar chart: per-metric score at noise = 0, per scenario.

    Parameters
    ----------
    normalize:
        If *True* (default) each metric is min-max normalised across methods
        per scenario (uses the ``_norm_{key}`` column added by
        ``add_composite_score``).  If *False* the raw metric values are
        plotted directly on the y-axis.
    """
    from pathlib import Path as _Path

    methods   = sorted(df["method"].unique())
    scenarios = sorted(df["scenario"].unique())
    df0       = df[df["noise"] == df["noise"].min()].copy()

    fig, axes = plt.subplots(
        len(composite_metrics), 1,
        figsize=(max(8, len(scenarios) * 1.2), 2.8 * len(composite_metrics)),
        squeeze=False,
    )
    x     = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)

    for row_i, (key, label) in enumerate(composite_metrics):
        ax      = axes[row_i][0]
        col     = f"_norm_{key}" if normalize else key
        for j, method in enumerate(methods):
            sub  = df0[df0["method"] == method].set_index("scenario")
            def _scalar(v):
                import pandas as _pd
                if isinstance(v, _pd.Series):
                    return float(v.mean()) if len(v) else float("nan")
                return float(v)

            vals = [
                _scalar(sub.loc[s, col]) if s in sub.index and col in sub.columns
                else float("nan")
                for s in scenarios
            ]
            ax.bar(
                x + j * width - 0.4 + width / 2, vals,
                width=width * 0.9,
                color=_ms_method_color(method),
                alpha=0.85,
                label=_ms_method_label(method),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=8)
        if normalize:
            ax.set_ylabel("Norm. score", fontsize=9)
            ax.set_title(f"{label} (normalised, noise = 0)", fontsize=10)
            ax.set_ylim(-0.05, 1.15)
        else:
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(f"{label} (raw, noise = 0)", fontsize=10)
            if key == "cat_sep":
                ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        if row_i == 0:
            ax.legend(fontsize=8, ncol=len(methods))

    suptitle = "Per-metric normalised scores at zero noise" if normalize else "Per-metric raw scores at zero noise"
    fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()
    out = _Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# FLA correlation heatmaps
# ---------------------------------------------------------------------------

#: Preferred display order for interpolation methods (top → bottom).
_FLA_METHOD_ROW_ORDER: list[str] = [
    "input_difference",
    "gradient_x_input",
    "linear",
    "slerp",
    "nearest_neighbor",
]


def plot_fla_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str,
    output_path: "Path | str",
    *,
    method_labels: "dict[str, str] | None" = None,
    row_order: "list[str] | None" = None,
    figsize: "tuple[float, float] | None" = None,
    dpi: int = 150,
) -> None:
    """Heatmap of Spearman ρ between FLA metrics and AUC-over-noise.

    Parameters
    ----------
    corr_df:
        DataFrame with methods as the index and FLA metrics as columns,
        values are Spearman ρ in [-1, 1].
    title:
        Figure title.
    output_path:
        Destination file (PNG/PDF).
    method_labels:
        Optional mapping ``{method_key: display_label}``.  If *None*, the
        raw index values are used as-is.
    row_order:
        Preferred row order (top → bottom).  Missing rows are appended at
        the bottom; extra entries that are not in the index are ignored.
    figsize:
        Override figure size.  Defaults to ``(max(12, n_cols * 1.1), 4.5)``.
    dpi:
        Output resolution.
    """
    from pathlib import Path as _Path

    if method_labels is None:
        method_labels = {}

    # ── Apply row order ───────────────────────────────────────────────────────
    if row_order is None:
        row_order = _FLA_METHOD_ROW_ORDER
    ordered_rows = [r for r in row_order if r in corr_df.index]
    remaining    = [r for r in corr_df.index if r not in ordered_rows]
    corr_df = corr_df.loc[ordered_rows + remaining]

    n_rows, n_cols = corr_df.shape
    if figsize is None:
        figsize = (max(12, n_cols * 1.15), max(3.5, n_rows * 0.75 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        corr_df.values.astype(float),
        cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto",
    )

    # x-axis: leave room so rotated labels don't overlap
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        corr_df.columns,
        rotation=40, ha="right", rotation_mode="anchor",
        fontsize=9,
    )
    ax.tick_params(axis="x", pad=4)

    # y-axis
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [method_labels.get(m, m) for m in corr_df.index],
        fontsize=10,
    )

    # Cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = corr_df.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(val) >= 0.7 else "black",
                )

    cbar = plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.85)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    out = _Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_fla_category_heatmap(
    corr_df: pd.DataFrame,
    title: str,
    output_path: "Path | str",
    metric_categories: "dict[str, str]",
    *,
    method_labels: "dict[str, str] | None" = None,
    row_order: "list[str] | None" = None,
    dpi: int = 150,
) -> None:
    """Category-averaged version of :func:`plot_fla_correlation_heatmap`.

    FLA metrics are averaged within each category (Ruggedness, Epistasis,
    Navigability, Neutrality, Fitness Distribution) before plotting, giving a
    more compact 5 × n_categories heatmap.

    Parameters
    ----------
    corr_df:
        Per-metric Spearman ρ DataFrame (same format as
        :func:`plot_fla_correlation_heatmap`).
    metric_categories:
        Mapping ``{metric_key: category_name}``.
    All other parameters are forwarded to :func:`plot_fla_correlation_heatmap`.
    """
    cats = sorted(set(metric_categories.values()))
    cat_df = pd.DataFrame(index=corr_df.index, columns=cats, dtype=float)
    for cat in cats:
        cols = [c for c in corr_df.columns if metric_categories.get(c) == cat]
        if cols:
            cat_df[cat] = corr_df[cols].mean(axis=1)

    plot_fla_correlation_heatmap(
        cat_df, title, output_path,
        method_labels=method_labels,
        row_order=row_order,
        figsize=(max(7, len(cats) * 1.4), max(3.5, len(cat_df) * 0.75 + 1.5)),
        dpi=dpi,
    )


# ---------------------------------------------------------------------------
# Synthetic dataset overview plots
# ---------------------------------------------------------------------------

#: Colour for each fitness model in synthetic dataset plots.
_SYNTHETIC_MODEL_COLORS: dict[str, str] = {
    "additive":  "#2196F3",   # blue
    "epistatic": "#FF9800",   # orange
    "nonlinear": "#4CAF50",   # green
}

#: Colour for each mutation-count tier.
_MUT_TIER_COLORS: dict[str, str] = {
    "wildtype": "#607D8B",   # grey-blue
    "single":   "#1a78c2",   # strong blue
    "double":   "#2ecc71",   # vivid green
    "triple":   "#e67e22",   # strong orange
    "multi":    "#8e24aa",   # purple (≥4 mutations)
}

#: Display-friendly label for each tier.
_MUT_TIER_LABELS: dict[str, str] = {
    "wildtype": "WT",
    "single":   "Single (1)",
    "double":   "Double (2)",
    "triple":   "Triple (3)",
    "multi":    "Multi (≥4)",
}


def plot_synthetic_dataset_overview(
    datasets: dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
    dpi: int = 200,
    deep: bool = False,
    facecolor: str = FIG_BG,
    transparent: bool = False,
) -> None:
    """
    Plot an overview of one or more synthetic datasets.

    Each dataset produces **two panels** laid out in a grid:

    * **Left** – Mutation-count bar chart breaking down how many variants are
      WT / single / double / triple / multi (≥4).
    * **Right** – Fitness distribution (KDE + rug) coloured by variant tier.

    Parameters
    ----------
    datasets:
        Mapping of ``"<protein>_<model>"`` labels to DataFrames. DataFrames
        must contain at least the columns ``variant_type``, ``n_mutations``,
        and ``fitness``.
    save_path:
        Where to write the finished figure (PNG). If *None* the figure is
        displayed interactively.
    dpi:
        Resolution for the saved figure.
    deep:
        When *True* the title block notes that these are the *deep* (high-order
        mutation) variants, and the ``multi`` tier is shown prominently.
    """
    n_datasets = len(datasets)
    if n_datasets == 0:
        return

    ncols = 2  # bar chart | fitness distribution
    nrows = n_datasets

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(10, 3.2 * nrows),
        squeeze=False,
    )
    _bg = "none" if transparent else facecolor
    fig.patch.set_facecolor(_bg)

    sup_suffix = " (deep variants)" if deep else ""
    fig.suptitle(
        f"Synthetic dataset overview{sup_suffix}",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    for row_idx, (label, df) in enumerate(datasets.items()):
        ax_bar = axes[row_idx][0]
        ax_fit = axes[row_idx][1]
        for ax in (ax_bar, ax_fit):
            ax.set_facecolor("none" if transparent else facecolor)

        # ── Determine ordered tiers present in this dataset ──────────────
        all_tiers_ordered = ["wildtype", "single", "double", "triple", "multi"]
        present_tiers = [t for t in all_tiers_ordered if t in df["variant_type"].values]

        counts = df["variant_type"].value_counts()

        # ── Left panel: stacked/grouped bar of mutation tiers ────────────
        bar_tiers   = [t for t in present_tiers if t != "wildtype"]
        bar_heights = [counts.get(t, 0) for t in bar_tiers]
        bar_colors  = [_MUT_TIER_COLORS.get(t, "#9E9E9E") for t in bar_tiers]
        bar_labels  = [_MUT_TIER_LABELS.get(t, t) for t in bar_tiers]
        x_pos = np.arange(len(bar_tiers))

        bars = ax_bar.bar(x_pos, bar_heights, color=bar_colors, edgecolor="white",
                          linewidth=0.6, width=0.6)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(bar_labels, fontsize=7.5)
        ax_bar.set_ylabel("Count", fontsize=8)
        ax_bar.set_title(f"{label} – mutation tiers", fontsize=8, fontweight="bold")
        ax_bar.tick_params(axis="y", labelsize=7)
        ax_bar.grid(True, axis="y", alpha=0.25)
        ax_bar.set_axisbelow(True)

        # Annotate bar heights
        for bar, h in zip(bars, bar_heights):
            if h > 0:
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max(bar_heights) * 0.01,
                    f"{h:,}",
                    ha="center", va="bottom", fontsize=6.5,
                )

        # ── Right panel: fitness KDE per tier ────────────────────────────
        for tier in present_tiers:
            sub = df.loc[df["variant_type"] == tier, "fitness"].dropna()
            if len(sub) < 2:
                continue
            color = _MUT_TIER_COLORS.get(tier, "#9E9E9E")
            tier_label = _MUT_TIER_LABELS.get(tier, tier)
            sns.kdeplot(
                sub,
                ax=ax_fit,
                color=color,
                fill=True,
                alpha=0.25,
                linewidth=1.5,
                label=tier_label,
            )
            # Rug tick marks at the bottom
            ax_fit.plot(
                sub,
                np.full(len(sub), ax_fit.get_ylim()[0] if ax_fit.get_ylim()[0] != 0 else 0),
                "|",
                color=color,
                alpha=0.15,
                markersize=3,
            )

        ax_fit.axvline(
            df.loc[df["is_wildtype"] == True, "fitness"].iloc[0]
            if "is_wildtype" in df.columns and df["is_wildtype"].any()
            else df["fitness"].median(),
            color="#333333",
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
            label="WT fitness",
        )
        ax_fit.set_xscale("symlog", linthresh=0.1)
        ax_fit.set_xlabel("Fitness (symlog scale)", fontsize=8)
        ax_fit.set_ylabel("Density", fontsize=8)
        ax_fit.set_title(f"{label} – fitness distribution", fontsize=8, fontweight="bold")
        ax_fit.tick_params(labelsize=7)
        ax_fit.legend(fontsize=6.5, framealpha=0.6, loc="upper left")
        ax_fit.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
        print(f"Synthetic dataset overview saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def load_synthetic_datasets(
    data_dir: str | Path,
    proteins: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    deep: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Load all (or a subset of) synthetic CSV files into a labelled dict.

    Parameters
    ----------
    data_dir:
        Root directory of the synthetic data (the one that contains
        ``GB1_additive.csv`` etc.).  Pass the *deep/* sub-directory for deep
        variants.
    proteins:
        If given, only load data for these proteins (e.g. ``["GB1", "p53"]``).
    models:
        If given, only load these fitness models
        (e.g. ``["additive", "nonlinear"]``).
    deep:
        When *True* the search is restricted to the ``deep/`` sub-directory of
        *data_dir* (a convenience shortcut).

    Returns
    -------
    dict mapping ``"<Protein>_<model>"`` → ``pd.DataFrame``.
    """
    data_dir = Path(data_dir)
    if deep:
        data_dir = data_dir / "deep"

    _proteins = [p.upper() for p in proteins] if proteins else None
    _models   = [m.lower() for m in models] if models else None

    datasets: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        stem = csv_path.stem                   # e.g. "GB1_nonlinear"
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        protein_key, model_key = parts[0].upper(), parts[1].lower()
        if protein_key == "GROUND":            # skip ground_truth_weights
            continue
        if _proteins and protein_key not in _proteins:
            continue
        if _models and model_key not in _models:
            continue
        df = pd.read_csv(csv_path)
        datasets[f"{protein_key}_{model_key}"] = df

    return datasets


# ---------------------------------------------------------------------------
# Attribution vs. mutational-coverage comparison
# ---------------------------------------------------------------------------

def plot_attribution_vs_mutation_coverage(
    results: dict[str, pd.DataFrame],
    mut_freq: pd.Series,
    seq_length: int,
    ground_truth: Optional[dict] = None,
    uncovered_threshold_pct: float = 10.0,
    save_path: Optional[str] = None,
    dpi: int = 200,
) -> pd.DataFrame:
    """
    Compare per-position attribution importance against training-set mutational
    frequency, and flag positions with high attribution but no mutational coverage.

    Parameters
    ----------
    results:
        Mapping of ``label → DataFrame`` where each DataFrame is the contents
        of a ``*_position_importance_summary.csv`` file (columns: ``position``,
        ``mean_abs_gradient``, ``importance_score``, …).
    mut_freq:
        ``pd.Series`` indexed by 1-based position (int) giving the number of
        variants in the training dataset that carry a mutation at that position.
    seq_length:
        Full sequence length (used to build a complete position axis).
    ground_truth:
        Optional dict loaded from ``*_ground_truth_weights.json``.  When
        provided, annotated positions are shaded in the background.
    uncovered_threshold_pct:
        Positions whose normalised attribution exceeds this percentile AND
        whose mutation frequency is zero are flagged as *uncovered hot-spots*.
    save_path:
        Where to write the PNG.  Interactive display if *None*.
    dpi:
        Resolution for saved figure.

    Returns
    -------
    DataFrame of uncovered hot-spot positions (one row per label × position).
    """
    all_positions = np.arange(1, seq_length + 1)

    # Normalise mutation frequency to [0, 1]
    mut_arr = np.array([mut_freq.get(p, 0) for p in all_positions], dtype=float)
    mut_norm = mut_arr / mut_arr.max() if mut_arr.max() > 0 else mut_arr

    # Ground-truth annotation colours
    _GT_CATEGORY_COLORS = {
        "binding_site": "#d73027",
        "active_site":  "#d73027",
        "structural":   "#4575b4",
        "folding":      "#74add1",
        "catalytic":    "#a50026",
        "interface":    "#f46d43",
        "allosteric":   "#fee090",
    }
    gt_annotations: dict[int, str] = {}
    if ground_truth and "annotations" in ground_truth:
        gt_annotations = {int(k): v for k, v in ground_truth["annotations"].items()}

    n_methods = len(results)
    fig_height = 3.5 + 1.8 * n_methods
    fig, axes = plt.subplots(
        n_methods + 1, 1,
        figsize=(max(14, seq_length * 0.22), fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5] + [2.0] * n_methods},
    )
    fig.patch.set_facecolor(FIG_BG)

    # ── Top panel: mutational frequency bar ──────────────────────────────
    ax_mut = axes[0]
    ax_mut.set_facecolor(FIG_BG)

    # Shade ground-truth regions
    gt_shaded: dict[str, bool] = {}
    for pos in all_positions:
        cat = gt_annotations.get(pos)
        if cat:
            color = _GT_CATEGORY_COLORS.get(cat, "#9E9E9E")
            ax_mut.axvspan(pos - 0.5, pos + 0.5, color=color, alpha=0.12, linewidth=0)
            if cat not in gt_shaded:
                gt_shaded[cat] = True

    ax_mut.bar(all_positions, mut_arr, color="#78909C", alpha=0.75,
               width=0.8, label="Mutation count")
    ax_mut.set_ylabel("Mutation\ncount", fontsize=8)
    ax_mut.set_title("Training-set mutational frequency per position", fontsize=8,
                     fontweight="bold")
    ax_mut.tick_params(labelsize=7)
    ax_mut.grid(True, axis="y", alpha=0.2)
    ax_mut.set_axisbelow(True)

    if gt_shaded:
        handles = [
            mpatches.Patch(
                color=_GT_CATEGORY_COLORS.get(c, "#9E9E9E"),
                alpha=0.35,
                label=c.replace("_", " ").title(),
            )
            for c in gt_shaded
        ]
        ax_mut.legend(handles=handles, fontsize=6.5, framealpha=0.6, loc="upper left",
                      title="Ground truth", title_fontsize=6.5)

    # ── Per-method attribution panels ─────────────────────────────────────
    uncovered_rows = []
    threshold_val = np.percentile(
        np.concatenate([
            df.set_index("position").reindex(all_positions, fill_value=0)["mean_abs_gradient"].values
            for df in results.values()
        ]),
        100 - uncovered_threshold_pct,
    )

    method_colors = list(_INTERP_COLORS.values()) + ["#795548", "#00BCD4", "#E91E63"]

    for idx, (label, imp_df) in enumerate(results.items()):
        ax = axes[idx + 1]
        ax.set_facecolor(FIG_BG)
        color = method_colors[idx % len(method_colors)]

        # Reindex to all positions (fill missing with 0)
        pos_series = (
            imp_df.set_index("position")
            .reindex(all_positions, fill_value=0)["mean_abs_gradient"]
        )
        attr_arr = pos_series.values.astype(float)
        attr_norm = attr_arr / attr_arr.max() if attr_arr.max() > 0 else attr_arr

        # Ground-truth shading
        for pos in all_positions:
            cat = gt_annotations.get(pos)
            if cat:
                ax.axvspan(pos - 0.5, pos + 0.5,
                           color=_GT_CATEGORY_COLORS.get(cat, "#9E9E9E"),
                           alpha=0.10, linewidth=0)

        # Dual: normalised mutation freq (grey fill) + attribution (coloured line)
        ax.fill_between(all_positions, mut_norm, alpha=0.18, color="#78909C",
                        label="Mut. freq (norm)")
        ax.plot(all_positions, attr_norm, color=color, linewidth=1.4,
                label="Attribution (norm)")
        ax.scatter(all_positions, attr_norm, color=color, s=12, zorder=3, alpha=0.7)

        # Flag uncovered hot-spots (high attribution, zero mutation count)
        uncovered_mask = (attr_arr >= threshold_val) & (mut_arr == 0)
        uncovered_pos = all_positions[uncovered_mask]
        if len(uncovered_pos) > 0:
            ax.scatter(uncovered_pos, attr_norm[uncovered_mask],
                       color="#d73027", s=55, zorder=5, marker="*",
                       label=f"Uncovered hot-spot ({len(uncovered_pos)})")
            for pos in uncovered_pos:
                uncovered_rows.append({"label": label, "position": int(pos),
                                       "mean_abs_gradient": float(attr_arr[pos - 1]),
                                       "mutation_count": 0})

        ax.set_ylabel("Norm.\nvalue", fontsize=7.5)
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.set_ylim(-0.02, 1.08)
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=6.5, framealpha=0.6, loc="upper left", ncol=3)

    axes[-1].set_xlabel("Position (1-indexed)", fontsize=8)
    plt.tight_layout(h_pad=0.5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
        print(f"Attribution vs. mutation coverage saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)

    return pd.DataFrame(uncovered_rows)
