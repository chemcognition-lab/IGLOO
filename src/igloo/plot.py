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
    "axes.titlesize":      9,
    "axes.titleweight":    "bold",
    "axes.labelsize":      8,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     7,
    "lines.linewidth":     2.0,
    "lines.markersize":    7,
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
            plt.savefig(out_path, dpi=120, bbox_inches='tight')
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
        fig1.savefig(save_path, dpi=150, bbox_inches='tight')
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
        fig2.savefig(summary_path, dpi=150, bbox_inches='tight')
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
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
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sequence alignment plot saved to {save_path}")
    plt.close()


def setup_matplotlib_style() -> None:
    """Set up consistent matplotlib style for all plots."""
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    # Set Noto Sans font with scalable size
    base_font_size = 14
    plt.rcParams.update({
        'font.family': 'Noto Sans',
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
    fig.savefig(output_path, bbox_inches="tight")
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
        'font.family': 'Noto Sans',
        'font.size': 20,           # Increased from 16
        'axes.titlesize': 24,      # Increased from 18
        'axes.labelsize': 20,      # Increased from 16
        'xtick.labelsize': 16,     # Increased from 12
        'ytick.labelsize': 16,     # Increased from 12
        'legend.fontsize': 16      # Increased from 12
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
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
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
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
            plt.savefig(save_path, dpi=150)
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
        plt.savefig(save_path, dpi=150)
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
                ha="center", va="top", fontsize=7, fontweight="bold", color="dimgray",
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
    ax.legend(fontsize=7, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    _apply_group_decorations(ax, fm_groups, x_pos)

    fname = f"{protein_prefix}_ig_sign_accuracy.png" if protein_prefix else "synthetic_ig_sign_accuracy.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
            ax.annotate(n, (d, ig), textcoords="offset points", xytext=(4, 3), fontsize=7)
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
              fontsize=7)

    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    out_path = output_dir / f"{safe_label}_cliff_damage.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
            ax.legend(handles=prot_handles + extra_handles, fontsize=7, loc="upper right")
        elif extra_handles:
            ax.legend(handles=extra_handles, fontsize=7)
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
            ax.legend(handles=prot_handles + extra_handles, fontsize=7, loc="upper right")
        elif extra_handles:
            ax.legend(handles=extra_handles, fontsize=7)
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
                  [(plt.cm.Blues(s), lb) for _, lb, s in tols]], fontsize=7, loc="lower right")
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
        ax.legend(handles=config_handles, fontsize=7)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        _apply_group_decorations(ax, fm_groups, x_pos)

    fname = f"{protein_prefix}_completeness.png" if protein_prefix else "synthetic_completeness.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
    ax.legend(handles=config_handles, fontsize=7, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    _apply_group_decorations(ax, fm_groups, x_pos)

    fname = f"{protein_prefix}_precision_at_k.png" if protein_prefix else "synthetic_precision_at_k.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
        ax.set_xlabel("Noise level", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xticks(noise_levels)
        ax.set_xticklabels([str(n) for n in noise_levels], fontsize=7, rotation=45)
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
            leg = ax.legend(title="Interp path", fontsize=7, title_fontsize=7)
            if _has_near_zero:
                leg.get_frame().set_alpha(0.9)
                ax.plot([], [], color="dimgrey", linewidth=1.0, linestyle=":",
                        alpha=0.7, label="near-zero frac (right axis)")
                ax.legend(title="Interp path", fontsize=7, title_fontsize=7)

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
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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

        ax.set_xlabel("Noise level", fontsize=8)
        ax.set_ylabel(label.split("\n")[0], fontsize=8)
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.set_xticks(range(n_noise))
        ax.set_xticklabels([str(nl) for nl in noise_levels], fontsize=7, rotation=45)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

        if ax_idx == 0 and legend_handles:
            ax.legend(
                handles=legend_handles,
                title="Interp path",
                fontsize=7,
                title_fontsize=7,
            )

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    prot_label = protein_prefix or f"{protein}_{fitness_model}"
    split_label = f" [{split}]" if split and split != "all" else ""
    fig.suptitle(
        f"IG quality vs noise level — {protein} / {fitness_model}{split_label}\n"
        f"(dots = runs; error bars on per-seq metrics = ±1 SD over sequences; "
        f"★ = PolyAlanine reference; dashed grey = random baseline)",
        fontsize=9,
    )
    fname = f"{prot_label}_noise_vs_interp_boxplots.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
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
    _model_perf_xs: list = []
    _model_test_ys: list = []
    _model_val_ys:  list = []
    for _nd in data.values():
        for _n, _m in _nd.items():
            if _n not in _model_perf_xs:
                _ts = _m.get(_model_test_key, float("nan"))
                _vs = _m.get(_model_val_key,  float("nan"))
                if np.isfinite(_ts) or np.isfinite(_vs):
                    _model_perf_xs.append(_n)
                    _model_test_ys.append(_ts)
                    _model_val_ys.append(_vs)
    if _model_perf_xs:
        _order = np.argsort(_model_perf_xs)
        _xs_s  = [_model_perf_xs[i] for i in _order]
        _ts_s  = [_model_test_ys[i]  for i in _order]
        _vs_s  = [_model_val_ys[i]   for i in _order]
        ax.plot(_xs_s, _ts_s, color="black", lw=2.0, ls="--", marker="x", ms=8,
                label=_model_test_lbl, zorder=10)
        ax.plot(_xs_s, _vs_s, color="#555",  lw=1.5, ls=":",  marker="+", ms=8,
                label=_model_val_lbl, zorder=10)
    _style_ax(ax, f"Global {_metric_label}\n({track} vs GT importance)",
              ylabel=f"{_metric_label} (↑ better)")
    ax.legend(handles=_interp_legend(interps) + [
        Line2D([0],[0], color="black", lw=2, ls="--", marker="x", ms=8, label=_model_test_lbl),
        Line2D([0],[0], color="#555",  lw=1.5, ls=":", marker="+", ms=8, label=_model_val_lbl),
    ], fontsize=7)

    ax = fig.add_subplot(gs[0, 1])
    _plot_metric_line(ax, data, f"auroc{sfx}", hline=0.5, **kw)
    _style_ax(ax, "AUROC\n(annotated vs background)", ylabel="AUROC (↑ better)")
    ax.legend(handles=_interp_legend(interps), fontsize=7)

    ax = fig.add_subplot(gs[0, 2])
    _plot_metric_line(ax, data, f"auprc{sfx}", **kw)
    _style_ax(ax, "AUPRC\n(annotated vs background)", ylabel="AUPRC (↑ better)")
    ax.legend(handles=_interp_legend(interps), fontsize=7)

    ax = fig.add_subplot(gs[0, 3])
    _plot_metric_line(ax, data, f"{_prec_k_key}{sfx}", **kw)
    _style_ax(ax, _prec_k_title, ylabel=f"{_prec_k_lbl} (↑ better)")
    ax.legend(handles=_interp_legend(interps), fontsize=7)

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
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 2:4])
    _plot_metric_line(ax, data, "sign_acc", hline=0.5, **kw)
    _style_ax(ax, "Sign accuracy\n(mutated annotated positions)",
              ylabel="Fraction correct (↑ better)")
    ax.set_ylim(0, 1)
    ax.legend(handles=_interp_legend(interps), fontsize=7)

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
        ax.legend(fontsize=7)

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
        ax.legend(handles=_cat_legend_handles, fontsize=6)

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
        ax.legend(fontsize=7)

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
        ax.legend(fontsize=6)

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
        ax.set_xticklabels([f"{n:.2g}" for n in sorted_noises], fontsize=8)
        _style_ax(
            ax,
            f"Per-variant {_metric_label} dist. vs GT ({track})\n{INTERP_LABELS.get(interp, interp)}",
            xlabel="Noise level",
            ylabel=f"Per-variant {_metric_label} (↑ better)",
        )
        if col_idx == 2:
            sm = plt.cm.ScalarMappable(cmap=cmap_noise, norm=noise_norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02).set_label("Noise level", fontsize=8)

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
    ax.set_xticklabels(metric_lbls, fontsize=7, rotation=35, ha="right")
    ax.set_yticks(range(len(row_lbls_hm)))
    ax.set_yticklabels(row_lbls_hm, fontsize=7)
    ax.set_title("Metric heatmap (column-normalised)\ngreen = best, red = worst",
                 fontsize=8, fontweight="bold")
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
        ax.set_xticklabels([f"{n:.2g}" for n in sorted_noises], fontsize=8)
        _style_ax(
            ax,
            f"Per-sequence mutation ρ distribution\n{INTERP_LABELS.get(interp, interp)}",
            xlabel="Noise level",
            ylabel=f"{_metric_label} — LOO damage vs {track} (↑ better)",
        )
        ax.legend(fontsize=7)
        if col_idx == 2:
            sm = plt.cm.ScalarMappable(cmap=cmap_noise, norm=noise_norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02).set_label("Noise level", fontsize=8)

    ax = fig.add_subplot(gs[4, 3])
    _plot_metric_line(ax, data, _mut_sp_key, hline=None if _is_ndcg else 0.0, **kw)
    _style_ax(
        ax,
        f"Mean per-sequence mutation {_metric_label}\n(LOO damage vs {track})",
        ylabel=f"{_metric_label} (↑ better)",
    )
    ax.legend(handles=_interp_legend(interps), fontsize=7)
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
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")
