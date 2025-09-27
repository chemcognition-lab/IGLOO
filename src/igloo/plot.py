"""
Plotting utilities for IGLOO analysis.

This module contains all plotting functions used across the IGLOO analysis scripts,
providing a centralized location for visualization code.
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_integrated_gradients(results_dict: dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot integrated gradients for multiple sequences.

    Args:
        results_dict: Dictionary containing analysis results for each sequence
        save_path: Optional path to save the plot
    """
    n_sequences = len(results_dict)
    fig, axes = plt.subplots(n_sequences, 1, figsize=(20, 4 * n_sequences))

    if n_sequences == 1:
        axes = [axes]

    for idx, (name, result) in enumerate(results_dict.items()):
        ax = axes[idx]
        positions = result['positions']
        gradients = result['position_gradients']
        amino_acids = result['amino_acids']

        # Create bar plot
        colors = ['red' if g < 0 else 'blue' for g in gradients]
        ax.bar(positions, gradients, color=colors, alpha=0.7)

        # Add amino acid labels
        for pos, aa, grad in zip(positions, amino_acids, gradients):
            if abs(grad) > np.percentile(np.abs(gradients), 90):  # Label top 10% gradients
                ax.text(pos, grad + np.sign(grad) * 0.01 * max(np.abs(gradients)),
                       aa, ha='center', va='bottom' if grad > 0 else 'top',
                       fontweight='bold', fontsize=8)

        ax.set_xlabel('Position')
        ax.set_ylabel('Integrated Gradient')
        ax.set_title(f'{name}\nPredicted: {result["predicted_activity"]:.3f}, '
                    f'Effect vs Baseline: {result["effect_vs_baseline"]:.3f}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Integrated gradients plot saved to {save_path}")
    plt.close()


def plot_gradients_heatmap(results_df: pd.DataFrame, title: str, save_path: Optional[str] = None) -> None:
    """
    Create a heatmap of gradients across sequences.

    Args:
        results_df: DataFrame with columns ['sequence_name', 'position', 'integrated_gradient']
        title: Title for the heatmap
        save_path: Optional path to save the plot
    """
    pivot_data = results_df.pivot(index='sequence_name', columns='position', values='integrated_gradient')

    # Calculate dynamic width based on number of positions
    num_positions = len(pivot_data.columns)
    # Use at least 0.1 inches per position, with a minimum of 30 inches
    width = max(30, num_positions * 0.1)

    plt.figure(figsize=(width, 8))

    # Show every 10th position label to reduce crowding
    x_tick_frequency = max(1, num_positions // 50)  # Show ~50 labels max
    x_tick_labels = [str(pos) if i % x_tick_frequency == 0 else ''
                     for i, pos in enumerate(pivot_data.columns)]

    sns.heatmap(
        pivot_data,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Integrated Gradient'},
        xticklabels=x_tick_labels,
        yticklabels=True
    )
    plt.title(title)
    plt.xlabel('Position')
    plt.ylabel('Sequence')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gradient heatmap saved to {save_path}")
    plt.close()


def plot_variants_vs_wt_gradients(ig_results: dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot gradient differences between variants and wild-type.

    Args:
        ig_results: Dictionary containing integrated gradients results
        save_path: Optional path to save the plot
    """
    # Find wildtype result - try common keys
    wt_key = None
    for key in ['WT', 'Wildtype', 'wildtype', 'wt']:
        if key in ig_results:
            wt_key = key
            break

    if wt_key is None:
        # If no wildtype found, use the first result as reference
        wt_key = list(ig_results.keys())[0]

    wt_result = ig_results[wt_key]
    wt_gradients = wt_result['position_gradients']

    # Check if we have any variants to compare (more than just wildtype)
    variant_count = len([name for name in ig_results.keys() if name != wt_key])

    if variant_count == 0:
        print("Only wildtype sequence found, skipping variant comparison plot.")
        return

    plt.figure(figsize=(15, 10))
    plot_idx = 0

    for _i, (name, result) in enumerate(ig_results.items()):
        if name == wt_key:
            continue

        plot_idx += 1
        plt.subplot(variant_count, 1, plot_idx)
        positions = result['positions']
        gradient_diff = result['position_gradients'] - wt_gradients

        colors = ['red' if g < 0 else 'blue' for g in gradient_diff]
        plt.bar(positions, gradient_diff, color=colors, alpha=0.7)

        plt.xlabel('Position')
        plt.ylabel('Gradient Difference from WT')
        plt.title(f'{name} - WT (Activity: {result["predicted_activity"]:.1f} '
                  f'vs {wt_result["predicted_activity"]:.1f})')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    from scipy import stats
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
