"""
Plotting utilities for IGLOO analysis.

This module contains all plotting functions used across the IGLOO analysis scripts,
providing a centralized location for visualization code.
"""

import os
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


def plot_signed_gradient_effects(
    results_dict: dict[str, Any],
    save_path: Optional[str] = None,
    top_n: int = 10,
    figsize: tuple[int, int] = (15, 8)
) -> None:
    """
    Create a scatter plot showing signed gradient effects by position, similar to a mutation effect plot.

    This function creates a visualization with:
    - Red/orange points for beneficial effects (positive gradients)
    - Blue points for detrimental effects (negative gradients)
    - Points scattered by position with some vertical jitter for clarity
    - Top N most significant effects labeled
    Args:
        results_dict: Dictionary containing analysis results for each sequence
        save_path: Optional path to save the plot
        top_n: Number of top effects to label (default: 10)
        figsize: Figure size as tuple (width, height)
    """
    # Collect all gradient data
    all_effects = []

    for name, result in results_dict.items():
        positions = result['positions']
        gradients = result['position_gradients']
        amino_acids = result['amino_acids']

        for pos, grad, aa in zip(positions, gradients, amino_acids):
            all_effects.append({
                'sequence_name': name,
                'position': pos,
                'amino_acid': aa,
                'effect': grad,
                'abs_effect': abs(grad)
            })

    effects_df = pd.DataFrame(all_effects)

    # Separate beneficial and detrimental effects
    beneficial = effects_df[effects_df['effect'] > 0].copy()
    detrimental = effects_df[effects_df['effect'] < 0].copy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Add some vertical jitter for better visibility
    np.random.seed(42)  # For reproducible jitter

    # Plot beneficial effects (positive gradients) in red/orange
    if len(beneficial) > 0:
        beneficial_jitter = np.random.normal(0, 5, len(beneficial))  # Small jitter
        ax.scatter(
            beneficial['position'],
            beneficial['effect'] + beneficial_jitter,
            c='coral',
            alpha=0.6,
            s=30,
            label=f'Beneficial ({len(beneficial)} positions)'
        )

    # Plot detrimental effects (negative gradients) in blue
    if len(detrimental) > 0:
        detrimental_jitter = np.random.normal(0, 5, len(detrimental))  # Small jitter
        ax.scatter(
            detrimental['position'],
            detrimental['effect'] + detrimental_jitter,
            c='steelblue',
            alpha=0.6,
            s=30,
            label=f'Detrimental ({len(detrimental)} positions)'
        )

    # Add horizontal line at zero
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Label top N most significant effects
    top_effects = effects_df.nlargest(top_n, 'abs_effect')

    for _, row in top_effects.iterrows():
        # Add small random jitter for label positioning
        jitter_y = np.random.normal(0, 2)

        ax.annotate(
            f"{row['amino_acid']}{int(row['position'])}",
            xy=(row['position'], row['effect']),
            xytext=(row['position'], row['effect'] + jitter_y + np.sign(row['effect']) * 10),
            fontsize=8,
            ha='center',
            va='bottom' if row['effect'] > 0 else 'top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
            arrowprops=dict(arrowstyle='->', color='black', alpha=0.5, lw=0.5)
        )

    # Formatting
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Signed Gradient Effect', fontsize=12)
    ax.set_title('Signed Gradient Effects by Position', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Set y-axis limits with some padding
    y_min, y_max = effects_df['effect'].min(), effects_df['effect'].max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    # Extract predicted activities and sort sequences
    sequence_activities = []
    for name, result in results_dict.items():
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
        {'beneficial': 'orangered', 'detrimental': 'darkblue'}
    )

    # Plot bottom sequences (use cooler colors)
    bottom_avg_activity = np.mean([seq['predicted_activity'] for seq in bottom_sequences])
    plot_effects(
        ax2, bottom_effects,
        f'Bottom {bottom_n} Sequences (Avg Activity: {bottom_avg_activity:.2f})',
        {'beneficial': 'coral', 'detrimental': 'steelblue'}
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
