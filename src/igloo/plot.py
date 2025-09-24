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

    plt.figure(figsize=(20, 8))
    sns.heatmap(
        pivot_data,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Integrated Gradient'},
        xticklabels=True,
        yticklabels=True
    )
    plt.title(title)
    plt.xlabel('Position')
    plt.ylabel('Sequence')
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
    wt_result = ig_results['WT']
    wt_gradients = wt_result['position_gradients']

    plt.figure(figsize=(15, 10))
    plot_idx = 0

    for _i, (name, result) in enumerate(ig_results.items()):
        if name == 'WT':
            continue

        plot_idx += 1
        plt.subplot(len(ig_results)-1, 1, plot_idx)
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
