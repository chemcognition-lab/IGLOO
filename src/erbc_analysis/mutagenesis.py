"""
In-silico mutagenesis analysis for protein variant effect prediction.

This module provides functions to systematically mutate protein sequences
and predict the effects using trained models.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_processing import CustomDataset, batch_pad_and_normalize
from .embedding import get_embeddings


def compute_mutation_effects(
    model: torch.nn.Module,
    wt_sequence: str,
    device: torch.device,
    esm_layer: int = 33,
    y_log_mean: float = 0.0,
    y_log_std: float = 1.0,
    pooling_method: str = 'linear',
    batch_size: int = 16,
    progress: bool = True
) -> pd.DataFrame:
    """
    Compute the predicted effect of all possible single amino acid mutations.

    Args:
        model: Trained PyTorch model for prediction
        wt_sequence: Wild-type protein sequence
        device: PyTorch device (cuda/cpu)
        esm_layer: ESM layer to use for embeddings
        y_log_mean: Mean used for target normalization (to convert back)
        y_log_std: Std used for target normalization (to convert back)
        pooling_method: Model pooling method ('linear', 'esm_attention', etc.)
        batch_size: Batch size for ESM embedding computation
        progress: Whether to show progress bar

    Returns:
        DataFrame with columns: position, wt_aa, mut_aa, mutation,
                               predicted_activity, effect_vs_wt, percentile_rank
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    model.eval()

    # Get WT prediction for baseline using the same method as mutants
    print("Computing WT baseline prediction...")
    wt_embedding = get_embeddings([wt_sequence], layer=esm_layer, batch_size=1, dataset_name='PafA_mutagenesis')

    # Use DataLoader approach for consistency
    dummy_target = np.zeros(1)
    wt_dataset = CustomDataset(wt_embedding, dummy_target)
    wt_loader = DataLoader(wt_dataset, batch_size=1, shuffle=False, collate_fn=batch_pad_and_normalize)

    with torch.no_grad():
        for xb, lengths, _yb in wt_loader:
            xb = xb.to(device)
            if pooling_method == 'esm_attention':
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                wt_pred_norm = model(xb, input_mask).item()
            else:
                wt_pred_norm = model(xb).item()

    # Convert back to original scale
    wt_pred = wt_pred_norm * y_log_std + y_log_mean
    print(f"WT predicted activity: {wt_pred:.4f}")

    mutation_effects = []
    sequences_to_embed = []
    mutation_info = []

    # Prepare all mutant sequences
    for pos in range(len(wt_sequence)):
        wt_aa = wt_sequence[pos]
        for mut_aa in amino_acids:
            if mut_aa == wt_aa:
                continue

            # Create mutant sequence
            mut_seq = wt_sequence[:pos] + mut_aa + wt_sequence[pos+1:]
            sequences_to_embed.append(mut_seq)
            mutation_info.append({
                'position': pos + 1,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa,
                'mutation': f"{wt_aa}{pos+1}{mut_aa}"
            })

    print(f"Computing embeddings for {len(sequences_to_embed)} mutations...")

    # Get embeddings for all mutants
    all_embeddings = get_embeddings(sequences_to_embed,
                                    layer=esm_layer,
                                    batch_size=batch_size,
                                    dataset_name='PafA_mutagenesis')

    print(f"Predicting activities for {len(sequences_to_embed)} mutations...")

    # Predict activities using DataLoader (to match training/eval logic)
    predictions = []
    model.eval()

    # Create dummy target values (not used for prediction, but required by CustomDataset)
    dummy_targets = np.zeros(len(all_embeddings))

    # Create dataset and dataloader
    eval_dataset = CustomDataset(all_embeddings, dummy_targets)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_pad_and_normalize)

    pred_norm = []
    with torch.no_grad():
        for xb, lengths, _yb in tqdm(eval_loader, disable=not progress, desc="Predicting"):
            xb = xb.to(device)
            if pooling_method == 'esm_attention':
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                out = model(xb, input_mask).squeeze(-1).cpu().numpy()
            else:
                out = model(xb).squeeze(-1).cpu().numpy()
            pred_norm.append(out)

    pred_norm = np.concatenate(pred_norm)
    # Convert back to original scale
    predictions = pred_norm * y_log_std + y_log_mean

    # Combine results
    for info, pred in zip(mutation_info, predictions):
        effect_vs_wt = pred - wt_pred
        mutation_effects.append({
            **info,
            'predicted_activity': pred,
            'wt_activity': wt_pred,
            'effect_vs_wt': effect_vs_wt,
            'fold_change': pred / wt_pred if wt_pred != 0 else np.inf
        })

    df = pd.DataFrame(mutation_effects)

    # Add percentile ranks
    df['percentile_rank'] = df['predicted_activity'].rank(pct=True) * 100
    df['effect_percentile_rank'] = df['effect_vs_wt'].rank(pct=True) * 100

    # Sort by predicted effect (best mutations first)
    df = df.sort_values('effect_vs_wt', ascending=False).reset_index(drop=True)

    return df


def analyze_position_preferences(
    mutation_df: pd.DataFrame,
    top_n: int = 10,
    min_effect: float = None
) -> dict:
    """
    Analyze amino acid preferences at each position.

    Args:
        mutation_df: DataFrame from compute_mutation_effects
        top_n: Number of top/bottom mutations to highlight
        min_effect: Minimum effect size to consider for preferences

    Returns:
        Dictionary with analysis results
    """
    if min_effect is not None:
        filtered_df = mutation_df[mutation_df['effect_vs_wt'].abs() >= min_effect]
    else:
        filtered_df = mutation_df

    # Best and worst mutations overall
    top_mutations = mutation_df.head(top_n)
    worst_mutations = mutation_df.tail(top_n)

    # Position-level analysis
    position_stats = filtered_df.groupby('position').agg({
        'effect_vs_wt': ['mean', 'std', 'min', 'max', 'count'],
        'predicted_activity': ['mean', 'std']
    }).round(3)

    # Amino acid preferences
    aa_stats = filtered_df.groupby('mut_aa').agg({
        'effect_vs_wt': ['mean', 'std', 'count'],
        'predicted_activity': 'mean'
    }).round(3)

    # Most variable positions
    position_variability = filtered_df.groupby('position')['effect_vs_wt'].std().sort_values(ascending=False)

    return {
        'top_mutations': top_mutations,
        'worst_mutations': worst_mutations,
        'position_stats': position_stats,
        'aa_preferences': aa_stats,
        'most_variable_positions': position_variability,
        'summary': {
            'total_mutations': len(mutation_df),
            'beneficial_mutations': len(mutation_df[mutation_df['effect_vs_wt'] > 0]),
            'detrimental_mutations': len(mutation_df[mutation_df['effect_vs_wt'] < 0]),
            'best_effect': mutation_df['effect_vs_wt'].max(),
            'worst_effect': mutation_df['effect_vs_wt'].min(),
            'mean_effect': mutation_df['effect_vs_wt'].mean()
        }
    }


def plot_mutation_landscape(
    mutation_df: pd.DataFrame,
    wt_sequence: str,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (20, 12),
    top_n_label: int = 10
) -> None:
    """
    Create comprehensive visualization of mutation effects.

    Args:
        mutation_df: DataFrame from compute_mutation_effects
        wt_sequence: Wild-type sequence for reference
        save_path: Path to save the plot
        figsize: Figure size
        top_n_label: Number of top mutations to label
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Heatmap of mutation effects
    pivot_data = mutation_df.pivot(index='mut_aa', columns='position', values='effect_vs_wt')

    sns.heatmap(
        pivot_data,
        ax=axes[0, 0],
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Effect vs WT'},
        xticklabels=True,
        yticklabels=True
    )
    axes[0, 0].set_title('Mutation Effect Heatmap')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Mutant Amino Acid')

    # 2. Best mutations per position
    best_per_position = mutation_df.loc[mutation_df.groupby('position')['effect_vs_wt'].idxmax()]

    axes[0, 1].bar(best_per_position['position'], best_per_position['effect_vs_wt'],
                   color=['green' if x > 0 else 'red' for x in best_per_position['effect_vs_wt']])
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Best Mutation Effect per Position')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Best Effect vs WT')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Distribution of mutation effects
    axes[1, 0].hist(mutation_df['effect_vs_wt'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', label='WT baseline')
    axes[1, 0].set_title('Distribution of Mutation Effects')
    axes[1, 0].set_xlabel('Effect vs WT')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()

    # 4. Top mutations scatter plot
    top_mutations = mutation_df.head(top_n_label)

    axes[1, 1].scatter(
        mutation_df['position'],
        mutation_df['effect_vs_wt'],
        c=mutation_df['effect_vs_wt'],
        cmap='RdBu_r',
        alpha=0.6,
        s=30
    )

    # Label top mutations
    for _, row in top_mutations.iterrows():
        axes[1, 1].annotate(
            row['mutation'],
            (row['position'], row['effect_vs_wt']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, alpha=0.8
        )

    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title(f'Mutation Effects by Position (Top {top_n_label} labeled)')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Effect vs WT')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mutation landscape plot saved to {save_path}")

    plt.show()


def export_top_mutations(
    mutation_df: pd.DataFrame,
    output_path: str,
    top_n: int = 100,
    include_negative: bool = True
) -> None:
    """
    Export top mutations to CSV with detailed information.

    Args:
        mutation_df: DataFrame from compute_mutation_effects
        output_path: Path to save CSV file
        top_n: Number of top/bottom mutations to include
        include_negative: Whether to include worst mutations
    """
    top_positive = mutation_df.head(top_n)

    if include_negative:
        top_negative = mutation_df.tail(top_n)
        export_df = pd.concat([top_positive, top_negative])
    else:
        export_df = top_positive

    # Add some additional annotations
    export_df = export_df.copy()
    export_df['effect_category'] = export_df['effect_vs_wt'].apply(
        lambda x: 'Highly beneficial' if x > export_df['effect_vs_wt'].quantile(0.95)
        else 'Beneficial' if x > 0
        else 'Detrimental' if x > export_df['effect_vs_wt'].quantile(0.05)
        else 'Highly detrimental'
    )

    export_df.to_csv(output_path, index=False)
    print(f"Top mutations exported to {output_path}")
    print(f"Exported {len(export_df)} mutations")
