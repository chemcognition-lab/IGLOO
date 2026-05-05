"""
Advanced Integrated Gradients analysis for protein sequences.

This module provides functions for computing integrated gradients with respect
to sequence positions and interpreting the results in terms of amino acid
contributions and sequence motifs.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

from .embedding import get_embeddings


def integrated_gradients_embeddings(
    model: torch.nn.Module,
    input_embedding: np.ndarray,
    baseline_embedding: np.ndarray,
    device: torch.device,
    pooling_method: str = 'linear',
    steps: int = 50,
    batch_size: int = 10
) -> np.ndarray:
    """
    Compute integrated gradients with respect to embedding space.
    
    Args:
        model: Trained PyTorch model
        input_embedding: Input sequence embedding (1, L, D)
        baseline_embedding: Baseline sequence embedding (1, L, D)
        device: PyTorch device
        pooling_method: Model pooling method
        steps: Number of integration steps
        batch_size: Batch size for gradient computation
        
    Returns:
        Integrated gradients array (1, L, D)
    """
    model.eval()
    
    # Convert to tensors
    input_tensor = torch.tensor(input_embedding, dtype=torch.float32, device=device)
    baseline_tensor = torch.tensor(baseline_embedding, dtype=torch.float32, device=device)
    
    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, steps + 1, device=device)
    interpolated_inputs = []
    
    for alpha in alphas:
        interpolated = baseline_tensor + alpha * (input_tensor - baseline_tensor)
        interpolated_inputs.append(interpolated)
    
    # Compute gradients for batches
    all_gradients = []
    
    for i in range(0, len(interpolated_inputs), batch_size):
        batch = torch.stack(interpolated_inputs[i:i + batch_size])
        batch.requires_grad_(True)
        
        # Forward pass
        if pooling_method == 'esm_attention':
            # Create input masks
            seq_len = batch.size(2)
            input_masks = torch.ones(batch.size(0), seq_len, device=device)
            outputs = model(batch, input_masks)
        else:
            outputs = model(batch)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=batch,
            create_graph=False,
            retain_graph=False
        )[0]
        
        all_gradients.append(gradients.detach())
    
    # Combine all gradients
    all_gradients = torch.cat(all_gradients, dim=0)
    
    # Compute integrated gradients
    avg_gradients = torch.mean(all_gradients, dim=0, keepdim=True)
    integrated_grads = (input_tensor - baseline_tensor) * avg_gradients
    
    return integrated_grads.cpu().numpy()


def compute_residue_attributions(
    integrated_grads: np.ndarray,
    method: str = 'l2_norm'
) -> np.ndarray:
    """
    Convert embedding-level gradients to residue-level attributions.
    
    Args:
        integrated_grads: Integrated gradients (1, L, D)
        method: Method for aggregation ('l2_norm', 'sum', 'mean', 'max')
        
    Returns:
        Residue attributions (L,)
    """
    grads = integrated_grads[0]  # Remove batch dimension
    
    if method == 'l2_norm':
        return np.linalg.norm(grads, axis=1)
    elif method == 'sum':
        return np.sum(np.abs(grads), axis=1)
    elif method == 'mean':
        return np.mean(np.abs(grads), axis=1)
    elif method == 'max':
        return np.max(np.abs(grads), axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_amino_acid_contributions(
    model: torch.nn.Module,
    sequence: str,
    baseline_sequence: str,
    device: torch.device,
    esm_layer: int = 33,
    pooling_method: str = 'linear',
    perturbation_size: float = 0.1,
    n_samples: int = 100
) -> pd.DataFrame:
    """
    Estimate amino acid contributions using embedding perturbations.
    
    This method perturbs embeddings in directions corresponding to different
    amino acids and measures the effect on predictions.
    
    Args:
        model: Trained PyTorch model
        sequence: Input sequence
        baseline_sequence: Baseline sequence (usually WT)
        device: PyTorch device
        esm_layer: ESM layer for embeddings
        pooling_method: Model pooling method
        perturbation_size: Size of perturbations
        n_samples: Number of random perturbations per position
        
    Returns:
        DataFrame with amino acid contributions per position
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Get embeddings for all amino acids
    aa_embeddings = {}
    for aa in amino_acids:
        # Create single AA sequence for embedding
        aa_seq = aa
        emb = get_embeddings([aa_seq], layer=esm_layer, batch_size=1)
        aa_embeddings[aa] = emb[0, 0, :]  # Single residue embedding
    
    # Get input and baseline embeddings
    input_emb = get_embeddings([sequence], layer=esm_layer, batch_size=1)
    baseline_emb = get_embeddings([baseline_sequence], layer=esm_layer, batch_size=1)
    
    # Get baseline prediction
    baseline_tensor = torch.tensor(baseline_emb, dtype=torch.float32).to(device)
    with torch.no_grad():
        if pooling_method == 'esm_attention':
            input_mask = torch.ones(1, len(baseline_sequence), device=device)
            baseline_pred = model(baseline_tensor, input_mask).item()
        else:
            baseline_pred = model(baseline_tensor).item()
    
    contributions = []
    
    for pos in range(len(sequence)):
        current_aa = sequence[pos]
        
        for target_aa in amino_acids:
            if target_aa == current_aa:
                continue
            
            # Create perturbed embedding
            perturbed_emb = input_emb.copy()
            aa_direction = aa_embeddings[target_aa] - aa_embeddings[current_aa]
            aa_direction = aa_direction / np.linalg.norm(aa_direction)  # Normalize
            
            # Apply perturbation
            perturbed_emb[0, pos, :] += perturbation_size * aa_direction
            
            # Get prediction
            perturbed_tensor = torch.tensor(perturbed_emb, dtype=torch.float32).to(device)
            with torch.no_grad():
                if pooling_method == 'esm_attention':
                    input_mask = torch.ones(1, len(sequence), device=device)
                    pred = model(perturbed_tensor, input_mask).item()
                else:
                    pred = model(perturbed_tensor).item()
            
            effect = pred - baseline_pred
            
            contributions.append({
                'position': pos + 1,
                'current_aa': current_aa,
                'target_aa': target_aa,
                'mutation': f"{current_aa}{pos+1}{target_aa}",
                'contribution': effect
            })
    
    return pd.DataFrame(contributions)


def analyze_sequence_motifs(
    attributions: np.ndarray,
    sequence: str,
    window_size: int = 5,
    top_k: int = 10
) -> Dict:
    """
    Identify important sequence motifs based on attribution scores.
    
    Args:
        attributions: Residue attribution scores
        sequence: Protein sequence
        window_size: Size of motif windows
        top_k: Number of top motifs to return
        
    Returns:
        Dictionary with motif analysis results
    """
    motifs = []
    
    for i in range(len(sequence) - window_size + 1):
        motif_seq = sequence[i:i + window_size]
        motif_score = np.sum(attributions[i:i + window_size])
        
        motifs.append({
            'start_pos': i + 1,
            'end_pos': i + window_size,
            'sequence': motif_seq,
            'score': motif_score,
            'mean_score': motif_score / window_size
        })
    
    motifs_df = pd.DataFrame(motifs)
    motifs_df = motifs_df.sort_values('score', ascending=False)
    
    return {
        'top_motifs': motifs_df.head(top_k),
        'bottom_motifs': motifs_df.tail(top_k),
        'all_motifs': motifs_df
    }


def plot_attribution_landscape(
    attributions: np.ndarray,
    sequence: str,
    title: str = "Residue Attribution Landscape",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 8),
    highlight_threshold: float = None
) -> None:
    """
    Create comprehensive visualization of attribution scores.
    
    Args:
        attributions: Residue attribution scores
        sequence: Protein sequence
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        highlight_threshold: Threshold for highlighting important residues
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    positions = np.arange(1, len(sequence) + 1)
    
    # 1. Attribution bar plot
    colors = ['red' if attr > 0 else 'blue' for attr in attributions]
    if highlight_threshold:
        colors = ['gold' if abs(attr) > highlight_threshold else c for attr, c in zip(attributions, colors)]
    
    axes[0].bar(positions, attributions, color=colors, alpha=0.7)
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0].set_title(f'{title} - Attribution Scores')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Attribution Score')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Cumulative attribution
    cumulative = np.cumsum(attributions)
    axes[1].plot(positions, cumulative, linewidth=2, color='purple')
    axes[1].fill_between(positions, cumulative, alpha=0.3, color='purple')
    axes[1].set_title('Cumulative Attribution')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Cumulative Score')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Sequence with attribution heatmap
    # Create a heatmap-style visualization
    attr_matrix = attributions.reshape(1, -1)
    im = axes[2].imshow(attr_matrix, aspect='auto', cmap='RdBu_r', 
                       extent=[0.5, len(sequence) + 0.5, -0.5, 0.5])
    
    # Add sequence labels
    for i, aa in enumerate(sequence):
        axes[2].text(i + 1, 0, aa, ha='center', va='center', 
                    fontsize=8, fontweight='bold')
    
    axes[2].set_title('Sequence with Attribution Heatmap')
    axes[2].set_xlabel('Position')
    axes[2].set_yticks([])
    axes[2].set_xticks(positions[::10])  # Show every 10th position
    
    # Add colorbar
    plt.colorbar(im, ax=axes[2], orientation='horizontal', pad=0.1, shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Attribution landscape saved to {save_path}")
    
    plt.show()


def compare_attribution_methods(
    model: torch.nn.Module,
    sequence: str,
    baseline_sequence: str,
    device: torch.device,
    esm_layer: int = 33,
    pooling_method: str = 'linear'
) -> Dict[str, np.ndarray]:
    """
    Compare different attribution methods on the same sequence.
    
    Args:
        model: Trained PyTorch model
        sequence: Input sequence
        baseline_sequence: Baseline sequence
        device: PyTorch device
        esm_layer: ESM layer for embeddings
        pooling_method: Model pooling method
        
    Returns:
        Dictionary with attribution scores from different methods
    """
    # Get embeddings
    input_emb = get_embeddings([sequence], layer=esm_layer, batch_size=1)
    baseline_emb = get_embeddings([baseline_sequence], layer=esm_layer, batch_size=1)
    
    results = {}
    
    # Method 1: Integrated Gradients (L2 norm)
    ig_emb = integrated_gradients_embeddings(
        model, input_emb, baseline_emb, device, pooling_method
    )
    results['integrated_gradients_l2'] = compute_residue_attributions(ig_emb, 'l2_norm')
    
    # Method 2: Integrated Gradients (sum of absolute values)
    results['integrated_gradients_sum'] = compute_residue_attributions(ig_emb, 'sum')
    
    # Method 3: Simple gradient (for comparison)
    input_tensor = torch.tensor(input_emb, dtype=torch.float32, device=device)
    input_tensor.requires_grad_(True)
    
    if pooling_method == 'esm_attention':
        input_mask = torch.ones(1, len(sequence), device=device)
        output = model(input_tensor, input_mask)
    else:
        output = model(input_tensor)
    
    gradients = torch.autograd.grad(output, input_tensor)[0]
    results['simple_gradients'] = np.linalg.norm(gradients[0].cpu().numpy(), axis=1)
    
    # Method 4: Occlusion-based (mask each position)
    occlusion_scores = []
    baseline_tensor = torch.tensor(baseline_emb, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        if pooling_method == 'esm_attention':
            input_mask = torch.ones(1, len(baseline_sequence), device=device)
            baseline_pred = model(baseline_tensor, input_mask).item()
        else:
            baseline_pred = model(baseline_tensor).item()
    
    for pos in range(len(sequence)):
        occluded_emb = input_emb.copy()
        occluded_emb[0, pos, :] = baseline_emb[0, pos, :]  # Replace with baseline
        
        occluded_tensor = torch.tensor(occluded_emb, dtype=torch.float32).to(device)
        with torch.no_grad():
            if pooling_method == 'esm_attention':
                input_mask = torch.ones(1, len(sequence), device=device)
                occluded_pred = model(occluded_tensor, input_mask).item()
            else:
                occluded_pred = model(occluded_tensor).item()
        
        # Score is the difference when this position is occluded
        if pooling_method == 'esm_attention':
            input_mask = torch.ones(1, len(sequence), device=device)
            original_pred = model(input_tensor.detach(), input_mask).item()
        else:
            original_pred = model(input_tensor.detach()).item()
        
        occlusion_scores.append(original_pred - occluded_pred)
    
    results['occlusion'] = np.array(occlusion_scores)
    
    return results


def export_attribution_results(
    attributions: Dict[str, np.ndarray],
    sequence: str,
    output_path: str,
    include_motifs: bool = True
) -> None:
    """
    Export attribution results to CSV with comprehensive analysis.
    
    Args:
        attributions: Dictionary of attribution scores from different methods
        sequence: Protein sequence
        output_path: Path to save CSV file
        include_motifs: Whether to include motif analysis
    """
    results = []
    
    for pos, aa in enumerate(sequence):
        row = {
            'position': pos + 1,
            'amino_acid': aa,
        }
        
        # Add attribution scores from all methods
        for method, scores in attributions.items():
            row[f'{method}_score'] = scores[pos]
            row[f'{method}_rank'] = len(scores) - np.argsort(np.argsort(scores))[pos]
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Add consensus ranking (average of ranks)
    rank_columns = [col for col in df.columns if col.endswith('_rank')]
    df['consensus_rank'] = df[rank_columns].mean(axis=1)
    df['consensus_score'] = df[[col for col in df.columns if col.endswith('_score')]].mean(axis=1)
    
    df = df.sort_values('consensus_rank').reset_index(drop=True)
    
    df.to_csv(output_path, index=False)
    print(f"Attribution results exported to {output_path}")
    print(f"Methods compared: {list(attributions.keys())}")
