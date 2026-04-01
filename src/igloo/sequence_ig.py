"""
True sequence-level Integrated Gradients using ESM model directly.

This module provides functions for computing integrated gradients with respect
to the actual sequence tokens, not embeddings, allowing direct interpretation
of which amino acids at which positions contribute to predictions.
"""

import hashlib
from typing import Optional

import esm  # kept for ESMBasedPredictor / token-level IG internals (ESM-2 only)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_processing import CustomDataset, batch_pad_and_normalize
from .embedding import get_embeddings


def load_esm_model(model_name: str = 'esmc_300m', device: torch.device = None):
    """Load an ESM model and return ``(raw_model, alphabet)``.

    Delegates to :func:`~src.igloo.esm_adapter.load_esm_backend`.  For ESM-2
    the return is the classic ``(model, alphabet)`` pair.  For ESM-C the
    alphabet position returns the backend object (``alphabet=None`` is set
    inside the backend, so callers should not attempt to use it for
    tokenisation — token-level IG is ESM-2 only).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from .esm_adapter import load_esm_backend
    backend = load_esm_backend(model_name, device)
    # Return the raw model + alphabet (None for ESM-C) for backward compat
    return backend.model, backend.alphabet


def sequence_to_tokens(sequence: str, alphabet) -> torch.Tensor:
    """Convert sequence to ESM tokens."""
    batch_converter = alphabet.get_batch_converter()
    batch = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    return batch_tokens


def tokens_to_embeddings(
    tokens: torch.Tensor,
    esm_model: torch.nn.Module,
    layer: int = 33
) -> torch.Tensor:
    """Get embeddings from tokens using ESM model."""
    with torch.no_grad():
        results = esm_model(tokens, repr_layers=[layer], return_contacts=False)
        embeddings = results['representations'][layer]
    return embeddings


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Sample from Gumbel-Softmax distribution.

    Args:
        logits: Logits tensor of shape (..., num_classes)
        temperature: Temperature parameter (lower = more discrete)
        hard: If True, returns one-hot vectors but uses soft vectors for gradients

    Returns:
        Soft (or hard) categorical samples
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)

    if hard:
        # Straight-through estimator: hard forward, soft backward
        y_hard = F.one_hot(y_soft.argmax(dim=-1), num_classes=logits.size(-1)).float()
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y


def tokens_to_logits(tokens: torch.Tensor, vocab_size: int, device: torch.device) -> torch.Tensor:
    """
    Convert discrete tokens to logits (one-hot with high temperature).

    Args:
        tokens: Discrete token tensor
        vocab_size: Size of vocabulary
        device: PyTorch device

    Returns:
        Logits tensor that can be used with Gumbel-Softmax
    """
    # Create one-hot encoding
    one_hot = F.one_hot(tokens, num_classes=vocab_size).float()

    # Convert to logits (inverse softmax with high confidence)
    # Use a large value to make the softmax peaked at the correct token
    logits = torch.log(one_hot + 1e-8) * 10.0  # High confidence logits

    return logits


class ESMBasedPredictor(torch.nn.Module):
    """
    Wrapper that combines ESM model with your prediction model.
    This allows end-to-end gradients from tokens to predictions.
    """

    def __init__(
        self,
        esm_model: torch.nn.Module,
        predictor_model: torch.nn.Module,
        esm_layer: int = 33,
        pooling_method: str = 'linear'
    ):
        super().__init__()
        self.esm_model = esm_model
        self.predictor_model = predictor_model
        self.esm_layer = esm_layer
        self.pooling_method = pooling_method

        # Freeze ESM parameters to avoid changing them
        for param in self.esm_model.parameters():
            param.requires_grad = False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from tokens to prediction.

        Args:
            tokens: Tokenized sequences (batch_size, seq_len)

        Returns:
            Predictions (batch_size, 1)
        """
        print(f"ESMBasedPredictor: Input tokens shape: {tokens.shape}")

        # Get embeddings from ESM
        results = self.esm_model(tokens, repr_layers=[self.esm_layer], return_contacts=False)
        embeddings = results['representations'][self.esm_layer]
        print(f"ESMBasedPredictor: Raw ESM embeddings shape: {embeddings.shape}")

        # Remove start/end tokens to match training data format
        # ESM adds <cls> at start and <eos> at end
        seq_embeddings = embeddings[:, 1:-1, :]  # (batch, L, D)
        print(f"ESMBasedPredictor: After removing special tokens: {seq_embeddings.shape}")

        # Apply the same global normalization as used during training
        from .data_processing import _global_scaler_mean, _global_scaler_scale

        batch_size, seq_len, embed_dim = seq_embeddings.shape
        print(
            f"ESMBasedPredictor: Before normalization - batch_size: {batch_size}, "
            f"seq_len: {seq_len}, embed_dim: {embed_dim}"
        )
        flat_embeddings = seq_embeddings.reshape(batch_size, -1)  # (batch_size, seq_len * embed_dim)
        print(f"ESMBasedPredictor: Flattened embeddings shape: {flat_embeddings.shape}")

        # Apply global normalization (same as batch_pad_and_normalize with use_global_normalization=True)
        if _global_scaler_mean is not None and _global_scaler_scale is not None:
            print("ESMBasedPredictor: Using global normalization parameters")
            print(f"ESMBasedPredictor: Global mean: {_global_scaler_mean:.6f}, scale: {_global_scaler_scale:.6f}")
            flat_normalized = (flat_embeddings - _global_scaler_mean) / _global_scaler_scale
        else:
            print("ESMBasedPredictor: WARNING - No global normalization parameters found, "
                  "using per-sample normalization")
            # Fallback to per-sample normalization if global parameters not available
            mean = flat_embeddings.mean(dim=1, keepdim=True)
            std = flat_embeddings.std(dim=1, keepdim=True)
            flat_normalized = (flat_embeddings - mean) / (std + 1e-8)

        # Reshape back to sequence format
        normalized_embeddings = flat_normalized.reshape(batch_size, seq_len, embed_dim)
        print(f"ESMBasedPredictor: After normalization shape: {normalized_embeddings.shape}")

        # Apply prediction model
        if self.pooling_method == 'esm_attention':
            # Create input mask (all positions are valid since we removed special tokens)
            input_mask = torch.ones(batch_size, seq_len, device=normalized_embeddings.device)
            print(f"ESMBasedPredictor: Using esm_attention with mask shape: {input_mask.shape}")
            predictions = self.predictor_model(normalized_embeddings, input_mask)
        else:
            print(f"ESMBasedPredictor: Using {self.pooling_method} pooling")
            predictions = self.predictor_model(normalized_embeddings)

        print(f"ESMBasedPredictor: Final predictions shape: {predictions.shape}")
        print(f"ESMBasedPredictor: Prediction values: {predictions.squeeze()}")
        return predictions


class ESMBasedPredictorGumbel(torch.nn.Module):
    """
    Gumbel-Softmax version of ESMBasedPredictor.

    This version uses Gumbel-Softmax to make token selection differentiable,
    allowing direct gradients with respect to token choices.
    """

    def __init__(
        self,
        esm_model: torch.nn.Module,
        predictor_model: torch.nn.Module,
        esm_layer: int = 33,
        pooling_method: str = 'linear',
        temperature: float = 1.0,
        hard_gumbel: bool = False
    ):
        super().__init__()
        self.esm_model = esm_model
        self.predictor_model = predictor_model
        self.esm_layer = esm_layer
        self.pooling_method = pooling_method
        self.temperature = temperature
        self.hard_gumbel = hard_gumbel
        self.vocab_size = esm_model.embed_tokens.num_embeddings

        # Freeze ESM parameters to avoid changing them
        for param in self.esm_model.parameters():
            param.requires_grad = False

    def forward_from_logits(self, token_logits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from token logits (soft tokens) to prediction.

        Args:
            token_logits: Token logits tensor (batch_size, seq_len, vocab_size)

        Returns:
            Predictions (batch_size, 1)
        """
        print(f"ESMBasedPredictorGumbel: Input token_logits shape: {token_logits.shape}")

        # Convert logits to soft tokens using Gumbel-Softmax
        soft_tokens = gumbel_softmax_sample(
            token_logits,
            temperature=self.temperature,
            hard=self.hard_gumbel
        )  # (batch_size, seq_len, vocab_size)
        print(f"ESMBasedPredictorGumbel: Soft tokens shape: {soft_tokens.shape}")

        # Get embeddings using soft token weights
        # soft_tokens @ embedding_matrix gives weighted combination of embeddings
        embeddings = torch.matmul(soft_tokens, self.esm_model.embed_tokens.weight)
        # (batch_size, seq_len, embed_dim)

        # Apply position embeddings
        if hasattr(self.esm_model, 'embed_positions'):
            seq_len = embeddings.size(1)
            positions = torch.arange(seq_len, device=embeddings.device).unsqueeze(0).expand(embeddings.size(0), -1)
            embeddings = embeddings + self.esm_model.embed_positions(positions)

        # Apply layer norm if present
        if hasattr(self.esm_model, 'emb_layer_norm_before'):
            embeddings = self.esm_model.emb_layer_norm_before(embeddings)

        # Pass through transformer layers
        x = embeddings
        for layer in self.esm_model.layers[:self.esm_layer]:
            x, _ = layer(x, None)

        # Get final layer norm if present
        if hasattr(self.esm_model, 'emb_layer_norm_after'):
            x = self.esm_model.emb_layer_norm_after(x)

        # Remove start/end tokens to match training data format
        # ESM adds <cls> at start and <eos> at end
        seq_embeddings = x[:, 1:-1, :]  # (batch, L, D)

        # Apply the same global normalization as used during training
        from .data_processing import _global_scaler_mean, _global_scaler_scale

        batch_size, seq_len, embed_dim = seq_embeddings.shape
        flat_embeddings = seq_embeddings.reshape(batch_size, -1)

        # Apply global normalization (same as batch_pad_and_normalize with use_global_normalization=True)
        if _global_scaler_mean is not None and _global_scaler_scale is not None:
            flat_normalized = (flat_embeddings - _global_scaler_mean) / _global_scaler_scale
        else:
            # Fallback to per-sample normalization if global parameters not available
            mean = flat_embeddings.mean(dim=1, keepdim=True)
            std = flat_embeddings.std(dim=1, keepdim=True)
            flat_normalized = (flat_embeddings - mean) / (std + 1e-8)

        normalized_embeddings = flat_normalized.reshape(batch_size, seq_len, embed_dim)

        # Apply prediction model
        if self.pooling_method == 'esm_attention':
            input_mask = torch.ones(batch_size, seq_len, device=normalized_embeddings.device)
            predictions = self.predictor_model(normalized_embeddings, input_mask)
        else:
            predictions = self.predictor_model(normalized_embeddings)

        return predictions

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from discrete tokens to prediction.
        Converts tokens to logits first, then uses forward_from_logits.
        """
        # Convert discrete tokens to logits
        token_logits = tokens_to_logits(tokens, self.vocab_size, tokens.device)
        return self.forward_from_logits(token_logits)


def integrated_gradients_gumbel(
    combined_model: ESMBasedPredictorGumbel,
    input_tokens: torch.Tensor,
    baseline_tokens: torch.Tensor,
    alphabet,
    steps: int = 50,
    batch_size: int = 10
) -> torch.Tensor:
    """
    Compute integrated gradients using Gumbel-Softmax approach.

    This version directly computes gradients with respect to token logits,
    making the process more straightforward than the embedding interpolation approach.

    Args:
        combined_model: Gumbel-Softmax ESM + predictor combined model
        input_tokens: Input sequence tokens (1, seq_len)
        baseline_tokens: Baseline sequence tokens (1, seq_len)
        alphabet: ESM alphabet
        steps: Number of integration steps
        batch_size: Batch size for gradient computation

    Returns:
        Token-level gradients (1, seq_len)
    """
    combined_model.eval()
    device = next(combined_model.parameters()).device

    input_tokens = input_tokens.to(device)
    baseline_tokens = baseline_tokens.to(device)

    # Convert tokens to logits
    input_logits = tokens_to_logits(input_tokens, combined_model.vocab_size, device)
    baseline_logits = tokens_to_logits(baseline_tokens, combined_model.vocab_size, device)

    # Generate interpolated logits
    alphas = torch.linspace(0, 1, steps + 1, device=device)
    all_gradients = []

    for i in range(0, len(alphas), batch_size):
        batch_alphas = alphas[i:i + batch_size]
        interpolated_logits = []

        for alpha in batch_alphas:
            interpolated = baseline_logits + alpha * (input_logits - baseline_logits)
            interpolated_logits.append(interpolated)

        # Stack batch
        batch_logits = torch.cat(interpolated_logits, dim=0)  # (batch_size, seq_len, vocab_size)
        batch_logits.requires_grad_(True)

        # Forward pass
        outputs = combined_model.forward_from_logits(batch_logits)

        # Compute gradients with respect to logits
        logit_gradients = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=batch_logits,
            create_graph=False,
            retain_graph=False
        )[0]  # (batch_size, seq_len, vocab_size)

        all_gradients.append(logit_gradients.detach())

    # Combine all gradients
    all_gradients = torch.cat(all_gradients, dim=0)  # (total_steps, seq_len, vocab_size)

    # Average gradients across integration steps
    avg_gradients = torch.mean(all_gradients, dim=0, keepdim=True)  # (1, seq_len, vocab_size)

    # Convert logit gradients to position-level gradients
    # For each position, compute the gradient magnitude for the actual token at that position
    position_gradients = []

    for pos in range(avg_gradients.size(1)):
        # Get the actual token at this position
        actual_token_id = input_tokens[0, pos].item()

        # Get the gradient for this specific token
        token_gradient = avg_gradients[0, pos, actual_token_id].item()
        position_gradients.append(token_gradient)

    return torch.tensor(position_gradients).unsqueeze(0)  # (1, seq_len)


def integrated_gradients_tokens(
    combined_model: ESMBasedPredictor,
    input_tokens: torch.Tensor,
    baseline_tokens: torch.Tensor,
    alphabet,
    steps: int = 50,
    batch_size: int = 10
) -> torch.Tensor:
    """
    Compute integrated gradients with respect to input tokens.

    Since tokens are discrete, we'll compute gradients with respect to
    the token embeddings and then project back to token space.

    Args:
        combined_model: ESM + predictor combined model
        input_tokens: Input sequence tokens (1, seq_len)
        baseline_tokens: Baseline sequence tokens (1, seq_len)
        alphabet: ESM alphabet
        steps: Number of integration steps
        batch_size: Batch size for gradient computation

    Returns:
        Token-level gradients (1, seq_len)
    """
    combined_model.eval()
    device = next(combined_model.parameters()).device

    input_tokens = input_tokens.to(device)
    baseline_tokens = baseline_tokens.to(device)

    # Get token embeddings from ESM embedding matrix
    token_embedding_matrix = combined_model.esm_model.embed_tokens.weight  # (vocab_size, embed_dim)

    input_embeddings = token_embedding_matrix[input_tokens]  # (1, seq_len, embed_dim)
    baseline_embeddings = token_embedding_matrix[baseline_tokens]  # (1, seq_len, embed_dim)

    # Generate interpolated embeddings
    alphas = torch.linspace(0, 1, steps + 1, device=device)
    all_gradients = []

    for i in range(0, len(alphas), batch_size):
        batch_alphas = alphas[i:i + batch_size]
        interpolated_embeddings = []

        for alpha in batch_alphas:
            interpolated = baseline_embeddings + alpha * (input_embeddings - baseline_embeddings)
            interpolated_embeddings.append(interpolated)

        # Stack batch
        batch_embeddings = torch.cat(interpolated_embeddings, dim=0)  # (batch_size, seq_len, embed_dim)
        batch_embeddings.requires_grad_(True)

        # Forward pass through ESM layers (skip embedding layer)
        # We need to manually pass through ESM layers starting from layer 0
        x = batch_embeddings

        # Apply position embeddings if ESM uses them
        if hasattr(combined_model.esm_model, 'embed_positions'):
            positions = torch.arange(x.size(1), device=device).unsqueeze(0).expand(x.size(0), -1)
            x = x + combined_model.esm_model.embed_positions(positions)

        # Apply layer norm if present
        if hasattr(combined_model.esm_model, 'emb_layer_norm_before'):
            x = combined_model.esm_model.emb_layer_norm_before(x)

        # Pass through transformer layers
        for layer in combined_model.esm_model.layers:
            x, _ = layer(x, None)  # No attention mask for simplicity

        # Get the representation from the desired layer
        if hasattr(combined_model.esm_model, 'emb_layer_norm_after'):
            x = combined_model.esm_model.emb_layer_norm_after(x)

        # Remove start/end tokens to match training format
        seq_embeddings = x[:, 1:-1, :]

        # Apply the same global normalization as used during training
        from .data_processing import _global_scaler_mean, _global_scaler_scale

        batch_size_actual, seq_len, embed_dim = seq_embeddings.shape
        flat_embeddings = seq_embeddings.reshape(batch_size_actual, -1)

        # Apply global normalization (same as batch_pad_and_normalize with use_global_normalization=True)
        if _global_scaler_mean is not None and _global_scaler_scale is not None:
            flat_normalized = (flat_embeddings - _global_scaler_mean) / _global_scaler_scale
        else:
            # Fallback to per-sample normalization if global parameters not available
            mean = flat_embeddings.mean(dim=1, keepdim=True)
            std = flat_embeddings.std(dim=1, keepdim=True)
            flat_normalized = (flat_embeddings - mean) / (std + 1e-8)

        normalized_embeddings = flat_normalized.reshape(batch_size_actual, seq_len, embed_dim)

        # Apply prediction model
        if combined_model.pooling_method == 'esm_attention':
            input_mask = torch.ones(batch_size_actual, seq_len, device=device)
            outputs = combined_model.predictor_model(normalized_embeddings, input_mask)
        else:
            outputs = combined_model.predictor_model(normalized_embeddings)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=batch_embeddings,
            create_graph=False,
            retain_graph=False
        )[0]

        all_gradients.append(gradients.detach())

    # Combine all gradients
    all_gradients = torch.cat(all_gradients, dim=0)

    # Average gradients across integration steps
    avg_gradients = torch.mean(all_gradients, dim=0, keepdim=True)  # (1, seq_len, embed_dim)

    # Project embedding gradients back to token space
    # For each position, find which token direction has highest gradient magnitude
    token_gradients = []

    for pos in range(avg_gradients.size(1)):
        pos_gradient = avg_gradients[0, pos, :]  # (embed_dim,)

        # Compute similarity with all token embeddings
        similarities = torch.matmul(token_embedding_matrix, pos_gradient)  # (vocab_size,)

        # The token with highest similarity represents the gradient direction
        token_gradients.append(similarities.max().item())

    return torch.tensor(token_gradients).unsqueeze(0)  # (1, seq_len)


def create_combined_model(
    esm_model: torch.nn.Module,
    predictor_model: torch.nn.Module,
    esm_layer: int = 33,
    pooling_method: str = 'linear',
    method: str = 'embedding',
    gumbel_temperature: float = 1.0,
    hard_gumbel: bool = False,
    device: torch.device = None
):
    """
    Create the appropriate combined model based on the integration method.

    Args:
        esm_model: Pre-trained ESM model
        predictor_model: Your trained prediction model
        esm_layer: ESM layer to use
        pooling_method: Pooling method ('linear', 'mean', 'attention', etc.)
        method: Integration method ('embedding' or 'gumbel')
        gumbel_temperature: Temperature for Gumbel-Softmax (only for gumbel method)
        hard_gumbel: Whether to use hard Gumbel-Softmax (only for gumbel method)
        device: PyTorch device

    Returns:
        Combined model (ESMBasedPredictor or ESMBasedPredictorGumbel)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if method == 'gumbel':
        model = ESMBasedPredictorGumbel(
            esm_model=esm_model,
            predictor_model=predictor_model,
            esm_layer=esm_layer,
            pooling_method=pooling_method,
            temperature=gumbel_temperature,
            hard_gumbel=hard_gumbel
        ).to(device)
    else:  # method == 'embedding'
        model = ESMBasedPredictor(
            esm_model=esm_model,
            predictor_model=predictor_model,
            esm_layer=esm_layer,
            pooling_method=pooling_method
        ).to(device)

    return model


def compute_amino_acid_attributions(
    combined_model: ESMBasedPredictor,
    sequence: str,
    baseline_sequence: str,
    alphabet,
    device: torch.device
) -> pd.DataFrame:
    """
    Compute attribution scores for each amino acid at each position.

    This function systematically replaces each amino acid with every other
    amino acid and measures the effect on the prediction.

    Args:
        combined_model: Combined ESM + predictor model
        sequence: Input protein sequence
        baseline_sequence: Baseline sequence (usually WT)
        alphabet: ESM alphabet
        device: PyTorch device

    Returns:
        DataFrame with attribution scores for each position and amino acid
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    combined_model.eval()

    # Get baseline prediction
    baseline_tokens = sequence_to_tokens(baseline_sequence, alphabet).to(device)
    with torch.no_grad():
        baseline_pred = combined_model(baseline_tokens).item()

    attributions = []

    print(f"Computing amino acid attributions for {len(sequence)} positions...")

    for pos in tqdm(range(len(sequence)), desc="Position"):
        current_aa = sequence[pos]

        for target_aa in amino_acids:
            # Create mutant sequence
            mutant_seq = sequence[:pos] + target_aa + sequence[pos+1:]

            # Get prediction for mutant
            mutant_tokens = sequence_to_tokens(mutant_seq, alphabet).to(device)
            with torch.no_grad():
                mutant_pred = combined_model(mutant_tokens).item()

            # Calculate attribution
            attribution = mutant_pred - baseline_pred

            attributions.append({
                'position': pos + 1,
                'current_aa': current_aa,
                'target_aa': target_aa,
                'mutation': f"{current_aa}{pos+1}{target_aa}" if target_aa != current_aa else "WT",
                'predicted_activity': mutant_pred,
                'baseline_activity': baseline_pred,
                'attribution': attribution,
                'is_mutation': target_aa != current_aa
            })

    return pd.DataFrame(attributions)


def plot_amino_acid_heatmap(
    attributions_df: pd.DataFrame,
    sequence: str,
    title: str = "Amino Acid Attribution Heatmap",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (20, 8)
) -> None:
    """
    Create a heatmap showing attribution scores for all amino acids at all positions.

    Args:
        attributions_df: DataFrame from compute_amino_acid_attributions
        sequence: Original protein sequence
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Filter to only mutations (not WT)
    mutations_df = attributions_df[attributions_df['is_mutation']].copy()

    # Create pivot table for heatmap
    heatmap_data = mutations_df.pivot(
        index='target_aa',
        columns='position',
        values='attribution'
    )

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[4, 1])

    # Main heatmap
    sns.heatmap(
        heatmap_data,
        ax=ax1,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Attribution Score vs Baseline'},
        xticklabels=True,
        yticklabels=True
    )
    ax1.set_title(title)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Target Amino Acid')

    # Add WT sequence as reference
    for i, aa in enumerate(sequence):
        ax2.text(i + 0.5, 0.5, aa, ha='center', va='center',
                fontsize=8, fontweight='bold')
        ax2.add_patch(plt.Rectangle((i, 0), 1, 1, fill=False, edgecolor='black'))

    ax2.set_xlim(0, len(sequence))
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('WT')
    ax2.set_yticks([])
    ax2.set_title('Wild Type Sequence')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Amino acid heatmap saved to {save_path}")

    plt.show()


def analyze_position_sensitivity(
    attributions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze which positions are most sensitive to mutations.

    Args:
        attributions_df: DataFrame from compute_amino_acid_attributions

    Returns:
        DataFrame with position-level sensitivity metrics
    """
    # Filter to only mutations
    mutations_df = attributions_df[attributions_df['is_mutation']].copy()

    position_stats = mutations_df.groupby('position').agg({
        'attribution': ['mean', 'std', 'min', 'max', 'count'],
        'predicted_activity': ['mean', 'std']
    }).round(4)

    # Flatten column names
    position_stats.columns = ['_'.join(col).strip() for col in position_stats.columns]

    # Add derived metrics
    position_stats['sensitivity'] = position_stats['attribution_std']  # Standard deviation as sensitivity measure
    position_stats['max_improvement'] = position_stats['attribution_max']
    position_stats['max_detriment'] = position_stats['attribution_min']
    position_stats['beneficial_mutations'] = mutations_df.groupby(
        'position')['attribution'].apply(lambda x: (x > 0).sum())
    position_stats['detrimental_mutations'] = mutations_df.groupby(
        'position')['attribution'].apply(lambda x: (x < 0).sum())

    # Sort by sensitivity (most variable positions first)
    position_stats = position_stats.sort_values('sensitivity', ascending=False)

    return position_stats


def export_sequence_analysis_results(
    attributions_df: pd.DataFrame,
    position_stats: pd.DataFrame,
    output_dir: str,
    sequence: str,
    baseline_sequence: str
) -> None:
    """
    Export comprehensive sequence analysis results.

    Args:
        attributions_df: DataFrame from compute_amino_acid_attributions
        position_stats: DataFrame from analyze_position_sensitivity
        output_dir: Directory to save results
        sequence: Input sequence
        baseline_sequence: Baseline sequence
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. Full attribution matrix
    attributions_df.to_csv(
        os.path.join(output_dir, 'full_amino_acid_attributions.csv'),
        index=False
    )

    # 2. Position sensitivity analysis
    position_stats.to_csv(
        os.path.join(output_dir, 'position_sensitivity_analysis.csv')
    )

    # 3. Top beneficial mutations
    beneficial = attributions_df[
        (attributions_df['is_mutation']) & (attributions_df['attribution'] > 0)
    ].sort_values('attribution', ascending=False)

    beneficial.head(50).to_csv(
        os.path.join(output_dir, 'top_beneficial_mutations.csv'),
        index=False
    )

    # 4. Most detrimental mutations
    detrimental = attributions_df[
        (attributions_df['is_mutation']) & (attributions_df['attribution'] < 0)
    ].sort_values('attribution', ascending=True)

    detrimental.head(50).to_csv(
        os.path.join(output_dir, 'most_detrimental_mutations.csv'),
        index=False
    )

    # 5. Summary report
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("SEQUENCE-LEVEL ATTRIBUTION ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input sequence length: {len(sequence)}\n")
        f.write(f"Baseline sequence length: {len(baseline_sequence)}\n")
        f.write(f"Total mutations analyzed: {len(attributions_df[attributions_df['is_mutation']])}\n")
        f.write(f"Beneficial mutations: {len(beneficial)}\n")
        f.write(f"Detrimental mutations: {len(detrimental)}\n")
        f.write(f"Neutral mutations: {len(attributions_df) - len(beneficial) - len(detrimental) - len(sequence)}\n\n")

        f.write("TOP 10 MOST SENSITIVE POSITIONS:\n")
        for i, (pos, row) in enumerate(position_stats.head(10).iterrows()):
            f.write(f"{i+1:2d}. Position {pos}: sensitivity={row['sensitivity']:.4f}\n")

        f.write("\nTOP 10 BENEFICIAL MUTATIONS:\n")
        for i, (_, row) in enumerate(beneficial.head(10).iterrows()):
            f.write(f"{i+1:2d}. {row['mutation']}: {row['attribution']:.4f}\n")

    print(f"Analysis results exported to {output_dir}")


# ---------------------------------------------------------------------------
# Interpolation path helpers
# ---------------------------------------------------------------------------

def interpolate_embeddings(
    baseline: torch.Tensor,
    sequence: torch.Tensor,
    alpha: float,
    method: str = 'linear',
) -> torch.Tensor:
    """
    Interpolate between baseline and sequence embeddings at a given alpha.

    Parameters
    ----------
    baseline : Tensor of shape (..., embed_dim)
        Starting point; alpha=0 returns baseline, alpha=1 returns sequence.
    sequence : Tensor of the same shape as baseline.
    alpha    : Scalar in [0, 1].
    method   : Interpolation strategy:
        'linear'           — straight-line path (standard IG).
        'slerp'            — spherical linear interpolation applied
                             independently to each embedding vector
                             (preserves L2 norm along the path); falls
                             back to linear when vectors are nearly
                             parallel (omega ≈ 0).
        'nearest_neighbor' — discrete staircase path: positions switch
                             from baseline to sequence one at a time as
                             alpha increases (ordered by position index).
                             Position i activates when alpha >= i / L,
                             where L = number of positions. Produces a
                             piecewise-constant path through embedding
                             space; useful for probing position-order
                             sensitivity of attributions.

    Returns
    -------
    Tensor of same shape as inputs.
    """
    if method == 'linear':
        return baseline + alpha * (sequence - baseline)

    elif method == 'slerp':
        eps = 1e-8
        v0 = baseline.float()
        v1 = sequence.float()

        v0_norm = torch.nn.functional.normalize(v0, dim=-1)
        v1_norm = torch.nn.functional.normalize(v1, dim=-1)

        # Angle between the two unit vectors at each position
        dot = torch.clamp((v0_norm * v1_norm).sum(dim=-1, keepdim=True), -1.0, 1.0)
        omega = torch.acos(dot)         # (..., 1)
        sin_omega = torch.sin(omega)    # (..., 1)

        slerp = (
            (torch.sin((1.0 - alpha) * omega) / (sin_omega + eps)) * v0
            + (torch.sin(alpha * omega) / (sin_omega + eps)) * v1
        )
        linear = v0 + alpha * (v1 - v0)

        # Fall back to linear wherever sin_omega is too small
        return torch.where(sin_omega > eps, slerp, linear)

    elif method == 'nearest_neighbor':
        # Discrete staircase: position i switches from baseline to sequence
        # embedding when alpha >= i / seq_len. For alpha=0 every position
        # uses baseline; for alpha=1 every position uses sequence (since
        # i/L < 1 for all i in [0, L-1]).
        seq_len = baseline.shape[-2]
        thresholds = torch.arange(seq_len, dtype=torch.float32, device=baseline.device) / seq_len
        switched = (thresholds <= alpha).to(baseline.dtype)  # (seq_len,)

        # Reshape for broadcasting against (..., seq_len, embed_dim):
        #   2-D input (seq_len, embed_dim)      → switched: (seq_len, 1)
        #   3-D input (batch, seq_len, embed_dim) → switched: (1, seq_len, 1)
        shape = [1] * (baseline.dim() - 2) + [seq_len, 1]
        switched = switched.view(*shape)
        return baseline + switched * (sequence - baseline)

    else:
        raise ValueError(
            f"Unknown interpolation method '{method}'. "
            f"Choose from: 'linear', 'slerp', 'nearest_neighbor'. "
            f"(note: 'guided_ig' is pre-built by build_guided_ig_path() and does not use this function)"
        )


# ── Guided IG helpers ──────────────────────────────────────────────────────

def _guided_ig_model_forward(
    emb: "torch.Tensor",
    predictor_model,
    pooling_method: str,
    y_log_mean: float,
    y_log_std: float,
) -> "torch.Tensor":
    """Normalise *emb* exactly as the training pipeline does, then run the predictor.

    Returns the rescaled scalar output (or batch of outputs).
    """
    from .data_processing import _global_scaler_mean, _global_scaler_scale  # noqa: PLC0415
    device = emb.device
    b, L, D = emb.shape
    flat = emb.reshape(b, -1)
    if _global_scaler_mean is not None and _global_scaler_scale is not None:
        flat_norm = (flat - _global_scaler_mean) / _global_scaler_scale
    else:
        mu = flat.mean(dim=1, keepdim=True)
        sigma = flat.std(dim=1, keepdim=True)
        flat_norm = (flat - mu) / (sigma + 1e-8)
    norm_emb = flat_norm.reshape(b, L, D)
    if pooling_method == 'esm_attention':
        mask = torch.ones(b, L, device=device)
        out = predictor_model(norm_emb, mask)
    else:
        out = predictor_model(norm_emb)
    return out * y_log_std + y_log_mean


def build_guided_ig_path(
    baseline_emb: "torch.Tensor",
    seq_emb: "torch.Tensor",
    predictor_model,
    pooling_method: str,
    y_log_mean: float,
    y_log_std: float,
    device: "torch.device",
    steps: int,
) -> list:
    """Build a Guided IG interpolation path (Kapishnikov et al., 2021).

    Constructs a position-level gradient-reordered staircase path:

    1. Compute the gradient \u2207F at the linear midpoint between baseline and sequence.
    2. For each sequence position *p*, compute relevance::

           r_p = \u2016\u2207F_p\u2016\u2082  \u00d7  \u2016e\u2081_p \u2212 e\u2080_p\u2016\u2082

    3. Sort positions in *ascending* relevance order so that low-importance
       positions activate first (small \u03b1) and high-importance positions activate
       last (\u03b1 \u2192 1).  This concentrates attribution mass near the sequence
       endpoint where the model is most sensitive.

    The result is a staircase identical in structure to *nearest_neighbor* but with
    steps occurring in gradient-priority order rather than positional order.

    Parameters
    ----------
    baseline_emb, seq_emb:
        Tensors of shape ``(1, L, D)`` (a single sequence).
    steps:
        Number of integration steps (same as the IG ``steps`` argument).

    Returns
    -------
    list of ``(1, L, D)`` tensors, one per \u03b1 value in
    ``torch.linspace(0, 1, steps + 1)``.
    """
    predictor_model.eval()
    baseline_emb = baseline_emb.to(device).float()
    seq_emb = seq_emb.to(device).float()
    L = baseline_emb.shape[-2]

    # ── Gradient at the linear midpoint ────────────────────────────────────
    midpoint = (baseline_emb + seq_emb) / 2.0
    x_mid = midpoint.clone().requires_grad_(True)
    out_mid = _guided_ig_model_forward(x_mid, predictor_model, pooling_method, y_log_mean, y_log_std)
    grad = torch.autograd.grad(out_mid.sum(), x_mid)[0].detach()  # (1, L, D)

    # ── Per-position relevance: ||grad_p||\u2082 \u00d7 ||displacement_p||\u2082 ─────────┤
    displacement = seq_emb - baseline_emb  # (1, L, D)
    relevance = (
        grad.norm(dim=-1) * displacement.norm(dim=-1)
    ).squeeze(0)  # (L,)

    # Ascending sort: low relevance \u2192 activates early (small \u03b1)
    sorted_positions = torch.argsort(relevance)  # (L,)

    # ── Build staircase path ────────────────────────────────────────────────
    alphas = torch.linspace(0, 1, steps + 1)
    path: list = []
    for alpha in alphas:
        n_activated = min(int(alpha.item() * L), L)
        x_step = baseline_emb.clone()
        if n_activated > 0:
            active_pos = sorted_positions[:n_activated]  # (n_activated,)
            x_step[0, active_pos, :] = seq_emb[0, active_pos, :]
        path.append(x_step.detach())
    return path


# ── End Guided IG helpers ───────────────────────────────────────────────────


def analyze_sequence_with_integrated_gradients(
    sequence: str,
    baseline_sequence: str,
    model,  # Can be either ESMBasedPredictor or ESMBasedPredictorGumbel
    alphabet,
    device: torch.device,
    y_log_mean: float = 0.0,
    y_log_std: float = 1.0,
    steps: int = 50,
    batch_size: int = 10,
    method: str = 'embedding',  # 'embedding' or 'gumbel'
    attribution_method: str = 'signed',  # 'magnitude' (L2 norm) or 'signed' (dot product)
    actual_activity: float | None = None,
    baseline_actual_activity: float | None = None,
    plot_predictions: bool = False,
    sequence_name: str = "Sequence",
    baseline_name: str = "Baseline",
    save_plot_path: str | None = None,
    clip_min: float | None = None,
    clip_max: float | None = None,
    embeddings_subdir: str | None = None,
    interpolation_path: str = 'linear',
    plot_interpolation_umap: bool = False,
    interpolation_umap_save_path: str | None = None,
    _esm_backend=None,
) -> dict:
    """
    Analyze a sequence using integrated gradients.

    Args:
        sequence: Input protein sequence
        baseline_sequence: Baseline sequence (usually wild-type)
        model: Combined ESM + predictor model (ESMBasedPredictor or ESMBasedPredictorGumbel)
        _esm_backend: Optional pre-loaded ESM backend
            (:class:`~src.igloo.esm_adapter.ESM2Backend` or
            :class:`~src.igloo.esm_adapter.ESMCBackend`) to pass to
            :func:`~src.igloo.embedding.get_embeddings` so the ESM model is
            not loaded a second time for cache-miss sequences.
        alphabet: ESM alphabet
        device: PyTorch device
        y_log_mean: Mean for denormalization (default 0.0 for no denormalization)
        y_log_std: Std for denormalization (default 1.0 for no denormalization)
        steps: Number of integration steps
        batch_size: Batch size for gradient computation
        method: Integration method ('embedding' or 'gumbel')
        attribution_method: Attribution computation method ('magnitude' for L2 norm, 'signed' for dot product)
        actual_activity: Optional actual/experimental activity for the sequence
        baseline_actual_activity: Optional actual activity for the baseline
        plot_predictions: Whether to create predicted vs actual plot
        sequence_name: Name for the sequence in plots
        baseline_name: Name for the baseline in plots
        save_plot_path: Optional path to save the prediction plot
        clip_min: Optional minimum value to clip predictions (e.g., min of training data)
        clip_max: Optional maximum value to clip predictions (e.g., max of training data)

    Returns:
        dict with keys: sequence, baseline_sequence, predicted_activity,
                       baseline_activity, effect_vs_baseline, position_gradients,
                       positions, amino_acids
    """

    # Get ESM embeddings for sequences (same as training pipeline)
    print("Getting ESM embeddings for sequence and baseline...")
    esm_layer = model.esm_layer if hasattr(model, 'esm_layer') else 33
    # Use unique dataset names to avoid caching conflicts between different sequences
    seq_hash = hashlib.md5(sequence.encode()).hexdigest()[:8]
    baseline_hash = hashlib.md5(baseline_sequence.encode()).hexdigest()[:8]

    seq_embedding = get_embeddings(
        [sequence], layer=esm_layer, batch_size=1, dataset_name=f'sequence_ig_seq_{seq_hash}',
        subdir=embeddings_subdir,
        esm_backend=_esm_backend,
    )
    baseline_embedding = get_embeddings(
        [baseline_sequence], layer=esm_layer, batch_size=1, dataset_name=f'sequence_ig_baseline_{baseline_hash}',
        subdir=embeddings_subdir,
        esm_backend=_esm_backend,
    )

    # Get predictions using DataLoader approach (exactly like training/evaluation)
    print("Computing predictions using DataLoader method...")

    # Extract the predictor model and pooling method from combined model
    predictor_model = model.predictor_model if hasattr(model, 'predictor_model') else model
    pooling_method = model.pooling_method if hasattr(model, 'pooling_method') else 'linear'

    predictor_model.eval()

    # Use the same global normalization parameters that were used during training
    # This ensures the model sees inputs in the same distribution as during training

    # Sequence prediction using DataLoader with global normalization
    dummy_target = np.zeros(1)
    seq_dataset = CustomDataset(seq_embedding, dummy_target)
    seq_loader = DataLoader(seq_dataset, batch_size=1, shuffle=False, collate_fn=batch_pad_and_normalize)
    with torch.no_grad():
        for xb, lengths, yb in seq_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if pooling_method == 'esm_attention':
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                seq_pred_norm = predictor_model(xb, input_mask).squeeze(-1).cpu().numpy()[0]
            else:
                seq_pred_norm = predictor_model(xb).squeeze(-1).cpu().numpy()[0]
            break

    # Baseline prediction using DataLoader with global normalization
    baseline_dataset = CustomDataset(baseline_embedding, dummy_target)
    baseline_loader = DataLoader(
        baseline_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=batch_pad_and_normalize
    )

    with torch.no_grad():
        for xb, lengths, yb in baseline_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if pooling_method == 'esm_attention':
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                baseline_pred_norm = predictor_model(xb, input_mask).squeeze(-1).cpu().numpy()[0]
            else:
                baseline_pred_norm = predictor_model(xb).squeeze(-1).cpu().numpy()[0]
            break

    # Convert to original scale (same as training code)
    seq_pred = seq_pred_norm * y_log_std + y_log_mean
    baseline_pred = baseline_pred_norm * y_log_std + y_log_mean

    # Apply clipping if specified
    if clip_min is not None or clip_max is not None:
        print(f"Applying prediction clipping: min={clip_min}, max={clip_max}")
        if clip_min is not None and clip_max is not None:
            seq_pred = np.clip(seq_pred, clip_min, clip_max)
            baseline_pred = np.clip(baseline_pred, clip_min, clip_max)
        elif clip_min is not None:
            seq_pred = np.maximum(seq_pred, clip_min)
            baseline_pred = np.maximum(baseline_pred, clip_min)
        elif clip_max is not None:
            seq_pred = np.minimum(seq_pred, clip_max)
            baseline_pred = np.minimum(baseline_pred, clip_max)

    print(f"Sequence prediction (DataLoader method): {seq_pred:.4f}")
    print(f"Baseline prediction (DataLoader method): {baseline_pred:.4f}")

    # For integrated gradients, we need to create a wrapper that uses the same DataLoader approach
    # but allows gradients to flow through. We'll create interpolated embeddings and use the
    # predictor model directly (not the combined model)
    print(f"Computing integrated gradients using embedding interpolation (steps={steps})...")

    # Get baseline embeddings for interpolation
    device = next(predictor_model.parameters()).device

    # Generate interpolated embeddings between baseline and sequence
    alphas = torch.linspace(0, 1, steps + 1, device=device)
    all_gradients = []
    umap_trajectory: list = []
    umap_alphas: list = []
    path_predictions: list = []  # model output at each interpolation step

    # Convert to torch tensors for interpolation
    seq_emb_torch = torch.from_numpy(seq_embedding).to(device).float()
    baseline_emb_torch = torch.from_numpy(baseline_embedding).to(device).float()

    # Quick sanity-check: print mean embedding norm at 3 alpha checkpoints.
    # slerp  → norms stay constant  (all three values ≈ equal)
    # linear → norm dips at midpoint (middle value < endpoints)
    # guided_ig → path is built later; skip this check
    if interpolation_path == 'guided_ig':
        print(f"  Interpolation path check [guided_ig] — path built adaptively from gradients")
    else:
        with torch.no_grad():
            _ref_alphas = [0.0, 0.5, 1.0]
            _ref_norms = [
                interpolate_embeddings(baseline_emb_torch, seq_emb_torch, a, method=interpolation_path)
                .squeeze(0).norm(dim=-1).mean().item()
                for a in _ref_alphas
            ]
        print(
            f"  Interpolation path check [{interpolation_path}]  "
            f"norm(α=0)={_ref_norms[0]:.4f}  "
            f"norm(α=0.5)={_ref_norms[1]:.4f}  "
            f"norm(α=1)={_ref_norms[2]:.4f}"
            + ("  ✓ norms stable (slerp working)" if abs(_ref_norms[1] - _ref_norms[0]) < 0.01 * _ref_norms[0]
               else "  (midpoint dip expected for linear/nn)")
        )

    # ── Guided IG: pre-build the complete path before integration ──────────
    _guided_path: list = []
    if interpolation_path == 'guided_ig':
        print("  Building guided IG path (gradient-relevance-ordered staircase)...")
        _guided_path = build_guided_ig_path(
            baseline_emb_torch, seq_emb_torch,
            predictor_model, pooling_method,
            y_log_mean, y_log_std, device, steps,
        )
        print(f"  Guided IG path built: {len(_guided_path)} steps.")

    _global_step = 0  # tracks position in _guided_path across batches
    for i in range(0, len(alphas), batch_size):
        batch_alphas = alphas[i:i + batch_size]
        interpolated_embeddings = []

        for alpha in batch_alphas:
            if interpolation_path == 'guided_ig':
                interpolated = _guided_path[_global_step]
            else:
                interpolated = interpolate_embeddings(
                    baseline_emb_torch, seq_emb_torch, alpha.item(), method=interpolation_path
                )
            interpolated_embeddings.append(interpolated)
            if plot_interpolation_umap:
                pooled = interpolated.squeeze(0).mean(dim=0).detach().cpu().numpy()
                umap_trajectory.append(pooled)
                umap_alphas.append(alpha.item())
            _global_step += 1

        # Stack batch - but DON'T apply DataLoader normalization again
        # The embeddings are already in the right format from ESM
        batch_embeddings = torch.cat(interpolated_embeddings, dim=0)  # (batch_size, seq_len, embed_dim)
        batch_embeddings.requires_grad_(True)  # Enable gradients for embeddings

        # Apply the same normalization as used during training (same as batch_pad_and_normalize)
        from .data_processing import _global_scaler_mean, _global_scaler_scale

        batch_size_actual, seq_len, embed_dim = batch_embeddings.shape
        flat_embeddings = batch_embeddings.reshape(batch_size_actual, -1)

        if _global_scaler_mean is not None and _global_scaler_scale is not None:
            flat_normalized = (flat_embeddings - _global_scaler_mean) / _global_scaler_scale
        else:
            # Fallback to per-sample normalization if global parameters not available
            mean = flat_embeddings.mean(dim=1, keepdim=True)
            std = flat_embeddings.std(dim=1, keepdim=True)
            flat_normalized = (flat_embeddings - mean) / (std + 1e-8)

        normalized_embeddings = flat_normalized.reshape(batch_size_actual, seq_len, embed_dim)

        if pooling_method == 'esm_attention':
            # Create attention mask that matches the DataLoader approach
            # For consistency with DataLoader method, we need to account for sequence lengths
            # Since we're working with single sequences from ESM embeddings (no padding),
            # all positions should be valid, but we need to match the exact mask format
            input_mask = torch.ones(batch_size_actual, seq_len, device=device)
            outputs = predictor_model(normalized_embeddings, input_mask)
        else:
            outputs = predictor_model(normalized_embeddings)

        # Convert outputs to original scale for gradient computation
        # Note: We do NOT apply clipping here to preserve mathematical correctness of IG
        outputs_rescaled = outputs * y_log_std + y_log_mean

        if plot_interpolation_umap:
            path_predictions.extend(outputs_rescaled.detach().cpu().numpy().flatten().tolist())

        # Compute gradients with respect to the normalized embeddings
        # This is mathematically correct because the model sees normalized embeddings
        gradients = torch.autograd.grad(
            outputs=outputs_rescaled.sum(),
            inputs=normalized_embeddings,  # Gradients w.r.t. normalized embeddings
            create_graph=False,
            retain_graph=False
        )[0]

        # Transform gradients back to original embedding space for attribution calculation
        # Since normalized = (original - mean) / scale, we need to divide by scale
        if _global_scaler_mean is not None and _global_scaler_scale is not None:
            # For global normalization: grad_original = grad_normalized / scale
            gradients_original_space = gradients / _global_scaler_scale
        else:
            # For per-sample normalization, the transformation is more complex
            # We'll use the same std that was computed during normalization
            flat_gradients = gradients.reshape(batch_size_actual, -1)
            flat_embeddings = batch_embeddings.reshape(batch_size_actual, -1)
            std = flat_embeddings.std(dim=1, keepdim=True)
            gradients_original_space = (flat_gradients / (std + 1e-8)).reshape(batch_size_actual, seq_len, embed_dim)

        all_gradients.append(gradients_original_space.detach())

    # Combine all gradients and average across integration steps
    all_gradients = torch.cat(all_gradients, dim=0)  # (total_steps, seq_len, embed_dim)
    avg_gradients = torch.mean(all_gradients, dim=0)  # (seq_len, embed_dim)

    # Optionally visualise the interpolation path in UMAP-projected embedding space
    if plot_interpolation_umap and umap_trajectory:
        # ── Norm diagnostics ──────────────────────────────────────────────
        # slerp  → norms stay constant throughout (CV ≈ 0, no midpoint dip)
        # linear → norms decrease toward the midpoint then recover (CV > 0)
        # nearest_neighbor → norms jump in discrete steps
        _norms = np.linalg.norm(np.stack(umap_trajectory), axis=1)
        _mid = len(_norms) // 2
        _cv = _norms.std() / _norms.mean() if _norms.mean() > 0 else float('nan')
        print(
            f"[interpolation diagnostics | method={interpolation_path}]\n"
            f"  mean-pooled embedding norm:  "
            f"α=0 → {_norms[0]:.4f}   "
            f"α≈0.5 → {_norms[_mid]:.4f}   "
            f"α=1 → {_norms[-1]:.4f}\n"
            f"  norm CV across path: {_cv:.5f}  "
            f"(slerp → ~0; linear → >0 with midpoint dip; nn → step changes)"
        )
        # ─────────────────────────────────────────────────────────────────
        from .plot import plot_interpolation_path
        print(f"Plotting interpolation path figure ({len(umap_trajectory)} steps)...")
        # Mean-pool the endpoint embeddings for the chord-plane projection
        _baseline_pooled = baseline_emb_torch.squeeze(0).mean(dim=0).detach().cpu().numpy()
        _seq_pooled = seq_emb_torch.squeeze(0).mean(dim=0).detach().cpu().numpy()
        plot_interpolation_path(
            emb_trajectory=umap_trajectory,
            alphas=np.array(umap_alphas),
            baseline_emb=_baseline_pooled,
            seq_emb=_seq_pooled,
            sequence_name=sequence_name,
            baseline_name=baseline_name,
            interpolation_method=interpolation_path,
            predictions=np.array(path_predictions) if path_predictions else None,
            save_path=interpolation_umap_save_path,
        )

    # Compute integrated gradients using the specified attribution method
    # Attribution = (input - baseline) × average_gradient

    # Get the embedding differences
    seq_emb_torch = torch.from_numpy(seq_embedding).to(device).float().squeeze(0)  # (seq_len, embed_dim)
    baseline_emb_torch = torch.from_numpy(baseline_embedding).to(device).float().squeeze(0)  # (seq_len, embed_dim)
    embed_diff = seq_emb_torch - baseline_emb_torch  # (seq_len, embed_dim)

    if attribution_method == 'signed':
        # Compute signed attributions using dot product with embedding differences
        # This is the theoretically correct integrated gradients formula
        position_gradients = torch.sum(avg_gradients * embed_diff, dim=1).cpu().numpy()  # (seq_len,)
    else:  # attribution_method == 'magnitude'
        # Compute magnitude attributions using L2 norm of gradients
        # Note: This violates the completeness axiom but may be useful for interpretation
        position_gradients = torch.norm(avg_gradients, dim=1).cpu().numpy()  # (seq_len,)
        print("WARNING: Using magnitude attribution method - this violates the completeness axiom!")
        print("For theoretically correct integrated gradients, use attribution_method='signed'")

    print(f"Using attribution method: {attribution_method}")
    print(f"Sum of attributions: {np.sum(position_gradients):.6f}")
    print(f"Predicted difference (sequence - baseline): {seq_pred - baseline_pred:.6f}")
    print(f"Completeness axiom error: {abs(np.sum(position_gradients) - (seq_pred - baseline_pred)):.6f}")

    # Additional check: Compare with unclipped predictions for theoretical correctness
    print("\n=== COMPLETENESS AXIOM VALIDATION ===")

    # Compute unclipped predictions for theoretical comparison
    seq_pred_unclipped = seq_pred_norm * y_log_std + y_log_mean
    baseline_pred_unclipped = baseline_pred_norm * y_log_std + y_log_mean

    theoretical_diff = seq_pred_unclipped - baseline_pred_unclipped
    theoretical_error = abs(np.sum(position_gradients) - theoretical_diff)

    print(f"Theoretical (unclipped) difference: {theoretical_diff:.6f}")
    print(f"Theoretical completeness error: {theoretical_error:.6f}")

    if theoretical_error < 0.01:
        print("✅ Completeness axiom satisfied for unclipped predictions!")
    elif theoretical_error < 0.1:
        print("⚠️ Minor completeness axiom violation (< 0.1)")
    else:
        print("❌ Significant completeness axiom violation")

    if clip_min is not None or clip_max is not None:
        clipping_effect = abs((seq_pred - baseline_pred) - theoretical_diff)
        print(f"Clipping effect on difference: {clipping_effect:.6f}")
        if clipping_effect > 0.01:
            print("📝 Note: Large clipping effect may explain apparent completeness violation")

    print("=" * 50)

    # Additional normalization consistency check
    print("\n=== NORMALIZATION CONSISTENCY CHECK ===")

    # Check if we can reproduce the predictions using the same method as IG
    # This helps identify normalization mismatches

    # Method 1: DataLoader prediction (what we currently use for final predictions)
    seq_pred_str = f"Sequence={seq_pred:.6f}"
    baseline_pred_str = f"Baseline={baseline_pred:.6f}"
    diff_str = f"Diff={seq_pred - baseline_pred:.6f}"
    print(f"Method 1 (DataLoader): {seq_pred_str}, {baseline_pred_str}, {diff_str}")

    # Method 2: Direct model prediction (what IG uses during interpolation)
    # This should match Method 1 if normalization is consistent
    try:
        from .data_processing import _global_scaler_mean, _global_scaler_scale

        # Test direct prediction without DataLoader
        test_seq_emb = torch.from_numpy(seq_embedding).to(device).float()  # (1, seq_len, embed_dim)
        test_baseline_emb = torch.from_numpy(baseline_embedding).to(device).float()  # (1, seq_len, embed_dim)

        # Apply same normalization as IG does
        def normalize_like_ig(embeddings):
            batch_size, seq_len, embed_dim = embeddings.shape
            flat_embeddings = embeddings.reshape(batch_size, -1)

            if _global_scaler_mean is not None and _global_scaler_scale is not None:
                flat_normalized = (flat_embeddings - _global_scaler_mean) / _global_scaler_scale
            else:
                mean = flat_embeddings.mean(dim=1, keepdim=True)
                std = flat_embeddings.std(dim=1, keepdim=True)
                flat_normalized = (flat_embeddings - mean) / (std + 1e-8)

            return flat_normalized.reshape(batch_size, seq_len, embed_dim)

        with torch.no_grad():
            seq_normalized = normalize_like_ig(test_seq_emb)
            baseline_normalized = normalize_like_ig(test_baseline_emb)

            if pooling_method == 'esm_attention':
                seq_len = seq_normalized.size(1)
                input_mask = torch.ones(1, seq_len, device=device)
                seq_pred_direct = predictor_model(seq_normalized, input_mask).item()
                baseline_pred_direct = predictor_model(baseline_normalized, input_mask).item()
            else:
                seq_pred_direct = predictor_model(seq_normalized).item()
                baseline_pred_direct = predictor_model(baseline_normalized).item()

            # Convert to original scale
            seq_pred_direct = seq_pred_direct * y_log_std + y_log_mean
            baseline_pred_direct = baseline_pred_direct * y_log_std + y_log_mean

            # Apply the same clipping as the DataLoader method for consistency
            if clip_min is not None or clip_max is not None:
                if clip_min is not None and clip_max is not None:
                    seq_pred_direct = np.clip(seq_pred_direct, clip_min, clip_max)
                    baseline_pred_direct = np.clip(baseline_pred_direct, clip_min, clip_max)
                elif clip_min is not None:
                    seq_pred_direct = np.maximum(seq_pred_direct, clip_min)
                    baseline_pred_direct = np.maximum(baseline_pred_direct, clip_min)
                elif clip_max is not None:
                    seq_pred_direct = np.minimum(seq_pred_direct, clip_max)
                    baseline_pred_direct = np.minimum(baseline_pred_direct, clip_max)

            seq_direct_str = f"Sequence={seq_pred_direct:.6f}"
            baseline_direct_str = f"Baseline={baseline_pred_direct:.6f}"
            diff_direct_str = f"Diff={seq_pred_direct - baseline_pred_direct:.6f}"
            print(f"Method 2 (Direct): {seq_direct_str}, {baseline_direct_str}, {diff_direct_str}")

            # Check consistency
            seq_diff = abs(seq_pred - seq_pred_direct)
            baseline_diff = abs(baseline_pred - baseline_pred_direct)
            pred_diff_error = abs((seq_pred - baseline_pred) - (seq_pred_direct - baseline_pred_direct))

            consistency_str = f"Seq_error={seq_diff:.6f}, Baseline_error={baseline_diff:.6f}"
            diff_error_str = f"Diff_error={pred_diff_error:.6f}"
            print(f"Consistency: {consistency_str}, {diff_error_str}")

            if pred_diff_error > 1e-4:
                print("WARNING: Significant prediction difference between DataLoader and direct methods!")
                print("This suggests a normalization inconsistency that could affect IG completeness.")
            else:
                print("✓ Normalization appears consistent between methods.")

    except Exception as e:
        print(f"Normalization check failed: {e}")

    print("=" * 45)

    # Position gradients are already computed above as L2 norm of embedding gradients
    # No need to exclude special tokens since we're working with embeddings directly

    # Create predicted vs actual plot if requested and actual values are provided
    if plot_predictions and actual_activity is not None and baseline_actual_activity is not None:
        print("Creating predicted vs actual activity plot...")
        from .plot import plot_predicted_vs_actual_single
        plot_predicted_vs_actual_single(
            predicted_value=seq_pred,
            actual_value=actual_activity,
            baseline_predicted=baseline_pred,
            baseline_actual=baseline_actual_activity,
            sequence_name=sequence_name,
            baseline_name=baseline_name,
            title="Predicted vs Actual Activity Comparison",
            save_path=save_plot_path
        )

    # Trim gradients to actual sequence length — DataLoader padding may add extra positions
    seq_len_actual = len(sequence)
    if len(position_gradients) > seq_len_actual:
        position_gradients = position_gradients[:seq_len_actual]

    return {
        'sequence': sequence,
        'baseline_sequence': baseline_sequence,
        'predicted_activity': seq_pred,
        'baseline_activity': baseline_pred,
        'effect_vs_baseline': seq_pred - baseline_pred,
        'position_gradients': position_gradients,
        'positions': list(range(1, seq_len_actual + 1)),
        'amino_acids': list(sequence),
        'actual_activity': actual_activity,
        'baseline_actual_activity': baseline_actual_activity
    }


def analyze_sequence_with_token_level_integrated_gradients(
    sequence: str,
    baseline_sequence: str,
    esm_model: torch.nn.Module,
    predictor_model: torch.nn.Module,
    alphabet,
    device: torch.device,
    esm_layer: int = 33,
    pooling_method: str = 'linear',
    y_log_mean: float = 0.0,
    y_log_std: float = 1.0,
    steps: int = 50,
    batch_size: int = 10,
    attribution_method: str = 'magnitude',  # 'magnitude' (L2 norm) or 'signed' (dot product)
    actual_activity: float | None = None,
    baseline_actual_activity: float | None = None,
    plot_predictions: bool = False,
    sequence_name: str = "Sequence",
    baseline_name: str = "Baseline",
    save_plot_path: str | None = None,
    clip_min: float | None = None,
    clip_max: float | None = None,
    embeddings_subdir: str | None = None,
    interpolation_path: str = 'linear',
    _esm_backend=None,
) -> dict:
    """
    Analyze a sequence using direct token-level integrated gradients.

    .. note::
        Token-level IG requires direct access to ESM-2 internals (token embedding
        matrix, transformer layers).  **ESM-C is not supported** for this analysis
        mode.  Use ``method: embedding`` in your config when using ESM-C.

    Args:
        sequence: Input protein sequence
        baseline_sequence: Baseline sequence (usually wild-type)
        esm_model: Pre-trained ESM model
        predictor_model: Trained prediction model
        alphabet: ESM alphabet
        device: PyTorch device
        esm_layer: ESM layer to extract features from
        pooling_method: Pooling method ('linear', 'esm_attention', etc.)
        y_log_mean: Mean for denormalization (default 0.0 for no denormalization)
        y_log_std: Std for denormalization (default 1.0 for no denormalization)
        steps: Number of integration steps
        batch_size: Batch size for gradient computation
        attribution_method: Attribution computation method ('magnitude' for L2 norm, 'signed' for dot product)
        actual_activity: Optional actual/experimental activity for the sequence
        baseline_actual_activity: Optional actual activity for the baseline
        plot_predictions: Whether to create predicted vs actual plot
        sequence_name: Name for the sequence in plots
        baseline_name: Name for the baseline in plots
        save_plot_path: Optional path to save the prediction plot
        clip_min: Optional minimum value to clip predictions
        clip_max: Optional maximum value to clip predictions

    Returns:
        dict with keys: sequence, baseline_sequence, predicted_activity,
                       baseline_activity, effect_vs_baseline, position_gradients,
                       positions, amino_acids
    """

    # Convert sequences to tokens
    seq_tokens = sequence_to_tokens(sequence, alphabet).to(device)
    baseline_tokens = sequence_to_tokens(baseline_sequence, alphabet).to(device)

    # Guard: token-level IG is ESM-2 only
    if alphabet is None:
        raise ValueError(
            "Token-level integrated gradients requires direct access to ESM-2 model "
            "internals and is not supported with ESM-C.  "
            "Set 'integrated_gradients.method: embedding' in your config when using "
            "an ESM-C model."
        )

    print(f"Sequence tokens shape: {seq_tokens.shape}")
    print(f"Baseline tokens shape: {baseline_tokens.shape}")

    # Set models to eval mode
    esm_model.eval()
    predictor_model.eval()

    # Get token embedding matrix from ESM model
    token_embed_matrix = esm_model.embed_tokens.weight  # (vocab_size, embed_dim)

    # Function to get prediction from token embeddings
    def predict_from_token_embeddings(token_embeds):
        """Forward pass from token embeddings to prediction"""
        # Forward pass through ESM transformer layers
        x = token_embeds

        # Apply position embeddings if ESM uses them
        if hasattr(esm_model, 'embed_positions'):
            positions = torch.arange(x.size(1), device=device).unsqueeze(0).expand(x.size(0), -1)
            x = x + esm_model.embed_positions(positions)

        # Apply layer norm if present
        if hasattr(esm_model, 'emb_layer_norm_before'):
            x = esm_model.emb_layer_norm_before(x)

        # Pass through transformer layers up to specified layer
        for layer_idx, layer in enumerate(esm_model.layers):
            x, _ = layer(x, None)
            if layer_idx == esm_layer - 1:  # ESM layers are 0-indexed
                break

        # Apply final layer norm if present
        if hasattr(esm_model, 'emb_layer_norm_after'):
            x = esm_model.emb_layer_norm_after(x)

        # Remove special tokens to match training format
        seq_embeddings = x[:, 1:-1, :]  # (batch, seq_len, embed_dim)

        # Apply the same global normalization as used during training
        from .data_processing import _global_scaler_mean, _global_scaler_scale

        batch_size_cur, seq_len, embed_dim = seq_embeddings.shape
        flat_embeddings = seq_embeddings.reshape(batch_size_cur, -1)

        # Apply global normalization (same as batch_pad_and_normalize with use_global_normalization=True)
        if _global_scaler_mean is not None and _global_scaler_scale is not None:
            flat_normalized = (flat_embeddings - _global_scaler_mean) / _global_scaler_scale
        else:
            # Fallback to per-sample normalization if global parameters not available
            mean = flat_embeddings.mean(dim=1, keepdim=True)
            std = flat_embeddings.std(dim=1, keepdim=True)
            flat_normalized = (flat_embeddings - mean) / (std + 1e-8)

        normalized_embeddings = flat_normalized.reshape(batch_size_cur, seq_len, embed_dim)

        # Apply predictor model
        if pooling_method == 'esm_attention':
            input_mask = torch.ones(batch_size_cur, seq_len, device=device)
            outputs = predictor_model(normalized_embeddings, input_mask)
        else:
            outputs = predictor_model(normalized_embeddings)

        return outputs

    # Get baseline predictions using DataLoader approach (same as embedding-level IG)
    print("Computing predictions using DataLoader method (same as embedding analysis)...")

    # First get ESM embeddings using the standard pipeline
    import hashlib

    import numpy as np

    from .data_processing import CustomDataset
    from .embedding import get_embeddings

    # Use unique dataset names to avoid caching conflicts
    seq_hash = hashlib.md5(sequence.encode()).hexdigest()[:8]
    baseline_hash = hashlib.md5(baseline_sequence.encode()).hexdigest()[:8]

    seq_embedding = get_embeddings(
        [sequence], layer=esm_layer, batch_size=1, dataset_name=f'token_ig_seq_{seq_hash}',
        subdir=embeddings_subdir,
        esm_backend=_esm_backend,
    )
    baseline_embedding = get_embeddings(
        [baseline_sequence], layer=esm_layer, batch_size=1, dataset_name=f'token_ig_baseline_{baseline_hash}',
        subdir=embeddings_subdir,
        esm_backend=_esm_backend,
    )

    # Use DataLoader approach for predictions (exactly like embedding-level IG)

    predictor_model.eval()

    # Sequence prediction using DataLoader with global normalization
    dummy_target = np.zeros(1)
    seq_dataset = CustomDataset(seq_embedding, dummy_target)
    seq_loader = DataLoader(seq_dataset, batch_size=1, shuffle=False, collate_fn=batch_pad_and_normalize)
    with torch.no_grad():
        for xb, lengths, yb in seq_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if pooling_method == 'esm_attention':
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                seq_pred_norm = predictor_model(xb, input_mask).squeeze(-1).cpu().numpy()[0]
            else:
                seq_pred_norm = predictor_model(xb).squeeze(-1).cpu().numpy()[0]
            break

    # Baseline prediction using DataLoader with global normalization
    baseline_dataset = CustomDataset(baseline_embedding, dummy_target)
    baseline_loader = DataLoader(
        baseline_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=batch_pad_and_normalize
    )

    with torch.no_grad():
        for xb, lengths, yb in baseline_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if pooling_method == 'esm_attention':
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                baseline_pred_norm = predictor_model(xb, input_mask).squeeze(-1).cpu().numpy()[0]
            else:
                baseline_pred_norm = predictor_model(xb).squeeze(-1).cpu().numpy()[0]
            break

    # Convert to original scale
    seq_pred = seq_pred_norm * y_log_std + y_log_mean
    baseline_pred = baseline_pred_norm * y_log_std + y_log_mean

    # Apply clipping if specified
    if clip_min is not None or clip_max is not None:
        print(f"Applying prediction clipping: min={clip_min}, max={clip_max}")
        if clip_min is not None and clip_max is not None:
            seq_pred = np.clip(seq_pred, clip_min, clip_max)
            baseline_pred = np.clip(baseline_pred, clip_min, clip_max)
        elif clip_min is not None:
            seq_pred = np.maximum(seq_pred, clip_min)
            baseline_pred = np.maximum(baseline_pred, clip_min)
        elif clip_max is not None:
            seq_pred = np.minimum(seq_pred, clip_max)
            baseline_pred = np.minimum(baseline_pred, clip_max)

    print(f"Sequence prediction: {seq_pred:.4f}")
    print(f"Baseline prediction: {baseline_pred:.4f}")

    # Compute integrated gradients using direct token-level approach
    print(f"Computing token-level integrated gradients (steps={steps})...")

    # Generate interpolation steps
    alphas = torch.linspace(0, 1, steps + 1, device=device)
    all_token_gradients = []

    for i in range(0, len(alphas), batch_size):
        batch_alphas = alphas[i:i + batch_size]
        batch_token_gradients = []

        for alpha in batch_alphas:
            # Get token embeddings for sequence and baseline
            seq_token_embeds = token_embed_matrix[seq_tokens]  # (1, seq_len+2, embed_dim)
            baseline_token_embeds = token_embed_matrix[baseline_tokens]  # (1, seq_len+2, embed_dim)

            # Interpolate at token embedding level
            interpolated_embeds = interpolate_embeddings(
                baseline_token_embeds, seq_token_embeds, alpha.item(), method=interpolation_path
            )
            interpolated_embeds = interpolated_embeds.clone().detach().requires_grad_(True)

            # Forward pass to get prediction
            outputs = predict_from_token_embeddings(interpolated_embeds)

            # Convert outputs to original scale for gradient computation
            # Note: We do NOT apply clipping here to preserve mathematical correctness of IG
            outputs_rescaled = outputs * y_log_std + y_log_mean

            # Compute gradients w.r.t. interpolated token embeddings
            token_gradients = torch.autograd.grad(
                outputs=outputs_rescaled.sum(),
                inputs=interpolated_embeds,
                create_graph=False,
                retain_graph=False
            )[0]  # (1, seq_len+2, embed_dim) - includes CLS/EOS

            # Remove special tokens (CLS and EOS) and compute position-wise attributions
            seq_token_grads = token_gradients[:, 1:-1, :]  # (1, seq_len, embed_dim)

            if attribution_method == 'signed':
                # Compute signed attributions using dot product with token embedding differences
                seq_token_embeds_clean = seq_token_embeds[:, 1:-1, :]  # Remove CLS/EOS
                baseline_token_embeds_clean = baseline_token_embeds[:, 1:-1, :]  # Remove CLS/EOS
                embed_diff = seq_token_embeds_clean - baseline_token_embeds_clean  # (1, seq_len, embed_dim)

                # Dot product between gradients and embedding differences gives signed attribution
                position_grads = torch.sum(seq_token_grads * embed_diff, dim=2).cpu().detach().numpy()[0]  # (seq_len,)
            else:  # attribution_method == 'magnitude'
                # Compute magnitude attributions using L2 norm of gradients (default behavior)
                position_grads = torch.norm(seq_token_grads, dim=2).cpu().detach().numpy()[0]  # (seq_len,)

            batch_token_gradients.append(position_grads)

        all_token_gradients.extend(batch_token_gradients)

    # Average gradients across integration steps
    position_gradients = np.mean(all_token_gradients, axis=0)  # (seq_len,)

    print(f"Computed gradients for {len(position_gradients)} positions")
    print(f"Gradient range: [{np.min(position_gradients):.6f}, {np.max(position_gradients):.6f}]")

    # Completeness axiom validation
    print(f"Using attribution method: {attribution_method}")
    print(f"Sum of attributions: {np.sum(position_gradients):.6f}")
    print(f"Predicted difference (sequence - baseline): {seq_pred - baseline_pred:.6f}")
    print(f"Completeness axiom error: {abs(np.sum(position_gradients) - (seq_pred - baseline_pred)):.6f}")

    # Additional check: Compare with unclipped predictions for theoretical correctness
    print("\n=== COMPLETENESS AXIOM VALIDATION ===")

    # Compute unclipped predictions for theoretical comparison
    seq_pred_unclipped = seq_pred_norm * y_log_std + y_log_mean
    baseline_pred_unclipped = baseline_pred_norm * y_log_std + y_log_mean

    theoretical_diff = seq_pred_unclipped - baseline_pred_unclipped
    theoretical_error = abs(np.sum(position_gradients) - theoretical_diff)

    print(f"Theoretical (unclipped) difference: {theoretical_diff:.6f}")
    print(f"Theoretical completeness error: {theoretical_error:.6f}")

    if theoretical_error < 0.01:
        print("✅ Completeness axiom satisfied for unclipped predictions!")
    elif theoretical_error < 0.1:
        print("⚠️ Minor completeness axiom violation (< 0.1)")
    else:
        print("❌ Significant completeness axiom violation")

    if clip_min is not None or clip_max is not None:
        clipping_effect = abs((seq_pred - baseline_pred) - theoretical_diff)
        print(f"Clipping effect on difference: {clipping_effect:.6f}")
        if clipping_effect > 0.01:
            print("📝 Note: Large clipping effect may explain apparent completeness violation")

    print("=" * 50)

    # Create predicted vs actual plot if requested
    if plot_predictions and actual_activity is not None and baseline_actual_activity is not None:
        print("Creating predicted vs actual activity plot...")
        from .plot import plot_predicted_vs_actual_single
        plot_predicted_vs_actual_single(
            predicted_value=seq_pred,
            actual_value=actual_activity,
            baseline_predicted=baseline_pred,
            baseline_actual=baseline_actual_activity,
            sequence_name=sequence_name,
            baseline_name=baseline_name,
            title="Token-Level IG: Predicted vs Actual Activity",
            save_path=save_plot_path
        )

    return {
        'sequence': sequence,
        'baseline_sequence': baseline_sequence,
        'predicted_activity': seq_pred,
        'baseline_activity': baseline_pred,
        'effect_vs_baseline': seq_pred - baseline_pred,
        'position_gradients': position_gradients,
        'positions': list(range(1, len(sequence) + 1)),
        'amino_acids': list(sequence),
        'actual_activity': actual_activity,
        'baseline_actual_activity': baseline_actual_activity,
        'method': 'token_level_integrated_gradients'
    }
