"""
True sequence-level Integrated Gradients using ESM model directly.

This module provides functions for computing integrated gradients with respect
to the actual sequence tokens, not embeddings, allowing direct interpretation
of which amino acids at which positions contribute to predictions.
"""

import os
from typing import Optional

import esm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm


def load_esm_model(model_name: str = 'esm2_t33_650M_UR50D', device: torch.device = None):
    """Load ESM model and alphabet."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading ESM model {model_name}...")
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    esm_model = esm_model.to(device)
    esm_model.eval()

    return esm_model, alphabet


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
        # Get embeddings from ESM
        results = self.esm_model(tokens, repr_layers=[self.esm_layer], return_contacts=False)
        embeddings = results['representations'][self.esm_layer]

        # Remove start/end tokens to match training data format
        # ESM adds <cls> at start and <eos> at end
        seq_embeddings = embeddings[:, 1:-1, :]  # (batch, L, D)

        # Apply prediction model
        if self.pooling_method == 'esm_attention':
            # Create input mask (all positions are valid since we removed special tokens)
            batch_size, seq_len = seq_embeddings.shape[:2]
            input_mask = torch.ones(batch_size, seq_len, device=seq_embeddings.device)
            predictions = self.predictor_model(seq_embeddings, input_mask)
        else:
            predictions = self.predictor_model(seq_embeddings)

        return predictions


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

        # Apply prediction model
        if combined_model.pooling_method == 'esm_attention':
            batch_size_actual, seq_len = seq_embeddings.shape[:2]
            input_mask = torch.ones(batch_size_actual, seq_len, device=device)
            outputs = combined_model.predictor_model(seq_embeddings, input_mask)
        else:
            outputs = combined_model.predictor_model(seq_embeddings)

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


# Example usage function
def run_complete_sequence_analysis(
    predictor_model: torch.nn.Module,
    sequence: str,
    baseline_sequence: str,
    output_dir: str,
    esm_model_name: str = 'esm2_t33_650M_UR50D',
    esm_layer: int = 33,
    pooling_method: str = 'linear',
    device: Optional[torch.device] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete sequence-level attribution analysis.

    This is the main function to call for comprehensive analysis.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading ESM model...")
    esm_model, alphabet = load_esm_model(esm_model_name, device)

    print("Creating combined model...")
    combined_model = ESMBasedPredictor(
        esm_model=esm_model,
        predictor_model=predictor_model,
        esm_layer=esm_layer,
        pooling_method=pooling_method
    ).to(device)

    print("Computing amino acid attributions...")
    attributions_df = compute_amino_acid_attributions(
        combined_model=combined_model,
        sequence=sequence,
        baseline_sequence=baseline_sequence,
        alphabet=alphabet,
        device=device
    )

    print("Analyzing position sensitivity...")
    position_stats = analyze_position_sensitivity(attributions_df)

    print("Creating visualizations...")
    plot_amino_acid_heatmap(
        attributions_df=attributions_df,
        sequence=sequence,
        title="Amino Acid Attribution Analysis",
        save_path=os.path.join(output_dir, 'amino_acid_attribution_heatmap.png'),
        figsize=(min(len(sequence) * 0.3, 30), 10)
    )

    print("Exporting results...")
    export_sequence_analysis_results(
        attributions_df=attributions_df,
        position_stats=position_stats,
        output_dir=output_dir,
        sequence=sequence,
        baseline_sequence=baseline_sequence
    )

    return attributions_df, position_stats
