"""
Example script demonstrating both in-silico mutagenesis and advanced IG analysis.

This script shows how to use both approaches for protein variant analysis
after training a model.
"""

import os
import sys

import numpy as np
import torch

from src.igloo.advanced_ig import (
    analyze_sequence_motifs,
    compare_attribution_methods,
    compute_amino_acid_contributions,
    export_attribution_results,
    plot_attribution_landscape,
)
from src.igloo.mutagenesis import (
    analyze_position_preferences,
    compute_mutation_effects,
    export_top_mutations,
    plot_mutation_landscape,
)

# Add the project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)




def demo_mutagenesis_analysis(
    model,
    wt_sequence,
    device,
    results_dir,
    esm_layer=33,
    pooling_method='linear',
    y_log_mean=0.0,
    y_log_std=1.0
):
    """Demonstrate in-silico mutagenesis analysis."""

    print("\\n" + "="*50)
    print("IN-SILICO MUTAGENESIS ANALYSIS")
    print("="*50)

    # Compute all mutation effects
    print("Computing mutation effects for all possible single mutations...")
    mutation_df = compute_mutation_effects(
        model=model,
        wt_sequence=wt_sequence,
        device=device,
        esm_layer=esm_layer,
        y_log_mean=y_log_mean,
        y_log_std=y_log_std,
        pooling_method=pooling_method,
        batch_size=16,
        progress=True
    )

    # Analyze results
    print("\\nAnalyzing mutation effects...")
    analysis = analyze_position_preferences(mutation_df, top_n=10)

    # Print summary
    print("\\nSUMMARY:")
    for key, value in analysis['summary'].items():
        print(f"  {key}: {value}")

    print("\\nTOP 10 BENEFICIAL MUTATIONS:")
    print(analysis['top_mutations'][['mutation', 'predicted_activity', 'effect_vs_wt', 'fold_change']].round(3))

    print("\\nTOP 10 DETRIMENTAL MUTATIONS:")
    print(analysis['worst_mutations'][['mutation', 'predicted_activity', 'effect_vs_wt', 'fold_change']].round(3))

    # Create visualizations
    print("\\nCreating mutation landscape plot...")
    plot_path = os.path.join(results_dir, 'mutation_landscape.png')
    plot_mutation_landscape(
        mutation_df=mutation_df,
        wt_sequence=wt_sequence,
        save_path=plot_path,
        figsize=(20, 12),
        top_n_label=15
    )

    # Export results
    print("\\nExporting results...")
    csv_path = os.path.join(results_dir, 'all_mutations.csv')
    mutation_df.to_csv(csv_path, index=False)

    top_mutations_path = os.path.join(results_dir, 'top_mutations.csv')
    export_top_mutations(
        mutation_df=mutation_df,
        output_path=top_mutations_path,
        top_n=50,
        include_negative=True
    )

    return mutation_df, analysis


def demo_advanced_ig_analysis(
    model,
    target_sequence,
    wt_sequence,
    device,
    results_dir,
    esm_layer=33,
    pooling_method='linear'
):
    """Demonstrate advanced integrated gradients analysis."""

    print("\\n" + "="*50)
    print("ADVANCED INTEGRATED GRADIENTS ANALYSIS")
    print("="*50)

    # Compare different attribution methods
    print("Computing attributions using multiple methods...")
    attributions = compare_attribution_methods(
        model=model,
        sequence=target_sequence,
        baseline_sequence=wt_sequence,
        device=device,
        esm_layer=esm_layer,
        pooling_method=pooling_method
    )

    # Print attribution scores for top positions
    print("\\nTOP 10 POSITIONS BY CONSENSUS ATTRIBUTION:")
    consensus_scores = np.mean([scores for scores in attributions.values()], axis=0)
    top_positions = np.argsort(consensus_scores)[::-1][:10]

    for i, pos in enumerate(top_positions):
        aa = target_sequence[pos]
        score = consensus_scores[pos]
        print(f"  {i+1:2d}. Position {pos+1:3d} ({aa}): {score:.4f}")

    # Analyze sequence motifs
    print("\\nAnalyzing sequence motifs...")
    motif_analysis = analyze_sequence_motifs(
        attributions=consensus_scores,
        sequence=target_sequence,
        window_size=5,
        top_k=10
    )

    print("\\nTOP 5 IMPORTANT MOTIFS:")
    for _, motif in motif_analysis['top_motifs'].head(5).iterrows():
        print(f"  {motif['sequence']} (pos {motif['start_pos']}-{motif['end_pos']}): {motif['score']:.4f}")

    # Create visualization
    print("\\nCreating attribution landscape plot...")
    plot_path = os.path.join(results_dir, 'attribution_landscape.png')
    plot_attribution_landscape(
        attributions=consensus_scores,
        sequence=target_sequence,
        title="Attribution Analysis: Target vs WT",
        save_path=plot_path,
        figsize=(15, 10),
        highlight_threshold=np.std(consensus_scores)
    )

    # Compute amino acid contributions
    print("\\nComputing amino acid contributions...")
    aa_contributions = compute_amino_acid_contributions(
        model=model,
        sequence=target_sequence,
        baseline_sequence=wt_sequence,
        device=device,
        esm_layer=esm_layer,
        pooling_method=pooling_method,
        perturbation_size=0.1,
        n_samples=50
    )

    # Show top amino acid substitutions
    top_contributions = aa_contributions.sort_values('contribution', ascending=False).head(10)
    print("\\nTOP 10 BENEFICIAL AA SUBSTITUTIONS:")
    for _, row in top_contributions.iterrows():
        print(f"  {row['mutation']}: {row['contribution']:.4f}")

    # Export results
    print("\\nExporting results...")
    attribution_path = os.path.join(results_dir, 'attribution_comparison.csv')
    export_attribution_results(
        attributions=attributions,
        sequence=target_sequence,
        output_path=attribution_path,
        include_motifs=True
    )

    aa_contrib_path = os.path.join(results_dir, 'amino_acid_contributions.csv')
    aa_contributions.to_csv(aa_contrib_path, index=False)

    motifs_path = os.path.join(results_dir, 'sequence_motifs.csv')
    motif_analysis['all_motifs'].to_csv(motifs_path, index=False)

    return attributions, motif_analysis, aa_contributions


def main():
    """Main demo function."""

    # This is a demo - you would replace these with your actual trained model and data
    print("DEMO: Protein Variant Analysis with Mutagenesis and Advanced IG")
    print("Note: This is a demonstration script. Replace with your actual model and data.")

    # Example parameters (replace with your actual values)
    results_dir = "demo_results"
    os.makedirs(results_dir, exist_ok=True)

    # Example sequences (replace with your actual sequences)
    wt_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    target_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"  # Same as WT for demo

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # You would load your actual trained model here
    print(f"\\nDevice: {device}")
    print(f"WT sequence length: {len(wt_sequence)}")
    print(f"Target sequence length: {len(target_sequence)}")

    # Example model creation (replace with loading your trained model)
    print("\\nNote: Create and load your actual trained model here")
    print("Example model creation shown below:")
    print("""
    # Load your trained model like this:
    model = YourModelClass(...)
    model.load_state_dict(torch.load('path_to_your_model.pth'))
    model = model.to(device)
    model.eval()

    # Then run the analyses:
    mutation_df, analysis = demo_mutagenesis_analysis(
        model=model,
        wt_sequence=wt_sequence,
        device=device,
        results_dir=results_dir,
        pooling_method='your_pooling_method',
        y_log_mean=your_y_log_mean,
        y_log_std=your_y_log_std
    )

    attributions, motifs, aa_contrib = demo_advanced_ig_analysis(
        model=model,
        target_sequence=target_sequence,
        wt_sequence=wt_sequence,
        device=device,
        results_dir=results_dir,
        pooling_method='your_pooling_method'
    )
    """)


if __name__ == "__main__":
    main()
