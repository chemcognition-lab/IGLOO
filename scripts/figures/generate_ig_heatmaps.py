#!/usr/bin/env python3
"""
Generate integrated gradients heatmaps from analysis results.

This script reads configuration files and their corresponding results to generate
p    plot_attribution_heatmap(
        data=data_matrix,
        x_labels=position_labels,
        y_labels=variant_labels,
        title=f"IG Attributions: All {data_name} Variants",
        value_label="IG Score",
        figsize=(max(12, len(position_labels) * 0.05), max(3, len(variant_labels) * 0.5)),
        output_path=output_path
    )n-ready heatmaps showing integrated gradient attributions across protein sequences.

Usage:
    python scripts/figures/generate_ig_heatmaps.py config_file.yaml
    python scripts/figures/generate_ig_heatmaps.py configs/betalactamase_config_embedding-signed.yaml

Features:
- Generates individual heatmaps for each variant with individual color scales
- Generates combined heatmap with unified color scale across all variants
- Automatically names variants based on mutations (e.g., V67G) instead of generic names
- Uses red-white-blue color scheme (red=negative, white=0, blue=positive)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import after adding to path
from src.igloo.plot import plot_attribution_heatmap  # noqa: E402


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_results_file(config: dict) -> str:
    """Find the amino acid effects detailed CSV file based on config."""
    results_dir = config['results_dir']
    data_name = config['data']['name']

    # Expected filename pattern
    results_file = os.path.join(results_dir, f"{data_name}_amino_acid_effects_detailed.csv")

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    return results_file


def identify_mutations(df: pd.DataFrame, wt_sequence: str) -> dict:
    """
    Identify mutations for each variant compared to wild-type.

    Args:
        df: DataFrame with results
        wt_sequence: Wild-type sequence from config

    Returns:
        dict: Mapping from sequence_name to mutation name (e.g., 'V67G')
    """
    mutation_names = {}

    # Get unique sequence names excluding WT
    sequence_names = [name for name in df['sequence_name'].unique() if name != 'WT']

    for seq_name in sequence_names:
        # Get this sequence's amino acids
        seq_data = df[df['sequence_name'] == seq_name].sort_values('position')

        if len(seq_data) == 0:
            continue

        # Find differences from WT
        mutations = []
        for _, row in seq_data.iterrows():
            pos = row['position'] - 1  # Convert to 0-based indexing
            if pos < len(wt_sequence):
                wt_aa = wt_sequence[pos]
                var_aa = row['amino_acid']
                if wt_aa != var_aa:
                    mutations.append(f"{wt_aa}{pos+1}{var_aa}")

        # Create mutation name
        if mutations:
            mutation_names[seq_name] = "_".join(mutations)
        else:
            mutation_names[seq_name] = seq_name  # Keep original if no clear mutations

    return mutation_names


def prepare_heatmap_data(df: pd.DataFrame, mutation_names: dict) -> tuple:
    """
    Prepare data for heatmap visualization.

    Args:
        df: DataFrame with integrated gradient results
        mutation_names: Mapping from sequence names to mutation names

    Returns:
        tuple: (data_matrix, position_labels, variant_labels)
    """
    # Create pivot table excluding WT
    variant_df = df[df['sequence_name'] != 'WT']
    pivot_data = variant_df.pivot(index='sequence_name', columns='position', values='integrated_gradient')

    # Replace sequence names with mutation names
    pivot_data.index = [mutation_names.get(name, name) for name in pivot_data.index]

    # Prepare labels
    position_labels = [f"Position {pos}" for pos in pivot_data.columns]
    variant_labels = list(pivot_data.index)

    return pivot_data.values, position_labels, variant_labels


def generate_individual_heatmaps(df: pd.DataFrame, mutation_names: dict, output_dir: str,
                                data_name: str) -> list:
    """Generate individual heatmaps for each variant with individual color scales."""
    output_paths = []

    for seq_name, mutation_name in mutation_names.items():
        # Get variant data only (no WT)
        var_data = df[df['sequence_name'] == seq_name].sort_values('position')

        if len(var_data) == 0:
            continue

        # Prepare data matrix (1 row: just the variant)
        positions = var_data['position'].tolist()
        var_gradients = var_data['integrated_gradient'].tolist()

        data_matrix = np.array([var_gradients])

        # Prepare labels
        position_labels = [f"Position {pos}" for pos in positions]
        variant_labels = [mutation_name]

        # Generate heatmap
        output_path = os.path.join(output_dir, f"{data_name}_{mutation_name}_individual_heatmap.png")

        plot_attribution_heatmap(
            data=data_matrix,
            x_labels=position_labels,
            y_labels=variant_labels,
            title=f"IG Attributions: {mutation_name}",
            value_label="IG Score",
            figsize=(max(12, len(positions) * 0.05), 2),
            output_path=output_path,
            font_scale=1.0  # Normal font size for individual heatmaps
        )

        output_paths.append(output_path)
        print(f"Generated individual heatmap: {output_path}")

    return output_paths


def generate_combined_heatmap(df: pd.DataFrame, mutation_names: dict, output_dir: str,
                             data_name: str) -> str:
    """Generate combined heatmap with all variants using unified color scale."""
    # Prepare data
    data_matrix, position_labels, variant_labels = prepare_heatmap_data(df, mutation_names)

    # Sort variants by performance (best to worst)
    # Calculate functional performance for each variant (higher = better performing)
    variant_performances = []
    for i, variant_label in enumerate(variant_labels):
        # Find original sequence name for this variant
        original_seq_name = None
        for seq_name, mut_name in mutation_names.items():
            if mut_name == variant_label:
                original_seq_name = seq_name
                break

        if original_seq_name:
            # Get functional performance value for this variant
            variant_data = df[df['sequence_name'] == original_seq_name]
            if len(variant_data) > 0:
                # Try to find functional performance column (common names)
                performance_value = None
                for col in ['functional_score', 'activity', 'fitness', 'performance', 'value', 'score']:
                    if col in variant_data.columns:
                        # Take first value (should be same for all positions)
                        performance_value = variant_data[col].iloc[0]
                        break

                # If no specific performance column found, use the mean of integrated gradients as fallback
                if performance_value is None:
                    performance_value = variant_data['integrated_gradient'].mean()
                    print(f"Warning: No functional performance column found, using mean IG for {variant_label}")

                variant_performances.append((i, variant_label, performance_value))

    # Sort by functional performance (descending - best to worst)
    variant_performances.sort(key=lambda x: x[2], reverse=True)

    # Print performance ranking
    print("Variant performance ranking (by functional performance):")
    for i, (_, variant_label, performance) in enumerate(variant_performances, 1):
        print(f"  {i}. {variant_label}: {performance:.4f}")

    # Reorder data matrix and labels
    sorted_indices = [x[0] for x in variant_performances]
    sorted_variant_labels = [x[1] for x in variant_performances]
    sorted_data_matrix = data_matrix[sorted_indices, :]

    # Generate combined heatmap
    output_path = os.path.join(output_dir, f"{data_name}_all_variants_combined_heatmap.png")

    plot_attribution_heatmap(
        data=sorted_data_matrix,
        x_labels=position_labels,
        y_labels=sorted_variant_labels,
        title="IG Attributions: All Variants (Ordered by Performance)",
        value_label="IG Score",
        figsize=(max(12, len(position_labels) * 0.08), max(6, len(sorted_variant_labels) * 0.4)),
        output_path=output_path,
        font_scale=1.5  # Larger font size for combined heatmap
    )

    print(f"Generated combined heatmap: {output_path}")
    return output_path


def main():
    """Main function to process config and generate heatmaps."""
    parser = argparse.ArgumentParser(description="Generate IG attribution heatmaps from results")
    parser.add_argument("config_path", help="Path to configuration YAML file")
    parser.add_argument("--output-dir", help="Override output directory")

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config_path)
        print(f"Loaded config from: {args.config_path}")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Find results file
    try:
        results_file = find_results_file(config)
        print(f"Found results file: {results_file}")
    except Exception as e:
        print(f"Error finding results file: {e}")
        sys.exit(1)

    # Load results data
    try:
        df = pd.read_csv(results_file)
        print(f"Loaded {len(df)} rows of results data")
    except Exception as e:
        print(f"Error loading results data: {e}")
        sys.exit(1)

    # Get wildtype sequence from config or extract from data
    if 'sequence' in config['data']['wildtype']:
        wt_sequence = config['data']['wildtype']['sequence']
    else:
        # Extract wildtype sequence from WT entries in the data
        wt_data = df[df['sequence_name'] == 'WT'].sort_values('position')
        if len(wt_data) == 0:
            raise ValueError("No WT sequence found in data and no sequence specified in config")
        wt_sequence = ''.join(wt_data['amino_acid'].tolist())
        print(f"Extracted WT sequence from data: {len(wt_sequence)} amino acids")

    data_name = config['data']['name']

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(config['results_dir'], 'heatmaps')

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Identify mutations
    mutation_names = identify_mutations(df, wt_sequence)
    print(f"Identified {len(mutation_names)} variants with mutations:")
    for seq_name, mut_name in mutation_names.items():
        print(f"  {seq_name} -> {mut_name}")

    # Generate individual heatmaps
    print("\nGenerating individual heatmaps...")
    individual_paths = generate_individual_heatmaps(df, mutation_names, output_dir, data_name)

    # Generate combined heatmap
    print("\nGenerating combined heatmap...")
    generate_combined_heatmap(df, mutation_names, output_dir, data_name)

    # Summary
    print(f"\n✅ Successfully generated {len(individual_paths)} individual heatmaps")
    print("✅ Successfully generated 1 combined heatmap")
    print(f"📁 All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
