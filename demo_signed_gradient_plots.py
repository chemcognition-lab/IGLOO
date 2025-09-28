#!/usr/bin/env python3
"""
Demo script to show the new signed gradient plotting functions.

This script creates mock data to demonstrate the new visualization functions
that create mutation effect-style plots for signed gradients.
"""

import os
import sys

import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from igloo.plot import plot_signed_gradient_effects, plot_averaged_gradient_effects_comparison

def create_mock_ig_results():
    """Create mock integrated gradients results for demonstration."""
    np.random.seed(42)

    # Create mock sequences with different activity levels
    sequences = {
        'High_Activity_1': {'predicted_activity': 220.5, 'actual_activity': 230.0},
        'High_Activity_2': {'predicted_activity': 195.3, 'actual_activity': 200.0},
        'High_Activity_3': {'predicted_activity': 180.7, 'actual_activity': 175.0},
        'Medium_Activity_1': {'predicted_activity': 120.2, 'actual_activity': 115.0},
        'Medium_Activity_2': {'predicted_activity': 98.4, 'actual_activity': 105.0},
        'Low_Activity_1': {'predicted_activity': 45.1, 'actual_activity': 50.0},
        'Low_Activity_2': {'predicted_activity': 32.8, 'actual_activity': 35.0},
        'Low_Activity_3': {'predicted_activity': 28.3, 'actual_activity': 21.0},
    }

    ig_results = {}

    for name, info in sequences.items():
        # Generate mock sequence (546 positions like PafA)
        sequence_length = 546
        positions = list(range(1, sequence_length + 1))

        # Generate amino acids (randomly choose from common ones)
        amino_acids = np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=sequence_length)

        # Generate signed gradients that correlate with activity level
        # Higher activity sequences tend to have more beneficial positions
        activity_level = info['predicted_activity']

        if activity_level > 180:  # High activity
            # More positive gradients, fewer negative ones
            positive_bias = 0.3
            gradient_scale = 0.05
        elif activity_level > 100:  # Medium activity
            # Balanced positive and negative
            positive_bias = 0.0
            gradient_scale = 0.03
        else:  # Low activity
            # More negative gradients
            positive_bias = -0.2
            gradient_scale = 0.04

        # Generate gradients with realistic distribution
        base_gradients = np.random.normal(positive_bias, gradient_scale, sequence_length)

        # Add some hotspot positions with larger effects
        n_hotspots = np.random.randint(10, 25)
        hotspot_positions = np.random.choice(sequence_length, n_hotspots, replace=False)
        for pos in hotspot_positions:
            base_gradients[pos] *= np.random.uniform(2, 5)

        # Ensure some positions have very small gradients (neutral)
        neutral_positions = np.random.choice(sequence_length, sequence_length//4, replace=False)
        for pos in neutral_positions:
            base_gradients[pos] *= np.random.uniform(0.1, 0.3)

        ig_results[name] = {
            'sequence': ''.join(amino_acids),
            'baseline_sequence': ''.join(amino_acids),  # Assume same for demo
            'predicted_activity': info['predicted_activity'],
            'baseline_activity': 107.24,  # Mock baseline (like wildtype)
            'effect_vs_baseline': info['predicted_activity'] - 107.24,
            'position_gradients': base_gradients,
            'positions': positions,
            'amino_acids': list(amino_acids),
            'actual_activity': info['actual_activity'],
            'baseline_actual_activity': 120.0
        }

    return ig_results

def main():
    """Run the demo."""
    print("Creating mock integrated gradients results...")
    ig_results = create_mock_ig_results()

    # Create output directory
    output_dir = "demo_plots"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generated mock data for {len(ig_results)} sequences")

    # Demo 1: Single scatter plot showing all signed effects
    print("\n1. Creating signed gradient effects scatter plot...")
    signed_effects_path = os.path.join(output_dir, 'demo_signed_gradient_effects.png')
    plot_signed_gradient_effects(ig_results, signed_effects_path, top_n=20)

    # Demo 2: Comparison plot for top vs bottom sequences
    print("2. Creating top vs bottom sequences comparison plot...")
    comparison_effects_path = os.path.join(output_dir, 'demo_top_vs_bottom_effects.png')
    plot_averaged_gradient_effects_comparison(
        ig_results, comparison_effects_path,
        top_n=3, bottom_n=3
    )

    print(f"\nDemo plots saved to: {output_dir}/")
    print("Generated files:")
    print("  - demo_signed_gradient_effects.png: All sequences with beneficial/detrimental effects")
    print("  - demo_top_vs_bottom_effects.png: Comparison of high vs low activity sequences")

    # Print some statistics
    print("\nMock data statistics:")
    activities = [result['predicted_activity'] for result in ig_results.values()]
    print(f"  Predicted activities range: {min(activities):.1f} - {max(activities):.1f}")

    all_gradients = []
    for result in ig_results.values():
        all_gradients.extend(result['position_gradients'])

    beneficial = [g for g in all_gradients if g > 0]
    detrimental = [g for g in all_gradients if g < 0]

    print(f"  Total positions: {len(all_gradients)}")
    print(f"  Beneficial positions: {len(beneficial)} ({100*len(beneficial)/len(all_gradients):.1f}%)")
    print(f"  Detrimental positions: {len(detrimental)} ({100*len(detrimental)/len(all_gradients):.1f}%)")
    print(f"  Gradient range: {min(all_gradients):.4f} to {max(all_gradients):.4f}")

if __name__ == "__main__":
    main()
