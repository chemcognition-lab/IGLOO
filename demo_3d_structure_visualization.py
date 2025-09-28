#!/usr/bin/env python3
"""
Demo script showing how to visualize integrated gradient attributions on 3D protein structure.

This script demonstrates how to map integrated gradient attributions onto a 3D protein
structure using PyMOL and/or py3Dmol for interactive visualization.

Usage:
    python demo_3d_structure_visualization.py path/to/pdb_file.pdb
"""

import argparse
import os
import sys

import pandas as pd

# Add src to path
sys.path.insert(0, 'src')

from igloo.plot import plot_attributions_on_3d_structure


def create_demo_attribution_data():
    """Create mock attribution data for demonstration."""
    # This simulates the data you would get from the integrated gradients analysis
    demo_data = {
        'position': list(range(1, 547)),  # PafA has 546 residues
        'integrated_gradient': [],
        'sequence_name': ['Demo'] * 546,
        'amino_acid': ['A'] * 546  # Simplified for demo
    }

    # Create realistic gradient distribution
    import numpy as np
    np.random.seed(42)

    # Most positions have small gradients near zero
    base_gradients = np.random.normal(0, 0.001, 546)

    # Add some hotspot positions with significant effects
    hotspots = [50, 75, 120, 155, 200, 250, 300, 322, 350, 400, 450, 500]
    for pos in hotspots:
        if pos <= 546:
            # Some beneficial, some detrimental
            effect = np.random.choice([-1, 1]) * np.random.uniform(0.01, 0.05)
            base_gradients[pos-1] = effect

    demo_data['integrated_gradient'] = base_gradients.tolist()

    return pd.DataFrame(demo_data)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Demo 3D protein structure visualization with integrated gradients",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'pdb_path',
        type=str,
        help='Path to the PDB file of the protein structure'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='pafa_3d_attributions.html',
        help='Output path for the HTML visualization (default: pafa_3d_attributions.html)'
    )

    parser.add_argument(
        '--use-real-data',
        action='store_true',
        help='Use real attribution data from CSV file instead of demo data'
    )

    parser.add_argument(
        '--attribution-csv',
        type=str,
        help='Path to CSV file with real attribution data (requires --use-real-data)'
    )

    args = parser.parse_args()

    # Check if PDB file exists
    if not os.path.exists(args.pdb_path):
        print(f"Error: PDB file not found: {args.pdb_path}")
        sys.exit(1)

    print("Creating 3D protein structure visualization with integrated gradient attributions...")
    print(f"PDB file: {args.pdb_path}")
    print(f"Output file: {args.output}")

    # Get attribution data
    if args.use_real_data:
        if not args.attribution_csv:
            print("Error: --attribution-csv required when using --use-real-data")
            sys.exit(1)

        if not os.path.exists(args.attribution_csv):
            print(f"Error: Attribution CSV file not found: {args.attribution_csv}")
            sys.exit(1)

        print(f"Loading real attribution data from: {args.attribution_csv}")
        attribution_data = pd.read_csv(args.attribution_csv)

        # Verify required columns
        required_columns = ['position', 'integrated_gradient']
        missing_columns = [col for col in required_columns if col not in attribution_data.columns]
        if missing_columns:
            print(f"Error: Missing required columns in CSV: {missing_columns}")
            print(f"Available columns: {list(attribution_data.columns)}")
            sys.exit(1)
    else:
        print("Using demo attribution data...")
        attribution_data = create_demo_attribution_data()

    print(f"Attribution data shape: {attribution_data.shape}")
    print(f"Attribution range: {attribution_data['integrated_gradient'].min():.6f} to {attribution_data['integrated_gradient'].max():.6f}")

    # Create the 3D visualization
    try:
        plot_attributions_on_3d_structure(
            attribution_data=attribution_data,
            pdb_path=args.pdb_path,
            save_path=args.output,
            attribution_column='integrated_gradient',
            aggregation_method='mean',
            color_scheme='RdBu_r',
            html_output=True,
            pymol_output=True,
            figsize=(800, 600)
        )

        print("\n3D visualization created successfully!")
        print(f"Open {args.output} in your web browser to view the interactive visualization.")

        pymol_script = args.output.replace('.html', '.pml')
        if os.path.exists(pymol_script):
            print(f"PyMOL script saved to: {pymol_script}")
            print("To use with PyMOL: pymol " + pymol_script)

    except Exception as e:
        print(f"Error creating 3D visualization: {e}")
        print("Make sure you have the required dependencies installed:")
        print("  pip install py3Dmol")
        sys.exit(1)


if __name__ == "__main__":
    main()
