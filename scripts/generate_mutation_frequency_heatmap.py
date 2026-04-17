#!/usr/bin/env python3
"""
Generate mutation frequency heatmap for a dataset.

This script analyzes the frequency of amino acid mutations at each position
in a dataset and creates a heatmap visualization showing:
- Amino acids on the y-axis
- Positions on the x-axis
- Color intensity representing frequency
- Wildtype amino acids marked with red circles

Usage:
    python scripts/generate_mutation_frequency_heatmap.py config_file.yaml
    python scripts/generate_mutation_frequency_heatmap.py configs/betalactamase_config.yaml
    python scripts/generate_mutation_frequency_heatmap.py configs/PafA_config.yaml

Output:
    - Mutation frequency heatmap PNG file
    - Summary statistics about mutation patterns
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import after adding to path
from src.igloo.plot import plot_mutation_frequency_heatmap  # noqa: E402


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_wildtype_sequence(config: dict, df: pd.DataFrame = None) -> str:
    """Extract wildtype sequence from config or dataset."""
    if 'wildtype' in config['data'] and 'sequence' in config['data']['wildtype']:
        return config['data']['wildtype']['sequence']
    elif 'wildtype' in config['data'] and config['data']['wildtype'].get('method') == 'position':
        # Handle position-based wildtype extraction
        if df is None:
            raise ValueError("Dataset required for position-based wildtype extraction")

        position = config['data']['wildtype'].get('position', 'last')

        # Find sequence column
        sequence_col = None
        for col in ['sequence', 'Sequence', 'seq', 'Seq', 'SEQ', 'protein_sequence', 'aa_seq', 'amino_acid_sequence']:
            if col in df.columns:
                sequence_col = col
                break

        if sequence_col is None:
            available_cols = list(df.columns)
            raise ValueError(f"No sequence column found in dataset for wildtype extraction. "
                           f"Available columns: {available_cols}")

        if position == 'last':
            wildtype_seq = df[sequence_col].iloc[-1]
        elif position == 'first':
            wildtype_seq = df[sequence_col].iloc[0]
        elif isinstance(position, int):
            wildtype_seq = df[sequence_col].iloc[position]
        else:
            raise ValueError(f"Unsupported wildtype position: {position}")

        return wildtype_seq
    else:
        raise ValueError("Wildtype sequence not found in config")


def find_dataset_file(config: dict) -> str:
    """Find the dataset CSV file based on config."""
    data_config = config['data']

    # Try different possible file paths
    possible_paths = []

    # Check for direct path specification in config
    if 'path' in data_config:
        possible_paths.append(data_config['path'])

    if 'file_path' in data_config:
        possible_paths.append(data_config['file_path'])

    if 'train_path' in data_config:
        possible_paths.append(data_config['train_path'])

    # Look for common file patterns
    data_name = data_config.get('name', '')
    possible_patterns = [
        f"data/{data_name}.csv",
        f"data/{data_name}_train.csv",
        f"data/{data_name}_data.csv",
        f"data/{data_name}_processed.csv",
        f"data/{data_name.lower()}.csv",
        f"data/{data_name.lower()}_train.csv",
        f"data/{data_name.lower()}_data.csv",
        f"data/{data_name.lower()}_processed.csv"
    ]
    possible_paths.extend(possible_patterns)

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Dataset file not found. Tried: {possible_paths}")


def analyze_mutation_frequencies(df: pd.DataFrame, wildtype_sequence: str) -> pd.DataFrame:
    """
    Analyze mutation frequencies from dataset.

    Args:
        df: DataFrame containing sequence data
        wildtype_sequence: String of wildtype sequence

    Returns:
        DataFrame with columns ['position', 'amino_acid', 'frequency']
    """
    # Standard amino acids
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    # Initialize frequency matrix
    frequencies = []

    # Determine sequence column
    sequence_col = None

    # Print available columns for debugging
    print(f"Available columns in dataset: {list(df.columns)}")

    for col in ['sequence', 'Sequence', 'seq', 'Seq', 'SEQ', 'protein_sequence', 'aa_seq', 'amino_acid_sequence']:
        if col in df.columns:
            sequence_col = col
            break

    if sequence_col is None:
        raise ValueError(f"No sequence column found in dataset. Available columns: {list(df.columns)}")

    print(f"Using sequence column: '{sequence_col}'")

    # Get all sequences
    sequences = df[sequence_col].dropna().tolist()

    print(f"Analyzing {len(sequences)} sequences...")
    print(f"Wildtype sequence length: {len(wildtype_sequence)}")

    # Count mutations at each position
    for pos in range(len(wildtype_sequence)):
        wt_aa = wildtype_sequence[pos]

        # Count occurrences of each amino acid at this position
        aa_counts = {aa: 0 for aa in amino_acids}
        total_valid_sequences = 0

        for seq in sequences:
            if pos < len(seq) and seq[pos] in amino_acids:
                aa_counts[seq[pos]] += 1
                total_valid_sequences += 1

        # Calculate frequencies and add to results
        for aa in amino_acids:
            frequency = aa_counts[aa] / total_valid_sequences if total_valid_sequences > 0 else 0
            frequencies.append({
                'position': pos + 1,  # 1-based indexing
                'amino_acid': aa,
                'frequency': frequency,
                'count': aa_counts[aa],
                'is_wildtype': aa == wt_aa
            })

    return pd.DataFrame(frequencies)


def print_mutation_statistics(freq_df: pd.DataFrame, wildtype_sequence: str):
    """Print summary statistics about mutation patterns."""
    print("\n" + "="*60)
    print("MUTATION FREQUENCY ANALYSIS SUMMARY")
    print("="*60)

    # Overall statistics
    total_positions = len(wildtype_sequence)
    len(freq_df['position'].unique())

    print("Dataset Overview:")
    print(f"  - Total positions analyzed: {total_positions}")
    print(f"  - Wildtype sequence length: {len(wildtype_sequence)}")

    # Wildtype frequencies
    wt_freq = freq_df[freq_df['is_wildtype']]['frequency']
    print("\nWildtype Conservation:")
    print(f"  - Average wildtype frequency: {wt_freq.mean():.3f}")
    print(f"  - Min wildtype frequency: {wt_freq.min():.3f}")
    print(f"  - Max wildtype frequency: {wt_freq.max():.3f}")

    # Most variable positions
    position_diversity = freq_df.groupby('position').apply(
        lambda x: 1 - (x['frequency'] ** 2).sum()  # Simpson's diversity index
    ).sort_values(ascending=False)

    print("\nMost Variable Positions (top 10):")
    for i, (pos, diversity) in enumerate(position_diversity.head(10).items(), 1):
        wt_aa = wildtype_sequence[pos-1]
        wt_freq = freq_df[(freq_df['position'] == pos) & (freq_df['is_wildtype'])]['frequency'].iloc[0]
        print(f"  {i:2d}. Position {pos:3d} ({wt_aa}): diversity={diversity:.3f}, WT freq={wt_freq:.3f}")

    # Most conserved positions
    print("\nMost Conserved Positions (top 10):")
    for i, (pos, diversity) in enumerate(position_diversity.tail(10).items(), 1):
        wt_aa = wildtype_sequence[pos-1]
        wt_freq = freq_df[(freq_df['position'] == pos) & (freq_df['is_wildtype'])]['frequency'].iloc[0]
        print(f"  {i:2d}. Position {pos:3d} ({wt_aa}): diversity={diversity:.3f}, WT freq={wt_freq:.3f}")

    # Most frequent mutations
    mutations = freq_df[~freq_df['is_wildtype'] & (freq_df['frequency'] > 0)].copy()
    mutations = mutations.sort_values('frequency', ascending=False)

    print("\nMost Frequent Mutations (top 15):")
    for i, (_, row) in enumerate(mutations.head(15).iterrows(), 1):
        pos = int(row['position'])
        wt_aa = wildtype_sequence[pos-1]
        mut_aa = row['amino_acid']
        freq = row['frequency']
        count = row['count']
        print(f"  {i:2d}. {wt_aa}{pos}{mut_aa}: {freq:.3f} ({count} occurrences)")

    print("="*60)


def main():
    """Main function to generate mutation frequency heatmap."""
    parser = argparse.ArgumentParser(
        description="Generate mutation frequency heatmap for a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate heatmap for betalactamase dataset
    python scripts/generate_mutation_frequency_heatmap.py configs/betalactamase_config.yaml

    # Generate heatmap for PafA dataset
    python scripts/generate_mutation_frequency_heatmap.py configs/PafA_config.yaml

    # Specify custom output directory
    python scripts/generate_mutation_frequency_heatmap.py configs/betalactamase_config.yaml --output-dir custom_output

    # Use different wildtype marker style
    python scripts/generate_mutation_frequency_heatmap.py configs/betalactamase_config.yaml --marker x
        """
    )

    parser.add_argument("config_path", help="Path to configuration YAML file")
    parser.add_argument("--output-dir", help="Output directory for heatmap (default: results_dir from config)")
    parser.add_argument("--marker", default="o", help="Wildtype marker style (o, x, s, ^, etc.)")
    parser.add_argument("--marker-size", type=int, default=50, help="Size of wildtype markers")
    parser.add_argument("--colormap", default="Greens", help="Colormap for frequency values")
    parser.add_argument("--figsize", default="24,8", help="Figure size as 'width,height'")

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config_path)
        print(f"Loaded config from: {args.config_path}")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Get dataset info
    dataset_name = config['data']['name']
    print(f"Processing dataset: {dataset_name}")

    # Find dataset file
    try:
        dataset_file = find_dataset_file(config)
        print(f"Found dataset file: {dataset_file}")
    except Exception as e:
        print(f"Error finding dataset file: {e}")
        sys.exit(1)

    # Load dataset
    try:
        df = pd.read_csv(dataset_file)
        print(f"Loaded dataset: {len(df)} rows")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Get wildtype sequence
    try:
        wildtype_sequence = get_wildtype_sequence(config, df)
        print(f"Wildtype sequence: {len(wildtype_sequence)} amino acids")
    except Exception as e:
        print(f"Error getting wildtype sequence: {e}")
        sys.exit(1)

    # Analyze mutation frequencies
    print("\nAnalyzing mutation frequencies...")
    freq_df = analyze_mutation_frequencies(df, wildtype_sequence)

    # Print statistics
    print_mutation_statistics(freq_df, wildtype_sequence)

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(config.get('results_dir', 'results'), 'mutation_analysis')

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Parse figure size
    try:
        width, height = map(float, args.figsize.split(','))
        figsize = (width, height)
    except ValueError:
        print("Warning: Invalid figsize format, using default (24,8)")
        figsize = (24, 8)

    # Generate heatmap
    output_path = os.path.join(output_dir, f"{dataset_name}_mutation_frequency_heatmap.png")

    print("\nGenerating mutation frequency heatmap...")
    plot_mutation_frequency_heatmap(
        data=freq_df,
        wildtype_sequence=wildtype_sequence,
        title=f"Mutation Frequency: {dataset_name} Dataset",
        figsize=figsize,
        output_path=output_path,
        wildtype_marker=args.marker,
        wildtype_marker_size=args.marker_size,
        cmap=args.colormap
    )

    # Save frequency data
    freq_data_path = os.path.join(output_dir, f"{dataset_name}_mutation_frequencies.csv")
    freq_df.to_csv(freq_data_path, index=False)

    print("\n✅ Analysis complete!")
    print(f"📊 Heatmap saved to: {output_path}")
    print(f"📋 Frequency data saved to: {freq_data_path}")
    print(f"📁 All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
