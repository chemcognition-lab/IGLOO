#!/usr/bin/env python3
"""
Generate PDB files colored by Integrated Gradients values.

- Reads IG data from a *_amino_acid_effects_detailed.csv file
- Normalizes IG values per-variant into [-1, 1]
- Writes out PDB files with B-factors set to signed IG values
  (so PyMOL can color directly with spectrum b, red_white_blue, -1, 1)
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from Bio import PDB
from matplotlib.colors import LinearSegmentedColormap

# Add the src directory to the path to import igloo modules
sys.path.append(str(Path(__file__).parent.parent / "src"))


# -----------------------------
# Load CSV and normalize IG data
# -----------------------------
def load_and_normalize_ig(csv_path, normalization_method="global"):
    df = pd.read_csv(csv_path)

    if normalization_method == "per_variant":
        # Normalize per variant (ensures each variant spans [-1, 1])
        df["ig_normalized"] = df.groupby("sequence_name")["integrated_gradient"].transform(
            lambda x: x / x.abs().max() if x.abs().max() > 0 else x
        )
    elif normalization_method == "global":
        # Use global scaling to match heatmap exactly (same as TwoSlopeNorm approach)
        # This matches how the heatmap normalizes colors around zero
        # TwoSlopeNorm preserves the actual vmin/vmax, it does NOT make them symmetric

        # Don't force symmetry - use the actual range like TwoSlopeNorm does
        # This preserves the exact color mapping that appears in heatmaps
        df["ig_normalized"] = df["integrated_gradient"]
    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")

    return df


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def find_pdb_file(dataset_name, pdb_dir="data/PDBs"):
    """Find the appropriate PDB file for the dataset."""
    pdb_patterns = {
        'betalactamase': ['betalactamase_AF3.pdb', 'betalactamase.pdb'],
        'pafa': ['pafa_5TJ3.pdb', 'PafA.pdb'],
        'erbc': ['erbc_AF3.pdb', 'ERBC.pdb']
    }

    dataset_lower = dataset_name.lower()
    pdb_dir_path = Path(pdb_dir)

    if dataset_lower in pdb_patterns:
        for pattern in pdb_patterns[dataset_lower]:
            pdb_path = pdb_dir_path / pattern
            if pdb_path.exists():
                return str(pdb_path)

    # If no specific pattern found, look for any PDB file containing the dataset name
    for pdb_file in pdb_dir_path.glob("*.pdb"):
        if dataset_lower in pdb_file.name.lower():
            return str(pdb_file)

    raise FileNotFoundError(f"No PDB file found for dataset '{dataset_name}' in {pdb_dir}")


def load_ig_data(results_dir, normalization_method="global"):
    """Load integrated gradients data from the detailed CSV file."""
    csv_files = list(Path(results_dir).glob("*amino_acid_effects_detailed.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No detailed amino acid effects CSV found in {results_dir}")

    csv_file = csv_files[0]  # Use the first one found
    print(f"Loading IG data from: {csv_file}")
    return load_and_normalize_ig(csv_file, normalization_method)


# -----------------------------
# Write modified PDB
# -----------------------------
def write_pdb_with_ig(variant_name, df, pdb_in, pdb_out, chain_id="A"):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_in))

    # Subset for this variant
    subset = df[df["sequence_name"] == variant_name]
    pos_to_ig = dict(zip(subset["position"], subset["ig_normalized"]))

    # Debug: Print IG value statistics for this variant
    ig_values = subset["ig_normalized"].values
    print(f"  Variant {variant_name} IG stats:")
    print(f"    Min: {ig_values.min():.4f}, Max: {ig_values.max():.4f}")
    print(f"    Mean: {ig_values.mean():.4f}, Std: {ig_values.std():.4f}")
    near_zero_count = len(ig_values[(ig_values >= -0.01) & (ig_values <= 0.01)])
    print(f"    Values near zero (±0.01): {near_zero_count}/{len(ig_values)}")

    modified_residues = 0
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                res_id = residue.get_id()[1]  # residue number
                if res_id in pos_to_ig:
                    ig_val = float(pos_to_ig[res_id])
                    # Scale up the IG value so it's visible in PDB B-factor field (2 decimal places)
                    # Multiply by 100 to make small values visible
                    scaled_ig_val = ig_val * 100.0
                    for atom in residue:
                        atom.set_bfactor(scaled_ig_val)
                    modified_residues += 1
                else:
                    for atom in residue:
                        atom.set_bfactor(0.0)  # neutral if no data

    io = PDB.PDBIO()
    io.set_structure(structure)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(pdb_out), exist_ok=True)

    io.save(str(pdb_out))
    print(f"✔ Saved {pdb_out} ({modified_residues} residues modified)")
    return True


def get_available_variants(df):
    """Get list of available sequence variants in the data."""
    variants = df['sequence_name'].unique()
    return sorted(variants)


# -----------------------------
# Generate PyMOL script for red_white_blue coloring
# -----------------------------
def generate_pymol_script(output_dir, pdb_files, dataset_name, df=None, normalization_method="global"):
    """Generate a PyMOL script to visualize PDB files with red_white_blue coloring."""
    script_path = output_dir / f"{dataset_name}_ig_visualization.pml"

    # Determine color range based on normalization method
    if normalization_method == "global" and df is not None:
        # For global normalization, use the actual data range to match heatmap exactly
        vmin = df["ig_normalized"].min()
        vmax = df["ig_normalized"].max()
        color_range_text = f"range {vmin:.3f} to {vmax:.3f} (global scale - matches heatmap, scaled x100 in PDB)"
    elif normalization_method == "per_variant":
        color_range_text = "range -100 to 100 (per-variant scale, x100)"
    else:
        # Fallback for any other normalization method
        if df is not None:
            vmin = df["ig_normalized"].min()
            vmax = df["ig_normalized"].max()
            color_range_text = f"range {vmin:.3f} to {vmax:.3f} (scaled x100 in PDB)"
        else:
            color_range_text = "range -100 to 100 (default scale, x100)"

    script_lines = [
        f"# PyMOL script for visualizing {dataset_name} integrated gradients",
        "# Generated automatically by generate_ig_colored_pdbs.py",
        "#",
        f"# Usage: Open PyMOL and run: @{script_path.name}",
        "",
        "# Clear any existing objects",
        "delete all",
        "",
        "# Load all PDB files"
    ]

    for pdb_file in pdb_files:
        pdb_path = Path(pdb_file)
        object_name = pdb_path.stem.replace('-', '_').replace('%', 'pct')
        script_lines.append(f"load {pdb_path.name}, {object_name}")

    script_lines.extend([
        "",
        "# Set up the visualization",
        "hide everything",
        "show cartoon, all",
        "",
        f"# Color by B-factor with red_white_blue spectrum ({color_range_text})",
        "# Red = detrimental (negative IG), Blue = beneficial (positive IG)",
        "",
        "# Step 1: Color everything white as baseline",
        "color white, all",
        "",
        "# Step 2: Create selections for different IG ranges (scaled x100)",
        "select strong_negative, (b < -10)",
        "select weak_negative, (b > -10) and (b < -2)",
        "select near_zero, (b > -2) and (b < 2)",
        "select weak_positive, (b > 2) and (b < 10)",
        "select strong_positive, (b > 10)",
        "",
        "# Step 3: Apply colors manually for better control",
        "color red, strong_negative",
        "color salmon, weak_negative",
        "color white, near_zero",
        "color lightblue, weak_positive",
        "color blue, strong_positive",
        "",
        "# Step 4: Apply spectrum only to non-zero values for fine-tuning",
        "select non_zero, (b < -2) or (b > 2)",
        "spectrum b, red_white_blue, non_zero, -100, 100",
        "",
        "# Deselect all for clean view",
        "deselect",
        "",
        "# Set cartoon transparency for better visualization",
        "set cartoon_transparency, 0.1",
        "",
        "# Align all structures (if multiple variants)",
        "python",
        "objects = cmd.get_names(\"objects\")",
        "if len(objects) > 1:",
        "    for obj in objects[1:]:",
        "        cmd.align(obj, objects[0])",
        "python end",
        "",
        "# Set up a nice view",
        "bg_color white",
        "set ray_shadows, 0",
        "set depth_cue, 0",
        "set antialias, 2",
        "",
        "# Center and zoom",
        "zoom all",
        "center all",
        "",
        "print(\"\\n\" + \"=\"*60)",
        "print(\"INTEGRATED GRADIENTS COLOR SCALE:\")",
        "print(\"=\"*60)",
        "print(\"RED    = Detrimental effects (negative IG values)\")",
        "print(\"WHITE  = Neutral (zero IG values)\")",
        "print(\"BLUE   = Beneficial effects (positive IG values)\")",
        f"print(\"RANGE  = {color_range_text}\")",
        "print(\"=\"*60)"
    ])

    script_content = "\n".join(script_lines)

    with open(script_path, 'w') as f:
        f.write(script_content)

    print(f"Generated PyMOL script: {script_path}")
    return script_path


def generate_color_legend(output_dir, dataset_name, df=None, normalization_method="global",
                          figsize=(2, 8), dpi=300):
    """
    Generate a standalone color gradient legend for IG values.

    Args:
        output_dir: Directory to save the legend
        dataset_name: Name of the dataset (for filename)
        df: DataFrame containing IG data
        normalization_method: "global" or "per_variant"
        figsize: Figure size as (width, height)
        dpi: Resolution for the saved image

    Returns:
        str: Path to the saved legend image
    """
    # Create red-white-blue colormap (red=negative/detrimental, blue=positive/beneficial)
    rwb_colormap = LinearSegmentedColormap.from_list(
        "red_white_blue",
        ["red", "white", "blue"],
        N=256
    )

    # Determine value range
    if normalization_method == "global" and df is not None:
        # For global normalization, use the actual data range (like TwoSlopeNorm)
        vmin = df["ig_normalized"].min()
        vmax = df["ig_normalized"].max()
        title = "IG Score\n(Global Scale)"
    else:
        # For per-variant normalization, show the actual original range
        if df is not None:
            # Get the actual min/max from the original integrated_gradient values
            actual_vmin = df["integrated_gradient"].min()
            actual_vmax = df["integrated_gradient"].max()
            # Ensure symmetric range around zero for proper diverging colormap
            abs_max = max(abs(actual_vmin), abs(actual_vmax))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = -1.0, 1.0
        title = "IG Score\n(Per-Variant Scale)"

    # Create figure and colorbar
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=rwb_colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    # Create vertical colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=1.0, pad=0.0)

    # Set title and labels - position to the left of colorbar
    cbar.set_label(title, fontsize=14, fontweight='bold', labelpad=-50, rotation=0, ha='right', va='center')

    # Set specific ticks at key values
    tick_values = [vmin, 0, vmax]
    tick_labels = [f'{vmin:.2f}', '0.00', f'{vmax:.2f}']
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(tick_labels)

    # Style the colorbar
    cbar.ax.tick_params(labelsize=12)

    # Add text annotations with better spacing to prevent overlap
    text_offset = 2.0  # Horizontal offset from colorbar

    # Position text at specific fractions of the range with closer spacing
    cbar.ax.text(text_offset, vmax * 0.85, 'Beneficial',
                 ha='left', va='center', fontsize=10, color='darkblue', weight='bold')
    cbar.ax.text(text_offset, vmax * 0.7, '(Positive)',
                 ha='left', va='center', fontsize=9, color='darkblue')

    cbar.ax.text(text_offset, 0, 'Neutral',
                 ha='left', va='center', fontsize=10, color='black', weight='bold')

    cbar.ax.text(text_offset, vmin * 0.7, 'Detrimental',
                 ha='left', va='center', fontsize=10, color='darkred', weight='bold')
    cbar.ax.text(text_offset, vmin * 0.85, '(Negative)',
                 ha='left', va='center', fontsize=9, color='darkred')

    # Remove the main axis
    ax.set_visible(False)

    # Save the legend
    legend_path = output_dir / f"{dataset_name}_ig_color_legend.png"
    try:
        plt.tight_layout()
    except UserWarning:
        # Ignore tight layout warnings for colorbar-only figures
        pass
    plt.savefig(legend_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"Generated color legend: {legend_path}")
    return str(legend_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDB files colored by integrated gradients values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate PDBs with global normalization (matches heatmaps) + color legend
    python generate_ig_colored_pdbs.py configs/betalactamase_config.yaml

    # Generate PDBs with per-variant normalization (each variant spans [-1,1])
    python generate_ig_colored_pdbs.py configs/betalactamase_config.yaml --normalization per_variant

    # Generate PDBs for specific variants only
    python generate_ig_colored_pdbs.py configs/betalactamase_config.yaml --variants "Highest_Activity,Lowest_Activity"

    # Custom legend size (width, height in inches)
    python generate_ig_colored_pdbs.py configs/betalactamase_config.yaml --legend-size "3,10"

    # Specify custom output directory
    python generate_ig_colored_pdbs.py configs/betalactamase_config.yaml --output-dir custom_pdbs

    # List available variants
    python generate_ig_colored_pdbs.py configs/betalactamase_config.yaml --list-variants

Note: The generated color legend PNG can be used directly in figures alongside structure snapshots!
        """
    )

    parser.add_argument("config", help="Path to the configuration YAML file")
    parser.add_argument("--variants", help="Comma-separated list of variants to process (default: all)")
    parser.add_argument("--output-dir", help="Output directory for colored PDB files")
    parser.add_argument("--pdb-file", help="Specific PDB file to use (overrides auto-detection)")
    parser.add_argument("--chain-id", default="A", help="Chain ID to modify (default: A)")
    parser.add_argument("--normalization", choices=["global", "per_variant"], default="global",
                       help="IG normalization: 'global' matches heatmaps, 'per_variant' individual (default: global)")
    parser.add_argument("--legend-size", default="2,8",
                       help="Color legend size as 'width,height' in inches (default: 2,8)")

    parser.add_argument("--list-variants", action="store_true",
                       help="List available variants and exit")

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return 1

    # Get dataset info
    dataset_name = config['data']['name']
    results_dir = config['results_dir']

    print(f"Processing dataset: {dataset_name}")
    print(f"Results directory: {results_dir}")

    # Load IG data
    try:
        df = load_ig_data(results_dir, args.normalization)
    except Exception as e:
        print(f"Error loading IG data: {e}")
        return 1

    if args.normalization == "global":
        print("Using global normalization (matches heatmap color scaling)")
    else:
        print("Applied per-variant normalization to IG values (range [-1, 1])")

    # List available variants if requested
    if args.list_variants:
        variants = get_available_variants(df)
        print(f"\nAvailable variants ({len(variants)}):")
        for variant in variants:
            count = len(df[df['sequence_name'] == variant])
            print(f"  - {variant} ({count} residues)")
        return 0

    # Find PDB file
    if args.pdb_file:
        pdb_input = args.pdb_file
        if not os.path.exists(pdb_input):
            print(f"Error: PDB file not found: {pdb_input}")
            return 1
    else:
        try:
            pdb_input = find_pdb_file(dataset_name)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

    print(f"Using PDB file: {pdb_input}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(results_dir) / "colored_pdbs"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine which variants to process
    if args.variants:
        variants_to_process = [v.strip() for v in args.variants.split(',')]
    else:
        variants_to_process = get_available_variants(df)

    print(f"Processing {len(variants_to_process)} variant(s): {', '.join(variants_to_process)}")

    # Generate colored PDB files
    success_count = 0
    for variant in variants_to_process:
        # Create output filename
        pdb_base = Path(pdb_input).stem
        output_file = output_dir / f"{pdb_base}_{variant}_ig_colored.pdb"

        success = write_pdb_with_ig(
            variant_name=variant,
            df=df,
            pdb_in=pdb_input,
            pdb_out=str(output_file),
            chain_id=args.chain_id
        )

        if success:
            success_count += 1

    print(f"\nCompleted: {success_count}/{len(variants_to_process)} PDB files generated successfully")

    if success_count > 0:
        # Generate PyMOL script
        successful_pdb_files = []
        for variant in variants_to_process:
            pdb_base = Path(pdb_input).stem
            output_file = output_dir / f"{pdb_base}_{variant}_ig_colored.pdb"
            if output_file.exists():
                successful_pdb_files.append(str(output_file))

        if successful_pdb_files:
            pymol_script = generate_pymol_script(output_dir, successful_pdb_files, dataset_name, df, args.normalization)

        # Generate color legend for figures
        try:
            legend_width, legend_height = map(float, args.legend_size.split(','))
            legend_figsize = (legend_width, legend_height)
        except ValueError:
            print("Warning: Invalid legend size format, using default (2,8)")
            legend_figsize = (2, 8)

        legend_path = generate_color_legend(output_dir, dataset_name, df, args.normalization, legend_figsize)

        print("\nTo visualize in PyMOL:")
        print(f"  Option 1 (automatic): @{pymol_script.name}")
        print("  Option 2 (manual):")
        print(f"    load {output_dir}/*.pdb")
        if args.normalization == "global" and df is not None:
            vmin = df["ig_normalized"].min()
            vmax = df["ig_normalized"].max()
            vmin_scaled = vmin * 100
            vmax_scaled = vmax * 100
            print(f"    spectrum b, red_white_blue, all, {vmin_scaled:.1f}, {vmax_scaled:.1f}")
        else:
            print("    spectrum b, red_white_blue, all, -100, 100")
        print("    show cartoon")
        print(f"\nColor legend saved to: {legend_path}")
        print("Use this legend image alongside structure snapshots in figures!")

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
