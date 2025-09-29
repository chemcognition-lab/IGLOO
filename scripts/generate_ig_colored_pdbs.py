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

import pandas as pd
import yaml
from Bio import PDB

# Add the src directory to the path to import igloo modules
sys.path.append(str(Path(__file__).parent.parent / "src"))


# -----------------------------
# Load CSV and normalize IG data
# -----------------------------
def load_and_normalize_ig(csv_path):
    df = pd.read_csv(csv_path)

    # Normalize per variant (ensures each variant spans [-1, 1])
    df["ig_normalized"] = df.groupby("sequence_name")["integrated_gradient"].transform(
        lambda x: x / x.abs().max() if x.abs().max() > 0 else x
    )
    return df


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def find_pdb_file(dataset_name, pdb_dir="data/PDBs"):
    """Find the appropriate PDB file for the dataset."""
    pdb_patterns = {
        'betalactamase': ['betalactamase_4ZJ3.pdb', '4ZJ3.pdb', 'betalactamase.pdb'],
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


def load_ig_data(results_dir):
    """Load integrated gradients data from the detailed CSV file."""
    csv_files = list(Path(results_dir).glob("*amino_acid_effects_detailed.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No detailed amino acid effects CSV found in {results_dir}")

    csv_file = csv_files[0]  # Use the first one found
    print(f"Loading IG data from: {csv_file}")
    return load_and_normalize_ig(csv_file)


# -----------------------------
# Write modified PDB
# -----------------------------
def write_pdb_with_ig(variant_name, df, pdb_in, pdb_out, chain_id="A"):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_in))

    # Subset for this variant
    subset = df[df["sequence_name"] == variant_name]
    pos_to_ig = dict(zip(subset["position"], subset["ig_normalized"]))

    modified_residues = 0
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                res_id = residue.get_id()[1]  # residue number
                if res_id in pos_to_ig:
                    ig_val = float(pos_to_ig[res_id])
                    for atom in residue:
                        atom.set_bfactor(ig_val)
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
def generate_pymol_script(output_dir, pdb_files, dataset_name):
    """Generate a PyMOL script to visualize PDB files with red_white_blue coloring."""
    script_path = output_dir / f"{dataset_name}_ig_visualization.pml"

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
        "# Color by B-factor with red_white_blue spectrum (range -1 to 1)",
        "# Red = beneficial (positive IG), Blue = detrimental (negative IG)",
        "spectrum b, red_white_blue, all, -1, 1",
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
        "print(\"RED    = Beneficial effects (positive IG values)\")",
        "print(\"WHITE  = Neutral (zero IG values)\")",
        "print(\"BLUE   = Detrimental effects (negative IG values)\")",
        "print(\"=\"*60)"
    ])

    script_content = "\n".join(script_lines)

    with open(script_path, 'w') as f:
        f.write(script_content)

    print(f"Generated PyMOL script: {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDB files colored by integrated gradients values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate PDBs for all variants found in results
    python generate_ig_colored_pdbs.py configs/betalactamase_config.yaml

    # Generate PDBs for specific variants only
    python generate_ig_colored_pdbs.py configs/betalactamase_config.yaml --variants "Highest_Activity,Lowest_Activity"

    # Specify custom output directory
    python generate_ig_colored_pdbs.py configs/betalactamase_config.yaml --output-dir custom_pdbs

    # List available variants
    python generate_ig_colored_pdbs.py configs/betalactamase_config.yaml --list-variants
        """
    )

    parser.add_argument("config", help="Path to the configuration YAML file")
    parser.add_argument("--variants", help="Comma-separated list of variants to process (default: all)")
    parser.add_argument("--output-dir", help="Output directory for colored PDB files")
    parser.add_argument("--pdb-file", help="Specific PDB file to use (overrides auto-detection)")
    parser.add_argument("--chain-id", default="A", help="Chain ID to modify (default: A)")

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
        df = load_ig_data(results_dir)
    except Exception as e:
        print(f"Error loading IG data: {e}")
        return 1

    # List available variants if requested
    if args.list_variants:
        variants = get_available_variants(df)
        print(f"\nAvailable variants ({len(variants)}):")
        for variant in variants:
            count = len(df[df['sequence_name'] == variant])
            print(f"  - {variant} ({count} residues)")
        return 0

    print("Applied per-variant normalization to IG values (range [-1, 1])")

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
            pymol_script = generate_pymol_script(output_dir, successful_pdb_files, dataset_name)

        print("\nTo visualize in PyMOL:")
        print(f"  Option 1 (automatic): @{pymol_script.name}")
        print("  Option 2 (manual):")
        print(f"    load {output_dir}/*.pdb")
        print("    spectrum b, red_white_blue, all, -1, 1")
        print("    show cartoon")

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
