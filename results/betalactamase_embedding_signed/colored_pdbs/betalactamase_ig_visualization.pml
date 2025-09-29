# PyMOL script for visualizing betalactamase integrated gradients
# Generated automatically by generate_ig_colored_pdbs.py
#
# Usage: Open PyMOL and run: @betalactamase_ig_visualization.pml

# Clear any existing objects
delete all

# Load all PDB files
load betalactamase_4ZJ3_Bottom_4_err0.0%_ig_colored.pdb, betalactamase_4ZJ3_Bottom_4_err0.0pct_ig_colored

# Set up the visualization
hide everything
show cartoon, all

# Color by B-factor with red_white_blue spectrum (range -1 to 1)
# Red = beneficial (positive IG), Blue = detrimental (negative IG)
spectrum b, red_white_blue, all, -1, 1

# Set cartoon transparency for better visualization
set cartoon_transparency, 0.1

# Align all structures (if multiple variants)
python
objects = cmd.get_names("objects")
if len(objects) > 1:
    for obj in objects[1:]:
        cmd.align(obj, objects[0])
python end

# Set up a nice view
bg_color white
set ray_shadows, 0
set depth_cue, 0
set antialias, 2

# Center and zoom
zoom all
center all

print("\n" + "="*60)
print("INTEGRATED GRADIENTS COLOR SCALE:")
print("="*60)
print("RED    = Beneficial effects (positive IG values)")
print("WHITE  = Neutral (zero IG values)")
print("BLUE   = Detrimental effects (negative IG values)")
print("="*60)