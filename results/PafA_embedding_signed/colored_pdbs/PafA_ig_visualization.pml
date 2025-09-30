# PyMOL script for visualizing PafA integrated gradients
# Generated automatically by generate_ig_colored_pdbs.py
#
# Usage: Open PyMOL and run: @PafA_ig_visualization.pml

# Clear any existing objects
delete all

# Load all PDB files
load pafa_5TJ3_Top_3_err1.3%_ig_colored.pdb, pafa_5TJ3_Top_3_err1.3pct_ig_colored

# Set up the visualization
hide everything
show cartoon, all

# Color by B-factor with red_white_blue spectrum (range -100 to 100 (per-variant scale, x100))
# Red = detrimental (negative IG), Blue = beneficial (positive IG)

# Step 1: Color everything white as baseline
color white, all

# Step 2: Create selections for different IG ranges (scaled x100, adjusted for small values)
select strong_negative, (b < -2)
select weak_negative, (b > -2) and (b < -0.5)
select near_zero, (b > -0.5) and (b < 0.5)
select weak_positive, (b > 0.5) and (b < 2)
select strong_positive, (b > 2)

# Step 3: Apply colors manually for better control
color red, strong_negative
color salmon, weak_negative
color white, near_zero
color lightblue, weak_positive
color blue, strong_positive

# Step 4: Apply spectrum only to non-zero values for fine-tuning
select non_zero, (b < -0.5) or (b > 0.5)
spectrum b, red_white_blue, non_zero, -3, 3

# Deselect all for clean view
deselect

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
print("RED    = Detrimental effects (negative IG values)")
print("WHITE  = Neutral (zero IG values)")
print("BLUE   = Beneficial effects (positive IG values)")
print("RANGE  = range -100 to 100 (per-variant scale, x100)")
print("="*60)