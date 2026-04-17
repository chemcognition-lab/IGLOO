# PyMOL script for visualizing betalactamase integrated gradients
# Generated automatically by generate_ig_colored_pdbs.py
#
# Usage: Open PyMOL and run: @betalactamase_ig_visualization.pml

# Clear any existing objects
delete all

# Load all PDB files
load betalactamase_AF3_Bottom_10_err0.0%_ig_colored.pdb, betalactamase_AF3_Bottom_10_err0.0pct_ig_colored
load betalactamase_AF3_Bottom_11_err0.0%_ig_colored.pdb, betalactamase_AF3_Bottom_11_err0.0pct_ig_colored
load betalactamase_AF3_Bottom_2_err0.0%_ig_colored.pdb, betalactamase_AF3_Bottom_2_err0.0pct_ig_colored
load betalactamase_AF3_Bottom_3_err0.0%_ig_colored.pdb, betalactamase_AF3_Bottom_3_err0.0pct_ig_colored
load betalactamase_AF3_Bottom_4_err0.0%_ig_colored.pdb, betalactamase_AF3_Bottom_4_err0.0pct_ig_colored
load betalactamase_AF3_Bottom_5_err0.0%_ig_colored.pdb, betalactamase_AF3_Bottom_5_err0.0pct_ig_colored
load betalactamase_AF3_Bottom_6_err0.0%_ig_colored.pdb, betalactamase_AF3_Bottom_6_err0.0pct_ig_colored
load betalactamase_AF3_Bottom_7_err0.0%_ig_colored.pdb, betalactamase_AF3_Bottom_7_err0.0pct_ig_colored
load betalactamase_AF3_Bottom_8_err0.0%_ig_colored.pdb, betalactamase_AF3_Bottom_8_err0.0pct_ig_colored
load betalactamase_AF3_Bottom_9_err0.0%_ig_colored.pdb, betalactamase_AF3_Bottom_9_err0.0pct_ig_colored
load betalactamase_AF3_Highest_Activity_ig_colored.pdb, betalactamase_AF3_Highest_Activity_ig_colored
load betalactamase_AF3_Lowest_Activity_ig_colored.pdb, betalactamase_AF3_Lowest_Activity_ig_colored
load betalactamase_AF3_Median_Activity_ig_colored.pdb, betalactamase_AF3_Median_Activity_ig_colored
load betalactamase_AF3_Top_10_err6.6%_ig_colored.pdb, betalactamase_AF3_Top_10_err6.6pct_ig_colored
load betalactamase_AF3_Top_11_err8.3%_ig_colored.pdb, betalactamase_AF3_Top_11_err8.3pct_ig_colored
load betalactamase_AF3_Top_2_err3.6%_ig_colored.pdb, betalactamase_AF3_Top_2_err3.6pct_ig_colored
load betalactamase_AF3_Top_3_err6.6%_ig_colored.pdb, betalactamase_AF3_Top_3_err6.6pct_ig_colored
load betalactamase_AF3_Top_4_err9.4%_ig_colored.pdb, betalactamase_AF3_Top_4_err9.4pct_ig_colored
load betalactamase_AF3_Top_5_err5.4%_ig_colored.pdb, betalactamase_AF3_Top_5_err5.4pct_ig_colored
load betalactamase_AF3_Top_6_err9.6%_ig_colored.pdb, betalactamase_AF3_Top_6_err9.6pct_ig_colored
load betalactamase_AF3_Top_7_err7.8%_ig_colored.pdb, betalactamase_AF3_Top_7_err7.8pct_ig_colored
load betalactamase_AF3_Top_8_err1.6%_ig_colored.pdb, betalactamase_AF3_Top_8_err1.6pct_ig_colored
load betalactamase_AF3_Top_9_err5.7%_ig_colored.pdb, betalactamase_AF3_Top_9_err5.7pct_ig_colored
load betalactamase_AF3_WT_ig_colored.pdb, betalactamase_AF3_WT_ig_colored

# Set up the visualization
hide everything
show cartoon, all

# Color by B-factor with red_white_blue spectrum (range -0.674 to 0.058 (global scale - matches heatmap, scaled x100 in PDB))
# Red = detrimental (negative IG), Blue = beneficial (positive IG)

# Step 1: Color everything white as baseline
color white, all

# Step 2: Create selections for different IG ranges (scaled x100)
select strong_negative, (b < -10)
select weak_negative, (b > -10) and (b < -2)
select near_zero, (b > -2) and (b < 2)
select weak_positive, (b > 2) and (b < 10)
select strong_positive, (b > 10)

# Step 3: Apply colors manually for better control
color red, strong_negative
color salmon, weak_negative
color white, near_zero
color lightblue, weak_positive
color blue, strong_positive

# Step 4: Apply spectrum only to non-zero values for fine-tuning
select non_zero, (b < -2) or (b > 2)
spectrum b, red_white_blue, non_zero, -100, 100

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
print("RANGE  = range -0.674 to 0.058 (global scale - matches heatmap, scaled x100 in PDB)")
print("="*60)