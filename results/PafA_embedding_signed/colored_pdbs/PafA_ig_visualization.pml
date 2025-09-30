# PyMOL script for visualizing PafA integrated gradients
# Generated automatically by generate_ig_colored_pdbs.py
#
# Usage: Open PyMOL and run: @PafA_ig_visualization.pml

# Clear any existing objects
delete all

# Load all PDB files
load pafa_5TJ3_Bottom_10_err0.1%_ig_colored.pdb, pafa_5TJ3_Bottom_10_err0.1pct_ig_colored
load pafa_5TJ3_Bottom_11_err4.3%_ig_colored.pdb, pafa_5TJ3_Bottom_11_err4.3pct_ig_colored
load pafa_5TJ3_Bottom_2_err9.0%_ig_colored.pdb, pafa_5TJ3_Bottom_2_err9.0pct_ig_colored
load pafa_5TJ3_Bottom_3_err1.9%_ig_colored.pdb, pafa_5TJ3_Bottom_3_err1.9pct_ig_colored
load pafa_5TJ3_Bottom_4_err5.0%_ig_colored.pdb, pafa_5TJ3_Bottom_4_err5.0pct_ig_colored
load pafa_5TJ3_Bottom_5_err5.8%_ig_colored.pdb, pafa_5TJ3_Bottom_5_err5.8pct_ig_colored
load pafa_5TJ3_Bottom_6_err3.1%_ig_colored.pdb, pafa_5TJ3_Bottom_6_err3.1pct_ig_colored
load pafa_5TJ3_Bottom_7_err5.5%_ig_colored.pdb, pafa_5TJ3_Bottom_7_err5.5pct_ig_colored
load pafa_5TJ3_Bottom_8_err5.0%_ig_colored.pdb, pafa_5TJ3_Bottom_8_err5.0pct_ig_colored
load pafa_5TJ3_Bottom_9_err2.6%_ig_colored.pdb, pafa_5TJ3_Bottom_9_err2.6pct_ig_colored
load pafa_5TJ3_Highest_Activity_ig_colored.pdb, pafa_5TJ3_Highest_Activity_ig_colored
load pafa_5TJ3_Lowest_Activity_ig_colored.pdb, pafa_5TJ3_Lowest_Activity_ig_colored
load pafa_5TJ3_Median_Activity_ig_colored.pdb, pafa_5TJ3_Median_Activity_ig_colored
load pafa_5TJ3_Top_10_err0.3%_ig_colored.pdb, pafa_5TJ3_Top_10_err0.3pct_ig_colored
load pafa_5TJ3_Top_11_err3.3%_ig_colored.pdb, pafa_5TJ3_Top_11_err3.3pct_ig_colored
load pafa_5TJ3_Top_2_err3.2%_ig_colored.pdb, pafa_5TJ3_Top_2_err3.2pct_ig_colored
load pafa_5TJ3_Top_3_err1.3%_ig_colored.pdb, pafa_5TJ3_Top_3_err1.3pct_ig_colored
load pafa_5TJ3_Top_4_err2.6%_ig_colored.pdb, pafa_5TJ3_Top_4_err2.6pct_ig_colored
load pafa_5TJ3_Top_5_err0.2%_ig_colored.pdb, pafa_5TJ3_Top_5_err0.2pct_ig_colored
load pafa_5TJ3_Top_6_err2.7%_ig_colored.pdb, pafa_5TJ3_Top_6_err2.7pct_ig_colored
load pafa_5TJ3_Top_7_err7.3%_ig_colored.pdb, pafa_5TJ3_Top_7_err7.3pct_ig_colored
load pafa_5TJ3_Top_8_err1.4%_ig_colored.pdb, pafa_5TJ3_Top_8_err1.4pct_ig_colored
load pafa_5TJ3_Top_9_err0.5%_ig_colored.pdb, pafa_5TJ3_Top_9_err0.5pct_ig_colored
load pafa_5TJ3_WT_ig_colored.pdb, pafa_5TJ3_WT_ig_colored

# Set up the visualization
hide everything
show cartoon, all

# Color by B-factor with red_white_blue spectrum (range -0.372 to 0.525)
# Red = detrimental (negative IG), Blue = beneficial (positive IG)
spectrum b, red_white_blue, all, -0.372, 0.525

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
print("RANGE  = range -0.372 to 0.525")
print("="*60)