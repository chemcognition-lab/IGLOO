# betalactamase_linear_IG.py

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Train a model on beta-lactamase variant data using ESM embeddings,
# compute residue-level IG attributions, perform mutagenesis analysis,
# and sequence-level attribution analysis.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

# Import analysis modules
from src.erbc_analysis.data_processing import CustomDataset, batch_pad_and_normalize
from src.erbc_analysis.embedding import get_embeddings
from src.erbc_analysis.mutagenesis import (
    analyze_position_preferences,
    compute_mutation_effects,
    export_top_mutations,
    plot_mutation_landscape,
)
from src.erbc_analysis.sequence_ig import run_complete_sequence_analysis
from src.erbc_analysis.train_and_eval import train

# Determine script directory and set paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Paths
train_data_path = os.path.join(project_root, 'data', 'betalactamase_train.csv')
test_data_path = os.path.join(project_root, 'data', 'betalactamase_test.csv')
results_dir = os.path.join(project_root, 'results', 'betalactamase')
plot_path = os.path.join(results_dir, 'betalactamase_residue_ig_vs_WT.png')
df_path = os.path.join(results_dir, 'betalactamase_residue_ig_attributions.csv')

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# 1. Load datasets
print(f"Loading training data from {train_data_path}")
train_df = pd.read_csv(train_data_path)
print(f"Loading test data from {test_data_path}")
test_df = pd.read_csv(test_data_path)

# Combine both datasets for comprehensive analysis (like PafA approach)
df = pd.concat([train_df, test_df], ignore_index=True)

col = 'quantitative_function'
# Remove rows with NaN or Inf in target
df = df[df[col].notna()]
df = df[np.isfinite(df[col])]
sequences = df['SEQ'].astype(str).tolist()
y_raw = df[col].values.astype(np.float32)

print(f"Total dataset size: {len(sequences)} sequences")
print(f"Training set size: {len(train_df)} sequences")
print(f"Test set size: {len(test_df)} sequences")

# Config: select ESM layer and pooling method
ESM_LAYER = 33  # Change this to try other layers
POOLING_METHOD = 'esm_attention'  # Options: 'mean', 'attention', 'conv_attention', 'linear', 'esm_attention'
LOSS_TYPE = 'huber'  # Options: 'mse', 'mae', 'huber', 'quantile', 'combined'

# Hyperparameters (global scope)
hidden_dim1 = 1024
hidden_dim2 = 512
hidden_dim3 = 255
hidden_dim4 = 128
dropout_rate = 0.5
learning_rate = 5e-4  # Increased from 1e-4 for faster training
weight_decay = 1e-4  # L2 regularization
batch_size = 256  # Increased for more stable gradients
use_early_stopping = True  # Set to False to disable early stopping
patience = 15    # Reduced from 30 for faster training
lr_decay_factor = 0.5
lr_decay_patience = 20  # Reduced from 40
epochs = 200  # Reduced from 500
final_activation = None  # 'tanh', 'sigmoid', or None

# No normalization needed for beta-lactamase dataset
print("Using raw target values without normalization...")
y = y_raw  # Use raw values directly
y_log_mean = 0.0  # No normalization
y_log_std = 1.0   # No normalization
print(f"Target range: {np.min(y):.4f} to {np.max(y):.4f}")

# Plot distribution of target values
plt.figure(figsize=(8,4))
sns.histplot(y, bins=30, kde=True)
plt.xlabel('Target Value (Raw)')
plt.ylabel('Count')
plt.title('Distribution of Target Values')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'target_distribution.png'), bbox_inches='tight', dpi=200)
plt.close()

# Use the specific WT sequence provided
wt_seq = ("MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAG"
          "QEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELN"
          "EAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDG"
          "KPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW")
print(f"Using specified WT sequence as baseline: {wt_seq[:50]}...")
print(f"WT sequence length: {len(wt_seq)}")

# Check if WT sequence is in the dataset
wt_in_dataset = wt_seq in sequences
print(f"WT sequence found in dataset: {wt_in_dataset}")
if wt_in_dataset:
    wt_idx = sequences.index(wt_seq)
    dataset_wt_value = y_raw[wt_idx]
    print(f"WT functional value from dataset: {dataset_wt_value:.4f}")
    print("Note: Using specified WT functional value of 1.0 instead of dataset value")

# Set WT functional value to 1.0 as specified (raw, unnormalized value)
wt_functional_value = 1.0
print(f"Using WT functional value: {wt_functional_value:.4f}")

# 2. Tokenize & embed with ESM (directly, configurable layer/pooling)
print("Computing ESM embeddings...")
all_reps = get_embeddings(sequences, layer=ESM_LAYER, batch_size=16, dataset_name='betalactamase_combined')
N, L, D = all_reps.shape

# Embed WT for baseline
wt_reps = get_embeddings([wt_seq], layer=ESM_LAYER, batch_size=1, dataset_name='betalactamase_wt')  # (1, L, D)
baseline = wt_reps

# Use raw embeddings (not difference embeddings for beta-lactamase)
all_reps_combined = all_reps  # (N, L, D)

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 5-fold cross-validation (commented out for speed - uncomment if needed)
print("Skipping cross-validation for speed...")
# orig_idx = np.arange(N)
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# fold_metrics = []

# Train a final model on all data for analysis
print("\nTraining final model on all data for variant analysis...")
input_dim_combined = all_reps_combined.shape[2]
target_min = np.min(y)
target_max = np.max(y)

final_model, train_metrics = train(
    all_reps_combined, y,
    L=L,
    batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    hidden_dim1=hidden_dim1,
    hidden_dim2=hidden_dim2,
    hidden_dim3=hidden_dim3,
    hidden_dim4=hidden_dim4,
    dropout_rate=dropout_rate,
    epochs=50,  # Fewer epochs since we're using all data
    final_activation=final_activation,
    device=device,
    POOLING_METHOD=POOLING_METHOD,
    LOSS_TYPE=LOSS_TYPE,
    y_log_std=y_log_std,
    y_log_mean=y_log_mean
)
print("Final model training completed.")
print("Training set metrics:")
for k, v in train_metrics.items():
    print(f"  {k}: {v:.4f}")

# --- Predicted vs Actual Scatter Plot with Performance Metrics ---
# Get predictions on all data using DataLoader (to match train function)
final_model.eval()
pred_norm = []
y_true_norm = []
eval_dataset = CustomDataset(all_reps_combined, y)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_pad_and_normalize)
with torch.no_grad():
    for xb, lengths, yb in eval_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if POOLING_METHOD == 'esm_attention':
            input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
            for i, length in enumerate(lengths):
                input_mask[i, :length] = 1
            out = final_model(xb, input_mask).squeeze(-1).cpu().numpy()
        else:
            out = final_model(xb).squeeze(-1).cpu().numpy()
        pred_norm.append(out)
        y_true_norm.append(yb.squeeze(-1).cpu().numpy())
pred_norm = np.concatenate(pred_norm)
y_true_norm = np.concatenate(y_true_norm)
# Since we're not normalizing, predictions and targets are already in original scale
y_pred = pred_norm
y_true = y_true_norm

# Print a sample of predictions and targets
print("Sample predictions vs targets:")
for i in range(min(10, len(y_pred))):
    print(f"  Pred: {y_pred[i]:.3f}  Target: {y_true[i]:.3f}")

# Compute metrics
r2 = r2_score(y_true, y_pred)
pearson_r, _ = stats.pearsonr(y_true, y_pred)
spearman_rho, _ = stats.spearmanr(y_true, y_pred)

# Plot
plt.figure(figsize=(7,7))
sns.scatterplot(x=y_true, y=y_pred, s=40, alpha=0.7)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Final Model: Predicted vs Actual (Beta-lactamase)')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Ideal')

# Annotate metrics in top left
metrics_text = f"$R^2$ = {r2:.3f}\nPearson r = {pearson_r:.3f}\nSpearman $\\rho$ = {spearman_rho:.3f}"
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.legend()
plt.tight_layout()
scatter_path = os.path.join(results_dir, 'betalactamase_pred_vs_actual_final.png')
plt.savefig(scatter_path, dpi=200)
plt.close()
print(f"Saved predicted vs actual scatter plot to {scatter_path}")

# Perform in-silico mutagenesis analysis
print("\n" + "="*60)
print("PERFORMING IN-SILICO MUTAGENESIS ANALYSIS")
print("="*60)
print(f"Using WT functional value as baseline: {wt_functional_value:.4f}")

mutation_df = compute_mutation_effects(
    model=final_model,
    wt_sequence=wt_seq,
    device=device,
    esm_layer=ESM_LAYER,
    y_log_mean=y_log_mean,
    y_log_std=y_log_std,
    pooling_method=POOLING_METHOD,
    batch_size=16,
    progress=True
)

# Analyze results
analysis = analyze_position_preferences(mutation_df, top_n=10)

print("\nMUTAGENESIS SUMMARY:")
for key, value in analysis['summary'].items():
    print(f"  {key}: {value}")

print("\nTOP 10 BENEFICIAL MUTATIONS:")
print(analysis['top_mutations'][['mutation', 'predicted_activity', 'effect_vs_wt', 'fold_change']].head(10).round(3))

# Create mutation landscape plot
mutagenesis_plot_path = os.path.join(results_dir, 'betalactamase_mutation_landscape.png')
plot_mutation_landscape(
    mutation_df=mutation_df,
    wt_sequence=wt_seq,
    save_path=mutagenesis_plot_path,
    figsize=(20, 12),
    top_n_label=15
)

# Export mutagenesis results
mutations_csv_path = os.path.join(results_dir, 'betalactamase_all_mutations.csv')
mutation_df.to_csv(mutations_csv_path, index=False)

top_mutations_path = os.path.join(results_dir, 'betalactamase_top_mutations.csv')
export_top_mutations(
    mutation_df=mutation_df,
    output_path=top_mutations_path,
    top_n=50,
    include_negative=True
)

# Perform advanced attribution analysis on a few interesting variants
print("\n" + "="*60)
print("PERFORMING SEQUENCE-LEVEL ATTRIBUTION ANALYSIS")
print("="*60)

# Analyze top beneficial variants using sequence-level IG
top_variants = mutation_df.head(3)  # Analyze top 3 for speed
for _idx, variant in top_variants.iterrows():
    print(f"\nAnalyzing variant: {variant['mutation']}")

    # Reconstruct the mutant sequence
    pos = variant['position'] - 1  # Convert to 0-based
    mut_seq = wt_seq[:pos] + variant['mut_aa'] + wt_seq[pos+1:]

    # Create output directory for this variant
    variant_dir = os.path.join(results_dir, f"sequence_analysis_{variant['mutation']}")
    os.makedirs(variant_dir, exist_ok=True)

    # Run complete sequence-level analysis
    attributions_df, position_stats = run_complete_sequence_analysis(
        predictor_model=final_model,
        sequence=mut_seq,
        baseline_sequence=wt_seq,
        output_dir=variant_dir,
        esm_model_name='esm2_t33_650M_UR50D',
        esm_layer=ESM_LAYER,
        pooling_method=POOLING_METHOD,
        y_log_mean=y_log_mean,
        y_log_std=y_log_std,
        device=device
    )

    print(f"  Results saved to: {variant_dir}")
    print(f"  Most sensitive positions: {position_stats.head(5).index.tolist()}")

    # Show top beneficial mutations from this analysis
    beneficial_muts = attributions_df[
        (attributions_df['is_mutation']) & (attributions_df['attribution'] > 0)
    ].sort_values('attribution', ascending=False).head(5)

    print("  Top 5 beneficial mutations from this variant:")
    for _, mut in beneficial_muts.iterrows():
        print(f"    {mut['mutation']}: {mut['attribution']:.4f}")

# Also run sequence-level analysis on the wild-type for comparison
print("\nAnalyzing wild-type sequence for comparison...")
wt_analysis_dir = os.path.join(results_dir, "sequence_analysis_WT")
os.makedirs(wt_analysis_dir, exist_ok=True)

wt_attributions_df, wt_position_stats = run_complete_sequence_analysis(
    predictor_model=final_model,
    sequence=wt_seq,
    baseline_sequence=wt_seq,  # Self-comparison
    output_dir=wt_analysis_dir,
    esm_model_name='esm2_t33_650M_UR50D',
    esm_layer=ESM_LAYER,
    pooling_method=POOLING_METHOD,
    y_log_mean=y_log_mean,
    y_log_std=y_log_std,
    device=device
)

print(f"Wild-type analysis saved to: {wt_analysis_dir}")

# Additional analysis: Compare with training vs test performance
print("\n" + "="*50)
print("TRAINING VS TEST SET ANALYSIS")
print("="*50)

# Separate predictions for training and test sets
train_size = len(train_df)
train_pred = y_pred[:train_size]
train_actual = y_true[:train_size]
test_pred = y_pred[train_size:train_size+len(test_df)]
test_actual = y_true[train_size:train_size+len(test_df)]

# Calculate separate metrics
train_r2 = r2_score(train_actual, train_pred)
train_pearson, _ = stats.pearsonr(train_actual, train_pred)
test_r2 = r2_score(test_actual, test_pred)
test_pearson, _ = stats.pearsonr(test_actual, test_pred)

print("Training set performance:")
print(f"  R² = {train_r2:.4f}")
print(f"  Pearson r = {train_pearson:.4f}")
print("Test set performance:")
print(f"  R² = {test_r2:.4f}")
print(f"  Pearson r = {test_pearson:.4f}")

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training set
axes[0].scatter(train_actual, train_pred, alpha=0.6, s=30)
axes[0].plot([train_actual.min(), train_actual.max()],
             [train_actual.min(), train_actual.max()], 'k--', lw=2)
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title(f'Training Set (R² = {train_r2:.3f})')
axes[0].grid(True, alpha=0.3)

# Test set
axes[1].scatter(test_actual, test_pred, alpha=0.6, s=30, color='orange')
axes[1].plot([test_actual.min(), test_actual.max()],
             [test_actual.min(), test_actual.max()], 'k--', lw=2)
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
axes[1].set_title(f'Test Set (R² = {test_r2:.3f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'train_vs_test_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"Results saved to: {results_dir}")
print("Generated files:")
print("  - Prediction plots: betalactamase_pred_vs_actual_final.png, train_vs_test_comparison.png")
print("  - Mutation landscape: betalactamase_mutation_landscape.png")
print("  - All mutations: betalactamase_all_mutations.csv")
print("  - Top mutations: betalactamase_top_mutations.csv")
print("  - Sequence-level analyses: sequence_analysis_*/")
print("    * amino_acid_attribution_heatmap.png - Heatmap of all AA substitutions")
print("    * full_amino_acid_attributions.csv - Complete attribution matrix")
print("    * position_sensitivity_analysis.csv - Position-level sensitivity metrics")
print("    * top_beneficial_mutations.csv - Best mutations from sequence analysis")
print("    * analysis_summary.txt - Human-readable summary")
print("\nKey insights:")
print("1. Check 'betalactamase_top_mutations.csv' for the best mutations to try experimentally")
print("2. Review 'sequence_analysis_*/position_sensitivity_analysis.csv' for most mutable positions")
print("3. Use 'sequence_analysis_*/amino_acid_attribution_heatmap.png' to see AA preferences")
print("4. Compare different variants in their respective sequence_analysis_*/ folders")
print("5. Training vs test performance comparison shows generalization ability")

# Save the model and parameters
model_path = os.path.join(results_dir, 'betalactamase_model.pth')
torch.save({
    'model_state_dict': final_model.state_dict(),
    'model_config': {
        'input_dim': D,
        'L': L,
        'hidden_dim1': hidden_dim1,
        'hidden_dim2': hidden_dim2,
        'hidden_dim3': hidden_dim3,
        'hidden_dim4': hidden_dim4,
        'dropout_rate': dropout_rate,
        'final_activation': final_activation,
        'pooling_method': POOLING_METHOD
    },
    'normalization': {
        'y_log_mean': y_log_mean,
        'y_log_std': y_log_std
    },
    'performance': {
        'overall_r2': r2,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
}, model_path)
print("  - betalactamase_model.pth: Trained model with metadata")
