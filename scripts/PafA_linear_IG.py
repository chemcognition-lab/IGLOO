# erbc_linear_esm_residue_ig_baseline.py

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Train a model on PafA variant data using ESM embeddings,
# compute residue-level IG attributions using the WT variant as baseline,
# save both the IG plot and attribution DataFrame,
# and produce a sequence alignment highlighting hotspots.


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
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
from src.erbc_analysis.train_and_eval import train, train_and_evaluate  # noqa: F401

# Determine script directory and set paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Paths
data_path = os.path.join(project_root, 'data', 'PafA_data_processed.csv')
results_dir = os.path.join(project_root, 'results', 'PafA')
plot_path = os.path.join(results_dir, 'PafA_residue_ig_vs_WT.png')
df_path = os.path.join(results_dir, 'PafA_residue_ig_attributions.csv')
align_plot_path = os.path.join(results_dir, 'PafA_alignment_hotspots.png')

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)


# 1. Load dataset
print(f"Loading data from {data_path}")
df = pd.read_csv(data_path)
col = 'kcat_cMUP'
# Remove rows with NaN or Inf in target
df = df[df[col].notna()]
df = df[np.isfinite(df[col])]
sequences = df['SEQ'].astype(str).tolist()
y_raw = df[col].values.astype(np.float32)

# Config: select ESM layer and pooling method
ESM_LAYER = 33  # Change this to try other layers
POOLING_METHOD = 'esm_attention'  # Options: 'mean', 'attention', 'conv_attention', 'linear', 'esm_attention'
LOSS_TYPE = 'huber'  # Options: 'mse', 'mae', 'huber', 'quantile', 'combined'

# Hyperparameters (global scope)
hidden_dim1 = 1024 # 1024
hidden_dim2 = 512 # 512
hidden_dim3 = 255 # 255
hidden_dim4 = 128 # 128
dropout_rate = 0.5
learning_rate = 5e-4  # Increased from 1e-4 for faster training
weight_decay = 1e-4  # L2 regularization
batch_size = 256  # 32 # Increased for more stable gradients
use_early_stopping = True  # Set to False to disable early stopping
patience = 15    # Reduced from 30 for faster training
lr_decay_factor = 0.5
lr_decay_patience = 20  # Reduced from 40
epochs = 200  # Reduced from 500
final_activation = None  # 'tanh', 'sigmoid', or None


# Log-transform and z-score normalize target
print("Applying log1p transform to target...")
y_log = y_raw # np.log1p(y_raw)
y_log_mean = np.mean(y_log)
y_log_std = np.std(y_log)
y = (y_log - y_log_mean) / y_log_std
print(f"Target log1p mean: {y_log_mean:.4f}, std: {y_log_std:.4f}")

# Plot distribution of normalized target values
plt.figure(figsize=(8,4))
sns.histplot(y, bins=30, kde=True)
plt.xlabel('Normalized Target Value')
plt.ylabel('Count')
plt.title('Distribution of Normalized Target Values')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'target_distribution_normalized.png'), bbox_inches='tight', dpi=200)
# plt.show()

# Plot distribution of raw target values
plt.figure(figsize=(8,4))
sns.histplot(y_raw, bins=30, kde=True)
plt.xlabel('Raw Target Value')
plt.ylabel('Count')
plt.title('Distribution of Raw Target Values')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'target_distribution.png'), bbox_inches='tight', dpi=200)
# plt.show()


# y is now log-transformed and normalized

# Identify WT sequence for baseline
wt_seq = df['SEQ'].iloc[-1]
print(f"Using WT sequence as baseline: {wt_seq}")


# 2. Tokenize & embed with ESM (directly, configurable layer/pooling)

all_reps = get_embeddings(sequences, layer=ESM_LAYER, batch_size=16)
N, L, D = all_reps.shape

# Embed WT for baseline
data_wt = [('WT', wt_seq)]
wt_reps = get_embeddings([wt_seq], layer=ESM_LAYER, batch_size=1)  # (1, L, D)
baseline = wt_reps

# Compute difference embeddings: X_variant - X_WT

# Compute difference embeddings: X_variant - X_WT
all_reps_diff = all_reps - wt_reps[0]  # (N, L, D)
all_reps_combined = all_reps #all_reps_diff  # (N, L, D)

# 3. Train-test split, preserving original indices

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 5-fold cross-validation
print("Running 5-fold cross-validation...")
orig_idx = np.arange(N)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

wt_idx = len(sequences) - 1  # WT is last in the list
# for fold_idx, (train_index, val_index) in enumerate(kf.split(all_reps_combined)):
#     print(f"Fold {fold_idx+1}/5")
#     # Ensure WT is in training set
#     if wt_idx not in train_index:
#         train_index = np.append(train_index, wt_idx)
#         val_index = val_index[val_index != wt_idx]
#     X_train, X_val = all_reps_combined[train_index], all_reps_combined[val_index]
#     y_train, y_val = y[train_index], y[val_index]
#     metrics = train_and_evaluate(
#         X_train, y_train, X_val, y_val, fold_idx,
#         L=L,
#         results_dir=results_dir,
#         batch_size=batch_size,
#         learning_rate=learning_rate,
#         weight_decay=weight_decay,
#         hidden_dim1=hidden_dim1,
#         hidden_dim2=hidden_dim2,
#         hidden_dim3=hidden_dim3,
#         hidden_dim4=hidden_dim4,
#         dropout_rate=dropout_rate,
#         use_early_stopping=use_early_stopping,
#         patience=patience,
#         lr_decay_factor=lr_decay_factor,
#         lr_decay_patience=lr_decay_patience,
#         epochs=epochs,
#         final_activation=final_activation,
#         device=device,
#         POOLING_METHOD=POOLING_METHOD,
#         LOSS_TYPE=LOSS_TYPE,
#         y_log_std=y_log_std,
#         y_log_mean=y_log_mean
#     )
#     fold_metrics.append(metrics)
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

# metric_names = ['mse', 'mae', 'r2', 'pearson_r', 'spearman_rho']
# print("\nCross-validation results:")
# for name in metric_names:
#     values = [fm[name] for fm in fold_metrics]
#     print(f"{name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")




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
y_pred_unnorm = pred_norm * y_log_std + y_log_mean
y_true = y_true_norm * y_log_std + y_log_mean

# Print a sample of predictions and targets for debugging
print("Sample predictions vs targets (unnormalized):")
for i in range(min(10, len(y_pred_unnorm))):
    print(f"  Pred: {y_pred_unnorm[i]:.3f}  Target: {y_true[i]:.3f}")

# Compute metrics
r2 = r2_score(y_true, y_pred_unnorm)
pearson_r, _ = stats.pearsonr(y_true, y_pred_unnorm)
spearman_rho, _ = stats.spearmanr(y_true, y_pred_unnorm)

# Plot
plt.figure(figsize=(7,7))
sns.scatterplot(x=y_true, y=y_pred_unnorm, s=40, alpha=0.7)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Final Model: Predicted vs Actual')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Ideal')

# Annotate metrics in top left
metrics_text = f"$R^2$ = {r2:.3f}\nPearson r = {pearson_r:.3f}\nSpearman $\\rho$ = {spearman_rho:.3f}"
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.legend()
plt.tight_layout()
scatter_path = os.path.join(results_dir, 'PafA_pred_vs_actual_final.png')
plt.savefig(scatter_path, dpi=200)
plt.close()
print(f"Saved predicted vs actual scatter plot to {scatter_path}")

# Perform in-silico mutagenesis analysis
print("\n" + "="*60)
print("PERFORMING IN-SILICO MUTAGENESIS ANALYSIS")
print("="*60)

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
mutagenesis_plot_path = os.path.join(results_dir, 'PafA_mutation_landscape.png')
plot_mutation_landscape(
    mutation_df=mutation_df,
    wt_sequence=wt_seq,
    save_path=mutagenesis_plot_path,
    figsize=(20, 12),
    top_n_label=15
)

# Export mutagenesis results
mutations_csv_path = os.path.join(results_dir, 'PafA_all_mutations.csv')
mutation_df.to_csv(mutations_csv_path, index=False)

top_mutations_path = os.path.join(results_dir, 'PafA_top_mutations.csv')
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
    device=device
)

print(f"Wild-type analysis saved to: {wt_analysis_dir}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"Results saved to: {results_dir}")
print("Generated files:")
print("  - Cross-validation plots: PafA_training_loss_curve_fold*.png, PafA_pred_vs_actual_fold*.png")
print("  - Mutation landscape: PafA_mutation_landscape.png")
print("  - All mutations: PafA_all_mutations.csv")
print("  - Top mutations: PafA_top_mutations.csv")
print("  - Sequence-level analyses: sequence_analysis_*/")
print("    * amino_acid_attribution_heatmap.png - Heatmap of all AA substitutions")
print("    * full_amino_acid_attributions.csv - Complete attribution matrix")
print("    * position_sensitivity_analysis.csv - Position-level sensitivity metrics")
print("    * top_beneficial_mutations.csv - Best mutations from sequence analysis")
print("    * analysis_summary.txt - Human-readable summary")
print("\nKey insights:")
print("1. Check 'PafA_top_mutations.csv' for the best mutations to try experimentally")
print("2. Review 'sequence_analysis_*/position_sensitivity_analysis.csv' for most mutable positions")
print("3. Use 'sequence_analysis_*/amino_acid_attribution_heatmap.png' to see AA preferences")
print("4. Compare different variants in their respective sequence_analysis_*/ folders")
