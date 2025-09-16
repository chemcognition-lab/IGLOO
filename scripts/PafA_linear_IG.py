# erbc_linear_esm_residue_ig_baseline.py

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Train a linear model on ERBC variant data using ESM embeddings,
# compute residue-level IG attributions using the WT variant as baseline,
# save both the IG plot and attribution DataFrame,
# and produce a sequence alignment highlighting hotspots.

from pathlib import Path

import esm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

from src.erbc_analysis.embedding import get_embeddings
from src.erbc_analysis.train_and_eval import train_and_evaluate

# Determine script directory and set paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Paths
data_path = os.path.join(project_root, 'data', 'PafA_data_processed.csv')
results_dir = os.path.join(script_dir, 'results')
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
POOLING_METHOD = 'linear'  # Options: 'mean', 'attention', 'conv_attention', 'linear'
LOSS_TYPE = 'huber'  # Options: 'mse', 'mae', 'huber', 'quantile', 'combined'

# Hyperparameters (global scope)
hidden_dim1 = 1024 # 1024
hidden_dim2 = 512 # 512
hidden_dim3 = 255 # 255
hidden_dim4 = 128 # 128
dropout_rate = 0.6
learning_rate = 1e-3
weight_decay = 1e-4  # L2 regularization
batch_size = 64  # 32 # Increased for more stable gradients
use_early_stopping = True  # Set to False to disable early stopping
patience = 30    # Early stopping patience
lr_decay_factor = 0.5
lr_decay_patience = 40
epochs = 500
final_activation = None  # 'tanh', 'sigmoid', or None


# Log-transform and z-score normalize target
print("Applying log1p transform to target...")
y_log = np.log1p(y_raw)
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
all_reps_combined = all_reps_diff  # (N, L, D)

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
for fold_idx, (train_index, val_index) in enumerate(kf.split(all_reps_combined)):
    print(f"Fold {fold_idx+1}/5")
    # Ensure WT is in training set
    if wt_idx not in train_index:
        train_index = np.append(train_index, wt_idx)
        val_index = val_index[val_index != wt_idx]
    X_train, X_val = all_reps_combined[train_index], all_reps_combined[val_index]
    y_train, y_val = y[train_index], y[val_index]
    metrics = train_and_evaluate(
        X_train, y_train, X_val, y_val, fold_idx,
        L=L,
        results_dir=results_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        hidden_dim3=hidden_dim3,
        hidden_dim4=hidden_dim4,
        dropout_rate=dropout_rate,
        use_early_stopping=use_early_stopping,
        patience=patience,
        lr_decay_factor=lr_decay_factor,
        lr_decay_patience=lr_decay_patience,
        epochs=epochs,
        final_activation=final_activation,
        device=device,
        POOLING_METHOD=POOLING_METHOD,
        LOSS_TYPE=LOSS_TYPE,
        y_log_std=y_log_std,
        y_log_mean=y_log_mean
    )
    fold_metrics.append(metrics)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

metric_names = ['mse', 'mae', 'r2', 'pearson_r', 'spearman_rho']
print("\nCross-validation results:")
for name in metric_names:
    values = [fm[name] for fm in fold_metrics]
    print(f"{name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")


# # 7. Residue-level Integrated Gradients
# def integrated_gradients_res(x, model, baseline, steps=50):
#     device = next(model.parameters()).device
#     x = torch.tensor(x, dtype=torch.float32).to(device)
#     baseline = torch.tensor(baseline, dtype=torch.float32).to(device)
#     scaled = [baseline + (i/steps)*(x - baseline) for i in range(steps+1)]
#     grads = []
#     for s in scaled:
#         s = s.clone().detach().requires_grad_(True)
#         pred = model(s)
#         pred = pred.squeeze(-1)
#         grad = torch.autograd.grad(pred, s, grad_outputs=torch.ones_like(pred), retain_graph=True)[0]
#         grads.append(grad)
#     avg_grads = torch.mean(torch.stack(grads[1:], dim=0), dim=0)
#     return (x - baseline) * avg_grads
# test_idx = 0  # choose sample index in test set
# orig_index = idx_test[test_idx]
# sample_seq = sequences[orig_index]
# var_seq = sample_seq  # for alignment plot
# sample_input = X_test[test_idx:test_idx+1]
# print(f"Computing IG for variant index {orig_index} (sequence {sample_seq})...")
# attr = integrated_gradients_res(sample_input, model, baseline)
# residue_scores = attr[0].sum(dim=-1).detach().cpu().numpy()
# df_attr = pd.DataFrame({
#     'Position': np.arange(1, L+1),
#     'Residue': list(sample_seq),
#     'IG_Attribution': residue_scores
# })
# print(df_attr.head(10))
# print(f"Saving attribution DataFrame to {df_path}...")
# df_attr.to_csv(df_path, index=False)
# print(f"Saved attribution DataFrame to {df_path}")
# print(f"Saving IG plot to {plot_path}...")
# plt.figure(figsize=(12,4))
# plt.bar(df_attr['Position'], df_attr['IG_Attribution'])
# plt.xlabel('Residue Position')
# plt.ylabel('IG Attribution vs WT')
# plt.title(f'Residue-Level IG for sample {orig_index} (vs WT)')
# plt.savefig(plot_path, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"Saved IG plot to {plot_path}")
# hotspots = set(hot_pos + hot_neg)
# plt.figure(figsize=(15, 2))
# for i, (w, v) in enumerate(zip(wt_seq, var_seq)):
#     color = 'red' if (i+1) in hotspots else 'black'
#     plt.text(i, 1,   w, ha='center', va='center', color=color, fontsize=8)
#     plt.text(i, 0.5, v, ha='center', va='center', color=color, fontsize=8)

# plt.yticks([1, 0.5], ['WT', 'Variant'])
# plt.xticks([])
# plt.xlim(-1, len(wt_seq))
# plt.ylim(0, 1.2)
# plt.title('Sequence Alignment with Hotspots Highlighted (in red)')
# plt.tight_layout()
# plt.savefig(align_plot_path, dpi=300, bbox_inches='tight')
# plt.show()
# print(f"Saved alignment plot to {align_plot_path}")