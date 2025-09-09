# erbc_linear_esm_residue_ig_baseline.py

# Add project root to sys.path for src imports
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Train a linear model on ERBC variant data using ESM embeddings,
# compute residue-level IG attributions using the WT variant as baseline,
# save both the IG plot and attribution DataFrame,
# and produce a sequence alignment highlighting hotspots.

import hashlib
from pathlib import Path

import esm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.erbc_analysis.models import AttentionConvPoolingModel, MeanPoolingModel

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
POOLING_METHOD = 'attention'  # Options: 'mean', 'attention', 'conv_attention'


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
plt.show()

# Plot distribution of raw target values
plt.figure(figsize=(8,4))
sns.histplot(y_raw, bins=30, kde=True)
plt.xlabel('Raw Target Value')
plt.ylabel('Count')
plt.title('Distribution of Raw Target Values')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'target_distribution.png'), bbox_inches='tight', dpi=200)
plt.show()


# y is now log-transformed and normalized

# Identify WT sequence for baseline
wt_seq = df['SEQ'].iloc[-1]
print(f"Using WT sequence as baseline: {wt_seq}")


# 2. Tokenize & embed with ESM (directly, configurable layer/pooling)

def get_embeddings(sequences, layer=ESM_LAYER, batch_size=16):
    embeddings = []
    emb_dir = Path(project_root) / 'data' / 'embeddings'
    emb_dir.mkdir(parents=True, exist_ok=True)
    n = len(sequences)
    batch_files = []
    for i in range(0, n, batch_size):
        start = i
        end = min(i + batch_size, n)
        batch_file = emb_dir / f"PafA_embeddings_batch_{start}_{end-1}.safetensors"
        batch_files.append(batch_file)

    # If all batch files exist, load and return them (skip ESM model)
    if all(f.exists() for f in batch_files):
        for f in batch_files:
            emb = load_file(str(f))["embedding"]
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0).numpy()

    # Only load ESM model if needed
    print("Loading ESM model...")
    esm_model_name = 'esm2_t33_650M_UR50D'
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    for i in range(0, n, batch_size):
        start = i
        end = min(i + batch_size, n)
        batch_file = emb_dir / f"PafA_embeddings_batch_{start}_{end-1}.safetensors"
        if batch_file.exists():
            emb = load_file(str(batch_file))["embedding"]
            embeddings.append(emb)
            continue
        batch = [(str(j), sequences[j]) for j in range(start, end)]
        _, _, tokens = batch_converter(batch)
        with torch.no_grad():
            out = esm_model(tokens, repr_layers=[layer], return_contacts=False)
        reps = out['representations'][layer].cpu()  # (batch, L+2, D)
        Lp2 = reps.shape[1]
        emb = reps[:, 1:Lp2-1, :].contiguous()  # (batch, L, D)
        save_file({"embedding": emb}, str(batch_file))
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0).numpy()

all_reps = get_embeddings(sequences, layer=ESM_LAYER, batch_size=16)
N, L, D = all_reps.shape

# Embed WT for baseline
data_wt = [('WT', wt_seq)]
wt_reps = get_embeddings([wt_seq], layer=ESM_LAYER, batch_size=1)  # (1, L, D)
baseline = wt_reps

# 3. Train-test split, preserving original indices
# Hyperparameters (global scope)
hidden_dim1 = 1024
hidden_dim2 = 512
hidden_dim3 = 256
hidden_dim4 = 128
dropout_rate = 0.15
learning_rate = 5e-5
weight_decay = 1e-4  # L2 regularization
batch_size = 32  # Increased for more stable gradients
use_early_stopping = True  # Set to False to disable early stopping
patience = 50    # Early stopping patience
lr_decay_factor = 0.5
lr_decay_patience = 40
epochs = 500

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Hyperparameters (move above train_and_evaluate for global scope)
hidden_dim1 = 1024
hidden_dim2 = 512
hidden_dim3 = 256
hidden_dim4 = 128
dropout_rate = 0.15
learning_rate = 5e-5
weight_decay = 1e-4  # L2 regularization
batch_size = 32  # Increased for more stable gradients
use_early_stopping = True  # Set to False to disable early stopping
patience = 50    # Early stopping patience
lr_decay_factor = 0.5
lr_decay_patience = 40
epochs = 500

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 5-fold cross-validation
print("Running 5-fold cross-validation...")
orig_idx = np.arange(N)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

def train_and_evaluate(X_train, y_train, X_val, y_val, fold_idx):
    # Model/optimizer must be re-initialized each fold
    if POOLING_METHOD == 'mean':
        model = MeanPoolingModel(L, D, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, hidden_dim3=hidden_dim3, hidden_dim4=hidden_dim4, dropout_rate=dropout_rate)
    elif POOLING_METHOD == 'attention':
        model = AttentionConvPoolingModel(L, D, conv_dim=256, kernel_size=5, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, hidden_dim3=hidden_dim3, hidden_dim4=hidden_dim4, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown pooling method: {POOLING_METHOD}")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.HuberLoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_decay_factor, patience=lr_decay_patience, verbose=False)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    from tqdm import trange
    epoch_iter = trange(epochs, desc=f"Fold {fold_idx+1} Training", leave=True)
    for epoch in epoch_iter:
        model.train()
        batch_train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        train_loss = np.mean(batch_train_losses)
        train_losses.append(train_loss)

        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_output = model(xb)
                val_loss = criterion(val_output, yb)
                batch_val_losses.append(val_loss.item())
        val_loss = np.mean(batch_val_losses)
        val_losses.append(val_loss)

        # Update progress bar with current losses
        epoch_iter.set_postfix({"train_loss": train_loss, "val_loss": val_loss})

        lr_scheduler.step(val_loss)

        if use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    model.load_state_dict(best_model_state)
                    break
    # Save loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Fold {fold_idx+1})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'PafA_training_loss_curve_fold{fold_idx+1}.png'), bbox_inches='tight', dpi=200)
    plt.close()

    # Evaluate on validation set
    model.eval()
    pred_norm = []
    y_true_norm = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb).squeeze(-1).cpu().numpy()
            pred_norm.append(out)
            y_true_norm.append(yb.squeeze(-1).cpu().numpy())
    pred_norm = np.concatenate(pred_norm)
    y_true_norm = np.concatenate(y_true_norm)
    pred_log = pred_norm * y_log_std + y_log_mean
    y_true_log = y_true_norm * y_log_std + y_log_mean
    pred = np.expm1(pred_log)
    y_true = np.expm1(y_true_log)
    mse = np.mean((pred - y_true) ** 2)
    mae = np.mean(np.abs(pred - y_true))
    ss_res = np.sum((y_true - pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    pearson_r = np.corrcoef(y_true, pred)[0, 1]
    spearman_rho, spearman_p = spearmanr(y_true, pred)

    # Save actual vs predicted plot with metrics
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, pred, alpha=0.5, color='orange')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted (Fold {fold_idx+1})')
    plt.grid(True)
    # Add metrics to plot (top left)
    metrics_text = f"Pearson r: {pearson_r:.3f}\nR2: {r2:.3f}\nSpearman rho: {spearman_rho:.3f}"
    plt.gca().text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                  fontsize=10, verticalalignment='top', horizontalalignment='left',
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.savefig(os.path.join(results_dir, f'PafA_pred_vs_actual_fold{fold_idx+1}.png'), bbox_inches='tight', dpi=200)
    plt.close()

    # Return metrics
    return dict(mse=mse, mae=mae, r2=r2, pearson_r=pearson_r, spearman_rho=spearman_rho)

for fold_idx, (train_index, val_index) in enumerate(kf.split(all_reps)):
    print(f"Fold {fold_idx+1}/5")
    X_train, X_val = all_reps[train_index], all_reps[val_index]
    y_train, y_val = y[train_index], y[val_index]
    metrics = train_and_evaluate(X_train, y_train, X_val, y_val, fold_idx)
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