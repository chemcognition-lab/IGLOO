# erbc_linear_esm_residue_ig_baseline.py
# Train a linear model on ERBC variant data using ESM embeddings,
# compute residue-level IG attributions using the WT variant as baseline,
# save both the IG plot and attribution DataFrame,
# and produce a sequence alignment highlighting hotspots.

import os

import esm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, layers

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
import seaborn as sns

print(f"Loading data from {data_path}")
df = pd.read_csv(data_path)
col = 'kcat_cMUP'
# Remove rows with NaN or Inf in target
df = df[df[col].notna()]
df = df[np.isfinite(df[col])]
sequences = df['SEQ'].astype(str).tolist()
y_raw = df[col].values.astype(np.float32)

# Config: select ESM layer and pooling method
ESM_LAYER = 6  # Change this to try other layers
POOLING_METHOD = 'attention'  # Options: 'mean', 'attention', 'conv_attention'


# Log-transform and z-score normalize target
print("Applying log1p transform to target...")
y_log = np.log1p(y_raw)
y_log_mean = np.mean(y_log)
y_log_std = np.std(y_log)
y = (y_log - y_log_mean) / y_log_std
print(f"Target log1p mean: {y_log_mean:.4f}, std: {y_log_std:.4f}")

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

# Debug: Check for NaNs/Infs in input and target
print('NaNs in y:', np.isnan(y).sum())
print('Infs in y:', np.isinf(y).sum())

# Identify WT sequence for baseline
wt_seq = df['SEQ'].iloc[-1]
print(f"Using WT sequence as baseline: {wt_seq}")


# 2. Tokenize & embed with ESM (directly, configurable layer/pooling)
print("Loading ESM model...")
esm_model_name = 'esm2_t6_8M_UR50D'
esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
esm_model.eval()
batch_converter = alphabet.get_batch_converter()

def get_embeddings(sequences, layer=ESM_LAYER, batch_size=16):
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = [(str(j), sequences[j]) for j in range(i, min(i+batch_size, len(sequences)))]
        _, _, tokens = batch_converter(batch)
        with torch.no_grad():
            out = esm_model(tokens, repr_layers=[layer], return_contacts=False)
        reps = out['representations'][layer].numpy()
        # Remove [CLS] and [EOS] tokens
        Lp2 = reps.shape[1]
        embeddings.append(reps[:, 1:Lp2-1, :])
    return np.concatenate(embeddings, axis=0)

all_reps = get_embeddings(sequences, layer=ESM_LAYER, batch_size=16)
N, L, D = all_reps.shape

# Optionally normalize input features (per feature z-score)
# Uncomment if needed:
# all_reps_mean = np.mean(all_reps, axis=(0, 1), keepdims=True)
# all_reps_std = np.std(all_reps, axis=(0, 1), keepdims=True)
# all_reps = (all_reps - all_reps_mean) / (all_reps_std + 1e-8)

# Embed WT for baseline
data_wt = [('WT', wt_seq)]
wt_reps = get_embeddings([wt_seq], layer=ESM_LAYER, batch_size=1)  # (1, L, D)
baseline = wt_reps

# 3. Train-test split, preserving original indices
print("Splitting data...")
orig_idx = np.arange(N)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    all_reps, y, orig_idx, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# 4. Build TensorFlow model: (L, D) -> mean pool -> linear

# Build PyTorch model: (N, L, D) -> mean pool -> linear
print("Building PyTorch model with attention pooling and hidden layer...")
import torch.nn as nn
import torch.optim as optim





# Hyperparameters
hidden_dim1 = 256
hidden_dim2 = 128
dropout_rate = 0.3
learning_rate = 5e-4
weight_decay = 1e-3  # L2 regularization
batch_size = 32  # You can increase this for less noisy training
use_early_stopping = False  # Set to False to disable early stopping
patience = 30    # Early stopping patience
lr_decay_factor = 0.5
lr_decay_patience = 20


# Model selection based on pooling method
class MeanPoolingModel(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim1=256, hidden_dim2=128, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, 1)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        pooled = torch.mean(x, dim=1)
        x = self.fc1(pooled)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class AttentionConvPoolingModel(nn.Module):
    def __init__(self, seq_len, input_dim, conv_dim=128, kernel_size=5, hidden_dim1=256, hidden_dim2=128, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu_conv = nn.ReLU()
        self.attn = nn.Linear(conv_dim, 1)
        self.fc1 = nn.Linear(conv_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, 1)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.conv1(x)      # (batch, conv_dim, seq_len)
        x = self.relu_conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, conv_dim)
        attn_weights = torch.softmax(self.attn(x), dim=1)
        pooled = torch.sum(attn_weights * x, dim=1)
        x = self.fc1(pooled)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Select model
if POOLING_METHOD == 'mean':
    model = MeanPoolingModel(L, D, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout_rate=dropout_rate)
elif POOLING_METHOD == 'attention':
    model = AttentionConvPoolingModel(L, D, conv_dim=128, kernel_size=5, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout_rate=dropout_rate)
else:
    raise ValueError(f"Unknown pooling method: {POOLING_METHOD}")


optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.MSELoss()
# Learning rate scheduler
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay_factor, patience=lr_decay_patience, verbose=True)

# 5. Train


print("Training PyTorch model...")

# Training loop with early stopping and batch size
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
val_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_losses = []
val_losses = []
epochs = 500
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    batch_train_losses = []
    for xb, yb in train_loader:
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
            val_output = model(xb)
            val_loss = criterion(val_output, yb)
            batch_val_losses.append(val_loss.item())
    val_loss = np.mean(batch_val_losses)
    val_losses.append(val_loss)

    # Learning rate scheduler step
    lr_scheduler.step(val_loss)

    # Early stopping (optional)
    if use_early_stopping:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

# Plot training and validation loss curves
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'PafA_training_loss_curve.png'), bbox_inches='tight', dpi=200)
plt.show()

# 6. Evaluate




print("Evaluating...")
model.eval()
pred_norm = []
y_true_norm = []
with torch.no_grad():
    for xb, yb in val_loader:
        out = model(xb).squeeze(-1).cpu().numpy()
        pred_norm.append(out)
        y_true_norm.append(yb.squeeze(-1).cpu().numpy())
pred_norm = np.concatenate(pred_norm)
y_true_norm = np.concatenate(y_true_norm)
# Inverse transform to original scale
pred_log = pred_norm * y_log_std + y_log_mean
y_true_log = y_true_norm * y_log_std + y_log_mean
pred = np.expm1(pred_log)
y_true = np.expm1(y_true_log)
mse = np.mean((pred - y_true) ** 2)
mae = np.mean(np.abs(pred - y_true))
# R2 score
ss_res = np.sum((y_true - pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
# Pearson correlation
pearson_r = np.corrcoef(y_true, pred)[0, 1]
# Spearman correlation
from scipy.stats import spearmanr
spearman_rho, spearman_p = spearmanr(y_true, pred)
print(f"Test MSE (original scale): {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
print(f"Pearson r: {pearson_r:.4f}")
print(f"Spearman rho: {spearman_rho:.4f}, p={spearman_p:.4g}")

# 7. Residue-level Integrated Gradients

# PyTorch Integrated Gradients
def integrated_gradients_res(x, model, baseline, steps=50):
    x = torch.tensor(x, dtype=torch.float32)
    baseline = torch.tensor(baseline, dtype=torch.float32)
    scaled = [baseline + (i/steps)*(x - baseline) for i in range(steps+1)]
    grads = []
    for s in scaled:
        s = s.clone().detach().requires_grad_(True)
        pred = model(s)
        pred = pred.squeeze(-1)
        grad = torch.autograd.grad(pred, s, grad_outputs=torch.ones_like(pred), retain_graph=True)[0]
        grads.append(grad)
    avg_grads = torch.mean(torch.stack(grads[1:], dim=0), dim=0)
    return (x - baseline) * avg_grads

# 8. Compute attributions for a sample variant

test_idx = 0  # choose sample index in test set
orig_index = idx_test[test_idx]
sample_seq = sequences[orig_index]
var_seq = sample_seq  # for alignment plot
sample_input = X_test[test_idx:test_idx+1]
print(f"Computing IG for variant index {orig_index} (sequence {sample_seq})...")
attr = integrated_gradients_res(sample_input, model, baseline)
residue_scores = attr[0].sum(dim=-1).detach().cpu().numpy()

# 9. Map back to positions and residues
df_attr = pd.DataFrame({
    'Position': np.arange(1, L+1),
    'Residue': list(sample_seq),
    'IG_Attribution': residue_scores
})
print(df_attr.head(10))

# 10. Save the attribution DataFrame
print(f"Saving attribution DataFrame to {df_path}...")
df_attr.to_csv(df_path, index=False)
print(f"Saved attribution DataFrame to {df_path}")

# 11. Plot and save IG bar chart
print(f"Saving IG plot to {plot_path}...")
plt.figure(figsize=(12,4))
plt.bar(df_attr['Position'], df_attr['IG_Attribution'])
plt.xlabel('Residue Position')
plt.ylabel('IG Attribution vs WT')
plt.title(f'Residue-Level IG for sample {orig_index} (vs WT)')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved IG plot to {plot_path}")

# 12. Sequence alignment with hotspots
# Identify your top 5 positive and top 5 negative hotspots
hot_pos = df_attr.nlargest(5, 'IG_Attribution')['Position'].tolist()
hot_neg = df_attr.nsmallest(5, 'IG_Attribution')['Position'].tolist()
hotspots = set(hot_pos + hot_neg)

print(f"Hotspot positions: {sorted(hotspots)}")

print(f"Saving alignment plot to {align_plot_path}...")
plt.figure(figsize=(15, 2))
for i, (w, v) in enumerate(zip(wt_seq, var_seq)):
    color = 'red' if (i+1) in hotspots else 'black'
    plt.text(i, 1,   w, ha='center', va='center', color=color, fontsize=8)
    plt.text(i, 0.5, v, ha='center', va='center', color=color, fontsize=8)

plt.yticks([1, 0.5], ['WT', 'Variant'])
plt.xticks([])
plt.xlim(-1, len(wt_seq))
plt.ylim(0, 1.2)
plt.title('Sequence Alignment with Hotspots Highlighted (in red)')
plt.tight_layout()
plt.savefig(align_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved alignment plot to {align_plot_path}")