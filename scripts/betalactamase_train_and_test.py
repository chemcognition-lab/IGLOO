import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Train a linear model on beta-lactamase variant data using ESM embeddings,
# Train on betalactamase_train.csv and test on betalactamase_test.csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from src.igloo.data_processing import CustomDataset, batch_pad_and_normalize
from src.igloo.embedding import get_embeddings
from src.igloo.train_and_eval import train

# Determine script directory and set paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Paths
train_data_path = os.path.join(project_root, 'data', 'betalactamase_train.csv')
test_data_path = os.path.join(project_root, 'data', 'betalactamase_test.csv')
results_dir = os.path.join(project_root, 'results', 'betalactamase')

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# 1. Load datasets
print(f"Loading training data from {train_data_path}")
train_df = pd.read_csv(train_data_path)
print(f"Loading test data from {test_data_path}")
test_df = pd.read_csv(test_data_path)

col = 'quantitative_function'

# Remove rows with NaN or Inf in target
train_df = train_df[train_df[col].notna()]
train_df = train_df[np.isfinite(train_df[col])]
test_df = test_df[test_df[col].notna()]
test_df = test_df[np.isfinite(test_df[col])]

train_sequences = train_df['SEQ'].astype(str).tolist()
test_sequences = test_df['SEQ'].astype(str).tolist()
y_train_raw = train_df[col].values.astype(np.float32)
y_test_raw = test_df[col].values.astype(np.float32)

print(f"Training set size: {len(train_sequences)} sequences")
print(f"Test set size: {len(test_sequences)} sequences")

# Config: select ESM layer and pooling method
ESM_LAYER = 33  # Change this to try other layers
POOLING_METHOD = 'esm_attention'  # Options: 'mean', 'attention', 'conv_attention', 'linear', 'esm_attention'
LOSS_TYPE = 'huber'  # Options: 'mse', 'mae', 'huber', 'quantile', 'combined'

# Hyperparameters
hidden_dim1 = 1024
hidden_dim2 = 512
hidden_dim3 = 255
hidden_dim4 = 128
dropout_rate = 0.5
learning_rate = 5e-4
weight_decay = 1e-4
batch_size = 256
use_early_stopping = True
patience = 15
lr_decay_factor = 0.5
lr_decay_patience = 20
epochs = 200
final_activation = None  # 'tanh', 'sigmoid', or None

# Normalize target based on training data only
print("Normalizing targets based on training data...")
y_train_log = y_train_raw  # No log transform, use raw values
y_train_log_mean = np.mean(y_train_log)
y_train_log_std = np.std(y_train_log)
y_train = (y_train_log - y_train_log_mean) / y_train_log_std

# Apply same normalization to test data
y_test_log = y_test_raw
y_test = (y_test_log - y_train_log_mean) / y_train_log_std

print(f"Training target mean: {y_train_log_mean:.4f}, std: {y_train_log_std:.4f}")
print(f"Training target range: [{y_train_raw.min():.4f}, {y_train_raw.max():.4f}]")
print(f"Test target range: [{y_test_raw.min():.4f}, {y_test_raw.max():.4f}]")

# Plot distribution of normalized target values
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Training data - normalized
axes[0, 0].hist(y_train, bins=30, alpha=0.7, density=True, label='Training')
axes[0, 0].set_xlabel('Normalized Target Value')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Training Data - Normalized')
axes[0, 0].grid(True)

# Training data - raw
axes[0, 1].hist(y_train_raw, bins=30, alpha=0.7, density=True, label='Training')
axes[0, 1].set_xlabel('Raw Target Value')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Training Data - Raw')
axes[0, 1].grid(True)

# Test data - normalized
axes[1, 0].hist(y_test, bins=30, alpha=0.7, density=True, label='Test', color='orange')
axes[1, 0].set_xlabel('Normalized Target Value')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Test Data - Normalized')
axes[1, 0].grid(True)

# Test data - raw
axes[1, 1].hist(y_test_raw, bins=30, alpha=0.7, density=True, label='Test', color='orange')
axes[1, 1].set_xlabel('Raw Target Value')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Test Data - Raw')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'target_distributions.png'), bbox_inches='tight', dpi=200)
plt.close()

# 2. Get ESM embeddings
print("Computing ESM embeddings for training data...")
train_reps = get_embeddings(train_sequences, layer=ESM_LAYER, batch_size=16, dataset_name='betalactamase_train')
N_train, L, D = train_reps.shape

print("Computing ESM embeddings for test data...")
test_reps = get_embeddings(test_sequences, layer=ESM_LAYER, batch_size=16, dataset_name='betalactamase_test')
N_test = test_reps.shape[0]

print(f"Training embeddings shape: {train_reps.shape}")
print(f"Test embeddings shape: {test_reps.shape}")

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Train model on training data
print("Training model on training data...")
model, train_metrics = train(
    train_reps, y_train,
    L=L,
    batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    hidden_dim1=hidden_dim1,
    hidden_dim2=hidden_dim2,
    hidden_dim3=hidden_dim3,
    hidden_dim4=hidden_dim4,
    dropout_rate=dropout_rate,
    epochs=epochs,
    final_activation=final_activation,
    device=device,
    POOLING_METHOD=POOLING_METHOD,
    LOSS_TYPE=LOSS_TYPE,
    y_log_std=y_train_log_std,
    y_log_mean=y_train_log_mean
)

print("Training completed.")
print("Training metrics:")
for k, v in train_metrics.items():
    print(f"  {k}: {v:.4f}")

# 4. Evaluate on training data
print("\nEvaluating on training data...")
model.eval()
train_pred_norm = []
train_true_norm = []
train_dataset = CustomDataset(train_reps, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_pad_and_normalize)

with torch.no_grad():
    for xb, lengths, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if POOLING_METHOD == 'esm_attention':
            input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
            for i, length in enumerate(lengths):
                input_mask[i, :length] = 1
            out = model(xb, input_mask).squeeze(-1).cpu().numpy()
        else:
            out = model(xb).squeeze(-1).cpu().numpy()
        train_pred_norm.append(out)
        train_true_norm.append(yb.squeeze(-1).cpu().numpy())

train_pred_norm = np.concatenate(train_pred_norm)
train_true_norm = np.concatenate(train_true_norm)
train_pred_unnorm = train_pred_norm * y_train_log_std + y_train_log_mean
train_true_unnorm = train_true_norm * y_train_log_std + y_train_log_mean

# Training metrics
train_r2 = r2_score(train_true_unnorm, train_pred_unnorm)
train_pearson_r, _ = stats.pearsonr(train_true_unnorm, train_pred_unnorm)
train_spearman_rho, _ = stats.spearmanr(train_true_unnorm, train_pred_unnorm)

print(f"Training R²: {train_r2:.4f}")
print(f"Training Pearson r: {train_pearson_r:.4f}")
print(f"Training Spearman ρ: {train_spearman_rho:.4f}")

# 5. Evaluate on test data
print("\nEvaluating on test data...")
test_pred_norm = []
test_true_norm = []
test_dataset = CustomDataset(test_reps, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_pad_and_normalize)

with torch.no_grad():
    for xb, lengths, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if POOLING_METHOD == 'esm_attention':
            input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
            for i, length in enumerate(lengths):
                input_mask[i, :length] = 1
            out = model(xb, input_mask).squeeze(-1).cpu().numpy()
        else:
            out = model(xb).squeeze(-1).cpu().numpy()
        test_pred_norm.append(out)
        test_true_norm.append(yb.squeeze(-1).cpu().numpy())

test_pred_norm = np.concatenate(test_pred_norm)
test_true_norm = np.concatenate(test_true_norm)
test_pred_unnorm = test_pred_norm * y_train_log_std + y_train_log_mean
test_true_unnorm = test_true_norm * y_train_log_std + y_train_log_mean

# Test metrics
test_r2 = r2_score(test_true_unnorm, test_pred_unnorm)
test_pearson_r, _ = stats.pearsonr(test_true_unnorm, test_pred_unnorm)
test_spearman_rho, _ = stats.spearmanr(test_true_unnorm, test_pred_unnorm)

print(f"Test R²: {test_r2:.4f}")
print(f"Test Pearson r: {test_pearson_r:.4f}")
print(f"Test Spearman ρ: {test_spearman_rho:.4f}")

# 6. Create visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training data scatter plot
axes[0].scatter(train_true_unnorm, train_pred_unnorm, alpha=0.6, s=20)
axes[0].plot([train_true_unnorm.min(), train_true_unnorm.max()],
             [train_true_unnorm.min(), train_true_unnorm.max()], 'k--', lw=2, label='Ideal')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Training Data: Predicted vs Actual')

# Annotate training metrics
train_metrics_text = f"R² = {train_r2:.3f}\nPearson r = {train_pearson_r:.3f}\nSpearman ρ = {train_spearman_rho:.3f}"
axes[0].text(0.05, 0.95, train_metrics_text, transform=axes[0].transAxes,
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test data scatter plot
axes[1].scatter(test_true_unnorm, test_pred_unnorm, alpha=0.6, s=20, color='orange')
axes[1].plot([test_true_unnorm.min(), test_true_unnorm.max()],
             [test_true_unnorm.min(), test_true_unnorm.max()], 'k--', lw=2, label='Ideal')
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
axes[1].set_title('Test Data: Predicted vs Actual')

# Annotate test metrics
test_metrics_text = f"R² = {test_r2:.3f}\nPearson r = {test_pearson_r:.3f}\nSpearman ρ = {test_spearman_rho:.3f}"
axes[1].text(0.05, 0.95, test_metrics_text, transform=axes[1].transAxes,
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'pred_vs_actual_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()

# 7. Save predictions and results
results_summary = {
    'training_size': len(train_sequences),
    'test_size': len(test_sequences),
    'esm_layer': ESM_LAYER,
    'pooling_method': POOLING_METHOD,
    'loss_type': LOSS_TYPE,
    'target_mean': y_train_log_mean,
    'target_std': y_train_log_std,
    'train_r2': train_r2,
    'train_pearson_r': train_pearson_r,
    'train_spearman_rho': train_spearman_rho,
    'test_r2': test_r2,
    'test_pearson_r': test_pearson_r,
    'test_spearman_rho': test_spearman_rho
}

# Save results summary
results_df = pd.DataFrame([results_summary])
results_df.to_csv(os.path.join(results_dir, 'results_summary.csv'), index=False)

# Save detailed predictions
train_results_df = pd.DataFrame({
    'SEQ': train_sequences,
    'actual': train_true_unnorm,
    'predicted': train_pred_unnorm,
    'residual': train_true_unnorm - train_pred_unnorm
})
train_results_df.to_csv(os.path.join(results_dir, 'train_predictions.csv'), index=False)

test_results_df = pd.DataFrame({
    'SEQ': test_sequences,
    'actual': test_true_unnorm,
    'predicted': test_pred_unnorm,
    'residual': test_true_unnorm - test_pred_unnorm
})
test_results_df.to_csv(os.path.join(results_dir, 'test_predictions.csv'), index=False)

# 8. Print summary
print("\n" + "="*60)
print("BETA-LACTAMASE TRAINING AND TESTING COMPLETE!")
print("="*60)
print(f"Results saved to: {results_dir}")
print("\nFinal Results:")
print("Training Performance:")
print(f"  R² = {train_r2:.4f}")
print(f"  Pearson r = {train_pearson_r:.4f}")
print(f"  Spearman ρ = {train_spearman_rho:.4f}")
print("\nTest Performance:")
print(f"  R² = {test_r2:.4f}")
print(f"  Pearson r = {test_pearson_r:.4f}")
print(f"  Spearman ρ = {test_spearman_rho:.4f}")
print("\nGenerated files:")
print("  - results_summary.csv: Overall performance metrics")
print("  - train_predictions.csv: Training set predictions")
print("  - test_predictions.csv: Test set predictions")
print("  - pred_vs_actual_comparison.png: Scatter plots")
print("  - target_distributions.png: Target distribution plots")

# Save the model
model_path = os.path.join(results_dir, 'betalactamase_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
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
        'y_log_mean': y_train_log_mean,
        'y_log_std': y_train_log_std
    }
}, model_path)
print("  - betalactamase_model.pth: Trained model")
