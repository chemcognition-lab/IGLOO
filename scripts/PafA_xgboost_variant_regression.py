# erbc_xgboost_variant_regression.py

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

from src.igloo.embedding import get_embeddings

# Determine script directory and set paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Paths
data_path = os.path.join(project_root, 'data', 'PafA_data_processed.csv')
results_dir = os.path.join(project_root, 'results', 'PafA_XGB')
os.makedirs(results_dir, exist_ok=True)

# 1. Load dataset
print(f"Loading data from {data_path}")
df = pd.read_csv(data_path)
col = 'kcat_cMUP'
df = df[df[col].notna()]
df = df[np.isfinite(df[col])]
sequences = df['SEQ'].astype(str).tolist()
y_raw = df[col].values.astype(np.float32)

# Log-transform and z-score normalize target
y_log = y_raw #np.log1p(y_raw)
y_log_mean = np.mean(y_log)
y_log_std = np.std(y_log)
y = (y_log - y_log_mean) / y_log_std
print(f"Target log1p mean: {y_log_mean:.4f}, std: {y_log_std:.4f}")

# Identify WT sequence for baseline
wt_seq = df['SEQ'].iloc[-1]
print(f"Using WT sequence as baseline: {wt_seq}")

# 2. Tokenize & embed with ESM
ESM_LAYER = 33
print("Embedding all sequences...")
all_reps = get_embeddings(sequences, layer=ESM_LAYER, batch_size=16)
N, L, D = all_reps.shape
print(f"Embeddings shape: {all_reps.shape}")

# Embed WT for baseline
wt_reps = get_embeddings([wt_seq], layer=ESM_LAYER, batch_size=1)  # (1, L, D)

# Compute difference embeddings: X_variant - X_WT
all_reps_diff = all_reps - wt_reps[0]  # (N, L, D)

# Pooling: mean over sequence length (L) to get (N, D)
X_features = all_reps_diff.mean(axis=1)  # (N, D)

# GridSearchCV for XGBoost hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0.5, 1.0, 2.0]
}

# Use only the first fold for grid search to save time
train_index, val_index = next(KFold(n_splits=5, shuffle=True, random_state=42).split(X_features))
X_train, X_val = X_features[train_index], X_features[val_index]
y_train, y_val = y[train_index], y[val_index]

xgb_model = xgb.XGBRegressor(random_state=42)
gs = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=-1
)
gs.fit(X_train, y_train)
print(f"Best parameters from GridSearchCV: {gs.best_params_}")
print(f"Best CV score (neg MSE): {gs.best_score_}")

# Use best params for cross-validation below
best_params = gs.best_params_

# 5-fold cross-validation
print("Running 5-fold cross-validation with XGBoost...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []
wt_idx = len(sequences) - 1  # WT is last in the list

for fold_idx, (train_index, val_index) in enumerate(kf.split(X_features)):
    print(f"Fold {fold_idx+1}/5")
    # Ensure WT is in training set
    if wt_idx not in train_index:
        train_index = np.append(train_index, wt_idx)
        val_index = val_index[val_index != wt_idx]
    X_train, X_val = X_features[train_index], X_features[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = xgb.XGBRegressor(
        **best_params,
        random_state=42
    )
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_val)

    # Inverse normalization for metrics
    y_pred_log = y_pred * y_log_std + y_log_mean
    y_val_log = y_val * y_log_std + y_log_mean
    y_pred_raw = np.expm1(y_pred_log)
    y_val_raw = np.expm1(y_val_log)

    mse = mean_squared_error(y_val_raw, y_pred_raw)
    mae = mean_absolute_error(y_val_raw, y_pred_raw)
    r2 = r2_score(y_val_raw, y_pred_raw)
    pearson_r = pearsonr(y_val_raw, y_pred_raw)[0]
    spearman_rho = spearmanr(y_val_raw, y_pred_raw)[0]
    fold_metrics.append(dict(mse=mse, mae=mae, r2=r2, pearson_r=pearson_r, spearman_rho=spearman_rho))

    # Plot predictions vs actual
    plt.figure(figsize=(6,6))
    plt.scatter(y_val_raw, y_pred_raw, alpha=0.5, color='blue')
    plt.plot([min(y_val_raw), max(y_val_raw)], [min(y_val_raw), max(y_val_raw)], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'XGBoost Actual vs Predicted (Fold {fold_idx+1})')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'PafA_xgb_pred_vs_actual_fold{fold_idx+1}.png'), bbox_inches='tight', dpi=200)
    plt.close()

# Print cross-validation results
metric_names = ['mse', 'mae', 'r2', 'pearson_r', 'spearman_rho']
print("\nCross-validation results (XGBoost):")
for name in metric_names:
    values = [fm[name] for fm in fold_metrics]
    print(f"{name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

# Save metrics to CSV
metrics_df = pd.DataFrame(fold_metrics)
metrics_df.to_csv(os.path.join(results_dir, 'PafA_xgb_cv_metrics.csv'), index=False)
print(f"Saved cross-validation metrics to {os.path.join(results_dir, 'PafA_xgb_cv_metrics.csv')}")
