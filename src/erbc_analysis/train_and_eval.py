

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import trange

from src.erbc_analysis.data_processing import CustomDataset, batch_pad_and_normalize
from src.erbc_analysis.loss import combined_loss, huber_loss, mae_loss, mse_loss, quantile_loss
from src.erbc_analysis.models import AttentionConvPoolingModel, LinearRegressionModel, MeanPoolingModel, ESMAttention1d


def train_and_evaluate(
    X_train, y_train, X_val, y_val, fold_idx,
    L,
    results_dir,
    batch_size,
    learning_rate,
    weight_decay,
    hidden_dim1,
    hidden_dim2,
    hidden_dim3,
    hidden_dim4,
    dropout_rate,
    use_early_stopping,
    patience,
    lr_decay_factor,
    lr_decay_patience,
    epochs,
    final_activation,
    device,
    POOLING_METHOD,
    LOSS_TYPE,
    y_log_std,
    y_log_mean
):
    """
    Train and evaluate a protein variant regression model for one fold.
    Returns metrics dict for the fold.
    """
    input_dim_combined = X_train.shape[2]
    target_min = np.min(y_train)
    target_max = np.max(y_train)
    if POOLING_METHOD == 'mean':
        model = MeanPoolingModel(
            L, input_dim_combined,
            hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2,
            hidden_dim3=hidden_dim3, hidden_dim4=hidden_dim4,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
            target_min=target_min, target_max=target_max
        )
    elif POOLING_METHOD == 'attention':
        model = AttentionConvPoolingModel(
            L, input_dim_combined,
            conv_dim=256, kernel_size=5,
            hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2,
            hidden_dim3=hidden_dim3, hidden_dim4=hidden_dim4,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
            target_min=target_min, target_max=target_max
        )
    elif POOLING_METHOD == 'linear':
        model = LinearRegressionModel(
            L, input_dim_combined,
            final_activation=final_activation,
            target_min=target_min, target_max=target_max
        )
    elif POOLING_METHOD == 'esm_attention':
        model = ESMAttention1d(max_length=L, d_embedding=input_dim_combined)
    else:
        raise ValueError(f"Unknown pooling method: {POOLING_METHOD}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    quantile_weight = 0.3
    mse_weight = 0.7

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_decay_factor, patience=lr_decay_patience)

    print(f"Training {POOLING_METHOD} model for fold {fold_idx+1} with {LOSS_TYPE} loss...")

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=batch_pad_and_normalize)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_pad_and_normalize)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    epoch_iter = trange(epochs, desc=f"Fold {fold_idx+1} Training", leave=True)
    for epoch in epoch_iter:
        model.train()
        batch_train_losses = []
        for xb, lengths, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            if LOSS_TYPE == 'mse':
                loss = mse_loss(output, yb)
            elif LOSS_TYPE == 'mae':
                loss = mae_loss(output, yb)
            elif LOSS_TYPE == 'huber':
                loss = huber_loss(output, yb, delta=1.0)
            elif LOSS_TYPE == 'quantile':
                loss = quantile_loss(output, yb, quantile=0.5)
            elif LOSS_TYPE == 'combined':
                loss = combined_loss(
                    output, yb,
                    loss_fns=[(quantile_loss, {'quantile': 0.5}), (mse_loss, {})],
                    weights=[quantile_weight, mse_weight]
                )
            else:
                raise ValueError(f"Unknown LOSS_TYPE: {LOSS_TYPE}")
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        train_loss = np.mean(batch_train_losses)
        train_losses.append(train_loss)

        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for xb, lengths, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_output = model(xb)
                if LOSS_TYPE == 'mse':
                    val_loss = mse_loss(val_output, yb)
                elif LOSS_TYPE == 'mae':
                    val_loss = mae_loss(val_output, yb)
                elif LOSS_TYPE == 'huber':
                    val_loss = huber_loss(val_output, yb, delta=1.0)
                elif LOSS_TYPE == 'quantile':
                    val_loss = quantile_loss(val_output, yb, quantile=0.5)
                elif LOSS_TYPE == 'combined':
                    val_loss = combined_loss(
                        val_output, yb,
                        loss_fns=[(quantile_loss, {'quantile': 0.5}), (mse_loss, {})],
                        weights=[0.3, 0.7]
                    )
                else:
                    raise ValueError(f"Unknown LOSS_TYPE: {LOSS_TYPE}")
                batch_val_losses.append(val_loss.item())
        val_loss = np.mean(batch_val_losses)
        val_losses.append(val_loss)

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
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel(f'Loss ({LOSS_TYPE})')
    plt.title(f'Training and Validation Loss (Fold {fold_idx+1})')
    plt.legend()
    plt.grid(True)
    if results_dir:
        plt.savefig(
            os.path.join(results_dir, f'PafA_training_loss_curve_fold{fold_idx+1}.png'),
            bbox_inches='tight', dpi=200
        )
    plt.close()

    model.eval()
    pred_norm = []
    y_true_norm = []
    with torch.no_grad():
        for xb, lengths, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb).squeeze(-1).cpu().numpy()
            pred_norm.append(out)
            y_true_norm.append(yb.squeeze(-1).cpu().numpy())
    pred_norm = np.concatenate(pred_norm)
    y_true_norm = np.concatenate(y_true_norm)
    pred_log = pred_norm * y_log_std + y_log_mean
    y_true_log = y_true_norm * y_log_std + y_log_mean
    pred = pred_log #np.expm1(pred_log)
    y_true = y_true_log # np.expm1(y_true_log)
    mse = np.mean((pred - y_true) ** 2)
    mae = np.mean(np.abs(pred - y_true))
    ss_res = np.sum((y_true - pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    pearson_r = np.corrcoef(y_true, pred)[0, 1]
    spearman_rho, spearman_p = spearmanr(y_true, pred)

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, pred, alpha=0.5, color='orange')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted (Fold {fold_idx+1})')
    plt.grid(True)
    metrics_text = f"Pearson r: {pearson_r:.3f}\nR2: {r2:.3f}\nSpearman rho: {spearman_rho:.3f}"
    plt.gca().text(
        0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    if results_dir:
        plt.savefig(
            os.path.join(results_dir, f'PafA_pred_vs_actual_fold{fold_idx+1}.png'),
            bbox_inches='tight', dpi=200
        )
    plt.close()

    return dict(mse=mse, mae=mae, r2=r2, pearson_r=pearson_r, spearman_rho=spearman_rho)
