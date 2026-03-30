import hashlib
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import trange

from src.igloo.data_processing import (
    CustomDataset,
    batch_pad_and_normalize,
    compute_global_normalization_params,
    set_global_normalization_params,
)
from src.igloo.embedding import get_embeddings
from src.igloo.loss import combined_loss, huber_loss, mae_loss, mse_loss, quantile_loss
from src.igloo.models import AttentionConvPoolingModel, ESMAttention1d, LinearRegressionModel, MeanPoolingModel
from src.igloo.utils import create_model


def _create_epoch_plot(model, train_loader, val_loader, train_targets, val_targets,
                      epoch, pooling_method, results_dir, dataset_name, device):
    """Create and save predicted vs actual plot for current epoch."""
    model.eval()

    # Generate hash for unique filename
    hash_input = f"{dataset_name}_epoch_{epoch}_{pooling_method}"
    hash_obj = hashlib.md5(hash_input.encode())
    file_hash = hash_obj.hexdigest()[:8]

    with torch.no_grad():
        # Get training predictions
        train_preds = []
        train_actuals = []
        for xb, lengths, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if pooling_method == 'esm_attention':
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                predictions = model(xb, input_mask)
            else:
                predictions = model(xb)

            train_preds.extend(predictions.squeeze(-1).cpu().numpy())
            train_actuals.extend(yb.squeeze(-1).cpu().numpy())

        # Get validation predictions if available
        val_preds = []
        val_actuals = []
        if val_loader is not None:
            for xb, lengths, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                if pooling_method == 'esm_attention':
                    input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                    for i, length in enumerate(lengths):
                        input_mask[i, :length] = 1
                    predictions = model(xb, input_mask)
                else:
                    predictions = model(xb)

                val_preds.extend(predictions.squeeze(-1).cpu().numpy())
                val_actuals.extend(yb.squeeze(-1).cpu().numpy())

    # Create plot
    plt.figure(figsize=(10, 6))

    if val_loader is not None:
        plt.subplot(1, 2, 1)
        plt.scatter(train_actuals, train_preds, alpha=0.6, s=20, label='Training')
        plt.plot([min(train_actuals), max(train_actuals)],
                 [min(train_actuals), max(train_actuals)], 'r--', lw=2)
        plt.xlabel('Actual (Normalized)')
        plt.ylabel('Predicted (Normalized)')
        plt.title(f'Training - Epoch {epoch+1}')
        plt.grid(True, alpha=0.3)

        # Calculate metrics
        train_r2 = r2_score(train_actuals, train_preds)
        plt.text(0.05, 0.95, f'R² = {train_r2:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.subplot(1, 2, 2)
        plt.scatter(val_actuals, val_preds, alpha=0.6, s=20, label='Validation', color='orange')
        plt.plot([min(val_actuals), max(val_actuals)],
                 [min(val_actuals), max(val_actuals)], 'r--', lw=2)
        plt.xlabel('Actual (Normalized)')
        plt.ylabel('Predicted (Normalized)')
        plt.title(f'Validation - Epoch {epoch+1}')
        plt.grid(True, alpha=0.3)

        val_r2 = r2_score(val_actuals, val_preds)
        plt.text(0.05, 0.95, f'R² = {val_r2:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        plt.scatter(train_actuals, train_preds, alpha=0.6, s=20)
        plt.plot([min(train_actuals), max(train_actuals)],
                 [min(train_actuals), max(train_actuals)], 'r--', lw=2)
        plt.xlabel('Actual (Normalized)')
        plt.ylabel('Predicted (Normalized)')
        plt.title(f'Training - Epoch {epoch+1}')
        plt.grid(True, alpha=0.3)

        train_r2 = r2_score(train_actuals, train_preds)
        plt.text(0.05, 0.95, f'R² = {train_r2:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    # Save plot with hashed filename
    filename = f"epoch_{epoch+1:03d}_{file_hash}_pred_vs_actual.png"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    # Print brief progress info
    if val_loader is not None:
        print(f"    Saved plot: {filename} (Train R²={train_r2:.3f}, Val R²={val_r2:.3f})")
    else:
        print(f"    Saved plot: {filename} (Train R²={train_r2:.3f})")


def train_model_with_embeddings(
    model: torch.nn.Module,
    train_sequences: list,
    train_targets: np.ndarray,
    val_sequences: list = None,
    val_targets: np.ndarray = None,
    esm_layer: int = 33,
    pooling_method: str = 'linear',
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_epochs: int = 100,
    device: torch.device = None,
    patience: int = 10,
    min_delta: float = 1e-4,
    dataset_name: str = 'training',
    save_path: Optional[str] = None,
    loss_type: str = 'mse',
    results_dir: Optional[str] = None,
    model_config: Optional[dict[str, Any]] = None,
    embeddings_subdir: Optional[str] = None,
    esm_model_name: str = 'esmc_300m',
) -> dict:
    """
    Train a model using the same embedding + DataLoader approach as integrated gradients.

    This ensures perfect consistency between training and evaluation/integrated gradients.

    Args:
        model: PyTorch model to train (e.g., LinearModel, AttentionModel)
        train_sequences: List of protein sequences for training
        train_targets: Training target values
        val_sequences: Optional validation sequences
        val_targets: Optional validation targets
        esm_layer: ESM layer to extract embeddings from
        pooling_method: Pooling method ('linear', 'esm_attention', etc.)
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_epochs: Maximum number of training epochs
        device: PyTorch device
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        dataset_name: Name for embedding caching
        save_path: Optional path to save the trained model

    Returns:
        Dictionary with training history and metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    print(f"Getting ESM embeddings for {len(train_sequences)} training sequences...")
    train_embeddings = get_embeddings(
        train_sequences,
        layer=esm_layer,
        batch_size=batch_size,
        dataset_name=f"{dataset_name}_train",
        subdir=embeddings_subdir,
        esm_model_name=esm_model_name,
    )

    val_embeddings = None
    if val_sequences is not None:
        print(f"Getting ESM embeddings for {len(val_sequences)} validation sequences...")
        val_embeddings = get_embeddings(
            val_sequences,
            layer=esm_layer,
            batch_size=batch_size,
            dataset_name=f"{dataset_name}_val",
            subdir=embeddings_subdir,
            esm_model_name=esm_model_name,
        )

    # Compute global normalization parameters for IG analysis (but don't use them for training)
    all_embeddings_for_norm = [train_embeddings]
    if val_embeddings is not None:
        all_embeddings_for_norm.append(val_embeddings)
    norm_params = compute_global_normalization_params(all_embeddings_for_norm)
    print(f"Computed global normalization for IG: mean={norm_params['mean']:.4f}, scale={norm_params['scale']:.4f}")

    # Set global normalization parameters before training

    set_global_normalization_params(norm_params['mean'], norm_params['scale'])
    print(f"Set global normalization for training: mean={norm_params['mean']:.4f}, scale={norm_params['scale']:.4f}")

    # Create a custom collate function that uses global normalization for training
    def batch_pad_and_normalize_training(batch):
        return batch_pad_and_normalize(batch, use_global_normalization=True)

    # Create datasets and dataloaders
    train_dataset = CustomDataset(train_embeddings, train_targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=batch_pad_and_normalize_training  # Use global normalization for training
    )

    val_loader = None
    if val_embeddings is not None:
        val_dataset = CustomDataset(val_embeddings, val_targets)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=batch_pad_and_normalize_training  # Use global normalization for validation
        )

    # Setup optimizer and loss function with better hyperparameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4,
                                  betas=(0.9, 0.999), eps=1e-8)

    # Add cosine annealing scheduler with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
    )

    # Create a warmup scheduler for the first few epochs
    warmup_epochs = min(10, num_epochs // 10)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    ) if warmup_epochs > 0 else None

    if loss_type == 'huber':
        criterion = torch.nn.HuberLoss()
    elif loss_type == 'mae':
        criterion = torch.nn.L1Loss()
    else:  # default to MSE
        criterion = torch.nn.MSELoss()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs with {loss_type} loss...")

    # Model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []

        for _batch_idx, (xb, lengths, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            # Forward pass (same as in integrated gradients)
            if pooling_method == 'esm_attention':
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                predictions = model(xb, input_mask)
            else:
                predictions = model(xb)

            loss = criterion(predictions.squeeze(-1), yb.squeeze(-1))
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['epoch'].append(epoch + 1)



        # Validation phase
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for xb, lengths, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    if pooling_method == 'esm_attention':
                        input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                        for i, length in enumerate(lengths):
                            input_mask[i, :length] = 1
                        predictions = model(xb, input_mask)
                    else:
                        predictions = model(xb)

                    loss = criterion(predictions.squeeze(-1), yb.squeeze(-1))
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)
            history['val_loss'].append(val_loss)

            # Update learning rate (cosine annealing doesn't need loss)

            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model if path provided
                if save_path:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'train_loss': avg_train_loss,
                        'val_loss': val_loss,
                        'pooling_method': pooling_method,
                        'esm_layer': esm_layer,
                        'model_config': model_config or {},
                        'normalization_params': norm_params
                    }, save_path)
            else:
                patience_counter += 1
        else:
            # No validation set - save model periodically and don't use early stopping
            if save_path and (epoch + 1) % 50 == 0:  # Save every 50 epochs
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': None,
                    'pooling_method': pooling_method,
                    'esm_layer': esm_layer,
                    'model_config': model_config or {},
                    'normalization_params': norm_params
                }, save_path)

        # Update learning rate scheduler
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Print progress with current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if val_loss is not None:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, LR: {current_lr:.2e}")

        # Add debugging info every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Sample a few predictions to check if model is learning
                sample_batch = next(iter(train_loader))
                xb, lengths, yb = sample_batch
                xb, yb = xb.to(device), yb.to(device)

                if pooling_method == 'esm_attention':
                    input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                    for i, length in enumerate(lengths):
                        input_mask[i, :length] = 1
                    sample_preds = model(xb, input_mask).squeeze(-1)
                else:
                    sample_preds = model(xb).squeeze(-1)

                pred_mean = sample_preds.mean().item()
                pred_std = sample_preds.std().item()
                target_mean = yb.squeeze(-1).mean().item()
                target_std = yb.squeeze(-1).std().item()

                print(f"  Debug - Pred mean: {pred_mean:.4f}, std: {pred_std:.4f} | "
                      f"Target mean: {target_mean:.4f}, std: {target_std:.4f}")
            model.train()

        # Create and save predicted vs actual plot for this epoch
        if results_dir is not None and (epoch + 1) % 5 == 0:  # Save every 5 epochs to avoid too many plots
            _create_epoch_plot(
                model, train_loader, val_loader, train_targets, val_targets,
                epoch, pooling_method, results_dir, dataset_name, device
            )

        # Early stopping check
        if val_loader is not None and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break

    # Load best model if saved (only when using validation set)
    if save_path and os.path.exists(save_path) and val_loader is not None:
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    elif val_loader is None:
        print("No validation set used - keeping final model from last epoch")

    # Final evaluation
    model.eval()
    final_metrics = {}

    with torch.no_grad():
        # Training metrics
        train_predictions = []
        train_actuals = []

        for xb, lengths, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if pooling_method == 'esm_attention':
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                predictions = model(xb, input_mask)
            else:
                predictions = model(xb)

            train_predictions.extend(predictions.squeeze(-1).cpu().numpy())
            train_actuals.extend(yb.cpu().numpy())

        final_metrics['train_r2'] = r2_score(train_actuals, train_predictions)
        final_metrics['train_mae'] = mean_absolute_error(train_actuals, train_predictions)
        final_metrics['train_rmse'] = np.sqrt(mean_squared_error(train_actuals, train_predictions))

        # Validation metrics
        if val_loader is not None:
            val_predictions = []
            val_actuals = []

            for xb, lengths, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                if pooling_method == 'esm_attention':
                    input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                    for i, length in enumerate(lengths):
                        input_mask[i, :length] = 1
                    predictions = model(xb, input_mask)
                else:
                    predictions = model(xb)

                val_predictions.extend(predictions.squeeze(-1).cpu().numpy())
                val_actuals.extend(yb.cpu().numpy())

            final_metrics['val_r2'] = r2_score(val_actuals, val_predictions)
            final_metrics['val_mae'] = mean_absolute_error(val_actuals, val_predictions)
            final_metrics['val_rmse'] = np.sqrt(mean_squared_error(val_actuals, val_predictions))

    print("\nTraining completed!")

    # Create and save loss vs epoch plot
    if results_dir is not None:
        plt.figure(figsize=(10, 6))

        epochs = history['epoch']
        train_losses = history['train_loss']

        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

        if history['val_loss'] and any(v is not None for v in history['val_loss']):
            val_losses = history['val_loss']
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel(f'Loss ({loss_type.upper()})')
        plt.title(f'{dataset_name} - Training Loss vs Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add final loss values as text
        final_train_loss = train_losses[-1]
        plt.text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.6f}',
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        if history['val_loss'] and any(v is not None for v in history['val_loss']):
            final_val_loss = history['val_loss'][-1]
            plt.text(0.02, 0.88, f'Final Val Loss: {final_val_loss:.6f}',
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Save the plot
        loss_plot_path = os.path.join(results_dir, f'{dataset_name}_loss_vs_epoch.png')
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss vs epoch plot saved to: {loss_plot_path}")

    # Final model save
    if save_path:
        if val_loader is None:
            # No validation set - always save final model (may overwrite periodic saves)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': num_epochs - 1,
                'train_loss': history['train_loss'][-1],
                'val_loss': None,
                'pooling_method': pooling_method,
                'esm_layer': esm_layer,
                'model_config': model_config or {},
                'normalization_params': norm_params,
                'final_metrics': final_metrics
            }, save_path)
            print(f"Final model (epoch {num_epochs}) saved to {save_path}")
        elif not os.path.exists(save_path):
            # Validation set used but no model saved yet (shouldn't happen)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': num_epochs - 1,
                'train_loss': history['train_loss'][-1],
                'val_loss': history['val_loss'][-1] if history['val_loss'] else None,
                'pooling_method': pooling_method,
                'esm_layer': esm_layer,
                'model_config': model_config or {},
                'normalization_params': norm_params,
                'final_metrics': final_metrics
            }, save_path)
            print(f"Final model saved to {save_path}")

    print("Final Training Metrics:")
    print(f"  R² = {final_metrics['train_r2']:.4f}")
    print(f"  MAE = {final_metrics['train_mae']:.4f}")
    print(f"  RMSE = {final_metrics['train_rmse']:.4f}")

    if 'val_r2' in final_metrics:
        print("Final Validation Metrics:")
        print(f"  R² = {final_metrics['val_r2']:.4f}")
        print(f"  MAE = {final_metrics['val_mae']:.4f}")
        print(f"  RMSE = {final_metrics['val_rmse']:.4f}")

    return {
        'model': model,
        'history': history,
        'metrics': final_metrics,
        'pooling_method': pooling_method,
        'esm_layer': esm_layer
    }


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
    for _epoch in epoch_iter:
        model.train()
        batch_train_losses = []
        for xb, lengths, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            if POOLING_METHOD == 'esm_attention':
                # Create input mask from lengths
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                output = model(xb, input_mask)
            else:
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
                if POOLING_METHOD == 'esm_attention':
                    # Create input mask from lengths
                    input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                    for i, length in enumerate(lengths):
                        input_mask[i, :length] = 1
                    val_output = model(xb, input_mask)
                else:
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
            if POOLING_METHOD == 'esm_attention':
                # Create input mask from lengths
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                out = model(xb, input_mask).squeeze(-1).cpu().numpy()
            else:
                out = model(xb).squeeze(-1).cpu().numpy()
            pred_norm.append(out)
            y_true_norm.append(yb.squeeze(-1).cpu().numpy())
    pred_norm = np.concatenate(pred_norm)
    y_true_norm = np.concatenate(y_true_norm)
    pred_log = pred_norm * y_log_std + y_log_mean
    y_true_log = y_true_norm * y_log_std + y_log_mean
    pred = pred_log #np.expm1(pred_log)
    y_true = y_true_log # np.expm1(y_true_log)

    # Clip predictions to reasonable range based on training data
    y_train_original = y_train * y_log_std + y_log_mean  # Convert back to original scale
    data_min = np.min(y_train_original)
    data_max = np.max(y_train_original)
    # Add small buffer (5% of range) to avoid being too restrictive
    buffer = 0.05 * (data_max - data_min)
    pred_unclipped = pred.copy()
    pred = np.clip(pred, data_min - buffer, data_max + buffer)

    # Report clipping statistics
    n_clipped = np.sum((pred_unclipped < data_min - buffer) | (pred_unclipped > data_max + buffer))
    if n_clipped > 0:
        print(
            f"  Clipped {n_clipped}/{len(pred)} predictions to range "
            f"[{data_min - buffer:.2f}, {data_max + buffer:.2f}]"
        )
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


def train(
    X_train, y_train,
    L,
    batch_size,
    learning_rate,
    weight_decay,
    hidden_dim1,
    hidden_dim2,
    hidden_dim3,
    hidden_dim4,
    dropout_rate,
    epochs,
    final_activation,
    device,
    POOLING_METHOD,
    LOSS_TYPE,
    y_log_std,
    y_log_mean
):
    """
    Train a protein variant regression model on all data (no validation set).
    Returns the trained model and metrics on the training set.
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

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=batch_pad_and_normalize)

    train_losses = []
    from tqdm import trange
    epoch_iter = trange(epochs, desc="Training", leave=True)
    for _epoch in epoch_iter:
        model.train()
        batch_train_losses = []
        for xb, lengths, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            if POOLING_METHOD == 'esm_attention':
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                output = model(xb, input_mask)
            else:
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
        epoch_iter.set_postfix({"train_loss": train_loss})

    # Compute metrics on training set
    model.eval()
    eval_dataset = CustomDataset(X_train, y_train)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_pad_and_normalize)
    pred_norm = []
    y_true_norm = []
    with torch.no_grad():
        for xb, lengths, yb in eval_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if POOLING_METHOD == 'esm_attention':
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                out = model(xb, input_mask).squeeze(-1).cpu().numpy()
            else:
                out = model(xb).squeeze(-1).cpu().numpy()
            pred_norm.append(out)
            y_true_norm.append(yb.squeeze(-1).cpu().numpy())
    pred_norm = np.concatenate(pred_norm)
    y_true_norm = np.concatenate(y_true_norm)
    pred_log = pred_norm * y_log_std + y_log_mean
    y_true_log = y_true_norm * y_log_std + y_log_mean
    pred = pred_log
    y_true = y_true_log



    # Clip predictions to reasonable range based on training data
    y_train_original = y_train * y_log_std + y_log_mean
    data_min = np.min(y_train_original)
    data_max = np.max(y_train_original)
    buffer = 0.05 * (data_max - data_min)
    pred = np.clip(pred, data_min - buffer, data_max + buffer)
    mse = np.mean((pred - y_true) ** 2)
    mae = np.mean(np.abs(pred - y_true))
    ss_res = np.sum((y_true - pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    pearson_r = np.corrcoef(y_true, pred)[0, 1]
    spearman_rho, spearman_p = spearmanr(y_true, pred)

    return model, dict(mse=mse, mae=mae, r2=r2, pearson_r=pearson_r, spearman_rho=spearman_rho)


def train_or_load_model(config, sequences, y, results_dir, device):
    embeddings_subdir = 'synthetic' if 'synthetic' in config.get('data', {}).get('path', '') else None
    """Train a new model or load existing one"""

    model_config = config['model']
    training_model_path = os.path.join(results_dir, f"{config['data']['name']}_training_model.pth")

    # Get embedding dimensions
    print("Getting embedding dimensions...")
    esm_model_name = model_config.get('esm_model_name', 'esmc_300m')
    sample_embeddings = get_embeddings(
        [sequences[0]],
        layer=model_config['esm_layer'],
        batch_size=1,
        dataset_name=f"{config['data']['name']}_sample",
        subdir=embeddings_subdir,
        esm_model_name=esm_model_name,
    )
    L, D = sample_embeddings.shape[1], sample_embeddings.shape[2]

    # Check for existing model
    train_model = True
    if os.path.exists(training_model_path) and not config.get('force_retrain', False):
        print(f"\nFound existing trained model at {training_model_path}")
        try:
            checkpoint = torch.load(training_model_path, map_location=device, weights_only=False)

            if 'model_state_dict' in checkpoint and 'pooling_method' in checkpoint:
                # Load model configuration
                saved_config = checkpoint.get('model_config', {})

                # Compare configurations with debugging
                current_config_values = {
                    'input_dim': D,
                    'L': L,
                    'hidden_dim1': model_config['hidden_dim1'],
                    'hidden_dim2': model_config['hidden_dim2'],
                    'hidden_dim3': model_config['hidden_dim3'],
                    'hidden_dim4': model_config['hidden_dim4'],
                    'dropout_rate': model_config['dropout_rate'],
                    'pooling_method': model_config['pooling_method']
                }

                print("Comparing model configurations:")
                print(f"Current config: {current_config_values}")
                print(f"Saved config: {saved_config}")

                # Check if config matches
                config_matches = True
                for key in ['input_dim', 'L', 'hidden_dim1', 'hidden_dim2',
                           'hidden_dim3', 'hidden_dim4', 'dropout_rate', 'pooling_method']:
                    current_val = current_config_values.get(key)
                    saved_val = saved_config.get(key)
                    if current_val != saved_val:
                        print(f"Config mismatch for {key}: current={current_val}, saved={saved_val}")
                        config_matches = False

                if config_matches:
                    print("✓ Model configurations match. Loading pre-trained model...")
                    model_cfg = {
                        'input_dim': D,
                        'L': L,
                        'hidden_dim1': model_config['hidden_dim1'],
                        'hidden_dim2': model_config['hidden_dim2'],
                        'hidden_dim3': model_config['hidden_dim3'],
                        'hidden_dim4': model_config['hidden_dim4'],
                        'dropout_rate': model_config['dropout_rate'],
                        'final_activation': model_config.get('final_activation'),
                        'pooling_method': model_config['pooling_method']
                    }

                    final_model = create_model(model_cfg, device)
                    final_model.load_state_dict(checkpoint['model_state_dict'])
                    final_model.eval()
                    train_model = False

                    # Restore global normalization parameters
                    if 'normalization_params' in checkpoint:
                        from src.igloo.data_processing import set_global_normalization_params
                        norm_params = checkpoint['normalization_params']
                        set_global_normalization_params(norm_params['mean'], norm_params['scale'])
                        print(f"Restored normalization params: mean={norm_params['mean']:.4f}, "
                              f"scale={norm_params['scale']:.4f}")
                    else:
                        print("Warning: No normalization parameters found in saved model")

                    print("Successfully loaded pre-trained model!")
                else:
                    print("✗ Model configuration doesn't match. Training new model.")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model.")

    if train_model:
        print("\nTraining model...")

        # Create model
        model_cfg = {
            'input_dim': D,
            'L': L,
            'hidden_dim1': model_config['hidden_dim1'],
            'hidden_dim2': model_config['hidden_dim2'],
            'hidden_dim3': model_config['hidden_dim3'],
            'hidden_dim4': model_config['hidden_dim4'],
            'dropout_rate': model_config['dropout_rate'],
            'final_activation': model_config.get('final_activation'),
            'pooling_method': model_config['pooling_method']
        }
        model = create_model(model_cfg, device)

        # Train model
        training_results = train_model_with_embeddings(
            model=model,
            train_sequences=sequences,
            train_targets=y,
            val_sequences=None,
            val_targets=None,
            esm_layer=model_config['esm_layer'],
            pooling_method=model_config['pooling_method'],
            batch_size=model_config['batch_size'],
            learning_rate=model_config['learning_rate'],
            num_epochs=model_config['epochs'],
            device=device,
            dataset_name=f"{config['data']['name']}_integrated_gradients",
            loss_type=model_config.get('loss_type', 'huber').lower(),
            results_dir=results_dir,
            save_path=training_model_path,
            model_config=model_cfg,
            embeddings_subdir=embeddings_subdir,
            esm_model_name=esm_model_name,
        )

        final_model = training_results['model']
        train_metrics = training_results['metrics']
    else:
        train_metrics = {'note': 'using_pretrained_model'}

    return final_model, train_model, train_metrics
