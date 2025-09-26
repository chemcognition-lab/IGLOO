import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.metrics import r2_score

from src.igloo.models import AttentionConvPoolingModel, ESMAttention1d, LinearRegressionModel, MeanPoolingModel


def print_data_info(df, col, sequences, y_raw):
    """Print dataset information and statistics."""
    print(f"Total dataset size: {len(sequences)} sequences")
    print("Applying z-score normalization to target...")
    y_log_mean = np.mean(y_raw)
    y_log_std = np.std(y_raw)
    y = (y_raw - y_log_mean) / y_log_std
    print(f"Target mean: {y_log_mean:.4f}, std: {y_log_std:.4f}")
    print(f"Normalized target range: {np.min(y):.4f} to {np.max(y):.4f}")
    return y, y_log_mean, y_log_std


def print_model_performance(y_true, y_pred, y_raw, y_log_mean, y_log_std, clip_predictions=True):
    """Print comprehensive model performance metrics and diagnostics."""
    # Compute metrics
    r2 = r2_score(y_true, y_pred)
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    spearman_rho, _ = stats.spearmanr(y_true, y_pred)

    print("Model Performance:")
    print(f"  R² = {r2:.4f}")
    print(f"  Pearson r = {pearson_r:.4f}")
    print(f"  Spearman ρ = {spearman_rho:.4f}")

    # Add diagnostic information about prediction distribution
    y_pred_denorm = y_pred * y_log_std + y_log_mean

    # Apply clipping to keep predictions within realistic bounds
    if clip_predictions:
        data_min, data_max = np.min(y_raw), np.max(y_raw)
        y_pred_denorm_unclipped = y_pred_denorm.copy()
        y_pred_denorm = np.clip(y_pred_denorm, data_min, data_max)

        # Report clipping statistics
        n_clipped_low = np.sum(y_pred_denorm_unclipped < data_min)
        n_clipped_high = np.sum(y_pred_denorm_unclipped > data_max)
        if n_clipped_low > 0 or n_clipped_high > 0:
            print("\nPrediction Clipping Applied:")
            print(f"  Clipped {n_clipped_low} predictions below {data_min:.4f}")
            print(f"  Clipped {n_clipped_high} predictions above {data_max:.4f}")
            total_clipped = n_clipped_low + n_clipped_high
            clipped_pct = 100 * total_clipped / len(y_pred_denorm)
            print(f"  Total clipped: {total_clipped}/{len(y_pred_denorm)} ({clipped_pct:.1f}%)")
    print("\nPrediction Diagnostics:")
    print(f"  Raw target range: [{np.min(y_raw):.4f}, {np.max(y_raw):.4f}]")
    print(f"  Raw target mean: {np.mean(y_raw):.4f}, std: {np.std(y_raw):.4f}")
    print(f"  Predicted range: [{np.min(y_pred_denorm):.4f}, {np.max(y_pred_denorm):.4f}]")
    print(f"  Predicted mean: {np.mean(y_pred_denorm):.4f}, std: {np.std(y_pred_denorm):.4f}")
    print(f"  Prediction std ratio (predicted/actual): {np.std(y_pred_denorm)/np.std(y_raw):.3f}")

    return r2, pearson_r, spearman_rho, y_pred_denorm


def print_extreme_value_analysis(y_raw, y_pred_denorm):
    """Print analysis of model performance on extreme values."""
    # Show how well the model predicts extreme values
    sorted_indices = np.argsort(y_raw)
    extreme_low_idx = sorted_indices[:5]  # 5 lowest
    extreme_high_idx = sorted_indices[-5:]  # 5 highest

    print("\nExtreme Value Performance:")
    print("5 Lowest actual values vs predictions:")
    for i, idx in enumerate(extreme_low_idx):
        actual_val = y_raw[idx]
        pred_val = y_pred_denorm[idx]
        error_val = abs(actual_val - pred_val)
        print(f"  #{i+1}: actual={actual_val:.4f}, predicted={pred_val:.4f}, error={error_val:.4f}")

    print("5 Highest actual values vs predictions:")
    for i, idx in enumerate(extreme_high_idx):
        actual_val = y_raw[idx]
        pred_val = y_pred_denorm[idx]
        error_val = abs(actual_val - pred_val)
        print(f"  #{i+1}: actual={actual_val:.4f}, predicted={pred_val:.4f}, error={error_val:.4f}")


def print_analysis_sections(section_name, width=60):
    """Print formatted section headers."""
    print("\n" + "="*width)
    print(section_name)
    print("="*width)


def print_sequence_analysis_info(sequences_to_analyze, y_raw, sequences):
    """Print information about sequences to be analyzed."""
    print(f"\nAnalyzing {len(sequences_to_analyze)} sequences with integrated gradients:")
    for name, seq, _ in sequences_to_analyze:
        actual_activity = y_raw[sequences.index(seq)] if seq in sequences else "N/A"
        print(f"  {name}: actual activity = {actual_activity}")


def print_ig_sequence_results(name, actual_activity_raw, result, pred_raw_training_method=None):
    """Print integrated gradients results for a single sequence."""
    if isinstance(actual_activity_raw, (int, float)):
        print(f"  Actual activity: {actual_activity_raw:.4f}")
    else:
        print(f"  Actual activity: {actual_activity_raw}")
    print(f"  IG method prediction: {result['predicted_activity']:.4f}")
    print(f"  Baseline activity: {result['baseline_activity']:.4f}")
    print(f"  Effect vs baseline: {result['effect_vs_baseline']:.4f}")

    # Compare prediction methods if training method prediction is available
    if (pred_raw_training_method is not None and
        isinstance(actual_activity_raw, (int, float))):
        training_error = pred_raw_training_method - actual_activity_raw
        ig_error = result['predicted_activity'] - actual_activity_raw
        print(f"  Training method error: {training_error:.4f}")
        print(f"  IG method error: {ig_error:.4f}")
        print(f"  Prediction difference: {abs(pred_raw_training_method - result['predicted_activity']):.4f}")
        print(f"  Relative error (training): {abs(training_error) / abs(actual_activity_raw) * 100:.1f}%")
        print(f"  Relative error (IG): {abs(ig_error) / abs(actual_activity_raw) * 100:.1f}%")

        # Check if the problem is with embedding differences
        if abs(pred_raw_training_method - result['predicted_activity']) > 1.0:
            print("  WARNING: Large difference between training and IG predictions!")
            print("  This suggests embedding/preprocessing differences.")

    print(f"  Top 5 gradient positions: {np.argsort(np.abs(result['position_gradients']))[-5:][::-1] + 1}")


def print_training_info(train_metrics, train_model):
    """Print training completion information."""
    if train_model:
        print("Training completed.")
        print("Training metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("Using pre-trained model for analysis.")


def plot_predicted_vs_actual(y_true, y_pred, save_path, title="Predicted vs Actual Values",
                            r2=None, pearson_r=None, spearman_rho=None):
    """
    Create a predicted vs actual scatter plot with performance metrics.

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        save_path: Path to save the plot
        title: Title for the plot
        r2: R² score (optional, will be calculated if not provided)
        pearson_r: Pearson correlation (optional, will be calculated if not provided)
        spearman_rho: Spearman correlation (optional, will be calculated if not provided)
    """
    # Calculate metrics if not provided
    if r2 is None:
        r2 = r2_score(y_true, y_pred)
    if pearson_r is None:
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
    if spearman_rho is None:
        spearman_rho, _ = stats.spearmanr(y_true, y_pred)

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

    # Perfect prediction line (y=x)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Add best fit line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), 'b-', alpha=0.8, linewidth=2, label=f'Best Fit (slope={z[0]:.3f})')

    # Labels and title
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    # Add metrics text box
    metrics_text = f'R² = {r2:.4f}\nPearson r = {pearson_r:.4f}\nSpearman ρ = {spearman_rho:.4f}'
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top', fontsize=11)

    # Add sample size
    plt.text(0.95, 0.05, f'n = {len(y_true)}', transform=plt.gca().transAxes,
             horizontalalignment='right', verticalalignment='bottom', fontsize=10)

    # Grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')

    # Make plot square and tight
    plt.axis('equal')
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Predicted vs Actual plot saved to: {save_path}")


def print_completion_summary(results_dir):
    """Print final completion summary and file information."""
    # Extract protein name from results_dir path
    import os
    dir_name = os.path.basename(results_dir)
    # Get the part before the first underscore
    protein_name = dir_name.split('_')[0]
    # Capitalize first letter for display
    protein_display = protein_name.capitalize()

    print("\n" + "="*60)
    print(f"{protein_display.upper()} INTEGRATED GRADIENTS ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Results saved to: {results_dir}")
    print("\nGenerated files:")
    print(f"  - {protein_name}_integrated_gradients_analysis.png: Position-wise gradients for each sequence")
    print(f"  - {protein_name}_integrated_gradients_heatmap.png: Heatmap comparing gradients across sequences")
    print(f"  - {protein_name}_variants_vs_WT_gradients.png: Gradient differences compared to wild-type")
    print(f"  - {protein_name}_predicted_vs_actual.png: Model performance scatter plot")
    print(f"  - {protein_name}_integrated_gradients_results.csv: Detailed results for all positions")
    print(f"  - {protein_name}_position_importance_summary.csv: Position importance ranking")
    print(f"  - {protein_name}_integrated_gradients_summary.txt: Human-readable summary")
    print("\nKey insights:")
    print(f"1. Check '{protein_name}_position_importance_summary.csv' for most important positions")
    print(f"2. Use '{protein_name}_integrated_gradients_heatmap.png' to compare gradient patterns")
    print(f"3. Review '{protein_name}_integrated_gradients_analysis.png' for position-specific effects")
    print(f"4. Compare '{protein_name}_variants_vs_WT_gradients.png' to see variant-specific changes")
    print(f"5. View '{protein_name}_predicted_vs_actual.png' for overall model performance")
    print("6. Integrated gradients show which positions contribute most to predictions")


def create_model(config, device):
    """Create a model from configuration dictionary."""
    input_dim = config['input_dim']
    L = config['L']
    pooling_method = config['pooling_method']

    if pooling_method == 'mean':
        model = MeanPoolingModel(
            L, input_dim,
            hidden_dim1=config['hidden_dim1'],
            hidden_dim2=config['hidden_dim2'],
            hidden_dim3=config['hidden_dim3'],
            hidden_dim4=config['hidden_dim4'],
            dropout_rate=config['dropout_rate'],
            final_activation=config['final_activation'],
            target_min=None, target_max=None
        )
    elif pooling_method == 'attention':
        model = AttentionConvPoolingModel(
            L, input_dim,
            conv_dim=256, kernel_size=5,
            hidden_dim1=config['hidden_dim1'],
            hidden_dim2=config['hidden_dim2'],
            hidden_dim3=config['hidden_dim3'],
            hidden_dim4=config['hidden_dim4'],
            dropout_rate=config['dropout_rate'],
            final_activation=config['final_activation'],
            target_min=None, target_max=None
        )
    elif pooling_method == 'linear':
        model = LinearRegressionModel(
            L, input_dim,
            final_activation=config['final_activation'],
            target_min=None, target_max=None
        )
    elif pooling_method == 'esm_attention':
        model = ESMAttention1d(max_length=L, d_embedding=input_dim)
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method}")

    return model.to(device)
