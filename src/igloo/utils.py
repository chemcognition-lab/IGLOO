import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from src.igloo.models import AttentionConvPoolingModel, ESMAttention1d, LinearRegressionModel, MeanPoolingModel

from .embedding import get_embeddings
from .plot import plot_gradients_heatmap, plot_integrated_gradients, plot_variants_vs_wt_gradients


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


def print_completion_summary(n_sequences, n_analyzed, eval_results, results_dir, dataset_name):
    """Print a summary of the completed analysis."""
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Total sequences: {n_sequences}")
    print(f"Sequences analyzed: {n_analyzed}")
    print(f"Results saved to: {results_dir}")

    if eval_results:
        print("\nModel Performance:")
        print(f"  R² = {eval_results.get('r2', 'N/A'):.4f}")
        print(f"  Pearson r = {eval_results.get('pearson_r', 'N/A'):.4f}")
        print(f"  Spearman ρ = {eval_results.get('spearman_rho', 'N/A'):.4f}")

    print("\nFiles generated:")
    print("  - Model performance plots")
    print("  - Attribution heatmaps")
    print("  - Integrated gradients plots")
    print("  - Summary CSV files")
    print("="*60)


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


def evaluate_model_performance(model, sequences, y, y_raw, y_log_mean, y_log_std, config, results_dir, device):
    """Evaluate the trained model"""
    from .data_processing import CustomDataset, batch_pad_and_normalize

    print("\nEvaluating model performance...")
    model.eval()

    model_config = config['model']

    # Get embeddings for evaluation
    print("Getting embeddings for evaluation...")

    all_embeddings = get_embeddings(
        sequences,
        layer=model_config['esm_layer'],
        batch_size=16,
        dataset_name=f"{config['data']['name']}_eval"
    )

    # Evaluate using DataLoader
    eval_dataset = CustomDataset(all_embeddings, y)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=model_config['batch_size'],
        shuffle=False,
        collate_fn=batch_pad_and_normalize
    )

    pred_norm = []
    y_true_norm = []

    with torch.no_grad():
        for xb, lengths, yb in eval_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if model_config['pooling_method'] == 'esm_attention':
                input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                for i, length in enumerate(lengths):
                    input_mask[i, :length] = 1
                out = model(xb, input_mask).squeeze(-1)
            else:
                out = model(xb).squeeze(-1)

            if out.ndim > 1:
                out = out.squeeze(-1)

            pred_norm.append(out.cpu().numpy())
            y_true_norm.append(yb.cpu().numpy().flatten())

    y_pred = np.concatenate(pred_norm)
    y_true = np.concatenate(y_true_norm)

    # Get performance metrics
    r2, pearson_r, spearman_rho, y_pred_denorm = print_model_performance(
        y_true, y_pred, y_raw, y_log_mean, y_log_std
    )

    print_extreme_value_analysis(y_raw, y_pred_denorm)

    # Create predicted vs actual plot
    print("Creating predicted vs actual plot...")
    pred_vs_actual_path = os.path.join(
        results_dir,
        f"{config['data']['name']}_predicted_vs_actual.png"
    )
    plot_predicted_vs_actual(
        y_raw, y_pred_denorm, pred_vs_actual_path,
        title=f"{config['data']['name']} Model: Predicted vs Actual {config['data']['target_column']} Values",
        r2=r2, pearson_r=pearson_r, spearman_rho=spearman_rho
    )

    return r2, pearson_r, spearman_rho, y_pred_denorm


def create_visualizations_and_exports(ig_results, r2, pearson_r, spearman_rho, train_metrics,
                                    config, results_dir, wt_seq, final_model):
    """Create visualizations and export results"""

    print("\nCreating visualizations...")

    dataset_name = config['data']['name']
    ig_config = config['integrated_gradients']

    # Create integrated gradients plots
    ig_plot_path = os.path.join(results_dir, f'{dataset_name}_integrated_gradients_analysis.png')
    plot_integrated_gradients(ig_results, ig_plot_path)

    print("\nExporting results...")

    # Create detailed results dataframe
    all_results = []
    for name, result in ig_results.items():
        for pos, aa, grad in zip(result['positions'], result['amino_acids'], result['position_gradients']):
            all_results.append({
                'sequence_name': name,
                'position': pos,
                'amino_acid': aa,
                'integrated_gradient': grad,
                'abs_gradient': abs(grad),
                'predicted_activity': result['predicted_activity'],
                'baseline_activity': result['baseline_activity'],
                'effect_vs_baseline': result['effect_vs_baseline'],
                'actual_activity': result.get('actual_activity', 'N/A'),
                'baseline_actual_activity': result.get('baseline_actual_activity', 'N/A')
            })

    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(results_dir, f'{dataset_name}_integrated_gradients_results.csv')
    results_df.to_csv(results_csv_path, index=False)

    # Create position importance summary
    position_importance = results_df.groupby('position').agg({
        'integrated_gradient': ['mean', 'std'],
        'abs_gradient': ['mean', 'max']
    }).round(4)
    position_importance.columns = ['mean_gradient', 'std_gradient', 'mean_abs_gradient', 'max_abs_gradient']
    position_importance['importance_score'] = (position_importance['mean_abs_gradient'] +
                                              position_importance['std_gradient'])
    position_importance = position_importance.sort_values('importance_score', ascending=False)

    importance_csv_path = os.path.join(results_dir, f'{dataset_name}_position_importance_summary.csv')
    position_importance.to_csv(importance_csv_path)

    # Create predicted vs actual summary
    pred_vs_actual_summary_path = os.path.join(results_dir, f'{dataset_name}_predicted_vs_actual_summary.csv')
    prediction_summary_data = []
    for name, result in ig_results.items():
        if result.get('actual_activity') is not None and isinstance(result['actual_activity'], (int, float)):
            prediction_summary_data.append({
                'sequence_name': name,
                'predicted_activity': result['predicted_activity'],
                'actual_activity': result['actual_activity'],
                'prediction_error': result['predicted_activity'] - result['actual_activity'],
                'baseline_predicted': result['baseline_activity'],
                'baseline_actual': result.get('baseline_actual_activity', 'N/A'),
                'effect_vs_baseline': result['effect_vs_baseline']
            })

    if prediction_summary_data:
        pred_summary_df = pd.DataFrame(prediction_summary_data)
        pred_summary_df.to_csv(pred_vs_actual_summary_path, index=False)
        print(f"Predicted vs actual summary saved to: {pred_vs_actual_summary_path}")

    # Create summary report
    summary_path = os.path.join(results_dir, f'{dataset_name}_integrated_gradients_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{dataset_name.upper()} INTEGRATED GRADIENTS ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("Analysis Configuration:\n")
        f.write(f"  Integration Method: {ig_config['method']}\n")
        f.write(f"  Attribution Method: {ig_config.get('attribution_method', 'magnitude')}\n")
        f.write(f"  ESM Layer: {config['model']['esm_layer']}\n")
        f.write(f"  Pooling Method: {config['model']['pooling_method']}\n")
        f.write(f"  Integration Steps: {ig_config.get('steps', 50)}\n\n")

        f.write("Model Performance:\n")
        f.write(f"  R² = {r2:.4f}\n")
        f.write(f"  Pearson r = {pearson_r:.4f}\n")
        f.write(f"  Spearman ρ = {spearman_rho:.4f}\n\n")

        f.write("Sequences Analyzed:\n")
        for name, result in ig_results.items():
            f.write(f"  {name}:\n")
            f.write(f"    Predicted Activity: {result['predicted_activity']:.4f}\n")
            f.write(f"    Effect vs Baseline: {result['effect_vs_baseline']:.4f}\n")
            if result.get('actual_activity') is not None:
                f.write(f"    Actual Activity: {result['actual_activity']:.4f}\n")
            f.write("\n")

        f.write("TOP 10 MOST IMPORTANT POSITIONS (across all sequences):\n")
        for i, (pos, row) in enumerate(position_importance.head(10).iterrows()):
            aa = wt_seq[pos-1] if pos <= len(wt_seq) else 'N/A'
            f.write(f"{i+1:2d}. Position {pos} ({aa}): importance = {row['importance_score']:.4f}\n")

    # Create heatmap
    print("Creating gradient heatmap...")
    heatmap_path = os.path.join(results_dir, f'{dataset_name}_integrated_gradients_heatmap.png')
    plot_gradients_heatmap(results_df, f'{dataset_name} Integrated Gradients Heatmap Across Sequences',
                          heatmap_path)

    # Create comparison plots
    print("Creating comparison plots...")
    comparison_path = os.path.join(results_dir, f'{dataset_name}_variants_vs_WT_gradients.png')
    plot_variants_vs_wt_gradients(ig_results, comparison_path)

    # Save model and analysis parameters
    analysis_model_path = os.path.join(results_dir, f'{dataset_name}_ig_analysis.pth')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_config': {
            'input_dim': config['model'].get('input_dim'),
            'L': config['model'].get('L'),
            'hidden_dim1': config['model']['hidden_dim1'],
            'hidden_dim2': config['model']['hidden_dim2'],
            'hidden_dim3': config['model']['hidden_dim3'],
            'hidden_dim4': config['model']['hidden_dim4'],
            'dropout_rate': config['model']['dropout_rate'],
            'final_activation': config['model'].get('final_activation'),
            'pooling_method': config['model']['pooling_method']
        },
        'analysis_config': {
            'esm_layer': config['model']['esm_layer'],
            'ig_method': ig_config['method'],
            'attribution_method': ig_config.get('attribution_method', 'magnitude'),
            'ig_steps': ig_config.get('steps', 50),
            'ig_batch_size': ig_config.get('batch_size', 10),
            'target_column': config['data']['target_column'],
            'normalization': config['data'].get('normalization', 'z-score')
        },
        'performance': {
            'r2': r2,
            'pearson_r': pearson_r,
            'spearman_rho': spearman_rho
        },
        'data_info': {
            'target_mean': config.get('y_log_mean', 0.0),
            'target_std': config.get('y_log_std', 1.0),
            'wt_sequence': wt_seq
        },
        'config': config
    }, analysis_model_path)

    # Create eval_results dict from the passed values
    eval_results_dict = {
        'r2': r2,
        'pearson_r': pearson_r,
        'spearman_rho': spearman_rho
    }

    print_completion_summary(
        len(ig_results),  # n_sequences
        len(ig_results),  # n_analyzed (same as n_sequences for this case)
        eval_results_dict,  # eval_results
        results_dir,  # results_dir
        config['data']['name']  # dataset_name
    )
