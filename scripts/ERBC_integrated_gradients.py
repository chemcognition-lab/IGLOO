# ERBC_integrated_gradients.py

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Train a model on ERBC variant data using ESM embeddings,
# and perform TRUE integrated gradients analysis using the integrated_gradients_tokens function.
# This is an alternative to the brute-force mutation approach.
#
# UPDATED: Now uses the new train_model_with_embeddings function and consistent
# embedding-based integrated gradients approach for perfect training-inference consistency.

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import analysis modules
from src.igloo.data_processing import CustomDataset, batch_pad_and_normalize
from src.igloo.embedding import get_embeddings
from src.igloo.plot import plot_gradients_heatmap, plot_integrated_gradients, plot_variants_vs_wt_gradients
from src.igloo.sequence_ig import (
    analyze_sequence_with_integrated_gradients,
    create_combined_model,
    load_esm_model,
)
from src.igloo.train_and_eval import train_model_with_embeddings
from src.igloo.utils import (
    create_model,
    plot_predicted_vs_actual,
    print_analysis_sections,
    print_completion_summary,
    print_data_info,
    print_extreme_value_analysis,
    print_ig_sequence_results,
    print_model_performance,
    print_sequence_analysis_info,
    print_training_info,
)

# Determine script directory and set paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Paths
data_path = os.path.join(project_root, 'data', 'Data_Donya_ERBC_Converted.csv')
results_dir = os.path.join(project_root, 'results', 'ERBC_ig')

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# 1. Load dataset
print(f"Loading data from {data_path}")
df = pd.read_csv(data_path)
col = 'log_production'
# Remove rows with NaN or Inf in target
df = df[df[col].notna()]
df = df[np.isfinite(df[col])]
sequences = df['SEQ'].astype(str).tolist()
y_raw = df[col].values.astype(np.float32)

# Data range for prediction clipping (keep predictions within realistic bounds)
DATA_MIN = float(np.min(y_raw))
DATA_MAX = float(np.max(y_raw))
print(f"Data range for clipping: [{DATA_MIN:.4f}, {DATA_MAX:.4f}]")

# Config: select ESM layer and pooling method
ESM_LAYER = 33
POOLING_METHOD = 'esm_attention'  # Changed from 'esm_attention' to use deeper model architecture
LOSS_TYPE = 'huber'

# Integrated Gradients Configuration
IG_METHOD = 'embedding'  # 'embedding' or 'gumbel' - using embedding for consistency with new approach
GUMBEL_TEMPERATURE = 1.0  # Temperature for Gumbel-Softmax (only used if method='gumbel')
HARD_GUMBEL = False  # Whether to use hard Gumbel-Softmax

# Hyperparameters
hidden_dim1 = 1024
hidden_dim2 = 512
hidden_dim3 = 255
hidden_dim4 = 128
dropout_rate = 0.2  # Reduced from 0.5 - high dropout prevents learning
learning_rate = 1e-3  # Increased from 5e-4 - model needs higher learning rate
weight_decay = 1e-4
batch_size = 256
epochs = 100  # Increased to allow more learning
final_activation = None

# Z-score normalize target and print data info
y, y_log_mean, y_log_std = print_data_info(df, col, sequences, y_raw)

# Get wild-type sequence (marked as 'WT')
wt_df = df[df['Variant'] == 'WT']
if len(wt_df) == 0:
    raise ValueError("No wild-type sequence found! Expected a variant marked 'WT'")
wt_seq = wt_df['SEQ'].iloc[0]
wt_idx = df[df['Variant'] == 'WT'].index[0]
print(f"Using WT sequence as baseline: {wt_seq[:50]}...")
print(f"WT sequence length: {len(wt_seq)}")
print(f"WT log_production: {wt_df['log_production'].iloc[0]:.4f}")

# 2. Get embedding dimensions for model configuration
print("Getting embedding dimensions...")
# Get a sample embedding to determine dimensions
sample_embeddings = get_embeddings([sequences[0]], layer=ESM_LAYER, batch_size=1, dataset_name='ERBC_sample')
N, L, D = len(sequences), sample_embeddings.shape[1], sample_embeddings.shape[2]

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Check for existing trained model and load if available
training_model_path = os.path.join(results_dir, 'ERBC_training_model.pth')
final_model = None
train_model = True

if os.path.exists(training_model_path):
    print(f"\nFound existing trained model at {training_model_path}")
    print("Loading pre-trained model...")
    try:
        checkpoint = torch.load(training_model_path, map_location=device, weights_only=False)

        # Check if this was saved by train_model_with_embeddings (has specific keys)
        if 'model_state_dict' in checkpoint and 'pooling_method' in checkpoint and 'esm_layer' in checkpoint:
            # Verify configuration matches current parameters
            config_match = (
                checkpoint.get('pooling_method') == POOLING_METHOD and
                checkpoint.get('esm_layer') == ESM_LAYER
            )

            if config_match:
                # Create model with current configuration
                model_config = {
                    'input_dim': D,
                    'L': L,
                    'hidden_dim1': hidden_dim1,
                    'hidden_dim2': hidden_dim2,
                    'hidden_dim3': hidden_dim3,
                    'hidden_dim4': hidden_dim4,
                    'dropout_rate': dropout_rate,
                    'final_activation': final_activation,
                    'pooling_method': POOLING_METHOD
                }
                final_model = create_model(model_config, device)
                final_model.load_state_dict(checkpoint['model_state_dict'])
                final_model.eval()

                print(f"Successfully loaded pre-trained model from epoch {checkpoint.get('epoch', 'unknown')+1}")
                if 'train_loss' in checkpoint and 'val_loss' in checkpoint:
                    print(f"  Final training loss: {checkpoint['train_loss']:.6f}")
                    if checkpoint['val_loss'] is not None:
                        print(f"  Final validation loss: {checkpoint['val_loss']:.6f}")

                train_model = False  # Skip training
            else:
                print("Model configuration doesn't match current parameters. Training new model.")
        else:
            print("Model format not recognized. Training new model.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model.")
else:
    print("\nNo existing trained model found. Training new model.")

# Note: Global normalization is disabled in favor of per-batch normalization
# This approach provides better training stability and prediction consistency

# Train model if needed
if train_model:
    print("\nTraining model on all data for integrated gradients analysis...")

    # Create the model first
    model_config = {
        'input_dim': D,
        'L': L,
        'hidden_dim1': hidden_dim1,
        'hidden_dim2': hidden_dim2,
        'hidden_dim3': hidden_dim3,
        'hidden_dim4': hidden_dim4,
        'dropout_rate': dropout_rate,
        'final_activation': final_activation,
        'pooling_method': POOLING_METHOD
    }
    model = create_model(model_config, device)

    # Use the new training function that ensures consistency with integrated gradients
    training_model_path = os.path.join(results_dir, 'ERBC_training_model.pth')
    training_results = train_model_with_embeddings(
        model=model,
        train_sequences=sequences,
        train_targets=y,
        val_sequences=None,  # Train on all data
        val_targets=None,
        esm_layer=ESM_LAYER,
        pooling_method=POOLING_METHOD,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=epochs,
        device=device,
        dataset_name='ERBC_integrated_gradients',
        loss_type=LOSS_TYPE.lower(),
        results_dir=results_dir,
        save_path=training_model_path  # Save the best model during training
    )

    final_model = training_results['model']
    train_metrics = training_results['metrics']
else:
    # Create dummy train_metrics for consistency when using pre-trained model
    train_metrics = {'note': 'using_pretrained_model'}

print_training_info(train_metrics, train_model)

# 4. Evaluate model performance
print("\nEvaluating model performance...")
final_model.eval()

# Get embeddings for evaluation (same approach as training)
print("Getting embeddings for evaluation...")
all_embeddings = get_embeddings(sequences, layer=ESM_LAYER, batch_size=16, dataset_name='ERBC_eval')

# Use DataLoader approach (same as training) for consistency
eval_dataset = CustomDataset(all_embeddings, y)
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
            out = final_model(xb, input_mask).cpu().numpy()
        else:
            out = final_model(xb).cpu().numpy()

        # Handle different output shapes - ensure we get (batch_size,)
        if out.ndim > 1:
            out = out.squeeze(-1)
        pred_norm.append(out)
        y_true_norm.append(yb.cpu().numpy().flatten())

y_pred = np.concatenate(pred_norm)
y_true = np.concatenate(y_true_norm)

# Print comprehensive model performance metrics
r2, pearson_r, spearman_rho, y_pred_denorm = print_model_performance(y_true, y_pred, y_raw, y_log_mean, y_log_std)

# Print extreme value analysis
print_extreme_value_analysis(y_raw, y_pred_denorm)

# Create predicted vs actual plot
print("Creating predicted vs actual plot...")
pred_vs_actual_path = os.path.join(results_dir, 'ERBC_predicted_vs_actual.png')
plot_predicted_vs_actual(
    y_raw, y_pred_denorm, pred_vs_actual_path,
    title="ERBC Model: Predicted vs Actual log_production Values",
    r2=r2, pearson_r=pearson_r, spearman_rho=spearman_rho
)

# 5. Perform integrated gradients analysis
print_analysis_sections("PERFORMING INTEGRATED GRADIENTS ANALYSIS")

# Load ESM model for sequence analysis
print("Loading ESM model for integrated gradients...")
esm_model, alphabet = load_esm_model('esm2_t33_650M_UR50D', device)

# Create combined model using the specified method
print(f"Creating combined model using {IG_METHOD} method...")
# Note: The new integrated gradients approach uses the underlying predictor model
# with the same DataLoader pipeline as training for perfect consistency
combined_model = create_combined_model(
    esm_model=esm_model,
    predictor_model=final_model,
    esm_layer=ESM_LAYER,
    pooling_method=POOLING_METHOD,
    method=IG_METHOD,
    gumbel_temperature=GUMBEL_TEMPERATURE,
    hard_gumbel=HARD_GUMBEL,
    device=device
)

# Analyze several interesting sequences using integrated gradients
sequences_to_analyze = []

# 1. Wild-type sequence
sequences_to_analyze.append(('WT', wt_seq, wt_seq))

# 2. Find some high and low activity sequences from the dataset
sorted_indices = np.argsort(y_raw)
low_activity_idx = sorted_indices[0]  # Lowest activity
high_activity_idx = sorted_indices[-1]  # Highest activity
mid_activity_idx = sorted_indices[len(sorted_indices)//2]  # Median activity

# Get a few more interesting variants
high_10_idx = sorted_indices[-10]  # 10th highest
low_10_idx = sorted_indices[9]  # 10th lowest

sequences_to_analyze.extend([
    ('Lowest_Activity', sequences[low_activity_idx], wt_seq),
    ('Highest_Activity', sequences[high_activity_idx], wt_seq),
    ('Median_Activity', sequences[mid_activity_idx], wt_seq),
    ('High_10th', sequences[high_10_idx], wt_seq),
    ('Low_10th', sequences[low_10_idx], wt_seq)
])

print_sequence_analysis_info(sequences_to_analyze, y_raw, sequences)

# Perform integrated gradients analysis
ig_results = {}
for name, sequence, baseline in sequences_to_analyze:
    print(f"\nAnalyzing {name}...")
    # Get actual activity for comparison
    if sequence in sequences:
        seq_idx = sequences.index(sequence)
        actual_activity_raw = y_raw[seq_idx]
        actual_activity_normalized = y[seq_idx]
    else:
        actual_activity_raw = "N/A"
        actual_activity_normalized = "N/A"

    # Get actual activity for plotting (if available)
    actual_activity_plot = actual_activity_raw if isinstance(actual_activity_raw, (int, float)) else None
    baseline_idx = sequences.index(baseline) if baseline in sequences else None
    baseline_actual_activity = y_raw[baseline_idx] if baseline_idx is not None else None

    result = analyze_sequence_with_integrated_gradients(
        sequence=sequence,
        baseline_sequence=baseline,
        model=combined_model,
        alphabet=alphabet,
        device=device,
        y_log_mean=y_log_mean,
        y_log_std=y_log_std,
        steps=50,
        batch_size=10,
        method=IG_METHOD,
        actual_activity=actual_activity_plot,
        baseline_actual_activity=baseline_actual_activity,
        plot_predictions=True,  # Enable plotting
        sequence_name=name,
        baseline_name="WT",
        save_plot_path=os.path.join(results_dir, f'ERBC_{name}_predicted_vs_actual.png'),
        clip_min=DATA_MIN,
        clip_max=DATA_MAX
    )
    ig_results[name] = result

    print_ig_sequence_results(name, actual_activity_raw, result, None)

# 6. Create visualizations
print("\nCreating visualizations...")

# Create integrated gradients plots
ig_plot_path = os.path.join(results_dir, 'ERBC_integrated_gradients_analysis.png')
plot_integrated_gradients(ig_results, ig_plot_path)

# 7. Export results
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
results_csv_path = os.path.join(results_dir, 'ERBC_integrated_gradients_results.csv')
results_df.to_csv(results_csv_path, index=False)

# Create position importance summary
position_importance = results_df.groupby('position').agg({
    'integrated_gradient': ['mean', 'std'],
    'abs_gradient': ['mean', 'max']
}).round(4)
position_importance.columns = ['mean_gradient', 'std_gradient', 'mean_abs_gradient', 'max_abs_gradient']
position_importance['importance_score'] = position_importance['mean_abs_gradient'] + position_importance['std_gradient']
position_importance = position_importance.sort_values('importance_score', ascending=False)

importance_csv_path = os.path.join(results_dir, 'ERBC_position_importance_summary.csv')
position_importance.to_csv(importance_csv_path)

# Create predicted vs actual summary
pred_vs_actual_summary_path = os.path.join(results_dir, 'ERBC_predicted_vs_actual_summary.csv')
prediction_summary_data = []
for name, result in ig_results.items():
    if result.get('actual_activity') is not None and isinstance(result['actual_activity'], (int, float)):
        pred_error = abs(result['predicted_activity'] - result['actual_activity'])
        prediction_summary_data.append({
            'sequence_name': name,
            'predicted_activity': result['predicted_activity'],
            'actual_activity': result['actual_activity'],
            'prediction_error': pred_error,
            'baseline_predicted': result['baseline_activity'],
            'baseline_actual': result.get('baseline_actual_activity', 'N/A'),
            'effect_vs_baseline': result['effect_vs_baseline']
        })

if prediction_summary_data:
    pred_summary_df = pd.DataFrame(prediction_summary_data)
    pred_summary_df.to_csv(pred_vs_actual_summary_path, index=False)
    print(f"Predicted vs actual summary saved to: {pred_vs_actual_summary_path}")

# Create summary report
summary_path = os.path.join(results_dir, 'ERBC_integrated_gradients_summary.txt')
with open(summary_path, 'w') as f:
    f.write("ERBC INTEGRATED GRADIENTS ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n\n")

    f.write("Analysis Configuration:\n")
    f.write(f"  Integration Method: {IG_METHOD}\n")
    if IG_METHOD == 'gumbel':
        f.write(f"  Gumbel Temperature: {GUMBEL_TEMPERATURE}\n")
        f.write(f"  Hard Gumbel: {HARD_GUMBEL}\n")
    f.write(f"  ESM Layer: {ESM_LAYER}\n")
    f.write(f"  Pooling Method: {POOLING_METHOD}\n\n")

    f.write("Model Performance:\n")
    f.write(f"  R² = {r2:.4f}\n")
    f.write(f"  Pearson r = {pearson_r:.4f}\n")
    f.write(f"  Spearman ρ = {spearman_rho:.4f}\n\n")

    f.write("Sequences Analyzed:\n")
    for name, result in ig_results.items():
        f.write(f"  {name}:\n")
        f.write(f"    Predicted Activity: {result['predicted_activity']:.4f}\n")
        if result.get('actual_activity') is not None and isinstance(result['actual_activity'], (int, float)):
            f.write(f"    Actual Activity: {result['actual_activity']:.4f}\n")
            pred_error = abs(result['predicted_activity'] - result['actual_activity'])
            f.write(f"    Prediction Error: {pred_error:.4f}\n")
        f.write(f"    Effect vs Baseline: {result['effect_vs_baseline']:.4f}\n")
        top_positions = np.argsort(np.abs(result['position_gradients']))[-5:][::-1] + 1
        f.write(f"    Top 5 Important Positions: {list(top_positions)}\n\n")

    f.write("TOP 10 MOST IMPORTANT POSITIONS (across all sequences):\n")
    for i, (pos, row) in enumerate(position_importance.head(10).iterrows()):
        aa = wt_seq[pos-1] if pos <= len(wt_seq) else 'N/A'
        f.write(f"{i+1:2d}. Position {pos} ({aa}): importance = {row['importance_score']:.4f}\n")

# 8. Create heatmap of gradients across sequences
print("Creating gradient heatmap...")
heatmap_path = os.path.join(results_dir, 'ERBC_integrated_gradients_heatmap.png')
plot_gradients_heatmap(results_df, 'ERBC Integrated Gradients Heatmap Across Sequences', heatmap_path)

# 9. Create comparison with wild-type
print("Creating comparison plots...")

# Plot showing difference between variants and WT
comparison_path = os.path.join(results_dir, 'ERBC_variants_vs_WT_gradients.png')
plot_variants_vs_wt_gradients(ig_results, comparison_path)

print_completion_summary(results_dir)

# Save model and analysis parameters (analysis results file)
analysis_model_path = os.path.join(results_dir, 'ERBC_ig_analysis.pth')
torch.save({
    'model_state_dict': final_model.state_dict(),
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
    'analysis_config': {
        'esm_layer': ESM_LAYER,
        'ig_steps': 50,
        'ig_batch_size': 10,
        'target_column': col,
        'normalization': 'z-score'
    },
    'performance': {
        'r2': r2,
        'pearson_r': pearson_r,
        'spearman_rho': spearman_rho
    },
    'data_info': {
        'target_mean': y_log_mean,
        'target_std': y_log_std,
        'wt_sequence': wt_seq
    },
    'training_info': {
        'was_retrained': train_model,
        'train_metrics': train_metrics if train_model else 'using_pretrained'
    }
}, analysis_model_path)
print("  - ERBC_ig_analysis.pth: Analysis results and model saved")
if train_model:
    print("  - ERBC_training_model.pth: Best trained model saved automatically during training")
