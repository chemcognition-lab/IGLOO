# PafA_integrated_gradients.py

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Train a model on PafA variant data using ESM embeddings,
# and perform TRUE integrated gradients analysis using the integrated_gradients_tokens function.
# This is an alternative to the brute-force mutation approach.

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import r2_score

# Import analysis modules
from src.igloo.embedding import get_embeddings
from src.igloo.plot import plot_gradients_heatmap, plot_integrated_gradients, plot_variants_vs_wt_gradients
from src.igloo.sequence_ig import (
    ESMBasedPredictor,
    analyze_sequence_with_integrated_gradients,
    load_esm_model,
)
from src.igloo.train_and_eval import train
from src.igloo.utils import create_model

# Determine script directory and set paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Paths
data_path = os.path.join(project_root, 'data', 'PafA_data_processed.csv')
results_dir = os.path.join(project_root, 'results', 'PafA_ig')

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

print(f"Total dataset size: {len(sequences)} sequences")

# Config: select ESM layer and pooling method
ESM_LAYER = 33
POOLING_METHOD = 'mean'  # Changed from 'esm_attention' to use deeper model architecture
LOSS_TYPE = 'huber'

# Hyperparameters
hidden_dim1 = 1024
hidden_dim2 = 512
hidden_dim3 = 255
hidden_dim4 = 128
dropout_rate = 0.5
learning_rate = 5e-4
weight_decay = 1e-4
batch_size = 256
epochs = 50
final_activation = None

# Z-score normalize target (no log transformation for PafA)
print("Applying z-score normalization to target...")
y_log_mean = np.mean(y_raw)
y_log_std = np.std(y_raw)
y = (y_raw - y_log_mean) / y_log_std
print(f"Target mean: {y_log_mean:.4f}, std: {y_log_std:.4f}")
print(f"Normalized target range: {np.min(y):.4f} to {np.max(y):.4f}")

# Get wild-type sequence (last in the dataset)
wt_seq = df['SEQ'].iloc[-1]
print(f"Using WT sequence as baseline: {wt_seq[:50]}...")
print(f"WT sequence length: {len(wt_seq)}")

# 2. Get ESM embeddings
print("Computing ESM embeddings...")
all_reps = get_embeddings(sequences, layer=ESM_LAYER, batch_size=16, dataset_name='PafA_combined')
N, L, D = all_reps.shape

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Check for existing model and load if available
model_path = os.path.join(results_dir, 'PafA_ig_model.pth')
final_model = None

if os.path.exists(model_path):
    print(f"\nFound existing model at {model_path}")
    print("Loading pre-trained model...")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_config = checkpoint['model_config']

        # Verify configuration matches current parameters
        config_match = (
            model_config.get('hidden_dim1') == hidden_dim1 and
            model_config.get('hidden_dim2') == hidden_dim2 and
            model_config.get('hidden_dim3') == hidden_dim3 and
            model_config.get('hidden_dim4') == hidden_dim4 and
            model_config.get('dropout_rate') == dropout_rate and
            model_config.get('final_activation') == final_activation and
            model_config.get('pooling_method') == POOLING_METHOD and
            model_config.get('input_dim') == D and
            model_config.get('L') == L
        )

        if config_match:
            final_model = create_model(model_config, device)
            final_model.load_state_dict(checkpoint['model_state_dict'])
            final_model.eval()

            # Load performance metrics if available
            if 'performance' in checkpoint:
                perf = checkpoint['performance']
                print("Pre-trained model performance:")
                for k, v in perf.items():
                    print(f"  {k}: {v:.4f}")

            print("Successfully loaded pre-trained model!")
            train_model = False
        else:
            print("Model configuration doesn't match current parameters. Training new model.")
            train_model = True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model.")
        train_model = True
else:
    print("\nNo existing model found. Training new model.")
    train_model = True

# Train model if needed
if train_model:
    print("\nTraining model on all data for integrated gradients analysis...")
    final_model, train_metrics = train(
        all_reps, y,
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
        y_log_std=y_log_std,
        y_log_mean=y_log_mean
    )

    print("Training completed.")
    print("Training metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
else:
    print("Using pre-trained model for analysis.")
    # Create dummy train_metrics for consistency
    train_metrics = {'note': 'using_pretrained_model'}

# 4. Evaluate model performance
print("\nEvaluating model performance...")
final_model.eval()

# Use direct tensor conversion (same as verification method) to avoid DataLoader normalization issues
with torch.no_grad():
    all_reps_tensor = torch.tensor(all_reps, dtype=torch.float32, device=device)
    if POOLING_METHOD == 'esm_attention':
        # Create proper input mask for all sequences
        input_mask = torch.ones(all_reps_tensor.size(0), all_reps_tensor.size(1), device=device)
        y_pred = final_model(all_reps_tensor, input_mask).squeeze(-1).cpu().numpy()
    else:
        y_pred = final_model(all_reps_tensor).squeeze(-1).cpu().numpy()

y_true = y  # Use normalized targets for comparison

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
print("\nPrediction Diagnostics:")
print(f"  Raw target range: [{np.min(y_raw):.4f}, {np.max(y_raw):.4f}]")
print(f"  Raw target mean: {np.mean(y_raw):.4f}, std: {np.std(y_raw):.4f}")
print(f"  Predicted range: [{np.min(y_pred_denorm):.4f}, {np.max(y_pred_denorm):.4f}]")
print(f"  Predicted mean: {np.mean(y_pred_denorm):.4f}, std: {np.std(y_pred_denorm):.4f}")
print(f"  Prediction std ratio (predicted/actual): {np.std(y_pred_denorm)/np.std(y_raw):.3f}")

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

# 5. Perform integrated gradients analysis
print("\n" + "="*60)
print("PERFORMING INTEGRATED GRADIENTS ANALYSIS")
print("="*60)

# Load ESM model for sequence analysis
print("Loading ESM model for integrated gradients...")
esm_model, alphabet = load_esm_model('esm2_t33_650M_UR50D', device)

# Create combined model
combined_model = ESMBasedPredictor(
    esm_model=esm_model,
    predictor_model=final_model,
    esm_layer=ESM_LAYER,
    pooling_method=POOLING_METHOD
).to(device)

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

print(f"\nAnalyzing {len(sequences_to_analyze)} sequences with integrated gradients:")
for name, seq, _ in sequences_to_analyze:
    actual_activity = y_raw[sequences.index(seq)] if seq in sequences else "N/A"
    print(f"  {name}: actual activity = {actual_activity}")

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

    # VERIFICATION: Use the same prediction method as training
    pred_raw_training_method = None
    if sequence in sequences:
        print("  Verifying prediction using training method...")
        # Get the pre-computed embedding (same as used in training)
        seq_idx = sequences.index(sequence)
        seq_embedding = all_reps[seq_idx:seq_idx+1]  # Keep as numpy, add batch dimension

        # Use the same evaluation pipeline as training
        final_model.eval()
        with torch.no_grad():
            seq_tensor = torch.tensor(seq_embedding, dtype=torch.float32, device=device)
            if POOLING_METHOD == 'esm_attention':
                input_mask = torch.ones(1, seq_embedding.shape[1], device=device)
                pred_normalized = final_model(seq_tensor, input_mask).squeeze(-1).cpu().numpy()[0]
            else:
                pred_normalized = final_model(seq_tensor).squeeze(-1).cpu().numpy()[0]
            pred_raw_training_method = pred_normalized * y_log_std + y_log_mean
        print(f"  Training method prediction: {pred_raw_training_method:.4f}")
        print(f"  Training method (normalized): {pred_normalized:.4f}")
    else:
        print("  Sequence not found in training data - skipping training method verification")

    print(f"  Sequence length: {len(sequence)}")
    print(f"  Baseline length: {len(baseline)}")

    result = analyze_sequence_with_integrated_gradients(
        sequence=sequence,
        baseline_sequence=baseline,
        model=combined_model,
        alphabet=alphabet,
        device=device,
        y_log_mean=y_log_mean,
        y_log_std=y_log_std,
        steps=50,
        batch_size=10
    )
    ig_results[name] = result

    if isinstance(actual_activity_raw, (int, float)):
        print(f"  Actual activity: {actual_activity_raw:.4f}")
    else:
        print(f"  Actual activity: {actual_activity_raw}")
    print(f"  IG method prediction: {result['predicted_activity']:.4f}")
    print(f"  Baseline activity: {result['baseline_activity']:.4f}")
    print(f"  Effect vs baseline: {result['effect_vs_baseline']:.4f}")

    # Compare prediction methods
    if sequence in sequences and isinstance(actual_activity_raw, (int, float)) and pred_raw_training_method is not None:
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

# 6. Create visualizations
print("\nCreating visualizations...")

# Create integrated gradients plots
ig_plot_path = os.path.join(results_dir, 'PafA_integrated_gradients_analysis.png')
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
            'effect_vs_baseline': result['effect_vs_baseline']
        })

results_df = pd.DataFrame(all_results)
results_csv_path = os.path.join(results_dir, 'PafA_integrated_gradients_results.csv')
results_df.to_csv(results_csv_path, index=False)

# Create position importance summary
position_importance = results_df.groupby('position').agg({
    'integrated_gradient': ['mean', 'std'],
    'abs_gradient': ['mean', 'max']
}).round(4)
position_importance.columns = ['mean_gradient', 'std_gradient', 'mean_abs_gradient', 'max_abs_gradient']
position_importance['importance_score'] = position_importance['mean_abs_gradient'] + position_importance['std_gradient']
position_importance = position_importance.sort_values('importance_score', ascending=False)

importance_csv_path = os.path.join(results_dir, 'PafA_position_importance_summary.csv')
position_importance.to_csv(importance_csv_path)

# Create summary report
summary_path = os.path.join(results_dir, 'PafA_integrated_gradients_summary.txt')
with open(summary_path, 'w') as f:
    f.write("PAFA INTEGRATED GRADIENTS ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n\n")

    f.write("Model Performance:\n")
    f.write(f"  R² = {r2:.4f}\n")
    f.write(f"  Pearson r = {pearson_r:.4f}\n")
    f.write(f"  Spearman ρ = {spearman_rho:.4f}\n\n")

    f.write("Sequences Analyzed:\n")
    for name, result in ig_results.items():
        f.write(f"  {name}:\n")
        f.write(f"    Predicted Activity: {result['predicted_activity']:.4f}\n")
        f.write(f"    Effect vs Baseline: {result['effect_vs_baseline']:.4f}\n")
        top_positions = np.argsort(np.abs(result['position_gradients']))[-5:][::-1] + 1
        f.write(f"    Top 5 Important Positions: {list(top_positions)}\n\n")

    f.write("TOP 10 MOST IMPORTANT POSITIONS (across all sequences):\n")
    for i, (pos, row) in enumerate(position_importance.head(10).iterrows()):
        aa = wt_seq[pos-1] if pos <= len(wt_seq) else 'N/A'
        f.write(f"{i+1:2d}. Position {pos} ({aa}): importance = {row['importance_score']:.4f}\n")

# 8. Create heatmap of gradients across sequences
print("Creating gradient heatmap...")
heatmap_path = os.path.join(results_dir, 'PafA_integrated_gradients_heatmap.png')
plot_gradients_heatmap(results_df, 'PafA Integrated Gradients Heatmap Across Sequences', heatmap_path)

# 9. Create comparison with wild-type
print("Creating comparison plots...")

# Plot showing difference between variants and WT
comparison_path = os.path.join(results_dir, 'PafA_variants_vs_WT_gradients.png')
plot_variants_vs_wt_gradients(ig_results, comparison_path)

print("\n" + "="*60)
print("PAFA INTEGRATED GRADIENTS ANALYSIS COMPLETE!")
print("="*60)
print(f"Results saved to: {results_dir}")
print("\nGenerated files:")
print("  - PafA_integrated_gradients_analysis.png: Position-wise gradients for each sequence")
print("  - PafA_integrated_gradients_heatmap.png: Heatmap comparing gradients across sequences")
print("  - PafA_variants_vs_WT_gradients.png: Gradient differences compared to wild-type")
print("  - PafA_integrated_gradients_results.csv: Detailed results for all positions")
print("  - PafA_position_importance_summary.csv: Position importance ranking")
print("  - PafA_integrated_gradients_summary.txt: Human-readable summary")
print("\nKey insights:")
print("1. Check 'PafA_position_importance_summary.csv' for most important positions")
print("2. Use 'PafA_integrated_gradients_heatmap.png' to compare gradient patterns")
print("3. Review 'PafA_integrated_gradients_analysis.png' for position-specific effects")
print("4. Compare 'PafA_variants_vs_WT_gradients.png' to see variant-specific changes")
print("5. Integrated gradients show which positions contribute most to kcat_cMUP predictions")

# Save model and analysis parameters
model_path = os.path.join(results_dir, 'PafA_ig_model.pth')
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
}, model_path)
print("  - PafA_ig_model.pth: Model saved with updated analysis parameters")
