# betalactamase_integrated_gradients_train_test.py

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Train a model on beta-lactamase training data using ESM embeddings,
# and perform integrated gradients analysis on the TEST SET to analyze
# performance on unseen data.
#
# This script uses the new train_model_with_embeddings function and consistent
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
train_data_path = os.path.join(project_root, 'data', 'betalactamase_train.csv')
test_data_path = os.path.join(project_root, 'data', 'betalactamase_test.csv')
results_dir = os.path.join(project_root, 'results', 'betalactamase_ig_train_test')

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# 1. Load datasets
print(f"Loading training data from {train_data_path}")
train_df = pd.read_csv(train_data_path)
print(f"Loading test data from {test_data_path}")
test_df = pd.read_csv(test_data_path)

col = 'quantitative_function'

# Process training data
train_df = train_df[train_df[col].notna()]
train_df = train_df[np.isfinite(train_df[col])]
train_sequences = train_df['SEQ'].astype(str).tolist()
y_train_raw = train_df[col].values.astype(np.float32)

# Process test data
test_df = test_df[test_df[col].notna()]
test_df = test_df[np.isfinite(test_df[col])]
test_sequences = test_df['SEQ'].astype(str).tolist()
y_test_raw = test_df[col].values.astype(np.float32)

print(f"Training dataset size: {len(train_sequences)} sequences")
print(f"Test dataset size: {len(test_sequences)} sequences")

# Data range for prediction clipping (use combined range)
all_y_raw = np.concatenate([y_train_raw, y_test_raw])
DATA_MIN = float(np.min(all_y_raw))
DATA_MAX = float(np.max(all_y_raw))
print(f"Combined data range for clipping: [{DATA_MIN:.4f}, {DATA_MAX:.4f}]")

# Config: select ESM layer and pooling method
ESM_LAYER = 33
POOLING_METHOD = 'esm_attention'
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

# Use raw target values without normalization for beta-lactamase dataset
print("Using raw target values without normalization...")
y_train = y_train_raw
y_test = y_test_raw
y_log_mean = 0.0
y_log_std = 1.0
print(f"Training target range: {np.min(y_train):.4f} to {np.max(y_train):.4f}")
print(f"Test target range: {np.min(y_test):.4f} to {np.max(y_test):.4f}")

# Use the specific WT sequence for beta-lactamase
wt_seq = ("MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAG"
          "QEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELN"
          "EAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDG"
          "KPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW")
print(f"Using beta-lactamase WT sequence as baseline: {wt_seq[:50]}...")
print(f"WT sequence length: {len(wt_seq)}")

# 2. Get embedding dimensions for model configuration
print("Getting embedding dimensions...")
# Get a sample embedding to determine dimensions
sample_embeddings = get_embeddings(
    [train_sequences[0]], layer=ESM_LAYER, batch_size=1,
    dataset_name='betalactamase_sample'
)
N_train, L, D = len(train_sequences), sample_embeddings.shape[1], sample_embeddings.shape[2]

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Check for existing trained model and load if available
training_model_path = os.path.join(results_dir, 'betalactamase_training_model.pth')
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

# Train model if needed
if train_model:
    print("\nTraining model on TRAINING SET for integrated gradients analysis on TEST SET...")

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

    # Use the new training function - train ONLY on training set
    training_results = train_model_with_embeddings(
        model=model,
        train_sequences=train_sequences,
        train_targets=y_train,
        val_sequences=None,  # No validation split, train on all training data
        val_targets=None,
        esm_layer=ESM_LAYER,
        pooling_method=POOLING_METHOD,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=epochs,
        device=device,
        dataset_name='betalactamase_train_only',
        loss_type=LOSS_TYPE.lower(),
        results_dir=results_dir,
        save_path=training_model_path  # Save the best model during training
    )

    final_model = training_results['model']
    train_metrics = training_results['metrics']

    # Ensure the final model is saved (in case train_model_with_embeddings didn't save it)
    print(f"Saving final trained model to {training_model_path}")
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
        'esm_layer': ESM_LAYER,
        'pooling_method': POOLING_METHOD,
        'train_loss': train_metrics.get('train_loss', 0.0),
        'val_loss': train_metrics.get('val_loss', None),
        'epoch': train_metrics.get('epoch', epochs-1),
        'training_complete': True
    }, training_model_path)
    print("Final trained model saved successfully.")
else:
    # Create dummy train_metrics for consistency when using pre-trained model
    train_metrics = {'note': 'using_pretrained_model'}

print_training_info(train_metrics, train_model)

# 4. Evaluate model performance on TEST SET
print("\nEvaluating model performance on TEST SET (unseen data)...")
final_model.eval()

# Get embeddings for test set evaluation
print("Getting embeddings for test set evaluation...")
test_embeddings = get_embeddings(test_sequences, layer=ESM_LAYER, batch_size=16, dataset_name='betalactamase_test_eval')

# Use DataLoader approach (same as training) for consistency
test_dataset = CustomDataset(test_embeddings, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_pad_and_normalize)

pred_norm = []
y_true_norm = []
with torch.no_grad():
    for xb, lengths, yb in test_loader:
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

y_pred_test = np.concatenate(pred_norm)
y_true_test = np.concatenate(y_true_norm)

# Print comprehensive model performance metrics on test set
print("\n" + "="*60)
print("TEST SET PERFORMANCE (Model has never seen this data)")
print("="*60)
r2, pearson_r, spearman_rho, y_pred_test_denorm = print_model_performance(
    y_true_test, y_pred_test, y_test_raw, y_log_mean, y_log_std
)

# Print extreme value analysis for test set
print_extreme_value_analysis(y_test_raw, y_pred_test_denorm)

# Create predicted vs actual plot for test set
print("Creating predicted vs actual plot for test set...")
pred_vs_actual_path = os.path.join(results_dir, 'betalactamase_test_predicted_vs_actual.png')
plot_predicted_vs_actual(
    y_test_raw, y_pred_test_denorm, pred_vs_actual_path,
    title="Beta-lactamase Model: Test Set Predicted vs Actual Values",
    r2=r2, pearson_r=pearson_r, spearman_rho=spearman_rho
)

# 5. Perform integrated gradients analysis ON TEST SET
print_analysis_sections("PERFORMING INTEGRATED GRADIENTS ANALYSIS ON TEST SET")

# Load ESM model for sequence analysis
print("Loading ESM model for integrated gradients...")
esm_model, alphabet = load_esm_model('esm2_t33_650M_UR50D', device)

# Create combined model using the specified method
print(f"Creating combined model using {IG_METHOD} method...")
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

# Select interesting sequences from TEST SET for analysis
sequences_to_analyze = []

# 1. Wild-type sequence (use as baseline)
sequences_to_analyze.append(('WT', wt_seq, wt_seq))

# 2. Find high and low activity sequences from TEST SET
test_sorted_indices = np.argsort(y_test_raw)
test_low_activity_idx = test_sorted_indices[0]  # Lowest activity in test set
test_high_activity_idx = test_sorted_indices[-1]  # Highest activity in test set
test_mid_activity_idx = test_sorted_indices[len(test_sorted_indices)//2]  # Median activity in test set

# Get a few more interesting variants from test set
test_high_10_idx = test_sorted_indices[-10] if len(test_sorted_indices) >= 10 else test_sorted_indices[-1]
test_low_10_idx = test_sorted_indices[9] if len(test_sorted_indices) >= 10 else test_sorted_indices[0]

sequences_to_analyze.extend([
    ('Test_Lowest_Activity', test_sequences[test_low_activity_idx], wt_seq),
    ('Test_Highest_Activity', test_sequences[test_high_activity_idx], wt_seq),
    ('Test_Median_Activity', test_sequences[test_mid_activity_idx], wt_seq),
    ('Test_High_10th', test_sequences[test_high_10_idx], wt_seq),
    ('Test_Low_10th', test_sequences[test_low_10_idx], wt_seq)
])

print_sequence_analysis_info(sequences_to_analyze, y_test_raw, test_sequences)

# Perform integrated gradients analysis on test sequences
ig_results = {}
for name, sequence, baseline in sequences_to_analyze:
    print(f"\nAnalyzing {name} from test set...")

    # Get actual activity for comparison (from test set)
    if sequence in test_sequences:
        seq_idx = test_sequences.index(sequence)
        actual_activity_raw = y_test_raw[seq_idx]
        actual_activity_normalized = y_test[seq_idx]
    else:
        # WT sequence might not be in test set
        actual_activity_raw = "N/A"
        actual_activity_normalized = "N/A"

    # Get actual activity for plotting (if available)
    actual_activity_plot = actual_activity_raw if isinstance(actual_activity_raw, (int, float)) else None

    # Baseline activity (WT) - might not be in test set, so check training set too
    baseline_actual_activity = None
    if baseline in test_sequences:
        baseline_idx = test_sequences.index(baseline)
        baseline_actual_activity = y_test_raw[baseline_idx]
    elif baseline in train_sequences:
        baseline_idx = train_sequences.index(baseline)
        baseline_actual_activity = y_train_raw[baseline_idx]

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
        save_plot_path=os.path.join(results_dir, f'betalactamase_{name}_predicted_vs_actual.png'),
        clip_min=DATA_MIN,
        clip_max=DATA_MAX
    )
    ig_results[name] = result

    print_ig_sequence_results(name, actual_activity_raw, result, None)

# 6. Create visualizations
print("\nCreating visualizations...")

# Create integrated gradients plots
ig_plot_path = os.path.join(results_dir, 'betalactamase_integrated_gradients_analysis.png')
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
results_csv_path = os.path.join(results_dir, 'betalactamase_integrated_gradients_results.csv')
results_df.to_csv(results_csv_path, index=False)

# Create position importance summary
position_importance = results_df.groupby('position').agg({
    'integrated_gradient': ['mean', 'std'],
    'abs_gradient': ['mean', 'max']
}).round(4)
position_importance.columns = ['mean_gradient', 'std_gradient', 'mean_abs_gradient', 'max_abs_gradient']
position_importance['importance_score'] = position_importance['mean_abs_gradient'] + position_importance['std_gradient']
position_importance = position_importance.sort_values('importance_score', ascending=False)

importance_csv_path = os.path.join(results_dir, 'betalactamase_position_importance_summary.csv')
position_importance.to_csv(importance_csv_path)

# Create predicted vs actual summary for test set
pred_vs_actual_summary_path = os.path.join(results_dir, 'betalactamase_test_predicted_vs_actual_summary.csv')
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
    print(f"Test set predicted vs actual summary saved to: {pred_vs_actual_summary_path}")

# Create summary report
summary_path = os.path.join(results_dir, 'betalactamase_integrated_gradients_summary.txt')
with open(summary_path, 'w') as f:
    f.write("BETA-LACTAMASE INTEGRATED GRADIENTS ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n\n")

    f.write("Analysis Configuration:\n")
    f.write(f"  Training Set Size: {len(train_sequences)} sequences\n")
    f.write(f"  Test Set Size: {len(test_sequences)} sequences\n")
    f.write(f"  Integration Method: {IG_METHOD}\n")
    if IG_METHOD == 'gumbel':
        f.write(f"  Gumbel Temperature: {GUMBEL_TEMPERATURE}\n")
        f.write(f"  Hard Gumbel: {HARD_GUMBEL}\n")
    f.write(f"  ESM Layer: {ESM_LAYER}\n")
    f.write(f"  Pooling Method: {POOLING_METHOD}\n\n")

    f.write("Model Performance on Test Set (Unseen Data):\n")
    f.write(f"  R² = {r2:.4f}\n")
    f.write(f"  Pearson r = {pearson_r:.4f}\n")
    f.write(f"  Spearman ρ = {spearman_rho:.4f}\n\n")

    f.write("Test Set Sequences Analyzed:\n")
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

    f.write("TOP 10 MOST IMPORTANT POSITIONS (across test sequences):\n")
    for i, (pos, row) in enumerate(position_importance.head(10).iterrows()):
        aa = wt_seq[pos-1] if pos <= len(wt_seq) else 'N/A'
        f.write(f"{i+1:2d}. Position {pos} ({aa}): importance = {row['importance_score']:.4f}\n")

# 8. Create heatmap of gradients across test sequences
print("Creating gradient heatmap for test sequences...")
heatmap_path = os.path.join(results_dir, 'betalactamase_test_integrated_gradients_heatmap.png')
plot_gradients_heatmap(results_df, 'Beta-lactamase Test Set Integrated Gradients Heatmap', heatmap_path)

# 9. Create comparison with wild-type
print("Creating comparison plots...")

# Plot showing difference between test variants and WT
comparison_path = os.path.join(results_dir, 'betalactamase_test_variants_vs_WT_gradients.png')
plot_variants_vs_wt_gradients(ig_results, comparison_path)

print_completion_summary(results_dir)

# Save model and analysis parameters (analysis results file)
analysis_model_path = os.path.join(results_dir, 'betalactamase_ig_analysis.pth')
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
        'normalization': 'none',
        'train_set_size': len(train_sequences),
        'test_set_size': len(test_sequences)
    },
    'test_performance': {
        'r2': r2,
        'pearson_r': pearson_r,
        'spearman_rho': spearman_rho
    },
    'data_info': {
        'target_mean': y_log_mean,
        'target_std': y_log_std,
        'wt_sequence': wt_seq,
        'data_min': DATA_MIN,
        'data_max': DATA_MAX
    },
    'training_info': {
        'was_retrained': train_model,
        'train_metrics': train_metrics if train_model else 'using_pretrained'
    }
}, analysis_model_path)
print("  - betalactamase_ig_analysis.pth: Analysis results and model saved")
if train_model:
    print("  - betalactamase_training_model.pth: Best trained model saved automatically during training")

print("\n" + "="*80)
print("SUMMARY: TRAIN-TEST INTEGRATED GRADIENTS ANALYSIS COMPLETE")
print("="*80)
print(f"✓ Model trained on {len(train_sequences)} training sequences")
print(f"✓ Integrated gradients analysis performed on {len(test_sequences)} test sequences")
print(f"✓ Test set performance: R² = {r2:.4f}, Pearson r = {pearson_r:.4f}")
print(f"✓ Results saved to: {results_dir}")
print("="*80)
