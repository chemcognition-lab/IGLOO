# betalactamase_integrated_gradients.py

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Train a model on beta-lactamase variant data using ESM embeddings,
# and perform TRUE integrated gradients analysis using the integrated_gradients_tokens function.
# This is an alternative to the brute-force mutation approach.

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

# Import analysis modules
from src.igloo.data_processing import CustomDataset, batch_pad_and_normalize
from src.igloo.embedding import get_embeddings
from src.igloo.plot import plot_gradients_heatmap, plot_integrated_gradients
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
train_data_path = os.path.join(project_root, 'data', 'betalactamase_train.csv')
test_data_path = os.path.join(project_root, 'data', 'betalactamase_test.csv')
results_dir = os.path.join(project_root, 'results', 'betalactamase_ig')

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# 1. Load datasets
print(f"Loading training data from {train_data_path}")
train_df = pd.read_csv(train_data_path)
print(f"Loading test data from {test_data_path}")
test_df = pd.read_csv(test_data_path)

# Combine both datasets for comprehensive analysis
df = pd.concat([train_df, test_df], ignore_index=True)

col = 'quantitative_function'
# Remove rows with NaN or Inf in target
df = df[df[col].notna()]
df = df[np.isfinite(df[col])]
sequences = df['SEQ'].astype(str).tolist()
y_raw = df[col].values.astype(np.float32)

print(f"Total dataset size: {len(sequences)} sequences")

# Config: select ESM layer and pooling method
ESM_LAYER = 33
POOLING_METHOD = 'esm_attention'
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

# No normalization for beta-lactamase dataset
print("Using raw target values without normalization...")
y = y_raw
y_log_mean = 0.0
y_log_std = 1.0
print(f"Target range: {np.min(y):.4f} to {np.max(y):.4f}")

# Use the specific WT sequence
wt_seq = ("MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAG"
          "QEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELN"
          "EAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDG"
          "KPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW")
print(f"Using specified WT sequence as baseline: {wt_seq[:50]}...")
print(f"WT sequence length: {len(wt_seq)}")

# 2. Get ESM embeddings
print("Computing ESM embeddings...")
all_reps = get_embeddings(sequences, layer=ESM_LAYER, batch_size=16, dataset_name='betalactamase_combined')
N, L, D = all_reps.shape

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Check for existing model and load if available
model_path = os.path.join(results_dir, 'betalactamase_ig_model.pth')
final_model = None

if os.path.exists(model_path):
    print(f"\nFound existing model at {model_path}")
    print("Loading pre-trained model...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
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
pred_values = []
true_values = []
eval_dataset = CustomDataset(all_reps, y)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_pad_and_normalize)

with torch.no_grad():
    for xb, lengths, yb in eval_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if POOLING_METHOD == 'esm_attention':
            input_mask = torch.zeros(xb.size(0), xb.size(1), device=device)
            for i, length in enumerate(lengths):
                input_mask[i, :length] = 1
            out = final_model(xb, input_mask).squeeze(-1).cpu().numpy()
        else:
            out = final_model(xb).squeeze(-1).cpu().numpy()
        pred_values.append(out)
        true_values.append(yb.squeeze(-1).cpu().numpy())

y_pred = np.concatenate(pred_values)
y_true = np.concatenate(true_values)

# Compute metrics
r2 = r2_score(y_true, y_pred)
pearson_r, _ = stats.pearsonr(y_true, y_pred)
spearman_rho, _ = stats.spearmanr(y_true, y_pred)

print("Model Performance:")
print(f"  R² = {r2:.4f}")
print(f"  Pearson r = {pearson_r:.4f}")
print(f"  Spearman ρ = {spearman_rho:.4f}")

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

sequences_to_analyze.extend([
    ('Lowest_Activity', sequences[low_activity_idx], wt_seq),
    ('Highest_Activity', sequences[high_activity_idx], wt_seq),
    ('Median_Activity', sequences[mid_activity_idx], wt_seq)
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
    actual_activity = y_raw[sequences.index(sequence)] if sequence in sequences else "N/A"

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

    print(f"  Actual activity: {actual_activity}")
    print(f"  Predicted activity: {result['predicted_activity']:.4f}")
    print(f"  Baseline activity: {result['baseline_activity']:.4f}")
    print(f"  Effect vs baseline: {result['effect_vs_baseline']:.4f}")
    if isinstance(actual_activity, (int, float)):
        prediction_error = result['predicted_activity'] - actual_activity
        print(f"  Prediction error: {prediction_error:.4f}")
    print(f"  Top 5 gradient positions: {np.argsort(np.abs(result['position_gradients']))[-5:][::-1] + 1}")

# 6. Create visualizations
print("\nCreating visualizations...")

# Create integrated gradients plots
ig_plot_path = os.path.join(results_dir, 'integrated_gradients_analysis.png')
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
results_csv_path = os.path.join(results_dir, 'integrated_gradients_results.csv')
results_df.to_csv(results_csv_path, index=False)

# Create position importance summary
position_importance = results_df.groupby('position').agg({
    'integrated_gradient': ['mean', 'std'],
    'abs_gradient': ['mean', 'max']
}).round(4)
position_importance.columns = ['mean_gradient', 'std_gradient', 'mean_abs_gradient', 'max_abs_gradient']
position_importance['importance_score'] = position_importance['mean_abs_gradient'] + position_importance['std_gradient']
position_importance = position_importance.sort_values('importance_score', ascending=False)

importance_csv_path = os.path.join(results_dir, 'position_importance_summary.csv')
position_importance.to_csv(importance_csv_path)

# Create summary report
summary_path = os.path.join(results_dir, 'integrated_gradients_summary.txt')
with open(summary_path, 'w') as f:
    f.write("INTEGRATED GRADIENTS ANALYSIS SUMMARY\n")
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
pivot_data = results_df.pivot(index='sequence_name', columns='position', values='integrated_gradient')

heatmap_path = os.path.join(results_dir, 'integrated_gradients_heatmap.png')
plot_gradients_heatmap(pivot_data, 'Integrated Gradients Heatmap Across Sequences', heatmap_path)

print("\n" + "="*60)
print("INTEGRATED GRADIENTS ANALYSIS COMPLETE!")
print("="*60)
print(f"Results saved to: {results_dir}")
print("\nGenerated files:")
print("  - integrated_gradients_analysis.png: Position-wise gradients for each sequence")
print("  - integrated_gradients_heatmap.png: Heatmap comparing gradients across sequences")
print("  - integrated_gradients_results.csv: Detailed results for all positions")
print("  - position_importance_summary.csv: Position importance ranking")
print("  - integrated_gradients_summary.txt: Human-readable summary")
print("\nKey insights:")
print("1. Check 'position_importance_summary.csv' for most important positions")
print("2. Use 'integrated_gradients_heatmap.png' to compare gradient patterns")
print("3. Review 'integrated_gradients_analysis.png' for position-specific effects")
print("4. Integrated gradients show which positions contribute most to predictions")
print("5. Unlike brute-force mutations, this shows the gradient-based importance")

# Save model and analysis parameters (update even if using pre-trained model)
model_path = os.path.join(results_dir, 'betalactamase_ig_model.pth')
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
        'target_column': 'quantitative_function',
        'normalization': 'log1p_zscore'
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
print("  - betalactamase_ig_model.pth: Model saved with updated analysis parameters")
