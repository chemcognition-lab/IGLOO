#!/usr/bin/env python3
"""
Unified Integrated Gradients Analysis Script

This script runs integrated gradients analysis based on a configuration file.
It supports multiple datasets, both token-level and embedding-level IG methods,
and both magnitude and signed attribution methods.

Usage:
    python run_integrated_gradients.py --config config.yaml
    python run_integrated_gradients.py --config config.json
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import analysis modules
from src.igloo.config_utils import (
    get_wildtype_sequence,
    load_config,
    select_sequences_for_analysis,
    validate_config,
)
from src.igloo.data_processing import (
    CustomDataset,
    batch_pad_and_normalize,
)
from src.igloo.embedding import get_embeddings
from src.igloo.plot import (
    plot_gradients_heatmap,
    plot_integrated_gradients,
    plot_predicted_vs_actual,
    plot_variants_vs_wt_gradients,
)
from src.igloo.sequence_ig import (
    analyze_sequence_with_integrated_gradients,
    analyze_sequence_with_token_level_integrated_gradients,
    create_combined_model,
    load_esm_model,
)
from src.igloo.train_and_eval import train_model_with_embeddings
from src.igloo.utils import (
    create_model,
    print_analysis_sections,
    print_completion_summary,
    print_data_info,
    print_extreme_value_analysis,
    print_ig_sequence_results,
    print_model_performance,
    print_sequence_analysis_info,
    print_training_info,
)


class IntegratedGradientsRunner:
    """Main class for running integrated gradients analysis from config"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Set up paths
        self.setup_paths()

        # Initialize data variables
        self.df = None
        self.sequences = None
        self.y_raw = None
        self.y = None
        self.y_log_mean = None
        self.y_log_std = None
        self.wt_seq = None
        self.final_model = None

    def setup_paths(self):
        """Set up all necessary paths"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

        # Data path
        self.data_path = os.path.join(project_root, self.config['data']['path'])

        # Results directory
        results_base = self.config.get('results_dir', f"results/{self.config['data']['name']}_ig")
        self.results_dir = os.path.join(project_root, results_base)
        os.makedirs(self.results_dir, exist_ok=True)

        print(f"Data path: {self.data_path}")
        print(f"Results directory: {self.results_dir}")

    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)

        # Get target column
        target_col = self.config['data']['target_column']

        # Remove rows with NaN or Inf in target
        self.df = self.df[self.df[target_col].notna()]
        self.df = self.df[np.isfinite(self.df[target_col])]

        self.sequences = self.df['SEQ'].astype(str).tolist()
        self.y_raw = self.df[target_col].values.astype(np.float32)

        print(f"Dataset size: {len(self.sequences)} sequences")
        print(f"Target range: {np.min(self.y_raw):.4f} to {np.max(self.y_raw):.4f}")

        # Handle normalization
        normalization = self.config['data'].get('normalization', 'z-score')
        if normalization == 'z-score':
            self.y, self.y_log_mean, self.y_log_std = print_data_info(
                self.df, target_col, self.sequences, self.y_raw
            )
        elif normalization == 'none' or normalization is None:
            print("Using raw target values without normalization...")
            self.y = self.y_raw
            self.y_log_mean = 0.0
            self.y_log_std = 1.0
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")

        # Get wild-type sequence
        self.get_wildtype_sequence()

    def get_wildtype_sequence(self):
        """Get the wild-type sequence based on config"""
        wt_config = self.config['data']['wildtype']
        self.wt_seq, wt_idx = get_wildtype_sequence(self.df, wt_config)

        # Print information about the WT sequence
        if wt_config['method'] == 'variant_column':
            variant_col = wt_config['variant_column']
            wt_value = wt_config['wildtype_value']
            wt_activity = self.df.loc[wt_idx, self.config['data']['target_column']]
            print(f"Found WT sequence via {variant_col}='{wt_value}'")
            print(f"WT {self.config['data']['target_column']}: {wt_activity:.4f}")
        elif wt_config['method'] == 'position':
            print(f"Using WT sequence from position: {wt_config['position']}")
        elif wt_config['method'] == 'sequence':
            print("Using provided WT sequence")

        print(f"WT sequence: {self.wt_seq[:50]}...")
        print(f"WT sequence length: {len(self.wt_seq)}")

    def train_or_load_model(self):
        """Train a new model or load existing one"""
        model_config = self.config['model']
        training_model_path = os.path.join(self.results_dir, f"{self.config['data']['name']}_training_model.pth")

        # Get embedding dimensions
        print("Getting embedding dimensions...")
        sample_embeddings = get_embeddings(
            [self.sequences[0]],
            layer=model_config['esm_layer'],
            batch_size=1,
            dataset_name=f"{self.config['data']['name']}_sample"
        )
        L, D = sample_embeddings.shape[1], sample_embeddings.shape[2]

        # Check for existing model
        train_model = True
        if os.path.exists(training_model_path) and not self.config.get('force_retrain', False):
            print(f"\nFound existing trained model at {training_model_path}")
            try:
                checkpoint = torch.load(training_model_path, map_location=self.device, weights_only=False)

                if 'model_state_dict' in checkpoint and 'pooling_method' in checkpoint:
                    # Load model configuration
                    saved_config = checkpoint.get('model_config', {})

                    # Check if config matches
                    config_matches = all(
                        saved_config.get(key) == model_config.get(key, saved_config.get(key))
                        for key in ['input_dim', 'L', 'hidden_dim1', 'hidden_dim2', 'hidden_dim3', 'hidden_dim4', 'dropout_rate', 'pooling_method']
                    )

                    if config_matches:
                        print("Loading pre-trained model...")
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

                        self.final_model = create_model(model_cfg, self.device)
                        self.final_model.load_state_dict(checkpoint['model_state_dict'])
                        self.final_model.eval()
                        train_model = False
                        print("Successfully loaded pre-trained model!")
                    else:
                        print("Model configuration doesn't match. Training new model.")

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
            model = create_model(model_cfg, self.device)

            # Train model
            training_results = train_model_with_embeddings(
                model=model,
                train_sequences=self.sequences,
                train_targets=self.y,
                val_sequences=None,
                val_targets=None,
                esm_layer=model_config['esm_layer'],
                pooling_method=model_config['pooling_method'],
                batch_size=model_config['batch_size'],
                learning_rate=model_config['learning_rate'],
                num_epochs=model_config['epochs'],
                device=self.device,
                dataset_name=f"{self.config['data']['name']}_integrated_gradients",
                loss_type=model_config.get('loss_type', 'huber').lower(),
                results_dir=self.results_dir,
                save_path=training_model_path
            )

            self.final_model = training_results['model']
            train_metrics = training_results['metrics']
        else:
            train_metrics = {'note': 'using_pretrained_model'}

        print_training_info(train_metrics, train_model)
        return train_model, train_metrics

    def evaluate_model_performance(self):
        """Evaluate the trained model"""
        print("\nEvaluating model performance...")
        self.final_model.eval()

        model_config = self.config['model']

        # Get embeddings for evaluation
        print("Getting embeddings for evaluation...")
        all_embeddings = get_embeddings(
            self.sequences,
            layer=model_config['esm_layer'],
            batch_size=16,
            dataset_name=f"{self.config['data']['name']}_eval"
        )

        # Evaluate using DataLoader
        eval_dataset = CustomDataset(all_embeddings, self.y)
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
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                if model_config['pooling_method'] == 'esm_attention':
                    input_mask = torch.zeros(xb.size(0), xb.size(1), device=self.device)
                    for i, length in enumerate(lengths):
                        input_mask[i, :length] = 1
                    out = self.final_model(xb, input_mask).squeeze(-1)
                else:
                    out = self.final_model(xb).squeeze(-1)

                if out.ndim > 1:
                    out = out.squeeze(-1)

                pred_norm.append(out.cpu().numpy())
                y_true_norm.append(yb.cpu().numpy().flatten())

        y_pred = np.concatenate(pred_norm)
        y_true = np.concatenate(y_true_norm)

        # Get performance metrics
        r2, pearson_r, spearman_rho, y_pred_denorm = print_model_performance(
            y_true, y_pred, self.y_raw, self.y_log_mean, self.y_log_std
        )

        print_extreme_value_analysis(self.y_raw, y_pred_denorm)

        # Create predicted vs actual plot
        print("Creating predicted vs actual plot...")
        pred_vs_actual_path = os.path.join(
            self.results_dir,
            f"{self.config['data']['name']}_predicted_vs_actual.png"
        )
        plot_predicted_vs_actual(
            self.y_raw, y_pred_denorm, pred_vs_actual_path,
            title=f"{self.config['data']['name']} Model: Predicted vs Actual {self.config['data']['target_column']} Values",
            r2=r2, pearson_r=pearson_r, spearman_rho=spearman_rho
        )

        return r2, pearson_r, spearman_rho, y_pred_denorm

    def select_sequences_to_analyze(self) -> list[tuple[str, str, str]]:
        """Select interesting sequences for analysis"""
        selection_config = self.config['analysis'].get('sequence_selection', {})
        return select_sequences_for_analysis(self.df, self.y_raw, self.wt_seq, selection_config)

    def run_integrated_gradients_analysis(self, sequences_to_analyze):
        """Run the integrated gradients analysis"""
        print_analysis_sections("PERFORMING INTEGRATED GRADIENTS ANALYSIS")

        ig_config = self.config['integrated_gradients']
        model_config = self.config['model']

        # Load ESM model
        print("Loading ESM model for integrated gradients...")
        esm_model, alphabet = load_esm_model('esm2_t33_650M_UR50D', self.device)

        print_sequence_analysis_info(sequences_to_analyze, self.y_raw, self.sequences)

        # Run analysis based on method
        ig_method = ig_config['method']  # 'embedding' or 'token'
        attribution_method = ig_config.get('attribution_method', 'magnitude')

        ig_results = {}

        if ig_method == 'embedding':
            # Create combined model for embedding-level IG
            print("Creating combined model for embedding-level integrated gradients...")
            combined_model = create_combined_model(
                esm_model=esm_model,
                predictor_model=self.final_model,
                esm_layer=model_config['esm_layer'],
                pooling_method=model_config['pooling_method'],
                method='embedding',
                device=self.device
            )

            # Run embedding-level analysis
            for name, sequence, baseline in sequences_to_analyze:
                print(f"\nAnalyzing {name} (embedding-level)...")

                # Get actual activity
                actual_activity = self.y_raw[self.sequences.index(sequence)] if sequence in self.sequences else None
                baseline_actual_activity = self.y_raw[self.sequences.index(baseline)] if baseline in self.sequences else None

                # Get data range for clipping
                data_min = float(np.min(self.y_raw))
                data_max = float(np.max(self.y_raw))

                result = analyze_sequence_with_integrated_gradients(
                    sequence=sequence,
                    baseline_sequence=baseline,
                    model=combined_model,
                    alphabet=alphabet,
                    device=self.device,
                    y_log_mean=self.y_log_mean,
                    y_log_std=self.y_log_std,
                    steps=ig_config.get('steps', 50),
                    batch_size=ig_config.get('batch_size', 10),
                    method='embedding',
                    attribution_method=attribution_method,
                    actual_activity=actual_activity,
                    baseline_actual_activity=baseline_actual_activity,
                    plot_predictions=True,
                    sequence_name=name,
                    baseline_name="WT",
                    save_plot_path=os.path.join(self.results_dir, f"{self.config['data']['name']}_{name}_predicted_vs_actual.png"),
                    clip_min=data_min,
                    clip_max=data_max
                )
                ig_results[name] = result
                print_ig_sequence_results(name, actual_activity, result, None)

        elif ig_method == 'token':
            # Run token-level analysis
            for name, sequence, baseline in sequences_to_analyze:
                print(f"\nAnalyzing {name} (token-level)...")

                # Get actual activity
                actual_activity = self.y_raw[self.sequences.index(sequence)] if sequence in self.sequences else None
                baseline_actual_activity = self.y_raw[self.sequences.index(baseline)] if baseline in self.sequences else None

                # Get data range for clipping
                data_min = float(np.min(self.y_raw))
                data_max = float(np.max(self.y_raw))

                result = analyze_sequence_with_token_level_integrated_gradients(
                    sequence=sequence,
                    baseline_sequence=baseline,
                    esm_model=esm_model,
                    predictor_model=self.final_model,
                    alphabet=alphabet,
                    device=self.device,
                    esm_layer=model_config['esm_layer'],
                    pooling_method=model_config['pooling_method'],
                    y_log_mean=self.y_log_mean,
                    y_log_std=self.y_log_std,
                    steps=ig_config.get('steps', 50),
                    batch_size=ig_config.get('batch_size', 10),
                    attribution_method=attribution_method,
                    actual_activity=actual_activity,
                    baseline_actual_activity=baseline_actual_activity,
                    plot_predictions=True,
                    sequence_name=name,
                    baseline_name="WT",
                    save_plot_path=os.path.join(self.results_dir, f"{self.config['data']['name']}_{name}_predicted_vs_actual.png"),
                    clip_min=data_min,
                    clip_max=data_max
                )
                ig_results[name] = result
                print_ig_sequence_results(name, actual_activity, result, None)
        else:
            raise ValueError(f"Unknown integrated gradients method: {ig_method}")

        return ig_results

    def create_visualizations_and_exports(self, ig_results, r2, pearson_r, spearman_rho, train_metrics):
        """Create visualizations and export results"""
        print("\nCreating visualizations...")

        dataset_name = self.config['data']['name']
        ig_config = self.config['integrated_gradients']

        # Create integrated gradients plots
        ig_plot_path = os.path.join(self.results_dir, f'{dataset_name}_integrated_gradients_analysis.png')
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
        results_csv_path = os.path.join(self.results_dir, f'{dataset_name}_integrated_gradients_results.csv')
        results_df.to_csv(results_csv_path, index=False)

        # Create position importance summary
        position_importance = results_df.groupby('position').agg({
            'integrated_gradient': ['mean', 'std'],
            'abs_gradient': ['mean', 'max']
        }).round(4)
        position_importance.columns = ['mean_gradient', 'std_gradient', 'mean_abs_gradient', 'max_abs_gradient']
        position_importance['importance_score'] = position_importance['mean_abs_gradient'] + position_importance['std_gradient']
        position_importance = position_importance.sort_values('importance_score', ascending=False)

        importance_csv_path = os.path.join(self.results_dir, f'{dataset_name}_position_importance_summary.csv')
        position_importance.to_csv(importance_csv_path)

        # Create predicted vs actual summary
        pred_vs_actual_summary_path = os.path.join(self.results_dir, f'{dataset_name}_predicted_vs_actual_summary.csv')
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
        summary_path = os.path.join(self.results_dir, f'{dataset_name}_integrated_gradients_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"{dataset_name.upper()} INTEGRATED GRADIENTS ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write("Analysis Configuration:\n")
            f.write(f"  Integration Method: {ig_config['method']}\n")
            f.write(f"  Attribution Method: {ig_config.get('attribution_method', 'magnitude')}\n")
            f.write(f"  ESM Layer: {self.config['model']['esm_layer']}\n")
            f.write(f"  Pooling Method: {self.config['model']['pooling_method']}\n")
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
                aa = self.wt_seq[pos-1] if pos <= len(self.wt_seq) else 'N/A'
                f.write(f"{i+1:2d}. Position {pos} ({aa}): importance = {row['importance_score']:.4f}\n")

        # Create heatmap
        print("Creating gradient heatmap...")
        heatmap_path = os.path.join(self.results_dir, f'{dataset_name}_integrated_gradients_heatmap.png')
        plot_gradients_heatmap(results_df, f'{dataset_name} Integrated Gradients Heatmap Across Sequences', heatmap_path)

        # Create comparison plots
        print("Creating comparison plots...")
        comparison_path = os.path.join(self.results_dir, f'{dataset_name}_variants_vs_WT_gradients.png')
        plot_variants_vs_wt_gradients(ig_results, comparison_path)

        # Save model and analysis parameters
        analysis_model_path = os.path.join(self.results_dir, f'{dataset_name}_ig_analysis.pth')
        torch.save({
            'model_state_dict': self.final_model.state_dict(),
            'model_config': {
                'input_dim': self.config['model'].get('input_dim'),
                'L': self.config['model'].get('L'),
                'hidden_dim1': self.config['model']['hidden_dim1'],
                'hidden_dim2': self.config['model']['hidden_dim2'],
                'hidden_dim3': self.config['model']['hidden_dim3'],
                'hidden_dim4': self.config['model']['hidden_dim4'],
                'dropout_rate': self.config['model']['dropout_rate'],
                'final_activation': self.config['model'].get('final_activation'),
                'pooling_method': self.config['model']['pooling_method']
            },
            'analysis_config': {
                'esm_layer': self.config['model']['esm_layer'],
                'ig_method': ig_config['method'],
                'attribution_method': ig_config.get('attribution_method', 'magnitude'),
                'ig_steps': ig_config.get('steps', 50),
                'ig_batch_size': ig_config.get('batch_size', 10),
                'target_column': self.config['data']['target_column'],
                'normalization': self.config['data'].get('normalization', 'z-score')
            },
            'performance': {
                'r2': r2,
                'pearson_r': pearson_r,
                'spearman_rho': spearman_rho
            },
            'data_info': {
                'target_mean': self.y_log_mean,
                'target_std': self.y_log_std,
                'wt_sequence': self.wt_seq
            },
            'config': self.config
        }, analysis_model_path)

        print_completion_summary(self.results_dir)

    def run(self):
        """Run the complete analysis pipeline"""
        print("="*60)
        print("INTEGRATED GRADIENTS ANALYSIS PIPELINE")
        print("="*60)
        print(f"Dataset: {self.config['data']['name']}")
        print(f"Method: {self.config['integrated_gradients']['method']}")
        print(f"Attribution: {self.config['integrated_gradients'].get('attribution_method', 'magnitude')}")
        print("="*60)

        # Load and prepare data
        self.load_and_prepare_data()

        # Train or load model
        train_model, train_metrics = self.train_or_load_model()

        # Evaluate model performance
        r2, pearson_r, spearman_rho, y_pred_denorm = self.evaluate_model_performance()

        # Select sequences to analyze
        sequences_to_analyze = self.select_sequences_to_analyze()

        # Run integrated gradients analysis
        ig_results = self.run_integrated_gradients_analysis(sequences_to_analyze)

        # Create visualizations and export results
        self.create_visualizations_and_exports(ig_results, r2, pearson_r, spearman_rho, train_metrics)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)


# Functions are now imported from src.igloo.config_utils


def main():
    parser = argparse.ArgumentParser(description='Run integrated gradients analysis from config file')
    parser.add_argument('--config', required=True, help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--validate-only', action='store_true', help='Only validate config without running analysis')

    args = parser.parse_args()

    try:
        # Load and validate configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        config = validate_config(config)

        if args.validate_only:
            print("✓ Configuration is valid!")
            return

        # Run analysis
        runner = IntegratedGradientsRunner(config)
        runner.run()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
