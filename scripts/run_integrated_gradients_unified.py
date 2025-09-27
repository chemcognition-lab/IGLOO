#!/usr/bin/env python3
"""
Unified Integrated Gradients Analysis Runner

A configuration-driven script for running integrated gradients analysis
on protein sequence data using ESM embeddings and various ML models.

Usage:
    python run_integrated_gradients.py config.yaml
    python run_integrated_gradients.py config.json

Example:
    python run_integrated_gradients.py configs/betalactamase_config.yaml
"""

import argparse
import os
import sys
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

# Import analysis modules
from src.igloo.config_utils import (
    load_config,
    validate_config,
)
from src.igloo.data_processing import load_and_prepare_data, prepare_wildtype_sequence
from src.igloo.integrated_gradients import run_integrated_gradients_analysis
from src.igloo.train_and_eval import train_or_load_model
from src.igloo.utils import (
    create_visualizations_and_exports,
    evaluate_model_performance,
    print_completion_summary,
)


class IntegratedGradientsRunner:
    """Unified runner for integrated gradients analysis with config support."""

    def __init__(self, config_path: str):
        """Initialize the runner with a configuration file."""
        self.config = load_config(config_path)
        validate_config(self.config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize attributes that will be set during execution
        self.df = None
        self.sequences = None
        self.y = None
        self.y_raw = None
        self.y_log_mean = None
        self.y_log_std = None
        self.wt_seq = None
        self.final_model = None
        self.model_cfg = None

    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        # Construct data path
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        data_path = os.path.join(project_root, self.config['data']['path'])

        # load_and_prepare_data returns a tuple: (df, sequences, y_raw, y, y_log_mean, y_log_std)
        result = load_and_prepare_data(self.config, data_path)
        self.df, self.sequences, self.y_raw, self.y, self.y_log_mean, self.y_log_std = result

    def prepare_wildtype_sequence(self):
        """Prepare the wildtype sequence for analysis."""
        self.wt_seq, self.wt_idx = prepare_wildtype_sequence(self.df, self.config)

    def get_embeddings_for_training(self):
        """Get embeddings for model training."""
        from src.igloo.embedding import get_embeddings

        model_cfg = self.config['model']

        # Get sample embeddings to determine dimensions
        print("Getting sample embeddings to determine dimensions...")
        sample_sequences = self.sequences[:min(5, len(self.sequences))]
        sample_embeddings = get_embeddings(
            sample_sequences,
            layer=model_cfg.get('esm_layer', 33),
            batch_size=model_cfg.get('batch_size', 16),
            dataset_name=f"{self.config['data']['name']}_sample"
        )

        # Determine embedding dimensions
        if isinstance(sample_embeddings, tuple):
            embedding_dim = sample_embeddings[0].shape[-1]
        else:
            embedding_dim = sample_embeddings.shape[-1]

        print(f"Embedding dimension: {embedding_dim}")
        return embedding_dim

    def train_or_load_model(self, embedding_dim: int):
        """Train a new model or load an existing one."""
        # Construct results directory path
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        results_dir = os.path.join(
            project_root,
            self.config.get('results_dir', f"results/{self.config['data']['name']}_ig")
        )
        os.makedirs(results_dir, exist_ok=True)

        print(f"Results directory: {results_dir}")

        # train_or_load_model returns a tuple: (final_model, train_model, train_metrics)
        self.final_model, train_model, train_metrics = train_or_load_model(
            self.config, self.sequences, self.y, results_dir, self.device
        )
        self.model_cfg = self.config['model']  # Use the model config from the configuration

        # Print model status
        if train_model:
            print("✓ Model training completed successfully")
            if 'train_r2' in train_metrics:
                print(f"  Training R² = {train_metrics['train_r2']:.4f}")
        else:
            print("✓ Pre-trained model loaded successfully")

    def evaluate_model_performance(self):
        """Evaluate the trained model's performance."""
        # Construct results directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        default_results_dir = f"results/{self.config['data']['name']}_ig"
        results_dir = os.path.join(project_root, self.config.get('results_dir', default_results_dir))

        return evaluate_model_performance(
            self.final_model, self.sequences, self.y, self.y_raw,
            self.y_log_mean, self.y_log_std, self.config, results_dir, self.device
        )

    def run_integrated_gradients_analysis(self):
        """Run the integrated gradients analysis."""
        # Construct results directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        default_results_dir = f"results/{self.config['data']['name']}_ig"
        results_dir = os.path.join(project_root, self.config.get('results_dir', default_results_dir))

        # Get sequences to analyze - format as expected: (name, sequence, baseline_sequence)
        # For token-level analysis, pass the actual wildtype sequence as baseline
        sequences_to_analyze = [("Wildtype", self.wt_seq, self.wt_seq)]

        return run_integrated_gradients_analysis(
            sequences_to_analyze, self.config, self.y_raw, self.sequences,
            self.final_model, results_dir, self.device
        )

    def create_visualizations_and_exports(self, results, eval_results):
        """Create visualizations and export results."""
        # Extract required metrics from eval_results tuple
        r2, pearson_r, spearman_rho, y_pred_denorm = eval_results

        # Create basic train metrics dict
        train_metrics = {'note': 'metrics_from_evaluation'}

        # Construct results directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        default_results_dir = f"results/{self.config['data']['name']}_ig"
        results_dir = os.path.join(project_root, self.config.get('results_dir', default_results_dir))

        create_visualizations_and_exports(
            results, r2, pearson_r, spearman_rho, train_metrics,
            self.config, results_dir, self.wt_seq, self.final_model
        )

    def run_complete_analysis(self):
        """Run the complete integrated gradients analysis pipeline."""
        try:
            print("="*60)
            print("INTEGRATED GRADIENTS ANALYSIS PIPELINE")
            print("="*60)
            print(f"Config file: {self.config.get('analysis', {}).get('dataset_name', 'Unknown')}")
            print(f"Device: {self.device}")
            print()

            # Step 1: Load and prepare data
            self.load_and_prepare_data()

            # Step 2: Prepare wildtype sequence
            self.prepare_wildtype_sequence()

            # Step 3: Get embedding dimensions
            embedding_dim = self.get_embeddings_for_training()

            # Step 4: Train or load model
            self.train_or_load_model(embedding_dim)

            # Step 5: Evaluate model performance
            eval_results = self.evaluate_model_performance()

            # Step 6: Run integrated gradients analysis
            ig_results = self.run_integrated_gradients_analysis()

            # Step 7: Create visualizations and exports
            if ig_results:
                self.create_visualizations_and_exports(ig_results, eval_results)

            # Step 8: Print completion summary
            # Convert eval_results tuple to dict for print function
            r2, pearson_r, spearman_rho, _ = eval_results
            eval_results_dict = {
                'r2': r2,
                'pearson_r': pearson_r,
                'spearman_rho': spearman_rho
            }

            print_completion_summary(
                len(self.sequences),
                len(ig_results) if ig_results else 0,
                eval_results_dict,
                self.config.get('results_dir', 'results'),
                self.config['data']['name']
            )

            return {
                'evaluation': eval_results,
                'integrated_gradients': ig_results
            }

        except Exception as e:
            print(f"Error during analysis: {e}")
            traceback.print_exc()
            raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run integrated gradients analysis with configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_integrated_gradients.py configs/betalactamase_config.yaml
    python run_integrated_gradients.py configs/erbc_config.json
    python run_integrated_gradients.py configs/pafa_config.yaml
        """
    )

    parser.add_argument(
        'config_path',
        type=str,
        help='Path to the configuration file (YAML or JSON)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate the configuration file without running analysis'
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config_path):
        print(f"Error: Configuration file '{args.config_path}' not found.")
        sys.exit(1)

    try:
        # Load and validate config
        config = load_config(args.config_path)
        validate_config(config)
        print(f"Configuration file '{args.config_path}' is valid!")

        if args.validate_only:
            print("Validation complete. Exiting.")
            return

        # Run analysis
        runner = IntegratedGradientsRunner(args.config_path)
        results = runner.run_complete_analysis()

        print("\nAnalysis completed successfully!")
        return results

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
