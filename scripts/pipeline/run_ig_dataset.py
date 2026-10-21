#!/usr/bin/env python3
"""
Integrated Gradients Analysis Runner — DatasetLoader edition.

Identical pipeline to run_integrated_gradients.py but loads and prepares
data through the DatasetLoader / Dataset classes in src/igloo/dataset.py
instead of calling load_and_prepare_data() / prepare_wildtype_sequence()
directly.

Usage:
    python scripts/pipeline/run_ig_dataset.py configs/pafa_config_embedding-signed.yaml
    python scripts/pipeline/run_ig_dataset.py configs/betalactamase_config_embedding-signed.yaml
    python scripts/pipeline/run_ig_dataset.py configs/erbc_config_embedding-signed.yaml

    # List all datasets the registry can discover, then exit
    python scripts/pipeline/run_ig_dataset.py --list-datasets

    # Validate config only
    python scripts/pipeline/run_ig_dataset.py configs/pafa_config.yaml --validate-only
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

# Add project root to path so imports work regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.igloo.config_utils import (
    load_config,
    select_sequences_for_analysis,
    validate_config,
)
from src.igloo.dataset import DatasetLoader
from src.igloo.embedding import get_embeddings
from src.igloo.integrated_gradients import run_integrated_gradients_analysis
from src.igloo.train_and_eval import train_or_load_model
from src.igloo.utils import (
    create_visualizations_and_exports,
    evaluate_model_performance,
    print_completion_summary,
)


def _resolve_results_dir(config: dict) -> str:
    """Return the absolute results directory for a given config."""
    dataset_name = config["data"]["name"]
    rel = config.get("results_dir", f"results/{dataset_name}_ig")
    return str(PROJECT_ROOT / rel)


class IGDatasetRunner:
    """
    Integrated gradients runner that uses DatasetLoader for data preparation.

    The only difference from IntegratedGradientsRunner is steps 1 and 2:
      - Step 1 uses DatasetLoader.load_prepared()  → Dataset.to_pipeline_tuple()
      - Step 2 uses Dataset.get_wildtype()
    All downstream steps (embedding, training, IG, plotting) are identical.
    """

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        validate_config(self.config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Will be populated during execution
        self.dataset = None
        self.df = None
        self.sequences = None
        self.y = None
        self.y_raw = None
        self.y_log_mean = None
        self.y_log_std = None
        self.wt_seq = None
        self.wt_idx = None
        self.final_model = None

    # ------------------------------------------------------------------
    # Step 1 — data loading via DatasetLoader
    # ------------------------------------------------------------------
    def load_and_prepare_data(self):
        """Load and prepare the dataset via DatasetLoader."""
        data_root = str(PROJECT_ROOT / "data")
        loader = DatasetLoader(data_root)

        dataset_name = self.config["data"]["name"]
        print(f"Loading dataset '{dataset_name}' via DatasetLoader …")
        self.dataset = loader.load_prepared(dataset_name, self.config)

        if self.dataset is None:
            # Fallback: try the CSV stem from the path field
            csv_stem = Path(self.config["data"]["path"]).stem
            print(
                f"  '{dataset_name}' not found in registry, trying stem '{csv_stem}' …"
            )
            self.dataset = loader.load_prepared(csv_stem, self.config)

        if self.dataset is None:
            raise RuntimeError(
                f"Could not load dataset '{dataset_name}'. "
                "Check that 'data.name' in the config matches the CSV filename "
                "(without extension) inside the data/ folder."
            )

        # Unpack into the same attributes the rest of the pipeline expects
        (
            self.df,
            self.sequences,
            self.y_raw,
            self.y,
            self.y_log_mean,
            self.y_log_std,
        ) = self.dataset.to_pipeline_tuple()

        print(f"  Loaded {len(self.sequences)} sequences.")

    # ------------------------------------------------------------------
    # Step 2 — wildtype via Dataset
    # ------------------------------------------------------------------
    def prepare_wildtype_sequence(self):
        """Retrieve the wildtype sequence from the prepared Dataset."""
        self.wt_seq, self.wt_idx = self.dataset.get_wildtype()
        if self.wt_seq is None:
            raise RuntimeError(
                "Wildtype sequence could not be identified. "
                "Check the 'data.wildtype' section of your config."
            )
        print(f"Wildtype sequence length: {len(self.wt_seq)}")
        print(f"Wildtype sequence: {self.wt_seq[:50]}…")

    # ------------------------------------------------------------------
    # Steps 3-8 — identical to run_integrated_gradients.py
    # ------------------------------------------------------------------
    def get_embeddings_for_training(self) -> int:
        """Get a sample embedding to determine the embedding dimension."""
        model_cfg = self.config["model"]
        print("Getting sample embeddings to determine dimensions…")
        sample = self.sequences[: min(5, len(self.sequences))]
        sample_embeddings = get_embeddings(
            sample,
            layer=model_cfg.get("esm_layer", 33),
            batch_size=model_cfg.get("batch_size", 16),
            dataset_name=f"{self.config['data']['name']}_sample",
        )
        if isinstance(sample_embeddings, tuple):
            embedding_dim = sample_embeddings[0].shape[-1]
        else:
            embedding_dim = sample_embeddings.shape[-1]
        print(f"Embedding dimension: {embedding_dim}")
        return embedding_dim

    def train_or_load_model(self, embedding_dim: int):
        """Train a new model or load a pre-trained one."""
        results_dir = _resolve_results_dir(self.config)
        os.makedirs(results_dir, exist_ok=True)
        print(f"Results directory: {results_dir}")

        self.final_model, trained, train_metrics = train_or_load_model(
            self.config, self.sequences, self.y, results_dir, self.device
        )

        if trained:
            print("✓ Model training completed successfully")
            if "train_r2" in train_metrics:
                print(f"  Training R² = {train_metrics['train_r2']:.4f}")
        else:
            print("✓ Pre-trained model loaded successfully")

    def evaluate_model_performance(self):
        """Evaluate the trained model's performance."""
        results_dir = _resolve_results_dir(self.config)
        return evaluate_model_performance(
            self.final_model,
            self.sequences,
            self.y,
            self.y_raw,
            self.y_log_mean,
            self.y_log_std,
            self.config,
            results_dir,
            self.device,
        )

    def run_integrated_gradients_analysis(self, eval_results=None):
        """Select sequences and run integrated gradients."""
        results_dir = _resolve_results_dir(self.config)

        selection_config = self.config.get(
            "sequence_selection",
            {
                "include_extremes": True,
                "include_median": True,
                "include_top_n": 0,
                "include_bottom_n": 0,
                "custom_sequences": [],
                "max_error_percent": 10.0,
            },
        )

        y_pred = eval_results[3] if eval_results is not None else None

        sequences_to_analyze = select_sequences_for_analysis(
            self.df,
            self.y_raw,
            self.wt_seq,
            selection_config,
            model=self.final_model,
            sequences=self.sequences,
            y_pred=y_pred,
        )

        print(f"\nSelected {len(sequences_to_analyze)} sequences for integrated gradients analysis:")
        for name, seq, _baseline in sequences_to_analyze:
            try:
                activity = self.y_raw[self.sequences.index(seq)]
                print(f"  {name}: activity = {activity:.4f}")
            except ValueError:
                print(f"  {name}: custom sequence (not in training data)")

        return run_integrated_gradients_analysis(
            sequences_to_analyze,
            self.config,
            self.y_raw,
            self.sequences,
            self.final_model,
            results_dir,
            self.device,
            self.y_log_mean,
            self.y_log_std,
        )

    def create_visualizations_and_exports(self, ig_results, eval_results):
        """Create plots and export CSV results."""
        r2, pearson_r, spearman_rho, _ = eval_results
        results_dir = _resolve_results_dir(self.config)
        create_visualizations_and_exports(
            ig_results,
            r2,
            pearson_r,
            spearman_rho,
            {"note": "metrics_from_evaluation"},
            self.config,
            results_dir,
            self.wt_seq,
            self.final_model,
        )

    def run_complete_analysis(self):
        """Run the full pipeline end-to-end."""
        print("=" * 60)
        print("INTEGRATED GRADIENTS ANALYSIS PIPELINE  (DatasetLoader)")
        print("=" * 60)
        print(f"Dataset : {self.config['data']['name']}")
        print(f"Device  : {self.device}")
        print()

        # 1. Load & prepare data via DatasetLoader
        self.load_and_prepare_data()

        # 2. Wildtype from Dataset
        self.prepare_wildtype_sequence()

        # 3. Embedding dimension
        embedding_dim = self.get_embeddings_for_training()

        # 4. Train / load model
        self.train_or_load_model(embedding_dim)

        # 5. Evaluate
        eval_results = self.evaluate_model_performance()

        # 6. Integrated gradients
        ig_results = self.run_integrated_gradients_analysis(eval_results)

        # 7. Visualisations & exports
        if ig_results:
            self.create_visualizations_and_exports(ig_results, eval_results)

        # 8. Summary
        r2, pearson_r, spearman_rho, _ = eval_results
        print_completion_summary(
            len(self.sequences),
            len(ig_results) if ig_results else 0,
            {"r2": r2, "pearson_r": pearson_r, "spearman_rho": spearman_rho},
            self.config.get("results_dir", "results"),
            self.config["data"]["name"],
        )

        return {"evaluation": eval_results, "integrated_gradients": ig_results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Integrated gradients analysis using DatasetLoader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with PafA embedding-signed config
  python scripts/pipeline/run_ig_dataset.py configs/pafa_config_embedding-signed.yaml

  # Run with betalactamase token-signed config
  python scripts/pipeline/run_ig_dataset.py configs/betalactamase_config_token-signed.yaml

  # List all datasets the registry can discover
  python scripts/pipeline/run_ig_dataset.py --list-datasets

  # Validate a config file without running
  python scripts/pipeline/run_ig_dataset.py configs/erbc_config_embedding-signed.yaml --validate-only
        """,
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        type=str,
        help="Path to a YAML or JSON configuration file",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all datasets discoverable in data/ and exit",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the configuration file without running the analysis",
    )

    args = parser.parse_args()

    # --list-datasets: no config needed
    if args.list_datasets:
        loader = DatasetLoader(str(PROJECT_ROOT / "data"))
        loader.print_registry_summary()
        return

    if not args.config_path:
        parser.error("config_path is required unless --list-datasets is used")

    if not os.path.exists(args.config_path):
        print(f"Error: config file '{args.config_path}' not found.")
        sys.exit(1)

    try:
        config = load_config(args.config_path)
        validate_config(config)
        print(f"✓ Configuration '{args.config_path}' is valid.")

        if args.validate_only:
            return

        runner = IGDatasetRunner(args.config_path)
        runner.run_complete_analysis()
        print("\nAnalysis completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
