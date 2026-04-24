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
import time
import traceback
from datetime import datetime

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

# Import analysis modules
from src.igloo.config_utils import (
    load_config,
    select_sequences_for_analysis,
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


class _Tee:
    """Duplicate writes to both the original stdout and a file."""
    def __init__(self, original, file_obj):
        self._orig = original
        self._file = file_obj
    def write(self, data):
        self._orig.write(data)
        self._file.write(data)
    def flush(self):
        self._orig.flush()
        self._file.flush()
    def __getattr__(self, name):
        return getattr(self._orig, name)


def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Merge dot-path key=value strings into cfg in-place.

    Supports nested keys separated by dots, e.g.::

        ig.interpolation_path=slerp
        training.noise.label_noise_std=0.1
        results_dir=results/myjob
    """
    def _cast(v: str):
        if v.lower() in ('true', 'false'):
            return v.lower() == 'true'
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    for item in overrides:
        if '=' not in item:
            print(f"Warning: ignoring malformed override (no '='): {item}")
            continue
        dotpath, _, raw_val = item.partition('=')
        keys = dotpath.split('.')
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _cast(raw_val)


class IntegratedGradientsRunner:
    """Unified runner for integrated gradients analysis with config support."""


    def __init__(self, config_path: str, overrides: list[str] | None = None):
        """Initialize the runner with a configuration file."""
        import random
        self.config = load_config(config_path)
        if overrides:
            _apply_overrides(self.config, overrides)
        validate_config(self.config)

        # Set global random seeds for reproducibility
        seed = self.config.get('training', {}).get('seed', 42)
        try:
            import numpy as np
            import torch
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"Global random seed set to {seed}")
        except Exception as e:
            print(f"[Warning] Could not set all random seeds: {e}")

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

        # Train+val / test split attributes (None when no split is configured)
        self.trainval_idx: "np.ndarray | None" = None
        self.test_idx:     "np.ndarray | None" = None

    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        import numpy as np
        # Construct data path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        data_path = os.path.join(project_root, self.config['data']['path'])

        # load_and_prepare_data returns a tuple: (df, sequences, y_raw, y, y_log_mean, y_log_std)
        result = load_and_prepare_data(self.config, data_path)
        self.df, self.sequences, self.y_raw, self.y, self.y_log_mean, self.y_log_std = result

    def inject_label_noise(self):
        """Optionally corrupt all labels with Gaussian noise before the train/test split.

        Reads ``training.noise.label_noise_std`` from the config (in raw target
        units).  The same noise realisation is applied to both ``y_raw`` and
        the (possibly normalised) ``y``, so model training, evaluation, and IG
        analysis all see consistently noisy labels.

        A fixed ``label_noise_seed`` (default 0) is used so runs are
        reproducible; set it to different values to sample different noise
        realisations.
        """
        import numpy as np

        noise_cfg  = self.config.get('training', {}).get('noise', {})
        noise_std  = float(noise_cfg.get('label_noise_std', 0.0))
        if noise_std <= 0.0:
            return

        seed = int(noise_cfg.get('label_noise_seed', 0))
        rng  = np.random.default_rng(seed)
        epsilon = rng.standard_normal(len(self.y_raw)).astype(np.float32) * noise_std

        self.y_raw = self.y_raw + epsilon
        # y may be z-score normalised: y = (y_raw - mean) / std
        # so the noise in normalised space is epsilon / y_log_std
        self.y = self.y + (epsilon / self.y_log_std)

        print(f"Label noise injected — std={noise_std:.4f} (seed={seed}), "
              f"applied to all {len(self.y_raw)} sequences before train/test split")

    def split_data(self):
        """Split the loaded dataset into train+val and held-out test sets.

        Reads ``training.test_fraction`` from the config.  If absent or 0 the
        method is a no-op and all sequences are used for training (existing
        behaviour).

        For datasets that have a ``variant_type`` column (synthetic data) a
        stratified split per mutation order (single / double / triple) is used
        and the wildtype sequence is always kept in train+val.  For all other
        datasets a simple random split is used.
        """
        import numpy as np

        training_cfg  = self.config.get('training', {})
        test_fraction = float(training_cfg.get('test_fraction', 0.0))

        if test_fraction <= 0.0:
            # No split — use all sequences for training (backward compatible)
            self.trainval_idx = np.arange(len(self.sequences))
            self.test_idx     = np.array([], dtype=int)
            return

        split_seed = int(training_cfg.get('split_seed', 42))

        from igloo.data_processing import split_dataset
        self.trainval_idx, self.test_idx = split_dataset(
            self.df, test_fraction=test_fraction, seed=split_seed
        )

        n_tv   = len(self.trainval_idx)
        n_test = len(self.test_idx)
        print(f"\nData split: {n_tv} train+val ({(1-test_fraction)*100:.0f}%), "
              f"{n_test} held-out test ({test_fraction*100:.0f}%)")

        if 'variant_type' in self.df.columns and n_test > 0:
            test_df_view = self.df.iloc[self.test_idx]
            vc = test_df_view['variant_type'].value_counts()
            for vtype, cnt in vc.items():
                print(f"  Test set — {vtype}: {cnt}")

    def _trainval_sequences(self):
        """Return the train+val sequence list (or all sequences if no split)."""
        import numpy as np
        if self.trainval_idx is None:
            return self.sequences
        return [self.sequences[i] for i in self.trainval_idx]

    def _trainval_y(self):
        """Return the train+val normalised targets (or all targets if no split)."""
        import numpy as np
        if self.trainval_idx is None:
            return self.y
        return self.y[self.trainval_idx]

    def prepare_wildtype_sequence(self):
        """Prepare the wildtype sequence for analysis."""
        self.wt_seq, self.wt_idx = prepare_wildtype_sequence(self.df, self.config)

    def train_or_load_model(self):
        """Train a new model or load an existing one."""
        import numpy as np
        # Construct results directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        results_dir = os.path.join(
            project_root,
            self.config.get('results_dir', f"results/{self.config['data']['name']}_ig")
        )
        os.makedirs(results_dir, exist_ok=True)

        print(f"Results directory: {results_dir}")

        # Save a snapshot of the full config to the results directory
        import yaml
        config_snapshot_path = os.path.join(
            results_dir,
            f"{self.config['data']['name']}_config_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        with open(config_snapshot_path, 'w') as _f:
            yaml.dump(self.config, _f, default_flow_style=False, sort_keys=False)
        print(f"Config snapshot saved to: {config_snapshot_path}")

        # Use only the train+val pool (test set is held out)
        tv_sequences = self._trainval_sequences()
        tv_y         = self._trainval_y()

        # train_or_load_model returns a tuple: (final_model, train_model, train_metrics)
        self.final_model, train_model, train_metrics = train_or_load_model(
            self.config, tv_sequences, tv_y, results_dir, self.device
        )
        self.model_cfg = self.config['model']  # Use the model config from the configuration

        # Print model status
        if train_model:
            print("✓ Model training completed successfully")
            # CV summary
            if 'mean_val_r2' in train_metrics:
                n_folds = self.config.get('training', {}).get('n_folds', 5)
                print(f"  {n_folds}-fold CV — mean val R² = {train_metrics['mean_val_r2']:.4f} "
                      f"(±{train_metrics.get('std_val_r2', 0):.4f})")
            elif 'train_r2' in train_metrics:
                print(f"  Training R² = {train_metrics['train_r2']:.4f}")
        else:
            print("✓ Pre-trained model loaded successfully")

    def evaluate_model_performance(self):
        """Evaluate the trained model's performance on the held-out test set.

        If no test split is configured, falls back to evaluating on all data
        (existing behaviour).
        """
        import numpy as np
        # Construct results directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        default_results_dir = f"results/{self.config['data']['name']}_ig"
        results_dir = os.path.join(project_root, self.config.get('results_dir', default_results_dir))

        has_test = self.test_idx is not None and len(self.test_idx) > 0

        if has_test:
            # Evaluate on held-out test set
            test_sequences = [self.sequences[i] for i in self.test_idx]
            test_y         = self.y[self.test_idx]
            test_y_raw     = self.y_raw[self.test_idx]

            print(f"\nTest-set evaluation ({len(test_sequences)} sequences):")
            test_eval = evaluate_model_performance(
                self.final_model, test_sequences, test_y, test_y_raw,
                self.y_log_mean, self.y_log_std,
                self.config, results_dir, self.device,
                split_name='test',
            )
            return test_eval
        else:
            # No test split — evaluate on all sequences (backward compatible)
            return evaluate_model_performance(
                self.final_model, self.sequences, self.y, self.y_raw,
                self.y_log_mean, self.y_log_std, self.config, results_dir, self.device,
                split_name='train',
            )

    def run_integrated_gradients_analysis(self, eval_results=None):
        """Run the integrated gradients analysis on all sequences (train+val and test).

        Sequences are selected from the full dataset; each selected sequence is
        labelled with ``[test]`` or ``[trainval]`` so results can be differentiated.
        The wildtype, lowest/highest/median activity sequences are still tagged as
        such (with their split appended).
        """
        import numpy as np

        # Construct results directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        default_results_dir = f"results/{self.config['data']['name']}_ig"
        results_dir = os.path.join(project_root, self.config.get('results_dir', default_results_dir))

        has_split = (self.test_idx is not None and len(self.test_idx) > 0)

        # Build a fast lookup: sequence string → split label
        if has_split:
            test_seqs  = {self.sequences[i] for i in self.test_idx}
            def _split_label(seq: str) -> str:
                return "test" if seq in test_seqs else "trainval"
        else:
            def _split_label(seq: str) -> str:  # noqa: F811
                return "all"

        # Get sequences to analyze based on configuration (from full dataset)
        # Config can store sequence_selection either at the top level or under 'analysis'.
        _default_sel = {
            'include_extremes': True,
            'include_median': True,
            'include_top_n': 0,
            'include_bottom_n': 0,
            'custom_sequences': [],
            'max_error_percent': 10.0,
        }
        selection_config = (
            self.config.get('analysis', {}).get('sequence_selection')
            or self.config.get('sequence_selection')
            or _default_sel
        )

        # y_pred is not available over the full dataset here (eval was on test only),
        # so pass None to avoid misleading error-filtering.
        sequences_to_analyze = select_sequences_for_analysis(
            self.df, self.y_raw, self.wt_seq, selection_config,
            model=self.final_model, sequences=self.sequences, y_pred=None
        )

        # Annotate each sequence name with its split label
        if has_split:
            annotated = []
            for name, seq, baseline in sequences_to_analyze:
                label = _split_label(seq)
                annotated.append((f"{name}[{label}]", seq, baseline))
            sequences_to_analyze = annotated

        # Always append the poly-alanine special case
        poly_ala = 'A' * len(self.wt_seq)
        sequences_to_analyze.append(('PolyAlanine', poly_ala, self.wt_seq))

        print(f"\nSelected {len(sequences_to_analyze)} sequences for integrated gradients "
              f"analysis (from {'train+val and test' if has_split else 'full dataset'}):")
        for name, seq, _baseline in sequences_to_analyze:
            try:
                seq_idx = self.sequences.index(seq)
                activity = self.y_raw[seq_idx]
                print(f"  {name}: activity = {activity:.4f}")
            except ValueError:
                print(f"  {name}: custom sequence (not in dataset)")

        return run_integrated_gradients_analysis(
            sequences_to_analyze, self.config, self.y_raw, self.sequences,
            self.final_model, results_dir, self.device, self.y_log_mean, self.y_log_std,
            trainval_sequences=self._trainval_sequences(),
            test_sequences=[self.sequences[i] for i in self.test_idx] if (self.test_idx is not None and len(self.test_idx) > 0) else None,
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

    def _zip_per_sequence_dirs(self, results_dir: str) -> None:
        """Zip any *_per_sequence directories in results_dir and remove the originals."""
        import zipfile
        import glob

        pattern = os.path.join(results_dir, '*_per_sequence')
        dirs = [p for p in glob.glob(pattern) if os.path.isdir(p)]
        if not dirs:
            return

        for dir_path in dirs:
            zip_path = dir_path + '.zip'
            print(f"Zipping {os.path.basename(dir_path)} → {os.path.basename(zip_path)} ...", end=' ', flush=True)
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(dir_path):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        arcname = os.path.relpath(fpath, os.path.dirname(dir_path))
                        zf.write(fpath, arcname)
            # Remove original directory only after successful zip
            import shutil
            shutil.rmtree(dir_path)
            print("done")

    def _write_timing_report(self, timing: dict, results_dir: str) -> None:
        """Write a timing summary to a .txt file in the results directory."""
        os.makedirs(results_dir, exist_ok=True)
        report_path = os.path.join(results_dir, f"{self.config['data']['name']}_timing_report.txt")

        total = timing.get('total', 0.0)
        n_ig = timing.get('n_ig_sequences', 0)

        lines = [
            "=" * 60,
            "PIPELINE TIMING REPORT",
            "=" * 60,
            f"Dataset      : {self.config['data']['name']}",
            f"Started at   : {timing.get('start_time', 'N/A')}",
            f"Finished at  : {timing.get('end_time', 'N/A')}",
            f"Device       : {self.device}",
            "-" * 60,
            f"{'Step':<40} {'Duration':>10}",
            "-" * 60,
        ]

        step_keys = [
            ('data_loading',        'Data loading & preparation'),
            ('wildtype_prep',       'Wildtype sequence preparation'),
            ('embedding_retrieval', 'Embedding retrieval (if needed)'),
            ('model_train_load',    'Model training / loading'),
            ('model_evaluation',    'Model evaluation'),
            ('integrated_gradients', f'Integrated gradients ({n_ig} sequence(s))'),
            ('visualizations',      'Visualizations & exports'),
        ]

        for key, label in step_keys:
            secs = timing.get(key)
            if secs is not None:
                lines.append(f"  {label:<38} {self._fmt_duration(secs):>10}")

        lines += [
            "-" * 60,
            f"  {'TOTAL':<38} {self._fmt_duration(total):>10}",
            "=" * 60,
        ]

        report = "\n".join(lines) + "\n"
        print(report)

        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Timing report saved to {report_path}")

    @staticmethod
    def _fmt_duration(secs: float) -> str:
        """Format seconds as a human-readable string."""
        if secs < 60:
            return f"{secs:.1f}s"
        m, s = divmod(int(secs), 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s"

    def run_complete_analysis(self):
        """Run the complete integrated gradients analysis pipeline."""
        # Set up results dir and log file before anything else
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        results_dir = os.path.join(
            project_root,
            self.config.get('results_dir', f"results/{self.config['data']['name']}_ig")
        )
        os.makedirs(results_dir, exist_ok=True)
        _run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(
            results_dir,
            f"{self.config['data']['name']}_run_log_{_run_ts}.txt"
        )
        _log_file = open(log_path, 'w')
        _orig_stdout = sys.stdout
        sys.stdout = _Tee(_orig_stdout, _log_file)

        timing: dict = {}
        pipeline_start = time.perf_counter()
        timing['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            print("="*60)
            print("INTEGRATED GRADIENTS ANALYSIS PIPELINE")
            print("="*60)
            print(f"Device: {self.device}")
            print()

            # Print key settings summary
            ig_cfg       = self.config.get('integrated_gradients', {})
            model_cfg    = self.config.get('model', {})
            training_cfg = self.config.get('training', {})
            noise_cfg    = training_cfg.get('noise', {})
            print("── Settings ─────────────────────────────────────────")
            print(f"  dataset          : {self.config['data']['name']}")
            print(f"  data path        : {self.config['data'].get('path', 'N/A')}")
            print(f"  esm model        : {model_cfg.get('esm_model_name', 'esmc_300m')}")
            print(f"  esm layer        : {model_cfg.get('esm_layer', 'N/A')}")
            print(f"  pooling          : {model_cfg.get('pooling_method', 'N/A')}")
            print(f"  ig method        : {ig_cfg.get('method', 'N/A')}")
            print(f"  interp path      : {ig_cfg.get('interpolation_path', 'N/A')}")
            print(f"  ig steps         : {ig_cfg.get('steps', 'N/A')}")
            print(f"  label noise std  : {noise_cfg.get('label_noise_std', 0.0)}")
            print(f"  test fraction    : {training_cfg.get('test_fraction', 0.0)}")
            print(f"  n_folds          : {training_cfg.get('n_folds', 5)}")
            print(f"  epochs           : {model_cfg.get('epochs', 'N/A')}")
            print(f"  results dir      : {self.config.get('results_dir', 'N/A')}")
            print(f"  model dir        : {self.config.get('model_dir', 'N/A')}")
            print(f"  force retrain    : {self.config.get('force_retrain', False)}")
            print("─────────────────────────────────────────────────────")
            print()

            # Step 1: Load and prepare data
            _t = time.perf_counter()
            self.load_and_prepare_data()
            timing['data_loading'] = time.perf_counter() - _t

            # Step 1b: Inject label noise (if configured) — before any split
            self.inject_label_noise()

            # Step 2: Split into train+val and held-out test (if configured)
            self.split_data()

            # Step 3: Prepare wildtype sequence
            _t = time.perf_counter()
            self.prepare_wildtype_sequence()
            timing['wildtype_prep'] = time.perf_counter() - _t

            # Step 4: Train or load model
            _t = time.perf_counter()
            self.train_or_load_model()
            timing['model_train_load'] = time.perf_counter() - _t
            timing['embedding_retrieval'] = 0.0

            # Step 5: Evaluate model performance (on test set if split is configured)
            _t = time.perf_counter()
            eval_results = self.evaluate_model_performance()
            timing['model_evaluation'] = time.perf_counter() - _t

            # Step 6: Run integrated gradients analysis
            _t = time.perf_counter()
            ig_results = self.run_integrated_gradients_analysis(eval_results)
            timing['integrated_gradients'] = time.perf_counter() - _t
            # Count analysed sequences (exclude internal keys starting with '_')
            timing['n_ig_sequences'] = sum(
                1 for k in (ig_results or {}) if not k.startswith('_')
            )

            # Step 7: Create visualizations and exports
            _t = time.perf_counter()
            if ig_results:
                self.create_visualizations_and_exports(ig_results, eval_results)
            # Zip per-sequence plot directories to save inodes/space
            self._zip_per_sequence_dirs(results_dir)
            timing['visualizations'] = time.perf_counter() - _t

            timing['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            timing['total'] = time.perf_counter() - pipeline_start

            # Step 8: Print completion summary
            r2, pearson_r, spearman_rho, _ = eval_results
            eval_results_dict = {
                'r2': r2,
                'pearson_r': pearson_r,
                'spearman_rho': spearman_rho
            }
            split_label = "test" if (self.test_idx is not None and len(self.test_idx) > 0) else "train"
            print(f"\n(Metrics above are on the held-out {split_label} set)")

            print_completion_summary(
                len(self.sequences),
                len(ig_results) if ig_results else 0,
                eval_results_dict,
                self.config.get('results_dir', 'results'),
                self.config['data']['name']
            )

            # Step 9: Write timing report
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
            results_dir = os.path.join(
                project_root,
                self.config.get('results_dir', f"results/{self.config['data']['name']}_ig")
            )
            self._write_timing_report(timing, results_dir)

            return {
                'evaluation': eval_results,
                'integrated_gradients': ig_results,
                'timing': timing,
            }

        except Exception as e:
            print(f"Error during analysis: {e}")
            traceback.print_exc()
            raise
        finally:
            sys.stdout = _orig_stdout
            _log_file.close()
            print(f"Run log saved to {log_path}")


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

    parser.add_argument(
        'overrides',
        nargs='*',
        metavar='key.subkey=value',
        help='Optional dot-path overrides applied on top of the config, e.g. '
             'ig.interpolation_path=slerp  training.noise.label_noise_std=0.1  '
             'results_dir=results/myjob'
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config_path):
        print(f"Error: Configuration file '{args.config_path}' not found.")
        sys.exit(1)

    try:
        # Load and validate config (apply any CLI overrides first)
        config = load_config(args.config_path)
        if args.overrides:
            _apply_overrides(config, args.overrides)
        validate_config(config)
        print(f"Configuration file '{args.config_path}' is valid!")

        if args.validate_only:
            print("Validation complete. Exiting.")
            return

        # Run analysis
        runner = IntegratedGradientsRunner(args.config_path, overrides=args.overrides)
        results = runner.run_complete_analysis()

        print("\nAnalysis completed successfully!")
        return results

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
