"""
Configuration utilities for integrated gradients analysis.

This module provides functions for loading, validating, and managing configuration files
for the unified integrated gradients analysis pipeline.
"""

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the file format is not supported
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    return config


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and set defaults for configuration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Validated configuration with defaults applied

    Raises:
        ValueError: If required configuration keys are missing
    """
    required_keys = [
        'data.name',
        'data.path',
        'data.target_column',
        'data.wildtype',
        'model.esm_layer',
        'model.pooling_method',
        'integrated_gradients.method'
    ]

    # Check required keys
    for key in required_keys:
        keys = key.split('.')
        current = config
        for k in keys:
            if k not in current:
                raise ValueError(f"Missing required config key: {key}")
            current = current[k]

    # Set defaults
    defaults = {
        'data': {
            'normalization': 'z-score'
        },
        'model': {
            'esm_model_name': 'esmc_300m',
            'hidden_dim1': 1024,
            'hidden_dim2': 512,
            'hidden_dim3': 255,
            'hidden_dim4': 128,
            'dropout_rate': 0.2,
            'learning_rate': 1e-3,
            'batch_size': 256,
            'epochs': 100,
            'loss_type': 'huber',
            'final_activation': None
        },
        'integrated_gradients': {
            'steps': 50,
            'batch_size': 10,
            'attribution_method': 'magnitude',
            'method': 'embedding',
        },
        'analysis': {
            'sequence_selection': {
                'include_extremes': True,
                'include_median': True,
                'include_top_n': 0,
                'include_bottom_n': 0,
                'custom_sequences': []
            }
        },
        'training': {
            'noise': {
                'label_noise_std': 0.0,   # σ in raw target units; 0 = no noise
                'label_noise_seed': 0,    # RNG seed for reproducibility
                'input_noise_std': 0.0,   # σ for per-batch embedding augmentation; 0 = off
            }
        },
        'force_retrain': False
    }

    # Apply defaults
    def apply_defaults(config_dict, defaults_dict):
        for key, value in defaults_dict.items():
            if isinstance(value, dict):
                if key not in config_dict:
                    config_dict[key] = {}
                apply_defaults(config_dict[key], value)
            else:
                if key not in config_dict:
                    config_dict[key] = value

    apply_defaults(config, defaults)

    return config


def get_wildtype_sequence(df, wt_config: dict[str, Any]) -> tuple[str, int]:
    """
    Get the wild-type sequence based on configuration.

    Args:
        df: Pandas DataFrame containing the sequence data
        wt_config: Wild-type configuration dictionary

    Returns:
        Tuple of (wild_type_sequence, wild_type_index)

    Raises:
        ValueError: If wild-type sequence cannot be found or invalid method specified
    """
    if wt_config['method'] == 'variant_column':
        # Find WT by variant name
        variant_col = wt_config['variant_column']
        wt_value = wt_config['wildtype_value']
        wt_df = df[df[variant_col] == wt_value]

        if len(wt_df) == 0:
            raise ValueError(f"No wild-type sequence found with {variant_col}='{wt_value}'")

        wt_seq = wt_df['SEQ'].iloc[0]
        wt_idx = wt_df.index[0]

    elif wt_config['method'] == 'position':
        # Get WT by position (e.g., last sequence)
        position = wt_config['position']
        if position == 'last':
            wt_idx = df.index[-1]
        elif position == 'first':
            wt_idx = df.index[0]
        else:
            wt_idx = position
        wt_seq = df['SEQ'].loc[wt_idx]

    elif wt_config['method'] == 'sequence':
        # Use provided sequence - find its index if it exists in the dataset
        wt_seq = wt_config['sequence']
        matching_rows = df[df['SEQ'] == wt_seq]
        if len(matching_rows) > 0:
            wt_idx = matching_rows.index[0]
        else:
            wt_idx = -1  # Not found in dataset

    else:
        raise ValueError(f"Unknown wildtype method: {wt_config['method']}")

    return wt_seq, wt_idx


def select_sequences_for_analysis(df, y_raw: list, wt_seq: str,
                                 selection_config: dict[str, Any],
                                 model=None, sequences=None, y_pred=None) -> list[tuple[str, str, str]]:
    """
    Select sequences for integrated gradients analysis based on configuration.

    Args:
        df: Pandas DataFrame containing sequence data
        y_raw: Raw target values
        wt_seq: Wild-type sequence
        selection_config: Sequence selection configuration
        model: Optional trained model for prediction error filtering
        sequences: Optional list of sequences (for prediction error filtering)
        y_pred: Optional pre-computed predictions (for prediction error filtering)

    Returns:
        List of tuples (name, sequence, baseline_sequence)
    """
    import numpy as np

    df_sequences = df['SEQ'].astype(str).tolist()
    sequences_to_analyze = []

    # Always include wild-type
    sequences_to_analyze.append(('WT', wt_seq, wt_seq))

    # Get prediction errors if model is provided for filtering
    prediction_errors = None
    if model is not None and sequences is not None and y_pred is not None:
        # Calculate percentage errors: |actual - predicted| / |actual| * 100
        prediction_errors = np.abs(np.array(y_raw) - np.array(y_pred)) / np.abs(np.array(y_raw)) * 100
        print(f"Computed prediction errors. Mean error: {np.mean(prediction_errors):.2f}%")

    # Add sequences based on activity levels
    sorted_indices = np.argsort(y_raw)

    if selection_config.get('include_extremes', True):
        # Lowest and highest activity (no error filtering for extremes)
        sequences_to_analyze.extend([
            ('Lowest_Activity', df_sequences[sorted_indices[0]], wt_seq),
            ('Highest_Activity', df_sequences[sorted_indices[-1]], wt_seq)
        ])

    if selection_config.get('include_median', True):
        # Median activity (no error filtering for median)
        mid_idx = sorted_indices[len(sorted_indices)//2]
        sequences_to_analyze.append(('Median_Activity', df_sequences[mid_idx], wt_seq))

    # For top_n and bottom_n, apply error filtering if available
    error_threshold = selection_config.get('max_error_percent', 10.0)  # Default 10% error threshold

    if selection_config.get('include_top_n', 0) > 0:
        # Top N sequences with optional error filtering
        top_n = selection_config['include_top_n']

        if prediction_errors is not None:
            # Filter by error first, then sort by activity
            good_prediction_mask = prediction_errors <= error_threshold
            good_indices = np.where(good_prediction_mask)[0]

            if len(good_indices) > 0:
                # Sort good predictions by activity (descending)
                good_activities = np.array(y_raw)[good_indices]
                good_sorted_indices = good_indices[np.argsort(good_activities)[::-1]]

                # Skip the highest if it's already included in extremes
                start_idx = 1 if selection_config.get('include_extremes', True) else 0
                selected_count = 0

                for i in range(start_idx, len(good_sorted_indices)):
                    if selected_count >= top_n:
                        break
                    idx = good_sorted_indices[i]
                    error_pct = prediction_errors[idx]
                    sequences_to_analyze.append((f'Top_{selected_count+2}_err{error_pct:.1f}%',
                                               df_sequences[idx], wt_seq))
                    selected_count += 1

                print(f"Selected {selected_count}/{top_n} top sequences with <{error_threshold}% error")
            else:
                print(f"Warning: No sequences found with <{error_threshold}% error for top_n selection")
        else:
            # No error filtering - use original logic
            for i in range(min(top_n, len(sorted_indices))):
                idx = sorted_indices[-(i+2)]  # Skip the highest (already included)
                sequences_to_analyze.append((f'Top_{i+2}', df_sequences[idx], wt_seq))

    if selection_config.get('include_bottom_n', 0) > 0:
        # Bottom N sequences with optional error filtering
        bottom_n = selection_config['include_bottom_n']

        if prediction_errors is not None:
            # Filter by error first, then sort by activity
            good_prediction_mask = prediction_errors <= error_threshold
            good_indices = np.where(good_prediction_mask)[0]

            if len(good_indices) > 0:
                # Sort good predictions by activity (ascending)
                good_activities = np.array(y_raw)[good_indices]
                good_sorted_indices = good_indices[np.argsort(good_activities)]

                # Skip the lowest if it's already included in extremes
                start_idx = 1 if selection_config.get('include_extremes', True) else 0
                selected_count = 0

                for i in range(start_idx, len(good_sorted_indices)):
                    if selected_count >= bottom_n:
                        break
                    idx = good_sorted_indices[i]
                    error_pct = prediction_errors[idx]
                    sequences_to_analyze.append((f'Bottom_{selected_count+2}_err{error_pct:.1f}%',
                                               df_sequences[idx], wt_seq))
                    selected_count += 1

                print(f"Selected {selected_count}/{bottom_n} bottom sequences with <{error_threshold}% error")
            else:
                print(f"Warning: No sequences found with <{error_threshold}% error for bottom_n selection")
        else:
            # No error filtering - use original logic
            for i in range(min(bottom_n, len(sorted_indices))):
                idx = sorted_indices[i+1]  # Skip the lowest (already included)
                sequences_to_analyze.append((f'Bottom_{i+2}', df_sequences[idx], wt_seq))

    # Add any custom sequences specified in config
    custom_sequences = selection_config.get('custom_sequences', [])
    for custom in custom_sequences:
        if 'sequence' in custom:
            sequences_to_analyze.append((custom['name'], custom['sequence'], wt_seq))
        elif 'index' in custom:
            idx = custom['index']
            sequences_to_analyze.append((custom['name'], sequences[idx], wt_seq))

    # include_all: add every sequence in the dataset (deduplicating against already-selected)
    if selection_config.get('include_all', False):
        already_seqs = {seq for _, seq, _ in sequences_to_analyze}
        # Try to find a human-readable name column (mutation_string preferred)
        name_col = None
        for candidate in ('mutation_string', 'name', 'variant_id', 'id'):
            if candidate in df.columns:
                name_col = candidate
                break
        for i, seq in enumerate(df_sequences):
            if seq in already_seqs:
                continue
            if name_col is not None:
                row_name = str(df.iloc[i][name_col])
            else:
                row_name = f'seq_{i}'
            sequences_to_analyze.append((row_name, seq, wt_seq))
            already_seqs.add(seq)
        print(f"include_all: {len(sequences_to_analyze)} total sequences queued for IG analysis.")

    return sequences_to_analyze


def validate_data_files(config: dict[str, Any], project_root: Path = None) -> dict[str, bool]:
    """
    Validate that data files referenced in config exist.

    Args:
        config: Configuration dictionary
        project_root: Project root directory (if None, infers from config path)

    Returns:
        Dictionary mapping file paths to existence status
    """
    if project_root is None:
        project_root = Path.cwd()

    results = {}

    # Check main data file
    data_path = project_root / config['data']['path']
    results[config['data']['path']] = data_path.exists()

    # For beta-lactamase, also check if test file exists
    if 'betalactamase' in config['data']['name'].lower():
        test_path = data_path.parent / 'betalactamase_test.csv'
        results['betalactamase_test.csv'] = test_path.exists()

    return results


def print_config_summary(config: dict[str, Any]) -> None:
    """
    Print a summary of the configuration.

    Args:
        config: Configuration dictionary
    """
    print("Configuration Summary:")
    print(f"  Dataset: {config['data']['name']}")
    print(f"  Target: {config['data']['target_column']}")
    print(f"  Normalization: {config['data'].get('normalization', 'z-score')}")
    print(f"  WT Method: {config['data']['wildtype']['method']}")
    print(f"  ESM Layer: {config['model']['esm_layer']}")
    print(f"  Pooling: {config['model']['pooling_method']}")
    print(f"  IG Method: {config['integrated_gradients']['method']}")
    print(f"  Attribution: {config['integrated_gradients'].get('attribution_method', 'magnitude')}")
    print(f"  Results Dir: {config.get('results_dir', 'auto')}")
