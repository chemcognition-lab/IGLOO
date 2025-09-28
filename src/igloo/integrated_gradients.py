"""
Integrated gradients analysis functionality.

This module contains functions for running integrated gradients analysis
on protein sequences using both embedding-level and token-level methods.
"""

import os

from .sequence_ig import (
    analyze_sequence_with_integrated_gradients,
    analyze_sequence_with_token_level_integrated_gradients,
    create_combined_model,
    load_esm_model,
)
from .utils import print_analysis_sections, print_ig_sequence_results, print_sequence_analysis_info


def run_integrated_gradients_analysis(sequences_to_analyze, config, y_raw, sequences,
                                     final_model, results_dir, device, y_log_mean, y_log_std):
    """Run the integrated gradients analysis"""
    print_analysis_sections("PERFORMING INTEGRATED GRADIENTS ANALYSIS")

    ig_config = config['integrated_gradients']
    model_config = config['model']

    # Load ESM model
    print("Loading ESM model for integrated gradients...")
    esm_model, alphabet = load_esm_model('esm2_t33_650M_UR50D', device)

    print_sequence_analysis_info(sequences_to_analyze, y_raw, sequences)

    # Run analysis based on method
    ig_method = ig_config['method']  # 'embedding' or 'token'
    attribution_method = ig_config.get('attribution_method', 'magnitude')

    ig_results = {}

    if ig_method == 'embedding':
        # Create combined model for embedding-level IG
        print("Creating combined model for embedding-level integrated gradients...")
        combined_model = create_combined_model(
            esm_model=esm_model,
            predictor_model=final_model,
            esm_layer=model_config['esm_layer'],
            pooling_method=model_config['pooling_method'],
            method='embedding',
            device=device
        )


        # Run embedding-level analysis
        for name, sequence, baseline in sequences_to_analyze:
            print(f"\nAnalyzing {name} (embedding-level)...")

            # Get actual activity
            actual_activity = y_raw[sequences.index(sequence)] if sequence in sequences else None
            baseline_actual_activity = (y_raw[sequences.index(baseline)]
                                      if baseline in sequences else None)

            # Get data range for clipping
            data_min = float(min(y_raw))
            data_max = float(max(y_raw))

            result = analyze_sequence_with_integrated_gradients(
                sequence=sequence,
                baseline_sequence=baseline,
                model=combined_model,
                alphabet=alphabet,
                device=device,
                y_log_mean=y_log_mean,
                y_log_std=y_log_std,
                steps=ig_config.get('steps', 50),
                batch_size=ig_config.get('batch_size', 10),
                method='embedding',
                attribution_method=attribution_method,
                actual_activity=actual_activity,
                baseline_actual_activity=baseline_actual_activity,
                plot_predictions=True,
                sequence_name=name,
                baseline_name="WT",
                save_plot_path=os.path.join(results_dir,
                                          f"{config['data']['name']}_{name}_predicted_vs_actual.png"),
                clip_min=data_min,
                clip_max=data_max
            )
            ig_results[name] = result
            print_ig_sequence_results(name, actual_activity, result, None)

    elif ig_method == 'token':
        # Run token-level analysis
        for name, sequence, baseline_sequence in sequences_to_analyze:
            print(f"\nAnalyzing {name} (token-level)...")

            # Get actual activity
            actual_activity = y_raw[sequences.index(sequence)] if sequence in sequences else None

            # baseline_sequence is now directly the sequence string
            baseline_actual_activity = (y_raw[sequences.index(baseline_sequence)]
                                      if baseline_sequence in sequences else None)

            # Get data range for clipping
            data_min = float(min(y_raw))
            data_max = float(max(y_raw))

            result = analyze_sequence_with_token_level_integrated_gradients(
                sequence=sequence,
                baseline_sequence=baseline_sequence,
                esm_model=esm_model,
                predictor_model=final_model,
                alphabet=alphabet,
                device=device,
                esm_layer=model_config['esm_layer'],
                pooling_method=model_config['pooling_method'],
                y_log_mean=y_log_mean,
                y_log_std=y_log_std,
                steps=ig_config.get('steps', 50),
                batch_size=ig_config.get('batch_size', 10),
                attribution_method=attribution_method,
                actual_activity=actual_activity,
                baseline_actual_activity=baseline_actual_activity,
                plot_predictions=True,
                sequence_name=name,
                baseline_name="WT",
                save_plot_path=os.path.join(results_dir,
                                          f"{config['data']['name']}_{name}_predicted_vs_actual.png"),
                clip_min=data_min,
                clip_max=data_max
            )
            ig_results[name] = result
            print_ig_sequence_results(name, actual_activity, result, None)
    else:
        raise ValueError(f"Unknown integrated gradients method: {ig_method}")

    return ig_results
