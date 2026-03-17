"""
Integrated gradients analysis functionality.

This module contains functions for running integrated gradients analysis
on protein sequences using both embedding-level and token-level methods.
"""

import os

from .completeness_validation import filter_sequences_by_completeness, validate_completeness_axiom_token_level
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
    embeddings_subdir = 'synthetic' if 'synthetic' in config.get('data', {}).get('path', '') else None

    # Load ESM model
    print("Loading ESM model for integrated gradients...")
    esm_model, alphabet = load_esm_model('esm2_t33_650M_UR50D', device)

    print_sequence_analysis_info(sequences_to_analyze, y_raw, sequences)

    # Run analysis based on method
    ig_method = ig_config['method']  # 'embedding' or 'token'
    attribution_method = ig_config.get('attribution_method', 'magnitude')

    # Check if completeness validation is enabled
    completeness_config = ig_config.get('completeness_validation', {})
    validation_enabled = completeness_config.get('enabled', True)  # Default to enabled
    completeness_threshold = completeness_config.get('threshold', 0.1)  # Default threshold
    validation_steps = completeness_config.get('validation_steps', 25)  # Faster validation

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

        # Apply completeness validation if enabled
        if validation_enabled:
            sequences_to_analyze, validation_results = filter_sequences_by_completeness(
                sequences_to_analyze=sequences_to_analyze,
                model=combined_model,
                alphabet=alphabet,
                device=device,
                y_log_mean=y_log_mean,
                y_log_std=y_log_std,
                validation_steps=validation_steps,
                attribution_method=attribution_method,
                completeness_threshold=completeness_threshold,
                max_sequences=completeness_config.get('max_sequences_to_validate', None),
                embeddings_subdir=embeddings_subdir
            )

            # Store validation results for later reporting
            ig_results['_validation_results'] = validation_results
        else:
            print("⚠️  Completeness validation disabled - proceeding with all sequences")

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
                clip_max=data_max,
                embeddings_subdir=embeddings_subdir
            )
            ig_results[name] = result
            print_ig_sequence_results(name, actual_activity, result, None)

    elif ig_method == 'token':
        # Apply completeness validation if enabled for token-level analysis
        if validation_enabled:
            # Create a token-level validation function
            def validate_token_sequence(name, sequence, baseline):
                print(f"Validating {name}...", end=" ")


                passes, error, info = validate_completeness_axiom_token_level(
                    sequence, baseline, esm_model, final_model, alphabet, device,
                    model_config['esm_layer'], model_config['pooling_method'],
                    y_log_mean, y_log_std, validation_steps, 10, attribution_method, completeness_threshold,
                    embeddings_subdir=embeddings_subdir
                )

                if passes:
                    print(f"✅ Passed (error={error:.4f})")
                else:
                    print(f"❌ Failed (error={error:.4f} > {completeness_threshold})")

                return passes, error, info

            print(f"\n🔍 VALIDATING TOKEN-LEVEL COMPLETENESS AXIOM (threshold ≤ {completeness_threshold})")
            print("=" * 60)

            filtered_sequences = []
            validation_results = {}

            # Always include WT (first sequence) regardless of completeness
            if sequences_to_analyze and sequences_to_analyze[0][0] == 'WT':
                wt_entry = sequences_to_analyze[0]
                filtered_sequences.append(wt_entry)
                print(f"✓ {wt_entry[0]}: Included (always include WT)")

                # Still validate WT for reporting
                passes, error, info = validate_token_sequence(wt_entry[0], wt_entry[1], wt_entry[2])
                validation_results[wt_entry[0]] = {
                    'passes': passes,
                    'error': error,
                    'info': info,
                    'included': True,
                    'reason': 'WT sequence (always included)'
                }

                sequences_to_validate = sequences_to_analyze[1:]  # Skip WT for validation filtering
            else:
                sequences_to_validate = sequences_to_analyze

            # Validate remaining sequences
            for name, sequence, baseline in sequences_to_validate:
                passes, error, info = validate_token_sequence(name, sequence, baseline)

                validation_results[name] = {
                    'passes': passes,
                    'error': error,
                    'info': info,
                    'included': passes,
                    'reason': 'Passes completeness validation' if passes else f'Fails validation (error={error:.4f})'
                }

                if passes:
                    filtered_sequences.append((name, sequence, baseline))

            # Report filtering results
            print("\n" + "=" * 60)
            print("📊 TOKEN-LEVEL VALIDATION SUMMARY:")
            print(f"   Original sequences: {len(sequences_to_analyze)}")
            print(f"   Sequences passing validation: {len(filtered_sequences)}")
            print(f"   Sequences filtered out: {len(sequences_to_analyze) - len(filtered_sequences)}")

            # List any failed sequences
            failed_sequences = [name for name, result in validation_results.items()
                              if not result['passes'] and 'error' not in result['info']]
            if failed_sequences:
                print("\n⚠️  FAILED SEQUENCES:")
                for name, result in validation_results.items():
                    if not result['passes'] and 'error' not in result['info']:
                        print(f"    {name}: error={result['error']:.4f}")

            print("=" * 60)

            # Update sequences_to_analyze with filtered list
            sequences_to_analyze = filtered_sequences

            # Store validation results for later reporting
            ig_results['_token_validation_results'] = validation_results
        else:
            print("⚠️  Token-level completeness validation disabled - proceeding with all sequences")

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
                clip_max=data_max,
                embeddings_subdir=embeddings_subdir
            )
            ig_results[name] = result
            print_ig_sequence_results(name, actual_activity, result, None)
    else:
        raise ValueError(f"Unknown integrated gradients method: {ig_method}")

    return ig_results
