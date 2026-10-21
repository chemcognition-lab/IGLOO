"""
Integrated gradients analysis functionality.

This module contains functions for running integrated gradients analysis
on protein sequences using both embedding-level and token-level methods.
"""

import os

import torch

from .completeness_validation import filter_sequences_by_completeness, validate_completeness_axiom_token_level
from .embedding import get_embeddings
from .esm_adapter import load_esm_backend
from .sequence_ig import (
    analyze_sequence_with_integrated_gradients,
    analyze_sequence_with_token_level_integrated_gradients,
    create_combined_model,
)
from .utils import print_analysis_sections, print_ig_sequence_results

_DEFAULT_ESM_MODEL = 'esmc_300m'


def run_integrated_gradients_analysis(sequences_to_analyze, config, y_raw, sequences,
                                     final_model, results_dir, device, y_log_mean, y_log_std,
                                     trainval_sequences=None, test_sequences=None):
    """Run the integrated gradients analysis"""
    print_analysis_sections("PERFORMING INTEGRATED GRADIENTS ANALYSIS")

    ig_config = config['integrated_gradients']
    model_config = config['model']
    embeddings_subdir = 'synthetic' if 'synthetic' in config.get('data', {}).get('path', '') else None
    esm_model_name = model_config.get('esm_model_name', _DEFAULT_ESM_MODEL)
    ig_method = ig_config['method']  # 'embedding' or 'token'

    # Load ESM backend (model + optional alphabet). 'onehot' bypasses the pLM
    # entirely (see embedding.py::get_embeddings) — no backend to load, and
    # the embedding-level combined model never calls esm_model(...) directly.
    if esm_model_name == 'onehot':
        print("Input representation: one-hot amino-acid encoding (no pLM)")
        esm_backend = None
        esm_model = None
        alphabet = None
    else:
        print(f"Loading ESM model for integrated gradients ({esm_model_name})...")
        esm_backend = load_esm_backend(esm_model_name, device)
        esm_model = esm_backend.model
        alphabet  = esm_backend.alphabet  # None for ESM-C

    # Guard: token-level IG requires ESM-2
    if ig_method == 'token' and esm_model_name == 'onehot':
        raise ValueError("Token-level integrated gradients is not supported for the one-hot baseline.")
    if ig_method == 'token' and esm_backend.family != 'esm2':
        raise ValueError(
            f"Token-level integrated gradients is not supported with "
            f"{esm_backend.family!r} models (ESM-2 only). "
            "Set 'integrated_gradients.method: embedding' in your config."
        )
    attribution_method = ig_config.get('attribution_method', 'magnitude')
    interpolation_path = ig_config.get('interpolation_path', 'linear')  # 'linear' | 'slerp' | 'nearest_neighbor' | 'guided_ig'
    plot_umap = ig_config.get('plot_interpolation_umap', False)

    # Pre-load ALL training embeddings from the existing disk cache into a
    # sequence → embedding lookup, so the per-sequence get_embeddings calls
    # inside analyze_sequence_with_integrated_gradients can skip ESM inference.
    # Use trainval_sequences if provided — their embeddings live under the
    # training cache name and the batch boundaries match exactly.
    _lookup_seqs = trainval_sequences if trainval_sequences is not None else sequences
    print(f"Building sequence embedding lookup from training cache ({len(_lookup_seqs)} sequences)...")
    esm_layer = model_config['esm_layer']
    _model_tag = model_config.get('esm_model_name', 'esmc_300m').lower().replace("-", "_")
    _train_cache_name = f"{config['data']['name']}_{_model_tag}_integrated_gradients"
    train_emb_np = get_embeddings(
        _lookup_seqs,
        layer=esm_layer,
        batch_size=model_config.get('batch_size', 16),
        dataset_name=_train_cache_name,
        subdir=embeddings_subdir,
        esm_backend=esm_backend,
        esm_model_name=esm_model_name,
    )  # (N, L, D) numpy array
    embedding_lookup: dict = {seq: train_emb_np[i : i + 1] for i, seq in enumerate(_lookup_seqs)}

    # Also pull in test-set embeddings from their evaluation cache.
    if test_sequences:
        print(f"  Adding test-set embeddings to lookup ({len(test_sequences)} sequences)...")
        _test_cache_name = f"{config['data']['name']}_{_model_tag}_integrated_gradients_test"
        test_emb_np = get_embeddings(
            test_sequences,
            layer=esm_layer,
            batch_size=model_config.get('batch_size', 16),
            dataset_name=_test_cache_name,
            subdir=embeddings_subdir,
            esm_backend=esm_backend,
            esm_model_name=esm_model_name,
        )  # (N, L, D) numpy array
        for i, seq in enumerate(test_sequences):
            embedding_lookup[seq] = test_emb_np[i : i + 1]

    print(f"  Embedding lookup ready: {len(embedding_lookup)} sequences")

    # For nearest_neighbor paths, also expose the full tensor for path grounding.
    training_embeddings: torch.Tensor | None = None
    if interpolation_path == 'nearest_neighbor':
        training_embeddings = torch.from_numpy(train_emb_np).float()
        print(f"  Training embeddings tensor for nearest-neighbour path: {tuple(training_embeddings.shape)}")

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
                embeddings_subdir=embeddings_subdir,
                esm_backend=esm_backend,
                esm_model_name=esm_model_name,
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
                plot_predictions=False,
                sequence_name=name,
                baseline_name="WT",
                save_plot_path=None,
                clip_min=data_min,
                clip_max=data_max,
                embeddings_subdir=embeddings_subdir,
                interpolation_path=interpolation_path,
                plot_interpolation_umap=plot_umap,
                nn_waypoints=ig_config.get('nn_waypoints', 10),
                interpolation_umap_save_path=(
                    os.path.join(results_dir, f"{name}_interpolation_umap.png")
                    if plot_umap else None
                ),
                training_embeddings=training_embeddings,
                _esm_backend=esm_backend,
                esm_model_name=esm_model_name,
                embedding_lookup=embedding_lookup,
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
                plot_predictions=False,
                sequence_name=name,
                baseline_name="WT",
                save_plot_path=None,
                clip_min=data_min,
                clip_max=data_max,
                embeddings_subdir=embeddings_subdir,
                interpolation_path=interpolation_path,
                _esm_backend=esm_backend,
            )
            ig_results[name] = result
            print_ig_sequence_results(name, actual_activity, result, None)
    else:
        raise ValueError(f"Unknown integrated gradients method: {ig_method}")

    return ig_results
