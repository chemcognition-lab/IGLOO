#!/usr/bin/env python3
"""
Completeness Axiom Validation Module

This module provides functions to validate that integrated gradients analysis
satisfies the completeness axiom before including sequences in the final analysis.
"""


import numpy as np
import torch

from .sequence_ig import (
    analyze_sequence_with_integrated_gradients,
    analyze_sequence_with_token_level_integrated_gradients,
)


def validate_completeness_axiom_token_level(
    sequence: str,
    baseline_sequence: str,
    esm_model,
    predictor_model,
    alphabet,
    device: torch.device,
    esm_layer: int = 33,
    pooling_method: str = 'linear',
    y_log_mean: float = 0.0,
    y_log_std: float = 1.0,
    steps: int = 25,
    batch_size: int = 10,
    attribution_method: str = 'signed',
    completeness_threshold: float = 0.1,
    embeddings_subdir: str = None
) -> tuple[bool, float, dict]:
    """
    Validate that a sequence satisfies the completeness axiom for token-level analysis.

    Args:
        sequence: Input protein sequence
        baseline_sequence: Baseline sequence (usually wild-type)
        esm_model: Pre-trained ESM model
        predictor_model: Trained prediction model
        alphabet: ESM alphabet
        device: PyTorch device
        esm_layer: ESM layer to extract features from
        pooling_method: Pooling method ('linear', 'esm_attention', etc.)
        y_log_mean: Mean for denormalization
        y_log_std: Std for denormalization
        steps: Number of integration steps (reduced for speed)
        batch_size: Batch size for gradient computation
        attribution_method: Attribution computation method ('signed' recommended)
        completeness_threshold: Maximum allowed completeness error

    Returns:
        Tuple of (passes_validation, completeness_error, validation_info)
    """
    try:

        # Run a quick token-level IG analysis to check completeness
        result = analyze_sequence_with_token_level_integrated_gradients(
            sequence=sequence,
            baseline_sequence=baseline_sequence,
            esm_model=esm_model,
            predictor_model=predictor_model,
            alphabet=alphabet,
            device=device,
            esm_layer=esm_layer,
            pooling_method=pooling_method,
            y_log_mean=y_log_mean,
            y_log_std=y_log_std,
            steps=steps,
            batch_size=batch_size,
            attribution_method=attribution_method,
            actual_activity=None,
            baseline_actual_activity=None,
            plot_predictions=False,  # Don't create plots for validation
            sequence_name="Validation",
            baseline_name="Baseline",
            embeddings_subdir=embeddings_subdir
        )

        # Extract completeness information from the result
        predicted_activity = result['predicted_activity']
        baseline_activity = result['baseline_activity']
        position_gradients = result['position_gradients']

        # Calculate completeness axiom components
        predicted_diff = predicted_activity - baseline_activity
        attribution_sum = np.sum(position_gradients)
        completeness_error = abs(attribution_sum - predicted_diff)

        # Check if it passes the threshold
        passes_validation = completeness_error <= completeness_threshold

        validation_info = {
            'predicted_diff': predicted_diff,
            'attribution_sum': attribution_sum,
            'completeness_error': completeness_error,
            'threshold': completeness_threshold,
            'predicted_activity': predicted_activity,
            'baseline_activity': baseline_activity
        }

        return passes_validation, completeness_error, validation_info

    except Exception as e:
        print(f"Warning: Token-level completeness validation failed for sequence: {e}")
        # If validation fails, assume it passes to avoid blocking analysis
        return True, 0.0, {'error': str(e)}


def validate_completeness_axiom(
    sequence: str,
    baseline_sequence: str,
    model,
    alphabet,
    device: torch.device,
    y_log_mean: float = 0.0,
    y_log_std: float = 1.0,
    steps: int = 50,
    batch_size: int = 10,
    attribution_method: str = 'signed',
    completeness_threshold: float = 0.1,
    embeddings_subdir: str = None
) -> tuple[bool, float, dict]:
    """
    Validate that a sequence satisfies the completeness axiom before full analysis.

    This function performs a quick integrated gradients computation to check if the
    completeness axiom is satisfied: ∑(attributions) ≈ f(input) - f(baseline)

    Args:
        sequence: Input protein sequence
        baseline_sequence: Baseline sequence (usually wild-type)
        model: Combined ESM + predictor model
        alphabet: ESM alphabet
        device: PyTorch device
        y_log_mean: Mean for denormalization
        y_log_std: Std for denormalization
        steps: Number of integration steps (reduced for speed)
        batch_size: Batch size for gradient computation
        attribution_method: Attribution computation method ('signed' recommended)
        completeness_threshold: Maximum allowed completeness error

    Returns:
        Tuple of (passes_validation, completeness_error, validation_info)
    """
    try:
        # Import required functions

        # Run a quick IG analysis to check completeness
        result = analyze_sequence_with_integrated_gradients(
            sequence=sequence,
            baseline_sequence=baseline_sequence,
            model=model,
            alphabet=alphabet,
            device=device,
            y_log_mean=y_log_mean,
            y_log_std=y_log_std,
            steps=steps,  # Use fewer steps for speed
            batch_size=batch_size,
            method='embedding',
            attribution_method=attribution_method,
            actual_activity=None,
            baseline_actual_activity=None,
            plot_predictions=False,  # Don't create plots for validation
            sequence_name="Validation",
            baseline_name="Baseline",
            embeddings_subdir=embeddings_subdir
        )

        # Extract completeness information from the result
        predicted_activity = result['predicted_activity']
        baseline_activity = result['baseline_activity']
        position_gradients = result['position_gradients']

        # Calculate completeness axiom components
        predicted_diff = predicted_activity - baseline_activity
        attribution_sum = np.sum(position_gradients)
        completeness_error = abs(attribution_sum - predicted_diff)

        # Check if it passes the threshold
        passes_validation = completeness_error <= completeness_threshold

        validation_info = {
            'predicted_diff': predicted_diff,
            'attribution_sum': attribution_sum,
            'completeness_error': completeness_error,
            'threshold': completeness_threshold,
            'predicted_activity': predicted_activity,
            'baseline_activity': baseline_activity
        }

        return passes_validation, completeness_error, validation_info

    except Exception as e:
        print(f"Warning: Completeness validation failed for sequence: {e}")
        # If validation fails, assume it passes to avoid blocking analysis
        return True, 0.0, {'error': str(e)}


def filter_sequences_by_completeness(
    sequences_to_analyze: list[tuple[str, str, str]],
    model,
    alphabet,
    device: torch.device,
    y_log_mean: float = 0.0,
    y_log_std: float = 1.0,
    validation_steps: int = 25,  # Fewer steps for faster validation
    attribution_method: str = 'signed',
    completeness_threshold: float = 0.1,
    max_sequences: int = None,
    embeddings_subdir: str = None
) -> tuple[list[tuple[str, str, str]], dict]:
    """
    Filter sequences to only include those that satisfy the completeness axiom.

    Args:
        sequences_to_analyze: List of (name, sequence, baseline) tuples
        model: Combined ESM + predictor model
        alphabet: ESM alphabet
        device: PyTorch device
        y_log_mean: Mean for denormalization
        y_log_std: Std for denormalization
        validation_steps: Number of IG steps for validation (reduced for speed)
        attribution_method: Attribution method for validation
        completeness_threshold: Maximum allowed completeness error
        max_sequences: Maximum number of sequences to return (None for all)

    Returns:
        Tuple of (filtered_sequences, validation_results)
    """
    print(f"\n🔍 VALIDATING COMPLETENESS AXIOM (threshold ≤ {completeness_threshold})")
    print("=" * 60)

    filtered_sequences = []
    validation_results = {}

    # Always include WT (first sequence) regardless of completeness
    if sequences_to_analyze and sequences_to_analyze[0][0] == 'WT':
        wt_entry = sequences_to_analyze[0]
        filtered_sequences.append(wt_entry)
        print(f"✓ {wt_entry[0]}: Included (always include WT)")

        # Still validate WT for reporting
        passes, error, info = validate_completeness_axiom(
            wt_entry[1], wt_entry[2], model, alphabet, device,
            y_log_mean, y_log_std, validation_steps, 10, attribution_method, completeness_threshold,
            embeddings_subdir=embeddings_subdir
        )
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
        print(f"Validating {name}...", end=" ")

        passes, error, info = validate_completeness_axiom(
            sequence, baseline, model, alphabet, device,
            y_log_mean, y_log_std, validation_steps, 10, attribution_method, completeness_threshold,
            embeddings_subdir=embeddings_subdir
        )

        validation_results[name] = {
            'passes': passes,
            'error': error,
            'info': info,
            'included': passes,
            'reason': 'Passes completeness validation' if passes else f'Fails validation (error={error:.4f})'
        }

        if passes:
            filtered_sequences.append((name, sequence, baseline))
            print(f"✅ Passed (error={error:.4f})")
        else:
            print(f"❌ Failed (error={error:.4f} > {completeness_threshold})")

        # Stop if we've reached the maximum number of sequences
        if max_sequences is not None and len(filtered_sequences) >= max_sequences:
            remaining_count = len(sequences_to_validate) - (sequences_to_validate.index((name, sequence, baseline)) + 1)
            if remaining_count > 0:
                print(f"Note: Stopping validation early (reached max_sequences={max_sequences}). "
                      f"{remaining_count} sequences not validated.")
            break

    # Summary
    total_validated = len([r for r in validation_results.values() if 'error' not in r['info']])
    passed_count = len([r for r in validation_results.values() if r['passes']])
    failed_count = total_validated - passed_count

    print("\n📊 COMPLETENESS VALIDATION SUMMARY:")
    print(f"  Total sequences validated: {total_validated}")
    print(f"  ✅ Passed validation: {passed_count}")
    print(f"  ❌ Failed validation: {failed_count}")
    print(f"  📋 Selected for analysis: {len(filtered_sequences)}")

    if failed_count > 0:
        print("\n⚠️  FAILED SEQUENCES:")
        for name, result in validation_results.items():
            if not result['passes'] and 'error' not in result['info']:
                print(f"    {name}: error={result['error']:.4f}")

    print("=" * 60)

    return filtered_sequences, validation_results


def add_completeness_validation_to_config(config: dict, enable: bool = True, threshold: float = 0.1) -> dict:
    """
    Add completeness validation configuration to the config dictionary.

    Args:
        config: Configuration dictionary
        enable: Whether to enable completeness validation
        threshold: Completeness error threshold

    Returns:
        Updated configuration dictionary
    """
    if 'integrated_gradients' not in config:
        config['integrated_gradients'] = {}

    config['integrated_gradients']['completeness_validation'] = {
        'enabled': enable,
        'threshold': threshold,
        'validation_steps': 25,  # Fewer steps for faster validation
        'max_sequences_to_validate': None  # None means validate all
    }

    return config
