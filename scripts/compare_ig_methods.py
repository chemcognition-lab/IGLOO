#!/usr/bin/env python3
"""
Compare Integrated Gradients Methods: Embedding vs Gumbel-Softmax

This script demonstrates how to switch between the two integrated gradients approaches
and compare their results.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch

from src.igloo.sequence_ig import (
    analyze_sequence_with_integrated_gradients,
    create_combined_model,
    load_esm_model,
)
from src.igloo.utils import create_model

def compare_ig_methods(
    predictor_model,
    sequence: str,
    baseline_sequence: str,
    y_log_mean: float,
    y_log_std: float,
    esm_layer: int = 33,
    pooling_method: str = 'mean',
    device: torch.device = None
):
    """
    Compare both integrated gradients methods on the same sequence.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading ESM model...")
    esm_model, alphabet = load_esm_model('esm2_t33_650M_UR50D', device)
    
    results = {}
    
    # Test both methods
    methods = [
        ('embedding', {}),
        ('gumbel', {'gumbel_temperature': 1.0, 'hard_gumbel': False}),
        ('gumbel_hard', {'gumbel_temperature': 0.5, 'hard_gumbel': True}),
    ]
    
    for method_name, method_kwargs in methods:
        print(f"\n" + "="*60)
        print(f"TESTING {method_name.upper()} METHOD")
        print("="*60)
        
        # Create appropriate model
        method = 'gumbel' if 'gumbel' in method_name else 'embedding'
        combined_model = create_combined_model(
            esm_model=esm_model,
            predictor_model=predictor_model,
            esm_layer=esm_layer,
            pooling_method=pooling_method,
            method=method,
            **method_kwargs,
            device=device
        )
        
        # Time the analysis
        start_time = time.time()
        
        result = analyze_sequence_with_integrated_gradients(
            sequence=sequence,
            baseline_sequence=baseline_sequence,
            model=combined_model,
            alphabet=alphabet,
            device=device,
            y_log_mean=y_log_mean,
            y_log_std=y_log_std,
            steps=20,  # Fewer steps for comparison
            batch_size=5,
            method=method
        )
        
        end_time = time.time()
        
        results[method_name] = {
            'result': result,
            'time': end_time - start_time
        }
        
        print(f"Method: {method_name}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Predicted activity: {result['predicted_activity']:.4f}")
        print(f"Baseline activity: {result['baseline_activity']:.4f}")
        print(f"Effect vs baseline: {result['effect_vs_baseline']:.4f}")
        print(f"Gradient range: [{np.min(result['position_gradients']):.4f}, {np.max(result['position_gradients']):.4f}]")
        print(f"Gradient std: {np.std(result['position_gradients']):.4f}")
    
    # Compare results
    print(f"\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print("\nPredicted Activities:")
    for method in results:
        pred = results[method]['result']['predicted_activity']
        print(f"  {method:15}: {pred:.4f}")
    
    print("\nEffect vs Baseline:")
    for method in results:
        effect = results[method]['result']['effect_vs_baseline']
        print(f"  {method:15}: {effect:.4f}")
    
    print("\nComputation Time:")
    for method in results:
        time_taken = results[method]['time']
        print(f"  {method:15}: {time_taken:.2f} seconds")
    
    print("\nGradient Statistics:")
    for method in results:
        grads = results[method]['result']['position_gradients']
        print(f"  {method:15}: mean={np.mean(grads):.4f}, std={np.std(grads):.4f}")
    
    # Gradient correlation between methods
    print("\nGradient Correlations:")
    method_names = list(results.keys())
    for i, method1 in enumerate(method_names):
        for method2 in method_names[i+1:]:
            grads1 = results[method1]['result']['position_gradients']
            grads2 = results[method2]['result']['position_gradients']
            correlation = np.corrcoef(grads1, grads2)[0, 1]
            print(f"  {method1} vs {method2}: r = {correlation:.4f}")
    
    return results

if __name__ == "__main__":
    print("This is a utility script for comparing IG methods.")
    print("Import and use the compare_ig_methods function in your analysis.")
