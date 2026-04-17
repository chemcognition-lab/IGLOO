# Integrated Gradients Analysis Configuration Summary

This document summarizes all the configuration files created for comprehensive integrated gradients analysis across all three datasets.

## Configuration Matrix

### PafA Dataset
- **pafa_config_embedding-magnitude.yaml** - Embedding-level IG with magnitude attribution
- **pafa_config_embedding-signed.yaml** - Embedding-level IG with signed attribution
- **pafa_config_token-magnitude.yaml** - Token-level IG with magnitude attribution
- **pafa_config_token-signed.yaml** - Token-level IG with signed attribution

### Beta-lactamase Dataset
- **betalactamase_config_embedding-magnitude.yaml** - Embedding-level IG with magnitude attribution
- **betalactamase_config_embedding-signed.yaml** - Embedding-level IG with signed attribution
- **betalactamase_config_token-magnitude.yaml** - Token-level IG with magnitude attribution
- **betalactamase_config_token-signed.yaml** - Token-level IG with signed attribution

### ERBC Dataset
- **erbc_config_embedding-magnitude.yaml** - Embedding-level IG with magnitude attribution
- **erbc_config_embedding-signed.yaml** - Embedding-level IG with signed attribution
- **erbc_config_token-magnitude.yaml** - Token-level IG with magnitude attribution
- **erbc_config_token-signed.yaml** - Token-level IG with signed attribution

## Key Differences Between Configurations

### Integration Method
- **embedding**: Standard integrated gradients on ESM embeddings (faster, sequence-level attribution)
- **token**: Token-level integrated gradients with normalization (slower, position-level attribution)

### Attribution Method
- **magnitude**: Absolute values of gradients (shows importance magnitude)
- **signed**: Raw signed gradients (shows beneficial vs detrimental effects)

### Dataset-Specific Settings

#### PafA
- Target: `kcat_cMUP`
- Normalization: `z-score`
- Wildtype: Position-based (last sequence)
- Results: `results/PafA_{method}_{attribution}`

#### Beta-lactamase
- Target: `quantitative_function`
- Normalization: `none`
- Wildtype: Full sequence provided
- Results: `results/betalactamase_{method}_{attribution}`

#### ERBC
- Target: `log_production`
- Normalization: `z-score`
- Wildtype: Variant column-based (WT label)
- Results: `results/ERBC_{method}_{attribution}`

## Model Hyperparameters (Optimized)

All configurations use improved hyperparameters to prevent model collapse:
- **hidden_dim1**: 512 (reduced from 1024)
- **hidden_dim2**: 256 (reduced from 512)
- **hidden_dim3**: 128 (reduced from 255)
- **hidden_dim4**: 64 (reduced from 128)
- **dropout_rate**: 0.1 (reduced from 0.2-0.5)
- **learning_rate**: 0.0001 (reduced)
- **batch_size**: 64 (reduced from 256)
- **epochs**: 1000 (increased)
- **loss_type**: "mse" (changed from huber)

## Sequence Selection

All configurations include enhanced sequence selection:
- **include_extremes**: true (lowest and highest activity)
- **include_median**: true (median activity sequence)
- **include_top_n**: 10 (additional high-activity sequences)
- **include_bottom_n**: 10 (additional low-activity sequences)
- **max_error_percent**: 10.0 (filter by prediction accuracy)

## Usage Examples

### Run Individual Analysis
```bash
# PafA with token-level signed gradients
python scripts/run_integrated_gradients_unified.py configs/pafa_config_token-signed.yaml

# Beta-lactamase with embedding magnitude
python scripts/run_integrated_gradients_unified.py configs/betalactamase_config_embedding-magnitude.yaml

# ERBC with token magnitude
python scripts/run_integrated_gradients_unified.py configs/erbc_config_token-magnitude.yaml
```

### Run Batch Analysis
```bash
# Run all configurations for all datasets
python run_all_ig_analyses.py

# Run only PafA analyses
python run_all_ig_analyses.py --dataset pafa

# Run only token-level analyses across all datasets
python run_all_ig_analyses.py --method token

# Run only signed attribution analyses
python run_all_ig_analyses.py --attribution signed

# Run specific combination
python run_all_ig_analyses.py --dataset betalactamase --method embedding --attribution signed
```

## Expected Outputs

Each analysis will generate:
1. **Model performance plots** - Predicted vs actual values
2. **Attribution heatmaps** - Gradient patterns across sequences
3. **Position importance rankings** - Most critical positions
4. **Summary CSV files** - Detailed numerical results
5. **Summary text reports** - Human-readable analysis summaries

### Additional Outputs for Signed Attribution
- **Beneficial/detrimental position rankings**
- **Amino acid type effect analysis**
- **Signed gradient scatter plots** (mutation effects style)
- **Top vs bottom sequence comparisons**

## Results Organization

Results will be saved in structured directories:
```
results/
├── PafA_embedding_magnitude/
├── PafA_embedding_signed/
├── PafA_token_magnitude/
├── PafA_token_signed/
├── betalactamase_embedding_magnitude/
├── betalactamase_embedding_signed/
├── betalactamase_token_magnitude/
├── betalactamase_token_signed/
├── ERBC_embedding_magnitude/
├── ERBC_embedding_signed/
├── ERBC_token_magnitude/
└── ERBC_token_signed/
```

## Comparative Analysis Opportunities

With these 12 configurations (4 per dataset), you can compare:
1. **Method effectiveness**: Token vs embedding-level IG
2. **Attribution insights**: Magnitude vs signed gradients
3. **Cross-dataset patterns**: Common important positions/features
4. **Model performance**: How different approaches affect prediction accuracy
5. **Biological relevance**: Which method identifies known functional sites

This comprehensive setup enables systematic exploration of integrated gradients across your protein datasets with different attribution strategies.
