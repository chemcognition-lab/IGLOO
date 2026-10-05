# Integrated Gradients Analysis - Configuration Guide

This directory contains configuration files for running integrated gradients analysis using the unified `run_integrated_gradients.py` script.

## Quick Start

```bash
# Run ERBC analysis with embedding-level IG and magnitude attribution
python scripts/pipeline/run_integrated_gradients.py configs/erbc_config.yaml

# Run PafA analysis with token-level IG and signed attribution
python scripts/pipeline/run_integrated_gradients.py configs/pafa_config.yaml

# Run beta-lactamase analysis
python scripts/pipeline/run_integrated_gradients.py configs/betalactamase_config.yaml

# Validate config without running analysis
python scripts/pipeline/run_integrated_gradients.py configs/erbc_config.yaml --validate-only
```

## Configuration File Structure

### Data Section
```yaml
data:
  name: "ERBC"                           # Dataset name (used for file naming)
  path: "data/Data_Donya_ERBC_Converted.csv"  # Path to CSV file
  target_column: "log_production"         # Column to predict
  normalization: "z-score"               # "z-score" or "none"
  wildtype:                              # How to identify wild-type sequence
    method: "variant_column"             # "variant_column", "position", or "sequence"
    variant_column: "Variant"            # Column containing variant names
    wildtype_value: "WT"                 # Value identifying wild-type
```

#### Wild-type Identification Methods

1. **variant_column**: Find WT by looking for a specific value in a column
   ```yaml
   wildtype:
     method: "variant_column"
     variant_column: "Variant"
     wildtype_value: "WT"
   ```

2. **position**: Use sequence at a specific position in the dataset
   ```yaml
   wildtype:
     method: "position"
     position: "last"  # "first", "last", or integer index
   ```

3. **sequence**: Provide the WT sequence directly
   ```yaml
   wildtype:
     method: "sequence"
     sequence: "MSKGEELFT..."
   ```

### Model Section
```yaml
model:
  esm_layer: 33                    # ESM layer to extract features from
  pooling_method: "esm_attention"  # "linear" or "esm_attention"
  hidden_dim1: 1024               # First hidden layer size
  hidden_dim2: 512                # Second hidden layer size
  hidden_dim3: 255                # Third hidden layer size
  hidden_dim4: 128                # Fourth hidden layer size
  dropout_rate: 0.2               # Dropout rate
  learning_rate: 0.001            # Learning rate
  batch_size: 256                 # Batch size
  epochs: 100                     # Number of training epochs
  loss_type: "huber"              # "mse", "huber", or "l1"
  final_activation: null          # Final activation function (usually null)
```

### Integrated Gradients Section
```yaml
integrated_gradients:
  method: "embedding"              # "embedding" or "token"
  attribution_method: "magnitude"  # "magnitude" or "signed"
  steps: 50                       # Number of integration steps
  batch_size: 10                  # Batch size for gradient computation
```

#### IG Method Comparison

- **embedding**: Uses embedding-level integrated gradients (faster, good for overall importance)
- **token**: Uses token-level integrated gradients (slower, more direct attribution)

#### Attribution Method Comparison

- **magnitude**: L2 norm of gradients (always positive, shows importance magnitude)
- **signed**: Dot product with embedding differences (can be positive/negative, shows directional effects)

### Analysis Section
```yaml
analysis:
  sequence_selection:
    include_extremes: true    # Include lowest and highest activity sequences
    include_median: true      # Include median activity sequence
    include_top_n: 2         # Include top N sequences (beyond highest)
    include_bottom_n: 2      # Include bottom N sequences (beyond lowest)
    custom_sequences:        # Custom sequences to analyze
      - name: "Custom_1"
        sequence: "MSKGEELFT..."
      - name: "Custom_2"
        index: 42
```

### Optional Settings
```yaml
results_dir: "results/ERBC_ig"  # Custom results directory
force_retrain: false           # Force model retraining even if model exists
```

## Example Configurations

### ERBC Dataset
- Uses variant column to identify WT
- Z-score normalization
- Embedding-level IG with magnitude attribution
- See: `erbc_config.yaml`

### PafA Dataset
- Uses last sequence as WT
- Z-score normalization
- Token-level IG with signed attribution (to see directional effects)
- See: `pafa_config.yaml`

### Beta-lactamase Dataset
- Uses provided WT sequence
- No normalization
- Embedding-level IG with magnitude attribution
- See: `betalactamase_config.yaml`

## Output Files

The script generates the same comprehensive output as the individual scripts:

- `{dataset}_predicted_vs_actual.png` - Model performance plot
- `{dataset}_integrated_gradients_analysis.png` - IG analysis plots
- `{dataset}_integrated_gradients_heatmap.png` - Gradient heatmap
- `{dataset}_variants_vs_WT_gradients.png` - Comparison plots
- `{dataset}_integrated_gradients_results.csv` - Detailed results
- `{dataset}_position_importance_summary.csv` - Position importance ranking
- `{dataset}_predicted_vs_actual_summary.csv` - Prediction accuracy summary
- `{dataset}_integrated_gradients_summary.txt` - Human-readable summary
- `{dataset}_training_model.pth` - Trained model
- `{dataset}_ig_analysis.pth` - Analysis results and parameters

## Advanced Usage

### Creating Custom Configurations

1. Copy an existing config file
2. Modify the data section for your dataset
3. Adjust model hyperparameters if needed
4. Choose IG method and attribution method
5. Customize sequence selection

### Comparing Methods

To compare different methods on the same dataset:

```bash
# Create configs with different methods
cp configs/erbc_config.yaml configs/erbc_embedding_magnitude.yaml
cp configs/erbc_config.yaml configs/erbc_embedding_signed.yaml
cp configs/erbc_config.yaml configs/erbc_token_magnitude.yaml
cp configs/erbc_config.yaml configs/erbc_token_signed.yaml

# Edit each config to use different method/attribution
# Then run each one with different results directories
```

### Troubleshooting

1. **Config validation**: Use `--validate-only` to check your config
2. **Model retraining**: Set `force_retrain: true` to retrain the model
3. **Memory issues**: Reduce `batch_size` in model and/or integrated_gradients sections
4. **File paths**: All paths in config are relative to the project root

## File Format Support

Both YAML (`.yaml`, `.yml`) and JSON (`.json`) formats are supported. YAML is recommended for readability and comments.
