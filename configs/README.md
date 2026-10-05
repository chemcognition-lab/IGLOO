# Integrated Gradients Analysis - Configuration Guide

This directory contains configuration files for running integrated gradients analysis using the unified `run_integrated_gradients.py` script.

## Quick Start

```bash
# Synthetic landscape with known ground truth
python scripts/pipeline/run_integrated_gradients.py configs/synthetic/GB1_additive_config.yaml

# Real data: FLIP2 alpha-amylase split
python scripts/pipeline/run_integrated_gradients.py configs/FLIP2/alpha_amylase__mutation__by_mutation.yaml

# Real data: ProteinGym beta-lactamase DMS assay
python scripts/pipeline/run_integrated_gradients.py configs/ProteinGym/BLAT_ECOLX_Firnberg_2014_config.yaml

# Validate config without running analysis
python scripts/pipeline/run_integrated_gradients.py configs/synthetic/GB1_additive_config.yaml --validate-only

# Override any value from the command line (dot paths into the YAML)
python scripts/pipeline/run_integrated_gradients.py configs/synthetic/GB1_additive_config.yaml \
    model.pooling_method=esm_perres_mean training.seed=1 results_dir=results/demo
```

## Configuration File Structure

### Data Section
```yaml
data:
  name: "alpha_amylase__mutation__by_mutation"   # Dataset name (used for file naming)
  path: "data/FLIP2_processed/alpha_amylase__mutation__by_mutation_combined.csv"
  target_column: "quantitative_function"  # Column to predict
  normalization: "z-score"               # "z-score" or "none"
  wildtype:                              # How to identify wild-type sequence
    method: "variant_column"             # "variant_column", "position", or "sequence"
    variant_column: "mutant"             # Column containing variant names
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
results_dir: "results/my_dataset_ig"  # Custom results directory
force_retrain: false           # Force model retraining even if model exists
```

## Example Configurations

### Synthetic landscapes (`synthetic/`)
- GB1, T4Lysozyme and p53 × additive / epistatic / nonlinear (plus `GB1_*_deep` variants)
- WT given as a sequence; no normalization (fitness is already WT-normalised)
- Ground-truth position weights in `data/synthetic/*_ground_truth_weights.json`

### FLIP2 alpha-amylase (`FLIP2/`)
- WT identified from the `mutant` column; z-score normalization
- Uses the pre-defined train/test split (`training.split_method: column`)

### ProteinGym DMS assays (`ProteinGym/`)
- One config per assay, generated by `scripts/data/generate_proteingym_configs.py`

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
# No need to copy configs: use CLI overrides with separate results directories
CFG=configs/FLIP2/alpha_amylase__mutation__by_mutation.yaml
for path in linear slerp nearest_neighbor; do
  python scripts/pipeline/run_integrated_gradients.py $CFG \
      integrated_gradients.interpolation_path=$path results_dir=results/compare/$path
done
```

### Troubleshooting

1. **Config validation**: Use `--validate-only` to check your config
2. **Model retraining**: Set `force_retrain: true` to retrain the model
3. **Memory issues**: Reduce `batch_size` in model and/or integrated_gradients sections
4. **File paths**: All paths in config are relative to the project root

## File Format Support

Both YAML (`.yaml`, `.yml`) and JSON (`.json`) formats are supported. YAML is recommended for readability and comments.
