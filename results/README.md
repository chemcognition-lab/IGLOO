# Results Directory

This directory contains analysis results organized by protein/dataset:

## Structure
```
results/
├── betalactamase/          # Beta-lactamase analysis results
│   ├── results_summary.csv
│   ├── train_predictions.csv
│   ├── test_predictions.csv
│   ├── pred_vs_actual_comparison.png
│   ├── target_distributions.png
│   └── betalactamase_model.pth
├── PafA/                   # PafA protein analysis results
│   ├── mutation analysis/
│   ├── sequence analysis/
│   └── model outputs/
└── [other proteins]/       # Future protein analyses

```

## File Types

### Model Performance
- `results_summary.csv`: Overall performance metrics (R², Pearson r, Spearman ρ)
- `*_predictions.csv`: Detailed predictions vs actual values
- `pred_vs_actual_comparison.png`: Scatter plots of model performance

### Analysis Results
- `mutation_landscape.png`: Heatmaps of mutation effects
- `sequence_analysis_*/`: Per-variant sequence-level attribution analysis
- `*_model.pth`: Trained PyTorch models with normalization parameters

### Data Distributions
- `target_distributions.png`: Distribution plots of target variables
- Training vs test set comparisons

## Usage
Each subfolder contains complete analysis results for a specific protein dataset, making it easy to compare across different proteins and experiments.
