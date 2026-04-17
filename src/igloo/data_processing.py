import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config_utils import get_wildtype_sequence

# Global normalization parameters - will be set by compute_global_normalization_params
_global_scaler_mean = None
_global_scaler_scale = None


def print_data_info(df, col, sequences, y_raw):
    """Print dataset information and statistics."""
    print(f"Total dataset size: {len(sequences)} sequences")
    print("Applying z-score normalization to target...")
    y_log_mean = np.mean(y_raw)
    y_log_std = np.std(y_raw)
    y = (y_raw - y_log_mean) / y_log_std
    print(f"Target mean: {y_log_mean:.4f}, std: {y_log_std:.4f}")
    print(f"Normalized target range: {np.min(y):.4f} to {np.max(y):.4f}")
    return y, y_log_mean, y_log_std

def set_global_normalization_params(mean, scale):
    """Set global normalization parameters."""
    global _global_scaler_mean, _global_scaler_scale
    _global_scaler_mean = mean
    _global_scaler_scale = scale

def compute_global_normalization_params(all_embeddings):
    """
    Compute global normalization parameters from all training embeddings.
    This ensures consistent normalization across all batches.

    Args:
        all_embeddings: List of embedding arrays or single array

    Returns:
        dict containing 'mean' and 'scale' for global normalization
    """
    global _global_scaler_mean, _global_scaler_scale

    if isinstance(all_embeddings, list):
        # Handle list of embedding arrays
        all_flat = []
        for emb in all_embeddings:
            if len(emb.shape) == 2:  # Single sequence (L, D)
                all_flat.append(emb.flatten())
            elif len(emb.shape) == 3:  # Batch of sequences (N, L, D)
                for seq in emb:
                    all_flat.append(seq.flatten())
        combined_flat = np.concatenate(all_flat)
    else:
        # Handle single array
        if len(all_embeddings.shape) == 3:  # (N, L, D)
            combined_flat = all_embeddings.reshape(-1, all_embeddings.shape[-1]).flatten()
        elif len(all_embeddings.shape) == 2:  # (L, D)
            combined_flat = all_embeddings.flatten()
        else:
            combined_flat = all_embeddings.flatten()

    # Compute global mean and std
    _global_scaler_mean = np.mean(combined_flat)
    _global_scaler_scale = np.std(combined_flat)

    # Avoid division by zero
    if _global_scaler_scale == 0:
        _global_scaler_scale = 1.0

    return {'mean': _global_scaler_mean, 'scale': _global_scaler_scale}

def batch_pad_and_normalize(batch, use_global_normalization=True):
    """
    Pad sequences in batch and normalize embeddings.

    Args:
        batch: List of (embedding, target) tuples
        use_global_normalization: If True, use global normalization parameters.
                                 If False, use batch-specific normalization (old behavior).
    """
    global _global_scaler_mean, _global_scaler_scale

    embeddings, targets = zip(*batch)
    batch_size = len(embeddings)
    embedding_dim = embeddings[0].shape[1]
    max_len = max(seq.shape[0] for seq in embeddings)
    padded = np.full((batch_size, max_len, embedding_dim), 0.0)
    lengths = []
    for i, seq in enumerate(embeddings):
        seq_len = seq.shape[0]
        padded[i, :seq_len, :] = seq
        lengths.append(seq_len)

    flat = padded.reshape(batch_size, -1)

    if use_global_normalization and _global_scaler_mean is not None and _global_scaler_scale is not None:
        # Use global normalization parameters
        flat_norm = (flat - _global_scaler_mean) / _global_scaler_scale
    else:
        # Use batch-specific normalization (old behavior)
        scaler = StandardScaler()
        flat_norm = scaler.fit_transform(flat)

    normed = flat_norm.reshape(batch_size, max_len, embedding_dim)
    return (
        torch.tensor(normed, dtype=torch.float32),
        torch.tensor(lengths, dtype=torch.int64),
        torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    )

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]

def load_and_prepare_data(config, data_path):
    """Load and prepare the dataset"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Get target column
    target_col = config['data']['target_column']

    # Replace NaN or Inf values with 0 instead of removing rows
    df[target_col] = df[target_col].fillna(0.0)
    df[target_col] = df[target_col].replace([np.inf, -np.inf], 0.0)

    # Set wildtype sequence target value to 1.0 (only for beta-lactamase dataset)
    dataset_name = config['data'].get('name', '').lower()
    if dataset_name == 'betalactamase':
        wt_config = config['data']['wildtype']
        try:
            wt_seq, wt_idx = get_wildtype_sequence(df, wt_config)
            if wt_idx >= 0:  # Wildtype found in dataset
                original_value = df.loc[wt_idx, target_col]
                df.loc[wt_idx, target_col] = 1.0
                print(f"Set beta-lactamase wildtype sequence target from {original_value:.4f} to 1.0")
            else:
                print("Beta-lactamase wildtype sequence not found in dataset - no target adjustment made")
        except Exception as e:
            print(f"Warning: Could not set beta-lactamase wildtype target value: {e}")
    else:
        print(f"Dataset '{dataset_name}' - skipping wildtype normalization")

    sequences = df['SEQ'].astype(str).tolist()
    y_raw = df[target_col].values.astype(np.float32)

    print(f"Dataset size: {len(sequences)} sequences")
    print(f"Target range: {np.min(y_raw):.4f} to {np.max(y_raw):.4f}")

    # Handle normalization
    normalization = config['data'].get('normalization', 'z-score')
    if normalization == 'z-score':
        y, y_log_mean, y_log_std = print_data_info(
            df, target_col, sequences, y_raw
        )
    elif normalization == 'none' or normalization is None:
        print("Using raw target values without normalization...")
        y = y_raw
        y_log_mean = 0.0
        y_log_std = 1.0
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    return df, sequences, y_raw, y, y_log_mean, y_log_std


def prepare_wildtype_sequence(df, config):
    """Get the wild-type sequence based on config"""
    wt_config = config['data']['wildtype']
    wt_seq, wt_idx = get_wildtype_sequence(df, wt_config)

    # Print information about the WT sequence
    if wt_config['method'] == 'variant_column':
        variant_col = wt_config['variant_column']
        wt_value = wt_config['wildtype_value']
        wt_activity = df.loc[wt_idx, config['data']['target_column']]
        print(f"Found WT sequence via {variant_col}='{wt_value}'")
        print(f"WT {config['data']['target_column']}: {wt_activity:.4f}")
    elif wt_config['method'] == 'position':
        print(f"Using WT sequence from position: {wt_config['position']}")
    elif wt_config['method'] == 'sequence':
        print("Using provided WT sequence")

    print(f"WT sequence: {wt_seq[:50]}...")
    print(f"WT sequence length: {len(wt_seq)}")

    return wt_seq, wt_idx


def split_dataset(
    df: pd.DataFrame,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(trainval_idx, test_idx)`` integer index arrays into *df*.

    Auto-detects the dataset type:

    * If a ``variant_type`` column is present (synthetic datasets) the split is
      **stratified by mutation order** (single / double / triple) and the
      wildtype sequence is always kept in the train+val pool.
    * Otherwise a plain **random split** via scikit-learn is used.

    Parameters
    ----------
    df            : DataFrame (as returned by :func:`load_and_prepare_data`).
    test_fraction : Fraction of non-wildtype variants to hold out (default 0.20).
    seed          : Random seed for reproducibility.

    Returns
    -------
    trainval_idx, test_idx
        1-D integer arrays suitable for indexing *df* or the parallel
        ``sequences`` / ``y`` arrays.
    """
    if 'variant_type' in df.columns:
        # Synthetic / stratified path — delegate to split_train_test which
        # handles WT pinning and mutation-order stratification.
        from igloo.synthetic_data import split_train_test  # local import to avoid circular dep
        trainval_df, _ = split_train_test(df, test_fraction=test_fraction, seed=seed)
        trainval_seqs = set(trainval_df['SEQ'].tolist())
        seqs = df['SEQ'].tolist()
        trainval_idx = np.array(
            [i for i, s in enumerate(seqs) if s in trainval_seqs], dtype=int
        )
        test_idx = np.array(
            [i for i, s in enumerate(seqs) if s not in trainval_seqs], dtype=int
        )
    else:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df))
        trainval_idx, test_idx = train_test_split(
            idx, test_size=test_fraction, random_state=seed
        )

    return trainval_idx, test_idx
