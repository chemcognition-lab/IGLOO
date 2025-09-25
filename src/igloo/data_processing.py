import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Global normalization parameters - will be set by compute_global_normalization_params
_global_scaler_mean = None
_global_scaler_scale = None

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
