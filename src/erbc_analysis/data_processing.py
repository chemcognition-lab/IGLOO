import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


# --- Batch-wise padding and normalization ---
def batch_pad_and_normalize(batch):
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
