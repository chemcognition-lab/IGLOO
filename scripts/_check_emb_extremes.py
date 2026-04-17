import sys
sys.path.insert(0, '/homes/bargho16/IGLOO')
import numpy as np
import yaml

from src.igloo.data_processing import load_and_prepare_data, prepare_wildtype_sequence
from src.igloo.embedding import get_embeddings

with open('/homes/bargho16/IGLOO/configs/synthetic/GB1_additive_config.yaml') as f:
    config = yaml.safe_load(f)

data_path = config['data']['path']
df, sequences, y_raw, y, *_ = load_and_prepare_data(config, data_path)
wt_seq, _ = prepare_wildtype_sequence(df, config)

all_embs = get_embeddings(sequences, layer=33, batch_size=16,
                          dataset_name='GB1_additive', subdir='synthetic')
pooled = all_embs.mean(axis=1)  # (N, D)

centroid = pooled.mean(axis=0)
dists = np.linalg.norm(pooled - centroid, axis=1)
order = np.argsort(dists)[::-1]

emb_extreme_idx = [i for i in order if sequences[i] != wt_seq][:20]
rng = np.random.default_rng(0)
random_idx = [i for i in rng.choice(len(sequences), 50, replace=False) if sequences[i] != wt_seq][:20]

print("Distance from centroid (960D L2):")
print(f"  Embedding extremes: {dists[emb_extreme_idx].mean():.4f} ± {dists[emb_extreme_idx].std():.4f}"
      f"  (range {dists[emb_extreme_idx].min():.4f}–{dists[emb_extreme_idx].max():.4f})")
print(f"  Random sequences:   {dists[random_idx].mean():.4f} ± {dists[random_idx].std():.4f}"
      f"  (range {dists[random_idx].min():.4f}–{dists[random_idx].max():.4f})")
print(f"\nAll distances: median={np.median(dists):.4f}, mean={dists.mean():.4f}, max={dists.max():.4f}")

from sklearn.decomposition import PCA
pca10 = PCA(n_components=10).fit(pooled)
cumvar = pca10.explained_variance_ratio_.cumsum() * 100
print("\nCumulative variance explained:")
for i, v in enumerate(cumvar):
    print(f"  PC1–{i+1}: {v:.1f}%")

pca2 = PCA(n_components=2)
proj2 = pca2.fit_transform(pooled)
pca_dists_emb = np.linalg.norm(proj2[emb_extreme_idx] - proj2.mean(axis=0), axis=1)
pca_dists_rnd = np.linalg.norm(proj2[random_idx]       - proj2.mean(axis=0), axis=1)
print(f"\nDistance from centroid in 2D PCA space:")
print(f"  Embedding extremes: {pca_dists_emb.mean():.4f} ± {pca_dists_emb.std():.4f}")
print(f"  Random sequences:   {pca_dists_rnd.mean():.4f} ± {pca_dists_rnd.std():.4f}")
print("\nConclusion: if 2D distances are similar but 960D distances differ,")
print("the extremes live in high-dim space that PCA doesn't capture.")
