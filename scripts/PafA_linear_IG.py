# erbc_linear_esm_residue_ig_baseline.py
# Train a linear model on ERBC variant data using ESM embeddings,
# compute residue-level IG attributions using the WT variant as baseline,
# save both the IG plot and attribution DataFrame,
# and produce a sequence alignment highlighting hotspots.

import os

import esm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, layers

# Determine script directory and set paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Paths
data_path = os.path.join(project_root, 'data', 'PafA_data_processed.csv')
results_dir = os.path.join(script_dir, 'results')
plot_path = os.path.join(results_dir, 'PafA_residue_ig_vs_WT.png')
df_path = os.path.join(results_dir, 'PafA_residue_ig_attributions.csv')
align_plot_path = os.path.join(results_dir, 'PafA_alignment_hotspots.png')

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# 1. Load dataset
print(f"Loading data from {data_path}")
df = pd.read_csv(data_path)
sequences = df['SEQ'].astype(str).tolist()
y = df['kcat_cMUP'].values.astype(np.float32)

# Identify WT sequence for baseline
wt_seq = df['SEQ'].iloc[-1]
print(f"Using WT sequence as baseline: {wt_seq}")

# 2. Tokenize & embed with ESM
print("Loading ESM model...")
esm_model_name = 'esm2_t6_8M_UR50D'
esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
esm_model.eval()
batch_converter = alphabet.get_batch_converter()

# Embed all variants
print("Embedding all sequences...")
data_all = [(str(i), seq) for i, seq in enumerate(sequences)]
labels_all, strs_all, tokens_all = batch_converter(data_all)
with torch.no_grad():
    results_all = esm_model(tokens_all, repr_layers=[6], return_contacts=False)

token_reps_all = results_all['representations'][6]  # (N, L+2, D)
N, Lp2, D = token_reps_all.shape
L = Lp2 - 2
all_reps = token_reps_all[:, 1:L+1, :].numpy()

# Embed WT for baseline
data_wt = [('WT', wt_seq)]
_, _, tokens_wt = batch_converter(data_wt)
with torch.no_grad():
    results_wt = esm_model(tokens_wt, repr_layers=[6], return_contacts=False)
wt_reps = results_wt['representations'][6][:, 1:L+1, :].numpy()  # (1, L, D)
baseline = tf.constant(wt_reps, dtype=tf.float32)

# 3. Train-test split, preserving original indices
print("Splitting data...")
orig_idx = np.arange(N)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    all_reps, y, orig_idx, test_size=0.2, random_state=42
)

# 4. Build TensorFlow model: (L, D) -> mean pool -> linear
print("Building model...")
input_tensor = layers.Input(shape=(L, D))
pooled = layers.GlobalAveragePooling1D()(input_tensor)
out = layers.Dense(1, activation='linear')(pooled)
model = Model(inputs=input_tensor, outputs=out)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# 5. Train
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16
)

# 6. Evaluate
print("Evaluating...")
mse, mae = model.evaluate(X_test, y_test)
print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")

# 7. Residue-level Integrated Gradients
def integrated_gradients_res(x, model, baseline, steps=50):
    x = tf.cast(x, tf.float32)
    baseline = tf.cast(baseline, tf.float32)
    scaled = [baseline + (i/steps)*(x - baseline) for i in range(steps+1)]
    grads = []
    for s in scaled:
        with tf.GradientTape() as tape:
            tape.watch(s)
            pred = tf.squeeze(model(s), axis=-1)
        grads.append(tape.gradient(pred, s))
    avg_grads = tf.reduce_mean(tf.stack(grads[1:], axis=0), axis=0)
    return (x - baseline) * avg_grads

# 8. Compute attributions for a sample variant
test_idx = 0  # choose sample index in test set
orig_index = idx_test[test_idx]
sample_seq = sequences[orig_index]
var_seq = sample_seq  # for alignment plot
sample_input = X_test[test_idx:test_idx+1]
print(f"Computing IG for variant index {orig_index} (sequence {sample_seq})...")
attr = integrated_gradients_res(sample_input, model, baseline)
residue_scores = tf.reduce_sum(attr[0], axis=-1).numpy()

# 9. Map back to positions and residues
df_attr = pd.DataFrame({
    'Position': np.arange(1, L+1),
    'Residue': list(sample_seq),
    'IG_Attribution': residue_scores
})
print(df_attr.head(10))

# 10. Save the attribution DataFrame
print(f"Saving attribution DataFrame to {df_path}...")
df_attr.to_csv(df_path, index=False)
print(f"Saved attribution DataFrame to {df_path}")

# 11. Plot and save IG bar chart
print(f"Saving IG plot to {plot_path}...")
plt.figure(figsize=(12,4))
plt.bar(df_attr['Position'], df_attr['IG_Attribution'])
plt.xlabel('Residue Position')
plt.ylabel('IG Attribution vs WT')
plt.title(f'Residue-Level IG for sample {orig_index} (vs WT)')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved IG plot to {plot_path}")

# 12. Sequence alignment with hotspots
# Identify your top 5 positive and top 5 negative hotspots
hot_pos = df_attr.nlargest(5, 'IG_Attribution')['Position'].tolist()
hot_neg = df_attr.nsmallest(5, 'IG_Attribution')['Position'].tolist()
hotspots = set(hot_pos + hot_neg)

print(f"Hotspot positions: {sorted(hotspots)}")

print(f"Saving alignment plot to {align_plot_path}...")
plt.figure(figsize=(15, 2))
for i, (w, v) in enumerate(zip(wt_seq, var_seq)):
    color = 'red' if (i+1) in hotspots else 'black'
    plt.text(i, 1,   w, ha='center', va='center', color=color, fontsize=8)
    plt.text(i, 0.5, v, ha='center', va='center', color=color, fontsize=8)

plt.yticks([1, 0.5], ['WT', 'Variant'])
plt.xticks([])
plt.xlim(-1, len(wt_seq))
plt.ylim(0, 1.2)
plt.title('Sequence Alignment with Hotspots Highlighted (in red)')
plt.tight_layout()
plt.savefig(align_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved alignment plot to {align_plot_path}")