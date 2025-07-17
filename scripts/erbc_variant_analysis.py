#!/usr/bin/env python3
"""
ERBC Variant Analysis Pipeline

This script performs the following steps:
 1. Load ERBC variant dataset and identify WT baseline.
 2. Generate ESM embeddings for sequences.
 3. Train a simple linear regression model (mean-pooled ESM) to predict production (log or raw), tracking training/validation loss.
 4. Compute residue-level Integrated Gradients (IG) attributions vs WT.
 5. Save IG attributions and bar-plot.
 6. Dynamically determine a hotspot threshold from the IG distribution.
 7. Classify amino-acid swaps by physicochemical changes and test enrichment (chi² only).
 8. Correlate side-chain property deltas (volume, hydrophobicity) with IG magnitude via Pearson correlation.
 9. Generate a PyMOL selection script for hotspot residues.
10. (Optional) Outline MSA + WebLogo generation for conservation analysis.
11. (Optional) Compute alternative attributions (Gradient × Input) for robustness.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, layers
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, pearsonr
import torch
import esm

# ------------- User parameters -------------
ESM_MODEL = 'esm2_t6_8M_UR50D'
EMB_LAYER = 6
STEPS_IG = 50
HIDDEN_BATCH = 16
EPOCHS = 50
HOTSPOT_PERCENTILE = 90  # percentile for hotspot selection

# Choose production column: if True, use log_production and suffix '_log'; else use 'Production_ug/mL' and no suffix
USE_LOG = False

# ------------- Paths and suffixes -------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Data_Donya_ERBC_Converted.csv')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# file suffix
suffix = '_log' if USE_LOG else ''
DF_ATTR_CSV = os.path.join(RESULTS_DIR, f'residue_ig_attributions{suffix}.csv')
FIG_IG = os.path.join(RESULTS_DIR, f'residue_ig_vs_WT{suffix}.png')
FIG_LOSS = os.path.join(RESULTS_DIR, f'training_loss{suffix}.png')
PYMOL_PML = os.path.join(RESULTS_DIR, f'hotspots{suffix}.pml')

# ------------- AA property lookup -------------
AA_PROPS = {
    'A': {'hydro':  1.8, 'vol':  88.6, 'charge': 'neutral'},
    'C': {'hydro':  2.5, 'vol': 108.5, 'charge': 'neutral'},
    'D': {'hydro': -3.5, 'vol': 111.1, 'charge': 'negative'},
    'E': {'hydro': -3.5, 'vol': 138.4, 'charge': 'negative'},
    'F': {'hydro':  2.8, 'vol': 189.9, 'charge': 'neutral'},
    'G': {'hydro': -0.4, 'vol':  60.1, 'charge': 'neutral'},
    'H': {'hydro': -3.2, 'vol': 153.2, 'charge': 'positive'},
    'I': {'hydro':  4.5, 'vol': 166.7, 'charge': 'neutral'},
    'K': {'hydro': -3.9, 'vol': 168.6, 'charge': 'positive'},
    'L': {'hydro':  3.8, 'vol': 166.7, 'charge': 'neutral'},
    'M': {'hydro':  1.9, 'vol': 162.9, 'charge': 'neutral'},
    'N': {'hydro': -3.5, 'vol': 114.1, 'charge': 'neutral'},
    'P': {'hydro': -1.6, 'vol': 112.7, 'charge': 'neutral'},
    'Q': {'hydro': -3.5, 'vol': 143.8, 'charge': 'neutral'},
    'R': {'hydro': -4.5, 'vol': 173.4, 'charge': 'positive'},
    'S': {'hydro': -0.8, 'vol':  89.0, 'charge': 'neutral'},
    'T': {'hydro': -0.7, 'vol': 116.1, 'charge': 'neutral'},
    'V': {'hydro':  4.2, 'vol': 140.0, 'charge': 'neutral'},
    'W': {'hydro': -0.9, 'vol': 227.8, 'charge': 'neutral'},
    'Y': {'hydro': -1.3, 'vol': 193.6, 'charge': 'neutral'},
}

def categorize_swap(wt, var):
    if wt not in AA_PROPS or var not in AA_PROPS:
        return 'other'
    p1, p2 = AA_PROPS[wt], AA_PROPS[var]
    if p1['charge'] != p2['charge']:
        return f"charge:{p1['charge']}->{p2['charge']}"
    hydro_pos = {aa for aa, v in AA_PROPS.items() if v['hydro'] > 0}
    polar = set(AA_PROPS.keys()) - hydro_pos
    if wt in hydro_pos and var in polar:
        return 'hydrophobic->polar'
    if wt in polar and var in hydro_pos:
        return 'polar->hydrophobic'
    if abs(p2['vol'] - p1['vol']) > 50:
        return 'large_volume_change'
    return 'other'


def integrated_gradients(x, model, baseline, steps=STEPS_IG):
    x = tf.cast(x, tf.float32)
    b = tf.cast(baseline, tf.float32)
    scaled = [b + (i/steps)*(x - b) for i in range(steps+1)]
    grads = []
    for xi in scaled:
        with tf.GradientTape() as tape:
            tape.watch(xi)
            pred = tf.squeeze(model(xi), axis=-1)
        grads.append(tape.gradient(pred, xi))
    avg_grads = tf.reduce_mean(tf.stack(grads[1:], axis=0), axis=0)
    return (x - b) * avg_grads

if __name__ == '__main__':
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    seqs = df['SEQ'].tolist()
    # Choose target column based on flag
    if USE_LOG:
        y = df['log_production'].values.astype(np.float32)
    else:
        y = df['Production_ug/mL'].values.astype(np.float32)
    wt_seq = df.loc[df['Variant'] == 'WT', 'SEQ'].iloc[0]

    # 2. Load ESM
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    # 3. Embed sequences
    batch = [(str(i), s) for i, s in enumerate(seqs)]
    _, _labels, tokens = batch_converter(batch)
    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[EMB_LAYER], return_contacts=False)
    reps = out['representations'][EMB_LAYER].numpy()
    N, Lp2, D = reps.shape
    L = Lp2 - 2
    X_all = reps[:, 1:L+1, :]

    # 4. WT baseline
    _, _, t_wt = batch_converter([('WT', wt_seq)])
    with torch.no_grad():
        o_wt = esm_model(t_wt, repr_layers=[EMB_LAYER], return_contacts=False)
    X_wt = o_wt['representations'][EMB_LAYER][:, 1:L+1, :]

    # 5. Train/test split
    idx = np.arange(N)
    X_tr, X_te, y_tr, y_te, ix_tr, ix_te = train_test_split(
        X_all, y, idx, test_size=0.2, random_state=42
    )

    # 6. Build & train model
    inp = layers.Input(shape=(L, D))
    x = layers.GlobalAveragePooling1D()(inp)
    out_layer = layers.Dense(1)(x)
    model = Model(inp, out_layer)
    model.compile(optimizer='adam', loss='mse')
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=EPOCHS,
        batch_size=HIDDEN_BATCH
    )

    # 6b. Save training history
    plt.figure()
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig(FIG_LOSS, bbox_inches='tight', dpi=200)
    plt.close()

    # 7. IG attributions for first test sample relative to WT baseline
    ig = integrated_gradients(X_te[0:1], model, X_wt)
    scores = tf.reduce_sum(ig[0], axis=-1).numpy()
    pos = np.arange(1, L+1)
    df_attr = pd.DataFrame({
        'Position': pos,
        'Residue': list(seqs[ix_te[0]]),
        f'IG_Attribution_vs_WT{suffix}': scores
    })
    df_attr.to_csv(DF_ATTR_CSV, index=False)
    plt.figure()
    plt.bar(pos, scores)
    plt.xlabel('Residue')
    plt.ylabel('IG vs WT')
    plt.title('IG vs WT baseline')
    plt.savefig(FIG_IG, bbox_inches='tight', dpi=200)
    plt.close()

    # 8. Threshold determination and hotspots
    ig_abs = np.abs(df_attr[f'IG_Attribution_vs_WT{suffix}'])
    thr = np.percentile(ig_abs, HOTSPOT_PERCENTILE)
    print(f"Threshold set at {HOTSPOT_PERCENTILE}th percentile: {thr:.2e}")
    hotspots = df_attr[ig_abs > thr]

    # 9. Classify swaps
    swaps = []
    for _, r in hotspots.iterrows():
        wt = wt_seq[int(r.Position)-1]
        var = seqs[ix_te[0]][int(r.Position)-1]
        swaps.append({
            'Position': r.Position,
            'WT': wt,
            'Var': var,
            'IG': r[f'IG_Attribution_vs_WT{suffix}'],
            'Category': categorize_swap(wt, var)
        })
    df_swaps = pd.DataFrame(swaps)

    # 10. Enrichment chi2
    if not df_swaps.empty:
        ct = pd.crosstab(df_swaps['Category'], df_swaps['IG'] > 0)
        p_chi = chi2_contingency(ct.values)[1]
        print('Chi2 p-value:', p_chi)

    # 11. Property correlation
    deltas = []
    for r in swaps:
        p1 = AA_PROPS[r['WT']]
        p2 = AA_PROPS[r['Var']]
        deltas.append({
            'dvol': p2['vol'] - p1['vol'],
            'dhyd': p2['hydro'] - p1['hydro'],
            'IG': r['IG']
        })
    df_del = pd.DataFrame(deltas)
    for c in ['dvol', 'dhyd']:
        if df_del[c].nunique() > 1:
            r_val, p_val = pearsonr(df_del[c], df_del['IG'])
            print(f"Corr {c} vs IG: r={r_val:.3f}, p={p_val:.3g}")

    # 12. PyMOL script
    if not df_swaps.empty:
        resnos = ','.join(map(str, df_swaps['Position'].tolist()))
        with open(PYMOL_PML, 'w') as f:
            f.write(f"select hotspots, resi {resnos}\ncolor red, hotspots\n")
        print(f"Wrote PyMOL script: {PYMOL_PML}")

    print('Done.')
