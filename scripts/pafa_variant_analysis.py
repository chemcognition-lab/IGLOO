#!/usr/bin/env python3
"""
ERBC Variant Analysis Pipeline with 5‐Fold CV and Averaged IG

Steps:
 1. Load ERBC variants and choose WT baseline.
 2. Generate ESM embeddings for all sequences.
 3. Perform 5‐fold cross‐validation: train & evaluate a mean‐pooled linear model, report fold metrics.
 4. Compute Integrated Gradients (IG) for the first test sample in each fold, then average across folds.
 5. Save averaged IG attributions, bar‐plot, determine hotspots by percentile, classify swaps, run enrichment and correlation tests.
 6. Generate a PyMOL script highlighting hotspot residues.
"""
import os

import esm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from scipy.stats import chi2_contingency, pearsonr
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import Model, layers

# -------- User parameters --------
ESM_MODEL = 'esm2_t6_8M_UR50D'
EMB_LAYER = 6
STEPS_IG = 50
HIDDEN_BATCH = 16
EPOCHS = 50
HOTSPOT_PERCENTILE = 90  # percentile for hotspot selection
USE_LOG = False

# -------- Paths & suffix --------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATA_PATH     = os.path.join(PROJECT_ROOT, 'data', 'PafA_data_processed.csv')
RESULTS_DIR   = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
suffix        = '_log' if USE_LOG else ''
DF_ATTR_CSV   = os.path.join(RESULTS_DIR, f'residue_ig_attributions{suffix}.csv')
FIG_IG        = os.path.join(RESULTS_DIR, f'residue_ig_vs_WT{suffix}.png')
FIG_CV        = os.path.join(RESULTS_DIR, f'cv_metrics{suffix}.txt')
FIG_LOSS      = os.path.join(RESULTS_DIR, f'training_loss{suffix}.png')
PYMOL_PML     = os.path.join(RESULTS_DIR, f'hotspots{suffix}.pml')

# -------- AA properties & swap categorization --------
AA_PROPS = {
    'A':{'hydro':1.8,'vol':88.6,'charge':'neutral'}, 'C':{'hydro':2.5,'vol':108.5,'charge':'neutral'},
    'D':{'hydro':-3.5,'vol':111.1,'charge':'negative'}, 'E':{'hydro':-3.5,'vol':138.4,'charge':'negative'},
    'F':{'hydro':2.8,'vol':189.9,'charge':'neutral'}, 'G':{'hydro':-0.4,'vol':60.1,'charge':'neutral'},
    'H':{'hydro':-3.2,'vol':153.2,'charge':'positive'}, 'I':{'hydro':4.5,'vol':166.7,'charge':'neutral'},
    'K':{'hydro':-3.9,'vol':168.6,'charge':'positive'}, 'L':{'hydro':3.8,'vol':166.7,'charge':'neutral'},
    'M':{'hydro':1.9,'vol':162.9,'charge':'neutral'}, 'N':{'hydro':-3.5,'vol':114.1,'charge':'neutral'},
    'P':{'hydro':-1.6,'vol':112.7,'charge':'neutral'}, 'Q':{'hydro':-3.5,'vol':143.8,'charge':'neutral'},
    'R':{'hydro':-4.5,'vol':173.4,'charge':'positive'}, 'S':{'hydro':-0.8,'vol':89.0,'charge':'neutral'},
    'T':{'hydro':-0.7,'vol':116.1,'charge':'neutral'}, 'V':{'hydro':4.2,'vol':140.0,'charge':'neutral'},
    'W':{'hydro':-0.9,'vol':227.8,'charge':'neutral'}, 'Y':{'hydro':-1.3,'vol':193.6,'charge':'neutral'} }

def categorize_swap(wt, var):
    if wt not in AA_PROPS or var not in AA_PROPS:
        return 'other'
    p1, p2 = AA_PROPS[wt], AA_PROPS[var]
    if p1['charge'] != p2['charge']:
        return f"charge:{p1['charge']}->{p2['charge']}"
    hydro_pos = {aa for aa,v in AA_PROPS.items() if v['hydro'] > 0}
    polar     = set(AA_PROPS) - hydro_pos
    if wt in hydro_pos and var in polar:
        return 'hydrophobic->polar'
    if wt in polar and var in hydro_pos:
        return 'polar->hydrophobic'
    if abs(p2['vol'] - p1['vol']) > 50:
        return 'large_volume_change'
    return 'other'

def integrated_gradients(x, model, baseline, steps=STEPS_IG):
    # PyTorch version of integrated gradients
    x = torch.tensor(x, dtype=torch.float32)
    b = torch.tensor(baseline, dtype=torch.float32)
    scaled = [b + (i/steps)*(x-b) for i in range(steps+1)]
    grads = []
    for xi in scaled:
        xi = xi.clone().detach().requires_grad_(True)
        pred = model(xi)
        pred = pred.squeeze(-1)
        grad = torch.autograd.grad(pred, xi, grad_outputs=torch.ones_like(pred), retain_graph=True)[0]
        grads.append(grad)
    avg = torch.mean(torch.stack(grads[1:], dim=0), dim=0)
    return (x - b) * avg

if __name__ == '__main__':
    
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    col = 'log_production' if USE_LOG else 'kcat_cMUP'
    df = df[df[col].notna()]  # Filter out rows where target is NaN
    seqs = df['SEQ'].tolist()
    y = df[col].astype(np.float32).values
    wt_seq = df['SEQ'].iloc[-1]

    # 2. Embed with ESM
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()
    # Embed sequences in batches to avoid memory errors
    X_all_list = []
    batch_size = 16  # Adjust as needed for your system
    for i in range(0, len(seqs), batch_size):
        batch = [(str(j), seqs[j]) for j in range(i, min(i+batch_size, len(seqs)))]
        _, _, tokens = batch_converter(batch)
        with torch.no_grad():
            out = esm_model(tokens, repr_layers=[EMB_LAYER], return_contacts=False)
        reps = out['representations'][EMB_LAYER].numpy()
        Lp2 = reps.shape[1]
        X_all_list.append(reps[:, 1:Lp2-1, :])
    X_all = np.concatenate(X_all_list, axis=0)
    N, L, D = X_all.shape

    # Debug: Check for NaNs in input data after creation
    print('NaNs in X_all:', np.isnan(X_all).sum())
    print('NaNs in y:', np.isnan(y).sum())
    # Debug: Check for NaNs in input data
    print('Checking for NaNs in input data...')
    print('NaNs in X_all:', np.isnan(X_all).sum())
    print('NaNs in y:', np.isnan(y).sum())

    # WT embedding baseline
    _,_,t_wt = batch_converter([('WT', wt_seq)])
    with torch.no_grad():
        o_wt = esm_model(t_wt, repr_layers=[EMB_LAYER], return_contacts=False)
    X_wt = o_wt['representations'][EMB_LAYER][:,1:L+1,:]

    # 3. 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mses, maes = [], []
    fold_models, fold_tests = [], []
    import torch.nn as nn
    import torch.optim as optim

    class MeanPooledLinear(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)
        def forward(self, x):
            x = x.mean(dim=1)  # mean pooling over sequence length
            return self.fc(x)

    train_losses = []
    val_losses = []
    for fold, (tr, te) in enumerate(kf.split(X_all), 1):
        X_tr = torch.tensor(X_all[tr], dtype=torch.float32)
        y_tr = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)

        X_te = torch.tensor(X_all[te], dtype=torch.float32)
        y_te = torch.tensor(y[te], dtype=torch.float32).unsqueeze(1)

        model = MeanPooledLinear(D)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.HuberLoss()

        fold_train_losses = []
        fold_val_losses = []
        # Training
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            output = model(X_tr)
            loss = criterion(output, y_tr)
            loss.backward()
            optimizer.step()
            fold_train_losses.append(loss.item())

            # Validation loss for this epoch
            model.eval()
            with torch.no_grad():
                val_output = model(X_te)
                val_loss = criterion(val_output, y_te)
                fold_val_losses.append(val_loss.item())
                # Debug: Check for NaNs in validation output
                print(f'Fold {fold}: NaNs in val_output:', torch.isnan(val_output).sum().item())
                print(f'Fold {fold}: NaNs in val_loss:', torch.isnan(val_loss).sum().item())

        train_losses.append(fold_train_losses)
        val_losses.append(fold_val_losses)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            pred = model(X_te)
            mse = nn.MSELoss()(pred, y_te).item()
            mae = nn.L1Loss()(pred, y_te).item()
            # Debug: Check for NaNs in final predictions
            print(f'Fold {fold}: NaNs in pred:', torch.isnan(pred).sum().item())
            print(f'Fold {fold}: mse={mse}, is_nan={np.isnan(mse)}')
            print(f'Fold {fold}: mae={mae}, is_nan={np.isnan(mae)}')
        print(f"Fold {fold}: MSE={mse:.4f}, MAE={mae:.4f}")
        mses.append(mse)
        maes.append(mae)
        fold_models.append(model)
        fold_tests.append(te)
    print(f"5-fold CV MSE: {np.mean(mses):.4f} ± {np.std(mses):.4f}")
    print(f"5-fold CV MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    # plot last fold loss (from PyTorch training)
    plt.figure()
    plt.plot(train_losses[-1], label='train')
    plt.plot(val_losses[-1], label='val')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Last Fold Training')
    plt.savefig(FIG_LOSS, bbox_inches='tight', dpi=200)
    plt.close()

    # 4. Averaged IG over folds
    all_igs = []
    for model, te in zip(fold_models, fold_tests):
        idx = te[0]
        ig = integrated_gradients(X_all[idx:idx+1], model, X_wt)
        # Debug: Check for NaNs in IG scores
        print(f'Fold IG: NaNs in IG:', torch.isnan(ig).sum().item())
        scores = ig[0].sum(dim=-1).detach().cpu().numpy()
        all_igs.append(scores)
    avg_scores = np.mean(np.stack(all_igs, axis=0), axis=0)
    print('NaNs in avg_scores:', np.isnan(avg_scores).sum())
    pos = np.arange(1, L+1)
    df_attr = pd.DataFrame({
        'Position': pos,
        'Residue': list(seqs[fold_tests[0][0]]),
        f'IG_Attribution_vs_WT{suffix}': avg_scores
    })
    df_attr.to_csv(DF_ATTR_CSV, index=False)
    plt.figure()
    plt.bar(pos, avg_scores)
    plt.xlabel('Residue')
    plt.ylabel('Avg IG vs WT')
    plt.title('Averaged IG across folds')
    plt.savefig(FIG_IG, bbox_inches='tight', dpi=200)
    plt.close()

    # 5. Hotspot selection & swap classification
    ig_abs = np.abs(df_attr[f'IG_Attribution_vs_WT{suffix}'])
    thr = np.percentile(ig_abs, HOTSPOT_PERCENTILE)
    print(f"Hotspot threshold at {HOTSPOT_PERCENTILE}th percentile: {thr:.2e}")
    hs = df_attr[ig_abs > thr]
    swaps=[]
    for _,r in hs.iterrows():
        wt  = wt_seq[int(r.Position)-1]
        var = seqs[fold_tests[0][0]][int(r.Position)-1]
        swaps.append({'Position': r.Position, 'WT': wt, 'Var': var,
                      'IG': r[f'IG_Attribution_vs_WT{suffix}'], 'Category': categorize_swap(wt,var)})
    df_swaps = pd.DataFrame(swaps)

    # 6. Enrichment
    if not df_swaps.empty:
        ct = pd.crosstab(df_swaps['Category'], df_swaps['IG']>0)
        p_chi = chi2_contingency(ct.values)[1]
        print('Chi² p-value:', p_chi)

    # 7. Correlation with property deltas
    deltas = [{'dvol':AA_PROPS[r['WT']]['vol']-AA_PROPS[r['Var']]['vol'],
               'dhyd':AA_PROPS[r['WT']]['hydro']-AA_PROPS[r['Var']]['hydro'],
               'IG':r['IG']} for r in swaps]
    df_del = pd.DataFrame(deltas)
    for c in ['dvol','dhyd']:
        if df_del[c].nunique()>1:
            rv,pv = pearsonr(df_del[c], df_del['IG'])
            print(f"Corr {c} vs IG: r={rv:.3f}, p={pv:.3g}")

    # 8. PyMOL
    if not df_swaps.empty:
        resnos = ','.join(map(str, df_swaps['Position']))
        with open(PYMOL_PML,'w') as f:
            f.write(f"select hotspots, resi {resnos}\ncolor red, hotspots\n")
        print('Wrote PyMOL script:', PYMOL_PML)

    print('Done.')