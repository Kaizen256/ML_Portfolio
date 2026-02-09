import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "features" / "external"

MEANS_PATH  = ROOT / "data" / "training_data_means.csv"
VALMAP_PATH = ROOT / "data" / "pert_ids_val.csv"

SEED = 6
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model settings
K_OUT    = 32
HID      = 256
DROPOUT  = 0.10
LR       = 2e-3
WD       = 1e-4
EPOCHS   = 1500
BATCH    = 32
PATIENCE = 120

class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_out),
        )

    def forward(self, x):
        return self.net(x)

def fit_model(Xtr, Ytr, Xva, Yva):
    model = MLP(Xtr.shape[1], HID, Ytr.shape[1], dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr, device=DEVICE)
    Ytr_t = torch.tensor(Ytr, device=DEVICE)
    Xva_t = torch.tensor(Xva, device=DEVICE)
    Yva_t = torch.tensor(Yva, device=DEVICE)

    best = float("inf")
    best_state = None
    bad = 0

    ds = TensorDataset(Xtr_t, Ytr_t)
    bs = min(BATCH, len(ds))

    for epoch in range(EPOCHS):
        model.train()
        dl = DataLoader(ds, batch_size=bs, shuffle=True)

        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_pred = model(Xva_t)
            va_loss = loss_fn(va_pred, Yva_t).item()

        if va_loss < best - 1e-6:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model

def main():
    df = pd.read_csv(MEANS_PATH)
    gene_cols = [c for c in df.columns if c != "pert_symbol"]

    base_row = df["pert_symbol"].astype(str) == "non-targeting"
    x_base = df.loc[base_row, gene_cols].iloc[0].to_numpy(np.float32)

    df_train = df.loc[~base_row].reset_index(drop=True)
    train_genes = df_train["pert_symbol"].astype(str).tolist()
    D = df_train[gene_cols].to_numpy(np.float32) - x_base[None, :]

    df_val = pd.read_csv(VALMAP_PATH)
    val_map = dict(zip(df_val["pert_id"].astype(str), df_val["pert"].astype(str)))
    val_genes = list(val_map.values())

    union = sorted(set(gene_cols) | set(train_genes) | set(val_genes))
    u2i = {g.upper(): i for i, g in enumerate(union)}

    ridx_train = np.array([u2i[g.upper()] for g in train_genes], dtype=np.int64)
    ridx_val   = np.array([u2i[g.upper()] for g in val_genes], dtype=np.int64)

    E_m3   = np.load(FEAT / "genept_m3_pca128.npy")
    E_ada  = np.load(FEAT / "genept_ada_pca128.npy")
    F_str  = np.load(FEAT / "string_graph_feats.npy")
    E_go   = np.load(FEAT / "go_svd128.npy")
    E_re   = np.load(FEAT / "reactome_svd128.npy")

    X_all = np.concatenate([E_m3, E_ada, F_str, E_go, E_re], axis=1).astype(np.float32)
    Xtr = X_all[ridx_train]
    Xva = X_all[ridx_val]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
    Xva = scaler.transform(Xva).astype(np.float32)

    svd_out = TruncatedSVD(n_components=K_OUT, random_state=SEED)
    Ytr = svd_out.fit_transform(D).astype(np.float32)

    n = len(train_genes)
    perm = np.random.permutation(n)
    cut = max(8, int(0.2 * n))
    va_idx = perm[:cut]
    tr_idx = perm[cut:]

    model = fit_model(Xtr[tr_idx], Ytr[tr_idx], Xtr[va_idx], Ytr[va_idx])

    model.eval()
    with torch.no_grad():
        coeff = model(torch.tensor(Xva, device=DEVICE)).cpu().numpy().astype(np.float32)

    Dhat = svd_out.inverse_transform(coeff).astype(np.float32)

    rows = []
    for k in range(1, 121):
        pert_id = f"pert_{k}"
        g = val_map.get(pert_id, None)

        if g is None:
            delta = D.mean(axis=0).astype(np.float32)
        else:
            j = list(val_map.keys()).index(pert_id)  # 0..119
            delta = Dhat[j]

        row = {"pert_id": pert_id}
        row.update({gene_cols[i]: float(delta[i]) for i in range(len(gene_cols))})
        rows.append(row)

    sub = pd.DataFrame(rows, columns=["pert_id"] + gene_cols)
    out_path = ROOT / "modelA_m3_all_submit.csv"
    sub.to_csv(out_path, index=False)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
