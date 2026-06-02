"""Hàm huấn luyện BPR-MF và NeuMF (binary cross-entropy với negative sampling 1:4)."""
from __future__ import annotations

import time

import numpy as np
import torch

from data_implicit import sample_train_negatives
from models_nn import BPRMF, NeuMF


def train_bpr(
    train_df, user_pos, n_users, n_items,
    dim: int = 64, epochs: int = 30, batch_size: int = 4096,
    lr: float = 0.005, weight_decay: float = 1e-5,
    n_neg_per_pos: int = 4, device: str = "cuda", verbose: bool = True,
) -> BPRMF:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = BPRMF(n_users, n_items, dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_users = train_df["user_id"].values.astype(np.int64)
    pos_items = train_df["item_id"].values.astype(np.int64)
    n = len(pos_users)
    rng = np.random.default_rng(0)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        # Sample 1 negative cho mỗi positive (mỗi epoch resample lại)
        neg_items = np.empty(n, dtype=np.int64)
        for k in range(n):
            u = int(pos_users[k])
            pos_set = user_pos.get(u, set())
            while True:
                j = int(rng.integers(1, n_items + 1))
                if j not in pos_set:
                    neg_items[k] = j
                    break
        perm = rng.permutation(n)
        losses = []
        for s in range(0, n, batch_size):
            b = perm[s:s + batch_size]
            u = torch.from_numpy(pos_users[b]).to(device)
            i = torch.from_numpy(pos_items[b]).to(device)
            j = torch.from_numpy(neg_items[b]).to(device)
            loss = model(u, i, j)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        if verbose:
            print(f"  [BPR-MF] epoch {ep:>2}/{epochs}  loss={np.mean(losses):.4f}  ({time.time()-t0:.1f}s)")
    return model


def train_ncf(
    train_df, user_pos, n_users, n_items,
    epochs: int = 25, batch_size: int = 4096, lr: float = 0.001,
    weight_decay: float = 1e-6, n_neg_per_pos: int = 4,
    device: str = "cuda", verbose: bool = True,
    gmf_dim: int = 32, mlp_dim: int = 32, mlp_hidden=(64, 32, 16),
) -> NeuMF:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = NeuMF(n_users, n_items, gmf_dim=gmf_dim, mlp_dim=mlp_dim, mlp_hidden=mlp_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = torch.nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        t0 = time.time()
        u_np, i_np, y_np = sample_train_negatives(train_df, user_pos, n_items, n_neg_per_pos, seed=ep)
        perm = np.random.permutation(len(u_np))
        u_np, i_np, y_np = u_np[perm], i_np[perm], y_np[perm]
        losses = []
        for s in range(0, len(u_np), batch_size):
            u = torch.from_numpy(u_np[s:s + batch_size]).to(device)
            i = torch.from_numpy(i_np[s:s + batch_size]).to(device)
            y = torch.from_numpy(y_np[s:s + batch_size]).to(device)
            logits = model(u, i)
            loss = bce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        if verbose:
            print(f"  [NeuMF]   epoch {ep:>2}/{epochs}  loss={np.mean(losses):.4f}  ({time.time()-t0:.1f}s)")
    return model


def make_torch_scorer(model, device: str = "cuda"):
    """Bọc PyTorch model thành score_fn(user, items) cho eval_loo."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()

    @torch.no_grad()
    def fn(user: int, items: list[int]):
        u = torch.full((len(items),), user, dtype=torch.long, device=device)
        i = torch.tensor(items, dtype=torch.long, device=device)
        if isinstance(model, BPRMF):
            s = model.score(u, i)
        else:
            s = model(u, i)
        return s.cpu().numpy()

    return fn
