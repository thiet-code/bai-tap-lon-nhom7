"""Hybrid v2: weighted score fusion giữa NeuMF + BPR-MF + Content-Based.

Công thức: score = w_ncf · z(s_ncf) + w_mf · z(s_mf) + w_cb · z(s_cb)
  (z = z-score normalize để khử bias scale giữa 3 nguồn)

Cold-start: user có n_train < threshold sẽ tăng trọng số CB + popularity.
"""
from __future__ import annotations

import numpy as np
import torch

from models_nn import BPRMF, NeuMF


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mu, sd = x.mean(), x.std()
    if sd < 1e-9:
        return x - mu
    return (x - mu) / sd


class HybridV2:
    def __init__(
        self,
        ncf: NeuMF,
        mf: BPRMF,
        cb,
        user_train_count: dict[int, int],
        popularity: np.ndarray,
        w_ncf: float = 0.5,
        w_mf: float = 0.3,
        w_cb: float = 0.2,
        cold_threshold: int = 10,
        device: str = "cuda",
    ):
        self.ncf = ncf
        self.mf = mf
        self.cb = cb
        self.user_train_count = user_train_count
        self.pop = popularity.astype(np.float32)
        if self.pop.max() > 0:
            self.pop_norm = self.pop / self.pop.max()
        else:
            self.pop_norm = self.pop
        self.w_ncf = w_ncf
        self.w_mf = w_mf
        self.w_cb = w_cb
        self.cold_threshold = cold_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ncf.eval()
        self.mf.eval()

    @torch.no_grad()
    def score(self, user: int, items: list[int]) -> np.ndarray:
        u_t = torch.full((len(items),), user, dtype=torch.long, device=self.device)
        i_t = torch.tensor(items, dtype=torch.long, device=self.device)
        s_ncf = self.ncf(u_t, i_t).cpu().numpy()
        s_mf = self.mf.score(u_t, i_t).cpu().numpy()
        cb_full = self.cb.predict_scores(user)
        s_cb = np.array([cb_full[i] if i < cb_full.shape[0] else 0.0 for i in items], dtype=np.float32)

        z_ncf = _zscore(s_ncf)
        z_mf = _zscore(s_mf)
        z_cb = _zscore(s_cb)

        n_train = self.user_train_count.get(int(user), 0)
        if n_train < self.cold_threshold:
            # User cold hoặc thưa: nghiêng về CB + popularity
            ramp = n_train / self.cold_threshold
            w_ncf_e = self.w_ncf * ramp
            w_mf_e = self.w_mf * ramp
            w_pop = (1 - ramp) * 0.4
            w_cb_e = 1 - w_ncf_e - w_mf_e - w_pop
            s_pop = np.array([self.pop_norm[i] if i < self.pop_norm.shape[0] else 0.0 for i in items], dtype=np.float32)
            z_pop = _zscore(s_pop)
            return w_ncf_e * z_ncf + w_mf_e * z_mf + w_cb_e * z_cb + w_pop * z_pop
        else:
            return self.w_ncf * z_ncf + self.w_mf * z_mf + self.w_cb * z_cb

    def make_scorer(self):
        return lambda u, items: self.score(u, items)
