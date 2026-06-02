"""Bốn mô hình PyTorch: BPR-MF (MF implicit) và NeuMF (Neural CF).

Tham khảo:
  - Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)
  - He et al. "Neural Collaborative Filtering" (WWW 2017)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRMF(nn.Module):
    """Matrix Factorization với BPR loss: f(u, i) = b_u + b_i + <p_u, q_i>."""

    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, dim)
        self.item_emb = nn.Embedding(n_items + 1, dim)
        self.user_bias = nn.Embedding(n_users + 1, 1)
        self.item_bias = nn.Embedding(n_items + 1, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (
            (self.user_emb(u) * self.item_emb(i)).sum(-1)
            + self.user_bias(u).squeeze(-1)
            + self.item_bias(i).squeeze(-1)
        )

    def forward(self, u, i_pos, i_neg):
        s_pos = self.score(u, i_pos)
        s_neg = self.score(u, i_neg)
        # BPR loss
        return -F.logsigmoid(s_pos - s_neg).mean()


class GMF(nn.Module):
    """Generalized Matrix Factorization: f = sigmoid(h^T (p_u ⊙ q_i))."""

    def __init__(self, n_users: int, n_items: int, dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, dim)
        self.item_emb = nn.Embedding(n_items + 1, dim)
        self.out = nn.Linear(dim, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        pu = self.user_emb(u)
        qi = self.item_emb(i)
        return self.out(pu * qi).squeeze(-1)


class MLP(nn.Module):
    """MLP: f = MLP([p_u ‖ q_i]) → 1."""

    def __init__(self, n_users: int, n_items: int, dim: int = 32, hidden=(64, 32, 16)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, dim)
        self.item_emb = nn.Embedding(n_items + 1, dim)
        layers = []
        prev = dim * 2
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        return self.mlp(x).squeeze(-1)


class NeuMF(nn.Module):
    """NeuMF = GMF + MLP fusion (He et al. 2017, paper NCF)."""

    def __init__(
        self, n_users: int, n_items: int,
        gmf_dim: int = 32, mlp_dim: int = 32, mlp_hidden=(64, 32, 16),
    ):
        super().__init__()
        self.gmf_user = nn.Embedding(n_users + 1, gmf_dim)
        self.gmf_item = nn.Embedding(n_items + 1, gmf_dim)
        self.mlp_user = nn.Embedding(n_users + 1, mlp_dim)
        self.mlp_item = nn.Embedding(n_items + 1, mlp_dim)
        layers = []
        prev = mlp_dim * 2
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(gmf_dim + mlp_hidden[-1], 1)

        for emb in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, u, i):
        gmf_vec = self.gmf_user(u) * self.gmf_item(i)
        mlp_vec = self.mlp(torch.cat([self.mlp_user(u), self.mlp_item(i)], dim=-1))
        return self.out(torch.cat([gmf_vec, mlp_vec], dim=-1)).squeeze(-1)
