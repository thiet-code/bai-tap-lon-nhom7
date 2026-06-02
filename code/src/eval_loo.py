"""Đánh giá leave-one-out theo chuẩn paper NCF: HR@K, NDCG@K, MRR.

Với mỗi user trong test:
  - 1 ground-truth + 99 negative = 100 ứng viên
  - Xếp hạng 100 item rồi tính HR@K (hit nếu ground-truth ∈ top-K)
    và NDCG@K = 1/log2(rank+2) nếu rank < K.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def evaluate_loo(
    score_fn: Callable[[int, list[int]], np.ndarray],
    test_users: list[int],
    test_pos: dict[int, int],
    test_negs: dict[int, list[int]],
    k_list: list[int] = [5, 10, 20],
) -> dict[str, float]:
    """score_fn(user, items) trả về np.ndarray score (càng cao càng thích)."""
    hrs = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}
    mrrs = []

    for uid in test_users:
        if uid not in test_pos or uid not in test_negs:
            continue
        pos = test_pos[uid]
        # Ground-truth nằm ở vị trí 0 trong candidates
        candidates = [pos] + test_negs[uid]
        scores = score_fn(uid, candidates)

        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0])

        mrrs.append(1.0 / (rank + 1))
        for k in k_list:
            if rank < k:
                hrs[k].append(1.0)
                ndcgs[k].append(1.0 / np.log2(rank + 2))
            else:
                hrs[k].append(0.0)
                ndcgs[k].append(0.0)

    out: dict[str, float] = {"n_users": len(mrrs), "mrr": float(np.mean(mrrs))}
    for k in k_list:
        out[f"hr@{k}"] = float(np.mean(hrs[k]))
        out[f"ndcg@{k}"] = float(np.mean(ndcgs[k]))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    test_pos = {1: 100, 2: 200}
    test_negs = {1: list(range(101, 200)), 2: list(range(201, 300))}

    def random_score(u, items):
        return rng.random(len(items))

    r = evaluate_loo(random_score, [1, 2], test_pos, test_negs)
    print("Random HR@10:", r["hr@10"], " NDCG@10:", r["ndcg@10"])
