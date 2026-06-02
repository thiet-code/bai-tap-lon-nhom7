"""Bộ metric top-N: Precision@K, Recall@K, MAP@K, NDCG@K."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for r in rec_k if r in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for r in rec_k if r in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def ndcg_at_k(
    recommended: list[int], relevant_with_rel: dict[int, float], k: int
) -> float:
    """NDCG với mức độ liên quan = rating. Trả 0 nếu không có item liên quan."""
    if not relevant_with_rel:
        return 0.0
    rec_k = recommended[:k]
    gains = [relevant_with_rel.get(item, 0.0) for item in rec_k]
    dcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(gains))

    ideal = sorted(relevant_with_rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def evaluate_recommender(
    recommend_fn,
    test_ratings,
    user_ids: Iterable[int],
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> dict[str, float]:
    """Chạy recommend_fn(user_id, top_k) cho từng user và lấy trung bình."""
    by_user = test_ratings.groupby("user_id")
    precisions, recalls, maps, ndcgs = [], [], [], []
    covered = 0

    for uid in user_ids:
        if uid not in by_user.groups:
            continue
        user_test = by_user.get_group(uid)
        # Item liên quan = rating >= ngưỡng
        liked = user_test[user_test["rating"] >= relevance_threshold]
        relevant_set = set(int(i) for i in liked["item_id"])
        rel_with_score = {int(r.item_id): float(r.rating) for r in liked.itertuples()}
        if not relevant_set:
            continue

        recs = recommend_fn(uid, k)
        if not recs:
            continue

        rec_ids = [int(r[0]) if isinstance(r, tuple) else int(r) for r in recs]
        precisions.append(precision_at_k(rec_ids, relevant_set, k))
        recalls.append(recall_at_k(rec_ids, relevant_set, k))
        maps.append(average_precision_at_k(rec_ids, relevant_set, k))
        ndcgs.append(ndcg_at_k(rec_ids, rel_with_score, k))
        covered += 1

    if covered == 0:
        return {"precision@k": 0, "recall@k": 0, "map@k": 0, "ndcg@k": 0, "n_users": 0}

    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        f"map@{k}": float(np.mean(maps)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "n_users": covered,
    }


if __name__ == "__main__":
    rec = [1, 2, 3, 4, 5]
    rel = {1, 3, 8}
    print("P@5:", precision_at_k(rec, rel, 5))
    print("R@5:", recall_at_k(rec, rel, 5))
    print("MAP@5:", average_precision_at_k(rec, rel, 5))
    print("NDCG@5:", ndcg_at_k(rec, {1: 5.0, 3: 4.0, 8: 5.0}, 5))
