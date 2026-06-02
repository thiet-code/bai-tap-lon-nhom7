"""Chuẩn bị dữ liệu implicit feedback và leave-one-out theo chuẩn paper NCF.

Mỗi rating coi là tín hiệu positive. Với mỗi user, giữ tương tác cuối (theo
timestamp) làm test, sample 99 negative để tính HR@K, NDCG@K.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loader import load as load_explicit


@dataclass
class ImplicitData:
    train: pd.DataFrame
    test: pd.DataFrame
    test_negatives: dict[int, list[int]]
    n_users: int
    n_items: int
    user_pos: dict[int, set[int]]
    movies: pd.DataFrame
    users: pd.DataFrame


def build_implicit(seed: int = 42, n_neg: int = 99) -> ImplicitData:
    ml = load_explicit()
    ratings = ml.ratings.copy()
    n_users, n_items = ml.n_users, ml.n_items

    # Leave-one-out: rating có timestamp lớn nhất của mỗi user vào test
    ratings = ratings.sort_values(["user_id", "timestamp"])
    test = ratings.groupby("user_id", group_keys=False).tail(1).reset_index(drop=True)
    train = ratings.merge(
        test[["user_id", "item_id", "timestamp"]],
        on=["user_id", "item_id", "timestamp"], how="left", indicator=True,
    ).query('_merge=="left_only"').drop(columns="_merge").reset_index(drop=True)

    train = train.rename(columns={"timestamp": "ts"})[["user_id", "item_id", "ts"]]
    test = test.rename(columns={"timestamp": "ts"})[["user_id", "item_id", "ts"]]

    user_pos: dict[int, set[int]] = {}
    for uid, grp in ratings.groupby("user_id"):
        user_pos[int(uid)] = set(int(i) for i in grp["item_id"])

    # Sample 99 negative cho mỗi user trong test
    rng = np.random.default_rng(seed)
    all_items = np.arange(1, n_items + 1)
    test_negs: dict[int, list[int]] = {}
    for uid in test["user_id"].unique():
        pos = user_pos.get(int(uid), set())
        negs: list[int] = []
        while len(negs) < n_neg:
            cand = rng.choice(all_items, size=n_neg * 2, replace=False)
            for c in cand:
                if int(c) not in pos and int(c) not in negs:
                    negs.append(int(c))
                    if len(negs) == n_neg:
                        break
        test_negs[int(uid)] = negs

    return ImplicitData(
        train=train,
        test=test,
        test_negatives=test_negs,
        n_users=n_users,
        n_items=n_items,
        user_pos=user_pos,
        movies=ml.movies,
        users=ml.users,
    )


def sample_train_negatives(
    train: pd.DataFrame, user_pos: dict[int, set[int]], n_items: int,
    n_neg_per_pos: int = 4, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sinh batch huấn luyện với tỉ lệ 1 positive : n negative."""
    rng = np.random.default_rng(seed)
    users, items, labels = [], [], []
    for u, i_pos in zip(train["user_id"].values, train["item_id"].values):
        users.append(u)
        items.append(i_pos)
        labels.append(1.0)
        pos_set = user_pos.get(int(u), set())
        neg_drawn = 0
        while neg_drawn < n_neg_per_pos:
            j = int(rng.integers(1, n_items + 1))
            if j not in pos_set:
                users.append(u)
                items.append(j)
                labels.append(0.0)
                neg_drawn += 1
    return (
        np.array(users, dtype=np.int64),
        np.array(items, dtype=np.int64),
        np.array(labels, dtype=np.float32),
    )


if __name__ == "__main__":
    d = build_implicit()
    print(f"Train: {len(d.train):,}  Test: {len(d.test):,}")
    print(f"n_users={d.n_users}  n_items={d.n_items}")
    print(f"Test negatives mau: user 1 -> {d.test_negatives[1][:10]}...")
    u, i, y = sample_train_negatives(d.train.head(100), d.user_pos, d.n_items, n_neg_per_pos=4)
    print(f"Sample batch: users={u.shape}  positives={y.sum():.0f}  negatives={(y==0).sum():.0f}")
