"""Đọc dữ liệu MovieLens 100K và chia train/test theo từng user (leave-N-out)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw", "ml-100k")

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


@dataclass
class MovieLens:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    users: pd.DataFrame
    n_users: int
    n_items: int


def load() -> MovieLens:
    ratings = pd.read_csv(
        os.path.join(RAW_DIR, "u.data"),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLS
    movies = pd.read_csv(
        os.path.join(RAW_DIR, "u.item"),
        sep="|",
        names=item_cols,
        encoding="latin-1",
        engine="python",
    )
    movies["genres"] = movies[GENRE_COLS].apply(
        lambda r: " ".join([g for g, v in zip(GENRE_COLS, r) if v == 1]) or "unknown",
        axis=1,
    )

    users = pd.read_csv(
        os.path.join(RAW_DIR, "u.user"),
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        engine="python",
    )

    return MovieLens(
        ratings=ratings,
        movies=movies,
        users=users,
        n_users=int(ratings["user_id"].max()),
        n_items=int(ratings["item_id"].max()),
    )


def train_test_split_per_user(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    min_test: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia tập test bằng cách lấy một phần rating của mỗi user."""
    rng = np.random.default_rng(seed)
    test_rows: list[int] = []
    for _, idxs in ratings.groupby("user_id").groups.items():
        idxs = np.array(list(idxs))
        n_test = max(min_test, int(round(len(idxs) * test_ratio)))
        # Giữ lại ít nhất 1 rating trong train
        n_test = min(n_test, len(idxs) - 1)
        if n_test <= 0:
            continue
        chosen = rng.choice(idxs, size=n_test, replace=False)
        test_rows.extend(chosen.tolist())

    test_mask = ratings.index.isin(test_rows)
    train = ratings.loc[~test_mask].reset_index(drop=True)
    test = ratings.loc[test_mask].reset_index(drop=True)
    return train, test


def build_user_item_matrix(
    ratings: pd.DataFrame, n_users: int, n_items: int
) -> np.ndarray:
    """Dựng ma trận user × item, giá trị là rating, 0 nếu chưa có."""
    matrix = np.zeros((n_users + 1, n_items + 1), dtype=np.float32)
    matrix[ratings["user_id"].values, ratings["item_id"].values] = ratings["rating"].values
    return matrix


if __name__ == "__main__":
    ml = load()
    print(f"So user: {ml.n_users}")
    print(f"So phim: {ml.n_items}")
    print(f"So rating: {len(ml.ratings):,}")
    print(f"Mat do du lieu: {len(ml.ratings) / (ml.n_users * ml.n_items) * 100:.2f}%")
    print(f"Trung binh rating: {ml.ratings['rating'].mean():.2f}")
    print("\nMau 5 phim:")
    print(ml.movies[["item_id", "title", "genres"]].head())

    train, test = train_test_split_per_user(ml.ratings, test_ratio=0.2)
    print(f"\nTrain size: {len(train):,}  |  Test size: {len(test):,}")
    print(f"So user trong test: {test['user_id'].nunique()}")
