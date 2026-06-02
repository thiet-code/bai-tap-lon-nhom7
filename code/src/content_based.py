"""Content-Based Filtering bằng TF-IDF trên genres + title."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _clean_title(text: str) -> str:
    # Bỏ năm phát hành "(1995)" cuối tiêu đề để TF-IDF khỏi nhiễu
    text = re.sub(r"\(\d{4}\)\s*$", "", str(text)).strip()
    return text.lower()


class ContentBasedRecommender:
    """TF-IDF trên genres × 3 + title. User profile = trung bình có trọng số item vector."""

    def __init__(self, min_rating_for_profile: float = 3.5):
        self.min_rating_for_profile = min_rating_for_profile
        self.vectorizer: TfidfVectorizer | None = None
        self.item_vectors: csr_matrix | None = None
        self.item_id_to_row: dict[int, int] = {}
        self.user_profiles: csr_matrix | None = None
        # Số rating mỗi item, dùng làm fallback cho cold-item
        self.popularity: np.ndarray | None = None
        self.n_users: int = 0
        self.n_items: int = 0

    def fit(
        self,
        movies: pd.DataFrame,
        train_ratings: pd.DataFrame,
        n_users: int,
        n_items: int,
    ) -> "ContentBasedRecommender":
        self.n_users = n_users
        self.n_items = n_items

        # Văn bản mỗi phim: genres lặp 3 lần để có trọng số cao hơn title
        docs: list[str] = [""]
        ids_in_order: list[int] = [0]
        for _, row in movies.sort_values("item_id").iterrows():
            genres = str(row.get("genres", "")).replace("-", "")
            title = _clean_title(row.get("title", ""))
            doc = f"{genres} {genres} {genres} {title}".strip()
            docs.append(doc or "unknown")
            ids_in_order.append(int(row["item_id"]))

        # Pad nếu n_items lớn hơn số phim có metadata
        while len(docs) <= n_items:
            docs.append("unknown")
            ids_in_order.append(len(ids_in_order))

        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
            ngram_range=(1, 1),
            min_df=2,
        )
        item_mat = self.vectorizer.fit_transform(docs)
        self.item_vectors = normalize(item_mat, norm="l2", axis=1)
        self.item_id_to_row = {iid: idx for idx, iid in enumerate(ids_in_order)}

        # Profile user = trung bình có trọng số các vector phim user rate >= ngưỡng
        liked = train_ratings[train_ratings["rating"] >= self.min_rating_for_profile]
        profiles_rows = []
        for uid in range(n_users + 1):
            user_likes = liked[liked["user_id"] == uid]
            if len(user_likes) == 0:
                profiles_rows.append(csr_matrix((1, self.item_vectors.shape[1]), dtype=np.float32))
                continue
            rows = [self.item_id_to_row[int(i)] for i in user_likes["item_id"] if int(i) in self.item_id_to_row]
            weights = user_likes["rating"].values
            if not rows:
                profiles_rows.append(csr_matrix((1, self.item_vectors.shape[1]), dtype=np.float32))
                continue
            sub = self.item_vectors[rows].multiply(weights[:, None])
            prof = sub.sum(axis=0)
            prof = csr_matrix(prof)
            profiles_rows.append(prof)
        self.user_profiles = normalize(vstack(profiles_rows), norm="l2", axis=1)

        pop = np.zeros(n_items + 1, dtype=np.int32)
        vc = train_ratings["item_id"].value_counts()
        pop[vc.index.values] = vc.values
        self.popularity = pop
        return self

    def _scores_for_user(self, user_id: int) -> np.ndarray:
        assert self.user_profiles is not None and self.item_vectors is not None
        if user_id < 0 or user_id >= self.user_profiles.shape[0]:
            return np.zeros(self.n_items + 1, dtype=np.float32)
        profile = self.user_profiles[user_id]
        if profile.nnz == 0:
            return np.zeros(self.n_items + 1, dtype=np.float32)
        sims = (self.item_vectors @ profile.T).toarray().ravel()
        if sims.shape[0] < self.n_items + 1:
            out = np.zeros(self.n_items + 1, dtype=np.float32)
            out[: sims.shape[0]] = sims
            return out
        return sims[: self.n_items + 1]

    def predict_scores(self, user_id: int) -> np.ndarray:
        """Trả về score cho mỗi item. User cold không profile thì fallback popularity."""
        scores = self._scores_for_user(user_id)
        if not np.any(scores > 0) and self.popularity is not None:
            pop = self.popularity.astype(np.float32)
            if pop.max() > 0:
                scores = pop / pop.max()
        return scores

    def recommend(
        self,
        user_id: int,
        train_mask_row: np.ndarray | None = None,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        scores = self.predict_scores(user_id).copy()
        scores[0] = -np.inf
        if train_mask_row is not None:
            scores[train_mask_row] = -np.inf
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx if np.isfinite(scores[i])]

    def item_similarity(self, item_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Tìm K phim tương tự nhất, dùng cho kịch bản cold-item."""
        if item_id not in self.item_id_to_row or self.item_vectors is None:
            return []
        row = self.item_id_to_row[item_id]
        sims = (self.item_vectors @ self.item_vectors[row].T).toarray().ravel()
        sims[row] = -np.inf
        top_idx = np.argpartition(-sims, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        inv = {v: k for k, v in self.item_id_to_row.items()}
        return [(inv[i], float(sims[i])) for i in top_idx if np.isfinite(sims[i])]


if __name__ == "__main__":
    from data_loader import load, train_test_split_per_user

    ml = load()
    train, _ = train_test_split_per_user(ml.ratings)
    cb = ContentBasedRecommender().fit(ml.movies, train, ml.n_users, ml.n_items)

    print("Top-5 phim giong 'Toy Story' (item 1):")
    for iid, sim in cb.item_similarity(1, top_k=5):
        title = ml.movies.set_index("item_id").loc[iid, "title"]
        print(f"  {iid:>4}  sim={sim:.3f}  {title}")

    print("\nTop-5 goi y content-based cho user 1:")
    for iid, sc in cb.recommend(user_id=1, top_k=5):
        title = ml.movies.set_index("item_id").loc[iid, "title"]
        print(f"  {iid:>4}  score={sc:.3f}  {title}")
