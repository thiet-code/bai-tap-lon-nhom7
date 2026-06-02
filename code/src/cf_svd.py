"""Collaborative Filtering bằng Matrix Factorization (Truncated SVD)."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class SVDRecommender:
    """SVD với mean centering theo user.

    Tách R = U S Vt sau khi trừ user-mean, tái tạo R_hat = U S Vt + user_mean.
    """

    def __init__(self, n_factors: int = 50, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.U: np.ndarray | None = None
        self.S: np.ndarray | None = None
        self.Vt: np.ndarray | None = None
        self.user_means: np.ndarray | None = None
        self.global_mean: float = 0.0
        self.pred_matrix: np.ndarray | None = None
        # True nếu user đã rate item này trong train
        self.train_mask: np.ndarray | None = None

    def fit(self, user_item: np.ndarray) -> "SVDRecommender":
        mat = user_item.astype(np.float32).copy()
        mask = mat > 0
        self.train_mask = mask

        # Chỉ tính user mean trên các rating thực sự có
        counts = mask.sum(axis=1)
        sums = mat.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            user_means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        self.user_means = user_means.astype(np.float32)
        self.global_mean = float(mat[mask].mean()) if mask.any() else 0.0

        # Mean-center, giữ 0 ở entry rỗng để SVD tập trung vào tín hiệu đã quan sát
        centered = mat - (mask.astype(np.float32) * user_means[:, None])

        k = min(self.n_factors, min(centered.shape) - 1)
        sparse = csr_matrix(centered)
        U, S, Vt = svds(sparse, k=k, random_state=self.random_state)
        # svds trả về theo singular value tăng dần, cần đảo ngược
        order = np.argsort(-S)
        self.U = U[:, order]
        self.S = S[order]
        self.Vt = Vt[order, :]

        self.pred_matrix = (self.U * self.S) @ self.Vt + user_means[:, None]
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        assert self.pred_matrix is not None
        if user_id >= self.pred_matrix.shape[0] or item_id >= self.pred_matrix.shape[1]:
            return self.global_mean
        return float(self.pred_matrix[user_id, item_id])

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        assert self.pred_matrix is not None and self.train_mask is not None
        if user_id <= 0 or user_id >= self.pred_matrix.shape[0]:
            return []
        scores = self.pred_matrix[user_id].copy()
        scores[0] = -np.inf
        if exclude_seen:
            seen = self.train_mask[user_id]
            scores[seen] = -np.inf
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx if np.isfinite(scores[i])]

    def is_cold_user(self, user_id: int) -> bool:
        if self.train_mask is None or user_id >= self.train_mask.shape[0]:
            return True
        return not bool(self.train_mask[user_id].any())


if __name__ == "__main__":
    from data_loader import load, train_test_split_per_user, build_user_item_matrix

    ml = load()
    train, _ = train_test_split_per_user(ml.ratings)
    R = build_user_item_matrix(train, ml.n_users, ml.n_items)

    model = SVDRecommender(n_factors=50).fit(R)
    recs = model.recommend(user_id=1, top_k=5)
    print("Top-5 goi y cho user 1 (SVD):")
    for item_id, score in recs:
        title = ml.movies.set_index("item_id").loc[item_id, "title"]
        print(f"  {item_id:>4}  {score:.3f}  {title}")
