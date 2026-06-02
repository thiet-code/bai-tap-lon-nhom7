"""Hybrid Recommender phiên bản v1: kết hợp SVD-CF và Content-Based, xử lý cold-start."""
from __future__ import annotations

import numpy as np

from cf_svd import SVDRecommender
from content_based import ContentBasedRecommender


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Chuẩn hoá min-max về [0, 1] để hai score so sánh được."""
    out = arr.copy().astype(np.float32)
    finite = np.isfinite(out)
    if not finite.any():
        return out
    lo = out[finite].min()
    hi = out[finite].max()
    if hi - lo < 1e-9:
        out[finite] = 0.5
        return out
    out[finite] = (out[finite] - lo) / (hi - lo)
    return out


class HybridRecommender:
    """Weighted hybrid: score = alpha · CF + (1 - alpha) · CB.

    Cold-start:
      - User không có rating trong train → dùng 100% content-based hoặc popularity.
      - Item mới (chưa có rating nào) → CF cho 0, hybrid vẫn chạy nhờ CB.
    """

    COLD_USER_THRESHOLD = 5

    def __init__(
        self,
        cf: SVDRecommender,
        cb: ContentBasedRecommender,
        alpha: float = 0.6,
        cold_cb_weight: float = 0.5,
    ):
        self.cf = cf
        self.cb = cb
        self.alpha = alpha
        self.cold_cb_weight = cold_cb_weight
        # Popularity bao gồm mọi item được rate trong train, dùng để fallback
        self._popularity = None
        if self.cb.popularity is not None:
            pop = self.cb.popularity.astype(np.float32)
            if pop.max() > 0:
                self._popularity = pop / pop.max()

    def _user_rating_count(self, user_id: int) -> int:
        if self.cf.train_mask is None or user_id >= self.cf.train_mask.shape[0]:
            return 0
        return int(self.cf.train_mask[user_id].sum())

    def predict_scores(self, user_id: int) -> tuple[np.ndarray, str]:
        """Trả về (scores, mode) - mode là chế độ định tuyến đã chọn."""
        n_ratings = self._user_rating_count(user_id)

        # Cold user: kết hợp popularity với CB nếu có profile
        if n_ratings == 0:
            cb_scores = self.cb.predict_scores(user_id)
            has_profile = (
                self.cb.user_profiles is not None
                and user_id < self.cb.user_profiles.shape[0]
                and self.cb.user_profiles[user_id].nnz > 0
            )
            if has_profile and self._popularity is not None:
                n = min(cb_scores.shape[0], self._popularity.shape[0])
                scores = self.cold_cb_weight * _minmax(cb_scores[:n]) + (1 - self.cold_cb_weight) * self._popularity[:n]
                return scores, "cold_user_cb_pop"
            if self._popularity is not None:
                return self._popularity.copy(), "cold_user_pop"
            return cb_scores, "cold_user_cb"

        # User thưa rating: kéo CB và popularity mạnh hơn
        if n_ratings < self.COLD_USER_THRESHOLD:
            assert self.cf.pred_matrix is not None
            cf_raw = self.cf.pred_matrix[user_id]
            cb_raw = self.cb.predict_scores(user_id)
            n = min(cf_raw.shape[0], cb_raw.shape[0])
            if self._popularity is not None:
                n = min(n, self._popularity.shape[0])
            cf_n = _minmax(cf_raw[:n])
            cb_n = _minmax(cb_raw[:n])
            ramp = n_ratings / self.COLD_USER_THRESHOLD
            # CF rất nhỏ khi user có rất ít rating
            a = self.alpha * (ramp ** 2)
            # Popularity chiếm đến 65% khi n=1
            pop_w = (1 - ramp) * 0.65
            cb_w = max(0.0, 1 - a - pop_w)
            scores = a * cf_n + cb_w * cb_n
            if self._popularity is not None:
                scores = scores + pop_w * self._popularity[:n]
            return scores, "sparse_user"

        # User bình thường: alpha · CF + (1-alpha) · CB
        assert self.cf.pred_matrix is not None
        cf_raw = self.cf.pred_matrix[user_id]
        cb_raw = self.cb.predict_scores(user_id)
        n = min(cf_raw.shape[0], cb_raw.shape[0])
        cf_n = _minmax(cf_raw[:n])
        cb_n = _minmax(cb_raw[:n])
        scores = self.alpha * cf_n + (1.0 - self.alpha) * cb_n
        return scores, "hybrid"

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float, str]]:
        scores, mode = self.predict_scores(user_id)
        scores = scores.copy()
        scores[0] = -np.inf
        if exclude_seen and self.cf.train_mask is not None and user_id < self.cf.train_mask.shape[0]:
            seen = self.cf.train_mask[user_id][: scores.shape[0]]
            scores[: seen.shape[0]][seen] = -np.inf
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i]), mode) for i in top_idx if np.isfinite(scores[i])]


if __name__ == "__main__":
    from data_loader import load, train_test_split_per_user, build_user_item_matrix

    ml = load()
    train, _ = train_test_split_per_user(ml.ratings)
    R = build_user_item_matrix(train, ml.n_users, ml.n_items)

    cf = SVDRecommender(n_factors=50).fit(R)
    cb = ContentBasedRecommender().fit(ml.movies, train, ml.n_users, ml.n_items)
    hy = HybridRecommender(cf, cb, alpha=0.6)

    print("Top-5 hybrid cho user 1 (active user):")
    for iid, sc, mode in hy.recommend(1, top_k=5):
        title = ml.movies.set_index("item_id").loc[iid, "title"]
        print(f"  [{mode}] {iid:>4}  score={sc:.3f}  {title}")

    print("\nTop-5 hybrid cho user 9999 (cold user, chua ton tai):")
    for iid, sc, mode in hy.recommend(9999, top_k=5):
        title = ml.movies.set_index("item_id").loc[iid, "title"]
        print(f"  [{mode}] {iid:>4}  score={sc:.3f}  {title}")
