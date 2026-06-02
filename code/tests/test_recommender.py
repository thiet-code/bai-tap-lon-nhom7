"""Unit test cho hệ thống gợi ý."""
from __future__ import annotations

import os
import sys
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_loader import build_user_item_matrix, load, train_test_split_per_user  # noqa: E402
from cf_svd import SVDRecommender  # noqa: E402
from content_based import ContentBasedRecommender  # noqa: E402
from hybrid import HybridRecommender  # noqa: E402
from metrics import (  # noqa: E402
    average_precision_at_k, ndcg_at_k, precision_at_k, recall_at_k,
)


class TestMetrics(unittest.TestCase):
    def test_precision_perfect(self):
        self.assertEqual(precision_at_k([1, 2, 3], {1, 2, 3}, 3), 1.0)

    def test_precision_zero(self):
        self.assertEqual(precision_at_k([4, 5, 6], {1, 2, 3}, 3), 0.0)

    def test_recall(self):
        self.assertAlmostEqual(recall_at_k([1, 2, 9], {1, 2, 3}, 3), 2 / 3)

    def test_map(self):
        # 1 hit ở vị trí 1, 1 hit ở vị trí 3 → (1/1 + 2/3) / 3
        ap = average_precision_at_k([1, 9, 3, 8], {1, 3, 5}, 4)
        self.assertAlmostEqual(ap, (1.0 + 2 / 3) / 3, places=4)

    def test_ndcg_perfect(self):
        rel = {1: 5.0, 2: 4.0, 3: 3.0}
        ndcg = ndcg_at_k([1, 2, 3], rel, 3)
        self.assertAlmostEqual(ndcg, 1.0, places=4)

    def test_ndcg_empty(self):
        self.assertEqual(ndcg_at_k([1, 2], {}, 2), 0.0)


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ml = load()
        cls.train, cls.test = train_test_split_per_user(
            cls.ml.ratings, test_ratio=0.2, seed=42
        )
        cls.R = build_user_item_matrix(cls.train, cls.ml.n_users, cls.ml.n_items)

    def test_svd_shape(self):
        cf = SVDRecommender(n_factors=20).fit(self.R)
        self.assertIsNotNone(cf.pred_matrix)
        self.assertEqual(cf.pred_matrix.shape, (self.ml.n_users + 1, self.ml.n_items + 1))

    def test_svd_recommend(self):
        cf = SVDRecommender(n_factors=20).fit(self.R)
        recs = cf.recommend(user_id=1, top_k=10)
        self.assertEqual(len(recs), 10)
        # Không được trùng item_id 0 và không được trùng item user đã rate trong train
        seen = set(self.train[self.train["user_id"] == 1]["item_id"])
        for iid, _ in recs:
            self.assertNotIn(iid, seen)
            self.assertGreater(iid, 0)

    def test_cb_item_similarity(self):
        cb = ContentBasedRecommender().fit(self.ml.movies, self.train, self.ml.n_users, self.ml.n_items)
        sims = cb.item_similarity(item_id=1, top_k=5)
        self.assertEqual(len(sims), 5)
        # Cosine similarity phải nằm trong [0, 1]
        for _, s in sims:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0 + 1e-6)

    def test_cb_cold_item_fallback(self):
        """Item không có metadata phải trả về list rỗng, không crash."""
        cb = ContentBasedRecommender().fit(self.ml.movies, self.train, self.ml.n_users, self.ml.n_items)
        result = cb.item_similarity(item_id=999999, top_k=5)
        self.assertEqual(result, [])

    def test_hybrid_cold_user_returns_recs(self):
        """User chưa tồn tại (cold-start hoàn toàn) vẫn phải có gợi ý."""
        cf = SVDRecommender(n_factors=20).fit(self.R)
        cb = ContentBasedRecommender().fit(self.ml.movies, self.train, self.ml.n_users, self.ml.n_items)
        hy = HybridRecommender(cf, cb, alpha=0.8)
        recs = hy.recommend(user_id=99999, top_k=10)
        self.assertEqual(len(recs), 10)
        modes = {m for _, _, m in recs}
        self.assertTrue(any("cold" in m for m in modes))

    def test_hybrid_active_user(self):
        cf = SVDRecommender(n_factors=20).fit(self.R)
        cb = ContentBasedRecommender().fit(self.ml.movies, self.train, self.ml.n_users, self.ml.n_items)
        hy = HybridRecommender(cf, cb, alpha=0.8)
        recs = hy.recommend(user_id=1, top_k=10)
        self.assertEqual(len(recs), 10)
        # Active user phải chạy chế độ "hybrid"
        modes = {m for _, _, m in recs}
        self.assertIn("hybrid", modes)

    def test_no_duplicate_recommendations(self):
        cf = SVDRecommender(n_factors=20).fit(self.R)
        cb = ContentBasedRecommender().fit(self.ml.movies, self.train, self.ml.n_users, self.ml.n_items)
        hy = HybridRecommender(cf, cb, alpha=0.8)
        recs = hy.recommend(user_id=10, top_k=20)
        ids = [i for i, _, _ in recs]
        self.assertEqual(len(ids), len(set(ids)), "Co item bi trung trong goi y")

    def test_split_no_leakage(self):
        train_pairs = set(zip(self.train["user_id"], self.train["item_id"]))
        test_pairs = set(zip(self.test["user_id"], self.test["item_id"]))
        self.assertEqual(len(train_pairs & test_pairs), 0, "Train/test bi ro ri")

    def test_minimum_users_in_test(self):
        self.assertGreater(self.test["user_id"].nunique(), 900)


class TestSparseUser(unittest.TestCase):
    """User có rất ít rating (sparse) - kịch bản bán-cold."""

    def test_sparse_user_uses_pop_blend(self):
        ml = load()
        # Cắt train: user 1 chỉ giữ lại 2 rating
        u1 = ml.ratings[ml.ratings["user_id"] == 1].head(2)
        others = ml.ratings[ml.ratings["user_id"] != 1]
        train = pd.concat([others, u1]).reset_index(drop=True)
        R = build_user_item_matrix(train, ml.n_users, ml.n_items)
        cf = SVDRecommender(n_factors=20).fit(R)
        cb = ContentBasedRecommender().fit(ml.movies, train, ml.n_users, ml.n_items)
        hy = HybridRecommender(cf, cb, alpha=0.9)
        recs = hy.recommend(user_id=1, top_k=10)
        modes = {m for _, _, m in recs}
        self.assertIn("sparse_user", modes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
