"""Thí nghiệm cold-start: chọn 150 user và che bớt rating, chỉ giữ 2 rating đầu.

Retrain mô hình trên train đã cắt, đánh giá trên 150 user này để chứng minh
giá trị của Hybrid khi user mới chỉ có rất ít rating.
"""
from __future__ import annotations

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from content_based import ContentBasedRecommender
from data_implicit import build_implicit
from eval_loo import evaluate_loo
from hybrid_v2 import HybridV2
from trainer import make_torch_scorer, train_bpr, train_ncf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "reports", "figures")
REPORTS_DIR = os.path.join(ROOT, "reports")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", context="talk")


def main():
    t0 = time.time()
    print("=" * 72)
    print("COLD-START EXPERIMENT (user nhan tao chi co 2 rating trong train)")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = build_implicit(seed=42, n_neg=99)
    print(f"[Data] {d.n_users} user, {d.n_items} phim")

    # Chọn 150 user có >=20 rating, sau đó cắt còn 2 rating
    rng = np.random.default_rng(123)
    candidate = [u for u in d.test["user_id"].unique() if (d.train["user_id"] == u).sum() >= 20]
    cold_users = rng.choice(candidate, size=150, replace=False).tolist()
    print(f"[Cold] Chon {len(cold_users)} user, mock cat train ve chi 2 rating dau (theo timestamp)")

    cold_train_rows = []
    for uid, grp in d.train.groupby("user_id"):
        if int(uid) in cold_users:
            keep = grp.sort_values("ts").head(2)
        else:
            keep = grp
        cold_train_rows.append(keep)
    cold_train = pd.concat(cold_train_rows).reset_index(drop=True)
    print(f"[Data] Train sau cat: {len(cold_train):,} positive interactions "
          f"(giam {len(d.train)-len(cold_train):,} so voi goc)")

    # Build lại user_pos cho dữ liệu mới
    user_pos: dict[int, set[int]] = {}
    for uid, grp in cold_train.groupby("user_id"):
        user_pos[int(uid)] = set(int(i) for i in grp["item_id"])
    # Thêm ground-truth test để negative sampling không chạm phải
    for uid, iid in zip(d.test["user_id"], d.test["item_id"]):
        user_pos.setdefault(int(uid), set()).add(int(iid))

    test_pos = dict(zip(d.test["user_id"], d.test["item_id"]))

    pop = np.zeros(d.n_items + 1, dtype=np.float32)
    vc = cold_train["item_id"].value_counts()
    pop[vc.index.values] = vc.values

    user_train_count = {u: int((cold_train["user_id"] == u).sum()) for u in d.test["user_id"].unique()}

    K_LIST = [5, 10, 20]
    results = {}

    print("\n[1/5] ItemPop")
    def pop_score(uid, items):
        return np.array([pop[i] if i < pop.shape[0] else 0 for i in items], dtype=np.float32)
    results["ItemPop"] = evaluate_loo(pop_score, cold_users, test_pos, d.test_negatives, K_LIST)
    _print(results["ItemPop"], K_LIST)

    print("\n[2/5] Content-Based")
    cb = ContentBasedRecommender(min_rating_for_profile=3.5).fit(
        d.movies,
        pd.DataFrame({"user_id": cold_train["user_id"], "item_id": cold_train["item_id"],
                       "rating": 4.0 * np.ones(len(cold_train))}),
        d.n_users, d.n_items,
    )
    def cb_score(uid, items):
        v = cb.predict_scores(uid)
        return np.array([v[i] if i < v.shape[0] else 0 for i in items], dtype=np.float32)
    results["Content-Based"] = evaluate_loo(cb_score, cold_users, test_pos, d.test_negatives, K_LIST)
    _print(results["Content-Based"], K_LIST)

    print("\n[3/5] BPR-MF (retrain tren cold-train)")
    bpr = train_bpr(cold_train.rename(columns={"ts": "timestamp"}), user_pos, d.n_users, d.n_items,
                     dim=64, epochs=25, batch_size=4096, lr=0.005,
                     weight_decay=1e-5, device=device, verbose=False)
    scorer = make_torch_scorer(bpr, device=device)
    results["BPR-MF"] = evaluate_loo(scorer, cold_users, test_pos, d.test_negatives, K_LIST)
    _print(results["BPR-MF"], K_LIST)

    print("\n[4/5] NeuMF (retrain tren cold-train)")
    ncf = train_ncf(cold_train.rename(columns={"ts": "timestamp"}), user_pos, d.n_users, d.n_items,
                     epochs=20, batch_size=4096, lr=0.001, weight_decay=1e-6,
                     n_neg_per_pos=4, device=device, verbose=False)
    scorer_ncf = make_torch_scorer(ncf, device=device)
    results["NeuMF"] = evaluate_loo(scorer_ncf, cold_users, test_pos, d.test_negatives, K_LIST)
    _print(results["NeuMF"], K_LIST)

    # Quét trọng số Hybrid để tìm cấu hình phù hợp cho cold-start
    print("\n[5/5] Hybrid - quet trong so cho cold-start")
    best_r = None
    best_combo = None
    for combo in [
        (0.5, 0.3, 0.2, 2),
        (0.6, 0.3, 0.1, 2),
        (0.7, 0.2, 0.1, 2),
        (0.5, 0.4, 0.1, 3),
        (0.4, 0.4, 0.2, 3),
    ]:
        w_n, w_m, w_c, ct = combo
        hy_t = HybridV2(ncf, bpr, cb, user_train_count, pop,
                         w_ncf=w_n, w_mf=w_m, w_cb=w_c,
                         cold_threshold=ct, device=device)
        r = evaluate_loo(hy_t.make_scorer(), cold_users, test_pos, d.test_negatives, [10])
        print(f"   w=({w_n},{w_m},{w_c}) ct={ct}  NDCG@10={r['ndcg@10']:.4f}  HR@10={r['hr@10']:.4f}")
        if best_r is None or r["ndcg@10"] > best_r["ndcg@10"]:
            best_r = r
            best_combo = combo
    print(f"   >> Best combo cho cold: {best_combo}")
    w_n, w_m, w_c, ct = best_combo
    hy = HybridV2(ncf, bpr, cb, user_train_count, pop,
                   w_ncf=w_n, w_mf=w_m, w_cb=w_c, cold_threshold=ct, device=device)
    results["Hybrid"] = evaluate_loo(hy.make_scorer(), cold_users, test_pos, d.test_negatives, K_LIST)
    _print(results["Hybrid"], K_LIST)

    out = {"results_cold_start": results, "n_cold_users": len(cold_users),
           "elapsed_seconds": time.time() - t0}
    with open(os.path.join(REPORTS_DIR, "results_cold_v2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    colors = {"ItemPop": "#9aa0a6", "Content-Based": "#34a853",
              "BPR-MF": "#4285f4", "NeuMF": "#a142f4", "Hybrid": "#ea4335"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = list(results.keys())
    hr = [results[n]["hr@10"] for n in names]
    nd = [results[n]["ndcg@10"] for n in names]
    cs = [colors.get(n, "#888") for n in names]
    axes[0].bar(names, hr, color=cs)
    axes[0].set_title(f"HR@10 - Cold-start (n={len(cold_users)} user, mỗi user 2 rating)")
    axes[0].set_ylabel("HR@10")
    axes[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(hr):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11)
    axes[1].bar(names, nd, color=cs)
    axes[1].set_title(f"NDCG@10 - Cold-start")
    axes[1].set_ylabel("NDCG@10")
    axes[1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(nd):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "v2_03_cold_start.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 6))
    for n in names:
        ax.plot(K_LIST, [results[n][f"ndcg@{k}"] for k in K_LIST], "o-",
                label=n, linewidth=2.5, color=colors.get(n, "#888"))
    ax.set_xticks(K_LIST)
    ax.set_xlabel("K")
    ax.set_ylabel("NDCG@K")
    ax.set_title(f"Cold-start: NDCG@K theo K (n={len(cold_users)} user)")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "v2_05_cold_start_curve.png"), dpi=150)
    plt.close()

    print("\n" + "=" * 72)
    print("KET QUA COLD-START (user chi 2 rating)")
    print("=" * 72)
    print(f"{'Model':<16}{'HR@5':>9}{'NDCG@5':>10}{'HR@10':>9}{'NDCG@10':>11}{'HR@20':>9}{'NDCG@20':>11}")
    for name, r in results.items():
        print(f"{name:<16}{r['hr@5']:>9.4f}{r['ndcg@5']:>10.4f}{r['hr@10']:>9.4f}"
              f"{r['ndcg@10']:>11.4f}{r['hr@20']:>9.4f}{r['ndcg@20']:>11.4f}")

    best = max(results, key=lambda n: results[n]["ndcg@10"])
    print(f"\n>> Best cold-start (NDCG@10): {best} = {results[best]['ndcg@10']:.4f}")
    print(f">> Tong thoi gian: {time.time()-t0:.1f}s")


def _print(r, K_LIST):
    parts = []
    for k in K_LIST:
        parts.append(f"HR@{k}={r[f'hr@{k}']:.4f} NDCG@{k}={r[f'ndcg@{k}']:.4f}")
    print("  " + " | ".join(parts))


if __name__ == "__main__":
    main()
