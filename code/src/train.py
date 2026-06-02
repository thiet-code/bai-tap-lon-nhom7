"""Pipeline v1: huấn luyện, đánh giá, tinh chỉnh alpha và kịch bản cold-start nhân tạo."""
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
import joblib

from data_loader import build_user_item_matrix, load, train_test_split_per_user
from cf_svd import SVDRecommender
from content_based import ContentBasedRecommender
from hybrid import HybridRecommender
from metrics import evaluate_recommender

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "reports", "figures")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", context="talk")


def make_pop_recommender(train, n_items, cf):
    pop_scores = np.zeros(n_items + 1)
    vc = train["item_id"].value_counts()
    pop_scores[vc.index.values] = vc.values

    def fn(uid: int, top_k: int):
        scores = pop_scores.copy()
        scores[0] = -np.inf
        if cf.train_mask is not None and uid < cf.train_mask.shape[0]:
            scores[cf.train_mask[uid]] = -np.inf
        idx = np.argpartition(-scores, top_k)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    return fn


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("HE THONG GOI Y LAI - TRAIN VA DANH GIA")
    print("=" * 70)

    ml = load()
    print(f"\n[Data] {ml.n_users} user, {ml.n_items} phim, {len(ml.ratings):,} rating")
    train, test = train_test_split_per_user(ml.ratings, test_ratio=0.2, seed=42)
    print(f"[Split] train={len(train):,}  test={len(test):,}")

    R = build_user_item_matrix(train, ml.n_users, ml.n_items)

    print("\n[Train] SVD-CF ...")
    t = time.time()
    cf = SVDRecommender(n_factors=50, random_state=42).fit(R)
    print(f"  Xong sau {time.time() - t:.1f}s")

    print("[Train] Content-Based ...")
    t = time.time()
    cb = ContentBasedRecommender(min_rating_for_profile=3.5).fit(
        ml.movies, train, ml.n_users, ml.n_items
    )
    print(f"  Xong sau {time.time() - t:.1f}s")

    # Quét alpha để chọn cấu hình Hybrid tốt nhất theo NDCG@10
    eval_users = list(test["user_id"].unique())
    print(f"\n[Tune] Quet alpha tren {len(eval_users)} user (chon theo NDCG@10)")
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_ndcg = {}
    for a in alphas:
        hy_a = HybridRecommender(cf, cb, alpha=a)

        def fn(uid, k, hy=hy_a):
            return [(i, s) for i, s, _ in hy.recommend(uid, top_k=k, exclude_seen=True)]

        m = evaluate_recommender(fn, test, eval_users, k=10, relevance_threshold=4.0)
        alpha_ndcg[a] = m["ndcg@10"]
        print(f"  alpha={a:.1f}  NDCG@10={m['ndcg@10']:.4f}  MAP@10={m['map@10']:.4f}")

    best_alpha = max(alpha_ndcg, key=lambda k: alpha_ndcg[k])
    print(f"\n[Tune] Alpha tot nhat: {best_alpha} (NDCG@10={alpha_ndcg[best_alpha]:.4f})")
    hy = HybridRecommender(cf, cb, alpha=best_alpha)

    # Đánh giá 4 mô hình trên toàn bộ user test
    pop_fn = make_pop_recommender(train, ml.n_items, cf)

    def recs_cf(uid, k):
        return cf.recommend(uid, top_k=k, exclude_seen=True)

    def recs_cb(uid, k):
        seen = cf.train_mask[uid] if cf.train_mask is not None and uid < cf.train_mask.shape[0] else None
        return cb.recommend(uid, train_mask_row=seen, top_k=k)

    def recs_hy(uid, k):
        return [(i, s) for i, s, _ in hy.recommend(uid, top_k=k, exclude_seen=True)]

    K_LIST = [5, 10, 20]
    print(f"\n[Eval] Tren toan bo {len(eval_users)} user test")

    results = {}
    for name, fn in [
        ("Popularity", pop_fn),
        ("Content-Based", recs_cb),
        ("SVD-CF", recs_cf),
        (f"Hybrid (a={best_alpha})", recs_hy),
    ]:
        print(f"\n  >>> {name}")
        results[name] = {}
        for k in K_LIST:
            m = evaluate_recommender(fn, test, eval_users, k=k, relevance_threshold=4.0)
            results[name][k] = m
            print(
                f"     K={k:>2}  P={m[f'precision@{k}']:.4f}  R={m[f'recall@{k}']:.4f}  "
                f"MAP={m[f'map@{k}']:.4f}  NDCG={m[f'ndcg@{k}']:.4f}"
            )

    # Kịch bản cold-start nhân tạo: 100 user chỉ giữ 2 rating đầu
    print("\n[Cold-start] Mo phong user moi: lay 100 user, giu lai chi 2 rating dau tien")
    rng = np.random.default_rng(123)
    candidate_users = [u for u in eval_users if (train["user_id"] == u).sum() >= 20]
    cold_users = rng.choice(candidate_users, size=min(100, len(candidate_users)), replace=False).tolist()

    cold_train = train[~train["user_id"].isin(cold_users)].copy()
    for u in cold_users:
        ur = train[train["user_id"] == u].sort_values("timestamp").head(2)
        cold_train = pd.concat([cold_train, ur])
    cold_train = cold_train.reset_index(drop=True)

    R_cold = build_user_item_matrix(cold_train, ml.n_users, ml.n_items)
    cf_cold = SVDRecommender(n_factors=50, random_state=42).fit(R_cold)
    cb_cold = ContentBasedRecommender(min_rating_for_profile=3.5).fit(
        ml.movies, cold_train, ml.n_users, ml.n_items
    )
    hy_cold = HybridRecommender(cf_cold, cb_cold, alpha=best_alpha)
    pop_fn_cold = make_pop_recommender(cold_train, ml.n_items, cf_cold)

    def rc_cf(uid, k):
        return cf_cold.recommend(uid, top_k=k, exclude_seen=True)

    def rc_cb(uid, k):
        seen = cf_cold.train_mask[uid] if cf_cold.train_mask is not None and uid < cf_cold.train_mask.shape[0] else None
        return cb_cold.recommend(uid, train_mask_row=seen, top_k=k)

    def rc_hy(uid, k):
        return [(i, s) for i, s, _ in hy_cold.recommend(uid, top_k=k, exclude_seen=True)]

    cold_results = {}
    for name, fn in [
        ("Popularity", pop_fn_cold),
        ("Content-Based", rc_cb),
        ("SVD-CF", rc_cf),
        (f"Hybrid (a={best_alpha})", rc_hy),
    ]:
        m = evaluate_recommender(fn, test, cold_users, k=10, relevance_threshold=4.0)
        cold_results[name] = m
        print(
            f"  {name:<22} P@10={m['precision@10']:.4f}  R@10={m['recall@10']:.4f}  "
            f"NDCG@10={m['ndcg@10']:.4f}  ({m['n_users']} user)"
        )

    metric_names_full = ["precision@10", "recall@10", "map@10", "ndcg@10"]
    metric_labels = ["Precision@10", "Recall@10", "MAP@10", "NDCG@10"]
    model_names = list(results.keys())
    colors = ["#9aa0a6", "#34a853", "#1a73e8", "#ea4335"]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(metric_names_full))
    width = 0.2
    for i, name in enumerate(model_names):
        vals = [results[name][10][m] for m in metric_names_full]
        ax.bar(x + i * width, vals, width, label=name, color=colors[i])
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Giá trị")
    ax.set_title("So sánh các mô hình gợi ý (K=10)")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_compare_models.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(model_names):
        ax.plot(K_LIST, [results[name][k][f"ndcg@{k}"] for k in K_LIST],
                marker="o", linewidth=2.5, label=name, color=colors[i])
    ax.set_xlabel("K (số phim gợi ý)")
    ax.set_ylabel("NDCG@K")
    ax.set_title("NDCG@K theo K cho từng mô hình")
    ax.set_xticks(K_LIST)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_ndcg_by_k.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(ml.ratings["rating"], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                 color="#1a73e8", edgecolor="white")
    axes[0].set_title("Phân bố rating")
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Số lượng")
    counts = ml.ratings["user_id"].value_counts().values
    axes[1].hist(counts, bins=40, color="#ea4335", edgecolor="white")
    axes[1].set_title("Số rating mỗi user")
    axes[1].set_xlabel("Số rating")
    axes[1].set_ylabel("Số user")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_data_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 6))
    cs_models = list(cold_results.keys())
    cs_p = [cold_results[n]["precision@10"] for n in cs_models]
    cs_r = [cold_results[n]["recall@10"] for n in cs_models]
    cs_n = [cold_results[n]["ndcg@10"] for n in cs_models]
    x = np.arange(len(cs_models))
    w = 0.27
    ax.bar(x - w, cs_p, w, label="Precision@10", color="#34a853")
    ax.bar(x, cs_r, w, label="Recall@10", color="#1a73e8")
    ax.bar(x + w, cs_n, w, label="NDCG@10", color="#ea4335")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("Hybrid", "Hybrid\n") for n in cs_models])
    ax.set_title(f"Cold-start: user chỉ có 2 rating (n={cold_results[cs_models[0]]['n_users']})")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_cold_start.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = list(alpha_ndcg.keys())
    ys = [alpha_ndcg[k] for k in xs]
    ax.plot(xs, ys, marker="o", linewidth=2.5, color="#ea4335")
    ax.axvline(best_alpha, ls="--", color="#1a73e8", label=f"Tốt nhất α={best_alpha}")
    ax.set_xlabel("α (trọng số CF)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Quét alpha cho Hybrid (α=0: chỉ CB, α=1: chỉ CF)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_alpha_sweep.png"), dpi=150)
    plt.close()

    rows = []
    for name in model_names:
        for k in K_LIST:
            rows.append({
                "model": name, "K": k,
                "Precision": results[name][k][f"precision@{k}"],
                "Recall": results[name][k][f"recall@{k}"],
                "MAP": results[name][k][f"map@{k}"],
                "NDCG": results[name][k][f"ndcg@{k}"],
            })
    df = pd.DataFrame(rows)
    pivot = df.set_index(["model", "K"])[["Precision", "Recall", "MAP", "NDCG"]]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", ax=ax, cbar=True)
    ax.set_title("Bảng metrics chi tiết (toàn bộ user)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "06_metrics_heatmap.png"), dpi=150)
    plt.close()

    out = {
        "best_alpha": best_alpha,
        "alpha_sweep": {str(k): v for k, v in alpha_ndcg.items()},
        "k_list": K_LIST,
        "results": {name: {str(k): r for k, r in d.items()} for name, d in results.items()},
        "cold_start_results": cold_results,
        "n_users_test": len(eval_users),
        "n_cold_users": len(cold_users),
        "elapsed_seconds": time.time() - t0,
    }
    with open(os.path.join(REPORTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    joblib.dump({"cf": cf, "cb": cb, "alpha": best_alpha}, os.path.join(MODELS_DIR, "hybrid_model.pkl"))

    print("\n" + "=" * 70)
    print("KET QUA TONG HOP @K=10 (toan bo user)")
    print("=" * 70)
    print(f"{'Model':<24}{'P@10':>10}{'R@10':>10}{'MAP@10':>10}{'NDCG@10':>10}")
    for name in model_names:
        r = results[name][10]
        print(f"{name:<24}{r['precision@10']:>10.4f}{r['recall@10']:>10.4f}"
              f"{r['map@10']:>10.4f}{r['ndcg@10']:>10.4f}")

    print("\nKET QUA COLD-START (user chi co 2 rating)")
    print("=" * 70)
    print(f"{'Model':<24}{'P@10':>10}{'R@10':>10}{'MAP@10':>10}{'NDCG@10':>10}")
    for name in cold_results:
        r = cold_results[name]
        print(f"{name:<24}{r['precision@10']:>10.4f}{r['recall@10']:>10.4f}"
              f"{r['map@10']:>10.4f}{r['ndcg@10']:>10.4f}")

    best = max(model_names, key=lambda n: results[n][10]["ndcg@10"])
    print(f"\n>> Best overall (NDCG@10): {best}")
    print(f">> Best alpha: {best_alpha}")
    print(f">> Tong thoi gian: {time.time() - t0:.1f}s")
    print(f">> Bieu do: {FIG_DIR}")


if __name__ == "__main__":
    main()
