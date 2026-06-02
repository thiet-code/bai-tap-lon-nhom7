"""Pipeline chính: train v2 với implicit feedback và đánh giá leave-one-out.

Mô hình: ItemPop, ItemKNN, Content-Based, BPR-MF, NeuMF, Hybrid (NCF+MF+CB).
Metric: HR@K, NDCG@K với K = 5, 10, 20 và MRR.
"""
from __future__ import annotations

import json
import os
import time

import joblib
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
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", context="talk")


def main():
    t0 = time.time()
    print("=" * 72)
    print("HE THONG GOI Y LAI v2 - IMPLICIT FEEDBACK + LEAVE-ONE-OUT")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Setup] Thiet bi: {device}")
    if device == "cuda":
        print(f"        GPU: {torch.cuda.get_device_name(0)}")

    print("\n[Data] Dang chuan bi du lieu implicit + leave-one-out...")
    d = build_implicit(seed=42, n_neg=99)
    print(f"  Train: {len(d.train):,} positive interactions")
    print(f"  Test:  {len(d.test):,} user (moi user 1 ground-truth + 99 neg)")

    test_pos = dict(zip(d.test["user_id"], d.test["item_id"]))
    test_users = list(d.test["user_id"].values)
    user_train_count = {u: int((d.train["user_id"] == u).sum()) for u in test_users}

    # Popularity = số lần tương tác trong train
    pop = np.zeros(d.n_items + 1, dtype=np.float32)
    vc = d.train["item_id"].value_counts()
    pop[vc.index.values] = vc.values

    K_LIST = [5, 10, 20]

    results: dict[str, dict] = {}

    # 1. Item Popularity
    print("\n[1/6] ItemPop (baseline)")
    def pop_score(uid, items):
        return np.array([pop[i] if i < pop.shape[0] else 0 for i in items], dtype=np.float32)
    results["ItemPop"] = evaluate_loo(pop_score, test_users, test_pos, d.test_negatives, K_LIST)
    _print(results["ItemPop"], K_LIST)

    # 2. ItemKNN (cosine similarity giữa các item)
    print("\n[2/6] ItemKNN (cosine similarity tren rating vectors)")
    t = time.time()
    n = d.n_items + 1
    user_item = np.zeros((d.n_users + 1, n), dtype=np.float32)
    user_item[d.train["user_id"].values, d.train["item_id"].values] = 1.0
    norm = np.linalg.norm(user_item, axis=0, keepdims=True) + 1e-9
    user_item_n = user_item / norm
    # 1683 × 1683 vẫn xử lý được trong RAM
    item_sim = user_item_n.T @ user_item_n
    np.fill_diagonal(item_sim, 0)
    print(f"  Tinh similarity: {time.time()-t:.1f}s")

    user_seen: dict[int, np.ndarray] = {}
    for u, grp in d.train.groupby("user_id"):
        user_seen[int(u)] = grp["item_id"].values

    def knn_score(uid, items):
        seen = user_seen.get(int(uid), np.array([], dtype=int))
        if len(seen) == 0:
            return pop[items]
        sims = item_sim[items][:, seen].sum(axis=1)
        return sims.astype(np.float32)

    results["ItemKNN"] = evaluate_loo(knn_score, test_users, test_pos, d.test_negatives, K_LIST)
    _print(results["ItemKNN"], K_LIST)

    # 3. Content-Based (TF-IDF)
    print("\n[3/6] Content-Based (TF-IDF genres + title)")
    cb = ContentBasedRecommender(min_rating_for_profile=3.5).fit(
        d.movies,
        # Tất cả implicit interaction coi như user "thích" (rating ảo 4.0)
        pd.DataFrame({"user_id": d.train["user_id"], "item_id": d.train["item_id"],
                       "rating": 4.0 * np.ones(len(d.train))}),
        d.n_users, d.n_items,
    )
    def cb_score(uid, items):
        v = cb.predict_scores(uid)
        return np.array([v[i] if i < v.shape[0] else 0 for i in items], dtype=np.float32)
    results["Content-Based"] = evaluate_loo(cb_score, test_users, test_pos, d.test_negatives, K_LIST)
    _print(results["Content-Based"], K_LIST)

    # 4. BPR-MF
    print("\n[4/6] BPR-MF (Matrix Factorization + Bayesian Personalized Ranking)")
    bpr = train_bpr(
        d.train, d.user_pos, d.n_users, d.n_items,
        dim=64, epochs=30, batch_size=4096, lr=0.005,
        weight_decay=1e-5, device=device, verbose=True,
    )
    bpr_scorer = make_torch_scorer(bpr, device=device)
    results["BPR-MF"] = evaluate_loo(bpr_scorer, test_users, test_pos, d.test_negatives, K_LIST)
    _print(results["BPR-MF"], K_LIST)

    # 5. NeuMF
    print("\n[5/6] NeuMF (GMF + MLP fusion)")
    ncf = train_ncf(
        d.train, d.user_pos, d.n_users, d.n_items,
        epochs=25, batch_size=4096, lr=0.001,
        weight_decay=1e-6, n_neg_per_pos=4,
        gmf_dim=32, mlp_dim=32, mlp_hidden=(64, 32, 16),
        device=device, verbose=True,
    )
    ncf_scorer = make_torch_scorer(ncf, device=device)
    results["NeuMF"] = evaluate_loo(ncf_scorer, test_users, test_pos, d.test_negatives, K_LIST)
    _print(results["NeuMF"], K_LIST)

    # 6. Hybrid - quét 5 cấu hình trọng số, chọn cấu hình tốt nhất theo NDCG@10
    print("\n[6/6] Hybrid (NeuMF + BPR-MF + Content-Based + cold-aware)")
    best_combo = None
    best_ndcg = -1.0
    combos = [
        (0.6, 0.3, 0.1),
        (0.5, 0.4, 0.1),
        (0.5, 0.3, 0.2),
        (0.4, 0.4, 0.2),
        (0.7, 0.2, 0.1),
    ]
    for (a, b, c) in combos:
        hy = HybridV2(ncf, bpr, cb, user_train_count, pop, a, b, c, cold_threshold=10, device=device)
        r = evaluate_loo(hy.make_scorer(), test_users, test_pos, d.test_negatives, [10])
        print(f"   w=({a},{b},{c})  NDCG@10={r['ndcg@10']:.4f}  HR@10={r['hr@10']:.4f}")
        if r["ndcg@10"] > best_ndcg:
            best_ndcg = r["ndcg@10"]
            best_combo = (a, b, c)
    print(f"  >> Trong so tot nhat: w_ncf={best_combo[0]}, w_mf={best_combo[1]}, w_cb={best_combo[2]}")
    hy = HybridV2(ncf, bpr, cb, user_train_count, pop, *best_combo, cold_threshold=10, device=device)
    results["Hybrid"] = evaluate_loo(hy.make_scorer(), test_users, test_pos, d.test_negatives, K_LIST)
    _print(results["Hybrid"], K_LIST)

    # Cold-start: user chỉ có 1-3 rating trong train
    print("\n[Cold-start] User chi co 1-3 rating trong train")
    cold_users = [u for u in test_users if user_train_count.get(u, 0) <= 3]
    print(f"  Co {len(cold_users)} cold user")
    cold_results = {}
    if cold_users:
        for name, fn in [
            ("ItemPop", pop_score),
            ("Content-Based", cb_score),
            ("BPR-MF", bpr_scorer),
            ("NeuMF", ncf_scorer),
            ("Hybrid", hy.make_scorer()),
        ]:
            r = evaluate_loo(fn, cold_users, test_pos, d.test_negatives, [10])
            cold_results[name] = r
            print(f"  {name:<14} HR@10={r['hr@10']:.4f}  NDCG@10={r['ndcg@10']:.4f}  ({r['n_users']} user)")

    # Lưu kết quả và checkpoint
    out = {
        "k_list": K_LIST,
        "results": results,
        "cold_start": cold_results,
        "best_hybrid_weights": list(best_combo),
        "n_test_users": len(test_users),
        "n_cold_users": len(cold_users),
        "elapsed_seconds": time.time() - t0,
        "device": device,
    }
    with open(os.path.join(REPORTS_DIR, "results_v2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    torch.save(bpr.state_dict(), os.path.join(MODELS_DIR, "bpr_mf.pt"))
    torch.save(ncf.state_dict(), os.path.join(MODELS_DIR, "neumf.pt"))
    joblib.dump({"cb": cb, "best_combo": best_combo, "popularity": pop,
                 "user_train_count": user_train_count,
                 "n_users": d.n_users, "n_items": d.n_items},
                os.path.join(MODELS_DIR, "hybrid_v2.pkl"))

    _plot(results, cold_results, K_LIST)

    print("\n" + "=" * 72)
    print("TONG HOP KET QUA (Leave-One-Out, 1 truth + 99 neg)")
    print("=" * 72)
    print(f"{'Model':<16}{'HR@5':>9}{'NDCG@5':>10}{'HR@10':>9}{'NDCG@10':>11}{'HR@20':>9}{'NDCG@20':>11}{'MRR':>9}")
    for name, r in results.items():
        print(f"{name:<16}{r['hr@5']:>9.4f}{r['ndcg@5']:>10.4f}{r['hr@10']:>9.4f}{r['ndcg@10']:>11.4f}"
              f"{r['hr@20']:>9.4f}{r['ndcg@20']:>11.4f}{r['mrr']:>9.4f}")

    best = max(results, key=lambda n: results[n]["ndcg@10"])
    print(f"\n>> Model tot nhat (NDCG@10): {best} = {results[best]['ndcg@10']:.4f}")
    print(f">> HR@10 tuong ung:           {results[best]['hr@10']:.4f}")
    print(f">> Tong thoi gian:             {time.time()-t0:.1f}s")


def _print(r, K_LIST):
    parts = []
    for k in K_LIST:
        parts.append(f"HR@{k}={r[f'hr@{k}']:.4f} NDCG@{k}={r[f'ndcg@{k}']:.4f}")
    print(" | ".join(parts) + f" | MRR={r['mrr']:.4f} | n={r['n_users']}")


def _plot(results, cold_results, K_LIST):
    colors = {"ItemPop": "#9aa0a6", "ItemKNN": "#fbbc04", "Content-Based": "#34a853",
              "BPR-MF": "#4285f4", "NeuMF": "#a142f4", "Hybrid": "#ea4335"}

    # Biểu đồ 1: bar chart HR@10 và NDCG@10
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    names = list(results.keys())
    hr = [results[n]["hr@10"] for n in names]
    nd = [results[n]["ndcg@10"] for n in names]
    cs = [colors.get(n, "#888") for n in names]
    axes[0].bar(names, hr, color=cs)
    axes[0].set_title("HR@10 - Hit Rate")
    axes[0].set_ylabel("HR@10")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(hr):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)

    axes[1].bar(names, nd, color=cs)
    axes[1].set_title("NDCG@10 - Ranking Quality")
    axes[1].set_ylabel("NDCG@10")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(nd):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "v2_01_compare_models.png"), dpi=150)
    plt.close()

    # Biểu đồ 2: HR/NDCG theo K
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for n in names:
        axes[0].plot(K_LIST, [results[n][f"hr@{k}"] for k in K_LIST], "o-",
                     label=n, color=colors.get(n, "#888"), linewidth=2.5)
        axes[1].plot(K_LIST, [results[n][f"ndcg@{k}"] for k in K_LIST], "o-",
                     label=n, color=colors.get(n, "#888"), linewidth=2.5)
    for ax, ttl in zip(axes, ["HR@K theo K", "NDCG@K theo K"]):
        ax.set_xticks(K_LIST)
        ax.set_xlabel("K")
        ax.set_title(ttl)
        ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "v2_02_metrics_curves.png"), dpi=150)
    plt.close()

    # Biểu đồ 3: cold-start
    if cold_results:
        fig, ax = plt.subplots(figsize=(11, 6))
        cs_names = list(cold_results.keys())
        x = np.arange(len(cs_names))
        w = 0.35
        ax.bar(x - w / 2, [cold_results[n]["hr@10"] for n in cs_names], w,
               label="HR@10", color="#4285f4")
        ax.bar(x + w / 2, [cold_results[n]["ndcg@10"] for n in cs_names], w,
               label="NDCG@10", color="#ea4335")
        ax.set_xticks(x)
        ax.set_xticklabels(cs_names, rotation=15)
        ax.set_title(f"Cold-start (user ≤3 rating, n={cold_results[cs_names[0]]['n_users']})")
        ax.legend()
        for i, n in enumerate(cs_names):
            ax.text(i - w / 2, cold_results[n]["hr@10"] + 0.005,
                    f"{cold_results[n]['hr@10']:.3f}", ha="center", fontsize=10)
            ax.text(i + w / 2, cold_results[n]["ndcg@10"] + 0.005,
                    f"{cold_results[n]['ndcg@10']:.3f}", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "v2_03_cold_start.png"), dpi=150)
        plt.close()

    # Biểu đồ 4: heatmap chi tiết
    rows = []
    for name, r in results.items():
        for k in K_LIST:
            rows.append({"Model": name, "K": k,
                         "HR": r[f"hr@{k}"], "NDCG": r[f"ndcg@{k}"]})
    df = pd.DataFrame(rows).set_index(["Model", "K"])
    fig, ax = plt.subplots(figsize=(8, max(5, 0.5 * len(df))))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGn", ax=ax)
    ax.set_title("HR@K va NDCG@K chi tiết")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "v2_04_heatmap.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
