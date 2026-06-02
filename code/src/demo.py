"""CLI demo phiên bản v1: nhập user_id (hoặc --new để mô phỏng user mới) và in top-N phim."""
from __future__ import annotations

import argparse
import os
import sys

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_loader import load  # noqa: E402
from hybrid import HybridRecommender  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Recommender Demo - MovieLens 100K"
    )
    parser.add_argument("--user", type=int, default=1, help="ID user (1..943)")
    parser.add_argument("--topk", type=int, default=10, help="So phim goi y")
    parser.add_argument("--new", action="store_true",
                        help="Mo phong user moi (chua co rating nao)")
    parser.add_argument("--similar-to", type=int, default=None,
                        help="Tim phim tuong tu voi item_id (cold-item demo)")
    args = parser.parse_args()

    model_path = os.path.join(ROOT, "models", "hybrid_model.pkl")
    if not os.path.exists(model_path):
        print(f"Khong tim thay {model_path}. Chay 'py src/train.py' truoc.")
        sys.exit(1)

    print("Dang nap model...")
    pkg = joblib.load(model_path)
    cf, cb, alpha = pkg["cf"], pkg["cb"], pkg["alpha"]
    hy = HybridRecommender(cf, cb, alpha=alpha)
    ml = load()
    title_map = ml.movies.set_index("item_id")[["title", "genres"]]

    print(f"Model: SVD-CF (k={cf.n_factors}) + Content-Based + Hybrid (alpha={alpha})")
    print(f"Co so du lieu: {ml.n_users} user, {ml.n_items} phim\n")

    # Demo 1: tìm phim tương tự (cold-item)
    if args.similar_to is not None:
        iid = args.similar_to
        if iid not in title_map.index:
            print(f"Khong tim thay phim {iid}")
            sys.exit(1)
        info = title_map.loc[iid]
        print(f"Tim {args.topk} phim tuong tu voi:")
        print(f"  [{iid}] {info['title']}  ({info['genres']})\n")
        for sim_id, sim in cb.item_similarity(iid, top_k=args.topk):
            t = title_map.loc[sim_id]
            print(f"  {sim_id:>5}  sim={sim:.3f}  {t['title']:<45} ({t['genres']})")
        return

    # Demo 2: gợi ý cho user
    user_id = 99999 if args.new else args.user

    if not args.new:
        u_train_count = int(cf.train_mask[user_id].sum()) if user_id < cf.train_mask.shape[0] else 0
        print(f">> User #{user_id}  ({u_train_count} rating trong train)")
        user_ratings = ml.ratings[(ml.ratings["user_id"] == user_id) & (ml.ratings["rating"] >= 4)]
        if len(user_ratings) > 0:
            print("Top 5 phim user da thich nhat:")
            for r in user_ratings.sort_values("rating", ascending=False).head(5).itertuples():
                t = title_map.loc[r.item_id]
                print(f"  [{r.rating}/5] {t['title']:<45} ({t['genres']})")
            print()
    else:
        print(">> User MOI (chua co rating nao trong he thong)\n")

    print(f"Top-{args.topk} goi y (Hybrid alpha={alpha}):")
    recs = hy.recommend(user_id=user_id, top_k=args.topk, exclude_seen=True)
    if not recs:
        print("  (khong co goi y)")
        return
    print(f"{'#':>3}  {'ItemID':>6}  {'Score':>7}  {'Mode':<18}  Title")
    for rank, (iid, sc, mode) in enumerate(recs, 1):
        if iid in title_map.index:
            t = title_map.loc[iid]
            line = f"{t['title']:<45} ({t['genres']})"
        else:
            line = "(unknown)"
        print(f"{rank:>3}  {iid:>6}  {sc:>7.3f}  {mode:<18}  {line}")


if __name__ == "__main__":
    main()
