"""CLI demo phiên bản v2 (BPR-MF + NeuMF + Content-Based + Hybrid)."""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_loader import load  # noqa: E402
from hybrid_v2 import HybridV2  # noqa: E402
from models_nn import BPRMF, NeuMF  # noqa: E402


def main():
    parser = argparse.ArgumentParser("Hybrid Recommender Demo v2 - MovieLens 100K")
    parser.add_argument("--user", type=int, default=1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--model", choices=["hybrid", "ncf", "mf"], default="hybrid")
    args = parser.parse_args()

    pkg_path = os.path.join(ROOT, "models", "hybrid_v2.pkl")
    if not os.path.exists(pkg_path):
        print(f"Khong tim thay {pkg_path}. Chay 'py src/train_v2.py' truoc.")
        sys.exit(1)

    pkg = joblib.load(pkg_path)
    cb = pkg["cb"]
    pop = pkg["popularity"]
    user_train_count = pkg["user_train_count"]
    n_users, n_items = pkg["n_users"], pkg["n_items"]
    best_combo = pkg["best_combo"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bpr = BPRMF(n_users, n_items, dim=64).to(device)
    bpr.load_state_dict(torch.load(os.path.join(ROOT, "models", "bpr_mf.pt"), map_location=device))
    bpr.eval()
    ncf = NeuMF(n_users, n_items).to(device)
    ncf.load_state_dict(torch.load(os.path.join(ROOT, "models", "neumf.pt"), map_location=device))
    ncf.eval()

    hy = HybridV2(ncf, bpr, cb, user_train_count, pop, *best_combo,
                   cold_threshold=10, device=str(device))

    ml = load()
    title_map = ml.movies.set_index("item_id")[["title", "genres"]]

    print(f"Model dang dung: {args.model}  |  Device: {device}")
    print(f"Hybrid weights (best): w_ncf={best_combo[0]}, w_mf={best_combo[1]}, w_cb={best_combo[2]}\n")

    user_id = args.user
    n_train = user_train_count.get(user_id, 0)
    print(f">> User #{user_id} - co {n_train} rating trong train")

    user_rates = ml.ratings[(ml.ratings["user_id"] == user_id) & (ml.ratings["rating"] >= 4)]
    if len(user_rates) > 0:
        print("Mau 5 phim user thich (rating >= 4):")
        for r in user_rates.sort_values("rating", ascending=False).head(5).itertuples():
            t = title_map.loc[r.item_id]
            print(f"  [{r.rating}/5] {t['title']:<45} ({t['genres']})")
        print()

    # Loại các item user đã rate, score tất cả ứng viên còn lại rồi rank
    all_items = list(range(1, n_items + 1))
    seen = set(int(i) for i in ml.ratings[ml.ratings["user_id"] == user_id]["item_id"])
    candidates = [i for i in all_items if i not in seen]

    if args.model == "hybrid":
        scores = hy.score(user_id, candidates)
    elif args.model == "ncf":
        with torch.no_grad():
            u_t = torch.full((len(candidates),), user_id, dtype=torch.long, device=device)
            i_t = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = ncf(u_t, i_t).cpu().numpy()
    else:
        with torch.no_grad():
            u_t = torch.full((len(candidates),), user_id, dtype=torch.long, device=device)
            i_t = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = bpr.score(u_t, i_t).cpu().numpy()

    order = np.argsort(-scores)[:args.topk]
    print(f"Top-{args.topk} goi y (model={args.model}):")
    print(f"{'#':>3}  {'Score':>7}  Title")
    for rank, idx in enumerate(order, 1):
        iid = candidates[idx]
        if iid in title_map.index:
            t = title_map.loc[iid]
            line = f"{t['title']:<45} ({t['genres']})"
        else:
            line = "(unknown)"
        print(f"{rank:>3}  {scores[idx]:>7.3f}  {line}")


if __name__ == "__main__":
    main()
