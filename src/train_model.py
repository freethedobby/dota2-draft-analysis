"""Train a win-prediction model on the fetched matches.

Features per match (all from radiant's perspective):
  - hero_radiant[h]:  +1 if radiant picked h
  - hero_dire[h]:     +1 if dire picked h
  - synergy_r[h1,h2]: +1 if both on radiant
  - synergy_d[h1,h2]: +1 if both on dire
  - counter[h_r,h_d]: +1 for each cross-team pair (radiant h_r vs dire h_d)

L2-regularized logistic regression. Symmetric features mean coefficients are interpretable:
  - hero_radiant - hero_dire = team-side-independent hero strength
  - synergy_r/d coefficients = same-team pair effect
  - counter coefficient = directed matchup (h_r favoured if positive)
"""

import argparse
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_heroes():
    heroes = json.load((DATA / "heroes.json").open())
    id_to_name = {h["id"]: h["localized_name"] for h in heroes}
    ids = sorted(id_to_name)
    idx = {hid: i for i, hid in enumerate(ids)}  # hero_id -> dense index [0..N-1]
    return ids, idx, id_to_name


def parse_team(s):
    return [int(x) for x in s.split(",")]


def build_features(df, idx):
    """Return CSR matrix and feature-index metadata.

    Layout (dense indices, N = num heroes):
        [0, N)              hero_radiant
        [N, 2N)             hero_dire
        [2N, 2N+P)          synergy_radiant   (P = N*(N-1)/2 unordered pairs)
        [2N+P, 2N+2P)       synergy_dire
        [2N+2P, 2N+2P+N*N)  counter[r,d]      (ordered: row=radiant, col=dire)
    """
    N = len(idx)
    P = N * (N - 1) // 2
    pair_id = {}  # (i,j) with i<j -> pair index [0..P)
    for i, j in combinations(range(N), 2):
        pair_id[(i, j)] = len(pair_id)

    offset_hr = 0
    offset_hd = N
    offset_sr = 2 * N
    offset_sd = 2 * N + P
    offset_ct = 2 * N + 2 * P
    n_feat = 2 * N + 2 * P + N * N

    rows, cols = [], []
    y = []
    out_row = 0  # only increments for valid matches; X.shape[0] == len(y)

    for _, row in df.iterrows():
        r = [idx[h] for h in parse_team(row["radiant_team"]) if h in idx]
        d = [idx[h] for h in parse_team(row["dire_team"]) if h in idx]
        if len(r) != 5 or len(d) != 5:
            continue
        if len(set(r)) != 5 or len(set(d)) != 5 or set(r) & set(d):
            continue  # duplicate hero on a side or hero on both sides
        for h in r:
            rows.append(out_row); cols.append(offset_hr + h)
        for h in d:
            rows.append(out_row); cols.append(offset_hd + h)
        for a, b in combinations(sorted(r), 2):
            rows.append(out_row); cols.append(offset_sr + pair_id[(a, b)])
        for a, b in combinations(sorted(d), 2):
            rows.append(out_row); cols.append(offset_sd + pair_id[(a, b)])
        for hr in r:
            for hd in d:
                rows.append(out_row); cols.append(offset_ct + hr * N + hd)
        y.append(int(row["radiant_win"]))
        out_row += 1

    data = np.ones(len(rows), dtype=np.float32)
    X = csr_matrix((data, (rows, cols)), shape=(out_row, n_feat), dtype=np.float32)
    y = np.array(y, dtype=np.int8)
    meta = {
        "N": N, "P": P,
        "offset_hr": offset_hr, "offset_hd": offset_hd,
        "offset_sr": offset_sr, "offset_sd": offset_sd, "offset_ct": offset_ct,
        "n_feat": n_feat, "pair_id": pair_id,
    }
    return X, y, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=0.5, help="inverse L2 regularization strength")
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--test-size", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(DATA / "matches.csv")
    print(f"loaded {len(df)} matches")
    df = df.drop_duplicates(subset=["match_id"])
    print(f"deduped: {len(df)}")

    ids, idx, id_to_name = load_heroes()
    print(f"{len(ids)} heroes")

    X, y, meta = build_features(df, idx)
    print(f"feature matrix: {X.shape}, nnz={X.nnz:,}, base radiant winrate={y.mean():.4f}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=42)
    print(f"train={Xtr.shape[0]}, test={Xte.shape[0]}")

    clf = LogisticRegression(
        C=args.C, solver="liblinear", penalty="l2", max_iter=args.max_iter, verbose=0,
    )
    clf.fit(Xtr, ytr)
    p_tr = clf.predict_proba(Xtr)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]
    print(f"train: logloss={log_loss(ytr, p_tr):.4f}  auc={roc_auc_score(ytr, p_tr):.4f}")
    print(f"test : logloss={log_loss(yte, p_te):.4f}  auc={roc_auc_score(yte, p_te):.4f}")
    print(f"(baseline logloss for {y.mean():.3f} prior: "
          f"{log_loss(yte, np.full_like(yte, y.mean(), dtype=float)):.4f})")

    out = DATA / "model.pkl"
    with out.open("wb") as f:
        pickle.dump({"clf": clf, "meta": meta, "hero_ids": ids, "id_to_name": id_to_name}, f)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
