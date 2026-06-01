"""Compare feature sets: hero-only vs hero+synergy vs hero+synergy+counter."""

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


def load():
    import json
    df = pd.read_csv(DATA / "matches.csv").drop_duplicates(subset=["match_id"])
    heroes = json.load((DATA / "heroes.json").open())
    id_to_name = {h["id"]: h["localized_name"] for h in heroes}
    ids = sorted(id_to_name)
    idx = {hid: i for i, hid in enumerate(ids)}
    return df, ids, idx, id_to_name


def build(df, idx, use_synergy, use_counter):
    N = len(idx)
    P = N * (N - 1) // 2 if use_synergy else 0
    pair_id = {}
    if use_synergy:
        for i, j in combinations(range(N), 2):
            pair_id[(i, j)] = len(pair_id)

    off_hr = 0
    off_hd = N
    off_sr = 2 * N
    off_sd = 2 * N + P
    off_ct = 2 * N + 2 * P
    n_feat = 2 * N + 2 * P + (N * N if use_counter else 0)

    rows, cols, y = [], [], []
    out = 0
    for _, row in df.iterrows():
        r = [idx[int(h)] for h in row["radiant_team"].split(",") if int(h) in idx]
        d = [idx[int(h)] for h in row["dire_team"].split(",") if int(h) in idx]
        if len(r) != 5 or len(d) != 5: continue
        if len(set(r)) != 5 or len(set(d)) != 5 or set(r) & set(d): continue
        for h in r: rows.append(out); cols.append(off_hr + h)
        for h in d: rows.append(out); cols.append(off_hd + h)
        if use_synergy:
            for a, b in combinations(sorted(r), 2):
                rows.append(out); cols.append(off_sr + pair_id[(a, b)])
            for a, b in combinations(sorted(d), 2):
                rows.append(out); cols.append(off_sd + pair_id[(a, b)])
        if use_counter:
            for hr in r:
                for hd in d:
                    rows.append(out); cols.append(off_ct + hr * N + hd)
        y.append(int(row["radiant_win"]))
        out += 1
    X = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                   shape=(out, n_feat), dtype=np.float32)
    return X, np.array(y, dtype=np.int8)


def evaluate(name, X, y, Cs):
    print(f"\n=== {name}  features={X.shape[1]:,}  nnz/row={X.nnz/X.shape[0]:.1f} ===")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=42)
    base = log_loss(yte, np.full_like(yte, y.mean(), dtype=float))
    print(f"  baseline logloss = {base:.4f}")
    for C in Cs:
        clf = LogisticRegression(C=C, solver="liblinear", penalty="l2", max_iter=400)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        ll = log_loss(yte, p)
        auc = roc_auc_score(yte, p)
        print(f"  C={C:>7.4f}  test logloss={ll:.4f}  AUC={auc:.4f}  Δ={ll-base:+.4f}")


def main():
    df, ids, idx, _ = load()
    print(f"matches: {len(df)}")
    Cs = [0.001, 0.005, 0.02, 0.1, 0.5]
    X1, y1 = build(df, idx, use_synergy=False, use_counter=False)
    evaluate("hero-only", X1, y1, Cs)
    X2, y2 = build(df, idx, use_synergy=True, use_counter=False)
    evaluate("hero + synergy", X2, y2, Cs)
    X3, y3 = build(df, idx, use_synergy=True, use_counter=True)
    evaluate("hero + synergy + counter", X3, y3, Cs)


if __name__ == "__main__":
    main()
