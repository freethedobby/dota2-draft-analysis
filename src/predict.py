"""Load the trained model and provide:

  - predict(radiant, dire) -> radiant win probability
  - hero_strengths()        -> ranked solo-strength table
  - top_synergies(k)        -> strongest same-team pairs
  - top_counters(k)         -> strongest cross-team counters

Heroes can be passed by id (int), localized name, or case-insensitive substring.
"""

import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "model.pkl"


class DotaModel:
    def __init__(self, path=MODEL_PATH):
        with Path(path).open("rb") as f:
            blob = pickle.load(f)
        self.clf = blob["clf"]
        self.meta = blob["meta"]
        self.hero_ids = blob["hero_ids"]               # dense_idx -> hero_id
        self.id_to_name = blob["id_to_name"]
        self.idx = {hid: i for i, hid in enumerate(self.hero_ids)}
        self.coef = self.clf.coef_.ravel()
        self.intercept = float(self.clf.intercept_[0])

    # ---------- hero lookup ----------
    def _resolve(self, h):
        if isinstance(h, (int, np.integer)):
            return int(h)
        if isinstance(h, str):
            s = h.strip().lower()
            exact = [hid for hid, n in self.id_to_name.items() if n.lower() == s]
            if exact:
                return exact[0]
            partial = [hid for hid, n in self.id_to_name.items() if s in n.lower()]
            if len(partial) == 1:
                return partial[0]
            if not partial:
                raise ValueError(f"no hero matches {h!r}")
            raise ValueError(f"ambiguous: {h!r} matches {[self.id_to_name[i] for i in partial]}")
        raise TypeError(f"bad hero type: {type(h)}")

    def name(self, hero_id):
        return self.id_to_name[hero_id]

    # ---------- prediction ----------
    def _features_for(self, radiant, dire):
        m = self.meta
        N = m["N"]
        r = [self.idx[self._resolve(h)] for h in radiant]
        d = [self.idx[self._resolve(h)] for h in dire]
        assert len(set(r)) == 5 and len(set(d)) == 5, "need 5 unique heroes per side"
        assert not (set(r) & set(d)), "hero overlap between teams"
        cols = []
        for h in r: cols.append(m["offset_hr"] + h)
        for h in d: cols.append(m["offset_hd"] + h)
        for a, b in combinations(sorted(r), 2):
            cols.append(m["offset_sr"] + m["pair_id"][(a, b)])
        for a, b in combinations(sorted(d), 2):
            cols.append(m["offset_sd"] + m["pair_id"][(a, b)])
        for hr in r:
            for hd in d:
                cols.append(m["offset_ct"] + hr * N + hd)
        rows = np.zeros(len(cols), dtype=np.int32)
        data = np.ones(len(cols), dtype=np.float32)
        return csr_matrix((data, (rows, cols)), shape=(1, m["n_feat"]), dtype=np.float32)

    def predict(self, radiant, dire):
        X = self._features_for(radiant, dire)
        return float(self.clf.predict_proba(X)[0, 1])

    # ---------- analysis ----------
    def hero_strengths(self):
        """Solo strength: (hero_radiant_coef - hero_dire_coef) / 2.

        With perfect mirror symmetry that equals the per-hero log-odds contribution.
        """
        m = self.meta
        N = m["N"]
        hr = self.coef[m["offset_hr"]: m["offset_hr"] + N]
        hd = self.coef[m["offset_hd"]: m["offset_hd"] + N]
        strength = (hr - hd) / 2.0
        rows = [{"hero_id": self.hero_ids[i],
                 "hero": self.id_to_name[self.hero_ids[i]],
                 "strength": float(strength[i]),
                 "coef_radiant": float(hr[i]),
                 "coef_dire": float(hd[i])} for i in range(N)]
        return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)

    def top_synergies(self, k=20, worst=False):
        """Strongest same-team pair effects (averaged radiant+dire)."""
        m = self.meta
        pair_id = m["pair_id"]
        sr = self.coef[m["offset_sr"]: m["offset_sr"] + m["P"]]
        sd = self.coef[m["offset_sd"]: m["offset_sd"] + m["P"]]
        avg = (sr + sd) / 2.0
        rows = []
        for (i, j), pid in pair_id.items():
            rows.append({"hero_a": self.id_to_name[self.hero_ids[i]],
                         "hero_b": self.id_to_name[self.hero_ids[j]],
                         "synergy": float(avg[pid])})
        df = pd.DataFrame(rows).sort_values("synergy", ascending=worst)
        return df.head(k).reset_index(drop=True)

    def top_counters(self, k=20, worst=False):
        """Strongest cross-team counter effects.

        Positive = radiant hero (row) favoured over dire hero (col).
        We symmetrize: counter(a vs b) = (ct[a,b] - ct[b,a]) / 2
        and report the top |value| pairs as "a counters b".
        """
        m = self.meta
        N = m["N"]
        ct = self.coef[m["offset_ct"]: m["offset_ct"] + N * N].reshape(N, N)
        sym = (ct - ct.T) / 2.0
        rows = []
        for i in range(N):
            for j in range(N):
                if i == j: continue
                rows.append({"hero_a": self.id_to_name[self.hero_ids[i]],
                             "hero_b": self.id_to_name[self.hero_ids[j]],
                             "counter": float(sym[i, j])})
        df = pd.DataFrame(rows)
        df = df.sort_values("counter", ascending=worst)
        return df.head(k).reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strengths", action="store_true")
    ap.add_argument("--synergies", action="store_true")
    ap.add_argument("--counters", action="store_true")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--predict", nargs=2, metavar=("RADIANT", "DIRE"),
                    help="two comma-separated hero lists, e.g. 'Anti-Mage,Lina,...' 'Pudge,...'")
    args = ap.parse_args()

    m = DotaModel()
    if args.strengths:
        print("=== TOP 15 SOLO-STRENGTH HEROES ===")
        print(m.hero_strengths().head(args.k).to_string(index=False))
        print("\n=== BOTTOM 15 ===")
        print(m.hero_strengths().tail(args.k).to_string(index=False))
    if args.synergies:
        print("\n=== TOP SYNERGIES (same-team pairs) ===")
        print(m.top_synergies(args.k).to_string(index=False))
        print("\n=== WORST SYNERGIES ===")
        print(m.top_synergies(args.k, worst=True).to_string(index=False))
    if args.counters:
        print("\n=== STRONGEST COUNTERS (a counters b) ===")
        print(m.top_counters(args.k).to_string(index=False))
    if args.predict:
        r = [s.strip() for s in args.predict[0].split(",")]
        d = [s.strip() for s in args.predict[1].split(",")]
        p = m.predict(r, d)
        print(f"\nradiant={r}\ndire   ={d}\nradiant win prob = {p:.4f}")
