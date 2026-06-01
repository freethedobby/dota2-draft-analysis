"""Minimal Dota 7.41 draft simulator.

Run with:
    streamlit run src/dashboard.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_image_select import image_select

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from predict import DotaModel  # noqa: E402

st.set_page_config(page_title="Dota 7.41 Draft", layout="wide")

PHASES = [
    ("Snowball", "< 33 min", 0, 33 * 60),
    ("Standard", "33–45 min", 33 * 60, 45 * 60),
    ("Late game", "45+ min", 45 * 60, 10**9),
]

# 5-minute slices for the time graph. Last bucket open-ended at the top.
DURATION_BUCKET_MIN = [(0, 25), (25, 30), (30, 35), (35, 40),
                       (40, 45), (45, 50), (50, 55), (55, 1000)]


@st.cache_resource
def load_model():
    return DotaModel()


@st.cache_data
def load_hero_meta():
    heroes = json.load((ROOT / "data" / "heroes.json").open())
    cdn = "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes"
    return {
        h["id"]: {
            "id": h["id"],
            "name": h["localized_name"],
            "primary_attr": h["primary_attr"],
            "portrait": f"{cdn}/{h['name'].replace('npc_dota_hero_', '')}.png",
        }
        for h in heroes
    }


@st.cache_data
def hero_stats():
    """Per-hero overall winrate + per-phase winrate + per-5-min-bucket winrate."""
    df = pd.read_csv(ROOT / "data" / "matches.csv")
    rows = []
    for _, r in df.iterrows():
        rh = [int(x) for x in r.radiant_team.split(",")]
        dh = [int(x) for x in r.dire_team.split(",")]
        for h in rh:
            rows.append((h, r.duration, int(r.radiant_win)))
        for h in dh:
            rows.append((h, r.duration, 1 - int(r.radiant_win)))
    hp = pd.DataFrame(rows, columns=["hero_id", "duration", "won"])
    out = {}
    for hid, sub in hp.groupby("hero_id"):
        d = {"picks": int(len(sub)), "winrate": float(sub.won.mean())}
        for name, _, lo, hi in PHASES:
            mask = (sub.duration >= lo) & (sub.duration < hi)
            n = int(mask.sum())
            d[f"wr_{name}"] = float(sub.loc[mask, "won"].mean()) if n else float("nan")
            d[f"n_{name}"] = n
        buckets = []
        for lo_m, hi_m in DURATION_BUCKET_MIN:
            mask = (sub.duration >= lo_m * 60) & (sub.duration < hi_m * 60)
            n = int(mask.sum())
            wr = float(sub.loc[mask, "won"].mean()) if n else float("nan")
            buckets.append({"lo": lo_m, "hi": hi_m, "n": n, "wr": wr})
        d["buckets"] = buckets
        out[hid] = d
    return out


@st.cache_data
def pair_stats():
    """Empirical same-team pair + cross-team counter stats from the raw matches.

    same[(min,max)] -> {"picks": n, "wins": w}   # n_pair-team games, won
    counter[(r,d)]  -> {"picks": n, "wins": w}   # r on radiant, d on dire, radiant_won
    """
    from collections import defaultdict
    from itertools import combinations
    df = pd.read_csv(ROOT / "data" / "matches.csv")
    same = defaultdict(lambda: [0, 0])
    counter = defaultdict(lambda: [0, 0])
    for _, r in df.iterrows():
        rh = [int(x) for x in r.radiant_team.split(",")]
        dh = [int(x) for x in r.dire_team.split(",")]
        if len(set(rh)) != 5 or len(set(dh)) != 5 or set(rh) & set(dh):
            continue
        rw = int(r.radiant_win)
        for a, b in combinations(rh, 2):
            k = (min(a, b), max(a, b))
            same[k][0] += 1
            same[k][1] += rw
        for a, b in combinations(dh, 2):
            k = (min(a, b), max(a, b))
            same[k][0] += 1
            same[k][1] += 1 - rw
        for hr in rh:
            for hd in dh:
                counter[(hr, hd)][0] += 1
                counter[(hr, hd)][1] += rw
    return ({k: {"picks": v[0], "wins": v[1]} for k, v in same.items()},
            {k: {"picks": v[0], "wins": v[1]} for k, v in counter.items()})


model = load_model()
hero_meta = load_hero_meta()
hstats = hero_stats()
same_pair_stats, counter_pair_stats = pair_stats()
NAME_TO_ID = {m["name"]: m["id"] for m in hero_meta.values()}

if "radiant" not in st.session_state:
    st.session_state.radiant = []
if "dire" not in st.session_state:
    st.session_state.dire = []
if "active_side" not in st.session_state:
    st.session_state.active_side = "Radiant"


def add_hero(hid):
    if hid in st.session_state.radiant or hid in st.session_state.dire:
        return
    side = st.session_state.active_side.lower()
    if len(st.session_state[side]) >= 5:
        other = "dire" if side == "radiant" else "radiant"
        if len(st.session_state[other]) < 5:
            st.session_state[other].append(hid)
            st.session_state.active_side = other.capitalize()
        return
    st.session_state[side].append(hid)
    if len(st.session_state[side]) == 5:
        other = "dire" if side == "radiant" else "radiant"
        if len(st.session_state[other]) < 5:
            st.session_state.active_side = other.capitalize()


def remove_hero(side, hid):
    if hid in st.session_state[side]:
        st.session_state[side].remove(hid)


def reset_draft():
    st.session_state.radiant = []
    st.session_state.dire = []
    st.session_state.active_side = "Radiant"


def quick_add_from_search():
    """Triggered when user presses Enter in the search box. Adds the best-matched
    unpicked hero to the active side and clears the search field."""
    val = st.session_state.get("search_box", "").strip().lower()
    if not val:
        return
    picked = set(st.session_state.radiant) | set(st.session_state.dire)
    attrs = st.session_state.get("attr_box", ["str", "agi", "int", "all"])
    pool = [m for m in hero_meta.values()
            if m["id"] not in picked and m["primary_attr"] in attrs]
    starts = [m for m in pool if m["name"].lower().startswith(val)]
    contains = [m for m in pool if val in m["name"].lower() and m not in starts]
    matches = starts + contains
    if not matches:
        return
    matches.sort(key=lambda m: (
        not m["name"].lower().startswith(val),
        -hstats.get(m["id"], {}).get("picks", 0),
        m["name"],
    ))
    add_hero(matches[0]["id"])
    st.session_state.search_box = ""


# ---------- header ----------
h1, h2 = st.columns([5, 1])
with h1:
    st.title("Dota 7.41 Draft")
with h2:
    st.button("Reset", on_click=reset_draft, width="stretch")

# ---------- picks ----------
r, d = st.columns(2)
for side_key, label, color, col in [("radiant", "Radiant", "green", r),
                                    ("dire", "Dire", "red", d)]:
    with col:
        st.markdown(f"### :{color}[{label}] &nbsp;<small>({len(st.session_state[side_key])}/5)</small>",
                    unsafe_allow_html=True)
        slot_cols = st.columns(5, gap="small")
        for i in range(5):
            with slot_cols[i]:
                if i < len(st.session_state[side_key]):
                    hid = st.session_state[side_key][i]
                    meta = hero_meta[hid]
                    st.image(meta["portrait"], width="stretch")
                    st.button("✕", key=f"rm_{side_key}_{hid}",
                              on_click=remove_hero, args=(side_key, hid), width="stretch")
                else:
                    st.markdown(
                        "<div style='border:2px dashed #333;border-radius:6px;"
                        "aspect-ratio:256/144;display:flex;align-items:center;"
                        "justify-content:center;color:#555;font-size:11px;'>—</div>",
                        unsafe_allow_html=True,
                    )

# ---------- result ----------
st.divider()
if len(st.session_state.radiant) == 5 and len(st.session_state.dire) == 5:
    radiant = [hero_meta[h]["name"] for h in st.session_state.radiant]
    dire = [hero_meta[h]["name"] for h in st.session_state.dire]
    try:
        p = model.predict(radiant, dire)
    except Exception as e:
        st.error(f"prediction failed: {e}")
    else:
        favored = "Radiant" if p >= 0.5 else "Dire"
        prob = p if p >= 0.5 else 1 - p
        c_hex = "#2e8b57" if favored == "Radiant" else "#c0392b"
        st.markdown(
            f"<div style='text-align:center;padding:18px 0 4px;'>"
            f"<div style='font-size:14px;color:#888;'>model prediction</div>"
            f"<div style='font-size:46px;font-weight:700;color:{c_hex};'>{favored} &nbsp; {prob:.1%}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ----- per-side raw winrate + phase profile -----
        def avg(side_ids, key):
            vals = [hstats[h][key] for h in side_ids if hstats.get(h)
                    and not (isinstance(hstats[h][key], float) and pd.isna(hstats[h][key]))]
            return sum(vals) / len(vals) if vals else float("nan")

        r_ids = st.session_state.radiant
        d_ids = st.session_state.dire
        r_wr = avg(r_ids, "winrate")
        d_wr = avg(d_ids, "winrate")
        cR, cD = st.columns(2)
        cR.metric("Radiant avg hero winrate", f"{r_wr:.1%}",
                  delta=f"{(r_wr - 0.5)*100:+.1f} pp")
        cD.metric("Dire avg hero winrate", f"{d_wr:.1%}",
                  delta=f"{(d_wr - 0.5)*100:+.1f} pp")

        st.subheader("Phase advantage by game length")
        st.caption("Each team's **score** at a duration = Σ(hero_winrate − 50%) over the 5 picks, "
                   "in percentage points. Positive = those heroes outperform when the game ends in "
                   "that 5-min window. The gold line is Radiant − Dire — when it's above zero, "
                   "Radiant is favored at that game length.")

        n_b = len(DURATION_BUCKET_MIN)
        xs, x_labels = [], []
        for lo, hi in DURATION_BUCKET_MIN:
            xs.append((lo + min(hi, 60)) / 2)
            x_labels.append(f"<{hi}" if lo == 0 else
                            f"{lo}+" if hi >= 1000 else f"{lo}-{hi}")

        def team_bucket_sum_dev(side_ids):
            """Σ (hero_wr − 0.5) × 100 across 5 heroes, per bucket. NaN-safe."""
            ys, ns = [], []
            for bi in range(n_b):
                tot, n_total = 0.0, 0
                ok = True
                for h in side_ids:
                    b = hstats.get(h, {}).get("buckets", [None] * n_b)[bi]
                    if b is None or pd.isna(b["wr"]):
                        ok = False
                        break
                    tot += (b["wr"] - 0.5) * 100
                    n_total += b["n"]
                ys.append(tot if ok else float("nan"))
                ns.append(n_total)
            return ys, ns

        r_y, r_n = team_bucket_sum_dev(r_ids)
        d_y, d_n = team_bucket_sum_dev(d_ids)
        diff = [(rv - dv) if not (pd.isna(rv) or pd.isna(dv)) else float("nan")
                for rv, dv in zip(r_y, d_y)]

        # ---- single differential line with split shading + spline smoothing ----
        diff_pos = [v if (not pd.isna(v) and v > 0) else 0 for v in diff]
        diff_neg = [v if (not pd.isna(v) and v < 0) else 0 for v in diff]

        fig = go.Figure()
        # green fill where radiant ahead
        fig.add_trace(go.Scatter(
            x=xs, y=diff_pos, mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="rgba(0,0,0,0)", shape="spline", smoothing=1.2),
            fill="tozeroy", fillcolor="rgba(46,139,87,0.35)",
        ))
        # red fill where dire ahead
        fig.add_trace(go.Scatter(
            x=xs, y=diff_neg, mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="rgba(0,0,0,0)", shape="spline", smoothing=1.2),
            fill="tozeroy", fillcolor="rgba(192,57,43,0.35)",
        ))
        # smoothed differential on top
        fig.add_trace(go.Scatter(
            x=xs, y=diff, mode="lines+markers", name="Radiant − Dire",
            line=dict(color="#d4a017", width=3, shape="spline", smoothing=1.2),
            marker=dict(size=9, color="#d4a017",
                        line=dict(color="#1a1a1a", width=1)),
            hovertemplate="%{y:+.1f} pp<extra></extra>",
        ))
        fig.add_hline(y=0, line_width=2, line_color="#666")

        valid = [v for v in diff if not pd.isna(v)]
        if valid:
            pad = max(2, (max(valid) - min(valid)) * 0.2)
            lo, hi = min(valid) - pad, max(valid) + pad
            # keep zero in view
            lo = min(lo, -2); hi = max(hi, 2)
        else:
            lo, hi = -10, 10

        fig.update_layout(
            xaxis=dict(title="Game duration (min)", tickmode="array",
                       tickvals=xs, ticktext=x_labels),
            yaxis=dict(title="Δ score: Radiant − Dire (pp)", range=[lo, hi],
                       zeroline=False, ticksuffix=" pp"),
            height=320, margin=dict(l=10, r=10, t=10, b=40),
            showlegend=False, hovermode="x unified",
        )
        # zero-line annotations to label the shaded regions
        if hi > 0:
            fig.add_annotation(x=xs[0], y=hi * 0.9, text="↑ Radiant favored",
                               showarrow=False, font=dict(color="#2e8b57", size=11),
                               xanchor="left")
        if lo < 0:
            fig.add_annotation(x=xs[0], y=lo * 0.9, text="↓ Dire favored",
                               showarrow=False, font=dict(color="#c0392b", size=11),
                               xanchor="left")
        st.plotly_chart(fig, width="stretch")

        thin = [x_labels[i] for i in range(n_b) if min(r_n[i], d_n[i]) < 100]
        if thin:
            st.caption(f"⚠ noisier buckets (fewer than 100 hero-picks per side): "
                       f"{', '.join(thin)}")
else:
    need = 10 - len(st.session_state.radiant) - len(st.session_state.dire)
    st.caption(f"Pick {need} more hero{'es' if need != 1 else ''}…")

# ---------- picker ----------
st.divider()
side_col, search_col, attr_col, hint_col = st.columns([1.0, 1.4, 1.8, 3.0])
with side_col:
    st.radio("Picking for", ["Radiant", "Dire"], key="active_side", horizontal=True,
             label_visibility="collapsed")
with search_col:
    st.text_input(
        "Search", key="search_box",
        placeholder="🔍 type + Enter to add",
        label_visibility="collapsed",
        on_change=quick_add_from_search,
    )
    search = st.session_state.get("search_box", "").strip().lower()
with attr_col:
    attr_filter = st.multiselect(
        "Attribute", ["str", "agi", "int", "all"],
        default=["str", "agi", "int", "all"],
        format_func=lambda s: {"str": "Str", "agi": "Agi",
                               "int": "Int", "all": "Uni"}[s],
        label_visibility="collapsed",
        key="attr_box",
    )
with hint_col:
    st.markdown(
        "<div style='color:#888;font-size:12px;padding-top:8px;'>"
        "💡 type a fragment (e.g. <code>void</code>) and press Enter — auto-adds "
        "to the active side.</div>",
        unsafe_allow_html=True,
    )

picked_set = set(st.session_state.radiant) | set(st.session_state.dire)
candidates = [m for m in hero_meta.values()
              if m["primary_attr"] in attr_filter
              and (not search or search in m["name"].lower())
              and m["id"] not in picked_set]
candidates.sort(key=lambda m: m["name"])

if not candidates:
    st.info("No heroes match (try clearing the search or adding all attributes).")
else:
    captions = []
    for m in candidates:
        stats = hstats.get(m["id"])
        wr = f"{stats['winrate']:.0%}" if stats else "—"
        captions.append(f"{m['name']} · {wr}")

    # dynamic key resets the widget after each click + on filter change so the
    # same image can be re-clicked and stale selections never cross-contaminate
    n_picked = len(st.session_state.radiant) + len(st.session_state.dire)
    picker_key = f"picker_{n_picked}_{search}_{''.join(sorted(attr_filter))}"

    sel = image_select(
        label="",
        images=[m["portrait"] for m in candidates],
        captions=captions,
        index=-1,
        return_value="index",
        use_container_width=False,
        key=picker_key,
    )
    if sel is not None and sel >= 0:
        add_hero(candidates[sel]["id"])
        st.rerun()

# ---------- footnote ----------
st.divider()
st.caption(
    "Trained on 18k Divine+ public matches on patch 7.41 · logistic regression on hero + "
    "synergy + counter features · test AUC 0.557. Pair effects are weak at this data size; "
    "individual hero strength is the cleanest signal."
)
