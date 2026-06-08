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

st.set_page_config(
    page_title="Dota Draft Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Mobile-friendly tweaks: tighten paddings, reduce min image-tile width, allow
# table cells to wrap, give the picker grid breathing room on narrow screens.
st.markdown(
    """
    <style>
      /* tighter top padding on small viewports */
      @media (max-width: 768px) {
        .block-container { padding-top: 1rem !important; padding-left: 0.75rem !important;
                           padding-right: 0.75rem !important; }
        h1, h2, h3 { font-size: 1.1rem !important; }
        [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
        [data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
        /* shrink hero portrait thumbs in the picker on phones */
        [data-testid="stImageContainer"] img { max-width: 56px !important; }
      }
      /* keep dataframes from overflowing horizontally */
      [data-testid="stDataFrame"] { overflow-x: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    out = {}
    for h in heroes:
        short = h["name"].replace("npc_dota_hero_", "")
        out[h["id"]] = {
            "id": h["id"],
            "name": h["localized_name"],
            "roles": h.get("roles", []),
            "primary_attr": h["primary_attr"],
            "portrait": f"{cdn}/{short}.png",
            "icon": f"{cdn}/icons/{short}.png",
        }
    return out


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


def _org_key(name):
    """Normalize a team name to an org key, collapsing sponsor variants.

    'Aurora Gaming', 'Aurora.1xBet' → 'aurora'
    'BB', 'BB Team' → 'bb'
    'Team Falcons' → 'falcons'
    'VGJ Storm' vs 'VGJ Thunder' → kept distinct ('vgj storm' / 'vgj thunder')
    """
    import re
    if not name:
        return None
    s = re.sub(r"\(.*?\)", "", name.lower())  # strip parentheticals
    parts = re.split(r"[.\-\s]+", s)
    skip = {"gaming", "esports", "esport", "e-sports", "team", "club", "fc", "1xbet",
            "academy", "junior", "jr", "the", ""}
    parts = [p for p in parts if p and p not in skip]
    return " ".join(parts) if parts else (s.strip() or None)


@st.cache_data(ttl=3600)
def top_team_ids(n=20, pull=40):
    """Top-N **organizations** by OpenDota rating, deduped by normalized name.

    Returns dict keyed by team_id (covering all sponsor variants of each org)
    so the live filter still matches whichever team_id the match was registered
    under. Canonical = highest-rated entry per org group.
    """
    import urllib.request
    from collections import defaultdict
    req = urllib.request.Request("https://api.opendota.com/api/teams",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        teams = json.load(resp)
    teams = sorted(teams, key=lambda t: -(t.get("rating") or 0))[:pull]

    by_org = defaultdict(list)
    for t in teams:
        key = _org_key(t.get("name"))
        if not key:
            continue
        by_org[key].append(t)

    # rank orgs by their top entry's rating, keep top N
    orgs = sorted(
        by_org.items(),
        key=lambda kv: -(kv[1][0].get("rating") or 0),
    )[:n]

    out = {}
    for org_key, members in orgs:
        members.sort(key=lambda t: -(t.get("rating") or 0))
        canonical = members[0]
        seen = {canonical.get("name")}
        alt_names = []
        for t in members[1:]:
            nm = t.get("name")
            if nm and nm not in seen:
                alt_names.append(nm)
                seen.add(nm)
        for t in members:
            out[t["team_id"]] = {
                "name": t.get("name") or f"Team {t['team_id']}",
                "canonical_name": canonical.get("name"),
                "logo": t.get("logo_url"),
                "rating": round(t.get("rating") or 0),
                "tag": t.get("tag"),
                "is_canonical": t["team_id"] == canonical["team_id"],
                "alt_names": alt_names if t["team_id"] == canonical["team_id"] else [],
            }
    return out


def _pandascore_key():
    """Look up PandaScore API key from Streamlit secrets, then env. None if neither."""
    import os
    try:
        return st.secrets["PANDASCORE_API_KEY"]
    except (FileNotFoundError, KeyError, Exception):
        pass
    return os.environ.get("PANDASCORE_API_KEY")


@st.cache_data(ttl=300)
def upcoming_pandascore_matches(window_hours=24):
    """Fetch upcoming Dota 2 matches within window from PandaScore. 5-min cache.

    Returns ([] , msg). msg is None on success or human-readable error.
    """
    key = _pandascore_key()
    if not key:
        return [], "missing_key"
    import urllib.parse
    import urllib.request
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc)
    until = now + _dt.timedelta(hours=window_hours)
    params = {
        "range[begin_at]": f"{now.isoformat()},{until.isoformat()}",
        "sort": "begin_at",
        "per_page": "50",
    }
    url = "https://api.pandascore.co/dota2/matches/upcoming?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, headers={"Accept": "application/json",
                      "Authorization": f"Bearer {key}",
                      "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.load(resp), None
    except Exception as e:
        return [], f"fetch_failed: {e}"


@st.cache_data(ttl=86400)
def player_profile(account_id):
    """One call for name + top heroes. Cached 24h per account."""
    if not account_id:
        return None
    import urllib.request
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        req = urllib.request.Request(
            f"https://api.opendota.com/api/players/{account_id}/heroes", headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            heroes_raw = json.load(resp)
    except Exception:
        heroes_raw = []
    try:
        req = urllib.request.Request(
            f"https://api.opendota.com/api/players/{account_id}", headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            prof = json.load(resp).get("profile") or {}
        name = prof.get("name") or prof.get("personaname") or f"#{account_id}"
    except Exception:
        name = f"#{account_id}"
    heroes = []
    for h in heroes_raw:
        if h.get("games", 0) >= 5:
            heroes.append({
                "hero_id": int(h["hero_id"]),
                "games": int(h["games"]),
                "wins": int(h.get("win", 0)),
                "wr": float(h["win"] / h["games"]) if h["games"] else 0.0,
            })
    heroes.sort(key=lambda h: -h["games"])
    return {"name": name, "heroes": heroes}


@st.cache_data(ttl=60)
def live_top_team_matches():
    """Currently-live matches involving at least one top-20 team. TTL 60s."""
    import urllib.request
    top = top_team_ids()
    req = urllib.request.Request("https://api.opendota.com/api/live",
                                 headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            live = json.load(resp)
    except Exception as e:
        return [], top, str(e)
    out = []
    for m in live:
        rt, dt = m.get("team_id_radiant", 0), m.get("team_id_dire", 0)
        if rt == 0 or dt == 0:
            continue  # skip pub matches
        if rt not in top and dt not in top:
            continue
        out.append(m)
    out.sort(key=lambda m: -m.get("spectators", 0))
    return out, top, None


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


RECENT_DRAFTS_FILE = ROOT / "data" / "recent_drafts.json"
MAX_RECENT_DRAFTS = 8


def load_recent_drafts():
    if not RECENT_DRAFTS_FILE.exists():
        return []
    try:
        return json.loads(RECENT_DRAFTS_FILE.read_text())
    except Exception:
        return []


def record_draft(r_ids, d_ids, prob):
    history = load_recent_drafts()
    key = (tuple(sorted(r_ids)), tuple(sorted(d_ids)))
    history = [h for h in history
               if (tuple(sorted(h["radiant"])), tuple(sorted(h["dire"]))) != key]
    history.insert(0, {"radiant": list(r_ids), "dire": list(d_ids), "prob": float(prob)})
    RECENT_DRAFTS_FILE.write_text(json.dumps(history[:MAX_RECENT_DRAFTS]))


def load_draft_into_state(idx):
    history = load_recent_drafts()
    if 0 <= idx < len(history):
        st.session_state.radiant = list(history[idx]["radiant"])
        st.session_state.dire = list(history[idx]["dire"])
        st.session_state.active_side = "Radiant"


def clear_recent_drafts():
    if RECENT_DRAFTS_FILE.exists():
        RECENT_DRAFTS_FILE.unlink()


def _fmt_dur(secs):
    m, s = divmod(max(0, int(secs)), 60)
    return f"{m}:{s:02d}"


def render_live_compact(m, top):
    """Sidebar-width version of the live-match card."""
    rt_id, dt_id = m["team_id_radiant"], m["team_id_dire"]
    r_name = m.get("team_name_radiant") or top.get(rt_id, {}).get("name", "?")
    d_name = m.get("team_name_dire") or top.get(dt_id, {}).get("name", "?")
    r_top = "⭐" if rt_id in top else ""
    d_top = "⭐" if dt_id in top else ""
    r_heroes = [p["hero_id"] for p in m["players"] if p["team"] == 0]
    d_heroes = [p["hero_id"] for p in m["players"] if p["team"] == 1]

    def icon_row(heroes):
        cols = st.columns(5, gap="small")
        for i, hid in enumerate(heroes[:5]):
            with cols[i]:
                meta = hero_meta.get(hid)
                if meta:
                    st.image(meta["icon"], width="stretch")

    with st.container(border=True):
        st.markdown(
            f"<div style='font-size:13px;line-height:1.3;'>"
            f"<b style='color:#2e8b57;'>{r_top}{r_name}</b><br>"
            f"<span style='color:#666;'>vs</span><br>"
            f"<b style='color:#c0392b;'>{d_top}{d_name}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:12px;color:#888;margin:4px 0 6px 0;'>"
            f"<b style='color:#2e8b57;'>{m['radiant_score']}</b> – "
            f"<b style='color:#c0392b;'>{m['dire_score']}</b> · "
            f"<code>{_fmt_dur(m['game_time'])}</code> · "
            f"👁 {m.get('spectators', 0):,}"
            f"</div>",
            unsafe_allow_html=True,
        )
        icon_row(r_heroes)
        icon_row(d_heroes)

        # model win prob if all heroes recognized
        r_valid = [h for h in r_heroes[:5] if h in hero_meta]
        d_valid = [h for h in d_heroes[:5] if h in hero_meta]
        if len(r_valid) == 5 and len(d_valid) == 5:
            try:
                p = model.predict([hero_meta[h]["name"] for h in r_valid],
                                  [hero_meta[h]["name"] for h in d_valid])
                fav = "Radiant" if p >= 0.5 else "Dire"
                c = "#2e8b57" if fav == "Radiant" else "#c0392b"
                st.markdown(
                    f"<div style='text-align:center;margin-top:6px;font-size:13px;'>"
                    f"model: <b style='color:{c};'>{fav} {max(p, 1-p):.0%}</b></div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        if st.button("📊 Details", key=f"details_{m['match_id']}", width="stretch"):
            show_live_match_details(m, top)


@st.dialog("Live match analytics", width="large")
def show_live_match_details(m, top):
    rt_id, dt_id = m["team_id_radiant"], m["team_id_dire"]
    r_name = m.get("team_name_radiant") or top.get(rt_id, {}).get("name", "Radiant")
    d_name = m.get("team_name_dire") or top.get(dt_id, {}).get("name", "Dire")
    r_heroes = [p["hero_id"] for p in m["players"] if p["team"] == 0]
    d_heroes = [p["hero_id"] for p in m["players"] if p["team"] == 1]

    # ----- header -----
    st.markdown(
        f"<div style='text-align:center;'>"
        f"<span style='font-size:22px;color:#2e8b57;font-weight:700;'>{r_name}</span>"
        f"&nbsp; <span style='color:#888;'>vs</span> &nbsp;"
        f"<span style='font-size:22px;color:#c0392b;font-weight:700;'>{d_name}</span><br>"
        f"<span style='font-size:18px;color:#2e8b57;font-weight:700;'>{m['radiant_score']}</span>"
        f"&nbsp;–&nbsp;"
        f"<span style='font-size:18px;color:#c0392b;font-weight:700;'>{m['dire_score']}</span>"
        f" · <code>{_fmt_dur(m['game_time'])}</code> · 👁 {m.get('spectators', 0):,}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ----- hero rows -----
    def portrait_row(heroes, label, color):
        st.markdown(f"<div style='margin-top:10px;color:{color};font-weight:600;'>{label}</div>",
                    unsafe_allow_html=True)
        cols = st.columns(5, gap="small")
        for i, hid in enumerate(heroes[:5]):
            with cols[i]:
                meta = hero_meta.get(hid)
                if not meta:
                    st.markdown("<div style='color:#666;font-size:11px;'>?</div>",
                                unsafe_allow_html=True)
                    continue
                st.image(meta["portrait"], width="stretch")
                wr = hstats.get(hid, {}).get("winrate")
                wr_txt = f"{wr:.0%}" if wr is not None else "—"
                st.markdown(
                    f"<div style='text-align:center;font-size:12px;line-height:1.2;'>"
                    f"<b>{meta['name']}</b><br><span style='color:#888;'>{wr_txt}</span></div>",
                    unsafe_allow_html=True,
                )

    portrait_row(r_heroes, "🟢 Radiant", "#2e8b57")
    portrait_row(d_heroes, "🔴 Dire", "#c0392b")

    r_valid = [h for h in r_heroes[:5] if h in hero_meta]
    d_valid = [h for h in d_heroes[:5] if h in hero_meta]

    if len(r_valid) == 5 and len(d_valid) == 5:
        st.divider()

        # ----- model prediction -----
        try:
            p = model.predict([hero_meta[h]["name"] for h in r_valid],
                              [hero_meta[h]["name"] for h in d_valid])
        except Exception:
            p = None

        if p is not None:
            fav = "Radiant" if p >= 0.5 else "Dire"
            c_hex = "#2e8b57" if fav == "Radiant" else "#c0392b"
            st.markdown(
                f"<div style='text-align:center;padding:6px 0;'>"
                f"<div style='font-size:12px;color:#888;'>model pre-game</div>"
                f"<div style='font-size:36px;font-weight:700;color:{c_hex};line-height:1.1;'>"
                f"{fav} {max(p, 1-p):.1%}</div></div>",
                unsafe_allow_html=True,
            )

        # ----- phase edge one-liner -----
        segs = []
        for name, _label, _, _ in PHASES:
            rd = sum((hstats[h][f"wr_{name}"] - 0.5) * 100 for h in r_valid
                     if h in hstats and not pd.isna(hstats[h][f"wr_{name}"]))
            dd = sum((hstats[h][f"wr_{name}"] - 0.5) * 100 for h in d_valid
                     if h in hstats and not pd.isna(hstats[h][f"wr_{name}"]))
            diff = rd - dd
            side = "R" if diff >= 0 else "D"
            cc = "#2e8b57" if diff >= 0 else "#c0392b"
            segs.append(
                f"<span style='color:#888;'>{name}</span> "
                f"<span style='color:{cc};font-weight:600;'>{side} {abs(diff):.1f}pp</span>"
            )
        st.markdown(
            "<div style='font-size:13px;text-align:center;padding:6px 0;'>"
            "<span style='color:#666;'>Phase edge:</span>&nbsp;&nbsp;"
            + " &nbsp;·&nbsp; ".join(segs) + "</div>",
            unsafe_allow_html=True,
        )

        # ----- strengths / weaknesses -----
        def hero_dev(hid):
            return (hstats[hid]["winrate"] - 0.5) * 100 if hid in hstats else 0.0

        def chip(hid):
            dev = hero_dev(hid)
            col = "#2e8b57" if dev > 0 else "#c0392b" if dev < 0 else "#888"
            return (f"<span style='background:rgba(255,255,255,0.04);padding:3px 7px;"
                    f"border-radius:9px;font-size:12px;margin-right:4px;'>"
                    f"<b style='color:#ddd;'>{hero_meta[hid]['name']}</b> "
                    f"<span style='color:{col};font-weight:600;'>{dev:+.1f}pp</span></span>")

        r_sorted = sorted(r_valid, key=hero_dev, reverse=True)
        d_sorted = sorted(d_valid, key=hero_dev, reverse=True)
        st.markdown(
            "<div style='margin-top:10px;'>"
            "<table style='width:100%;font-size:13px;'>"
            "<tr><td style='padding:4px 6px;color:#888;'></td>"
            "<td style='padding:4px 6px;color:#2e8b57;'>🟢 Radiant</td>"
            "<td style='padding:4px 6px;color:#c0392b;'>🔴 Dire</td></tr>"
            f"<tr><td style='padding:6px;color:#888;'>⭐ Best</td>"
            f"<td style='padding:6px;'>{chip(r_sorted[0])}{chip(r_sorted[1])}</td>"
            f"<td style='padding:6px;'>{chip(d_sorted[0])}{chip(d_sorted[1])}</td></tr>"
            f"<tr><td style='padding:6px;color:#888;'>⚠ Worst</td>"
            f"<td style='padding:6px;'>{chip(r_sorted[-1])}</td>"
            f"<td style='padding:6px;'>{chip(d_sorted[-1])}</td></tr>"
            "</table></div>",
            unsafe_allow_html=True,
        )

        st.divider()
        if st.button("📥 Load this draft into the main panel for full breakdown",
                     key=f"load_live_{m['match_id']}", width="stretch"):
            st.session_state.radiant = list(r_valid)
            st.session_state.dire = list(d_valid)
            st.session_state.active_side = "Radiant"
            st.rerun()
    else:
        st.info("Draft incomplete or contains an unrecognized hero — detailed analytics will "
                "appear once all 10 heroes are valid.")


def quick_add_from_search():
    """Triggered when user presses Enter in the search box. Adds the best-matched
    unpicked hero to the active side and clears the search field."""
    val = st.session_state.get("search_box", "").strip().lower()
    if not val:
        return
    picked = set(st.session_state.radiant) | set(st.session_state.dire)
    pool = [m for m in hero_meta.values() if m["id"] not in picked]
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


def render_sidebar():
    """Render Live / Upcoming / Recent as tabs inside Streamlit's left sidebar."""
    import datetime as _dt
    live, top, live_err = live_top_team_matches()
    n_orgs = sum(1 for v in top.values() if v["is_canonical"])
    recent = load_recent_drafts()

    tab_live, tab_up, tab_recent = st.tabs([
        f"Live ({len(live)})",
        "Upcoming",
        f"Recent ({len(recent)})",
    ])

    with tab_live:
        if live_err:
            st.error(live_err)
        elif not live:
            st.caption(f"No top-{n_orgs} matches right now.")
        else:
            for m in live:
                render_live_compact(m, top)

    with tab_up:
        up_data, up_err = upcoming_pandascore_matches(24)
        if up_err == "missing_key":
            st.caption("PandaScore API key required.")
            with st.expander("Enable"):
                st.markdown(
                    "1. Free key at <https://pandascore.co/>\n"
                    "2. Save to `.streamlit/secrets.toml`:\n"
                    "    `PANDASCORE_API_KEY = \"your_token\"`",
                    unsafe_allow_html=True,
                )
        elif up_err:
            st.error(up_err)
        else:
            top_orgs = {_org_key(v["canonical_name"]) for v in top.values()
                        if v["is_canonical"]}
            relevant = [
                m for m in up_data
                if any(_org_key(o.get("opponent", {}).get("name") or "") in top_orgs
                       for o in m.get("opponents") or [])
            ]
            if not relevant:
                st.caption(f"No top-{n_orgs} matches in the next 24h.")
            else:
                now = _dt.datetime.now(_dt.timezone.utc)
                for m in relevant[:6]:
                    opps = m.get("opponents", [])
                    if len(opps) < 2:
                        continue
                    t1 = opps[0].get("opponent", {}).get("name", "?")
                    t2 = opps[1].get("opponent", {}).get("name", "?")
                    tour = m.get("tournament", {}).get("name") or ""
                    try:
                        dt = _dt.datetime.fromisoformat(
                            m.get("begin_at").replace("Z", "+00:00"))
                        delta = dt - now
                        hrs = int(delta.total_seconds() // 3600)
                        mins = int((delta.total_seconds() % 3600) // 60)
                        when_str = f"in {hrs}h {mins:02d}m"
                    except Exception:
                        when_str = "?"
                    with st.container(border=True):
                        st.markdown(
                            f"<div style='font-size:13px;line-height:1.4;'>"
                            f"<b>{t1}</b> vs <b>{t2}</b><br>"
                            f"<span style='color:#888;font-size:11px;'>"
                            f"{when_str} · {tour}</span></div>",
                            unsafe_allow_html=True,
                        )

    with tab_recent:
        if not recent:
            st.caption("Complete a 5v5 draft to save it here.")
        else:
            st.button("Clear all", key="clear_recent",
                      on_click=clear_recent_drafts, width="stretch")
            for i, entry in enumerate(recent):
                prob = entry["prob"]
                fav = "R" if prob >= 0.5 else "D"
                c = "#2e8b57" if fav == "R" else "#c0392b"
                with st.container(border=True):
                    st.markdown(
                        f"<div style='font-size:13px;'><b style='color:{c};'>"
                        f"{fav} {max(prob, 1-prob):.0%}</b></div>",
                        unsafe_allow_html=True,
                    )
                    rc = st.columns(5, gap="small")
                    for j, hid in enumerate(entry["radiant"][:5]):
                        with rc[j]:
                            meta = hero_meta.get(hid)
                            if meta:
                                st.image(meta["icon"], width="stretch")
                    dc = st.columns(5, gap="small")
                    for j, hid in enumerate(entry["dire"][:5]):
                        with dc[j]:
                            meta = hero_meta.get(hid)
                            if meta:
                                st.image(meta["icon"], width="stretch")
                    st.button("Load", key=f"recent_load_{i}",
                              on_click=load_draft_into_state, args=(i,),
                              width="stretch")


# ---------- header ----------
h1, h2 = st.columns([5, 1])
with h1:
    st.markdown("## Dota Draft Lab")
    st.caption("Patch 7.41 · 35k Divine+ matches · model AUC 0.57")
with h2:
    st.button("Reset", on_click=reset_draft, width="stretch")


# ---------- sidebar (live + upcoming matches) ----------
with st.sidebar:
    render_sidebar()


# ---------- search + side toggle (top) ----------
sc1, sc2 = st.columns([4, 1])
with sc1:
    st.text_input(
        "Search", key="search_box",
        placeholder="Search hero — press Enter to draft",
        label_visibility="collapsed",
        on_change=quick_add_from_search,
    )
    _search = st.session_state.get("search_box", "").strip().lower()
with sc2:
    st.radio("Side", ["Radiant", "Dire"], key="active_side", horizontal=True,
             label_visibility="collapsed")


# ---------- picks (drafted heroes, visible at the top) ----------
r, d = st.columns(2)
for side_key, label, color, col in [("radiant", "Radiant", "green", r),
                                    ("dire", "Dire", "red", d)]:
    with col:
        st.markdown(
            f"<div style='font-size:14px;font-weight:600;color:"
            f"{'#2e8b57' if color == 'green' else '#c0392b'};"
            f"margin-bottom:6px;'>{label} "
            f"<span style='color:#666;font-weight:400;'>"
            f"{len(st.session_state[side_key])}/5</span></div>",
            unsafe_allow_html=True,
        )
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
                        "<div style='border:1px solid #2a2a2a;border-radius:4px;"
                        "aspect-ratio:256/144;'></div>",
                        unsafe_allow_html=True,
                    )

st.markdown("<div style='margin:14px 0;'></div>", unsafe_allow_html=True)


# ---------- result ----------
if len(st.session_state.radiant) == 5 and len(st.session_state.dire) == 5:
    radiant = [hero_meta[h]["name"] for h in st.session_state.radiant]
    dire = [hero_meta[h]["name"] for h in st.session_state.dire]
    try:
        p = model.predict(radiant, dire)
    except Exception as e:
        st.error(f"prediction failed: {e}")
    else:
        from itertools import combinations

        _draft_key = (tuple(sorted(st.session_state.radiant)),
                      tuple(sorted(st.session_state.dire)))
        if st.session_state.get("_last_recorded_key") != _draft_key:
            record_draft(st.session_state.radiant, st.session_state.dire, p)
            st.session_state._last_recorded_key = _draft_key

        favored = "Radiant" if p >= 0.5 else "Dire"
        prob = p if p >= 0.5 else 1 - p
        edge_pp = (p - 0.5) * 200

        # Prediction as a clean metric
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted winner", favored)
        m2.metric("Win probability", f"{prob:.1%}")
        m3.metric("Edge", f"{edge_pp:+.1f} pp",
                  delta_color="off")

        # Aggregates
        r_ids = st.session_state.radiant
        d_ids = st.session_state.dire

        def _solo(ids):
            devs = [(hstats[h]["winrate"] - 0.5) * 100 for h in ids if h in hstats]
            return (sum(devs) / len(devs)) if devs else 0.0

        def _pair(ids, min_picks=5):
            devs = []
            for a, b in combinations(ids, 2):
                k = (min(a, b), max(a, b))
                s = same_pair_stats.get(k)
                if s and s["picks"] >= min_picks:
                    devs.append((s["wins"] / s["picks"] - 0.5) * 100)
            return (sum(devs) / len(devs)) if devs else 0.0

        def _counter(r, d, min_picks=5):
            devs = []
            for hr in r:
                for hd in d:
                    s = counter_pair_stats.get((hr, hd))
                    if s and s["picks"] >= min_picks:
                        devs.append((s["wins"] / s["picks"] - 0.5) * 100)
            return (sum(devs) / len(devs)) if devs else 0.0

        ctr = _counter(r_ids, d_ids)
        breakdown = pd.DataFrame({
            "Component": ["Solo strength", "Pair synergy", "Cross-team counter"],
            "Radiant": [_solo(r_ids), _pair(r_ids), ctr],
            "Dire":    [_solo(d_ids), _pair(d_ids), -ctr],
        })
        st.dataframe(
            breakdown.style
                .format({"Radiant": "{:+.2f}", "Dire": "{:+.2f}"})
                .background_gradient(cmap="RdYlGn",
                                     subset=["Radiant", "Dire"], vmin=-5, vmax=5),
            hide_index=True, width="stretch",
        )

        # Strongest / weakest hero per side
        def _dev(hid):
            return (hstats[hid]["winrate"] - 0.5) * 100 if hid in hstats else 0.0

        r_sorted = sorted(r_ids, key=_dev, reverse=True)
        d_sorted = sorted(d_ids, key=_dev, reverse=True)

        def _label(hid):
            return f"{hero_meta[hid]['name']} ({_dev(hid):+.1f} pp)"

        strengths = pd.DataFrame([
            {"": "Strongest", "Radiant": _label(r_sorted[0]),
             "Dire": _label(d_sorted[0])},
            {"": "2nd",       "Radiant": _label(r_sorted[1]),
             "Dire": _label(d_sorted[1])},
            {"": "Weakest",   "Radiant": _label(r_sorted[-1]),
             "Dire": _label(d_sorted[-1])},
        ])
        st.dataframe(strengths, hide_index=True, width="stretch")

        # Role composition (from OpenDota hero role tags — the model itself doesn't
        # know who plays which role, but this flags structural gaps in a draft)
        TRACKED_ROLES = ["Carry", "Support", "Initiator",
                         "Disabler", "Nuker", "Durable", "Escape", "Pusher"]

        def _role_counts(ids):
            counts = {r: 0 for r in TRACKED_ROLES}
            for hid in ids:
                for role in hero_meta.get(hid, {}).get("roles", []):
                    if role in counts:
                        counts[role] += 1
            return counts

        r_roles = _role_counts(r_ids)
        d_roles = _role_counts(d_ids)
        comp = pd.DataFrame({
            "Role tag": TRACKED_ROLES,
            "Radiant":  [r_roles[r] for r in TRACKED_ROLES],
            "Dire":     [d_roles[r] for r in TRACKED_ROLES],
        })
        st.dataframe(
            comp.style
                .background_gradient(cmap="Greens",
                                     subset=["Radiant"], vmin=0, vmax=5)
                .background_gradient(cmap="Reds",
                                     subset=["Dire"], vmin=0, vmax=5),
            hide_index=True, width="stretch",
        )
        gaps = []
        for tag in ("Carry", "Support"):
            if r_roles[tag] == 0:
                gaps.append(f"Radiant has no tagged **{tag}**")
            if d_roles[tag] == 0:
                gaps.append(f"Dire has no tagged **{tag}**")
        if gaps:
            st.caption("Composition flags: " + " · ".join(gaps))
        st.caption("Roles are hero-tag overlaps from OpenDota — a hero is tagged "
                   "Carry/Support/etc. if they *can* fill that role, not necessarily "
                   "where they're played. The model ignores roles entirely.")

        # Phase advantage chart
        st.markdown("**Phase advantage by game length**")
        n_b = len(DURATION_BUCKET_MIN)
        xs, x_labels = [], []
        for lo, hi in DURATION_BUCKET_MIN:
            xs.append((lo + min(hi, 60)) / 2)
            x_labels.append(f"<{hi}" if lo == 0
                            else f"{lo}+" if hi >= 1000
                            else f"{lo}-{hi}")

        def _bucket_sum(side_ids):
            ys = []
            for bi in range(n_b):
                tot, ok = 0.0, True
                for h in side_ids:
                    b = hstats.get(h, {}).get("buckets", [None] * n_b)[bi]
                    if b is None or pd.isna(b["wr"]):
                        ok = False
                        break
                    tot += (b["wr"] - 0.5) * 100
                ys.append(tot if ok else float("nan"))
            return ys

        r_y = _bucket_sum(r_ids)
        d_y = _bucket_sum(d_ids)
        diff = [(rv - dv) if not (pd.isna(rv) or pd.isna(dv)) else float("nan")
                for rv, dv in zip(r_y, d_y)]
        diff_pos = [v if (not pd.isna(v) and v > 0) else 0 for v in diff]
        diff_neg = [v if (not pd.isna(v) and v < 0) else 0 for v in diff]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=diff_pos, mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="rgba(0,0,0,0)", shape="spline", smoothing=1.2),
            fill="tozeroy", fillcolor="rgba(46,139,87,0.25)",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=diff_neg, mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="rgba(0,0,0,0)", shape="spline", smoothing=1.2),
            fill="tozeroy", fillcolor="rgba(192,57,43,0.25)",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=diff, mode="lines+markers",
            line=dict(color="#bbb", width=2, shape="spline", smoothing=1.2),
            marker=dict(size=6, color="#bbb"),
            hovertemplate="%{y:+.1f} pp<extra></extra>", showlegend=False,
        ))
        fig.add_hline(y=0, line_width=1, line_color="#555")

        valid = [v for v in diff if not pd.isna(v)]
        if valid:
            pad = max(2, (max(valid) - min(valid)) * 0.2)
            lo, hi = min(valid) - pad, max(valid) + pad
            lo, hi = min(lo, -2), max(hi, 2)
        else:
            lo, hi = -10, 10

        fig.update_layout(
            xaxis=dict(title="Game length (min)", tickmode="array",
                       tickvals=xs, ticktext=x_labels),
            yaxis=dict(title="Radiant − Dire (pp)", range=[lo, hi], zeroline=False),
            height=240, margin=dict(l=10, r=10, t=10, b=40),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")
else:
    need = 10 - len(st.session_state.radiant) - len(st.session_state.dire)
    st.caption(f"Pick {need} more hero{'es' if need != 1 else ''}.")

# ---------- picker grid (choices) ----------
_picked = set(st.session_state.radiant) | set(st.session_state.dire)
_candidates = sorted(
    [m for m in hero_meta.values()
     if (not _search or _search in m["name"].lower()) and m["id"] not in _picked],
    key=lambda m: m["name"],
)
if not _candidates:
    st.info("No heroes match.")
else:
    _caps = []
    for m in _candidates:
        s = hstats.get(m["id"])
        wr = f"{s['winrate']:.0%}" if s else "—"
        _caps.append(f"{m['name']} · {wr}")
    _n_picked = len(st.session_state.radiant) + len(st.session_state.dire)
    _picker_key = f"picker_{_n_picked}_{_search}"
    # wsrv.nl serves a smaller pre-resized portrait so image_select packs
    # more tiles per row while keeping the recognizable landscape hero card.
    _thumb_w = 96
    _sel = image_select(
        label="",
        images=[
            f"https://wsrv.nl/?url={m['portrait'].replace('https://', '')}"
            f"&w={_thumb_w}&output=png"
            for m in _candidates
        ],
        captions=_caps,
        index=-1, return_value="index",
        use_container_width=False,
        key=_picker_key,
    )
    if _sel is not None and _sel >= 0:
        add_hero(_candidates[_sel]["id"])
        st.rerun()

# ---------- footnote ----------
st.markdown("<div style='margin:24px 0 0;'></div>", unsafe_allow_html=True)
st.caption("Data: OpenDota public matches (Divine+). Model: L2-regularized logistic regression.")
