"""Fetch high-rank public matches from OpenDota since the start of patch 7.41.

Usage:
    python src/fetch_matches.py --target 15000

Paginates publicMatches via less_than_match_id. Resumable: skips IDs already in the
on-disk CSV. Filters to full 5v5 drafted matches on the current patch.
"""

import argparse
import csv
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.opendota.com/api/publicMatches"
PATCH_741_START = 1742774400  # 2026-03-24 UTC (matches /constants/patch)
MIN_RANK = 70  # Divine+ (works empirically; 80 sparse)
ALLOWED_GAME_MODES = {1, 2, 3, 4, 16, 22}  # AP, CM, RD, SD, CD, Ranked AP — skip Turbo(23), AD(18)
OUT = Path(__file__).resolve().parent.parent / "data" / "matches.csv"
FIELDS = ["match_id", "start_time", "avg_rank_tier", "lobby_type", "game_mode",
          "duration", "radiant_win", "radiant_team", "dire_team"]


def load_existing():
    if not OUT.exists():
        return set(), None
    seen = set()
    min_id = None
    with OUT.open() as f:
        r = csv.DictReader(f)
        for row in r:
            mid = int(row["match_id"])
            seen.add(mid)
            min_id = mid if min_id is None else min(min_id, mid)
    return seen, min_id


def fetch_page(less_than_id):
    params = {"min_rank": MIN_RANK}
    if less_than_id is not None:
        params["less_than_match_id"] = less_than_id
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        import json
        return json.load(resp)


def keep(m):
    if m.get("game_mode") not in ALLOWED_GAME_MODES:
        return False
    if not isinstance(m.get("radiant_team"), list) or not isinstance(m.get("dire_team"), list):
        return False
    if len(m["radiant_team"]) != 5 or len(m["dire_team"]) != 5:
        return False
    if m.get("start_time", 0) < PATCH_741_START:
        return False
    if m.get("avg_rank_tier") is None:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=15000, help="stop when CSV has this many rows")
    ap.add_argument("--sleep", type=float, default=1.1, help="seconds between API calls (rate limit ~60/min)")
    ap.add_argument("--refresh", action="store_true",
                    help="Start from the newest match instead of resuming from the oldest "
                         "seen ID. Use this to top up with matches played since the last fetch. "
                         "Stops automatically after several consecutive pages of duplicates.")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen, min_id = load_existing()
    print(f"loaded {len(seen)} existing matches; oldest match_id={min_id}")

    write_header = not OUT.exists() or OUT.stat().st_size == 0
    f = OUT.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if write_header:
        w.writeheader()

    # Refresh mode starts from current matches and paginates backwards, dedupes via
    # `seen`, and stops after several consecutive empty pages (= caught up).
    less_than = None if args.refresh else min_id
    total = len(seen)
    pages = 0
    kept_this_run = 0
    empty_streak = 0
    MAX_EMPTY_STREAK = 4
    try:
        while total < args.target:
            try:
                page = fetch_page(less_than)
            except Exception as e:
                print(f"  fetch error: {e}; sleeping 5s and retrying")
                time.sleep(5)
                continue
            pages += 1
            if not page:
                print("empty page, stopping")
                break

            page_min = min(m["match_id"] for m in page)
            page_kept = 0
            all_dupes = True
            for m in page:
                if m["match_id"] in seen:
                    continue
                all_dupes = False
                seen.add(m["match_id"])
                if not keep(m):
                    continue
                w.writerow({
                    "match_id": m["match_id"],
                    "start_time": m["start_time"],
                    "avg_rank_tier": m["avg_rank_tier"],
                    "lobby_type": m["lobby_type"],
                    "game_mode": m["game_mode"],
                    "duration": m["duration"],
                    "radiant_win": int(bool(m["radiant_win"])),
                    "radiant_team": ",".join(str(x) for x in m["radiant_team"]),
                    "dire_team": ",".join(str(x) for x in m["dire_team"]),
                })
                page_kept += 1
                total += 1
                kept_this_run += 1
            f.flush()

            oldest_in_page = min(m["start_time"] for m in page)
            print(f"page {pages}: {len(page)} fetched, {page_kept} kept, "
                  f"oldest_ts={oldest_in_page}, total={total}")

            # Refresh mode: stop once we've caught up (consecutive duplicate pages)
            if args.refresh:
                empty_streak = empty_streak + 1 if all_dupes else 0
                if empty_streak >= MAX_EMPTY_STREAK:
                    print(f"{MAX_EMPTY_STREAK} consecutive all-duplicate pages — caught up, stopping")
                    break

            # Stop if we've paginated back past the patch start
            if oldest_in_page < PATCH_741_START:
                print(f"reached patch start ({PATCH_741_START}), stopping")
                break

            less_than = page_min
            time.sleep(args.sleep)
    finally:
        f.close()
    print(f"done. total matches on disk: {total} (added {kept_this_run} this run, {pages} pages)")


if __name__ == "__main__":
    main()
