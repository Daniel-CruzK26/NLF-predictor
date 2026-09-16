"""Script to fetch official titular depth charts (Slot 1 starters) for all 32 NFL teams from ESPN."""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STAR_PLAYERS_FILE = DATA_DIR / "star_players.json"
ACTIVE_STARTERS_FILE = DATA_DIR / "active_starters.json"
OFFICIAL_DEPTHCHARTS_FILE = DATA_DIR / "official_depthcharts_2026.json"

ESPN_TEAM_MAP = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL",
    7: "DEN", 8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC",
    13: "LV", 14: "LA", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO",
    19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC",
    25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX",
    33: "BAL", 34: "HOU",
}

# Standardized positional point impact values
STAR_VALUES = {
    "WR1": 1.7, "WR2": 1.2, "WR3": 0.9,
    "TE1": 1.3,
    "RB1": 1.3,
    "EDGE1": 1.6, "DE1": 1.4,
    "DT1": 1.4,
    "CB1": 1.5, "CB2": 1.1,
    "S1": 1.2, "FS1": 1.2, "SS1": 1.1,
    "LB1": 1.3, "MLB1": 1.3,
}


def fetch_team_depthchart(team_id: int, abbr: str) -> Dict[str, Any]:
    url = f"https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/depthcharts"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            starters = {}
            for grp in data.get("depthchart", []):
                for pos_key, pos_val in grp.get("positions", {}).items():
                    athletes = pos_val.get("athletes", [])
                    if athletes:
                        # Slot 1 / Rank 1 is the official starter
                        slot1 = athletes[0]
                        p_name = slot1.get("displayName") or slot1.get("shortName")
                        if p_name:
                            starters[pos_key.lower()] = {
                                "name": p_name,
                                "pos_name": pos_val.get("position", {}).get("displayName", pos_key),
                                "pos_abbr": pos_val.get("position", {}).get("abbreviation", pos_key.upper()),
                            }

            if starters:
                return {"abbr": abbr, "team_id": team_id, "starters": starters}
        except Exception as e:
            if attempt == 2:
                print(f"❌ Error fetching depth chart for {abbr} (ID {team_id}) after 3 attempts: {e}")
            import time
            time.sleep(1)
    return {"abbr": abbr, "team_id": team_id, "starters": {}}


def update_official_starters():
    print("📡 [1/3] Fetching official titular depth charts from ESPN for all 32 teams in parallel...")
    all_depths = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_team_depthchart, tid, abbr): abbr for tid, abbr in ESPN_TEAM_MAP.items()}
        for future in as_completed(futures):
            res = future.result()
            abbr = res["abbr"]
            all_depths[abbr] = res["starters"]
            st_cnt = len(res["starters"])
    # Load existing depth charts if available to avoid partial wipes
    existing_depths = {}
    if OFFICIAL_DEPTHCHARTS_FILE.exists():
        try:
            with open(OFFICIAL_DEPTHCHARTS_FILE, "r", encoding="utf-8") as f:
                existing_depths = json.load(f)
        except Exception:
            pass

    for abbr, st in all_depths.items():
        if len(st) < 15 and abbr in existing_depths:
            # Merge with existing so offensive/defensive starters are not lost
            merged = existing_depths[abbr].copy()
            merged.update(st)
            all_depths[abbr] = merged
            print(f"  ℹ️ {abbr:<4}: Preserved and merged {len(merged)} starter positions from previous verified depth chart.")

    # Save full depth charts
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OFFICIAL_DEPTHCHARTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_depths, f, indent=2)

    # 2. Extract Star Players (WR1, WR2, TE1, RB1, DE/EDGE, CB1, CB2, S1, MLB)
    print("\n⭐ [2/3] Extracting genuine titular star players into star_players.json...")
    star_players = {}
    active_qb_starters = {}

    existing_qbs = {}
    if ACTIVE_STARTERS_FILE.exists():
        try:
            with open(ACTIVE_STARTERS_FILE, "r", encoding="utf-8") as f:
                existing_qbs = json.load(f)
        except Exception:
            pass

    for abbr, st in all_depths.items():
        team_stars = []

        # QB
        qb_info = st.get("qb")
        if qb_info:
            full_name = qb_info["name"]
            parts = full_name.split()
            last = parts[-1]
            if len(parts) >= 3 and last in ["Jr.", "Sr.", "II", "III", "IV", "V"]:
                last = parts[-2]
            short_qb = f"{parts[0][0]}.{last}" if len(parts) >= 2 else full_name
            active_qb_starters[abbr] = short_qb
        elif abbr in existing_qbs:
            active_qb_starters[abbr] = existing_qbs[abbr]

        # Offensive Starters: WR1, WR2, TE1, RB1
        for pos_k in ["wr1", "wr2", "te", "rb"]:
            p = st.get(pos_k)
            if p:
                pos_lbl = "WR" if "wr" in pos_k else ("TE" if pos_k == "te" else "RB")
                val_key = "WR1" if pos_k == "wr1" else ("WR2" if pos_k == "wr2" else ("TE1" if pos_k == "te" else "RB1"))
                team_stars.append({
                    "name": p["name"],
                    "pos": pos_lbl,
                    "unit": "offense",
                    "value_pts": STAR_VALUES.get(val_key, 1.2),
                })

        # Defensive Starters: LDE, RDE, LCB, RCB, MLB/ILB, FS, SS
        def_keys = [
            ("lde", "EDGE", "EDGE1"),
            ("rde", "EDGE", "EDGE1"),
            ("lcb", "CB", "CB1"),
            ("rcb", "CB", "CB2"),
            ("mlb", "LB", "MLB1"),
            ("lilb", "LB", "LB1"),
            ("fs", "S", "FS1"),
            ("ss", "S", "SS1"),
            ("nt", "DT", "DT1"),
        ]

        for pos_k, pos_lbl, val_k in def_keys:
            p = st.get(pos_k)
            if p and len([s for s in team_stars if s["unit"] == "defense"]) < 4:
                # Avoid duplicate names
                if not any(s["name"] == p["name"] for s in team_stars):
                    team_stars.append({
                        "name": p["name"],
                        "pos": pos_lbl,
                        "unit": "defense",
                        "value_pts": STAR_VALUES.get(val_k, 1.2),
                    })

        star_players[abbr] = team_stars
        star_names = [f"{s['name']} ({s['pos']})" for s in team_stars]
        print(f"  • {abbr:<4}: {', '.join(star_names[:4])} ... ({len(team_stars)} stars)")

    # Save to disk
    with open(STAR_PLAYERS_FILE, "w", encoding="utf-8") as f:
        json.dump(star_players, f, indent=2)

    with open(ACTIVE_STARTERS_FILE, "w", encoding="utf-8") as f:
        json.dump(active_qb_starters, f, indent=2)

    print(f"\n✅ [3/3] Successfully updated:")
    print(f"   • {STAR_PLAYERS_FILE}")
    print(f"   • {ACTIVE_STARTERS_FILE}")
    print(f"   • {OFFICIAL_DEPTHCHARTS_FILE}")


if __name__ == "__main__":
    update_official_starters()
