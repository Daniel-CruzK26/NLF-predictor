"""Script to fetch live NFL depth charts / starting QBs from ESPN Core API."""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_team_qb(team_id: int) -> Optional[dict]:
    """Fetches team details and starting QB from ESPN depth chart."""
    try:
        team_url = f"http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2025/teams/{team_id}?lang=en&region=us"
        t_data = fetch_json(team_url)
        abbr = t_data.get("abbreviation")
        name = t_data.get("displayName")

        depth_url = f"http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2025/teams/{team_id}/depthcharts"
        d_data = fetch_json(depth_url)

        qb_name = None
        for item in d_data.get("items", []):
            positions = item.get("positions", {})
            if "qb" in positions:
                qb_info = positions["qb"]
                athletes = qb_info.get("athletes", [])
                if athletes:
                    slot1 = sorted(athletes, key=lambda x: x.get("slot", 99))[0]
                    ath_ref = slot1.get("athlete", {}).get("$ref")
                    if ath_ref:
                        ath_data = fetch_json(ath_ref)
                        full_name = ath_data.get("displayName", "")
                        parts = full_name.split()
                        qb_name = f"{parts[0][0]}.{parts[-1]}" if len(parts) >= 2 else full_name
                        break

        return {"id": team_id, "abbr": abbr, "name": name, "qb": qb_name}
    except Exception as e:
        return None


def fetch_all_espn_qbs() -> Dict[str, str]:
    """Fetches starting QBs for all 32 teams in parallel."""
    print("📡 Consultando la API de ESPN para obtener los QBs titulares activos...")
    starters = {}
    
    # ESPN NFL team IDs run from 1 to 34 (with some non-existent/relocated IDs)
    team_ids = list(range(1, 35))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_team_qb, tid): tid for tid in team_ids}
        for future in as_completed(futures):
            res = future.result()
            if res and res.get("abbr") and res.get("qb"):
                starters[res["abbr"]] = res["qb"]
                print(f"  ✓ {res['abbr']:<4} ({res['name']:<25}) ➔ QB1: {res['qb']}")

    return starters


if __name__ == "__main__":
    results = fetch_all_espn_qbs()
    print(f"\n✅ Total de equipos procesados exitosamente desde ESPN: {len(results)}/32")
