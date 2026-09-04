"""Non-QB Star Player Impact & Injury Adjustment Module."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
STAR_PLAYERS_FILE = DATA_DIR / "star_players.json"
INJURIES_FILE = DATA_DIR / "injuries_status.json"

# Multiplier on star value based on official injury designation
INJURY_AVAILABILITY_FACTOR = {
    "ACTIVE": 1.0,
    "QUESTIONABLE": 0.5,
    "DOUBTFUL": 0.2,
    "OUT": 0.0,
    "IR": 0.0,
}


def load_star_players() -> Dict[str, List[dict]]:
    """Loads the catalog of non-QB star players by team."""
    if not STAR_PLAYERS_FILE.exists():
        return {}
    try:
        with open(STAR_PLAYERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_injury_status() -> Dict[str, str]:
    """Loads current injury status overrides for star players."""
    if not INJURIES_FILE.exists():
        return {}
    try:
        with open(INJURIES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_injury_status(status_dict: Dict[str, str]) -> None:
    """Saves updated injury status overrides."""
    INJURIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INJURIES_FILE, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, indent=2)


def get_team_star_ratings(
    team: str,
    stars_catalog: Optional[Dict[str, List[dict]]] = None,
    injuries: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Calculates active offensive and defensive star player power ratings for a team."""
    if stars_catalog is None:
        stars_catalog = load_star_players()
    if injuries is None:
        injuries = load_injury_status()

    players = stars_catalog.get(team, [])
    off_val = 0.0
    def_val = 0.0
    off_lost = 0.0
    def_lost = 0.0

    for p in players:
        p_name = p.get("name", "")
        unit = p.get("unit", "offense")
        base_val = float(p.get("value_pts", 1.0))
        status = injuries.get(p_name, "ACTIVE").upper()
        avail = INJURY_AVAILABILITY_FACTOR.get(status, 1.0)

        effective_val = base_val * avail
        lost_val = base_val * (1.0 - avail)

        if unit == "offense":
            off_val += effective_val
            off_lost += lost_val
        else:
            def_val += effective_val
            def_lost += lost_val

    return {
        "off_stars_active": round(off_val, 2),
        "def_stars_active": round(def_val, 2),
        "off_stars_lost": round(off_lost, 2),
        "def_stars_lost": round(def_lost, 2),
        "net_stars_power": round(off_val + def_val, 2),
    }


def calculate_matchup_star_impact(
    home_team: str,
    away_team: str,
    stars_catalog: Optional[Dict[str, List[dict]]] = None,
    injuries: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Computes the net spread adjustment and total points impact due to star players and injuries.

    Positive spread_impact favors the HOME team.
    """
    h_ratings = get_team_star_ratings(home_team, stars_catalog, injuries)
    a_ratings = get_team_star_ratings(away_team, stars_catalog, injuries)

    # Home offense vs Away defense mismatch
    h_off_adv = h_ratings["off_stars_active"] - a_ratings["def_stars_active"]
    # Away offense vs Home defense mismatch
    a_off_adv = a_ratings["off_stars_active"] - h_ratings["def_stars_active"]

    # Net spread impact in points (positive = favors home)
    spread_impact = round((h_off_adv - a_off_adv) * 0.45, 2)

    # Total points impact (if both offenses missing stars, totals drop)
    tot_pts_lost = (h_ratings["off_stars_lost"] + a_ratings["off_stars_lost"]) * 0.65
    tot_pts_gained_by_missing_def = (h_ratings["def_stars_lost"] + a_ratings["def_stars_lost"]) * 0.45
    total_impact = round(tot_pts_gained_by_missing_def - tot_pts_lost, 2)

    return {
        "home_off_stars": h_ratings["off_stars_active"],
        "home_def_stars": h_ratings["def_stars_active"],
        "away_off_stars": a_ratings["off_stars_active"],
        "away_def_stars": a_ratings["def_stars_active"],
        "home_stars_lost": round(h_ratings["off_stars_lost"] + h_ratings["def_stars_lost"], 2),
        "away_stars_lost": round(a_ratings["off_stars_lost"] + a_ratings["def_stars_lost"], 2),
        "star_spread_adjustment": spread_impact,
        "star_total_adjustment": total_impact,
    }
