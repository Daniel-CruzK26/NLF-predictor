"""Active starting Quarterbacks configuration by team for current NFL season."""

import json
from pathlib import Path
from typing import Dict

ROSTERS_FILE = Path(__file__).resolve().parent.parent / "data" / "active_starters.json"

# Default updated starters for the current NFL season
DEFAULT_2026_STARTERS: Dict[str, str] = {
    # AFC East
    "BUF": "J.Allen",
    "MIA": "M.Willis",
    "NYJ": "G.Smith",        # Geno Smith with NY Jets
    "NE": "D.Maye",          # Drake Maye
    
    # AFC North
    "BAL": "L.Jackson",
    "CIN": "J.Burrow",
    "CLE": "D.Watson",
    "PIT": "A.Rodgers",
    
    # AFC South
    "HOU": "C.Stroud",
    "IND": "D.Jones",
    "JAX": "T.Lawrence",
    "TEN": "C.Ward",
    
    # AFC West
    "KC": "P.Mahomes",
    "LAC": "J.Herbert",
    "DEN": "B.Nix",
    "LV": "K.Cousins",
    
    # NFC East
    "PHI": "J.Hurts",
    "WAS": "J.Daniels",
    "DAL": "D.Prescott",
    "NYG": "J.Dart",
    
    # NFC North
    "DET": "J.Goff",
    "GB": "J.Love",
    "CHI": "C.Williams",
    "MIN": "J.McCarthy",
    
    # NFC South
    "ATL": "T.Tagovailoa",
    "TB": "B.Mayfield",
    "NO": "T.Shough",
    "CAR": "B.Young",
    
    # NFC West
    "LA": "M.Stafford",      # Los Angeles Rams (Top Contender)
    "SF": "B.Purdy",
    "SEA": "S.Darnold",      # Sam Darnold with Seattle Seahawks
    "ARI": "J.Brissett",
}


def get_active_starters() -> Dict[str, str]:
    """Returns active starting QBs, loading from JSON file if available or default."""
    if ROSTERS_FILE.exists():
        try:
            with open(ROSTERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_2026_STARTERS.copy()


def save_active_starters(starters: Dict[str, str]) -> None:
    """Saves customized starting QBs to disk."""
    ROSTERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ROSTERS_FILE, "w") as f:
        json.dump(starters, f, indent=2)
