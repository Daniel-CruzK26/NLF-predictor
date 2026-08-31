"""Configuration constants and path definitions for the NFL Predictor."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Seasons configuration
DEFAULT_SEASONS = list(range(2016, 2025))

# Team mapping / standard abbreviations (handling historical moves/renames)
TEAM_ABBR_MAP = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "WSH": "WAS",
}

# Elo Model Hyperparameters (FiveThirtyEight style)
ELO_INITIAL = 1500.0
ELO_K_FACTOR = 20.0
ELO_HFA = 48.0  # Home Field Advantage in Elo rating points (~ 2.0-2.5 pts of spread)
ELO_SEASON_REVERSION = 0.33  # Weight of reverting back to 1500 across seasons
ELO_SPREAD_DIVISOR = 25.0  # Elo points per 1 point of spread

# EWMA Parameters
DEFAULT_EWMA_ALPHA = 0.15  # Decay rate (~5-7 games half-life)
DEFAULT_EWMA_SPAN = 8
