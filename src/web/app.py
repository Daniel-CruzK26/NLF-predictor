"""FastAPI backend application for NFL Predictive Modeling & Betting Dashboard."""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config import PROCESSED_DATA_DIR, ELO_INITIAL
from ..models.inference import LiveInferenceEngine, GamePrediction
from ..data.loader import load_schedules, standardize_team_abbr
from ..features.situational import add_situational_features

app = FastAPI(title="NFL Gridiron AI Predictor", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

# Global engine singleton
engine = LiveInferenceEngine()


@app.on_event("startup")
def startup_event():
    """Pre-train production model on startup."""
    print("🚀 Initializing LiveInferenceEngine for Web Dashboard...")
    engine.fit_production_models()
    print("✅ Model weights and team priors loaded successfully.")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "models_loaded": bool(engine.trained_models)}


@app.get("/api/predictions")
def get_predictions(season: int = 2026, week: int = 1, min_edge: float = 1.5):
    """
    Returns live predictions and value betting recommendations for the requested week.
    """
    cache_csv = PROCESSED_DATA_DIR / f"predictions_{season}_week_{week}.csv"
    if cache_csv.exists():
        df_preds = pd.read_csv(cache_csv)
        return {"season": season, "week": week, "predictions": df_preds.to_dict(orient="records")}

    # Fallback to computing live
    schedules = load_schedules([season])
    week_games = schedules[schedules["week"] == week].copy()
    if len(week_games) == 0:
        return {"season": season, "week": week, "predictions": []}

    predictions = engine.predict_slate(week_games, edge_threshold_low=min_edge)
    pred_dicts = [p.to_dict() for p in predictions]
    return {"season": season, "week": week, "predictions": pred_dicts}


@app.get("/api/teams")
def get_teams():
    """
    Returns power rankings, Elo ratings, and Opponent-Adjusted EPA metrics for all 32 teams.
    """
    teams_data = []
    
    # Team division and conference mapping
    conf_map = {
        "BUF": "AFC East", "MIA": "AFC East", "NE": "AFC East", "NYJ": "AFC East",
        "BAL": "AFC North", "CIN": "AFC North", "CLE": "AFC North", "PIT": "AFC North",
        "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "TEN": "AFC South",
        "DEN": "AFC West", "KC": "AFC West", "LV": "AFC West", "LAC": "AFC West",
        "DAL": "NFC East", "NYG": "NFC East", "PHI": "NFC East", "WAS": "NFC East",
        "CHI": "NFC North", "DET": "NFC North", "GB": "NFC North", "MIN": "NFC North",
        "ATL": "NFC South", "CAR": "NFC South", "NO": "NFC South", "TB": "NFC South",
        "ARI": "NFC West", "LA": "NFC West", "SF": "NFC West", "SEA": "NFC West",
    }

    team_names = {
        "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
        "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
        "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
        "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
        "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams", "MIA": "Miami Dolphins",
        "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
        "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
        "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
    }

    from ..config_rosters import get_active_starters
    starters = get_active_starters()

    for t in sorted(list(set(list(engine.team_priors.keys()) + list(engine.elo_ratings.keys())))):
        elo = engine.elo_ratings.get(t, ELO_INITIAL)
        priors = engine.team_priors.get(t, {})
        qb_name = starters.get(t, f"{t}_QB")
        qb_stat = engine.qb_priors.get(qb_name, {"composite": 0.0, "epa": 0.0, "cpoe": 0.0})
        
        net_epa = priors.get("adj_off_total_epa", 0.0) - priors.get("adj_def_total_epa_allowed", 0.0)
        qb_comp = qb_stat.get("composite", 0.0)
        
        # Composite Power Rating: 40% Net EPA + 35% QB Impact + 25% Elo Prior
        power_rating = round(1500.0 + (net_epa * 8.5) + (qb_comp * 320.0) + ((elo - 1500.0) * 0.25), 1)

        teams_data.append({
            "abbr": t,
            "name": team_names.get(t, t),
            "division": conf_map.get(t, "NFL"),
            "starting_qb": qb_name,
            "elo_rating": round(elo, 1),
            "power_rating": power_rating,
            "adj_off_pass_epa": round(priors.get("adj_off_pass_epa", 0.0), 3),
            "adj_def_pass_epa_allowed": round(priors.get("adj_def_pass_epa_allowed", 0.0), 3),
            "adj_off_rush_epa": round(priors.get("adj_off_rush_epa", 0.0), 3),
            "adj_def_rush_epa_allowed": round(priors.get("adj_def_rush_epa_allowed", 0.0), 3),
            "adj_off_total_epa": round(priors.get("adj_off_total_epa", 0.0), 2),
            "adj_def_total_epa_allowed": round(priors.get("adj_def_total_epa_allowed", 0.0), 2),
            "net_adj_epa": round(net_epa, 2),
            "off_success_rate": round(priors.get("ewma_off_success_rate", 0.45) * 100, 1),
            "def_success_rate_allowed": round(priors.get("ewma_def_success_rate_allowed", 0.45) * 100, 1),
        })

    teams_data.sort(key=lambda x: x["power_rating"], reverse=True)
    for rank, item in enumerate(teams_data, start=1):
        item["rank"] = rank

    return {"teams": teams_data}


@app.get("/api/models")
def get_models_benchmark():
    """
    Returns out-of-fold validation metrics for all models vs Vegas.
    """
    return {
        "models": [
            {"name": "Las Vegas Closing Line", "mae": 9.491, "rmse": 12.482, "su_win_pct": 68.2, "brier": 0.205, "ats_pct": 50.0, "type": "Benchmark"},
            {"name": "LightGBM Regressor", "mae": 9.719, "rmse": 12.696, "su_win_pct": 67.14, "brier": 0.2152, "ats_pct": 55.57, "type": "ML Tree"},
            {"name": "XGBoost Regressor", "mae": 9.843, "rmse": 12.866, "su_win_pct": 67.02, "brier": 0.2182, "ats_pct": 53.87, "type": "ML Tree"},
            {"name": "Blended Ensemble", "mae": 9.841, "rmse": 12.825, "su_win_pct": 66.67, "brier": 0.2197, "ats_pct": 53.51, "type": "Ensemble"},
            {"name": "Gradient Boosting (GBDT)", "mae": 9.936, "rmse": 12.970, "su_win_pct": 66.67, "brier": 0.2208, "ats_pct": 53.75, "type": "ML Tree"},
            {"name": "Ridge Regularized", "mae": 10.227, "rmse": 13.179, "su_win_pct": 64.44, "brier": 0.2318, "ats_pct": 53.27, "type": "Linear L2"},
            {"name": "Elo Baseline Model", "mae": 10.286, "rmse": 13.303, "su_win_pct": 61.97, "brier": 0.2270, "ats_pct": 48.87, "type": "Rating System"},
        ],
        "top_features": [
            {"feature": "Opponent-Adjusted Net EPA Advantage", "importance": 0.1065, "category": "EPA"},
            {"feature": "Home Adj Offensive EPA Total", "importance": 0.0440, "category": "EPA"},
            {"feature": "Away Adj Defensive EPA Allowed", "importance": 0.0379, "category": "Defense"},
            {"feature": "Away QB Composite Score (EPA + CPOE)", "importance": 0.0374, "category": "QB"},
            {"feature": "Away Rush Defense Allowed", "importance": 0.0368, "category": "Defense"},
            {"feature": "Elo Spread Projection", "importance": 0.0347, "category": "Elo"},
            {"feature": "Away Pass Defense Allowed", "importance": 0.0345, "category": "Defense"},
            {"feature": "QB EPA / Dropback Differential", "importance": 0.0336, "category": "QB"},
            {"feature": "Home Adj Pass EPA", "importance": 0.0327, "category": "Offense"},
            {"feature": "Home Offensive Success Rate", "importance": 0.0326, "category": "Offense"},
        ]
    }


@app.get("/api/rosters")
def get_rosters_api():
    """Returns active starting QBs for all 32 teams."""
    from ..config_rosters import get_active_starters
    return {"starters": get_active_starters()}


@app.post("/api/rosters")
def update_rosters_api(payload: Dict[str, str]):
    """Updates active starting QBs and refreshes engine priors."""
    from ..config_rosters import save_active_starters
    save_active_starters(payload)
    engine.fit_production_models()
    return {"status": "success", "starters": payload}


@app.post("/api/rosters/sync-espn")
def sync_rosters_espn_api():
    """Fetches real-time QB depth charts from ESPN API and re-fits production models."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from scripts.fetch_espn_rosters import fetch_all_espn_qbs
    from ..config_rosters import save_active_starters
    espn_starters = fetch_all_espn_qbs()
    clean_map = {}
    for k, v in espn_starters.items():
        team_key = "LA" if k == "LAR" else ("WAS" if k == "WSH" else k)
        clean_map[team_key] = v
    save_active_starters(clean_map)
    engine.fit_production_models()
    return {"status": "success", "starters": clean_map}


class SimulationRequest(BaseModel):
    home_team: str
    away_team: str
    home_qb: Optional[str] = None
    away_qb: Optional[str] = None
    vegas_spread: float = 0.0
    wind_speed: float = 7.0
    temperature: float = 70.0
    is_dome: int = 0
    rest_diff: int = 0


@app.post("/api/simulate")
def simulate_game(req: SimulationRequest):
    """
    Runs custom interactive game simulation.
    """
    mock_game = pd.DataFrame([{
        "game_id": f"CUSTOM_{req.away_team}_{req.home_team}",
        "home_team": req.home_team,
        "away_team": req.away_team,
        "home_qb_name": req.home_qb or f"{req.home_team}_QB",
        "away_qb_name": req.away_qb or f"{req.away_team}_QB",
        "spread_line": req.vegas_spread,
        "total_line": 45.0,
        "wind": req.wind_speed,
        "temp": req.temperature,
        "roof": "dome" if req.is_dome else "outdoors",
        "rest_differential": req.rest_diff,
        "gameday": "2026-09-13",
    }])

    predictions = engine.predict_slate(mock_game, edge_threshold_low=1.5)
    return predictions[0].to_dict()


# Serve Static UI
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")
