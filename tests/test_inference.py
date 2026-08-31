import pandas as pd
import numpy as np
import pytest
from src.models.inference import LiveInferenceEngine, GamePrediction


def test_live_inference_engine_execution():
    engine = LiveInferenceEngine()
    engine.fit_production_models()

    # Create mock upcoming games slate
    upcoming_games = pd.DataFrame([
        {"game_id": "2026_01_NE_SEA", "gameday": "2026-09-09", "home_team": "SEA", "away_team": "NE", "spread_line": 3.5, "total_line": 44.5, "roof": "outdoors", "wind": 5.0, "temp": 68.0, "home_qb_name": "G.Smith", "away_qb_name": "D.Maye"},
        {"game_id": "2026_01_WAS_PHI", "gameday": "2026-09-13", "home_team": "PHI", "away_team": "WAS", "spread_line": 4.5, "total_line": 45.5, "roof": "outdoors", "wind": 8.0, "temp": 75.0, "home_qb_name": "J.Hurts", "away_qb_name": "J.Daniels"},
    ])

    predictions = engine.predict_slate(upcoming_games)
    assert len(predictions) == 2
    assert isinstance(predictions[0], GamePrediction)
    assert predictions[0].home_team == "SEA"
    assert predictions[0].home_win_prob >= 0.0
    assert predictions[0].home_win_prob <= 1.0
    assert predictions[0].recommendation is not None
    assert predictions[0].confidence in ["HIGH (3★)", "MEDIUM (2★)", "LOW (1★)", "NO VALUE"]
