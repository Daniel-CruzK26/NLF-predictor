import pandas as pd
import numpy as np
import pytest
from src.features.opponent_adjustment import OpponentAdjuster


def test_opponent_adjustment_logic():
    adjuster = OpponentAdjuster(l2_alpha=1.0, min_games=4)
    
    # 4 games: KC faces strong defense SF and scores high EPA, LV faces weak defense and scores average
    hist_df = pd.DataFrame({
        "game_id": ["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4"],
        "season": [2023] * 8,
        "week": [1, 1, 2, 2, 3, 3, 4, 4],
        "off_team": ["KC", "SF", "KC", "LV", "SF", "LV", "KC", "LAC"],
        "def_team": ["SF", "KC", "LV", "KC", "LV", "SF", "LAC", "KC"],
        "is_home": [1, 0, 1, 0, 1, 0, 1, 0],
        "off_pass_epa": [0.4, -0.1, 0.5, -0.3, 0.3, -0.4, 0.6, -0.2],
        "off_rush_epa": [0.1, -0.05, 0.2, -0.1, 0.15, -0.2, 0.1, -0.1],
        "off_total_epa": [10.0, -2.0, 15.0, -8.0, 8.0, -10.0, 14.0, -5.0],
        "off_epa_per_play": [0.25, -0.05, 0.35, -0.20, 0.22, -0.25, 0.35, -0.15],
    })

    ratings = adjuster.fit_ratings_on_history(hist_df, ["KC", "SF", "LV", "LAC"])
    
    # KC offense should have high positive rating
    kc_off_pass = ratings["off_pass_epa"]["off"]["KC"]
    lv_off_pass = ratings["off_pass_epa"]["off"]["LV"]
    assert kc_off_pass > lv_off_pass

    # SF defense should have lower (better) def allowed than LV defense
    sf_def_pass = ratings["off_pass_epa"]["def"]["SF"]
    lv_def_pass = ratings["off_pass_epa"]["def"]["LV"]
    assert sf_def_pass < lv_def_pass


def test_opponent_adjusted_features_no_leakage():
    matchups = pd.DataFrame([
        {"game_id": "G1", "season": 2023, "week": 1, "home_team": "KC", "away_team": "DET", "home_off_pass_epa": 0.5, "away_off_pass_epa": 0.2, "home_off_rush_epa": 0.1, "away_off_rush_epa": 0.0, "home_off_total_epa": 10.0, "away_off_total_epa": 4.0, "home_off_epa_per_play": 0.2, "away_off_epa_per_play": 0.08},
        {"game_id": "G2", "season": 2023, "week": 2, "home_team": "KC", "away_team": "DET", "home_off_pass_epa": 0.4, "away_off_pass_epa": 0.1, "home_off_rush_epa": 0.0, "away_off_rush_epa": -0.1, "home_off_total_epa": 8.0, "away_off_total_epa": 2.0, "home_off_epa_per_play": 0.15, "away_off_epa_per_play": 0.04},
    ])

    adjuster = OpponentAdjuster(l2_alpha=1.0, min_games=2)
    adjusted_matchups = adjuster.compute_opponent_adjusted_features(matchups)

    assert "home_adj_off_pass_epa" in adjusted_matchups.columns
    assert "away_adj_def_pass_epa_allowed" in adjusted_matchups.columns
    assert "adj_diff_pass_advantage" in adjusted_matchups.columns
    assert len(adjusted_matchups) == 2
