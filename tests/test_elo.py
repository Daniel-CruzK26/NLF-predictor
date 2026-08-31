import pandas as pd
import numpy as np
import pytest
from src.models.elo import EloModel
from src.models.evaluator import ModelEvaluator


def test_elo_initial_and_hfa():
    model = EloModel(initial_rating=1500.0, hfa=48.0, spread_divisor=25.0)
    
    assert model.get_rating("KC") == 1500.0
    assert model.get_rating("DET") == 1500.0
    
    # With equal ratings and HFA = 48, Home win probability should be > 50%
    prob_home = model.calculate_win_prob(1500.0, 1500.0)
    assert prob_home > 0.50
    assert pytest.approx(prob_home, 0.01) == 0.568
    
    # Spread should be 48 / 25 = 1.92 pts
    spread = model.calculate_spread_proj(1500.0, 1500.0)
    assert pytest.approx(spread, 0.01) == 1.92


def test_elo_mov_multiplier():
    model = EloModel()
    
    # Blowout victory (MOV = 30) should have higher multiplier than close game (MOV = 3)
    mult_close = model.calculate_mov_multiplier(3.0, 1500.0, 1500.0)
    mult_blowout = model.calculate_mov_multiplier(30.0, 1500.0, 1500.0)
    
    assert mult_blowout > mult_close
    assert mult_close > 0


def test_elo_simulation_and_mean_reversion():
    games = pd.DataFrame([
        {"game_id": "G1", "season": 2022, "week": 1, "home_team": "KC", "away_team": "LV", "home_score": 30, "away_score": 10},
        {"game_id": "G2", "season": 2022, "week": 2, "home_team": "KC", "away_team": "LAC", "home_score": 27, "away_score": 24},
        # New season (should trigger mean reversion)
        {"game_id": "G3", "season": 2023, "week": 1, "home_team": "KC", "away_team": "DET", "home_score": 20, "away_score": 21},
    ])
    
    model = EloModel()
    results = model.simulate_season(games)
    
    assert len(results) == 3
    assert "elo_proj_spread" in results.columns
    assert "home_elo_pre" in results.columns
    assert "home_elo_post" in results.columns
    
    # KC won 2 games in 2022, rating should increase
    kc_post_2022 = results.iloc[1]["home_elo_post"]
    assert kc_post_2022 > 1500.0
    
    # At start of 2023, KC rating should be reverted towards 1500 (lower than end of 2022)
    kc_pre_2023 = results.iloc[2]["home_elo_pre"]
    assert 1500.0 < kc_pre_2023 < kc_post_2022


def test_model_evaluator():
    df = pd.DataFrame({
        "actual_point_diff": [7, -3, 14, -10],
        "elo_proj_spread": [4.0, -2.5, 10.0, -7.0],
        "elo_prob_home_win": [0.65, 0.40, 0.80, 0.25],
        "spread_line": [3.0, -1.0, 7.0, -4.0],
    })
    
    res = ModelEvaluator.evaluate(df)
    assert res.n_games == 4
    assert res.mae_spread >= 0
    assert res.rmse_spread >= 0
    assert res.brier_score >= 0
    assert res.su_accuracy == 1.0  # All predictions picked right direction
    assert res.ats_accuracy is not None
