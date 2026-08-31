import pandas as pd
import numpy as np
import pytest
from src.features.ewma import EWMAEngine


def test_ewma_anti_leakage():
    """
    Test that game N only sees data from games 1..N-1 and not game N itself.
    """
    team_games = pd.DataFrame([
        {"team": "KC", "season": 2023, "week": 1, "off_pass_epa": 1.0, "game_id": "G1"},
        {"team": "KC", "season": 2023, "week": 2, "off_pass_epa": 2.0, "game_id": "G2"},
        {"team": "KC", "season": 2023, "week": 3, "off_pass_epa": 0.5, "game_id": "G3"},
    ])
    
    engine = EWMAEngine(alpha=0.5, metrics=["off_pass_epa"])
    ewma_df = engine.compute_team_game_ewma(team_games)
    
    # Week 1: Prior value (mean of series)
    w1_ewma = ewma_df[ewma_df["week"] == 1]["ewma_off_pass_epa"].iloc[0]
    
    # Week 2: EWMA should be strictly based on Week 1 data (1.0)
    w2_ewma = ewma_df[ewma_df["week"] == 2]["ewma_off_pass_epa"].iloc[0]
    assert w2_ewma == 1.0
    
    # Week 3: EWMA should be based on Week 1 and Week 2 (0.5 * 2.0 + 0.5 * 1.0 = 1.5)
    w3_ewma = ewma_df[ewma_df["week"] == 3]["ewma_off_pass_epa"].iloc[0]
    assert pytest.approx(w3_ewma, 0.01) == 1.5


def test_ewma_matchup_enrichment():
    team_games = pd.DataFrame([
        {"team": "KC", "season": 2023, "week": 1, "off_pass_epa": 0.8, "off_rush_epa": 0.1, "def_pass_epa_allowed": -0.2, "def_rush_epa_allowed": -0.1, "game_id": "G1"},
        {"team": "DET", "season": 2023, "week": 1, "off_pass_epa": 0.4, "off_rush_epa": 0.3, "def_pass_epa_allowed": 0.1, "def_rush_epa_allowed": 0.0, "game_id": "G1"},
        {"team": "KC", "season": 2023, "week": 2, "off_pass_epa": 0.6, "off_rush_epa": 0.2, "def_pass_epa_allowed": -0.1, "def_rush_epa_allowed": -0.05, "game_id": "G2"},
        {"team": "DET", "season": 2023, "week": 2, "off_pass_epa": 0.5, "off_rush_epa": 0.25, "def_pass_epa_allowed": 0.05, "def_rush_epa_allowed": -0.05, "game_id": "G3"},
    ])
    
    matchups = pd.DataFrame([
        {"game_id": "G1", "season": 2023, "week": 1, "home_team": "KC", "away_team": "DET"},
        {"game_id": "G2", "season": 2023, "week": 2, "home_team": "KC", "away_team": "DET"},
    ])
    
    engine = EWMAEngine(alpha=0.5)
    team_ewma = engine.compute_team_game_ewma(team_games)
    enriched = engine.enrich_matchups_with_ewma(matchups, team_ewma)
    
    assert "home_ewma_off_pass_epa" in enriched.columns
    assert "away_ewma_def_pass_epa_allowed" in enriched.columns
    assert "matchup_diff_home_pass_advantage" in enriched.columns
    assert "net_pass_advantage" in enriched.columns
