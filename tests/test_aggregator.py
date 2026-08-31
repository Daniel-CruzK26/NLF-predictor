import pandas as pd
import numpy as np
import pytest
from src.data.aggregator import filter_valid_plays, aggregate_pbp_to_team_games, build_game_matchup_dataset


def test_filter_valid_plays():
    sample_pbp = pd.DataFrame([
        # Valid pass play
        {"game_id": "2023_01_DET_KC", "posteam": "KC", "defteam": "DET", "play_type": "pass", "epa": 0.5, "two_point_attempt": 0, "qb_kneel": 0, "qb_spike": 0, "sack": 0, "qb_hit": 0},
        # Valid run play
        {"game_id": "2023_01_DET_KC", "posteam": "KC", "defteam": "DET", "play_type": "run", "epa": -0.2, "two_point_attempt": 0, "qb_kneel": 0, "qb_spike": 0, "sack": 0, "qb_hit": 0},
        # Invalid knee/spike play
        {"game_id": "2023_01_DET_KC", "posteam": "KC", "defteam": "DET", "play_type": "qb_kneel", "epa": -0.1, "two_point_attempt": 0, "qb_kneel": 1, "qb_spike": 0, "sack": 0, "qb_hit": 0},
        # Null epa
        {"game_id": "2023_01_DET_KC", "posteam": "KC", "defteam": "DET", "play_type": "pass", "epa": None, "two_point_attempt": 0, "qb_kneel": 0, "qb_spike": 0, "sack": 0, "qb_hit": 0},
    ])
    
    filtered = filter_valid_plays(sample_pbp)
    assert len(filtered) == 2
    assert "is_pass" in filtered.columns
    assert "is_rush" in filtered.columns
    assert "is_success" in filtered.columns


def test_aggregate_pbp_to_team_games():
    sample_pbp = pd.DataFrame([
        {"game_id": "2023_01_DET_KC", "posteam": "KC", "defteam": "DET", "play_type": "pass", "epa": 1.0, "two_point_attempt": 0, "qb_kneel": 0, "qb_spike": 0, "sack": 0, "qb_hit": 1, "cpoe": 5.0},
        {"game_id": "2023_01_DET_KC", "posteam": "KC", "defteam": "DET", "play_type": "run", "epa": -0.5, "two_point_attempt": 0, "qb_kneel": 0, "qb_spike": 0, "sack": 0, "qb_hit": 0},
        {"game_id": "2023_01_DET_KC", "posteam": "DET", "defteam": "KC", "play_type": "pass", "epa": 0.2, "two_point_attempt": 0, "qb_kneel": 0, "qb_spike": 0, "sack": 1, "qb_hit": 1, "cpoe": 2.0},
    ])
    
    tg = aggregate_pbp_to_team_games(sample_pbp)
    assert len(tg) == 2  # KC and DET
    
    kc_stats = tg[tg["team"] == "KC"].iloc[0]
    assert kc_stats["off_plays"] == 2
    assert kc_stats["off_pass_plays"] == 1
    assert kc_stats["off_rush_plays"] == 1
    assert kc_stats["off_pass_epa"] == 1.0
    assert kc_stats["off_rush_epa"] == -0.5
    assert kc_stats["off_total_epa"] == 0.5
    assert kc_stats["def_plays_faced"] == 1
    assert kc_stats["def_sacks_created"] == 1


def test_build_game_matchup_dataset():
    tg = pd.DataFrame([
        {"game_id": "2023_01_DET_KC", "team": "KC", "off_total_epa": 5.0, "def_total_epa_allowed": 3.0},
        {"game_id": "2023_01_DET_KC", "team": "DET", "off_total_epa": 3.0, "def_total_epa_allowed": 5.0},
    ])
    
    sched = pd.DataFrame([
        {"game_id": "2023_01_DET_KC", "season": 2023, "week": 1, "gameday": "2023-09-07", "home_team": "KC", "away_team": "DET", "home_score": 20, "away_score": 21, "spread_line": 4.5}
    ])
    
    matchups = build_game_matchup_dataset(tg, sched)
    assert len(matchups) == 1
    assert matchups.iloc[0]["actual_point_diff"] == -1
    assert matchups.iloc[0]["home_off_total_epa"] == 5.0
    assert matchups.iloc[0]["away_off_total_epa"] == 3.0
    assert "rest_differential" in matchups.columns
