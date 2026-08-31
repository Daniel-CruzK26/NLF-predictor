import pandas as pd
import numpy as np
import pytest
from src.data.qb_aggregator import extract_qb_dropbacks, aggregate_qb_game_stats, get_starting_qbs_per_game


def test_extract_qb_dropbacks():
    sample_pbp = pd.DataFrame([
        # Standard pass
        {"game_id": "G1", "posteam": "KC", "defteam": "DET", "play_type": "pass", "pass_attempt": 1, "passer_player_id": "00-0033873", "passer_player_name": "P.Mahomes", "epa": 0.8, "two_point_attempt": 0},
        # Scramble
        {"game_id": "G1", "posteam": "KC", "defteam": "DET", "play_type": "run", "qb_scramble": 1, "rusher_player_id": "00-0033873", "rusher_player_name": "P.Mahomes", "epa": 0.4, "two_point_attempt": 0},
        # Sack
        {"game_id": "G1", "posteam": "KC", "defteam": "DET", "play_type": "pass", "sack": 1, "passer_player_id": "00-0033873", "passer_player_name": "P.Mahomes", "epa": -1.2, "two_point_attempt": 0},
        # Regular running back rush (not a dropback)
        {"game_id": "G1", "posteam": "KC", "defteam": "DET", "play_type": "run", "pass_attempt": 0, "qb_scramble": 0, "sack": 0, "rusher_player_id": "00-0038134", "rusher_player_name": "I.Pacheco", "epa": 0.1, "two_point_attempt": 0},
    ])

    dropbacks = extract_qb_dropbacks(sample_pbp)
    assert len(dropbacks) == 3
    assert (dropbacks["qb_name"] == "P.Mahomes").all()


def test_aggregate_qb_game_stats_and_starters():
    sample_pbp = pd.DataFrame([
        # Mahomes 2 dropbacks
        {"game_id": "G1", "posteam": "KC", "defteam": "DET", "play_type": "pass", "pass_attempt": 1, "complete_pass": 1, "passing_yards": 25, "passer_player_id": "QB1", "passer_player_name": "P.Mahomes", "epa": 1.0, "cpoe": 6.0, "two_point_attempt": 0},
        {"game_id": "G1", "posteam": "KC", "defteam": "DET", "play_type": "pass", "pass_attempt": 1, "complete_pass": 0, "passing_yards": 0, "passer_player_id": "QB1", "passer_player_name": "P.Mahomes", "epa": -0.4, "cpoe": -4.0, "two_point_attempt": 0},
        # Backup QB Gabbert 1 dropback
        {"game_id": "G1", "posteam": "KC", "defteam": "DET", "play_type": "pass", "pass_attempt": 1, "complete_pass": 1, "passing_yards": 8, "passer_player_id": "QB2", "passer_player_name": "B.Gabbert", "epa": 0.1, "cpoe": 1.0, "two_point_attempt": 0},
        # Goff 2 dropbacks
        {"game_id": "G1", "posteam": "DET", "defteam": "KC", "play_type": "pass", "pass_attempt": 1, "complete_pass": 1, "passing_yards": 15, "passer_player_id": "QB3", "passer_player_name": "J.Goff", "epa": 0.5, "cpoe": 3.0, "two_point_attempt": 0},
        {"game_id": "G1", "posteam": "DET", "defteam": "KC", "play_type": "pass", "pass_attempt": 1, "sack": 1, "passer_player_id": "QB3", "passer_player_name": "J.Goff", "epa": -0.8, "two_point_attempt": 0},
    ])

    qb_stats = aggregate_qb_game_stats(sample_pbp)
    assert len(qb_stats) == 3

    starters = get_starting_qbs_per_game(qb_stats)
    assert len(starters) == 2  # KC starter is Mahomes, DET starter is Goff
    
    kc_starter = starters[starters["team"] == "KC"].iloc[0]
    assert kc_starter["qb_name"] == "P.Mahomes"
    assert kc_starter["qb_dropbacks"] == 2
    assert kc_starter["qb_composite_score"] > 0
