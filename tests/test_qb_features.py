import pandas as pd
import numpy as np
import pytest
from src.features.qb_features import QBFeatureEngine


def test_qb_ewma_and_backup_shrinkage():
    qb_games = pd.DataFrame([
        # Game 1: Mahomes 40 dropbacks, high performance
        {"game_id": "G1", "season": 2023, "week": 1, "team": "KC", "qb_id": "QB_MAHOMES", "qb_name": "P.Mahomes", "qb_dropbacks": 40, "qb_epa_per_dropback": 0.40, "qb_cpoe": 5.0, "qb_composite_score": 0.31, "qb_success_rate": 0.55, "qb_sack_rate": 0.02, "qb_turnover_rate": 0.0, "qb_air_yards_mean": 8.0},
        # Game 2: Mahomes 35 dropbacks
        {"game_id": "G2", "season": 2023, "week": 2, "team": "KC", "qb_id": "QB_MAHOMES", "qb_name": "P.Mahomes", "qb_dropbacks": 35, "qb_epa_per_dropback": 0.30, "qb_cpoe": 3.0, "qb_composite_score": 0.23, "qb_success_rate": 0.52, "qb_sack_rate": 0.03, "qb_turnover_rate": 0.01, "qb_air_yards_mean": 7.5},
        # Game 1: Rookie/Backup QB 5 dropbacks
        {"game_id": "G1", "season": 2023, "week": 1, "team": "LV", "qb_id": "QB_ROOKIE", "qb_name": "A.Rookie", "qb_dropbacks": 5, "qb_epa_per_dropback": 0.50, "qb_cpoe": 10.0, "qb_composite_score": 0.40, "qb_success_rate": 0.80, "qb_sack_rate": 0.0, "qb_turnover_rate": 0.0, "qb_air_yards_mean": 6.0},
        {"game_id": "G2", "season": 2023, "week": 2, "team": "LV", "qb_id": "QB_ROOKIE", "qb_name": "A.Rookie", "qb_dropbacks": 30, "qb_epa_per_dropback": -0.10, "qb_cpoe": -2.0, "qb_composite_score": -0.08, "qb_success_rate": 0.40, "qb_sack_rate": 0.05, "qb_turnover_rate": 0.03, "qb_air_yards_mean": 7.0},
    ])

    engine = QBFeatureEngine(alpha=0.2, shrinkage_dropbacks=50.0)
    df_ewma = engine.compute_qb_ewma(qb_games)

    # Week 1 Mahomes had 0 prior games -> defaulted to replacement prior
    mahomes_w1 = df_ewma[(df_ewma["qb_id"] == "QB_MAHOMES") & (df_ewma["week"] == 1)].iloc[0]
    assert mahomes_w1["qb_prior_dropbacks"] == 0

    # Week 2 Mahomes had 40 dropbacks -> score is shrunk towards replacement but positively reflects Game 1
    mahomes_w2 = df_ewma[(df_ewma["qb_id"] == "QB_MAHOMES") & (df_ewma["week"] == 2)].iloc[0]
    assert mahomes_w2["qb_prior_dropbacks"] == 40
    assert mahomes_w2["ewma_qb_composite_score"] > 0.0

    # Rookie with only 5 prior dropbacks in week 2 should be heavily shrunk towards replacement prior
    rookie_w2 = df_ewma[(df_ewma["qb_id"] == "QB_ROOKIE") & (df_ewma["week"] == 2)].iloc[0]
    assert rookie_w2["qb_prior_dropbacks"] == 5
    assert rookie_w2["ewma_qb_composite_score"] < mahomes_w2["ewma_qb_composite_score"]


def test_enrich_matchups_with_qb():
    matchups = pd.DataFrame([
        {"game_id": "G1", "season": 2023, "week": 1, "home_team": "KC", "away_team": "LV"},
    ])
    
    starters_ewma = pd.DataFrame([
        {"game_id": "G1", "team": "KC", "qb_id": "QB1", "qb_name": "P.Mahomes", "qb_prior_dropbacks": 500, "qb_prior_games": 20, "ewma_qb_composite_score": 0.25, "ewma_qb_epa_per_dropback": 0.30, "ewma_qb_cpoe": 4.5, "ewma_qb_sack_rate": 0.03},
        {"game_id": "G1", "team": "LV", "qb_id": "QB2", "qb_name": "J.Garoppolo", "qb_prior_dropbacks": 200, "qb_prior_games": 10, "ewma_qb_composite_score": 0.05, "ewma_qb_epa_per_dropback": 0.08, "ewma_qb_cpoe": 0.5, "ewma_qb_sack_rate": 0.08},
    ])

    engine = QBFeatureEngine()
    enriched = engine.enrich_matchups_with_qb(matchups, starters_ewma)

    assert "home_qb_name" in enriched.columns
    assert "away_qb_name" in enriched.columns
    assert enriched.iloc[0]["home_qb_name"] == "P.Mahomes"
    assert enriched.iloc[0]["away_qb_name"] == "J.Garoppolo"
    assert enriched.iloc[0]["diff_qb_composite_score"] > 0
    assert pytest.approx(enriched.iloc[0]["diff_qb_composite_score"], 0.01) == 0.20
