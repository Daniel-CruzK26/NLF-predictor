import pandas as pd
import numpy as np
import pytest
from src.features.situational import add_situational_features
from src.models.trainer import TimeSeriesSpreadTrainer


def test_add_situational_features():
    df = pd.DataFrame([
        {"game_id": "G1", "roof": "dome", "wind": None, "temp": None, "home_rest": 10, "away_rest": 7, "home_ewma_off_pass_epa": 0.3, "away_ewma_off_pass_epa": 0.2},
        {"game_id": "G2", "roof": "outdoors", "wind": 18, "temp": 28, "home_rest": 6, "away_rest": 7, "home_ewma_off_pass_epa": 0.1, "away_ewma_off_pass_epa": 0.2},
    ])

    enriched = add_situational_features(df)
    assert enriched.iloc[0]["is_dome"] == 1
    assert enriched.iloc[0]["wind_speed"] == 0.0
    assert enriched.iloc[0]["temperature"] == 70.0
    assert enriched.iloc[0]["rest_differential"] == 3

    assert enriched.iloc[1]["is_dome"] == 0
    assert enriched.iloc[1]["high_wind_flag"] == 1
    assert enriched.iloc[1]["freezing_temp_flag"] == 1
    assert enriched.iloc[1]["rest_differential"] == -1
    assert "wind_pass_decay_interaction" in enriched.columns


def test_time_series_splits_and_trainer():
    # 3 seasons of mock games
    games = []
    for s in [2022, 2023, 2024]:
        for w in range(1, 10):
            games.append({
                "game_id": f"{s}_{w}",
                "season": s,
                "week": w,
                "elo_proj_spread": 3.0,
                "adj_net_epa_advantage": 0.2,
                "diff_qb_composite_score": 0.15,
                "actual_point_diff": 4,
                "spread_line": 3.5,
            })
    df_games = pd.DataFrame(games)

    trainer = TimeSeriesSpreadTrainer(
        feature_cols=["elo_proj_spread", "adj_net_epa_advantage", "diff_qb_composite_score"]
    )
    splits = trainer.get_time_series_splits(df_games)
    assert len(splits) == 2  # Train 2022->Test 2023, Train 2022-23->Test 2024

    clean_df, eval_res, feat_imp = trainer.train_and_evaluate(df_games, model_type="ridge")
    assert eval_res.n_games == 18  # 9 games in 2023 + 9 games in 2024
    assert eval_res.mae_spread >= 0
    assert len(feat_imp) == 3
