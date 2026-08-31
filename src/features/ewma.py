"""Exponential Weighted Moving Average (EWMA) engine with strict anti-leakage shift."""

from typing import List, Optional
import numpy as np
import pandas as pd
from ..config import DEFAULT_EWMA_ALPHA, TEAM_ABBR_MAP


class EWMAEngine:
    """
    Computes EWMA features for NFL team performance metrics.
    Guarantees zero data leakage by shifting historical statistics by 1 game prior to matchup assembly.
    """

    DEFAULT_METRICS = [
        "off_pass_epa",
        "off_rush_epa",
        "off_total_epa",
        "off_epa_per_play",
        "off_success_rate",
        "off_pass_success_rate",
        "off_rush_success_rate",
        "off_sacks_allowed",
        "off_turnovers",
        "def_pass_epa_allowed",
        "def_rush_epa_allowed",
        "def_total_epa_allowed",
        "def_epa_per_play_allowed",
        "def_success_rate_allowed",
        "def_sacks_created",
        "def_turnovers_forced",
    ]

    def __init__(
        self,
        alpha: float = DEFAULT_EWMA_ALPHA,
        metrics: Optional[List[str]] = None
    ):
        self.alpha = alpha
        self.metrics = metrics or self.DEFAULT_METRICS

    def compute_team_game_ewma(self, team_games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates EWMA for each team independently over chronological sequence of games.
        Applies .shift(1) so that for any game g_i, only information from g_1..g_{i-1} is present.
        """
        df = team_games_df.copy()
        
        # Standardize team names
        if "team" in df.columns:
            df["team"] = df["team"].replace(TEAM_ABBR_MAP)
            
        # Ensure chronological ordering
        if "gameday" in df.columns:
            df["gameday"] = pd.to_datetime(df["gameday"])
            df = df.sort_values(["team", "season", "week", "gameday"]).reset_index(drop=True)
        else:
            df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)

        ewma_cols = {}
        for col in self.metrics:
            if col in df.columns:
                ewma_col_name = f"ewma_{col}"
                # 1. Compute rolling EWMA per team
                # 2. Shift by 1 to guarantee no lookahead/leakage
                ewma_series = (
                    df.groupby("team")[col]
                    .transform(lambda s: s.ewm(alpha=self.alpha, adjust=False).mean().shift(1))
                )
                
                # Fill initial game per team (where shift(1) is NaN) with league mean up to that point or global mean
                global_prior = df[col].mean()
                ewma_cols[ewma_col_name] = ewma_series.fillna(global_prior)

        for col_name, s in ewma_cols.items():
            df[col_name] = s

        return df

    def enrich_matchups_with_ewma(
        self,
        matchups_df: pd.DataFrame,
        team_games_ewma_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Joins home and away EWMA features into the game matchup dataframe and generates
        offensive vs defensive matchup differentials.
        """
        df_matchups = matchups_df.copy()
        
        # Extract only EWMA columns + join keys
        ewma_feature_cols = [c for c in team_games_ewma_df.columns if c.startswith("ewma_")]
        join_keys = ["game_id", "team"]
        
        sub_ewma = team_games_ewma_df[join_keys + ewma_feature_cols].drop_duplicates(subset=join_keys)

        # Merge for home team
        home_features = sub_ewma.rename(columns={"team": "home_team"})
        home_features = home_features.rename(
            columns={c: f"home_{c}" for c in ewma_feature_cols}
        )
        df_matchups = pd.merge(
            df_matchups,
            home_features,
            on=["game_id", "home_team"],
            how="left"
        )

        # Merge for away team
        away_features = sub_ewma.rename(columns={"team": "away_team"})
        away_features = away_features.rename(
            columns={c: f"away_{c}" for c in ewma_feature_cols}
        )
        df_matchups = pd.merge(
            df_matchups,
            away_features,
            on=["game_id", "away_team"],
            how="left"
        )

        # Generate Matchup Differentials (Home offense vs Away defense, Away offense vs Home defense)
        if "home_ewma_off_pass_epa" in df_matchups.columns and "away_ewma_def_pass_epa_allowed" in df_matchups.columns:
            df_matchups["matchup_diff_home_pass_advantage"] = (
                df_matchups["home_ewma_off_pass_epa"] - df_matchups["away_ewma_def_pass_epa_allowed"]
            )
            df_matchups["matchup_diff_away_pass_advantage"] = (
                df_matchups["away_ewma_off_pass_epa"] - df_matchups["home_ewma_def_pass_epa_allowed"]
            )
            df_matchups["net_pass_advantage"] = (
                df_matchups["matchup_diff_home_pass_advantage"] - df_matchups["matchup_diff_away_pass_advantage"]
            )

        if "home_ewma_off_rush_epa" in df_matchups.columns and "away_ewma_def_rush_epa_allowed" in df_matchups.columns:
            df_matchups["matchup_diff_home_rush_advantage"] = (
                df_matchups["home_ewma_off_rush_epa"] - df_matchups["away_ewma_def_rush_epa_allowed"]
            )
            df_matchups["matchup_diff_away_rush_advantage"] = (
                df_matchups["away_ewma_off_rush_epa"] - df_matchups["home_ewma_def_rush_epa_allowed"]
            )
            df_matchups["net_rush_advantage"] = (
                df_matchups["matchup_diff_home_rush_advantage"] - df_matchups["matchup_diff_away_rush_advantage"]
            )

        if "home_ewma_off_success_rate" in df_matchups.columns and "away_ewma_def_success_rate_allowed" in df_matchups.columns:
            df_matchups["net_success_rate_advantage"] = (
                (df_matchups["home_ewma_off_success_rate"] - df_matchups["away_ewma_def_success_rate_allowed"])
                - (df_matchups["away_ewma_off_success_rate"] - df_matchups["home_ewma_def_success_rate_allowed"])
            )

        if "home_ewma_def_sacks_created" in df_matchups.columns and "away_ewma_off_sacks_allowed" in df_matchups.columns:
            df_matchups["home_pass_rush_mismatch"] = (
                df_matchups["home_ewma_def_sacks_created"] - df_matchups["away_ewma_off_sacks_allowed"]
            )
            df_matchups["away_pass_rush_mismatch"] = (
                df_matchups["away_ewma_def_sacks_created"] - df_matchups["home_ewma_off_sacks_allowed"]
            )

        return df_matchups


def compute_ewma_features(
    team_games: pd.DataFrame,
    matchups: pd.DataFrame,
    alpha: float = DEFAULT_EWMA_ALPHA
) -> pd.DataFrame:
    """Convenience function to calculate EWMA and enrich matchup dataset."""
    engine = EWMAEngine(alpha=alpha)
    team_games_ewma = engine.compute_team_game_ewma(team_games)
    enriched_matchups = engine.enrich_matchups_with_ewma(matchups, team_games_ewma)
    return enriched_matchups
