"""Quarterback (QB) feature engineering with anti-leakage EWMA and backup shrinkage."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ..config import DEFAULT_EWMA_ALPHA, TEAM_ABBR_MAP


class QBFeatureEngine:
    """
    Computes rolling performance metrics for individual Quarterbacks with Bayesian-style shrinkage
    towards replacement-level for backups or inexperienced passers.
    """

    QB_METRICS = [
        "qb_epa_per_dropback",
        "qb_cpoe",
        "qb_composite_score",
        "qb_success_rate",
        "qb_sack_rate",
        "qb_turnover_rate",
        "qb_air_yards_mean",
    ]

    def __init__(
        self,
        alpha: float = DEFAULT_EWMA_ALPHA,
        shrinkage_dropbacks: float = 50.0,
        replacement_epa: float = -0.10,
        replacement_cpoe: float = -3.0,
        replacement_composite: float = -0.08,
    ):
        self.alpha = alpha
        self.shrinkage_dropbacks = shrinkage_dropbacks
        self.replacement_epa = replacement_epa
        self.replacement_cpoe = replacement_cpoe
        self.replacement_composite = replacement_composite

    def compute_qb_ewma(self, qb_games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes EWMA for each QB across their chronological sequence of games.
        Applies .shift(1) by qb_id so that game N only uses games 1..N-1.
        """
        df = qb_games_df.copy()
        
        # Standardize team names
        if "team" in df.columns:
            df["team"] = df["team"].replace(TEAM_ABBR_MAP)

        # Sort chronologically by QB
        sort_cols = ["qb_id"]
        if "season" in df.columns:
            sort_cols.append("season")
        if "week" in df.columns:
            sort_cols.append("week")
        if "gameday" in df.columns:
            sort_cols.append("gameday")
            
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # 1. Prior cumulative dropbacks before current game
        df["qb_prior_dropbacks"] = (
            df.groupby("qb_id")["qb_dropbacks"]
            .transform(lambda s: s.cumsum().shift(1))
            .fillna(0.0)
        )
        df["qb_prior_games"] = (
            df.groupby("qb_id").cumcount()
        )

        # 2. Compute rolling EWMA per metric
        for m in self.QB_METRICS:
            if m not in df.columns:
                continue
            ewma_col = f"ewma_{m}"
            
            # Prior rolling mean
            raw_ewma = (
                df.groupby("qb_id")[m]
                .transform(lambda s: s.ewm(alpha=self.alpha, adjust=False).mean().shift(1))
            )
            
            # Default prior for QBs with no historical games
            if m == "qb_cpoe":
                prior_val = self.replacement_cpoe
            elif m == "qb_composite_score":
                prior_val = self.replacement_composite
            elif m == "qb_epa_per_dropback":
                prior_val = self.replacement_epa
            else:
                prior_val = df[m].mean()

            raw_ewma = raw_ewma.fillna(prior_val)

            # 3. Shrinkage towards replacement level for low sample sizes (backups / rookies)
            weight = df["qb_prior_dropbacks"] / (
                df["qb_prior_dropbacks"] + self.shrinkage_dropbacks
            )
            shrunk_ewma = (weight * raw_ewma) + ((1.0 - weight) * prior_val)
            
            df[ewma_col] = shrunk_ewma

        return df

    def enrich_matchups_with_qb(
        self,
        matchups_df: pd.DataFrame,
        qb_starters_ewma_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges home and away starting QB features into the matchups dataframe
        and generates QB matchup differentials.
        """
        df_matchups = matchups_df.copy()

        qb_cols_to_keep = [
            "game_id",
            "team",
            "qb_id",
            "qb_name",
            "qb_prior_dropbacks",
            "qb_prior_games",
        ] + [c for c in qb_starters_ewma_df.columns if c.startswith("ewma_qb_")]

        sub_qb = qb_starters_ewma_df[qb_cols_to_keep].drop_duplicates(subset=["game_id", "team"])

        # Merge for Home QB
        home_qb = sub_qb.rename(columns={"team": "home_team"})
        home_qb = home_qb.rename(
            columns={c: f"home_{c}" for c in home_qb.columns if c not in ["game_id", "home_team"]}
        )
        df_matchups = pd.merge(df_matchups, home_qb, on=["game_id", "home_team"], how="left")

        # Merge for Away QB
        away_qb = sub_qb.rename(columns={"team": "away_team"})
        away_qb = away_qb.rename(
            columns={c: f"away_{c}" for c in away_qb.columns if c not in ["game_id", "away_team"]}
        )
        df_matchups = pd.merge(df_matchups, away_qb, on=["game_id", "away_team"], how="left")

        # Fill any missing QBs with replacement level
        df_matchups["home_ewma_qb_composite_score"] = df_matchups["home_ewma_qb_composite_score"].fillna(
            self.replacement_composite
        )
        df_matchups["away_ewma_qb_composite_score"] = df_matchups["away_ewma_qb_composite_score"].fillna(
            self.replacement_composite
        )
        df_matchups["home_ewma_qb_epa_per_dropback"] = df_matchups["home_ewma_qb_epa_per_dropback"].fillna(
            self.replacement_epa
        )
        df_matchups["away_ewma_qb_epa_per_dropback"] = df_matchups["away_ewma_qb_epa_per_dropback"].fillna(
            self.replacement_epa
        )

        # Differential QB Features
        df_matchups["diff_qb_composite_score"] = (
            df_matchups["home_ewma_qb_composite_score"] - df_matchups["away_ewma_qb_composite_score"]
        )
        df_matchups["diff_qb_epa_per_dropback"] = (
            df_matchups["home_ewma_qb_epa_per_dropback"] - df_matchups["away_ewma_qb_epa_per_dropback"]
        )
        
        if "home_ewma_qb_cpoe" in df_matchups.columns and "away_ewma_qb_cpoe" in df_matchups.columns:
            df_matchups["diff_qb_cpoe"] = (
                df_matchups["home_ewma_qb_cpoe"].fillna(self.replacement_cpoe)
                - df_matchups["away_ewma_qb_cpoe"].fillna(self.replacement_cpoe)
            )

        if "home_ewma_qb_sack_rate" in df_matchups.columns and "away_ewma_qb_sack_rate" in df_matchups.columns:
            # Positive diff means home QB is better at avoiding sacks
            df_matchups["diff_qb_sack_avoidance"] = (
                df_matchups["away_ewma_qb_sack_rate"].fillna(0.08)
                - df_matchups["home_ewma_qb_sack_rate"].fillna(0.08)
            )

        return df_matchups
