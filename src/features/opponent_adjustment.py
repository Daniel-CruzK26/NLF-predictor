"""Opponent-Adjusted EPA Engine using regularized Ridge regression with strict time-series splits."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class OpponentAdjuster:
    """
    Fits Ridge regression models to decouple team offensive and defensive EPA from schedule strength.
    Guarantees no data leakage by training only on games completed strictly prior to each week.
    """

    METRICS = ["off_pass_epa", "off_rush_epa", "off_total_epa", "off_epa_per_play"]

    def __init__(self, l2_alpha: float = 3.0, min_games: int = 16):
        self.l2_alpha = l2_alpha
        self.min_games = min_games

    def _prepare_team_matchup_long(self, matchups_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts matchups into a long format where each row represents one team's offensive performance
        against an opponent defense in a specific game.
        """
        df = matchups_df.copy()
        
        # Home offense vs Away defense
        home_rows = pd.DataFrame({
            "game_id": df["game_id"],
            "season": df["season"],
            "week": df["week"],
            "gameday": df.get("gameday"),
            "off_team": df["home_team"],
            "def_team": df["away_team"],
            "is_home": 1,
            "off_pass_epa": df.get("home_off_pass_epa", 0.0),
            "off_rush_epa": df.get("home_off_rush_epa", 0.0),
            "off_total_epa": df.get("home_off_total_epa", 0.0),
            "off_epa_per_play": df.get("home_off_epa_per_play", 0.0),
        })

        # Away offense vs Home defense
        away_rows = pd.DataFrame({
            "game_id": df["game_id"],
            "season": df["season"],
            "week": df["week"],
            "gameday": df.get("gameday"),
            "off_team": df["away_team"],
            "def_team": df["home_team"],
            "is_home": 0,
            "off_pass_epa": df.get("away_off_pass_epa", 0.0),
            "off_rush_epa": df.get("away_off_rush_epa", 0.0),
            "off_total_epa": df.get("away_off_total_epa", 0.0),
            "off_epa_per_play": df.get("away_off_epa_per_play", 0.0),
        })

        long_df = pd.concat([home_rows, away_rows], ignore_index=True)
        if "gameday" in long_df.columns and long_df["gameday"].notna().any():
            long_df["gameday"] = pd.to_datetime(long_df["gameday"])
            long_df = long_df.sort_values(["gameday", "season", "week"]).reset_index(drop=True)
        else:
            long_df = long_df.sort_values(["season", "week"]).reset_index(drop=True)

        return long_df

    def fit_ratings_on_history(
        self,
        historical_long_df: pd.DataFrame,
        all_teams: List[str]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Fits Ridge regression for each metric on historical games.
        Model: EPA_ij = mu + Off_i + Def_j + beta_HFA * is_home
        Returns ratings dict: {metric: {"off": {team: val}, "def": {team: val}}}
        """
        if len(historical_long_df) < self.min_games:
            # Not enough data: return neutral ratings (0.0)
            return {
                m: {"off": {t: 0.0 for t in all_teams}, "def": {t: 0.0 for t in all_teams}}
                for m in self.METRICS
            }

        team_to_idx = {t: i for i, t in enumerate(all_teams)}
        n_teams = len(all_teams)
        n_samples = len(historical_long_df)

        # Build feature matrix X: [Offense One-Hot (n_teams), Defense One-Hot (n_teams), is_home (1)]
        X = np.zeros((n_samples, 2 * n_teams + 1))
        
        for idx, row in historical_long_df.reset_index(drop=True).iterrows():
            off_t = row["off_team"]
            def_t = row["def_team"]
            if off_t in team_to_idx:
                X[idx, team_to_idx[off_t]] = 1.0
            if def_t in team_to_idx:
                X[idx, n_teams + team_to_idx[def_t]] = 1.0
            X[idx, -1] = float(row["is_home"])

        ratings_by_metric = {}
        for m in self.METRICS:
            if m not in historical_long_df.columns:
                continue
            y = historical_long_df[m].fillna(0.0).values
            
            # Ridge regression with L2 regularization
            ridge = Ridge(alpha=self.l2_alpha, fit_intercept=True)
            ridge.fit(X, y)
            
            coefs = ridge.coef_
            off_coefs = coefs[:n_teams]
            def_coefs = coefs[n_teams : 2 * n_teams]
            
            ratings_by_metric[m] = {
                "off": {t: float(off_coefs[i]) for t, i in team_to_idx.items()},
                # Defense coef > 0 means opponent scored more (weaker defense), so def_allowed = def_coef
                "def": {t: float(def_coefs[i]) for t, i in team_to_idx.items()},
            }

        return ratings_by_metric

    def compute_opponent_adjusted_features(
        self,
        matchups_df: pd.DataFrame,
        window_games: int = 300
    ) -> pd.DataFrame:
        """
        Computes rolling opponent-adjusted EPA for each game chronologically,
        fitting Ridge solely on prior games.
        """
        df_matchups = matchups_df.copy()
        long_df = self._prepare_team_matchup_long(df_matchups)
        
        all_teams = sorted(list(set(df_matchups["home_team"].dropna().unique()).union(
            set(df_matchups["away_team"].dropna().unique())
        )))

        # Ensure order
        if "gameday" in df_matchups.columns:
            df_matchups["gameday"] = pd.to_datetime(df_matchups["gameday"])
            df_matchups = df_matchups.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
        else:
            df_matchups = df_matchups.sort_values(["season", "week"]).reset_index(drop=True)

        home_adj_off_pass = []
        home_adj_def_pass = []
        away_adj_off_pass = []
        away_adj_def_pass = []

        home_adj_off_rush = []
        home_adj_def_rush = []
        away_adj_off_rush = []
        away_adj_def_rush = []

        home_adj_off_total = []
        home_adj_def_total = []
        away_adj_off_total = []
        away_adj_def_total = []

        for idx, row in df_matchups.iterrows():
            g_id = row["game_id"]
            h_team = row["home_team"]
            a_team = row["away_team"]

            # Filter historical long df strictly before this game
            # Long df has 2 rows per game, find all prior games
            prior_long = long_df[long_df["game_id"] != g_id]
            # Match games up to the current row index in matchups
            prior_long = long_df.iloc[: 2 * idx]

            if len(prior_long) > window_games * 2:
                prior_long = prior_long.iloc[-window_games * 2 :]

            ratings = self.fit_ratings_on_history(prior_long, all_teams)

            # Pass EPA ratings
            pass_rat = ratings.get("off_pass_epa", {"off": {}, "def": {}})
            home_adj_off_pass.append(pass_rat["off"].get(h_team, 0.0))
            home_adj_def_pass.append(pass_rat["def"].get(h_team, 0.0))
            away_adj_off_pass.append(pass_rat["off"].get(a_team, 0.0))
            away_adj_def_pass.append(pass_rat["def"].get(a_team, 0.0))

            # Rush EPA ratings
            rush_rat = ratings.get("off_rush_epa", {"off": {}, "def": {}})
            home_adj_off_rush.append(rush_rat["off"].get(h_team, 0.0))
            home_adj_def_rush.append(rush_rat["def"].get(h_team, 0.0))
            away_adj_off_rush.append(rush_rat["off"].get(a_team, 0.0))
            away_adj_def_rush.append(rush_rat["def"].get(a_team, 0.0))

            # Total EPA ratings
            total_rat = ratings.get("off_total_epa", {"off": {}, "def": {}})
            home_adj_off_total.append(total_rat["off"].get(h_team, 0.0))
            home_adj_def_total.append(total_rat["def"].get(h_team, 0.0))
            away_adj_off_total.append(total_rat["off"].get(a_team, 0.0))
            away_adj_def_total.append(total_rat["def"].get(a_team, 0.0))

        # Assign back to matchups dataframe
        df_matchups["home_adj_off_pass_epa"] = home_adj_off_pass
        df_matchups["home_adj_def_pass_epa_allowed"] = home_adj_def_pass
        df_matchups["away_adj_off_pass_epa"] = away_adj_off_pass
        df_matchups["away_adj_def_pass_epa_allowed"] = away_adj_def_pass

        df_matchups["home_adj_off_rush_epa"] = home_adj_off_rush
        df_matchups["home_adj_def_rush_epa_allowed"] = home_adj_def_rush
        df_matchups["away_adj_off_rush_epa"] = away_adj_off_rush
        df_matchups["away_adj_def_rush_epa_allowed"] = away_adj_def_rush

        df_matchups["home_adj_off_total_epa"] = home_adj_off_total
        df_matchups["home_adj_def_total_epa_allowed"] = home_adj_def_total
        df_matchups["away_adj_off_total_epa"] = away_adj_off_total
        df_matchups["away_adj_def_total_epa_allowed"] = away_adj_def_total

        # Opponent-Adjusted Matchup Differentials
        df_matchups["adj_diff_pass_advantage"] = (
            (df_matchups["home_adj_off_pass_epa"] - df_matchups["away_adj_def_pass_epa_allowed"])
            - (df_matchups["away_adj_off_pass_epa"] - df_matchups["home_adj_def_pass_epa_allowed"])
        )
        df_matchups["adj_diff_rush_advantage"] = (
            (df_matchups["home_adj_off_rush_epa"] - df_matchups["away_adj_def_rush_epa_allowed"])
            - (df_matchups["away_adj_off_rush_epa"] - df_matchups["home_adj_def_rush_epa_allowed"])
        )
        df_matchups["adj_net_epa_advantage"] = (
            (df_matchups["home_adj_off_total_epa"] - df_matchups["home_adj_def_total_epa_allowed"])
            - (df_matchups["away_adj_off_total_epa"] - df_matchups["away_adj_def_total_epa_allowed"])
        )

        return df_matchups
