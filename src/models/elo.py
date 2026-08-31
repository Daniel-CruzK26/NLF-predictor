"""FiveThirtyEight style NFL Elo Rating System with MOV multiplier and HFA."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ..config import (
    ELO_INITIAL,
    ELO_K_FACTOR,
    ELO_HFA,
    ELO_SEASON_REVERSION,
    ELO_SPREAD_DIVISOR,
    TEAM_ABBR_MAP,
)


class EloModel:
    """
    NFL Elo Rating Model adjusted for Margin of Victory (MOV) and Home Field Advantage (HFA).
    Calculates pre-game ratings and projections in strict chronological order to avoid data leakage.
    """

    def __init__(
        self,
        initial_rating: float = ELO_INITIAL,
        k_factor: float = ELO_K_FACTOR,
        hfa: float = ELO_HFA,
        season_reversion: float = ELO_SEASON_REVERSION,
        spread_divisor: float = ELO_SPREAD_DIVISOR,
    ):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.hfa = hfa
        self.season_reversion = season_reversion
        self.spread_divisor = spread_divisor
        
        # State: current rating for each team
        self.ratings: Dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        """Get current rating for a team, defaulting to initial rating."""
        norm_team = TEAM_ABBR_MAP.get(team, team)
        return self.ratings.get(norm_team, self.initial_rating)

    def calculate_win_prob(self, home_elo: float, away_elo: float, hfa: Optional[float] = None) -> float:
        """
        Calculate expected win probability for home team.
        P(Home) = 1 / (1 + 10 ^ (-(home_elo + HFA - away_elo) / 400))
        """
        if hfa is None:
            hfa = self.hfa
        elo_diff = (home_elo + hfa) - away_elo
        return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    def calculate_spread_proj(self, home_elo: float, away_elo: float, hfa: Optional[float] = None) -> float:
        """
        Projected point spread for home team (Positive = Home favored, Negative = Away favored).
        Spread = (home_elo + HFA - away_elo) / spread_divisor
        """
        if hfa is None:
            hfa = self.hfa
        return (home_elo + hfa - away_elo) / self.spread_divisor

    def calculate_mov_multiplier(self, mov: float, elo_winner: float, elo_loser: float) -> float:
        """
        Calculate Margin of Victory (MOV) multiplier to scale Elo updates.
        Formula: ln(|MOV| + 1) * (2.2 / ((elo_winner - elo_loser) * 0.001 + 2.2))
        """
        elo_diff = elo_winner - elo_loser
        denominator = (elo_diff * 0.001) + 2.2
        if denominator <= 0:
            denominator = 0.01
        return np.log(abs(mov) + 1.0) * (2.2 / denominator)

    def revert_towards_mean(self) -> None:
        """Revert all team ratings towards the mean (1500) between seasons."""
        for team in list(self.ratings.keys()):
            self.ratings[team] = (
                (1.0 - self.season_reversion) * self.ratings[team]
                + self.season_reversion * self.initial_rating
            )

    def simulate_season(
        self,
        games_df: pd.DataFrame,
        reset_ratings: bool = False
    ) -> pd.DataFrame:
        """
        Process a sequence of games chronologically, recording pre-game ratings,
        spread predictions, and updating team Elo ratings after each match.
        """
        if reset_ratings:
            self.ratings = {}

        df = games_df.copy()
        
        # Standardize team names
        df["home_team"] = df["home_team"].replace(TEAM_ABBR_MAP)
        df["away_team"] = df["away_team"].replace(TEAM_ABBR_MAP)
        
        # Sort chronologically
        if "gameday" in df.columns:
            df["gameday"] = pd.to_datetime(df["gameday"])
            df = df.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
        else:
            df = df.sort_values(["season", "week"]).reset_index(drop=True)

        pre_home_elos = []
        pre_away_elos = []
        prob_home_wins = []
        proj_spreads = []
        post_home_elos = []
        post_away_elos = []

        current_season = None

        for _, row in df.iterrows():
            season = row["season"]
            
            # Apply inter-season reversion when moving to a new season
            if current_season is not None and season != current_season:
                self.revert_towards_mean()
            current_season = season

            home_team = row["home_team"]
            away_team = row["away_team"]
            home_score = row.get("home_score")
            away_score = row.get("away_score")
            
            # 1. Pre-game ratings
            home_elo = self.get_rating(home_team)
            away_elo = self.get_rating(away_team)
            
            # Check for neutral site games if column exists
            hfa_val = 0.0 if row.get("location", "") == "Neutral" or row.get("neutral_site", False) else self.hfa

            # 2. Predictions before game is played
            prob_home = self.calculate_win_prob(home_elo, away_elo, hfa=hfa_val)
            proj_spread = self.calculate_spread_proj(home_elo, away_elo, hfa=hfa_val)

            pre_home_elos.append(home_elo)
            pre_away_elos.append(away_elo)
            prob_home_wins.append(prob_home)
            proj_spreads.append(proj_spread)

            # 3. Post-game rating update (only if scores are available)
            if pd.notna(home_score) and pd.notna(away_score):
                point_diff = home_score - away_score
                if point_diff > 0:
                    actual_home = 1.0
                    mov_multiplier = self.calculate_mov_multiplier(
                        point_diff, home_elo + hfa_val, away_elo
                    )
                elif point_diff < 0:
                    actual_home = 0.0
                    mov_multiplier = self.calculate_mov_multiplier(
                        point_diff, away_elo, home_elo + hfa_val
                    )
                else:
                    actual_home = 0.5
                    mov_multiplier = 1.0

                elo_shift = self.k_factor * mov_multiplier * (actual_home - prob_home)
                new_home_elo = home_elo + elo_shift
                new_away_elo = away_elo - elo_shift

                self.ratings[home_team] = new_home_elo
                self.ratings[away_team] = new_away_elo
            else:
                new_home_elo = home_elo
                new_away_elo = away_elo

            post_home_elos.append(new_home_elo)
            post_away_elos.append(new_away_elo)

        df["home_elo_pre"] = pre_home_elos
        df["away_elo_pre"] = pre_away_elos
        df["elo_prob_home_win"] = prob_home_wins
        df["elo_proj_spread"] = proj_spreads
        df["home_elo_post"] = post_home_elos
        df["away_elo_post"] = post_away_elos
        df["elo_diff_pre"] = (df["home_elo_pre"] + self.hfa) - df["away_elo_pre"]

        return df
