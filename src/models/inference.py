"""Inference and Betting Edge Engine for upcoming NFL game slates."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

from ..config import (
    DEFAULT_EWMA_ALPHA,
    ELO_HFA,
    ELO_INITIAL,
    ELO_K_FACTOR,
    ELO_SEASON_REVERSION,
    ELO_SPREAD_DIVISOR,
    PROCESSED_DATA_DIR,
    TEAM_ABBR_MAP,
)
from ..config_rosters import get_active_starters
from ..data.loader import standardize_team_abbr
from ..features.situational import add_situational_features
from .trainer import DEFAULT_FEATURE_COLS


@dataclass
class GamePrediction:
    game_id: str
    gameday: str
    home_team: str
    away_team: str
    home_qb: str
    away_qb: str
    vegas_spread: float  # Positive = Home Favored, Negative = Away Favored
    vegas_total: float
    model_spread: float  # Projected Home Point Margin
    home_win_prob: float
    away_win_prob: float
    edge: float          # model_spread - vegas_spread
    recommendation: str  # e.g., "BET HOME -4.5", "BET AWAY +7.0", "PASS"
    confidence: str      # "HIGH (3★)", "MEDIUM (2★)", "LOW (1★)", "NO VALUE"
    kelly_stake_pct: float  # Fractional Kelly stake recommendation
    key_drivers: List[str]  # Analytical rationales (e.g., "QB EPA Advantage +0.18", "Elite Pass Matchup")

    def to_dict(self) -> Dict:
        return {
            "game_id": self.game_id,
            "gameday": self.gameday,
            "matchup": f"{self.away_team} @ {self.home_team}",
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_qb": self.home_qb,
            "away_qb": self.away_qb,
            "vegas_spread": self.vegas_spread,
            "model_spread": round(self.model_spread, 2),
            "home_win_prob": round(self.home_win_prob * 100, 1),
            "away_win_prob": round(self.away_win_prob * 100, 1),
            "edge": round(self.edge, 2),
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "kelly_stake_pct": round(self.kelly_stake_pct, 2),
            "key_drivers": " | ".join(self.key_drivers),
        }


class LiveInferenceEngine:
    """
    Generates live spread projections, win probabilities, and betting edges for any NFL week.
    """

    def __init__(
        self,
        historical_dataset_path: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        self.dataset_path = historical_dataset_path or (PROCESSED_DATA_DIR / "final_ml_dataset.parquet")
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.trained_models = {}
        self.team_priors = {}
        self.qb_priors = {}
        self.elo_ratings = {}

    def fit_production_models(self) -> None:
        """
        Fits final production models (LightGBM, XGBoost, Ridge, and Ensemble)
        on the entire available historical completed dataset.
        """
        df = pd.read_parquet(self.dataset_path)
        df = add_situational_features(df)
        
        # Filter completed games
        valid_mask = df["actual_point_diff"].notna()
        clean_df = df[valid_mask].copy()

        avail_features = [c for c in self.feature_cols if c in clean_df.columns]
        X = clean_df[avail_features].fillna(0.0)
        y = clean_df["actual_point_diff"].values

        # 1. Train LightGBM
        m_lgb = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=3.0,
            random_state=42,
            verbose=-1,
        )
        m_lgb.fit(X, y)
        self.trained_models["lightgbm"] = m_lgb

        # 2. Train XGBoost
        m_xgb = xgb.XGBRegressor(
            n_estimators=120,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=3.0,
            random_state=42,
        )
        m_xgb.fit(X, y)
        self.trained_models["xgboost"] = m_xgb

        # 3. Train Ridge
        m_ridge = Ridge(alpha=12.0)
        m_ridge.fit(X, y)
        self.trained_models["ridge"] = m_ridge

        # Extract latest team Elo ratings
        if "home_elo_post" in clean_df.columns:
            latest_elos = {}
            for _, row in clean_df.sort_values(["season", "week"]).iterrows():
                latest_elos[row["home_team"]] = row["home_elo_post"]
                latest_elos[row["away_team"]] = row["away_elo_post"]
            
            # Apply inter-season reversion
            for t, val in latest_elos.items():
                self.elo_ratings[t] = (1.0 - ELO_SEASON_REVERSION) * val + ELO_SEASON_REVERSION * ELO_INITIAL

        # Extract latest team EWMA and Opponent-Adjusted EPA features
        for t in clean_df["home_team"].unique():
            home_t_games = clean_df[clean_df["home_team"] == t].sort_values(["season", "week"])
            away_t_games = clean_df[clean_df["away_team"] == t].sort_values(["season", "week"])
            
            last_home = home_t_games.iloc[-1] if len(home_t_games) > 0 else None
            last_away = away_t_games.iloc[-1] if len(away_t_games) > 0 else None

            # Average latest performance signals
            off_pass = float(last_home.get("home_adj_off_pass_epa", 0.0) if last_home is not None else 0.0)
            def_pass = float(last_home.get("home_adj_def_pass_epa_allowed", 0.0) if last_home is not None else 0.0)
            off_rush = float(last_home.get("home_adj_off_rush_epa", 0.0) if last_home is not None else 0.0)
            def_rush = float(last_home.get("home_adj_def_rush_epa_allowed", 0.0) if last_home is not None else 0.0)
            off_total = float(last_home.get("home_adj_off_total_epa", 0.0) if last_home is not None else 0.0)
            def_total = float(last_home.get("home_adj_def_total_epa_allowed", 0.0) if last_home is not None else 0.0)

            self.team_priors[t] = {
                "adj_off_pass_epa": off_pass,
                "adj_def_pass_epa_allowed": def_pass,
                "adj_off_rush_epa": off_rush,
                "adj_def_rush_epa_allowed": def_rush,
                "adj_off_total_epa": off_total,
                "adj_def_total_epa_allowed": def_total,
                "ewma_off_success_rate": float(last_home.get("home_ewma_off_success_rate", 0.45) if last_home is not None else 0.45),
                "ewma_def_success_rate_allowed": float(last_home.get("home_ewma_def_success_rate_allowed", 0.45) if last_home is not None else 0.45),
            }

        # Extract latest QB metrics
        if "home_qb_name" in clean_df.columns:
            for _, row in clean_df.iterrows():
                if pd.notna(row.get("home_qb_name")):
                    self.qb_priors[row["home_qb_name"]] = {
                        "composite": float(row.get("home_ewma_qb_composite_score", -0.05)),
                        "epa": float(row.get("home_ewma_qb_epa_per_dropback", -0.05)),
                        "cpoe": float(row.get("home_ewma_qb_cpoe", -1.5)),
                        "sack_rate": float(row.get("home_ewma_qb_sack_rate", 0.06)),
                    }
                if pd.notna(row.get("away_qb_name")):
                    self.qb_priors[row["away_qb_name"]] = {
                        "composite": float(row.get("away_ewma_qb_composite_score", -0.05)),
                        "epa": float(row.get("away_ewma_qb_epa_per_dropback", -0.05)),
                        "cpoe": float(row.get("away_ewma_qb_cpoe", -1.5)),
                        "sack_rate": float(row.get("away_ewma_qb_sack_rate", 0.06)),
                    }

        # Ensure active starters (like Geno Smith, Sam Darnold, Stafford) have their correct career metrics
        h_qb_col = clean_df["home_qb_name"] if "home_qb_name" in clean_df.columns else pd.Series("", index=clean_df.index)
        a_qb_col = clean_df["away_qb_name"] if "away_qb_name" in clean_df.columns else pd.Series("", index=clean_df.index)

        starters = get_active_starters()
        for t, qb_name in starters.items():
            if qb_name not in self.qb_priors:
                matched_rows = clean_df[(h_qb_col == qb_name) | (a_qb_col == qb_name)]
                if len(matched_rows) > 0:
                    last_r = matched_rows.iloc[-1]
                    is_h = last_r.get("home_qb_name") == qb_name
                    pfx = "home_" if is_h else "away_"
                    self.qb_priors[qb_name] = {
                        "composite": float(last_r.get(f"{pfx}ewma_qb_composite_score", 0.10)),
                        "epa": float(last_r.get(f"{pfx}ewma_qb_epa_per_dropback", 0.12)),
                        "cpoe": float(last_r.get(f"{pfx}ewma_qb_cpoe", 2.0)),
                        "sack_rate": float(last_r.get(f"{pfx}ewma_qb_sack_rate", 0.05)),
                    }

    def _build_upcoming_matchup_features(self, upcoming_df: pd.DataFrame) -> pd.DataFrame:
        """Constructs full feature vector for upcoming matches from team and QB priors."""
        df = upcoming_df.copy()
        df = standardize_team_abbr(df, ["home_team", "away_team"])
        df = add_situational_features(df)

        rows = []
        for _, row in df.iterrows():
            h_team = row["home_team"]
            a_team = row["away_team"]
            h_qb = row.get("home_qb_name", f"{h_team}_QB")
            a_qb = row.get("away_qb_name", f"{a_team}_QB")

            h_elo = self.elo_ratings.get(h_team, ELO_INITIAL)
            a_elo = self.elo_ratings.get(a_team, ELO_INITIAL)
            hfa_val = 0.0 if row.get("is_dome", 0) == 1 and row.get("location") == "Neutral" else ELO_HFA
            elo_proj_spread = (h_elo + hfa_val - a_elo) / ELO_SPREAD_DIVISOR
            elo_prob_home = 1.0 / (1.0 + 10.0 ** (-((h_elo + hfa_val) - a_elo) / 400.0))

            h_priors = self.team_priors.get(h_team, {})
            a_priors = self.team_priors.get(a_team, {})

            h_adj_off_pass = h_priors.get("adj_off_pass_epa", 0.0)
            a_adj_def_pass = a_priors.get("adj_def_pass_epa_allowed", 0.0)
            a_adj_off_pass = a_priors.get("adj_off_pass_epa", 0.0)
            h_adj_def_pass = h_priors.get("adj_def_pass_epa_allowed", 0.0)

            h_adj_off_rush = h_priors.get("adj_off_rush_epa", 0.0)
            a_adj_def_rush = a_priors.get("adj_def_rush_epa_allowed", 0.0)
            a_adj_off_rush = a_priors.get("adj_off_rush_epa", 0.0)
            h_adj_def_rush = h_priors.get("adj_def_rush_epa_allowed", 0.0)

            h_adj_off_tot = h_priors.get("adj_off_total_epa", 0.0)
            a_adj_def_tot = a_priors.get("adj_def_total_epa_allowed", 0.0)
            a_adj_off_tot = a_priors.get("adj_off_total_epa", 0.0)
            h_adj_def_tot = h_priors.get("adj_def_total_epa_allowed", 0.0)

            # Load Team Modifiers (e.g. Rams boost, Denver rise, coaching/roster upgrades)
            mod_file = PROCESSED_DATA_DIR.parent / "team_modifiers.json"
            team_mods = {}
            if mod_file.exists():
                try:
                    import json
                    with open(mod_file, "r") as f:
                        team_mods = json.load(f)
                except Exception:
                    pass

            h_mod = team_mods.get(h_team, 0.0)
            a_mod = team_mods.get(a_team, 0.0)

            # QB features
            h_qb_stat = self.qb_priors.get(h_qb, {"composite": 0.0, "epa": 0.0, "cpoe": 0.0, "sack_rate": 0.06})
            a_qb_stat = self.qb_priors.get(a_qb, {"composite": 0.0, "epa": 0.0, "cpoe": 0.0, "sack_rate": 0.06})

            feat = {
                "game_id": row["game_id"],
                "elo_proj_spread": elo_proj_spread + (h_mod - a_mod),
                "elo_diff_pre": (h_elo + hfa_val) - a_elo + ((h_mod - a_mod) * ELO_SPREAD_DIVISOR),
                "elo_prob_home_win": elo_prob_home,
                "adj_net_epa_advantage": (h_adj_off_tot - a_adj_def_tot) - (a_adj_off_tot - h_adj_def_tot) + (h_mod - a_mod) * 0.8,
                "adj_diff_pass_advantage": (h_adj_off_pass - a_adj_def_pass) - (a_adj_off_pass - h_adj_def_pass) + (h_mod - a_mod) * 0.4,
                "adj_diff_rush_advantage": (h_adj_off_rush - a_adj_def_rush) - (a_adj_off_rush - h_adj_def_rush) + (h_mod - a_mod) * 0.2,
                "home_adj_off_pass_epa": h_adj_off_pass + (h_mod * 0.03),
                "away_adj_def_pass_epa_allowed": a_adj_def_pass - (a_mod * 0.03),
                "home_adj_off_rush_epa": h_adj_off_rush + (h_mod * 0.02),
                "away_adj_def_rush_epa_allowed": a_adj_def_rush - (a_mod * 0.02),
                "home_adj_off_total_epa": h_adj_off_tot + (h_mod * 0.5),
                "away_adj_def_total_epa_allowed": a_adj_def_tot - (a_mod * 0.5),
                "diff_qb_composite_score": h_qb_stat["composite"] - a_qb_stat["composite"],
                "diff_qb_epa_per_dropback": h_qb_stat["epa"] - a_qb_stat["epa"],
                "diff_qb_cpoe": h_qb_stat["cpoe"] - a_qb_stat["cpoe"],
                "diff_qb_sack_avoidance": a_qb_stat["sack_rate"] - h_qb_stat["sack_rate"],
                "home_ewma_qb_composite_score": h_qb_stat["composite"],
                "away_ewma_qb_composite_score": a_qb_stat["composite"],
                "home_ewma_qb_epa_per_dropback": h_qb_stat["epa"],
                "away_ewma_qb_epa_per_dropback": a_qb_stat["epa"],
                "net_pass_advantage": (h_adj_off_pass - a_adj_def_pass) - (a_adj_off_pass - h_adj_def_pass),
                "net_rush_advantage": (h_adj_off_rush - a_adj_def_rush) - (a_adj_off_rush - h_adj_def_rush),
                "net_success_rate_advantage": (h_priors.get("ewma_off_success_rate", 0.45) - a_priors.get("ewma_def_success_rate_allowed", 0.45))
                - (a_priors.get("ewma_off_success_rate", 0.45) - h_priors.get("ewma_def_success_rate_allowed", 0.45)),
                "home_pass_rush_mismatch": 0.0,
                "away_pass_rush_mismatch": 0.0,
                "home_ewma_off_success_rate": h_priors.get("ewma_off_success_rate", 0.45),
                "away_ewma_def_success_rate_allowed": a_priors.get("ewma_def_success_rate_allowed", 0.45),
                "rest_differential": float(row.get("rest_differential", 0)),
                "is_dome": int(row.get("is_dome", 0)),
                "wind_speed": float(row.get("wind_speed", 7.0)),
                "high_wind_flag": int(row.get("high_wind_flag", 0)),
                "temperature": float(row.get("temperature", 70.0)),
                "freezing_temp_flag": int(row.get("freezing_temp_flag", 0)),
                "is_division_game": int(row.get("is_division_game", 0)),
                "wind_pass_decay_interaction": float(row.get("wind_pass_decay_interaction", 0.0)),
            }
            rows.append(feat)

        return pd.DataFrame(rows)

    def predict_slate(
        self,
        upcoming_df: pd.DataFrame,
        model_weights: Optional[Dict[str, float]] = None,
        edge_threshold_high: float = 3.0,
        edge_threshold_med: float = 2.0,
        edge_threshold_low: float = 1.5,
    ) -> List[GamePrediction]:
        """
        Generates comprehensive predictions and betting recommendations for the slate.
        """
        if not self.trained_models:
            self.fit_production_models()

        if model_weights is None:
            # Optimal blend: 50% LightGBM + 30% XGBoost + 20% Ridge
            model_weights = {"lightgbm": 0.50, "xgboost": 0.30, "ridge": 0.20}

        features_df = self._build_upcoming_matchup_features(upcoming_df)
        avail_features = [c for c in self.feature_cols if c in features_df.columns]
        X = features_df[avail_features].fillna(0.0)

        # Generate blended spread prediction
        blended_preds = np.zeros(len(X))
        for m_name, w in model_weights.items():
            if m_name in self.trained_models:
                blended_preds += w * self.trained_models[m_name].predict(X)

        predictions = []
        for idx, row in upcoming_df.reset_index(drop=True).iterrows():
            m_spread = float(blended_preds[idx])
            v_spread = float(row.get("spread_line", 0.0))
            v_total = float(row.get("total_line", 44.5)) if pd.notna(row.get("total_line")) else 44.5
            
            # Logistic win probability
            p_home = 1.0 / (1.0 + 10.0 ** (-m_spread / 13.5))
            p_away = 1.0 - p_home

            edge = m_spread - v_spread
            abs_edge = abs(edge)

            # Recommendation and Confidence Tier
            if edge >= edge_threshold_low:
                line_str = f"-{v_spread:.1f}" if v_spread > 0 else f"+{abs(v_spread):.1f}"
                rec = f"BET {row['home_team']} ({line_str})"
                # Win prob against the spread (estimated from edge)
                cover_prob = 0.50 + min(0.35, abs_edge * 0.035)
            elif edge <= -edge_threshold_low:
                line_str = f"+{v_spread:.1f}" if v_spread > 0 else f"-{abs(v_spread):.1f}"
                rec = f"BET {row['away_team']} ({line_str})"
                cover_prob = 0.50 + min(0.35, abs_edge * 0.035)
            else:
                rec = "PASS (No Edge)"
                cover_prob = 0.50

            # Confidence star rating
            if abs_edge >= edge_threshold_high:
                conf = "HIGH (3★)"
            elif abs_edge >= edge_threshold_med:
                conf = "MEDIUM (2★)"
            elif abs_edge >= edge_threshold_low:
                conf = "LOW (1★)"
            else:
                conf = "NO VALUE"

            # Fractional Kelly Criterion (1/4 Kelly for bankroll preservation)
            # b = odds payout = 100 / 110 = 0.909 for standard -110 spread
            b = 0.909
            kelly_full = (cover_prob * (b + 1.0) - 1.0) / b
            kelly_quarter = max(0.0, (kelly_full / 4.0) * 100.0) if abs_edge >= edge_threshold_low else 0.0

            # Key drivers explanation
            drivers = []
            h_priors = self.team_priors.get(row["home_team"], {})
            a_priors = self.team_priors.get(row["away_team"], {})
            
            net_epa = (h_priors.get("adj_off_total_epa", 0) - a_priors.get("adj_def_total_epa_allowed", 0)) - (a_priors.get("adj_off_total_epa", 0) - h_priors.get("adj_def_total_epa_allowed", 0))
            if abs(net_epa) > 2.0:
                drivers.append(f"Opponent-Adjusted EPA Advantage ({row['home_team'] if net_epa > 0 else row['away_team']})")

            h_qb = row.get("home_qb_name", f"{row['home_team']}_QB")
            a_qb = row.get("away_qb_name", f"{row['away_team']}_QB")
            diff_qb = self.qb_priors.get(h_qb, {}).get("composite", 0) - self.qb_priors.get(a_qb, {}).get("composite", 0)
            if abs(diff_qb) >= 0.10:
                drivers.append(f"QB Matchup Advantage ({h_qb if diff_qb > 0 else a_qb})")

            if abs(edge) >= 2.0:
                drivers.append(f"Market Discrepancy ({abs_edge:+.1f} pts Edge)")

            if not drivers:
                drivers.append("Balanced Matchup")

            pred = GamePrediction(
                game_id=row["game_id"],
                gameday=str(row.get("gameday", "")),
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_qb=h_qb,
                away_qb=a_qb,
                vegas_spread=v_spread,
                vegas_total=v_total,
                model_spread=m_spread,
                home_win_prob=p_home,
                away_win_prob=p_away,
                edge=edge,
                recommendation=rec,
                confidence=conf,
                kelly_stake_pct=kelly_quarter,
                key_drivers=drivers,
            )
            predictions.append(pred)

        return predictions
