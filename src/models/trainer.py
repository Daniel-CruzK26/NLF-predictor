"""Machine Learning Trainer for NFL Spread Prediction with strict Time-Series Splits."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..config import PROCESSED_DATA_DIR
from .evaluator import ModelEvaluator, EvaluationResult

# Default curated feature columns for ML modeling
DEFAULT_FEATURE_COLS = [
    # Elo Baseline
    "elo_proj_spread",
    "elo_diff_pre",
    "elo_prob_home_win",
    
    # Opponent-Adjusted EPA Features
    "adj_net_epa_advantage",
    "adj_diff_pass_advantage",
    "adj_diff_rush_advantage",
    "home_adj_off_pass_epa",
    "away_adj_def_pass_epa_allowed",
    "home_adj_off_rush_epa",
    "away_adj_def_rush_epa_allowed",
    "home_adj_off_total_epa",
    "away_adj_def_total_epa_allowed",
    
    # Advanced QB Features
    "diff_qb_composite_score",
    "diff_qb_epa_per_dropback",
    "diff_qb_cpoe",
    "diff_qb_sack_avoidance",
    "home_ewma_qb_composite_score",
    "away_ewma_qb_composite_score",
    "home_ewma_qb_epa_per_dropback",
    "away_ewma_qb_epa_per_dropback",
    
    # Team EWMA & Matchup Differentials
    "net_pass_advantage",
    "net_rush_advantage",
    "net_success_rate_advantage",
    "home_pass_rush_mismatch",
    "away_pass_rush_mismatch",
    "home_ewma_off_success_rate",
    "away_ewma_def_success_rate_allowed",
    
    # Situational & Weather Factors
    "rest_differential",
    "is_dome",
    "wind_speed",
    "high_wind_flag",
    "temperature",
    "freezing_temp_flag",
    "is_division_game",
    "wind_pass_decay_interaction",
]


class TimeSeriesSpreadTrainer:
    """
    Trains and validates NFL spread prediction models using chronological expanding-window time-series splits.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "actual_point_diff",
        vegas_col: str = "spread_line",
    ):
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.target_col = target_col
        self.vegas_col = vegas_col

    def get_time_series_splits(self, df: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Creates expanding-window time-series splits by season.
        Fold 1: Train Season S0 -> Test Season S1
        Fold 2: Train Seasons S0..S1 -> Test Season S2
        Fold 3: Train Seasons S0..S2 -> Test Season S3
        """
        seasons = sorted(df["season"].unique())
        splits = []
        
        for i in range(1, len(seasons)):
            train_seasons = seasons[:i]
            test_season = seasons[i]
            
            train_idx = df[df["season"].isin(train_seasons)].index
            test_idx = df[df["season"] == test_season].index
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
                
        return splits

    def _get_model_instance(self, model_type: str):
        """Helper to instantiate requested model algorithm."""
        if model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.04,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=3.0,
                random_state=42,
            )
        elif model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=100,
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
        elif model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.04,
                max_depth=3,
                subsample=0.8,
                random_state=42,
            )
        elif model_type == "ridge":
            return Ridge(alpha=10.0)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        model_type: str = "gradient_boosting"
    ) -> Tuple[pd.DataFrame, EvaluationResult, Dict[str, float]]:
        """
        Runs out-of-fold time-series cross-validation.
        Returns:
          - Dataframe with out-of-fold predictions
          - Overall EvaluationResult
          - Feature importances dictionary
        """
        valid_mask = df[self.target_col].notna()
        clean_df = df[valid_mask].copy().reset_index(drop=True)
        
        avail_features = [c for c in self.feature_cols if c in clean_df.columns]
        X = clean_df[avail_features].fillna(0.0)
        y = clean_df[self.target_col].values
        
        splits = self.get_time_series_splits(clean_df)
        if not splits:
            raise ValueError("Insufficient seasons for time-series splits.")

        oof_preds = np.full(len(clean_df), np.nan)
        feature_importance_accum = np.zeros(len(avail_features))

        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_test, y_test = X.iloc[test_idx], y[test_idx]

            test_season = clean_df.iloc[test_idx]["season"].iloc[0]
            print(f"   ⚙️  Fold {fold}: Train on {len(train_idx)} games -> Test on Season {test_season} ({len(test_idx)} games)...")

            if model_type == "ensemble":
                # Blended Ensemble: Gradient Boosting + Ridge + Elo baseline
                m_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=42)
                m_ridge = Ridge(alpha=15.0)

                m_gb.fit(X_train, y_train)
                m_ridge.fit(X_train, y_train)

                preds = (0.55 * m_gb.predict(X_test)) + (0.45 * m_ridge.predict(X_test))
                feature_importance_accum += m_gb.feature_importances_
            else:
                model = self._get_model_instance(model_type)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                if hasattr(model, "feature_importances_"):
                    feature_importance_accum += model.feature_importances_
                elif hasattr(model, "coef_"):
                    feature_importance_accum += np.abs(model.coef_)

            oof_preds[test_idx] = preds

        clean_df[f"{model_type}_proj_spread"] = oof_preds
        clean_df[f"{model_type}_prob_home_win"] = 1.0 / (1.0 + 10.0 ** (-oof_preds / 13.5))

        eval_mask = ~np.isnan(oof_preds)
        eval_df = clean_df[eval_mask].copy()

        eval_result = ModelEvaluator.evaluate(
            eval_df,
            pred_spread_col=f"{model_type}_proj_spread",
            pred_prob_col=f"{model_type}_prob_home_win",
            actual_diff_col=self.target_col,
            vegas_spread_col=self.vegas_col,
        )

        avg_importances = feature_importance_accum / len(splits)
        feat_imp_dict = dict(zip(avail_features, avg_importances))
        feat_imp_sorted = dict(sorted(feat_imp_dict.items(), key=lambda x: x[1], reverse=True))

        return clean_df, eval_result, feat_imp_sorted
