"""Evaluation metrics for NFL spread and outcome predictions."""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error


@dataclass
class EvaluationResult:
    n_games: int
    mae_spread: float
    rmse_spread: float
    brier_score: float
    log_loss_score: float
    su_accuracy: float  # Straight up winner accuracy
    vegas_mae: Optional[float] = None
    vegas_rmse: Optional[float] = None
    vegas_brier: Optional[float] = None
    ats_accuracy: Optional[float] = None  # Against the spread accuracy

    def to_dict(self) -> Dict[str, float]:
        return {
            "n_games": self.n_games,
            "mae_spread": self.mae_spread,
            "rmse_spread": self.rmse_spread,
            "brier_score": self.brier_score,
            "log_loss_score": self.log_loss_score,
            "su_accuracy": self.su_accuracy,
            "vegas_mae": self.vegas_mae,
            "vegas_rmse": self.vegas_rmse,
            "ats_accuracy": self.ats_accuracy,
        }

    def summary(self) -> str:
        lines = [
            f"=== Evaluation Summary ({self.n_games} games) ===",
            f"Spread MAE (Model):  {self.mae_spread:.3f} pts",
            f"Spread RMSE (Model): {self.rmse_spread:.3f} pts",
            f"SU Win Accuracy:     {self.su_accuracy * 100:.2f}%",
            f"Brier Score:         {self.brier_score:.4f}",
            f"Log Loss:            {self.log_loss_score:.4f}",
        ]
        if self.vegas_mae is not None:
            lines.append(f"Vegas Spread MAE:    {self.vegas_mae:.3f} pts")
            lines.append(f"Vegas Spread RMSE:   {self.vegas_rmse:.3f} pts")
            diff_mae = self.mae_spread - self.vegas_mae
            lines.append(f"Delta MAE vs Vegas:  {diff_mae:+.3f} pts")
        if self.ats_accuracy is not None:
            lines.append(f"ATS Pick Accuracy:   {self.ats_accuracy * 100:.2f}%")
        return "\n".join(lines)


class ModelEvaluator:
    """Evaluates prediction models against actual outcomes and Vegas closing lines."""

    @staticmethod
    def evaluate(
        df: pd.DataFrame,
        pred_spread_col: str = "elo_proj_spread",
        pred_prob_col: str = "elo_prob_home_win",
        actual_diff_col: str = "actual_point_diff",
        vegas_spread_col: str = "spread_line",
    ) -> EvaluationResult:
        """
        Evaluate spread and win probability predictions.
        Note on spread_line: In standard nflverse datasets, `spread_line` is positive when home team is favored
        (e.g., spread_line = 3.5 means Home is favored by 3.5 points).
        `actual_point_diff` = home_score - away_score.
        """
        valid_mask = (
            df[actual_diff_col].notna()
            & df[pred_spread_col].notna()
            & df[pred_prob_col].notna()
        )
        sub = df[valid_mask].copy()
        
        if len(sub) == 0:
            raise ValueError("No valid rows for evaluation.")

        y_true_diff = sub[actual_diff_col].values
        y_pred_diff = sub[pred_spread_col].values
        
        # Binary target: Home won (ties counted as 0.5)
        y_true_win = (y_true_diff > 0).astype(float)
        y_true_win[y_true_diff == 0] = 0.5
        
        y_pred_prob = sub[pred_prob_col].clip(1e-5, 1 - 1e-5).values

        # Spread metrics
        mae_spread = float(mean_absolute_error(y_true_diff, y_pred_diff))
        rmse_spread = float(np.sqrt(mean_squared_error(y_true_diff, y_pred_diff)))

        # Straight-up accuracy (excluding ties for strict binary classification)
        non_ties = y_true_diff != 0
        su_pred = (y_pred_diff[non_ties] > 0).astype(int)
        su_true = (y_true_diff[non_ties] > 0).astype(int)
        su_acc = float(np.mean(su_pred == su_true)) if len(su_pred) > 0 else 0.0

        # Probabilistic metrics
        # For Log Loss and Brier, evaluate on non-ties
        if len(su_true) > 0:
            brier = float(brier_score_loss(su_true, y_pred_prob[non_ties], pos_label=1))
            try:
                lloss = float(log_loss(su_true, y_pred_prob[non_ties], labels=[0, 1]))
            except Exception:
                lloss = 0.6931
        else:
            brier = 0.25
            lloss = 0.6931

        # Vegas benchmark metrics if available
        vegas_mae = None
        vegas_rmse = None
        ats_acc = None
        
        if vegas_spread_col in sub.columns and sub[vegas_spread_col].notna().sum() > 0:
            vegas_mask = sub[vegas_spread_col].notna()
            v_sub = sub[vegas_mask]
            
            y_v_true = v_sub[actual_diff_col].values
            y_v_spread = v_sub[vegas_spread_col].values
            y_m_spread = v_sub[pred_spread_col].values
            
            vegas_mae = float(mean_absolute_error(y_v_true, y_v_spread))
            vegas_rmse = float(np.sqrt(mean_squared_error(y_v_true, y_v_spread)))
            
            # ATS Accuracy: Bet on Home if Model Spread > Vegas Spread, else Away
            # Home covers if actual_diff > spread_line
            # Push if actual_diff == spread_line
            ats_bets = y_m_spread > y_v_spread  # True = Bet Home, False = Bet Away
            home_covered = y_v_true > y_v_spread
            away_covered = y_v_true < y_v_spread
            
            # Count wins and losses excluding pushes
            non_pushes = y_v_true != y_v_spread
            if np.sum(non_pushes) > 0:
                won_bet = (ats_bets[non_pushes] & home_covered[non_pushes]) | (
                    (~ats_bets[non_pushes]) & away_covered[non_pushes]
                )
                ats_acc = float(np.mean(won_bet))

        return EvaluationResult(
            n_games=len(sub),
            mae_spread=mae_spread,
            rmse_spread=rmse_spread,
            brier_score=brier,
            log_loss_score=lloss,
            su_accuracy=su_acc,
            vegas_mae=vegas_mae,
            vegas_rmse=vegas_rmse,
            ats_accuracy=ats_acc,
        )
