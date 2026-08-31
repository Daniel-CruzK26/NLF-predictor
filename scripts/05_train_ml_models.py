"""Script 05: Train and compare ML models (XGBoost, GBDT, LightGBM, Ridge, Ensemble) with Time-Series Split."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR
from src.features.situational import add_situational_features
from src.models.trainer import TimeSeriesSpreadTrainer, DEFAULT_FEATURE_COLS


def main():
    parser = argparse.ArgumentParser(description="Train ML models for NFL spread prediction.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "gradient_boosting", "xgboost", "lightgbm", "ensemble"],
        help="List of model architectures to train and evaluate",
    )
    args = parser.parse_args()

    data_path = PROCESSED_DATA_DIR / "final_ml_dataset.parquet"
    if not data_path.exists():
        print("❌ Error: final_ml_dataset.parquet not found. Please run scripts 01-04 first.")
        sys.exit(1)

    print("🤖 [1/3] Loading dataset and engineering weather/situational features...")
    df = pd.read_parquet(data_path)
    df = add_situational_features(df)
    print(f"   ✓ Loaded {len(df)} games across seasons {sorted(df['season'].unique().tolist())}.")

    print("🤖 [2/3] Executing Expanding-Window Time-Series Cross-Validation...")
    trainer = TimeSeriesSpreadTrainer(feature_cols=DEFAULT_FEATURE_COLS)
    
    results = {}
    feature_importances = {}
    pred_dfs = []

    for model_name in args.models:
        print(f"\n--- 🚀 Training {model_name.upper()} ---")
        try:
            pred_df, eval_res, feat_imp = trainer.train_and_evaluate(df, model_type=model_name)
            results[model_name] = eval_res
            feature_importances[model_name] = feat_imp
            pred_dfs.append(pred_df[[f"{model_name}_proj_spread", f"{model_name}_prob_home_win"]])
        except Exception as e:
            print(f"   ⚠️ Could not train {model_name}: {e}")

    # Combine all predictions
    final_preds_df = pd.concat([df] + pred_dfs, axis=1)
    # Remove duplicate columns if any
    final_preds_df = final_preds_df.loc[:, ~final_preds_df.columns.duplicated()]
    out_path = PROCESSED_DATA_DIR / "model_predictions_comparison.parquet"
    final_preds_df.to_parquet(out_path, index=False)

    print("\n" + "=" * 80)
    print("🏆 MODEL BENCHMARK COMPARISON (Out-of-Fold Time-Series Evaluation)")
    print("=" * 80)
    
    comparison_rows = []
    
    # Add Vegas benchmark row from the test set evaluation
    sample_eval = list(results.values())[0]
    comparison_rows.append({
        "Model": "Las Vegas Closing Line",
        "Spread MAE (pts)": f"{sample_eval.vegas_mae:.3f}",
        "Spread RMSE": f"{sample_eval.vegas_rmse:.3f}",
        "SU Win %": "—",
        "Brier Score": "—",
        "ATS Pick %": "50.00% (Market)",
        "Delta MAE vs Vegas": "0.000",
    })

    for m_name, res in results.items():
        delta_v = res.mae_spread - sample_eval.vegas_mae
        comparison_rows.append({
            "Model": m_name.upper(),
            "Spread MAE (pts)": f"{res.mae_spread:.3f}",
            "Spread RMSE": f"{res.rmse_spread:.3f}",
            "SU Win %": f"{res.su_accuracy * 100:.2f}%",
            "Brier Score": f"{res.brier_score:.4f}",
            "ATS Pick %": f"{res.ats_accuracy * 100:.2f}%",
            "Delta MAE vs Vegas": f"{delta_v:+.3f}",
        })

    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False))

    # Display Top Feature Importances for the best tree-based model
    best_tree = "xgboost" if "xgboost" in feature_importances else "gradient_boosting"
    if best_tree in feature_importances:
        print(f"\n📊 Top 10 Most Important Features in {best_tree.upper()}:")
        for rank, (feat, imp) in enumerate(list(feature_importances[best_tree].items())[:10], start=1):
            print(f"  {rank:2d}. {feat:35s} -> Importance = {imp:.4f}")

    print(f"\n✅ All model evaluations complete! Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
