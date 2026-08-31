"""Script 04: Extract QB metrics, compute Opponent-Adjusted EPA, and build advanced feature matrix."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_EWMA_ALPHA, PROCESSED_DATA_DIR, DEFAULT_SEASONS
from src.data.loader import load_pbp_data
from src.data.qb_aggregator import aggregate_qb_game_stats, get_starting_qbs_per_game
from src.features.opponent_adjustment import OpponentAdjuster
from src.features.qb_features import QBFeatureEngine


def main():
    parser = argparse.ArgumentParser(description="Build advanced QB and Opponent-Adjusted EPA features.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_EWMA_ALPHA, help="EWMA decay alpha")
    parser.add_argument("--ridge-alpha", type=float, default=3.0, help="Ridge regularization alpha for Opponent Adjustment")
    args = parser.parse_args()

    matchups_path = PROCESSED_DATA_DIR / "final_ml_dataset.parquet"
    if not matchups_path.exists():
        matchups_path = PROCESSED_DATA_DIR / "matchups.parquet"
    
    if not matchups_path.exists():
        print("❌ Error: Base matchups dataset not found. Please run scripts 01-03 first.")
        sys.exit(1)

    print("🧠 [1/4] Loading base matchup dataset and raw PBP data...")
    matchups_df = pd.read_parquet(matchups_path)
    seasons = sorted(matchups_df["season"].unique().tolist())
    pbp_df = load_pbp_data(seasons=seasons)
    print(f"   ✓ Loaded {len(matchups_df)} games and {len(pbp_df):,} plays for seasons {seasons}.")

    print("🧠 [2/4] Aggregating QB performance metrics and determining starting QBs...")
    qb_game_stats = aggregate_qb_game_stats(pbp_df)
    starters_df = get_starting_qbs_per_game(qb_game_stats)
    
    # Merge gameday / season / week to starters
    sched_info = matchups_df[["game_id", "season", "week", "gameday"]].drop_duplicates()
    starters_df = pd.merge(starters_df, sched_info, on="game_id", how="left")
    
    qb_stats_path = PROCESSED_DATA_DIR / "qb_game_stats.parquet"
    qb_game_stats.to_parquet(qb_stats_path, index=False)
    print(f"   ✓ Processed {len(qb_game_stats):,} QB-game appearances and saved to {qb_stats_path}")

    print("🧠 [3/4] Computing QB rolling EWMA & backup shrinkage features...")
    qb_engine = QBFeatureEngine(alpha=args.alpha)
    starters_ewma = qb_engine.compute_qb_ewma(starters_df)
    matchups_with_qb = qb_engine.enrich_matchups_with_qb(matchups_df, starters_ewma)
    print("   ✓ QB features and matchup differentials generated.")

    print(f"🧠 [4/4] Computing Opponent-Adjusted EPA via regularized Ridge (alpha={args.ridge_alpha})...")
    adj_engine = OpponentAdjuster(l2_alpha=args.ridge_alpha)
    final_advanced_df = adj_engine.compute_opponent_adjusted_features(matchups_with_qb)

    output_path = PROCESSED_DATA_DIR / "final_ml_dataset.parquet"
    final_advanced_df.to_parquet(output_path, index=False)
    print(f"   ✓ Advanced dataset successfully saved to {output_path}")

    # Feature Importance / Correlation Analysis with Spread Result
    print("\n📈 Correlation with Actual Point Differential (Home Score - Away Score):")
    feature_candidates = [
        "elo_proj_spread",
        "spread_line",
        "diff_qb_composite_score",
        "diff_qb_epa_per_dropback",
        "diff_qb_cpoe",
        "diff_qb_sack_avoidance",
        "adj_net_epa_advantage",
        "adj_diff_pass_advantage",
        "net_pass_advantage",
        "net_rush_advantage",
        "rest_differential",
    ]
    avail_features = [c for c in feature_candidates if c in final_advanced_df.columns]
    corrs = final_advanced_df[avail_features + ["actual_point_diff"]].corr()["actual_point_diff"].drop("actual_point_diff")
    corrs_sorted = corrs.sort_values(ascending=False)

    for rank, (feat, corr_val) in enumerate(corrs_sorted.items(), start=1):
        print(f"  {rank:2d}. {feat:30s} -> Pearson r = {corr_val:+.4f}")

    print("\n✅ Step 4 complete! Advanced QB and Opponent-Adjusted features ready.")


if __name__ == "__main__":
    main()
