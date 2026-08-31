"""Script 02: Run Elo baseline model, evaluate predictions vs Vegas, and save ratings."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ELO_HFA,
    ELO_INITIAL,
    ELO_K_FACTOR,
    ELO_SEASON_REVERSION,
    ELO_SPREAD_DIVISOR,
    PROCESSED_DATA_DIR,
)
from src.models.elo import EloModel
from src.models.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="Simulate and evaluate Elo baseline model.")
    parser.add_argument("--k-factor", type=float, default=ELO_K_FACTOR, help="Elo K-factor")
    parser.add_argument("--hfa", type=float, default=ELO_HFA, help="Home field advantage (points)")
    parser.add_argument("--reversion", type=float, default=ELO_SEASON_REVERSION, help="Interseason mean reversion")
    parser.add_argument("--divisor", type=float, default=ELO_SPREAD_DIVISOR, help="Elo to spread divisor")
    args = parser.parse_args()

    matchups_path = PROCESSED_DATA_DIR / "matchups.parquet"
    if not matchups_path.exists():
        print(f"❌ Error: {matchups_path} not found. Please run scripts/01_fetch_and_aggregate.py first.")
        sys.exit(1)

    print("📊 [1/3] Loading matchup dataset...")
    matchups_df = pd.read_parquet(matchups_path)

    # Filter regular and postseason completed games
    completed_games = matchups_df[
        matchups_df["home_score"].notna() & matchups_df["away_score"].notna()
    ].copy()
    print(f"   ✓ Loaded {len(completed_games)} completed games across {completed_games['season'].nunique()} seasons.")

    print("📊 [2/3] Simulating Elo ratings across historical seasons...")
    elo_model = EloModel(
        initial_rating=ELO_INITIAL,
        k_factor=args.k_factor,
        hfa=args.hfa,
        season_reversion=args.reversion,
        spread_divisor=args.divisor,
    )
    
    elo_df = elo_model.simulate_season(completed_games, reset_ratings=True)
    elo_out_path = PROCESSED_DATA_DIR / "elo_predictions.parquet"
    elo_df.to_parquet(elo_out_path, index=False)
    print(f"   ✓ Saved Elo projections to {elo_out_path}")

    print("📊 [3/3] Evaluating Elo performance vs actual outcomes and Vegas lines...")
    eval_result = ModelEvaluator.evaluate(
        elo_df,
        pred_spread_col="elo_proj_spread",
        pred_prob_col="elo_prob_home_win",
        actual_diff_col="actual_point_diff",
        vegas_spread_col="spread_line",
    )

    print("\n" + eval_result.summary())

    # Display Top 10 current team ratings
    print("\n🏆 Current Team Elo Ratings:")
    sorted_ratings = sorted(elo_model.ratings.items(), key=lambda x: x[1], reverse=True)
    for rank, (team, rating) in enumerate(sorted_ratings[:10], start=1):
        print(f"  {rank:2d}. {team:4s} -> {rating:.1f}")

    print("\n✅ Step 2 complete! Elo benchmark established.")


if __name__ == "__main__":
    main()
