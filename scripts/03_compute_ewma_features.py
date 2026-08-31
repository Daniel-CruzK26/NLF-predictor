"""Script 03: Compute anti-leakage EWMA features for pass/rush/pressure and build final ML dataset."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_EWMA_ALPHA, PROCESSED_DATA_DIR
from src.features.ewma import EWMAEngine


def main():
    parser = argparse.ArgumentParser(description="Compute EWMA performance metrics and matchup features.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_EWMA_ALPHA, help="EWMA decay alpha (default 0.15)")
    args = parser.parse_args()

    team_games_path = PROCESSED_DATA_DIR / "team_games.parquet"
    matchups_path = PROCESSED_DATA_DIR / "matchups.parquet"
    elo_path = PROCESSED_DATA_DIR / "elo_predictions.parquet"

    if not team_games_path.exists() or not matchups_path.exists():
        print("❌ Error: Processed data missing. Please run script 01 and 02 first.")
        sys.exit(1)

    print(f"⚙️  [1/3] Loading team games and matchup data...")
    team_games = pd.read_parquet(team_games_path)
    
    # If Elo predictions exist, use that as base so we have both Elo + EWMA features
    if elo_path.exists():
        matchups = pd.read_parquet(elo_path)
        print(f"   ✓ Loaded {len(matchups)} matchups (including Elo features).")
    else:
        matchups = pd.read_parquet(matchups_path)
        print(f"   ✓ Loaded {len(matchups)} matchups.")

    # Sort team games chronologically
    if "gameday" in matchups.columns:
        matchup_dates = matchups[["game_id", "gameday", "season", "week"]].drop_duplicates()
        team_games = pd.merge(team_games, matchup_dates, on="game_id", how="left")
    
    print(f"⚙️  [2/3] Computing EWMA metrics (alpha={args.alpha}) with zero-leakage shift(1)...")
    engine = EWMAEngine(alpha=args.alpha)
    team_games_ewma = engine.compute_team_game_ewma(team_games)
    
    print(f"⚙️  [3/3] Generating matchup differentials (Pass EPA, Rush EPA, Pressure, Success Rate)...")
    final_df = engine.enrich_matchups_with_ewma(matchups, team_games_ewma)
    
    output_path = PROCESSED_DATA_DIR / "final_ml_dataset.parquet"
    final_df.to_parquet(output_path, index=False)
    
    feature_cols = [c for c in final_df.columns if "ewma" in c or "diff" in c or "mismatch" in c or "elo" in c]
    print(f"   ✓ Successfully generated {len(feature_cols)} ML features across {len(final_df)} games.")
    print(f"   ✓ Final dataset saved to: {output_path}")

    # Display sample features for latest games
    print("\n📋 Sample of generated EWMA & Differential Features (first 5 games):")
    sample_cols = [
        "season", "week", "home_team", "away_team",
        "home_ewma_off_pass_epa", "away_ewma_def_pass_epa_allowed",
        "net_pass_advantage", "net_rush_advantage", "net_success_rate_advantage"
    ]
    avail_cols = [c for c in sample_cols if c in final_df.columns]
    print(final_df[avail_cols].dropna().tail(5).to_string(index=False))

    print("\n✅ Step 3 complete! EWMA features ready for ML modeling.")


if __name__ == "__main__":
    main()
