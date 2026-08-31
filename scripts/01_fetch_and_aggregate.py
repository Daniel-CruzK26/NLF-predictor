"""Script 01: Ingest NFL play-by-play data and generate game-level aggregations."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_SEASONS, PROCESSED_DATA_DIR
from src.data.loader import load_pbp_data, load_schedules
from src.data.aggregator import aggregate_pbp_to_team_games, build_game_matchup_dataset


def main():
    parser = argparse.ArgumentParser(description="Fetch and aggregate NFL play-by-play data.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        help="List of NFL seasons to process (e.g. 2021 2022 2023 2024)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of raw PBP and schedules data",
    )
    args = parser.parse_args()

    print(f"🏈 [1/4] Loading schedules for seasons {args.seasons}...")
    schedules_df = load_schedules(seasons=args.seasons, force_download=args.force_download)
    print(f"   ✓ Loaded {len(schedules_df)} scheduled games.")

    print(f"🏈 [2/4] Loading play-by-play (PBP) data...")
    pbp_df = load_pbp_data(seasons=args.seasons, force_download=args.force_download)
    print(f"   ✓ Loaded {len(pbp_df):,} raw plays.")

    print("🏈 [3/4] Aggregating PBP into team-game statistics...")
    team_games_df = aggregate_pbp_to_team_games(pbp_df)
    team_games_path = PROCESSED_DATA_DIR / "team_games.parquet"
    team_games_df.to_parquet(team_games_path, index=False)
    print(f"   ✓ Saved {len(team_games_df):,} team-game records to {team_games_path}")

    print("🏈 [4/4] Building game matchup dataset with situational context (rest, Vegas lines)...")
    matchups_df = build_game_matchup_dataset(team_games_df, schedules_df)
    matchups_path = PROCESSED_DATA_DIR / "matchups.parquet"
    matchups_df.to_parquet(matchups_path, index=False)
    print(f"   ✓ Saved {len(matchups_df):,} matchups to {matchups_path}")
    print("\n✅ Step 1 complete! Datasets ready in data/processed/")


if __name__ == "__main__":
    main()
