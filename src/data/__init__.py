"""Data ingestion and aggregation modules."""

from .loader import load_pbp_data, load_schedules
from .aggregator import aggregate_pbp_to_team_games, build_game_matchup_dataset
from .qb_aggregator import extract_qb_dropbacks, aggregate_qb_game_stats, get_starting_qbs_per_game

__all__ = [
    "load_pbp_data",
    "load_schedules",
    "aggregate_pbp_to_team_games",
    "build_game_matchup_dataset",
    "extract_qb_dropbacks",
    "aggregate_qb_game_stats",
    "get_starting_qbs_per_game",
]
