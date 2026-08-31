"""Aggregates play-by-play (PBP) data to team-game level with EPA and situational metrics."""

from typing import Optional
import numpy as np
import pandas as pd
from .loader import standardize_team_abbr


def filter_valid_plays(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Filter PBP to valid offensive plays (pass/rush), excluding kneels/spikes/garbage null EPA."""
    df = pbp_df.copy()
    
    # Standardize columns
    df = standardize_team_abbr(df, ["home_team", "away_team", "posteam", "defteam"])
    
    # Handle optional columns safely
    kneel_mask = (df["qb_kneel"] == 0) if "qb_kneel" in df.columns else True
    spike_mask = (df["qb_spike"] == 0) if "qb_spike" in df.columns else True
    
    # Filter conditions
    valid_play_types = ["pass", "run"]
    condition = (
        df["play_type"].isin(valid_play_types)
        & df["epa"].notna()
        & (df["two_point_attempt"] == 0)
        & kneel_mask
        & spike_mask
        & df["posteam"].notna()
        & df["defteam"].notna()
    )
    
    filtered = df[condition].copy()
    
    # Create indicator helper columns
    filtered["is_pass"] = (filtered["play_type"] == "pass").astype(int)
    filtered["is_rush"] = (filtered["play_type"] == "run").astype(int)
    filtered["is_success"] = (filtered["epa"] > 0).astype(int)
    
    interception_col = (
        filtered["interception"].fillna(0)
        if "interception" in filtered.columns
        else pd.Series(0, index=filtered.index)
    )
    fumble_col = (
        filtered["fumble_lost"].fillna(0)
        if "fumble_lost" in filtered.columns
        else pd.Series(0, index=filtered.index)
    )
    filtered["is_turnover"] = (interception_col + fumble_col).clip(upper=1)
    
    return filtered


def aggregate_pbp_to_team_games(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates filtered PBP data to a single row per (game_id, team),
    computing offensive and defensive EPA, success rates, and pressure metrics.
    """
    valid_pbp = filter_valid_plays(pbp_df)
    
    # 1. Offensive Aggregations (grouped by game_id and posteam)
    off_records = []
    for (game_id, team), group in valid_pbp.groupby(["game_id", "posteam"]):
        n_plays = len(group)
        pass_group = group[group["is_pass"] == 1]
        rush_group = group[group["is_rush"] == 1]
        
        n_pass = len(pass_group)
        n_rush = len(rush_group)
        
        off_pass_epa = pass_group["epa"].mean() if n_pass > 0 else 0.0
        off_rush_epa = rush_group["epa"].mean() if n_rush > 0 else 0.0
        off_total_epa = group["epa"].sum()
        off_epa_per_play = group["epa"].mean() if n_plays > 0 else 0.0
        
        off_success_rate = group["is_success"].mean() if n_plays > 0 else 0.0
        off_pass_success_rate = pass_group["is_success"].mean() if n_pass > 0 else 0.0
        off_rush_success_rate = rush_group["is_success"].mean() if n_rush > 0 else 0.0
        
        sacks_allowed = group.get("sack", 0).sum() if "sack" in group else 0
        qb_hits_allowed = group.get("qb_hit", 0).sum() if "qb_hit" in group else 0
        turnovers_committed = group["is_turnover"].sum()
        cpoe = pass_group["cpoe"].dropna().mean() if "cpoe" in pass_group and not pass_group["cpoe"].dropna().empty else 0.0
        
        off_records.append({
            "game_id": game_id,
            "team": team,
            "off_plays": n_plays,
            "off_pass_plays": n_pass,
            "off_rush_plays": n_rush,
            "off_pass_epa": off_pass_epa,
            "off_rush_epa": off_rush_epa,
            "off_total_epa": off_total_epa,
            "off_epa_per_play": off_epa_per_play,
            "off_success_rate": off_success_rate,
            "off_pass_success_rate": off_pass_success_rate,
            "off_rush_success_rate": off_rush_success_rate,
            "off_sacks_allowed": sacks_allowed,
            "off_qb_hits_allowed": qb_hits_allowed,
            "off_turnovers": turnovers_committed,
            "off_cpoe": cpoe,
        })
        
    df_off = pd.DataFrame(off_records)
    
    # 2. Defensive Aggregations (grouped by game_id and defteam)
    def_records = []
    for (game_id, team), group in valid_pbp.groupby(["game_id", "defteam"]):
        n_plays = len(group)
        pass_group = group[group["is_pass"] == 1]
        rush_group = group[group["is_rush"] == 1]
        
        n_pass = len(pass_group)
        n_rush = len(rush_group)
        
        def_pass_epa_allowed = pass_group["epa"].mean() if n_pass > 0 else 0.0
        def_rush_epa_allowed = rush_group["epa"].mean() if n_rush > 0 else 0.0
        def_total_epa_allowed = group["epa"].sum()
        def_epa_per_play_allowed = group["epa"].mean() if n_plays > 0 else 0.0
        
        def_success_rate_allowed = group["is_success"].mean() if n_plays > 0 else 0.0
        def_sacks_created = group.get("sack", 0).sum() if "sack" in group else 0
        def_qb_hits_created = group.get("qb_hit", 0).sum() if "qb_hit" in group else 0
        def_turnovers_forced = group["is_turnover"].sum()
        
        def_records.append({
            "game_id": game_id,
            "team": team,
            "def_plays_faced": n_plays,
            "def_pass_epa_allowed": def_pass_epa_allowed,
            "def_rush_epa_allowed": def_rush_epa_allowed,
            "def_total_epa_allowed": def_total_epa_allowed,
            "def_epa_per_play_allowed": def_epa_per_play_allowed,
            "def_success_rate_allowed": def_success_rate_allowed,
            "def_sacks_created": def_sacks_created,
            "def_qb_hits_created": def_qb_hits_created,
            "def_turnovers_forced": def_turnovers_forced,
        })
        
    df_def = pd.DataFrame(def_records)
    
    # Merge Offense and Defense metrics for each team in each game
    team_games = pd.merge(df_off, df_def, on=["game_id", "team"], how="outer")
    return team_games


def build_game_matchup_dataset(
    team_games: pd.DataFrame,
    schedules: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges team-game statistics with schedule and situational context,
    producing a tabular dataframe where each row is a game matchup with home and away perspectives.
    """
    sched = schedules.copy()
    sched = standardize_team_abbr(sched, ["home_team", "away_team"])
    
    # Sort chronologically to compute rest days accurately
    if "gameday" in sched.columns:
        sched["gameday"] = pd.to_datetime(sched["gameday"])
    else:
        sched["gameday"] = pd.to_datetime(sched.get("game_date", "2020-01-01"))
        
    sched = sched.sort_values(["gameday", "season", "week"]).reset_index(drop=True)
    
    # Compute rest days for each team
    team_last_date = {}
    home_rest_list = []
    away_rest_list = []
    
    for _, row in sched.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]
        g_date = row["gameday"]
        
        # Home team rest
        if h_team in team_last_date:
            h_rest = (g_date - team_last_date[h_team]).days
        else:
            h_rest = 7  # Default opening rest
        home_rest_list.append(h_rest)
        team_last_date[h_team] = g_date
        
        # Away team rest
        if a_team in team_last_date:
            a_rest = (g_date - team_last_date[a_team]).days
        else:
            a_rest = 7
        away_rest_list.append(a_rest)
        team_last_date[a_team] = g_date
        
    sched["home_rest_days"] = home_rest_list
    sched["away_rest_days"] = away_rest_list
    sched["rest_differential"] = sched["home_rest_days"] - sched["away_rest_days"]
    
    # Target values: point margin (spread target = home_score - away_score)
    if "home_score" in sched.columns and "away_score" in sched.columns:
        sched["actual_point_diff"] = sched["home_score"] - sched["away_score"]
        sched["home_won"] = (sched["actual_point_diff"] > 0).astype(int)
        
    # Merge home team game stats
    home_stats = team_games.add_prefix("home_")
    merged = pd.merge(
        sched,
        home_stats,
        left_on=["game_id", "home_team"],
        right_on=["home_game_id", "home_team"],
        how="left"
    )
    
    # Merge away team game stats
    away_stats = team_games.add_prefix("away_")
    merged = pd.merge(
        merged,
        away_stats,
        left_on=["game_id", "away_team"],
        right_on=["away_game_id", "away_team"],
        how="left"
    )
    
    # Drop redundant join columns
    drop_cols = [c for c in ["home_game_id", "away_game_id"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)
    
    return merged
