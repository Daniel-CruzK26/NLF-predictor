"""Aggregates play-by-play data at the Quarterback (QB) game level."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from .loader import standardize_team_abbr


def extract_qb_dropbacks(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts all QB dropbacks (pass attempts, sacks, and scrambles) from PBP data.
    """
    df = pbp_df.copy()
    df = standardize_team_abbr(df, ["home_team", "away_team", "posteam", "defteam"])

    # Identification of dropbacks
    is_pass = (df.get("play_type", "") == "pass") | (df.get("pass_attempt", 0) == 1)
    is_sack = df.get("sack", 0) == 1
    is_scramble = df.get("qb_scramble", 0) == 1

    dropback_mask = (
        (is_pass | is_sack | is_scramble)
        & df["epa"].notna()
        & (df.get("two_point_attempt", 0) == 0)
        & df["posteam"].notna()
    )

    df_dropbacks = df[dropback_mask].copy()

    # Identify primary QB ID and Name for each dropback
    qb_ids = []
    qb_names = []

    for _, row in df_dropbacks.iterrows():
        # 1. Check passer columns
        pid = row.get("passer_player_id")
        pname = row.get("passer_player_name")

        # 2. Check scramble/rusher if passer is missing
        if pd.isna(pid) or not pid:
            pid = row.get("rusher_player_id")
            pname = row.get("rusher_player_name")

        # 3. Fallback to general player name/id
        if pd.isna(pid) or not pid:
            pid = row.get("fantasy_player_id", row.get("id"))
            pname = row.get("fantasy_player_name", row.get("name"))

        qb_ids.append(pid if pd.notna(pid) else "UNKNOWN_QB")
        qb_names.append(pname if pd.notna(pname) else "Unknown")

    df_dropbacks["qb_id"] = qb_ids
    df_dropbacks["qb_name"] = qnames = qb_names
    df_dropbacks["is_success"] = (df_dropbacks["epa"] > 0).astype(int)

    interception_col = (
        df_dropbacks["interception"].fillna(0)
        if "interception" in df_dropbacks.columns
        else pd.Series(0, index=df_dropbacks.index)
    )
    fumble_col = (
        df_dropbacks["fumble_lost"].fillna(0)
        if "fumble_lost" in df_dropbacks.columns
        else pd.Series(0, index=df_dropbacks.index)
    )
    df_dropbacks["is_turnover"] = (interception_col + fumble_col).clip(upper=1)

    return df_dropbacks


def aggregate_qb_game_stats(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates dropbacks per QB per game, computing EPA, CPOE, Composite Score, and efficiency metrics.
    """
    dropbacks_df = extract_qb_dropbacks(pbp_df)

    records = []
    for (game_id, team, qb_id, qb_name), group in dropbacks_df.groupby(
        ["game_id", "posteam", "qb_id", "qb_name"]
    ):
        n_dropbacks = len(group)
        if n_dropbacks == 0 or qb_id == "UNKNOWN_QB":
            continue

        pass_plays = group[group.get("pass_attempt", 0) == 1]
        n_passes = len(pass_plays)

        completions = group.get("complete_pass", pd.Series(0, index=group.index)).sum()
        pass_yards = group.get("passing_yards", pd.Series(0, index=group.index)).sum()
        pass_tds = group.get("pass_touchdown", pd.Series(0, index=group.index)).sum()
        sacks = group.get("sack", pd.Series(0, index=group.index)).sum()
        turnovers = group["is_turnover"].sum()

        total_epa = group["epa"].sum()
        epa_per_dropback = total_epa / n_dropbacks
        success_rate = group["is_success"].mean()

        # CPOE (Completion Percentage Over Expected)
        cpoe_vals = pass_plays["cpoe"].dropna() if "cpoe" in pass_plays.columns else pd.Series(dtype=float)
        cpoe_mean = float(cpoe_vals.mean()) if not cpoe_vals.empty else 0.0

        # Air Yards / aDOT
        air_yards_vals = pass_plays["air_yards"].dropna() if "air_yards" in pass_plays.columns else pd.Series(dtype=float)
        air_yards_mean = float(air_yards_vals.mean()) if not air_yards_vals.empty else 0.0

        # NFL Analytics EPA + CPOE Composite Index (standard scaled)
        # Composite = 0.75 * EPA_per_dropback + 0.25 * (CPOE / 100)
        composite_score = (0.75 * epa_per_dropback) + (0.25 * (cpoe_mean / 100.0))

        records.append({
            "game_id": game_id,
            "team": team,
            "qb_id": qb_id,
            "qb_name": qb_name,
            "qb_dropbacks": n_dropbacks,
            "qb_passes": n_passes,
            "qb_completions": completions,
            "qb_passing_yards": pass_yards,
            "qb_pass_tds": pass_tds,
            "qb_sacks_taken": sacks,
            "qb_turnovers": turnovers,
            "qb_total_epa": total_epa,
            "qb_epa_per_dropback": epa_per_dropback,
            "qb_cpoe": cpoe_mean,
            "qb_composite_score": composite_score,
            "qb_success_rate": success_rate,
            "qb_air_yards_mean": air_yards_mean,
            "qb_sack_rate": sacks / n_dropbacks if n_dropbacks > 0 else 0.0,
            "qb_turnover_rate": turnovers / n_dropbacks if n_dropbacks > 0 else 0.0,
        })

    df_qb = pd.DataFrame(records)
    return df_qb


def get_starting_qbs_per_game(qb_game_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the primary starting QB for each team in each game (the QB with the most dropbacks).
    Returns a dataframe with exactly one row per (game_id, team).
    """
    df = qb_game_stats.sort_values(
        ["game_id", "team", "qb_dropbacks"], ascending=[True, True, False]
    )
    starters = df.drop_duplicates(subset=["game_id", "team"], keep="first").copy()
    return starters.reset_index(drop=True)
