"""NFL Data Loader using nfl_data_py with Parquet caching."""

from typing import List, Optional
import os
import pandas as pd
import nfl_data_py as nfl
from ..config import RAW_DATA_DIR, DEFAULT_SEASONS, TEAM_ABBR_MAP


def standardize_team_abbr(df: pd.DataFrame, team_cols: List[str]) -> pd.DataFrame:
    """Standardize historical team abbreviations (e.g. OAK -> LV, SD -> LAC)."""
    df = df.copy()
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_ABBR_MAP)
    return df


def load_schedules(
    seasons: Optional[List[int]] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load NFL schedules for specified seasons.
    Caches results to disk as parquet.
    """
    if seasons is None:
        seasons = DEFAULT_SEASONS
    
    seasons_sorted = sorted(seasons)
    cache_path = RAW_DATA_DIR / f"schedules_{seasons_sorted[0]}_{seasons_sorted[-1]}.parquet"
    
    if cache_path.exists() and not force_download:
        df_schedules = pd.read_parquet(cache_path)
    else:
        df_schedules = nfl.import_schedules(seasons)
        df_schedules = standardize_team_abbr(df_schedules, ["home_team", "away_team"])
        df_schedules.to_parquet(cache_path, index=False)
        
    return df_schedules


def load_pbp_data(
    seasons: Optional[List[int]] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load play-by-play (pbp) data for specified seasons.
    Caches each season individually as parquet for performance.
    """
    if seasons is None:
        seasons = DEFAULT_SEASONS
    
    dfs = []
    for season in seasons:
        season_cache = RAW_DATA_DIR / f"pbp_{season}.parquet"
        if season_cache.exists() and not force_download:
            df_season = pd.read_parquet(season_cache)
        else:
            df_season = nfl.import_pbp_data([season])
            df_season = standardize_team_abbr(df_season, ["home_team", "away_team", "posteam", "defteam"])
            df_season.to_parquet(season_cache, index=False)
        dfs.append(df_season)
        
    pbp_df = pd.concat(dfs, ignore_index=True)
    return pbp_df
