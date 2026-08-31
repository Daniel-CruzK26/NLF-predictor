"""Situational, weather, and rest feature engineering."""

import numpy as np
import pandas as pd


def add_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and engineers weather, dome/stadium, and rest features.
    """
    df = df.copy()

    # Stadium / Roof features
    roof_series = df["roof"].fillna("outdoors").str.lower() if "roof" in df.columns else pd.Series("outdoors", index=df.index)
    df["is_dome"] = roof_series.isin(["dome", "closed"]).astype(int)
    df["is_retractable"] = roof_series.isin(["retractable", "open"]).astype(int)
    df["is_outdoors"] = roof_series.isin(["outdoors", "open"]).astype(int)

    # Weather: Wind
    if "wind" in df.columns:
        wind_clean = pd.to_numeric(df["wind"], errors="coerce")
        # Domes have 0 wind; outdoor games default to median (~8 mph)
        wind_clean = np.where(df["is_dome"] == 1, 0.0, wind_clean)
        df["wind_speed"] = pd.Series(wind_clean, index=df.index).fillna(7.0)
    else:
        df["wind_speed"] = 0.0

    df["high_wind_flag"] = (df["wind_speed"] >= 15.0).astype(int)
    df["extreme_wind_flag"] = (df["wind_speed"] >= 20.0).astype(int)

    # Weather: Temperature
    if "temp" in df.columns:
        temp_clean = pd.to_numeric(df["temp"], errors="coerce")
        # Domes have climate control ~70°F; outdoor games default to ~60°F
        temp_clean = np.where(df["is_dome"] == 1, 70.0, temp_clean)
        df["temperature"] = pd.Series(temp_clean, index=df.index).fillna(60.0)
    else:
        df["temperature"] = 70.0

    df["freezing_temp_flag"] = (df["temperature"] <= 32.0).astype(int)
    df["cold_temp_flag"] = (df["temperature"] <= 45.0).astype(int)

    # Rest and division factors
    if "rest_differential" not in df.columns:
        h_rest = (
            df["home_rest"].fillna(7)
            if "home_rest" in df.columns
            else (
                df["home_rest_days"].fillna(7)
                if "home_rest_days" in df.columns
                else pd.Series(7, index=df.index)
            )
        )
        a_rest = (
            df["away_rest"].fillna(7)
            if "away_rest" in df.columns
            else (
                df["away_rest_days"].fillna(7)
                if "away_rest_days" in df.columns
                else pd.Series(7, index=df.index)
            )
        )
        df["rest_differential"] = h_rest - a_rest

    if "div_game" in df.columns:
        df["is_division_game"] = df["div_game"].fillna(0).astype(int)
    else:
        df["is_division_game"] = 0

    # Weather Interaction: Wind vs Passing EPA
    if "home_ewma_off_pass_epa" in df.columns and "away_ewma_off_pass_epa" in df.columns:
        total_pass_reliance = df["home_ewma_off_pass_epa"] + df["away_ewma_off_pass_epa"]
        df["wind_pass_decay_interaction"] = (df["wind_speed"] / 10.0) * total_pass_reliance

    return df
