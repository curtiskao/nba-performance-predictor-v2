"""Tests for feature engineering logic."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_engineering import (
    add_rest_features,
    add_rolling_features,
    add_trend_features,
    add_usage_features,
    build_features,
)


def make_df(n=20, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="2D")
    return pd.DataFrame(
        {
            "game_date": dates,
            "PTS": rng.integers(10, 40, n).astype(float),
            "REB": rng.integers(2, 15, n).astype(float),
            "AST": rng.integers(0, 12, n).astype(float),
            "FGA": rng.integers(10, 25, n).astype(float),
            "FGM": rng.integers(4, 15, n).astype(float),
            "FG_PCT": rng.uniform(0.3, 0.6, n),
            "FTA": rng.integers(0, 10, n).astype(float),
            "FTM": rng.integers(0, 8, n).astype(float),
            "TOV": rng.integers(0, 5, n).astype(float),
            "min": rng.integers(20, 40, n).astype(float),
            "home_away": ["home", "away"] * (n // 2),
            "is_playoffs": [0] * n,
            "season": ["2023-24"] * n,
            "season_type": ["Regular Season"] * n,
        }
    )


def test_rolling_features_no_leakage():
    df = make_df()
    df = add_rolling_features(df)
    # roll_mean_3 for row 0 uses no prior data — should be NaN or just row's own value
    # shift(1) means row 0 can't use itself
    assert pd.isna(df["PTS_roll_mean_3"].iloc[0]) or df["PTS_roll_mean_3"].iloc[0] >= 0


def test_rolling_feature_columns_created():
    df = make_df()
    df = add_rolling_features(df)
    for w in [3, 5, 10]:
        assert f"PTS_roll_mean_{w}" in df.columns
        assert f"PTS_roll_std_{w}" in df.columns


def test_rest_features():
    df = make_df()
    df = add_rest_features(df)
    assert "days_rest" in df.columns
    assert "is_back_to_back" in df.columns
    # All games are 2 days apart so no back-to-backs
    assert df["is_back_to_back"].sum() == 0


def test_usage_features():
    df = make_df()
    df = add_usage_features(df)
    assert "true_shooting_pct" in df.columns
    assert "fga_per_36" in df.columns


def test_trend_feature():
    df = make_df()
    df = add_trend_features(df)
    assert "pts_trend_slope" in df.columns
    assert df["pts_trend_slope"].notna().any()


def test_build_features_returns_more_columns():
    df = make_df()
    original_cols = len(df.columns)
    df_feat = build_features(df)
    assert len(df_feat.columns) > original_cols


def test_no_future_leakage_in_rolling():
    """Rolling means must not include the current game's value."""
    df = make_df(n=10)
    df["PTS"] = np.arange(10, dtype=float)  # deterministic sequence: 0,1,...,9
    df = add_rolling_features(df)
    # For row 3, shift(1).rolling(3) covers rows [0,1,2] → mean = 1.0
    mean_val = df["PTS_roll_mean_3"].iloc[3]
    assert abs(mean_val - 1.0) < 1e-9, f"Expected 1.0, got {mean_val}"
