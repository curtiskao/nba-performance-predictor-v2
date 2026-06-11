"""Tests for process_data normalization logic."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.process_data import normalize_schema


def make_raw_df():
    return pd.DataFrame(
        {
            "GAME_DATE": ["2024-01-01", "2024-01-03", "2024-01-05"],
            "MATCHUP": ["BOS vs. MIA", "BOS @ NYK", "BOS vs. PHI"],
            "WL": ["W", "L", "W"],
            "MIN": [36, 34, 38],
            "PTS": [28, 22, 31],
            "REB": [8, 6, 9],
            "AST": [5, 4, 6],
            "STL": [1, 0, 2],
            "BLK": [0, 1, 0],
            "TOV": [2, 3, 1],
            "FGA": [20, 18, 22],
            "FGM": [11, 9, 13],
            "FG_PCT": [0.55, 0.50, 0.59],
            "FG3A": [5, 4, 6],
            "FG3M": [2, 1, 3],
            "FG3_PCT": [0.40, 0.25, 0.50],
            "FTA": [6, 4, 5],
            "FTM": [4, 4, 2],
            "FT_PCT": [0.67, 1.0, 0.40],
            "PLUS_MINUS": [10, -5, 8],
            "GAME_ID": ["001", "002", "003"],
            "SEASON": ["2023-24", "2023-24", "2023-24"],
            "SEASON_TYPE": ["Regular Season", "Regular Season", "Regular Season"],
        }
    )


def test_normalize_adds_home_away():
    df = normalize_schema(make_raw_df())
    assert "home_away" in df.columns
    assert df["home_away"].iloc[0] == "home"
    assert df["home_away"].iloc[1] == "away"


def test_normalize_adds_opp_abbr():
    df = normalize_schema(make_raw_df())
    assert "opp_abbr" in df.columns
    assert df["opp_abbr"].iloc[0] == "MIA"
    assert df["opp_abbr"].iloc[1] == "NYK"


def test_normalize_sorts_by_date():
    raw = make_raw_df()
    raw = raw.iloc[::-1].reset_index(drop=True)  # reverse order
    df = normalize_schema(raw)
    dates = pd.to_datetime(df["game_date"])
    assert list(dates) == sorted(dates)


def test_normalize_is_playoffs_flag():
    raw = make_raw_df()
    raw.loc[2, "SEASON_TYPE"] = "Playoffs"
    df = normalize_schema(raw)
    assert df["is_playoffs"].iloc[2] == 1
    assert df["is_playoffs"].iloc[0] == 0


def test_normalize_deduplicates():
    raw = pd.concat([make_raw_df(), make_raw_df()], ignore_index=True)
    df = normalize_schema(raw)
    assert len(df) == 3
