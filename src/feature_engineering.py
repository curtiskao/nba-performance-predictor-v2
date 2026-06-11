"""Build feature matrix from processed game logs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURE_GROUPS_ENABLED, PROCESSED_DIR, ROLLING_WINDOWS

REQUIRED_PROCESSED_COLUMNS = ["game_date", "PTS", "season", "season_type"]


def _check(df: pd.DataFrame, cols: list[str], fn: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{fn}] Missing columns: {missing}. Available: {sorted(df.columns.tolist())}")


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, ["PTS"], "add_rolling_features")
    stat_cols = ["PTS", "REB", "AST", "FG_PCT", "min"]
    for col in stat_cols:
        if col not in df.columns:
            continue
        for w in ROLLING_WINDOWS:
            df[f"{col}_roll_mean_{w}"] = (
                df[col].shift(1).rolling(w, min_periods=1).mean()
            )
            df[f"{col}_roll_std_{w}"] = (
                df[col].shift(1).rolling(w, min_periods=1).std().fillna(0)
            )

    # Pandas EWMA with span=5 (alpha=1/3): all prior games weighted by recency.
    # Tree models latch onto this as the single strongest recent-form signal.
    df["pts_last_5_ewm"] = df["PTS"].shift(1).ewm(span=5, min_periods=1).mean()
    return df


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, ["game_date"], "add_rest_features")
    df["days_rest"] = df["game_date"].diff().dt.days.fillna(3).clip(0, 14)
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(int)
    df["games_last_7"] = (
        df["game_date"]
        .apply(lambda d: ((d - df["game_date"]).dt.days.between(1, 7)).sum())
    )
    return df


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    if "home_away" not in df.columns:
        print("  Warning: home_away column missing — defaulting to 0 (away)")
        df["home_away"] = "away"
    df["home_away_enc"] = (df["home_away"] == "home").astype(int)

    if "is_playoffs" not in df.columns:
        df["is_playoffs"] = 0

    df["playoff_round"] = 0
    df["game_in_series"] = 0
    playoff_mask = df["is_playoffs"] == 1
    if playoff_mask.any():
        if "season" not in df.columns:
            print("  Warning: season column missing — playoff round inference skipped.")
        else:
            playoff_df = df[playoff_mask].copy().sort_values("game_date")
            playoff_df["playoff_game_num"] = playoff_df.groupby("season").cumcount() + 1
            df.loc[playoff_mask, "playoff_game_num"] = playoff_df["playoff_game_num"]
            df.loc[playoff_mask, "playoff_round"] = (
                (df.loc[playoff_mask, "playoff_game_num"] - 1) // 14 + 1
            ).clip(1, 4)
            df["playoff_game_num"] = df["playoff_game_num"].fillna(0)
    return df


def add_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    # All usage features use shift(1) to avoid leakage.
    if "FGA" in df.columns and "FTA" in df.columns:
        fga_prev = df["FGA"].shift(1)
        fta_prev = df["FTA"].shift(1)
        pts_prev = df["PTS"].shift(1) if "PTS" in df.columns else pd.Series(np.nan, index=df.index)
        min_prev = df["min"].shift(1) if "min" in df.columns else pd.Series(np.nan, index=df.index)

        denom = 2 * (fga_prev + 0.44 * fta_prev)
        df["true_shooting_pct"] = np.where(denom > 0, pts_prev / denom, np.nan)
        df["fga_per_36"] = np.where(min_prev > 0, fga_prev / min_prev * 36, np.nan)
    else:
        print("  Warning: FGA/FTA missing — skipping true_shooting_pct and fga_per_36.")

    if "FGA" in df.columns and "FTA" in df.columns and "TOV" in df.columns:
        possession_usage = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
        df["usage_rate"] = possession_usage.shift(1).rolling(5, min_periods=1).mean()
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    if "PTS" not in df.columns:
        print("  Warning: PTS missing — skipping trend features.")
        return df

    window = 5

    def pts_slope(series) -> float:
        vals = np.asarray(series)
        if len(vals) < 2:
            return 0.0
        x = np.arange(len(vals))
        slope, *_ = scipy_stats.linregress(x, vals)
        return slope

    df["pts_trend_slope"] = (
        df["PTS"]
        .shift(1)
        .rolling(window, min_periods=2)
        .apply(pts_slope, raw=True)
        .fillna(0)
    )
    return df


def add_recency_weighted_features(df: pd.DataFrame) -> pd.DataFrame:
    if "PTS" not in df.columns:
        print("  Warning: PTS missing — skipping recency-weighted features.")
        return df

    decay = 0.9

    def weighted_mean(vals) -> float:
        vals = np.asarray(vals)
        weights = np.array([decay ** i for i in range(len(vals) - 1, -1, -1)])
        if weights.sum() == 0:
            return np.nan
        return np.dot(weights, vals) / weights.sum()

    for w in ROLLING_WINDOWS:
        df[f"PTS_ewm_{w}"] = (
            df["PTS"]
            .shift(1)
            .rolling(w, min_periods=1)
            .apply(weighted_mean, raw=True)
        )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, REQUIRED_PROCESSED_COLUMNS, "build_features")

    df = df.sort_values("game_date").reset_index(drop=True)
    enabled = FEATURE_GROUPS_ENABLED

    if enabled.get("rolling"):
        df = add_rolling_features(df)
    if enabled.get("rest"):
        df = add_rest_features(df)
    if enabled.get("context"):
        df = add_context_features(df)
    if enabled.get("usage"):
        df = add_usage_features(df)
    if enabled.get("trend"):
        df = add_trend_features(df)
    if enabled.get("rolling"):
        df = add_recency_weighted_features(df)

    return df


def load_and_engineer(
    player_name: str, processed_dir: str = PROCESSED_DIR
) -> pd.DataFrame:
    safe_name = player_name.lower().replace(" ", "_")
    path = Path(processed_dir) / f"{safe_name}_processed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}\n"
            f"  Run: python src/process_data.py --player \"{player_name}\""
        )
    try:
        df = pd.read_csv(path, parse_dates=["game_date"])
    except Exception as e:
        raise ValueError(f"Failed to read {path}: {e}") from e

    if df.empty:
        raise ValueError(f"Processed file is empty: {path}")

    return build_features(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    args = parser.parse_args()

    try:
        df = load_and_engineer(args.player)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
