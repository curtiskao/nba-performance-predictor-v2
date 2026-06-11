"""Correlation filter, feature importance, and SHAP analysis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORRELATION_THRESHOLD, TARGET_COL


# Raw per-game box score stats — these describe performance DURING the game being
# predicted so they must never be used as features (would be direct leakage).
_RAW_GAME_STATS = {
    "min", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF",
    "FGA", "FGM", "FG_PCT", "FG3A", "FG3M", "FG3_PCT",
    "FTA", "FTM", "FT_PCT", "plus_minus", "possession_usage",
}

# API metadata columns — identifiers or flags with no predictive value.
_METADATA = {"PLAYER_ID", "SEASON_ID", "VIDEO_AVAILABLE"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    non_feature = {
        TARGET_COL, "game_date", "matchup", "game_id", "season", "season_type",
        "win_loss", "home_away", "opp_abbr", "SEASON", "SEASON_TYPE",
    } | _RAW_GAME_STATS | _METADATA
    return [c for c in df.columns if c not in non_feature and pd.api.types.is_numeric_dtype(df[c])]


def correlation_filter(
    df: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
    verbose: bool = True,
) -> list[str]:
    feature_cols = get_feature_cols(df)
    target = df[TARGET_COL]

    scored: list[tuple[str, float]] = []
    skipped_sparse: list[str] = []

    for col in feature_cols:
        series = df[col].dropna()
        if len(series) < 10:
            skipped_sparse.append(col)
            continue
        aligned = series.align(target, join="inner")[0]
        t_aligned = target.loc[aligned.index]
        corr = abs(aligned.corr(t_aligned))
        scored.append((col, corr))

    scored.sort(key=lambda x: x[1], reverse=True)
    kept = [col for col, corr in scored if corr >= threshold]
    dropped = [(col, corr) for col, corr in scored if corr < threshold]

    if verbose:
        print(f"\n  Correlation filter  (threshold |r| ≥ {threshold},  target = {TARGET_COL})")
        print(f"  Candidates : {len(feature_cols)}  |  "
              f"Kept : {len(kept)}  |  "
              f"Dropped : {len(dropped)}  |  "
              f"Sparse (<10 non-null, skipped) : {len(skipped_sparse)}")
        print(f"\n  {'Feature':<35}  {'|r|':>6}  {'status'}")
        print(f"  {'─'*35}  {'─'*6}  {'─'*7}")
        for col, corr in scored:
            status = "KEEP" if corr >= threshold else "drop"
            print(f"  {col:<35}  {corr:>6.3f}  {status}")
        if skipped_sparse:
            print(f"\n  Skipped (sparse): {skipped_sparse}")

    return kept


def rf_feature_importance(
    X: pd.DataFrame, y: pd.Series, n_top: int = 20
) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X.fillna(0), y)
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
    return importance.head(n_top).reset_index(drop=True)


def shap_analysis(model, X: pd.DataFrame) -> pd.DataFrame:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.fillna(0))
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": X.columns, "shap_importance": mean_abs})
        .sort_values("shap_importance", ascending=False)
        .reset_index(drop=True)
    )


def select_features(
    df: pd.DataFrame,
    method: str = "correlation",
    model=None,
) -> list[str]:
    if method == "correlation":
        return correlation_filter(df)
    elif method == "rf_importance":
        cols = correlation_filter(df)
        X = df[cols].fillna(0)
        y = df[TARGET_COL]
        report = rf_feature_importance(X, y)
        return report["feature"].tolist()
    elif method == "shap":
        if model is None:
            raise ValueError("model required for shap method")
        cols = correlation_filter(df)
        X = df[cols].fillna(0)
        report = shap_analysis(model, X)
        return report["feature"].tolist()
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    import argparse

    from src.feature_engineering import load_and_engineer

    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--method", default="rf_importance")
    args = parser.parse_args()

    df = load_and_engineer(args.player)
    df = df.dropna(subset=[TARGET_COL])
    features = select_features(df, method=args.method)
    print(f"\nTop features ({len(features)}):")
    for i, f in enumerate(features[:20], 1):
        print(f"  {i:2d}. {f}")
