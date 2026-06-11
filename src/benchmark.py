"""Naive baselines: season avg, last-N avg, median, opponent-adjusted."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TARGET_COL, TEST_SPLIT_GAMES


def _metrics(y_true, y_pred, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:<35} MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return {"baseline": label, "MAE": mae, "RMSE": rmse, "R2": r2}


def run_baselines(df: pd.DataFrame, test_games: int = TEST_SPLIT_GAMES) -> pd.DataFrame:
    df = df.sort_values("game_date").dropna(subset=[TARGET_COL]).reset_index(drop=True)
    train = df.iloc[:-test_games]
    test = df.iloc[-test_games:]

    y_test = test[TARGET_COL].values
    results = []

    # Season average
    season_avg = train[TARGET_COL].mean()
    results.append(_metrics(y_test, np.full(len(y_test), season_avg), "Season average"))

    # Expanding mean (up to each test game)
    expanding_preds = []
    for i in range(len(test)):
        history = pd.concat([train[TARGET_COL], test[TARGET_COL].iloc[:i]])
        expanding_preds.append(history.mean())
    results.append(_metrics(y_test, expanding_preds, "Expanding mean"))

    # Last-5 average
    last5_preds = []
    for i in range(len(test)):
        history = pd.concat([train[TARGET_COL], test[TARGET_COL].iloc[:i]])
        last5_preds.append(history.iloc[-5:].mean())
    results.append(_metrics(y_test, last5_preds, "Last-5 average"))

    # Last-10 average
    last10_preds = []
    for i in range(len(test)):
        history = pd.concat([train[TARGET_COL], test[TARGET_COL].iloc[:i]])
        last10_preds.append(history.iloc[-10:].mean())
    results.append(_metrics(y_test, last10_preds, "Last-10 average"))

    # Median
    median_val = train[TARGET_COL].median()
    results.append(_metrics(y_test, np.full(len(y_test), median_val), "Season median"))

    # Opponent-adjusted (if opp_def_rating available)
    if "opp_def_rating" in df.columns:
        opp_avg = train["opp_def_rating"].mean()
        adj_preds = []
        for i, row in test.iterrows():
            base = pd.concat([train[TARGET_COL], test[TARGET_COL].iloc[:test.index.get_loc(i)]]).mean()
            if pd.notna(row.get("opp_def_rating")) and opp_avg > 0:
                adj = base * (opp_avg / row["opp_def_rating"])
            else:
                adj = base
            adj_preds.append(adj)
        results.append(_metrics(y_test, adj_preds, "Opponent-adjusted avg"))

    # Playoff-only subset
    if "is_playoffs" in test.columns and test["is_playoffs"].sum() > 0:
        po_mask = test["is_playoffs"] == 1
        y_po = test.loc[po_mask, TARGET_COL].values
        pred_po = np.full(len(y_po), train[TARGET_COL].mean())
        if len(y_po) > 0:
            results.append(_metrics(y_po, pred_po, "Season avg (playoff-only test)"))

    return pd.DataFrame(results)


def main():
    import argparse
    from src.feature_engineering import load_and_engineer

    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--test-games", type=int, default=TEST_SPLIT_GAMES)
    args = parser.parse_args()

    df = load_and_engineer(args.player)
    print(f"\nBaseline results for {args.player} (last {args.test_games} games as test):\n")
    results = run_baselines(df, args.test_games)
    print(f"\n{results.to_string(index=False)}")


if __name__ == "__main__":
    main()
