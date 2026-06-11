"""Train Linear Regression, Random Forest, and XGBoost models."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, PLAYOFF_WEIGHT_MULTIPLIER, RECENCY_DECAY_RATE,
    RF_PARAMS, TARGET_COL, TEST_SPLIT_GAMES, XGBOOST_PARAMS,
)
from src.feature_engineering import load_and_engineer
from src.feature_selection import correlation_filter


def compute_sample_weights(
    train_df: pd.DataFrame,
    decay_rate: float = RECENCY_DECAY_RATE,
    playoff_multiplier: float = PLAYOFF_WEIGHT_MULTIPLIER,
) -> np.ndarray:
    n = len(train_df)
    # Most recent game = 1.0; older games decay exponentially by position.
    recency = np.array([decay_rate ** (n - 1 - i) for i in range(n)])
    is_po = train_df["is_playoffs"].values if "is_playoffs" in train_df.columns else np.zeros(n)
    playoff_mult = np.where(is_po == 1, playoff_multiplier, 1.0)
    weights = recency * playoff_mult
    # Normalize to mean=1 so scale doesn't shift effective regularization strength.
    return weights / weights.mean()


def _split(df: pd.DataFrame, test_games: int):
    df = df.sort_values("game_date").reset_index(drop=True)
    train = df.iloc[:-test_games].copy()
    test = df.iloc[-test_games:].copy()
    return train, test


def _prep(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]):
    X_train = train[feature_cols].fillna(0)
    y_train = train[TARGET_COL]
    X_test = test[feature_cols].fillna(0)
    y_test = test[TARGET_COL]
    return X_train, y_train, X_test, y_test


def _report(name: str, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:<20} MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def train_linear(X_train, y_train, X_test, y_test, sample_weight=None) -> tuple:
    # Ridge regression: L2 regularization handles high feature count / small sample
    # without blowing up coefficients the way OLS does (fixes R²=-213 from prior runs).
    model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=500.0))])
    print(f"  [DEBUG] LinearRegression — type: {type(model).__name__}, "
          f"steps: {[name for name, _ in model.steps]}, "
          f"alpha={model.named_steps['model'].alpha}")
    fit_kwargs = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(X_train, y_train, **fit_kwargs)
    preds = model.predict(X_test)
    return model, _report("LinearRegression", y_test, preds)


def train_rf(X_train, y_train, X_test, y_test, sample_weight=None) -> tuple:
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(**RF_PARAMS, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    preds = model.predict(X_test)
    return model, _report("RandomForest", y_test, preds)


def train_xgboost(X_train, y_train, X_test, y_test, tune: bool = False, sample_weight=None) -> tuple:
    import xgboost as xgb

    if tune:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Carve a validation set from the END of training data (chronological).
        # Never touch X_test/y_test during tuning.
        val_size = max(10, len(X_train) // 5)
        X_tr = X_train.iloc[:-val_size]
        y_tr = y_train.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        sw_tr = sample_weight[:-val_size] if sample_weight is not None else None

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
            }
            m = xgb.XGBRegressor(**params, verbosity=0)
            m.fit(X_tr, y_tr, sample_weight=sw_tr)
            return mean_absolute_error(y_val, m.predict(X_val))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=40, show_progress_bar=False)
        best_params = {**study.best_params, "random_state": 42}
        print(f"  Best XGB params: {best_params}")
        model = xgb.XGBRegressor(**best_params, verbosity=0)
    else:
        model = xgb.XGBRegressor(**XGBOOST_PARAMS, random_state=42, verbosity=0)

    model.fit(X_train, y_train, sample_weight=sample_weight)
    preds = model.predict(X_test)
    return model, _report("XGBoost", y_test, preds)


_SEP = "─" * 60


def _section(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _print_feature_importance(name: str, model, feature_cols: list[str], top_n: int = 15) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\n  Top {top_n} features  ({name}):")
    print(f"  {'Feature':<35}  {'Importance':>10}")
    print(f"  {'─'*35}  {'─'*10}")
    for feat, score in imp[:top_n]:
        print(f"  {feat:<35}  {score:>10.4f}")


def train_all(
    player_name: str,
    test_games: int = TEST_SPLIT_GAMES,
    tune_xgb: bool = False,
    models_dir: str = MODELS_DIR,
    save: bool = True,
) -> dict:
    _section(f"train  ·  {player_name}")

    df = load_and_engineer(player_name)
    df = df.dropna(subset=[TARGET_COL])

    train_df, test_df = _split(df, test_games)

    # ── Dataset overview ──────────────────────────────────────────
    _section("Dataset overview")
    po_train = int(train_df["is_playoffs"].sum()) if "is_playoffs" in train_df.columns else 0
    po_test  = int(test_df["is_playoffs"].sum())  if "is_playoffs" in test_df.columns  else 0
    print(f"  Train : {len(train_df)} games  ({po_train} playoff)  "
          f"  {train_df['game_date'].min().date()} → {train_df['game_date'].max().date()}")
    print(f"  Test  : {len(test_df)} games  ({po_test} playoff)  "
          f"  {test_df['game_date'].min().date()} → {test_df['game_date'].max().date()}")

    for label, subset in [("Train", train_df), ("Test", test_df)]:
        pts = subset[TARGET_COL]
        print(f"\n  {label} {TARGET_COL}  →  "
              f"mean={pts.mean():.1f}  std={pts.std():.1f}  "
              f"min={pts.min():.0f}  p25={pts.quantile(0.25):.0f}  "
              f"median={pts.median():.0f}  p75={pts.quantile(0.75):.0f}  "
              f"max={pts.max():.0f}")

    # ── Feature selection (on train only) ─────────────────────────
    _section("Feature selection  (correlation filter on train set only)")
    feature_cols = correlation_filter(train_df, verbose=True)

    # ── Sample weights ────────────────────────────────────────────
    _section("Sample weights")
    weights = compute_sample_weights(train_df)
    print(f"  decay_rate={RECENCY_DECAY_RATE}  playoff_multiplier=×{PLAYOFF_WEIGHT_MULTIPLIER}")
    print(f"  Weight range : {weights.min():.3f} – {weights.max():.3f}  (mean=1.000)")
    oldest_w = weights[0]
    newest_w = weights[-1]
    print(f"  Oldest game  : weight={oldest_w:.3f}  "
          f"(game 1 of {len(train_df)})")
    print(f"  Newest game  : weight={newest_w:.3f}")
    if po_train > 0:
        po_weights = weights[train_df["is_playoffs"].values == 1]
        print(f"  Playoff games: {po_train} rows  weight range={po_weights.min():.3f}–{po_weights.max():.3f}")

    # ── NaN audit on selected features ───────────────────────────
    nan_counts = train_df[feature_cols].isnull().sum()
    nan_features = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if len(nan_features):
        print(f"\n  Features with NaN in training set (filled with 0 before fit):")
        for feat, n in nan_features.items():
            print(f"    {feat:<35}: {n:>3} NaN ({100*n/len(train_df):.0f}%)")
    else:
        print("\n  No NaN in selected training features.")

    # ── Training ──────────────────────────────────────────────────
    _section("Training models")
    X_train, y_train, X_test, y_test = _prep(train_df, test_df, feature_cols)
    print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}\n")

    linear_model, linear_metrics = train_linear(X_train, y_train, X_test, y_test, sample_weight=weights)
    rf_model,     rf_metrics     = train_rf(X_train, y_train, X_test, y_test, sample_weight=weights)
    xgb_model,   xgb_metrics    = train_xgboost(X_train, y_train, X_test, y_test, tune=tune_xgb, sample_weight=weights)

    # ── Feature importance ────────────────────────────────────────
    _section("Feature importance")
    _print_feature_importance("XGBoost", xgb_model, feature_cols)
    _print_feature_importance("RandomForest", rf_model, feature_cols)

    # ── Prediction distribution on test set ───────────────────────
    _section("Prediction distribution  (test set)")
    for name, model in [("LinearRegression", linear_model), ("RandomForest", rf_model), ("XGBoost", xgb_model)]:
        preds = model.predict(X_test)
        print(f"  {name:<20}  mean={preds.mean():.1f}  std={preds.std():.1f}  "
              f"min={preds.min():.1f}  max={preds.max():.1f}")
    print(f"  {'Actual':<20}  mean={y_test.mean():.1f}  std={y_test.std():.1f}  "
          f"min={y_test.min():.0f}  max={y_test.max():.0f}")

    # ── Playoff-only eval ─────────────────────────────────────────
    if "is_playoffs" in test_df.columns and po_test > 0:
        _section(f"Playoff-only eval  ({po_test} test games)")
        po_mask = test_df["is_playoffs"] == 1
        for name, model in [("LinearRegression", linear_model), ("RandomForest", rf_model), ("XGBoost", xgb_model)]:
            X_po = test_df.loc[po_mask, feature_cols].fillna(0)
            y_po = test_df.loc[po_mask, TARGET_COL]
            _report(f"{name} (playoff)", y_po, model.predict(X_po))

    # ── Final metrics summary ─────────────────────────────────────
    _section("Final metrics summary")
    metrics_df = pd.DataFrame([linear_metrics, rf_metrics, xgb_metrics])
    print(metrics_df.to_string(index=False))

    if save:
        safe_name = player_name.lower().replace(" ", "_")
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": linear_model, "features": feature_cols}, models_path / f"{safe_name}_linear.pkl")
        joblib.dump({"model": rf_model,     "features": feature_cols}, models_path / f"{safe_name}_rf.pkl")
        joblib.dump({"model": xgb_model,    "features": feature_cols}, models_path / f"{safe_name}_xgboost.pkl")
        print(f"\n  Models saved → {models_dir}/")

    return {
        "linear": linear_model,
        "rf": rf_model,
        "xgboost": xgb_model,
        "features": feature_cols,
        "metrics": metrics_df,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--test-games", type=int, default=TEST_SPLIT_GAMES)
    parser.add_argument("--tune-xgb", action="store_true")
    parser.add_argument("--models-dir", default=MODELS_DIR)
    args = parser.parse_args()

    try:
        train_all(args.player, args.test_games, args.tune_xgb, args.models_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"\nColumn error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nData error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
