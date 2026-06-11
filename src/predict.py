"""Load model and predict points for a player in an upcoming game."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, TARGET_COL, TEST_SPLIT_GAMES
from src.feature_engineering import load_and_engineer
from src.nba_api_client import NBAApiClient


def build_prediction_row(
    df: pd.DataFrame,
    opponent: str,
    home_away: str = "home",
    is_playoffs: int = 0,
    season: str = None,
    feature_cols: list = None,
    smooth: bool = True,
) -> pd.DataFrame:
    """Construct a feature row for the next game using latest known stats."""
    df = df.sort_values("game_date").reset_index(drop=True)
    row = df.iloc[-1].copy()

    # Smooth rolling-mean and EWM features: replace the last single game's value
    # with the median over the last 5 rows to avoid anchoring to one noisy game.
    # Skipped when smooth=False (--no-smooth flag) to use raw last-game values.
    if feature_cols is not None:
        smooth_cols = [c for c in feature_cols if "roll_mean" in c or "ewm" in c]
        if smooth:
            print(f"  [DEBUG] Smoothing {len(smooth_cols)} rolling/EWM cols "
                  f"(feature_cols={len(feature_cols)} total): {smooth_cols}")
            for col in smooth_cols:
                if col in df.columns:
                    old_val = row[col]
                    row[col] = df[col].iloc[-5:].median()
                    print(f"    smooth: {col}: {old_val:.2f} → {row[col]:.2f}")
        else:
            print(f"  [--no-smooth] Using raw last-game values for "
                  f"{len(smooth_cols)} rolling/EWM cols (no median replacement)")

    # Override context for the upcoming game
    row["home_away"] = home_away
    row["home_away_enc"] = int(home_away == "home")
    row["opp_abbr"] = opponent.upper()[:3]
    row["is_playoffs"] = is_playoffs
    row["days_rest"] = 1  # default assumption

    # Pull opponent stats: season-average (def rating, pace) + rolling last-10 pts allowed.
    if season:
        client = NBAApiClient()
        opp_abbr = opponent.upper()[:3]

        try:
            team_stats = client.get_league_team_stats(season)
            opp_row = team_stats[team_stats["TEAM_ABBREVIATION"].str.upper() == opp_abbr]
            if not opp_row.empty:
                if "E_DEF_RATING" in opp_row.columns:
                    row["opp_def_rating"] = opp_row["E_DEF_RATING"].values[0]
                if "PACE" in opp_row.columns:
                    row["opp_pace"] = opp_row["PACE"].values[0]
        except Exception as exc:
            print(f"  Warning: could not fetch season team stats: {exc}")

        try:
            # Compute opponent's rolling last-10 pts allowed from the league game log.
            all_frames = []
            for stype in ["Regular Season", "Playoffs"]:
                try:
                    g = client.get_league_game_log(season, stype)
                    if not g.empty:
                        all_frames.append(g)
                except Exception:
                    pass
            if all_frames:
                games = pd.concat(all_frames, ignore_index=True)
                games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
                opp_games = games[games["TEAM_ABBREVIATION"] == opp_abbr].copy()
                other_games = games[["TEAM_ABBREVIATION", "GAME_ID", "PTS"]].rename(
                    columns={"TEAM_ABBREVIATION": "OPP_TEAM", "PTS": "pts_allowed"}
                )
                opp_with_pa = opp_games[["GAME_ID", "GAME_DATE"]].merge(other_games, on="GAME_ID")
                opp_with_pa = opp_with_pa[opp_with_pa["OPP_TEAM"] != opp_abbr]
                opp_with_pa = opp_with_pa.sort_values("GAME_DATE")
                if len(opp_with_pa) >= 3:
                    row["opp_pts_allowed_last10"] = opp_with_pa["pts_allowed"].tail(10).mean()
        except Exception as exc:
            print(f"  Warning: could not compute rolling opponent defense: {exc}")

    return pd.DataFrame([row])


def predict(
    player_name: str,
    opponent: str,
    model_type: str = "xgboost",
    home_away: str = "home",
    is_playoffs: int = 0,
    season: str = None,
    models_dir: str = MODELS_DIR,
    n_bootstrap: int = 1000,
    smooth: bool = True,
) -> dict:
    safe_name = player_name.lower().replace(" ", "_")
    model_path = Path(models_dir) / f"{safe_name}_{model_type}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train.py first."
        )

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["features"]

    df = load_and_engineer(player_name)
    df = df.dropna(subset=[TARGET_COL])

    n_playoff_games = int(df["is_playoffs"].sum()) if "is_playoffs" in df.columns else 0
    limited_sample = n_playoff_games < 10 and is_playoffs

    pred_row = build_prediction_row(df, opponent, home_away, is_playoffs, season, feature_cols, smooth=smooth)

    # Align features
    X_pred = pred_row.reindex(columns=feature_cols, fill_value=0).fillna(0)

    # Sanity-check: print key feature values going into the model.
    sanity_features = [
        "PTS_roll_mean_3", "PTS_roll_mean_5", "PTS_roll_mean_10",
        "pts_last_5_ewm", "FG_PCT_roll_mean_10", "true_shooting_pct",
        "is_playoffs", "home_away_enc", "opp_def_rating", "opp_pts_allowed_last10",
    ]
    print("\n  Feature values at prediction time:")
    for f in sanity_features:
        if f in feature_cols:
            val = float(X_pred[f].iloc[0])
            print(f"    {f:<30}: {val:.3f}")

    point_estimate = float(model.predict(X_pred)[0])

    # Bootstrap CI: sample from TEST set residuals (out-of-sample), not training residuals.
    # Training residuals for tree ensembles are near-zero (memorization) → CI collapses to 0.
    # Test residuals reflect true generalization error and produce a realistic CI width.
    n_test = min(TEST_SPLIT_GAMES, len(df) - 1)
    test_df = df.iloc[-n_test:].copy()
    X_test_ci = test_df.reindex(columns=feature_cols, fill_value=0).fillna(0)
    residuals = test_df[TARGET_COL].values - model.predict(X_test_ci)
    rng = np.random.default_rng(seed=42)
    bootstrap_preds = point_estimate + rng.choice(residuals, size=n_bootstrap, replace=True)
    ci_low = float(np.percentile(bootstrap_preds, 10))
    ci_high = float(np.percentile(bootstrap_preds, 90))

    result = {
        "player": player_name,
        "opponent": opponent,
        "model": model_type,
        "predicted_pts": round(point_estimate, 1),
        "ci_80_low": round(ci_low, 1),
        "ci_80_high": round(ci_high, 1),
        "home_away": home_away,
        "is_playoffs": bool(is_playoffs),
        "limited_sample_warning": limited_sample,
    }

    print(f"\nPrediction: {player_name} vs {opponent}")
    print(f"  Model      : {model_type}")
    print(f"  Predicted  : {result['predicted_pts']} pts")
    print(f"  80% CI     : [{ci_low:.1f}, {ci_high:.1f}]  (residual std={residuals.std():.1f})")
    if limited_sample:
        print(f"  Warning    : Only {n_playoff_games} playoff games — limited sample size")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--opponent", required=True, help="Team abbreviation, e.g. OKC")
    parser.add_argument("--model", default="xgboost", choices=["linear", "rf", "xgboost"])
    parser.add_argument("--home-away", default="home", choices=["home", "away"])
    parser.add_argument("--playoffs", action="store_true")
    parser.add_argument("--season", default=None, help="Current season e.g. 2024-25")
    parser.add_argument("--models-dir", default=MODELS_DIR)
    parser.add_argument("--no-smooth", action="store_true",
                        help="Skip median-of-last-5 smoothing; use raw last-game row values")
    args = parser.parse_args()

    try:
        predict(
            args.player,
            args.opponent,
            args.model,
            args.home_away,
            int(args.playoffs),
            args.season,
            args.models_dir,
            smooth=not args.no_smooth,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"\nColumn error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
