# nba-performance-predictor-v2

## Project Overview

Predicts NBA player scoring performance for an upcoming game — with a primary focus on playoff/Finals contexts. Built on v1 (XGBoost, 25+ features, MAE ~5.0), v2 introduces a cleaner architecture, richer feature engineering, regular season + playoff data support, and a pluggable model pipeline.

## Goal

Given a player and their upcoming opponent, predict their points scored in the next game — targeting playoff/Finals use cases where sample sizes are small and context (matchup, fatigue, home/away) matters most.

---

## Repository Structure

```
nba-performance-predictor-v2/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── config.py                  # Seasons, stat categories, feature toggles, model hyperparams
│
├── data/
│   ├── raw/                   # Raw API output (JSON/CSV), never modified after write
│   └── processed/             # Processed/feature-engineered DataFrames ready for training
│
├── src/
│   ├── nba_api_client.py      # Thin wrapper around nba_api with rate limiting + caching
│   ├── get_stats.py           # Fetch regular season and/or playoff game logs per player/season
│   ├── process_data.py        # Cleaning, merging, rolling features, opponent context
│   ├── feature_engineering.py # Named feature groups: rolling stats, rest days, matchup, usage
│   ├── feature_selection.py   # Correlation filter, feature importance ranking, SHAP analysis
│   ├── train.py               # Train linear regression, Random Forest, XGBoost; save models
│   ├── benchmark.py           # Naive baselines: season avg, last-N avg, median; compute MAE/RMSE/R²
│   └── predict.py             # Load model + predict for a given player + upcoming opponent
│
├── models/                    # Serialized models (.pkl / .json)
├── notebooks/                 # Exploratory analysis, feature correlation plots
└── tests/                     # Unit tests for data processing and feature logic
```

---

## Key Modules

### `nba_api_client.py`
- Wraps `nba_api` endpoints: `PlayerGameLog`, `LeagueGameFinder`, `BoxScoreTraditionalV2`, `TeamDashboardByOpponent`
- Handles rate limiting (sleep between requests), retries, and local disk caching (JSON)
- Exposes clean fetch methods: `get_game_log(player_id, season, season_type)`, `get_team_defensive_stats(team_id, season)`

### `get_stats.py`
- Entry point for data collection
- Supports `--player`, `--seasons` (e.g. `2021-22,2022-23,2023-24`), `--season-type` (`Regular Season`, `Playoffs`, or `Both`)
- Writes raw game logs to `data/raw/`
- Resolves player name → player ID via `nba_api` `find_players_by_full_name`

### `process_data.py`
- Loads raw game logs, normalizes schema, deduplicates
- Merges regular season + playoff rows with a `season_type` flag
- Joins opponent defensive rating, pace, and opponent points-allowed-per-game
- Outputs cleaned DataFrame to `data/processed/`

### `feature_engineering.py`
Feature groups (all toggleable via `config.py`):
- **Rolling performance**: pts/reb/ast/fg%/min rolling mean + std over last 3, 5, 10 games
- **Rest & fatigue**: days_rest, is_back_to_back, games_played_last_7_days
- **Context flags**: home_away, is_playoffs, playoff_round (1–4), game_number_in_series
- **Matchup**: opponent_def_rating, opponent_pts_allowed_avg, opponent_pace
- **Usage**: usage_rate, true_shooting_pct, fga_per_36
- **Trend**: pts_trend_slope (linear regression over last 5 games)
- **Recency weighting**: exponential decay weights on rolling windows

### `feature_selection.py`
- Pearson correlation filter (drop features with |corr| < threshold vs target)
- Random Forest feature importance ranking
- SHAP value analysis on XGBoost model
- Outputs a ranked feature report

### `benchmark.py`
Naive baselines to compare models against:
- Season average points
- Last-5-game average
- Last-10-game average
- Opponent-adjusted season average
- Reports MAE, RMSE, R² for each

### `train.py`
- Trains three models: **Linear Regression**, **Random Forest**, **XGBoost**
- Time-series aware train/test split (no data leakage — test set is always chronologically after train)
- Optionally separate evaluation on playoff-only test rows
- Hyperparameter tuning via `GridSearchCV` or `Optuna` for XGBoost
- Saves models to `models/`, logs metrics to console + optional CSV

### `predict.py`
- CLI: `python predict.py --player "Jayson Tatum" --opponent "OKC Thunder" --model xgboost`
- Constructs feature vector for the upcoming game (last known stats + opponent context)
- Outputs predicted points + confidence interval (bootstrapped or quantile regression)
- Flags if prediction is based on limited playoff sample (< 10 games)

---

## Data Sources

| Source | Library / API | Notes |
|---|---|---|
| Player game logs | `nba_api` (`PlayerGameLog`) | Regular season + playoffs |
| Team defensive stats | `nba_api` (`TeamDashboardByOpponent`) | Opponent context |
| Advanced box scores | `nba_api` (`BoxScoreAdvancedV2`) | Usage rate, TS% |
| Pace/ratings | `nba_api` (`LeagueDashTeamStats`) | Team pace, off/def rating |

---

## Configuration (`config.py`)

```python
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
DEFAULT_SEASON_TYPE = "Both"          # "Regular Season" | "Playoffs" | "Both"
ROLLING_WINDOWS = [3, 5, 10]
TARGET_COL = "PTS"
TEST_SPLIT_GAMES = 20                 # Last N games held out as test set
CACHE_DIR = "data/raw/cache"
MODELS_DIR = "models"
FEATURE_GROUPS_ENABLED = {
    "rolling": True,
    "rest": True,
    "context": True,
    "matchup": True,
    "usage": True,
    "trend": True,
}
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
```

---

## Development Conventions

- **Python 3.11+**
- **Dependencies**: `nba_api`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `joblib`
- All raw data writes are append-safe and idempotent (check before re-fetching)
- No data leakage: features must only use information available *before* the game being predicted
- Playoff and regular season rows always tagged — models can be trained on combined or playoff-only splits
- All scripts runnable standalone via CLI with `argparse`

---

## Evaluation Protocol

1. **Baseline check**: every model must beat the last-5-game average baseline on MAE
2. **Train/test split**: chronological — never shuffle time-series data
3. **Playoff-specific eval**: report metrics separately on playoff-only test rows
4. **Target metric**: MAE (primary), RMSE (secondary), R² (tertiary)
5. **v1 benchmark**: MAE 5.0 — v2 target is MAE ≤ 4.5 on regular season, ≤ 5.5 on playoffs (smaller samples)

---

## Current Status

- [ ] Scaffolding and config
- [ ] `nba_api_client.py` with caching
- [ ] `get_stats.py` regular season + playoffs
- [ ] `process_data.py` cleaning + opponent join
- [ ] `feature_engineering.py`
- [ ] `feature_selection.py`
- [ ] `benchmark.py`
- [ ] `train.py` (Linear Regression, Random Forest, XGBoost)
- [ ] `predict.py` CLI
- [ ] SHAP analysis notebook
- [ ] README
