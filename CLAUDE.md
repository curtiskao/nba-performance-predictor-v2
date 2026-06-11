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
│   ├── train.py               # Train Ridge, Random Forest, XGBoost; save models
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
- Supports `--player`, `--seasons` (e.g. `2023-24,2024-25,2025-26`), `--season-type` (`Regular Season`, `Playoffs`, or `Both`)
- Writes raw game logs to `data/raw/`
- Resolves player name → player ID via `nba_api` `find_players_by_full_name`

### `process_data.py`
- Loads raw game logs, normalizes schema, deduplicates
- Merges regular season + playoff rows with a `season_type` flag
- Joins opponent defensive rating, pace, and opponent points-allowed-per-game (season avg + rolling last-10)
- Outputs cleaned DataFrame to `data/processed/`

### `feature_engineering.py`
Feature groups (all toggleable via `config.py`):
- **Rolling performance**: pts/reb/ast/fg%/fga/min rolling mean + std over last 3, 5, 10 games
- **Rest & fatigue**: days_rest, is_back_to_back, games_played_last_7_days
- **Context flags**: home_away, is_playoffs, playoff_round (1–4), game_number_in_series
- **Matchup**: opponent_def_rating, opponent_pts_allowed_avg (season + rolling last-10), opponent_pace
- **Usage**: usage_rate, true_shooting_pct, fga_per_36
- **Trend**: pts_trend_slope (linear regression over last 5 games)
- **EWM**: exponentially weighted means over spans 3, 5, 10 (pts_ewm_3/5/10, pts_last_5_ewm)

### `feature_selection.py`
- Pearson correlation filter (drop features with |corr| < 0.01 vs target — intentionally loose to avoid dropping weak-but-informative features on small samples)
- Random Forest feature importance ranking
- Outputs a ranked feature report

### `benchmark.py`
Naive baselines to compare models against:
- Season average points
- Last-5-game average
- Last-10-game average
- Opponent-adjusted season average
- Reports MAE, RMSE, R² for each

### `train.py`
- Trains three models: **Ridge Regression** (alpha=500), **Random Forest**, **XGBoost**
- Linear Regression replaced with Ridge (sklearn Pipeline + StandardScaler) due to underdetermination at high feature count / low sample count
- Time-series aware train/test split (no data leakage — test set is always chronologically after train)
- Sample weights: exponential recency decay (decay_rate=0.97) × playoff multiplier (3.0x) per row
- Separate evaluation on playoff-only test rows
- Saves models to `models/`

### `predict.py`
- CLI: `python predict.py --player "Victor Wembanyama" --opponent NYK --model rf --home-away home --playoffs`
- Flags: `--playoffs` (sets is_playoffs=1), `--no-smooth` (skips median smoothing, uses raw last-game row)
- Default smoothing: for rolling mean and EWM columns, replaces last-game value with median of last 5 rows to reduce single-game noise
- Outputs predicted points + 80% CI bootstrapped from out-of-sample residuals (1000 iterations, 10th/90th percentile)

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
SEASONS = ["2025-26"]                 # Use current season only for young/improving players
DEFAULT_SEASON_TYPE = "Both"          # "Regular Season" | "Playoffs" | "Both"
ROLLING_WINDOWS = [3, 5, 10]
TARGET_COL = "PTS"
TEST_SPLIT_GAMES = 10                 # Last N games held out as test set
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
    "n_estimators": 200,
    "max_depth": 3,            # Shallow trees prevent overfitting on small samples
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,     # Prevents splits on small subsets
}
RIDGE_ALPHA = 500.0            # Strong regularization needed: ~48 features, 75-192 train rows
DECAY_RATE = 0.97              # Recency decay per game (older games weighted less)
PLAYOFF_MULTIPLIER = 3.0       # Playoff rows get 3x weight in training
```

---

## Development Conventions

- **Python 3.11+**
- **Dependencies**: `nba_api`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `joblib`
- All raw data writes are append-safe and idempotent (check before re-fetching)
- No data leakage: features must only use information available *before* the game being predicted
- Playoff and regular season rows always tagged — models can be trained on combined or playoff-only splits
- All scripts runnable standalone via CLI with `argparse`
- Always pass `--playoffs` flag when predicting postseason games
- Use single-season data for young/rapidly-improving players; prior seasons dilute current form

---

## Model Selection: Per-Player Architecture

Each player gets their own independently trained model. Files are named per player:
- `data/processed/victor_wembanyama_processed.csv`
- `models/victor_wembanyama_rf.pkl`
- `models/victor_wembanyama_xgboost.pkl`

**Rationale**: Cross-player generalization is counterproductive — a Wembanyama game log and a role player game log have different statistical distributions. Per-player models learn player-specific patterns (usage, efficiency, context response) without noise from unrelated players.

**Tradeoff**: Small sample sizes per player (~75–200 rows) limit model complexity. XGBoost tends to overfit; Random Forest is more robust at this scale.

---

## Evaluation Protocol

1. **Baseline check**: every model must beat the last-5-game average baseline on MAE
2. **Train/test split**: chronological — never shuffle time-series data
3. **Playoff-specific eval**: report metrics separately on playoff-only test rows
4. **Target metric**: MAE (primary), RMSE (secondary), R² (tertiary)
5. **v1 benchmark**: MAE 5.0 — v2 target is MAE ≤ 4.5 on regular season, ≤ 5.5 on playoffs
6. **Prediction sanity check**: verify `is_playoffs`, `home_away_enc`, and opponent fields are set correctly before interpreting output

---

## Known Findings (Wembanyama 2025-26)

- **Best model**: Random Forest, MAE=3.70 on 10-game playoff test set
- **Dominant features**: `FG_PCT_roll_mean_10` (0.196), `true_shooting_pct` (0.161) — efficiency metrics drive RF more than raw scoring history
- **Prediction vs. sportsbook**: RF predicts ~24 pts; sportsbook lines 27–29.5. Gap explained by sportsbook pricing hot-streak continuation; model correctly reflects playoff average (~24.8 pts)
- **Multi-season data**: Adding 2023-24 and 2024-25 seasons hurt performance (MAE 3.70 → 4.25) because rookie-year lower scoring pulled the model's baseline down. For young improving players, use current season only.
- **Correlation filter threshold**: 0.01 (not 0.05) — aggressive filtering on small samples drops informative features like `pts_last_5_ewm` and `days_rest`
- **Linear Regression**: Must use Ridge (alpha=500) — plain LR explodes with 47+ features and 75 training rows (R²=-75 observed)
- **XGBoost on small samples**: Tends to overfit and underpredict; RF preferred. XGBoost max_depth=3, min_child_weight=3 partially mitigates this.
- **Smoothing**: Median-of-last-5 smoothing for rolling/EWM features stabilizes predictions but dampens hot-streak signal (~2–3 pts). Use `--no-smooth` to compare; effect on RF is minimal since RF doesn't rely heavily on PTS rolling means anyway.

---

## Current Status

- [x] Scaffolding and config
- [x] `nba_api_client.py` with caching
- [x] `get_stats.py` regular season + playoffs, multi-season
- [x] `process_data.py` cleaning + opponent join (season avg + rolling last-10)
- [x] `feature_engineering.py` (rolling, EWM, rest, context, matchup, usage, trend)
- [x] `feature_selection.py` (correlation filter)
- [x] `train.py` (Ridge, Random Forest, XGBoost + sample weights + playoff eval)
- [x] `predict.py` CLI (--playoffs, --no-smooth, 80% CI via residual bootstrap)
- [ ] `benchmark.py` — run and validate against naive baselines
- [ ] SHAP analysis notebook
- [ ] README
