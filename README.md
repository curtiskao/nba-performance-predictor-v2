# NBA Performance Predictor v2

Predicts NBA player point totals for an upcoming game. Built for playoff and Finals contexts where sample sizes are small and matchup/fatigue signals matter most. Models are trained per-player on regular season and playoff game logs fetched from the official NBA stats API.

---

## How it works

The pipeline has five stages that run in order:

```
get_stats.py  →  process_data.py  →  train.py  →  predict.py
                                  ↓
                            benchmark.py  (optional, runs independently)
```

1. **`get_stats.py`** — Fetches game-by-game box scores for a player from `nba_api` and writes a raw CSV to `data/raw/`.
2. **`process_data.py`** — Cleans the raw logs, joins opponent defensive stats (def rating, pace, pts allowed), and adds a rolling 10-game opponent defense column. Writes to `data/processed/`.
3. **`train.py`** — Engineers features, runs a correlation filter, trains Ridge regression / Random Forest / XGBoost, and saves `.pkl` model bundles to `models/`.
4. **`predict.py`** — Loads a saved model and constructs a feature row for the upcoming game, then outputs a point prediction with an 80% confidence interval.
5. **`benchmark.py`** — Evaluates naive baselines (season average, last-5, last-10, median, opponent-adjusted) on the same test split. Run this to see whether the models actually beat simple heuristics.

Each model file is saved as `models/{player_name}_{model_type}.pkl`. The bundle stores both the fitted model and the exact feature column list used during training, so predict.py always uses a consistent feature set.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.11+.

---

## Full walkthrough — Victor Wembanyama example

### Step 1 — Fetch raw game logs

```bash
python src/get_stats.py --player "Victor Wembanyama" --seasons 2023-24,2024-25 --season-type Both
```

Output: `data/raw/victor_wembanyama_gamelogs.csv`

API responses are cached to `data/raw/cache/` as JSON so reruns skip the network call. Clear the cache if you need fresh data.

### Step 2 — Process data

```bash
python src/process_data.py --player "Victor Wembanyama"
```

Output: `data/processed/victor_wembanyama_processed.csv`

This step prints a detailed diagnostic: date range, game counts by season/type, home/away split, PTS distribution, NaN counts, and opponent context coverage.

### Step 3 — Train models

```bash
python src/train.py --player "Victor Wembanyama"
```

Output: `models/victor_wembanyama_{linear,rf,xgboost}.pkl`

Training diagnostics include: dataset split overview, full correlation filter table, sample weight range, NaN audit, feature importances (top 15 for XGBoost and Random Forest), prediction distribution on the test set, and a playoff-only eval if playoff games are in the test window.

### Step 4 — Predict

```bash
python src/predict.py --player "Victor Wembanyama" --opponent OKC --model rf --season 2024-25
```

Output example:
```
  Feature values at prediction time:
    PTS_roll_mean_3               : 26.300
    PTS_roll_mean_5               : 25.800
    PTS_roll_mean_10              : 24.900
    pts_last_5_ewm                : 25.600
    FGA_roll_mean_3               : 17.100
    FGA_roll_mean_10              : 16.400
    FG_PCT_roll_mean_10           : 0.498
    true_shooting_pct             : 0.621
    is_playoffs                   : 0.000
    home_away_enc                 : 1.000

Prediction: Victor Wembanyama vs OKC
  Model      : rf
  Predicted  : 26.4 pts
  80% CI     : [21.2, 31.8]  (residual std=4.8)
```

### Step 5 — Check baselines (optional)

```bash
python src/benchmark.py --player "Victor Wembanyama"
```

If your model's MAE doesn't beat the last-5 average, the model isn't adding value.

---

## Flag reference

### `get_stats.py`

| Flag | Default | Description |
|---|---|---|
| `--player` | required | Player's full name as it appears on NBA.com, e.g. `"Victor Wembanyama"` |
| `--seasons` | `2021-22,2022-23,2023-24,2024-25` | Comma-separated list of seasons to fetch. Format: `YYYY-YY`. Add more seasons for a larger training set; remove older ones if the player's role changed significantly. |
| `--season-type` | `Both` | Which games to include: `Regular Season`, `Playoffs`, or `Both`. Use `Both` for maximum data; `Playoffs` if you only want postseason logs. |
| `--raw-dir` | `data/raw` | Where to write the raw CSV. Only change this if you're running multiple experiments in parallel with separate data directories. |

### `process_data.py`

| Flag | Default | Description |
|---|---|---|
| `--player` | required | Must match the name used in `get_stats.py` exactly — it's used to construct the filename. |
| `--seasons` | same as config | Controls which seasons' team defensive stats are fetched for the opponent join. Should match or be a subset of what was fetched in Step 1. |
| `--raw-dir` | `data/raw` | Where to read the raw CSV from. |
| `--processed-dir` | `data/processed` | Where to write the processed CSV. |

### `train.py`

| Flag | Default | Description |
|---|---|---|
| `--player` | required | Player name. Used to load `data/processed/{name}_processed.csv` and save to `models/{name}_*.pkl`. |
| `--test-games` | `20` | Number of most recent games held out as the test set. The rest become the training set. The split is always chronological — no shuffling. Increase this for a larger evaluation window; decrease it if you have limited total data (e.g. a rookie with 60 games). |
| `--tune-xgb` | off | When passed, runs 40 Optuna trials to find better XGBoost hyperparameters before the final fit. The tuning loop carves a validation set from the end of training data — it never touches the test set. Adds ~2–5 minutes. Useful for squeezing out extra performance once the rest of the pipeline is stable. |
| `--models-dir` | `models` | Where to save the `.pkl` files. |

### `predict.py`

| Flag | Default | Description |
|---|---|---|
| `--player` | required | Player name. Must match what was used in `train.py`. |
| `--opponent` | required | Team abbreviation for the upcoming opponent, e.g. `OKC`, `BOS`, `GSW`. Used to look up the opponent's defensive rating and rolling points-allowed when `--season` is also provided. |
| `--model` | `xgboost` | Which saved model to load: `linear` (Ridge), `rf` (Random Forest), or `xgboost`. `rf` tends to give the most stable point estimates; `xgboost` can be slightly more accurate but with higher variance. |
| `--home-away` | `home` | Whether the upcoming game is `home` or `away`. Home games carry a small positive encoding that the model can pick up if the player has a meaningful home/away split in the training data. |
| `--playoffs` | off | Pass this flag when predicting a playoff game. This sets `is_playoffs=1` in the feature row and also triggers a limited-sample warning if the player has fewer than 10 career playoff games in the training data. |
| `--season` | `None` | Current season string, e.g. `2024-25`. When provided, triggers a live API fetch for the opponent's current defensive rating, pace, and rolling last-10 points allowed. Without this, those features fall back to whatever values were in the player's most recent game row. |
| `--no-smooth` | off | By default, rolling-mean and EWM feature values in the prediction row are replaced with the **median of the player's last 5 games** to avoid anchoring to a single outlier game. Pass `--no-smooth` to use the raw last-game values instead. Compare both outputs to gauge how much noise was in the most recent game. |
| `--models-dir` | `models` | Where to read the `.pkl` files from. |

### `benchmark.py`

| Flag | Default | Description |
|---|---|---|
| `--player` | required | Player name. Loads the processed + feature-engineered data for this player. |
| `--test-games` | `20` | Same meaning as in `train.py`. Keep this consistent with your training run so the baselines and models are evaluated on the same test window. |

### `feature_selection.py` (standalone)

```bash
python src/feature_selection.py --player "Victor Wembanyama" --method rf_importance
```

| Flag | Default | Description |
|---|---|---|
| `--player` | required | Player name. |
| `--method` | `rf_importance` | `correlation` runs the Pearson filter only. `rf_importance` runs the correlation filter then ranks surviving features by Random Forest importance. `shap` requires a pre-fitted model and ranks by SHAP values. |

---

## Features the model uses

All features are computed from data available **before** the game being predicted. Raw box score stats from the current game (FGA, FGM, PTS, etc.) are never used as features.

| Feature group | Examples | What it captures |
|---|---|---|
| Rolling scoring | `PTS_roll_mean_3/5/10`, `PTS_roll_std_3` | Recent form and consistency |
| Rolling volume | `FGA_roll_mean_3/5/10`, `FGA_roll_std_3` | Shot creation / usage trend |
| Rolling efficiency | `FG_PCT_roll_mean_3/5/10` | Shooting form |
| EWM scoring | `pts_last_5_ewm` | Exponentially weighted recent form (more weight on recent games) |
| Recency-weighted | `PTS_ewm_3/5/10` | Custom exponential decay rolling window |
| Rest & fatigue | `days_rest`, `is_back_to_back`, `games_last_7` | Physical load |
| Context | `home_away_enc`, `is_playoffs`, `playoff_round` | Situational factors |
| Usage | `true_shooting_pct`, `fga_per_36`, `usage_rate` | Efficiency and role |
| Trend | `pts_trend_slope` | Whether the player is heating up or cooling down over the last 5 games |
| Opponent | `opp_def_rating`, `opp_pace`, `opp_pts_allowed_last10` | Defensive difficulty and pace context |

---

## Models

Three models are trained and saved per player. All use the same feature set selected by the correlation filter on the training split only.

| Model | Notes |
|---|---|
| **Ridge** (`linear`) | `StandardScaler` + `Ridge(alpha=500)`. High alpha provides heavy regularization needed when feature count (~50) approaches sample count (~65 training games). A functioning linear baseline — coefficients remain stable and interpretable. |
| **Random Forest** (`rf`) | 200 trees, max depth 6, min 2 samples per leaf. Tends to give the most reliable point estimates. Lower variance than XGBoost on small samples. |
| **XGBoost** (`xgboost`) | 200 trees, max depth 3. Shallower trees force the model to split on the strongest features (PTS rolling means, FGA volume) rather than finding deep noise splits. `min_child_weight=3` prevents overfitting to small playoff subsets. |

### Sample weighting

Training rows are weighted by two factors multiplied together:
- **Recency decay** (`RECENCY_DECAY_RATE = 0.97`): older games receive exponentially less weight. A game from 40 games ago has roughly 0.30× the weight of the most recent game.
- **Playoff multiplier** (`PLAYOFF_WEIGHT_MULTIPLIER = 3.0`): playoff rows count 3× more in the loss function, compensating for their small share of the overall dataset.

### Confidence interval

The 80% CI is computed by bootstrapping from **test set residuals** (not training residuals). Tree models memorize training data so training residuals are near zero; test residuals reflect real out-of-sample error. 1000 bootstrap samples of `point_estimate + residual` give stable p10/p90 percentiles.

---

## Configuration (`config.py`)

Key values you might want to tune:

```python
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]  # seasons to fetch by default
TEST_SPLIT_GAMES = 20          # games held out for evaluation
ROLLING_WINDOWS = [3, 5, 10]  # window sizes for rolling features

CORRELATION_THRESHOLD = 0.01  # minimum |Pearson r| with PTS to keep a feature
RECENCY_DECAY_RATE = 0.97     # per-game weight decay
PLAYOFF_WEIGHT_MULTIPLIER = 3.0

XGBOOST_PARAMS = {
    "max_depth": 3,
    "n_estimators": 200,
    "min_child_weight": 3,
    ...
}
```

Turning off feature groups (e.g. `"matchup": False` in `FEATURE_GROUPS_ENABLED`) removes that entire group from feature engineering before any correlation filtering happens.

---

## Project structure

```
nba-performance-predictor-v2/
├── config.py                  # All tunable constants
├── requirements.txt
│
├── src/
│   ├── nba_api_client.py      # API wrapper with rate limiting and JSON caching
│   ├── get_stats.py           # Step 1: fetch raw game logs
│   ├── process_data.py        # Step 2: clean, join opponent context
│   ├── feature_engineering.py # Step 3a: build feature matrix (called by train.py)
│   ├── feature_selection.py   # Step 3b: correlation filter, RF importance, SHAP
│   ├── train.py               # Step 3: train and save models
│   ├── predict.py             # Step 4: load model and predict
│   └── benchmark.py           # Naive baselines for comparison
│
├── data/
│   ├── raw/                   # Raw game log CSVs (never modified after write)
│   │   └── cache/             # JSON cache of API responses
│   └── processed/             # Feature-engineered DataFrames
│
├── models/                    # Saved model bundles (.pkl)
├── notebooks/                 # Exploratory analysis
└── tests/                     # Unit tests
```

---

## Running tests

```bash
python -m pytest tests/
```

Tests cover home/away parsing, opponent abbreviation extraction, sort order, is_playoffs flag assignment, deduplication, and the critical no-leakage check (verifies that rolling features use `shift(1)` and do not include the current game's stats).
