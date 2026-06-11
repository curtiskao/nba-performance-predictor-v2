SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
DEFAULT_SEASON_TYPE = "Both"  # "Regular Season" | "Playoffs" | "Both"
ROLLING_WINDOWS = [3, 5, 10]
TARGET_COL = "PTS"
TEST_SPLIT_GAMES = 20
CACHE_DIR = "data/raw/cache"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
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
    "max_depth": 3,          # shallower = forced to use strongest features (PTS rolling means)
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,   # prevents splits on tiny subsets; helps with small playoff samples
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 2,
}

# Correlation filter threshold (|corr| with target must exceed this)
CORRELATION_THRESHOLD = 0.01

# Sample weighting in training
RECENCY_DECAY_RATE = 0.97       # per-game exponential decay; game 40 ago ≈ 0.30 weight
PLAYOFF_WEIGHT_MULTIPLIER = 3.0  # playoff rows count this many times more in loss

# Rate limiting
API_SLEEP_SECONDS = 0.6
API_MAX_RETRIES = 3
