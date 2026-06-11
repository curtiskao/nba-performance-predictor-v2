"""Clean, merge, and enrich raw game logs with opponent context."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, RAW_DIR, SEASONS
from src.nba_api_client import NBAApiClient


# Map from normalized (uppercased) column names → output names.
# nba_api is inconsistent with case (e.g. Game_ID vs GAME_ID), so we uppercase
# all columns before applying this map.
GAME_LOG_RENAME = {
    "GAME_DATE": "game_date",
    "MATCHUP": "matchup",
    "WL": "win_loss",
    "MIN": "min",
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "TOV": "TOV",
    "FGA": "FGA",
    "FGM": "FGM",
    "FG_PCT": "FG_PCT",
    "FG3A": "FG3A",
    "FG3M": "FG3M",
    "FG3_PCT": "FG3_PCT",
    "FTA": "FTA",
    "FTM": "FTM",
    "FT_PCT": "FT_PCT",
    "PLUS_MINUS": "plus_minus",
    "GAME_ID": "game_id",
    "SEASON": "season",
    "SEASON_TYPE": "season_type",
}

REQUIRED_COLUMNS = ["game_date", "matchup", "season", "season_type", "game_id", "PTS"]

_SEP = "─" * 60


def _section(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _validate_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        available = sorted(df.columns.tolist())
        raise KeyError(
            f"[{context}] Missing required columns: {missing}\n"
            f"  Available columns: {available}"
        )


def load_raw_logs(player_name: str, raw_dir: str = RAW_DIR) -> pd.DataFrame:
    safe_name = player_name.lower().replace(" ", "_")
    path = Path(raw_dir) / f"{safe_name}_gamelogs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw logs not found: {path}\n"
            f"  Run: python src/get_stats.py --player \"{player_name}\""
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Raw log file is empty: {path}")
    return df


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Uppercase all column names first — nba_api is inconsistent (e.g. Game_ID vs GAME_ID).
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]

    df = df.rename(columns={k: v for k, v in GAME_LOG_RENAME.items() if k in df.columns})

    _validate_columns(df, REQUIRED_COLUMNS, "normalize_schema")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    bad_dates = df["game_date"].isna().sum()
    if bad_dates > 0:
        print(f"  Warning: {bad_dates} rows had unparseable game_date and will be dropped.")
        df = df.dropna(subset=["game_date"])

    df = df.sort_values("game_date").reset_index(drop=True)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["game_id", "season_type"]).reset_index(drop=True)
    if before_dedup != len(df):
        print(f"  Deduplication: {before_dedup} → {len(df)} rows ({before_dedup - len(df)} dropped)")

    df["home_away"] = df["matchup"].apply(
        lambda m: "home" if "vs." in str(m) else "away"
    )
    df["opp_abbr"] = df["matchup"].apply(
        lambda m: str(m).split(" vs. ")[-1].split(" @ ")[-1].strip()
    )
    df["is_playoffs"] = (df["season_type"] == "Playoffs").astype(int)

    # ── Normalization summary ──────────────────────────────────────
    print(f"  Date range  : {df['game_date'].min().date()}  →  {df['game_date'].max().date()}")
    by_type = df.groupby(["season", "season_type"]).size().reset_index(name="games")
    for _, row in by_type.iterrows():
        print(f"  {row['season']}  {row['season_type']:<16}: {row['games']} games")
    home = (df["home_away"] == "home").sum()
    print(f"  Home/Away   : {home} home, {len(df) - home} away")
    pts = df["PTS"]
    print(f"  PTS         : mean={pts.mean():.1f}  std={pts.std():.1f}  "
          f"min={pts.min():.0f}  max={pts.max():.0f}  median={pts.median():.1f}")

    return df


def join_opponent_context(df: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    client = NBAApiClient()
    team_stats_frames = []
    for season in seasons:
        try:
            ts = client.get_league_team_stats(season)
            ts["season"] = season
            team_stats_frames.append(ts)
        except Exception as exc:
            print(f"  Warning: could not fetch team stats for {season}: {exc}")

    if not team_stats_frames:
        print("  No opponent context available — skipping join.")
        return df

    team_stats = pd.concat(team_stats_frames, ignore_index=True)

    if "TEAM_ABBREVIATION" not in team_stats.columns:
        print("  Warning: TEAM_ABBREVIATION missing from team stats — skipping join.")
        return df

    keep_cols = ["TEAM_ABBREVIATION", "season"]
    for col in ["E_DEF_RATING", "DEF_RATING", "PACE", "OPP_PTS"]:
        if col in team_stats.columns:
            keep_cols.append(col)

    team_stats = team_stats[keep_cols].copy()
    team_stats = team_stats.rename(
        columns={
            "TEAM_ABBREVIATION": "opp_abbr",
            "E_DEF_RATING": "opp_def_rating",
            "DEF_RATING": "opp_def_rating_base",
            "PACE": "opp_pace",
            "OPP_PTS": "opp_pts_allowed_avg",
        }
    )

    before = len(df)
    df = df.merge(team_stats, on=["opp_abbr", "season"], how="left")

    added_cols = [c for c in ["opp_def_rating", "opp_def_rating_base", "opp_pace", "opp_pts_allowed_avg"]
                  if c in df.columns]
    print(f"  Opponent context columns added: {added_cols}")
    for col in added_cols:
        n_null = df[col].isna().sum()
        coverage = f"{len(df) - n_null}/{len(df)}"
        val_range = f"{df[col].min():.1f} – {df[col].max():.1f}" if df[col].notna().any() else "all NaN"
        print(f"    {col:<28}: {coverage} non-null, range [{val_range}]")

    return df


def add_rolling_opponent_defense(df: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Add opp_pts_allowed_last10: opponent's rolling 10-game avg pts allowed before each game."""
    client = NBAApiClient()
    all_games = []
    for season in seasons:
        for stype in ["Regular Season", "Playoffs"]:
            try:
                games = client.get_league_game_log(season, stype)
                if not games.empty:
                    all_games.append(games)
            except Exception as exc:
                print(f"  Warning: league game log fetch failed ({season} {stype}): {exc}")

    if not all_games:
        print("  No league game log available — skipping rolling opponent defense.")
        return df

    games = pd.concat(all_games, ignore_index=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    # Each game appears twice (once per team). Join on GAME_ID to get the other team's PTS,
    # which is the points allowed by each team in that game.
    scored = games[["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "PTS"]].copy()
    conceded = games[["TEAM_ABBREVIATION", "GAME_ID", "PTS"]].rename(
        columns={"TEAM_ABBREVIATION": "OPP_TEAM", "PTS": "pts_allowed"}
    )
    merged = scored.merge(conceded, on="GAME_ID")
    merged = merged[merged["TEAM_ABBREVIATION"] != merged["OPP_TEAM"]]

    # Rolling 10-game avg pts allowed per team; shift(1) so current game is not included.
    merged = merged.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    merged["opp_pts_allowed_last10"] = (
        merged.groupby("TEAM_ABBREVIATION")["pts_allowed"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

    lookup = (
        merged[["TEAM_ABBREVIATION", "GAME_DATE", "opp_pts_allowed_last10"]]
        .rename(columns={"TEAM_ABBREVIATION": "opp_abbr", "GAME_DATE": "game_date"})
        .drop_duplicates(subset=["opp_abbr", "game_date"])
    )

    df["game_date"] = pd.to_datetime(df["game_date"])
    before_cols = set(df.columns)
    df = df.merge(lookup, on=["opp_abbr", "game_date"], how="left")

    matched = df["opp_pts_allowed_last10"].notna().sum()
    total = len(df)
    print(f"  Rolling opponent defense: {matched}/{total} rows matched "
          f"({total - matched} NaN — expected for first ~10 games of each opponent).")
    return df


def process_player(
    player_name: str,
    seasons: list[str] = None,
    raw_dir: str = RAW_DIR,
    processed_dir: str = PROCESSED_DIR,
) -> pd.DataFrame:
    if seasons is None:
        seasons = SEASONS

    _section(f"process_data  ·  {player_name}")
    df = load_raw_logs(player_name, raw_dir)
    print(f"  Raw rows    : {len(df)}")
    print(f"  Raw columns : {sorted(df.columns.tolist())}")

    _section("Normalizing schema")
    df = normalize_schema(df)

    _section("Joining opponent context (season averages)")
    df = join_opponent_context(df, seasons)

    _section("Adding rolling opponent defense (last 10 games)")
    df = add_rolling_opponent_defense(df, seasons)

    _section("Processed file summary")
    nan_summary = df.isnull().sum()
    nan_cols = nan_summary[nan_summary > 0].sort_values(ascending=False)
    if len(nan_cols):
        print("  NaN counts (non-zero columns):")
        for col, n in nan_cols.items():
            print(f"    {col:<35}: {n:>3} NaN  ({100*n/len(df):.0f}%)")
    else:
        print("  No NaN values in processed data.")
    print(f"\n  Total columns : {len(df.columns)}")
    print(f"  Total rows    : {len(df)}")

    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    safe_name = player_name.lower().replace(" ", "_")
    out_path = Path(processed_dir) / f"{safe_name}_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Process raw NBA game logs")
    parser.add_argument("--player", required=True)
    parser.add_argument("--seasons", default=",".join(SEASONS))
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--processed-dir", default=PROCESSED_DIR)
    args = parser.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",")]
    try:
        process_player(args.player, seasons, args.raw_dir, args.processed_dir)
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
