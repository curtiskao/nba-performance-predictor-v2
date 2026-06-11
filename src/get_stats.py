"""Fetch regular season and/or playoff game logs per player/season."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, SEASONS
from src.nba_api_client import NBAApiClient


def fetch_player_logs(
    player_name: str,
    seasons: list[str],
    season_type: str = "Both",
    raw_dir: str = RAW_DIR,
) -> pd.DataFrame:
    client = NBAApiClient()
    player_id = client.find_player_id(player_name)
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    season_types = (
        ["Regular Season", "Playoffs"] if season_type == "Both" else [season_type]
    )

    all_frames = []
    for season in seasons:
        for stype in season_types:
            print(f"  Fetching {player_name} | {season} | {stype}...")
            try:
                df = client.get_game_log(player_id, season, stype)
                if df.empty:
                    continue
                df["SEASON"] = season
                df["SEASON_TYPE"] = stype
                all_frames.append(df)
            except Exception as exc:
                print(f"  Warning: {exc}")

    if not all_frames:
        raise ValueError(f"No game log data found for {player_name!r}")

    combined = pd.concat(all_frames, ignore_index=True)

    safe_name = player_name.lower().replace(" ", "_")
    out_path = raw_path / f"{safe_name}_gamelogs.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined)} rows to {out_path}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Fetch NBA player game logs")
    parser.add_argument("--player", required=True, help="Player full name")
    parser.add_argument(
        "--seasons",
        default=",".join(SEASONS),
        help="Comma-separated seasons, e.g. 2022-23,2023-24",
    )
    parser.add_argument(
        "--season-type",
        default="Both",
        choices=["Regular Season", "Playoffs", "Both"],
    )
    parser.add_argument("--raw-dir", default=RAW_DIR)
    args = parser.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",")]
    try:
        fetch_player_logs(args.player, seasons, args.season_type, args.raw_dir)
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nAPI error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
