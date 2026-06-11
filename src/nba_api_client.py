"""Thin wrapper around nba_api with rate limiting and local disk caching."""

import json
import os
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    BoxScoreAdvancedV2,
    LeagueDashTeamStats,
    LeagueGameFinder,
    PlayerGameLog,
)
from nba_api.stats.static import players, teams

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import API_MAX_RETRIES, API_SLEEP_SECONDS, CACHE_DIR


class NBAApiClient:
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Player / team lookups
    # ------------------------------------------------------------------

    def find_player_id(self, full_name: str) -> int:
        matches = players.find_players_by_full_name(full_name)
        if not matches:
            raise ValueError(f"No player found for: {full_name!r}")
        return matches[0]["id"]

    def find_team_id(self, team_name: str) -> int:
        all_teams = teams.get_teams()
        name_lower = team_name.lower()
        for t in all_teams:
            if (
                name_lower in t["full_name"].lower()
                or name_lower in t["nickname"].lower()
                or name_lower in t["abbreviation"].lower()
            ):
                return t["id"]
        raise ValueError(f"No team found for: {team_name!r}")

    # ------------------------------------------------------------------
    # Core fetch methods (with caching)
    # ------------------------------------------------------------------

    def get_game_log(
        self, player_id: int, season: str, season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        cache_key = f"gamelog_{player_id}_{season}_{season_type.replace(' ', '_')}.json"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return pd.DataFrame(cached)

        df = self._retry(
            lambda: PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=season_type,
                timeout=30,
            ).get_data_frames()[0]
        )
        self._save_cache(cache_key, df.to_dict(orient="records"))
        return df

    def get_league_team_stats(self, season: str) -> pd.DataFrame:
        """Return advanced team stats with TEAM_ABBREVIATION column added from static lookup."""
        cache_key = f"team_stats_{season}.json"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return pd.DataFrame(cached)

        df = self._retry(
            lambda: LeagueDashTeamStats(
                season=season,
                season_type_all_star="Regular Season",
                measure_type_detailed_defense="Advanced",
                timeout=30,
            ).get_data_frames()[0]
        )

        # LeagueDashTeamStats does not return TEAM_ABBREVIATION; join from static table.
        team_abbr = pd.DataFrame(
            [{"TEAM_ID": t["id"], "TEAM_ABBREVIATION": t["abbreviation"]} for t in teams.get_teams()]
        )
        df = df.merge(team_abbr, on="TEAM_ID", how="left")

        self._save_cache(cache_key, df.to_dict(orient="records"))
        return df

    def get_league_game_log(
        self, season: str, season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """All team-game rows for a season (each game appears once per team)."""
        cache_key = f"league_gamelog_{season}_{season_type.replace(' ', '_')}.json"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return pd.DataFrame(cached)

        df = self._retry(
            lambda: LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                player_or_team_abbreviation="T",
                timeout=60,
            ).get_data_frames()[0]
        )
        self._save_cache(cache_key, df.to_dict(orient="records"))
        return df

    def get_box_score_advanced(self, game_id: str) -> pd.DataFrame:
        cache_key = f"boxscore_adv_{game_id}.json"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return pd.DataFrame(cached)

        df = self._retry(
            lambda: BoxScoreAdvancedV2(
                game_id=game_id,
                timeout=30,
            ).get_data_frames()[0]
        )
        self._save_cache(cache_key, df.to_dict(orient="records"))
        return df

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key

    def _load_cache(self, key: str):
        path = self._cache_path(key)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, data):
        path = self._cache_path(key)
        with open(path, "w") as f:
            json.dump(data, f)

    # ------------------------------------------------------------------
    # Retry / rate limiting
    # ------------------------------------------------------------------

    def _retry(self, fn, retries: int = API_MAX_RETRIES):
        last_exc = None
        for attempt in range(retries):
            try:
                time.sleep(API_SLEEP_SECONDS)
                return fn()
            except Exception as exc:
                last_exc = exc
                wait = API_SLEEP_SECONDS * (2 ** attempt)
                print(f"  [retry {attempt + 1}/{retries}] {exc} — waiting {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"API call failed after {retries} attempts: {last_exc}") from last_exc
