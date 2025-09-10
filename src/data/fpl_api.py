"""
FPL API Data Collector Module
Handles fetching data from the official Fantasy Premier League API
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
from ratelimit import limits, sleep_and_retry

logger = logging.getLogger(__name__)


class FPLAPICollector:
    """Collector for Fantasy Premier League API data"""

    BASE_URL = "https://fantasy.premierleague.com/api"

    # API Endpoints
    ENDPOINTS = {
        "bootstrap": "/bootstrap-static/",
        "fixtures": "/fixtures/",
        "player": "/element-summary/{player_id}/",
        "live": "/event/{event_id}/live/",
        "history": "/entry/{team_id}/history/",
        "picks": "/entry/{team_id}/event/{event_id}/picks/",
    }

    def __init__(self, cache_dir: str = "cache/fpl"):
        """Initialize FPL API collector

        Args:
            cache_dir: Directory for caching API responses
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
        self._bootstrap_data = None
        self._last_fetch_time = {}

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    @sleep_and_retry
    @limits(calls=60, period=60)  # 60 requests per minute
    async def _fetch(self, url: str) -> dict:
        """Fetch data from URL with rate limiting

        Args:
            url: URL to fetch

        Returns:
            JSON response as dictionary
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                logger.info(f"Successfully fetched data from {url}")
                return data
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching {url}: {e}")
            raise

    def _get_cache_path(self, endpoint: str, params: str = "") -> Path:
        """Get cache file path for endpoint

        Args:
            endpoint: API endpoint name
            params: Additional parameters for cache key

        Returns:
            Path to cache file
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = (
            f"{endpoint}_{params}_{timestamp}.json" if params else f"{endpoint}_{timestamp}.json"
        )
        return self.cache_dir / filename

    def _load_from_cache(self, cache_path: Path, max_age_hours: int = 4) -> Optional[dict]:
        """Load data from cache if fresh enough

        Args:
            cache_path: Path to cache file
            max_age_hours: Maximum cache age in hours

        Returns:
            Cached data or None if stale/missing
        """
        if not cache_path.exists():
            return None

        # Check cache age
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(hours=max_age_hours):
            logger.info(f"Cache {cache_path} is stale (age: {cache_age})")
            return None

        try:
            with open(cache_path) as f:
                logger.info(f"Loading from cache: {cache_path}")
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading cache {cache_path}: {e}")
            return None

    def _save_to_cache(self, data: dict, cache_path: Path):
        """Save data to cache

        Args:
            data: Data to cache
            cache_path: Path to cache file
        """
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved to cache: {cache_path}")
        except OSError as e:
            logger.error(f"Error saving cache {cache_path}: {e}")

    async def get_bootstrap_data(self, use_cache: bool = True) -> dict:
        """Get bootstrap static data (players, teams, gameweeks)

        Args:
            use_cache: Whether to use cached data

        Returns:
            Bootstrap data dictionary
        """
        cache_path = self._get_cache_path("bootstrap")

        if use_cache:
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                self._bootstrap_data = cached_data
                return cached_data

        url = f"{self.BASE_URL}{self.ENDPOINTS['bootstrap']}"
        data = await self._fetch(url)

        self._save_to_cache(data, cache_path)
        self._bootstrap_data = data
        return data

    async def get_fixtures(
        self, gameweek: Optional[int] = None, use_cache: bool = True
    ) -> list[dict]:
        """Get fixtures data

        Args:
            gameweek: Specific gameweek to fetch (None for all)
            use_cache: Whether to use cached data

        Returns:
            List of fixture dictionaries
        """
        params = f"gw{gameweek}" if gameweek else "all"
        cache_path = self._get_cache_path("fixtures", params)

        if use_cache:
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                return cached_data

        url = f"{self.BASE_URL}{self.ENDPOINTS['fixtures']}"
        if gameweek:
            url += f"?event={gameweek}"

        data = await self._fetch(url)
        self._save_to_cache(data, cache_path)
        return data

    async def get_player_history(self, player_id: int, use_cache: bool = True) -> dict:
        """Get detailed player history

        Args:
            player_id: FPL player ID
            use_cache: Whether to use cached data

        Returns:
            Player history dictionary
        """
        cache_path = self._get_cache_path("player", str(player_id))

        if use_cache:
            cached_data = self._load_from_cache(cache_path, max_age_hours=24)
            if cached_data:
                return cached_data

        url = f"{self.BASE_URL}/element-summary/{player_id}/"
        data = await self._fetch(url)

        self._save_to_cache(data, cache_path)
        return data

    async def get_live_gameweek(self, gameweek: int, use_cache: bool = True) -> dict:
        """Get live gameweek data

        Args:
            gameweek: Gameweek number
            use_cache: Whether to use cached data

        Returns:
            Live gameweek data dictionary
        """
        cache_path = self._get_cache_path("live", f"gw{gameweek}")

        if use_cache:
            # Live data cache should be shorter
            cached_data = self._load_from_cache(cache_path, max_age_hours=1)
            if cached_data:
                return cached_data

        url = f"{self.BASE_URL}/event/{gameweek}/live/"
        data = await self._fetch(url)

        self._save_to_cache(data, cache_path)
        return data

    def process_players_data(self, bootstrap_data: dict) -> pd.DataFrame:
        """Process players data from bootstrap into DataFrame

        Args:
            bootstrap_data: Bootstrap API response

        Returns:
            DataFrame with processed player data
        """
        players_df = pd.DataFrame(bootstrap_data["elements"])
        teams_df = pd.DataFrame(bootstrap_data["teams"])
        positions_df = pd.DataFrame(bootstrap_data["element_types"])

        # Merge team names
        players_df = players_df.merge(
            teams_df[["id", "name", "short_name"]],
            left_on="team",
            right_on="id",
            suffixes=("", "_team"),
        )

        # Merge position names
        players_df = players_df.merge(
            positions_df[["id", "singular_name_short"]],
            left_on="element_type",
            right_on="id",
            suffixes=("", "_position"),
        )

        # Rename columns for clarity
        players_df.rename(
            columns={
                "singular_name_short": "position",
                "name": "team_name",
                "short_name": "team_short",
            },
            inplace=True,
        )

        # Map GKP to GK for consistency across the codebase
        players_df["position"] = players_df["position"].replace("GKP", "GK")

        # Convert cost from tenths to actual value
        players_df["price"] = players_df["now_cost"] / 10.0
        players_df["price_change"] = players_df["cost_change_event"] / 10.0

        # Select important columns
        important_cols = [
            "id",
            "web_name",
            "first_name",
            "second_name",
            "position",
            "team",
            "team_name",
            "team_short",
            "price",
            "price_change",
            "total_points",
            "points_per_game",
            "form",
            "selected_by_percent",
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "own_goals",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "expected_goals",
            "expected_assists",
            "expected_goal_involvements",
            "expected_goals_conceded",
            "value_form",
            "value_season",
            "transfers_in",
            "transfers_out",
            "transfers_in_event",
            "transfers_out_event",
            "chance_of_playing_next_round",
            "news",
            "news_added",
        ]

        # Filter to available columns
        available_cols = [col for col in important_cols if col in players_df.columns]
        players_df = players_df[available_cols]

        # Convert percentage strings to floats
        if "selected_by_percent" in players_df.columns:
            players_df["selected_by_percent"] = pd.to_numeric(
                players_df["selected_by_percent"], errors="coerce"
            )

        logger.info(f"Processed {len(players_df)} players")
        return players_df

    def process_fixtures_data(self, fixtures: list[dict]) -> pd.DataFrame:
        """Process fixtures data into DataFrame

        Args:
            fixtures: List of fixture dictionaries

        Returns:
            DataFrame with processed fixtures
        """
        fixtures_df = pd.DataFrame(fixtures)

        # Convert kickoff time
        if "kickoff_time" in fixtures_df.columns:
            fixtures_df["kickoff_time"] = pd.to_datetime(fixtures_df["kickoff_time"])

        # Add useful columns
        if "team_h_difficulty" in fixtures_df.columns:
            fixtures_df["home_difficulty"] = fixtures_df["team_h_difficulty"]
            fixtures_df["away_difficulty"] = fixtures_df["team_a_difficulty"]

        logger.info(f"Processed {len(fixtures_df)} fixtures")
        return fixtures_df

    async def get_gameweek_data(self, gameweek: int) -> dict[str, pd.DataFrame]:
        """Get all data for a specific gameweek

        Args:
            gameweek: Gameweek number

        Returns:
            Dictionary with DataFrames for players, fixtures, and live data
        """
        # Get bootstrap data if not already loaded
        if not self._bootstrap_data:
            self._bootstrap_data = await self.get_bootstrap_data()

        # Get fixtures for gameweek
        fixtures = await self.get_fixtures(gameweek)

        # Get live data if available
        try:
            live_data = await self.get_live_gameweek(gameweek)
        except Exception as e:
            logger.warning(f"Could not get live data for GW{gameweek}: {e}")
            live_data = None

        # Process into DataFrames
        result = {
            "players": self.process_players_data(self._bootstrap_data),
            "fixtures": self.process_fixtures_data(fixtures),
        }

        if live_data:
            result["live"] = pd.DataFrame(live_data.get("elements", []))

        return result

    async def fetch_historical_season(self, season: str = "2024-25") -> pd.DataFrame:
        """Fetch complete historical data for a season

        Args:
            season: Season identifier (e.g., "2024-25")

        Returns:
            DataFrame with all player-gameweek combinations
        """
        all_gameweeks = []

        # Get bootstrap data
        bootstrap = await self.get_bootstrap_data()
        total_gameweeks = len(bootstrap["events"])

        logger.info(f"Fetching {total_gameweeks} gameweeks for season {season}")

        for gw in range(1, total_gameweeks + 1):
            logger.info(f"Processing gameweek {gw}/{total_gameweeks}")

            try:
                gw_data = await self.get_gameweek_data(gw)
                players_df = gw_data["players"].copy()
                players_df["gameweek"] = gw
                players_df["season"] = season
                all_gameweeks.append(players_df)

                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching GW{gw}: {e}")
                continue

        if all_gameweeks:
            full_df = pd.concat(all_gameweeks, ignore_index=True)
            logger.info(f"Successfully fetched {len(full_df)} player-gameweek records")
            return full_df
        else:
            logger.error("No gameweek data could be fetched")
            return pd.DataFrame()


async def test_fpl_collector():
    """Test function for FPL API collector"""
    async with FPLAPICollector() as collector:
        # Test bootstrap data
        bootstrap = await collector.get_bootstrap_data()
        print(f"Found {len(bootstrap['elements'])} players")
        print(f"Found {len(bootstrap['teams'])} teams")
        print(f"Found {len(bootstrap['events'])} gameweeks")

        # Test processing players
        players_df = collector.process_players_data(bootstrap)
        print("\nTop 5 players by total points:")
        print(players_df.nlargest(5, "total_points")[["web_name", "total_points", "price"]])

        # Test fixtures
        fixtures = await collector.get_fixtures(gameweek=1)
        if fixtures:
            fixtures_df = collector.process_fixtures_data(fixtures)
            print(f"\nGameweek 1 has {len(fixtures_df)} fixtures")

        return players_df


if __name__ == "__main__":
    # Run test
    asyncio.run(test_fpl_collector())
