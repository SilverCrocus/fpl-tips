"""
Odds API Data Collector Module V2
Fetches REAL player prop betting odds from The Odds API
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry

from .name_matcher import PlayerNameMatcher

load_dotenv()
logger = logging.getLogger(__name__)


class PlayerPropsCollector:
    """Collector for real player prop odds from The Odds API"""

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "soccer_epl"

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "cache/player_props"):
        """Initialize Player Props collector

        Args:
            api_key: The Odds API key
            cache_dir: Directory for caching
        """
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.name_matcher = PlayerNameMatcher()  # Initialize name matcher

        if not self.api_key:
            raise ValueError(
                "The Odds API key is required. Please set ODDS_API_KEY in your .env file"
            )

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.cache_dir / f"{cache_key}_{timestamp}.json"

    def _load_from_cache(self, cache_key: str, max_age_hours: int = 24) -> Optional[dict]:
        """Load from cache if fresh (24 hours for player props)"""
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(hours=max_age_hours):
            logger.info(f"Cache expired for {cache_key}")
            return None

        try:
            with open(cache_path) as f:
                logger.info(f"Loading from cache: {cache_path}")
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _save_to_cache(self, data: dict, cache_key: str):
        """Save to cache"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved to cache: {cache_path}")
        except OSError as e:
            logger.error(f"Error saving cache: {e}")

    @sleep_and_retry
    @limits(calls=30, period=60)  # More generous for player props
    async def _fetch(self, url: str, params: dict = None) -> Any:
        """Fetch data from The Odds API"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.get(url, params=params) as response:
                # Log API usage
                remaining = response.headers.get("x-requests-remaining")
                used = response.headers.get("x-requests-used")
                if remaining:
                    logger.info(f"API Requests - Used: {used}, Remaining: {remaining}")

                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API Error {response.status}: {error_text}")
                    return None

                return await response.json()

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching from API: {e}")
            return None

    async def get_upcoming_matches(self) -> list[dict]:
        """Get upcoming EPL matches with event IDs"""
        cache_key = "epl_matches"
        cached = self._load_from_cache(cache_key, max_age_hours=12)

        if cached:
            return cached

        logger.info("Fetching upcoming EPL matches...")

        url = f"{self.BASE_URL}/sports/{self.SPORT}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",  # US region for player props
            "markets": "h2h",  # Basic market to get event IDs
            "oddsFormat": "decimal",
        }

        matches = await self._fetch(url, params)

        if matches:
            self._save_to_cache(matches, cache_key)
            logger.info(f"Found {len(matches)} upcoming matches")

        return matches or []

    async def get_player_goal_scorer_odds(self, event_id: str, event_name: str = "") -> dict:
        """Get anytime goal scorer odds for a specific match

        Args:
            event_id: The event ID from get_upcoming_matches
            event_name: Optional name for logging

        Returns:
            Dictionary with player odds data
        """
        cache_key = f"player_goals_{event_id}"
        cached = self._load_from_cache(cache_key)

        if cached:
            return cached

        logger.info(f"Fetching player goal scorer odds for {event_name or event_id}")

        url = f"{self.BASE_URL}/sports/{self.SPORT}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "player_goal_scorer_anytime",
            "oddsFormat": "decimal",
        }

        data = await self._fetch(url, params)

        if data:
            self._save_to_cache(data, cache_key)

        return data or {}

    async def get_all_player_props_for_gameweek(self) -> pd.DataFrame:
        """Get all player goal scorer odds for current gameweek

        Returns:
            DataFrame with player odds from all matches
        """
        # Step 1: Get all upcoming matches
        matches = await self.get_upcoming_matches()

        if not matches:
            logger.warning("No upcoming matches found")
            return pd.DataFrame()

        all_player_odds = []

        # Step 2: Fetch player props for each match
        for match in matches:
            event_id = match.get("id")
            home_team = match.get("home_team")
            away_team = match.get("away_team")
            commence_time = match.get("commence_time")

            match_name = f"{home_team} vs {away_team}"
            logger.info(f"Processing {match_name}...")

            # Get player goal scorer odds
            player_data = await self.get_player_goal_scorer_odds(event_id, match_name)

            if player_data and "bookmakers" in player_data:
                # Process each bookmaker's odds
                for bookmaker in player_data["bookmakers"]:
                    bookmaker_key = bookmaker.get("key")
                    bookmaker_title = bookmaker.get("title")

                    # Find player markets
                    for market in bookmaker.get("markets", []):
                        if market["key"] == "player_goal_scorer_anytime":
                            outcomes = market.get("outcomes", [])

                            for outcome in outcomes:
                                # Extract player name from description field
                                player_name = outcome.get("description", "")
                                if not player_name:
                                    # Fallback to name field if description not available
                                    player_name = outcome.get("name", "").replace(
                                        " - Anytime Goalscorer", ""
                                    )
                                    player_name = player_name.replace("Yes", "").strip()
                                odds = outcome.get("price", 0)

                                if player_name and odds > 0:
                                    all_player_odds.append(
                                        {
                                            "match_id": event_id,
                                            "home_team": home_team,
                                            "away_team": away_team,
                                            "commence_time": commence_time,
                                            "player_name": player_name,
                                            "odds_goal_anytime": odds,
                                            "prob_goal": (
                                                round(1 / max(odds, 0.01), 3) if odds > 0 else 0
                                            ),
                                            "bookmaker": bookmaker_key,
                                            "bookmaker_title": bookmaker_title,
                                            "is_real_odds": True,  # Flag to indicate these are real bookmaker odds
                                        }
                                    )

                            break  # We found the player market

        logger.info(f"Collected odds for {len(all_player_odds)} player-match combinations")

        return pd.DataFrame(all_player_odds)

    async def get_player_assists_odds(self, event_id: str, event_name: str = "") -> dict:
        """Get player assist odds for a specific match"""
        cache_key = f"player_assists_{event_id}"
        cached = self._load_from_cache(cache_key)

        if cached:
            return cached

        logger.info(f"Fetching player assist odds for {event_name or event_id}")

        url = f"{self.BASE_URL}/sports/{self.SPORT}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "player_assists",
            "oddsFormat": "decimal",
        }

        data = await self._fetch(url, params)

        if data:
            self._save_to_cache(data, cache_key)

        return data or {}

    def match_players_to_fpl(
        self, player_odds_df: pd.DataFrame, fpl_players_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Match bookmaker player names to FPL player data using enhanced matching

        Args:
            player_odds_df: DataFrame with bookmaker player odds
            fpl_players_df: DataFrame with FPL player data

        Returns:
            DataFrame with real odds matched to FPL players
        """
        if player_odds_df.empty:
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=["player_id", "odds_goal", "prob_goal", "is_real_odds"])

        # Import the enhanced matcher
        from .name_matcher import PlayerNameMatcher

        # Initialize matcher
        matcher = PlayerNameMatcher()

        # Prepare bookmaker data for matching
        player_odds_df = player_odds_df.copy()
        if (
            "player_name" not in player_odds_df.columns
            and "bookmaker_name" in player_odds_df.columns
        ):
            player_odds_df["player_name"] = player_odds_df["bookmaker_name"]

        # Rename column for matcher compatibility
        player_odds_df["bookmaker_name"] = player_odds_df["player_name"]

        # Create simplified odds DataFrame with unique players
        # Average odds if player appears multiple times (different bookmakers)
        player_summary = (
            player_odds_df.groupby("player_name")
            .agg(
                {
                    "odds_goal_anytime": "mean",
                    "prob_goal": "mean",
                    "home_team": "first",
                    "away_team": "first",
                }
            )
            .reset_index()
        )
        player_summary["bookmaker_name"] = player_summary["player_name"]

        # Match players using enhanced matcher
        matches = matcher.match_players(
            fpl_players_df, player_summary, threshold=0.75  # Lower threshold to catch more matches
        )

        # Merge with odds data
        matched_odds = []

        for _, match in matches[matches["matched"]].iterrows():
            # Find the corresponding odds
            odds_row = player_summary[player_summary["bookmaker_name"] == match["bookmaker_name"]]

            if not odds_row.empty:
                matched_odds.append(
                    {
                        "player_id": match["player_id"],
                        "odds_goal": odds_row.iloc[0]["odds_goal_anytime"],
                        "prob_goal": odds_row.iloc[0]["prob_goal"],
                        "is_real_odds": True,
                        "match_confidence": match["match_score"],
                    }
                )

        if matched_odds:
            result_df = pd.DataFrame(matched_odds)

            # Log matching statistics
            total_fpl = len(fpl_players_df)
            matched = len(matched_odds)
            logger.info(
                f"Matched {matched}/{total_fpl} FPL players to bookmaker odds ({matched*100/total_fpl:.1f}%)"
            )

            # Log some sample matches for verification
            if matched > 0:
                high_confidence = result_df[result_df["match_confidence"] > 0.9]
                if not high_confidence.empty:
                    logger.debug(f"High confidence matches: {len(high_confidence)}")

                low_confidence = result_df[result_df["match_confidence"] < 0.8]
                if not low_confidence.empty:
                    logger.debug(f"Low confidence matches: {len(low_confidence)}")

            return result_df[["player_id", "odds_goal", "prob_goal", "is_real_odds"]]
        else:
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=["player_id", "odds_goal", "prob_goal", "is_real_odds"])


async def test_real_player_props():
    """Test function for real player props collector"""
    async with PlayerPropsCollector() as collector:
        try:
            # Get all player goal scorer odds for gameweek
            player_odds_df = await collector.get_all_player_props_for_gameweek()

            if not player_odds_df.empty:
                print("\n" + "=" * 60)
                print("REAL PLAYER GOAL SCORER ODDS")
                print("=" * 60)

                # Group by match
                for match_id, match_group in player_odds_df.groupby("match_id"):
                    match = match_group.iloc[0]
                    print(f"\n{match['home_team']} vs {match['away_team']}")
                    print(f"Bookmaker: {match['bookmaker_title']}")
                    print("-" * 40)

                    # Show top 5 most likely scorers
                    top_scorers = match_group.nsmallest(5, "odds_goal_anytime")
                    for _, player in top_scorers.iterrows():
                        print(
                            f"  {player['player_name']:20} {player['odds_goal_anytime']:6.2f} "
                            f"({player['prob_goal']:.1%} chance)"
                        )

                    print(f"  ... and {len(match_group) - 5} more players")

                print(
                    f"\n\nTotal: {len(player_odds_df)} player odds across "
                    f"{player_odds_df['match_id'].nunique()} matches"
                )

                return player_odds_df
            else:
                print("No player odds data available")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            return pd.DataFrame()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_real_player_props())
