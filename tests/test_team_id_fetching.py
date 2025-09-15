"""
Test suite for FPL Team ID fetching functionality
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.fpl_api import FPLAPICollector
from src.my_team import MyTeam


class TestTeamIDFetching:
    """Test team fetching by ID functionality"""

    @pytest.fixture
    def mock_team_picks_response(self):
        """Mock response from FPL API for team picks"""
        return {
            "picks": [
                {"element": 1, "is_captain": True, "is_vice_captain": False},
                {"element": 15, "is_captain": False, "is_vice_captain": True},
                {"element": 234, "is_captain": False, "is_vice_captain": False},
                {"element": 567, "is_captain": False, "is_vice_captain": False},
                {"element": 890, "is_captain": False, "is_vice_captain": False},
                {"element": 123, "is_captain": False, "is_vice_captain": False},
                {"element": 456, "is_captain": False, "is_vice_captain": False},
                {"element": 789, "is_captain": False, "is_vice_captain": False},
                {"element": 101, "is_captain": False, "is_vice_captain": False},
                {"element": 202, "is_captain": False, "is_vice_captain": False},
                {"element": 303, "is_captain": False, "is_vice_captain": False},
                {"element": 404, "is_captain": False, "is_vice_captain": False},
                {"element": 505, "is_captain": False, "is_vice_captain": False},
                {"element": 606, "is_captain": False, "is_vice_captain": False},
                {"element": 707, "is_captain": False, "is_vice_captain": False},
            ],
            "entry_history": {
                "bank": 28,  # 2.8m in tenths
                "value": 1001,  # 100.1m in tenths
                "event_transfers": 1,
                "total_points": 236,
                "points": 45,
                "overall_rank": 1947545,
            },
        }

    @pytest.fixture
    def mock_team_history_response(self):
        """Mock response from FPL API for team history"""
        return {
            "entry": {
                "name": "Test FPL Team",
                "player_first_name": "John",
                "player_last_name": "Doe",
            },
            "chips": [
                {"name": "wildcard", "time": "2024-09-01T10:00:00Z", "event": 3},
            ],
        }

    @pytest.fixture
    def mock_bootstrap_response(self):
        """Mock bootstrap data response"""
        return {
            "events": [
                {"id": 1, "is_current": False, "finished": True},
                {"id": 2, "is_current": False, "finished": True},
                {"id": 3, "is_current": False, "finished": True},
                {"id": 4, "is_current": True, "finished": False},
            ],
            "elements": [],
            "teams": [],
        }

    @pytest.mark.asyncio
    async def test_get_team_history(self, mock_team_history_response):
        """Test fetching team history"""
        collector = FPLAPICollector()

        with patch.object(collector, '_fetch', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_team_history_response

            result = await collector.get_team_history(7954125, use_cache=False)

            assert result["entry"]["name"] == "Test FPL Team"
            assert result["entry"]["player_first_name"] == "John"
            assert len(result["chips"]) == 1
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_team_picks(self, mock_team_picks_response):
        """Test fetching team picks for a gameweek"""
        collector = FPLAPICollector()

        with patch.object(collector, '_fetch', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_team_picks_response

            result = await collector.get_team_picks(7954125, 4, use_cache=False)

            assert len(result["picks"]) == 15
            assert result["entry_history"]["bank"] == 28
            assert result["entry_history"]["total_points"] == 236
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_team(
        self, mock_bootstrap_response, mock_team_picks_response
    ):
        """Test fetching current team"""
        collector = FPLAPICollector()

        with patch.object(collector, 'get_bootstrap_data', new_callable=AsyncMock) as mock_bootstrap:
            mock_bootstrap.return_value = mock_bootstrap_response

            with patch.object(collector, 'get_team_picks', new_callable=AsyncMock) as mock_picks:
                mock_picks.return_value = mock_team_picks_response

                result = await collector.get_current_team(7954125, use_cache=False)

                assert len(result["picks"]) == 15
                mock_picks.assert_called_once_with(7954125, 4, False)

    def test_transform_api_team_to_myteam(
        self, mock_team_picks_response, mock_team_history_response
    ):
        """Test transforming API response to MyTeam structure"""
        collector = FPLAPICollector()

        result = collector.transform_api_team_to_myteam(
            mock_team_picks_response, mock_team_history_response
        )

        # Check player IDs
        assert len(result["players"]) == 15
        assert result["players"][0] == 1
        assert result["players"][1] == 15

        # Check captain and vice captain
        assert result["captain"] == 1
        assert result["vice_captain"] == 15

        # Check financial data
        assert result["bank"] == 2.8
        assert result["team_value"] == 100.1

        # Check chip availability
        assert result["wildcard_available"] == False  # Used in history
        assert result["free_hit_available"] == True
        assert result["bench_boost_available"] == True
        assert result["triple_captain_available"] == True

        # Check team metadata
        assert result["team_name"] == "Test FPL Team"
        assert result["manager_name"] == "John Doe"
        assert result["total_points"] == 236
        assert result["overall_rank"] == 1947545

    @pytest.mark.asyncio
    async def test_invalid_team_id(self):
        """Test handling of invalid team ID"""
        collector = FPLAPICollector()

        import aiohttp
        with patch.object(collector, '_fetch', new_callable=AsyncMock) as mock_fetch:
            # Simulate API error for invalid team
            mock_fetch.side_effect = aiohttp.ClientError("404 Not Found")

            with pytest.raises(ValueError) as exc_info:
                await collector.get_team_history(99999999, use_cache=False)

            assert "Could not fetch team" in str(exc_info.value)

    def test_myteam_creation_from_fetched_data(
        self, mock_team_picks_response, mock_team_history_response
    ):
        """Test creating MyTeam object from fetched data"""
        collector = FPLAPICollector()

        team_data = collector.transform_api_team_to_myteam(
            mock_team_picks_response, mock_team_history_response
        )

        # Create MyTeam object
        my_team = MyTeam(
            players=team_data["players"],
            captain=team_data["captain"],
            vice_captain=team_data["vice_captain"],
            bank=team_data["bank"],
            free_transfers=team_data["free_transfers"],
            wildcard_available=team_data["wildcard_available"],
            free_hit_available=team_data["free_hit_available"],
            bench_boost_available=team_data["bench_boost_available"],
            triple_captain_available=team_data["triple_captain_available"],
        )

        assert len(my_team.players) == 15
        assert my_team.captain == 1
        assert my_team.vice_captain == 15
        assert my_team.bank == 2.8
        assert my_team.free_transfers == 1
        assert my_team.wildcard_available == False