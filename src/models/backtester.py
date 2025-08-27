"""
Backtesting Framework for FPL Models
Simulates model performance over historical seasons
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class FPLTeam:
    """Represents an FPL team at a point in time"""
    players: List[int]  # Player IDs
    captain: int  # Captain ID
    vice_captain: int  # Vice captain ID
    bank: float  # Remaining budget
    total_points: int = 0
    transfers_made: int = 0
    wildcards_used: int = 0
    triple_captain_used: bool = False
    bench_boost_used: bool = False
    free_hit_used: bool = False


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    total_points: int
    gameweek_points: List[int]
    final_team_value: float
    transfers_made: int
    chips_used: Dict[str, int]  # Chip name -> gameweek used
    captain_points: int
    bench_points: int
    rank_percentile: float  # Percentile rank (0-100, higher is better)


class Backtester:
    """Backtest FPL strategies on historical data"""
    
    # FPL constraints
    MAX_PLAYERS = 15
    MAX_SAME_TEAM = 3
    POSITIONS = {
        'GK': {'min': 2, 'max': 2, 'playing': 1},
        'DEF': {'min': 5, 'max': 5, 'playing': {'min': 3, 'max': 5}},
        'MID': {'min': 5, 'max': 5, 'playing': {'min': 2, 'max': 5}},
        'FWD': {'min': 3, 'max': 3, 'playing': {'min': 1, 'max': 3}}
    }
    BUDGET = 100.0
    FREE_TRANSFERS_PER_WEEK = 1
    TRANSFER_COST = 4  # Points cost per extra transfer
    
    def __init__(self, historical_data: pd.DataFrame, model):
        """Initialize backtester
        
        Args:
            historical_data: Historical player-gameweek data
            model: Model to test (must have score_all_players method)
        """
        self.data = historical_data
        self.model = model
        self.current_team = None
        
    def _get_gameweek_data(self, gameweek: int, season: str) -> pd.DataFrame:
        """Get data for a specific gameweek
        
        Args:
            gameweek: Gameweek number
            season: Season identifier
            
        Returns:
            DataFrame with gameweek data
        """
        mask = (self.data['gameweek'] == gameweek) & (self.data['season'] == season)
        return self.data[mask].copy()
        
    def _initialize_team(self, budget: float = BUDGET) -> FPLTeam:
        """Initialize a team at the start of season
        
        Args:
            budget: Starting budget
            
        Returns:
            Initial FPL team
        """
        # Get first gameweek data
        gw1_data = self._get_gameweek_data(1, self.data['season'].iloc[0])
        
        # Score all players
        scored = self.model.score_all_players(gw1_data)
        
        # Select best team within constraints
        selected_players = self._select_optimal_team(scored, budget)
        
        # Choose captain (highest scoring player)
        captain = selected_players[0]
        vice_captain = selected_players[1]
        
        return FPLTeam(
            players=selected_players,
            captain=captain,
            vice_captain=vice_captain,
            bank=budget - self._calculate_team_cost(selected_players, gw1_data)
        )
        
    def _select_optimal_team(self, players_df: pd.DataFrame, 
                            budget: float) -> List[int]:
        """Select optimal team within FPL constraints
        
        Args:
            players_df: Scored players DataFrame
            budget: Available budget
            
        Returns:
            List of selected player IDs
        """
        selected = []
        spent = 0.0
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        team_counts = {}
        
        # Sort by score descending
        candidates = players_df.sort_values('model_score', ascending=False)
        
        for _, player in candidates.iterrows():
            player_id = int(player['player_id'])
            position = player['position']
            # Map GKP to GK for consistency
            if position == 'GKP':
                position = 'GK'
            team = player['team_name']
            price = player['price']
            
            # Check position limit
            if position not in self.POSITIONS:
                continue
            if position_counts[position] >= self.POSITIONS[position]['max']:
                continue
                
            # Check team limit
            if team_counts.get(team, 0) >= self.MAX_SAME_TEAM:
                continue
                
            # Check budget
            if spent + price > budget:
                continue
                
            # Add player
            selected.append(player_id)
            spent += price
            position_counts[position] += 1
            team_counts[team] = team_counts.get(team, 0) + 1
            
            # Check if team is complete
            if len(selected) == self.MAX_PLAYERS:
                break
                
        # Ensure minimum positions are met
        for position, reqs in self.POSITIONS.items():
            if position_counts[position] < reqs['min']:
                # Find cheapest players to fill position
                pos_players = candidates[
                    (candidates['position'] == position) & 
                    (~candidates['player_id'].isin(selected))
                ].nsmallest(reqs['min'] - position_counts[position], 'price')
                
                for _, player in pos_players.iterrows():
                    selected.append(int(player['player_id']))
                    
        return selected[:self.MAX_PLAYERS]
        
    def _calculate_team_cost(self, player_ids: List[int], 
                            players_df: pd.DataFrame) -> float:
        """Calculate total cost of team
        
        Args:
            player_ids: List of player IDs
            players_df: DataFrame with player data
            
        Returns:
            Total team cost
        """
        team_players = players_df[players_df['player_id'].isin(player_ids)]
        return team_players['price'].sum()
        
    def _simulate_transfers(self, current_team: FPLTeam, 
                          current_gw_data: pd.DataFrame,
                          free_transfers: int = 1) -> Tuple[FPLTeam, int]:
        """Simulate optimal transfers for a gameweek
        
        Args:
            current_team: Current team
            current_gw_data: Current gameweek data
            free_transfers: Number of free transfers available
            
        Returns:
            Updated team and points cost
        """
        # Score all players
        scored = self.model.score_all_players(current_gw_data)
        
        # Find worst performing players in team
        team_players = scored[scored['player_id'].isin(current_team.players)]
        worst_players = team_players.nsmallest(2, 'model_score')['player_id'].tolist()
        
        # Find best available players not in team
        available = scored[~scored['player_id'].isin(current_team.players)]
        
        transfers = []
        for worst_id in worst_players[:free_transfers + 1]:  # Allow 1 extra transfer
            # Get player to remove
            worst_player = team_players[team_players['player_id'] == worst_id].iloc[0]
            
            # Find best replacement within budget
            max_price = worst_player['price'] + current_team.bank
            replacements = available[
                (available['position'] == worst_player['position']) &
                (available['price'] <= max_price)
            ]
            
            if not replacements.empty:
                best_replacement = replacements.iloc[0]
                transfers.append((worst_id, int(best_replacement['player_id'])))
                
                # Update available budget
                current_team.bank += worst_player['price'] - best_replacement['price']
                
        # Apply transfers
        transfer_cost = 0
        for out_id, in_id in transfers:
            current_team.players.remove(out_id)
            current_team.players.append(in_id)
            current_team.transfers_made += 1
            
        # Calculate transfer cost
        if len(transfers) > free_transfers:
            transfer_cost = (len(transfers) - free_transfers) * self.TRANSFER_COST
            
        # Update captain
        team_scores = scored[scored['player_id'].isin(current_team.players)]
        if not team_scores.empty:
            current_team.captain = int(team_scores.iloc[0]['player_id'])
            if len(team_scores) > 1:
                current_team.vice_captain = int(team_scores.iloc[1]['player_id'])
                
        return current_team, transfer_cost
        
    def _calculate_gameweek_points(self, team: FPLTeam, 
                                  gw_data: pd.DataFrame) -> int:
        """Calculate points for a gameweek
        
        Args:
            team: FPL team
            gw_data: Gameweek data with actual points
            
        Returns:
            Total gameweek points
        """
        points = 0
        
        # Get team players' points
        team_players = gw_data[gw_data['player_id'].isin(team.players)]
        
        for _, player in team_players.iterrows():
            player_points = player.get('gameweek_points', 0)
            # Ensure player_points is not None
            if player_points is None:
                player_points = 0
            
            # Double points for captain
            if player['player_id'] == team.captain:
                player_points *= 2
                
            points += player_points
            
        return int(points)
        
    def run_backtest(self, season: str, start_gw: int = 1, 
                    end_gw: int = 38) -> BacktestResult:
        """Run backtest for a season
        
        Args:
            season: Season to test
            start_gw: Starting gameweek
            end_gw: Ending gameweek
            
        Returns:
            Backtest results
        """
        logger.info(f"Starting backtest for season {season}, GW{start_gw}-{end_gw}")
        
        # Initialize team
        self.current_team = self._initialize_team()
        
        # Track results
        gameweek_points = []
        total_points = 0
        captain_points = 0
        bench_points = 0
        chips_used = {}
        
        # Simulate each gameweek
        free_transfers = 1
        
        for gw in range(start_gw, end_gw + 1):
            # Get gameweek data
            gw_data = self._get_gameweek_data(gw, season)
            
            if gw_data.empty:
                logger.warning(f"No data for GW{gw}")
                continue
                
            # Make transfers (except GW1)
            transfer_cost = 0
            if gw > start_gw:
                self.current_team, transfer_cost = self._simulate_transfers(
                    self.current_team, gw_data, free_transfers
                )
                
                # Update free transfers
                free_transfers = min(2, free_transfers + 1)
                if transfer_cost > 0:
                    free_transfers = 1
                    
            # Calculate points
            gw_points = self._calculate_gameweek_points(self.current_team, gw_data)
            gw_points -= transfer_cost
            
            # Track captain points
            captain_data = gw_data[gw_data['player_id'] == self.current_team.captain]
            if not captain_data.empty:
                captain_pts = captain_data.iloc[0].get('gameweek_points', 0)
                if captain_pts is not None:
                    captain_points += captain_pts
                
            gameweek_points.append(gw_points)
            total_points += gw_points
            
            logger.debug(f"GW{gw}: {gw_points} points (transfer cost: {transfer_cost})")
            
        # Calculate final team value
        final_gw_data = self._get_gameweek_data(end_gw, season)
        final_team_value = self._calculate_team_cost(
            self.current_team.players, final_gw_data
        ) + self.current_team.bank
        
        # Estimate rank percentile (simplified)
        avg_points = 50 * (end_gw - start_gw + 1)  # Rough average
        if total_points > avg_points * 1.2:
            rank_percentile = 95  # Top 5%
        elif total_points > avg_points * 1.1:
            rank_percentile = 90  # Top 10%
        elif total_points > avg_points:
            rank_percentile = 70  # Top 30%
        else:
            rank_percentile = 50  # Average
            
        return BacktestResult(
            total_points=total_points,
            gameweek_points=gameweek_points,
            final_team_value=final_team_value,
            transfers_made=self.current_team.transfers_made,
            chips_used=chips_used,
            captain_points=captain_points,
            bench_points=bench_points,
            rank_percentile=rank_percentile
        )
        
    def compare_strategies(self, strategies: Dict[str, any], 
                         season: str) -> pd.DataFrame:
        """Compare multiple strategies
        
        Args:
            strategies: Dictionary of strategy_name -> model
            season: Season to test
            
        Returns:
            DataFrame comparing results
        """
        results = []
        
        for name, model in strategies.items():
            logger.info(f"Testing strategy: {name}")
            
            # Update model
            self.model = model
            
            # Run backtest
            result = self.run_backtest(season)
            
            results.append({
                'strategy': name,
                'total_points': result.total_points,
                'avg_gw_points': np.mean(result.gameweek_points),
                'final_value': result.final_team_value,
                'transfers': result.transfers_made,
                'rank_percentile': result.rank_percentile
            })
            
        return pd.DataFrame(results).sort_values('total_points', ascending=False)


def test_backtester():
    """Test backtesting framework"""
    # Create sample historical data
    np.random.seed(42)
    
    players = []
    for gw in range(1, 6):  # 5 gameweeks
        for player_id in range(1, 101):  # 100 players
            players.append({
                'player_id': player_id,
                'gameweek': gw,
                'season': '2024-25',
                'player_name': f'Player_{player_id}',
                'position': ['GK', 'DEF', 'MID', 'FWD'][player_id % 4],
                'team_name': f'Team_{player_id % 20}',
                'price': 4.0 + (player_id % 20) * 0.5,
                'gameweek_points': np.random.randint(0, 15),
                'total_points': np.random.randint(0, 200),
                'form': np.random.uniform(0, 10),
                'prob_goal': np.random.uniform(0, 0.5),
                'expected_goals': np.random.uniform(0, 1),
                'model_score': np.random.uniform(0, 100)
            })
            
    historical_data = pd.DataFrame(players)
    
    # Create mock model
    class MockModel:
        def score_all_players(self, df):
            df['model_score'] = df['form'] * 10 + df['prob_goal'] * 50
            return df
            
    # Test backtester
    model = MockModel()
    backtester = Backtester(historical_data, model)
    
    # Run backtest
    result = backtester.run_backtest('2024-25', start_gw=1, end_gw=5)
    
    print(f"Total Points: {result.total_points}")
    print(f"Gameweek Points: {result.gameweek_points}")
    print(f"Final Team Value: {result.final_team_value:.1f}")
    print(f"Transfers Made: {result.transfers_made}")
    print(f"Rank Percentile: {result.rank_percentile}%")
    
    return result


if __name__ == "__main__":
    # Run test
    test_backtester()