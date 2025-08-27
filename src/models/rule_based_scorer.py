"""
Rule-Based Scoring Model
Position-specific scoring formulas optimized through backtesting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy.optimize import differential_evolution
import json

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Weights for scoring formula"""
    # Position-specific weights
    gk_weights: Dict[str, float]
    def_weights: Dict[str, float]
    mid_weights: Dict[str, float]
    fwd_weights: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'GK': self.gk_weights,
            'DEF': self.def_weights,
            'MID': self.mid_weights,
            'FWD': self.fwd_weights
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScoringWeights':
        """Create from dictionary"""
        return cls(
            gk_weights=data['GK'],
            def_weights=data['DEF'],
            mid_weights=data['MID'],
            fwd_weights=data['FWD']
        )
    
    @classmethod
    def default(cls) -> 'ScoringWeights':
        """Get default weights"""
        return cls(
            gk_weights={
                'prob_clean_sheet': 2.0,
                'saves_per_game': 0.5,
                'team_elo': 0.001,
                'fixture_difficulty': -0.5,
                'form': 0.3,
                'value_ratio': 0.2
            },
            def_weights={
                'prob_clean_sheet': 2.5,
                'prob_goal': 1.0,
                'prob_assist': 1.2,
                'expected_assists': 1.5,
                'team_elo': 0.001,
                'fixture_difficulty': -0.5,
                'form': 0.4,
                'threat': 0.01,
                'value_ratio': 0.3
            },
            mid_weights={
                'prob_goal': 3.0,
                'prob_assist': 2.0,
                'expected_goals': 2.5,
                'expected_assists': 2.0,
                'creativity': 0.02,
                'threat': 0.02,
                'form': 0.5,
                'fixture_difficulty': -0.4,
                'value_ratio': 0.3,
                'ict_per_90': 0.01
            },
            fwd_weights={
                'prob_goal': 3.5,
                'expected_goals': 3.0,
                'prob_assist': 1.5,
                'threat': 0.03,
                'form': 0.6,
                'fixture_difficulty': -0.3,
                'value_ratio': 0.2,
                'goal_involvement': 2.0,
                'minutes': 0.01
            }
        )


class RuleBasedScorer:
    """Rule-based scoring model for FPL players"""
    
    def __init__(self, weights: Optional[ScoringWeights] = None):
        """Initialize scorer
        
        Args:
            weights: Scoring weights (uses default if None)
        """
        self.weights = weights or ScoringWeights.default()
        
    def score_player(self, player: pd.Series) -> float:
        """Score a single player
        
        Args:
            player: Player data as pandas Series
            
        Returns:
            Player score
        """
        position = player.get('position', 'MID')
        
        # Get position-specific weights
        if position == 'GK':
            weights = self.weights.gk_weights
        elif position == 'DEF':
            weights = self.weights.def_weights
        elif position == 'MID':
            weights = self.weights.mid_weights
        elif position == 'FWD':
            weights = self.weights.fwd_weights
        else:
            weights = self.weights.mid_weights  # Default
            
        # Calculate weighted score
        score = 0
        for feature, weight in weights.items():
            value = player.get(feature, 0)
            if pd.notna(value):
                score += weight * float(value)
                
        # Apply availability penalty
        if player.get('chance_of_playing_next_round', 100) < 75:
            score *= (player.get('chance_of_playing_next_round', 0) / 100)
            
        # Apply price consideration (slight preference for cheaper players of equal score)
        # Don't penalize expensive players - only give bonus to cheaper ones
        price = player.get('price', 5.0)
        price_factor = max(1.0, 1 + 0.01 * (10 - price))
        score = score * price_factor
        
        return max(0, score)  # Ensure non-negative
        
    def score_all_players(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Score all players - ONLY those with real odds
        
        Args:
            players_df: DataFrame with player data
            
        Returns:
            DataFrame with scores added (only players with real odds)
        """
        players_df = players_df.copy()
        
        # CRITICAL: Filter to only players with real odds
        if 'has_real_odds' in players_df.columns:
            initial_count = len(players_df)
            players_df = players_df[players_df['has_real_odds'] == True].copy()
            filtered_count = len(players_df)
            logger.info(f"Filtered to {filtered_count}/{initial_count} players with REAL odds only")
            
            if filtered_count == 0:
                logger.warning("No players with real odds found - no recommendations possible")
                return pd.DataFrame()  # Return empty DataFrame
        
        # Calculate scores only for players with real odds
        players_df['model_score'] = players_df.apply(self.score_player, axis=1)
        
        # Add position ranks
        players_df['position_rank'] = players_df.groupby('position')['model_score'].rank(
            ascending=False, method='dense'
        )
        
        # Add overall rank
        players_df['overall_rank'] = players_df['model_score'].rank(
            ascending=False, method='dense'
        )
        
        # Sort by score
        players_df = players_df.sort_values('model_score', ascending=False)
        
        logger.info(f"Scored {len(players_df)} players")
        return players_df
        
    def get_top_players(self, players_df: pd.DataFrame, 
                       position: Optional[str] = None,
                       max_price: Optional[float] = None,
                       top_n: int = 10) -> pd.DataFrame:
        """Get top players by score
        
        Args:
            players_df: DataFrame with player data
            position: Filter by position
            max_price: Maximum price filter
            top_n: Number of top players to return
            
        Returns:
            Top players DataFrame
        """
        # Score players first
        scored_df = self.score_all_players(players_df)
        
        # Apply filters
        filtered = scored_df.copy()
        
        if position:
            filtered = filtered[filtered['position'] == position]
            
        if max_price:
            filtered = filtered[filtered['price'] <= max_price]
            
        # Get top N
        top_players = filtered.head(top_n)
        
        # Select display columns
        display_cols = [
            'player_name', 'position', 'team_name', 'price', 
            'model_score', 'position_rank', 'total_points',
            'form', 'prob_goal', 'prob_assist', 'fixture_difficulty'
        ]
        
        available_cols = [col for col in display_cols if col in top_players.columns]
        
        return top_players[available_cols]
        
    def save_weights(self, filepath: str):
        """Save weights to JSON file
        
        Args:
            filepath: Path to save weights
        """
        with open(filepath, 'w') as f:
            json.dump(self.weights.to_dict(), f, indent=2)
        logger.info(f"Saved weights to {filepath}")
        
    def load_weights(self, filepath: str):
        """Load weights from JSON file
        
        Args:
            filepath: Path to load weights from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.weights = ScoringWeights.from_dict(data)
        logger.info(f"Loaded weights from {filepath}")


class WeightOptimizer:
    """Optimizer for finding optimal scoring weights"""
    
    def __init__(self, historical_data: pd.DataFrame):
        """Initialize optimizer
        
        Args:
            historical_data: Historical player-gameweek data
        """
        self.data = historical_data
        self.best_weights = None
        self.best_score = -np.inf
        
    def _create_weight_bounds(self) -> List[Tuple[float, float]]:
        """Create bounds for optimization
        
        Returns:
            List of (min, max) bounds for each weight
        """
        # Define bounds for different feature types
        bounds = []
        
        # GK weights (6 features)
        bounds.extend([
            (0, 5),    # prob_clean_sheet
            (0, 2),    # saves_per_game
            (0, 0.01), # team_elo
            (-2, 0),   # fixture_difficulty
            (0, 2),    # form
            (0, 1)     # value_ratio
        ])
        
        # DEF weights (9 features)
        bounds.extend([
            (0, 5),    # prob_clean_sheet
            (0, 3),    # prob_goal
            (0, 3),    # prob_assist
            (0, 3),    # expected_assists
            (0, 0.01), # team_elo
            (-2, 0),   # fixture_difficulty
            (0, 2),    # form
            (0, 0.1),  # threat
            (0, 1)     # value_ratio
        ])
        
        # MID weights (10 features)
        bounds.extend([
            (0, 5),    # prob_goal
            (0, 3),    # prob_assist
            (0, 4),    # expected_goals
            (0, 3),    # expected_assists
            (0, 0.1),  # creativity
            (0, 0.1),  # threat
            (0, 2),    # form
            (-2, 0),   # fixture_difficulty
            (0, 1),    # value_ratio
            (0, 0.05)  # ict_per_90
        ])
        
        # FWD weights (9 features)
        bounds.extend([
            (0, 5),    # prob_goal
            (0, 4),    # expected_goals
            (0, 3),    # prob_assist
            (0, 0.1),  # threat
            (0, 2),    # form
            (-2, 0),   # fixture_difficulty
            (0, 1),    # value_ratio
            (0, 3),    # goal_involvement
            (0, 0.02)  # minutes
        ])
        
        return bounds
        
    def _weights_from_vector(self, x: np.ndarray) -> ScoringWeights:
        """Convert optimization vector to weights
        
        Args:
            x: Flattened weight vector
            
        Returns:
            ScoringWeights object
        """
        idx = 0
        
        # GK weights
        gk_weights = {
            'prob_clean_sheet': x[idx],
            'saves_per_game': x[idx+1],
            'team_elo': x[idx+2],
            'fixture_difficulty': x[idx+3],
            'form': x[idx+4],
            'value_ratio': x[idx+5]
        }
        idx += 6
        
        # DEF weights
        def_weights = {
            'prob_clean_sheet': x[idx],
            'prob_goal': x[idx+1],
            'prob_assist': x[idx+2],
            'expected_assists': x[idx+3],
            'team_elo': x[idx+4],
            'fixture_difficulty': x[idx+5],
            'form': x[idx+6],
            'threat': x[idx+7],
            'value_ratio': x[idx+8]
        }
        idx += 9
        
        # MID weights
        mid_weights = {
            'prob_goal': x[idx],
            'prob_assist': x[idx+1],
            'expected_goals': x[idx+2],
            'expected_assists': x[idx+3],
            'creativity': x[idx+4],
            'threat': x[idx+5],
            'form': x[idx+6],
            'fixture_difficulty': x[idx+7],
            'value_ratio': x[idx+8],
            'ict_per_90': x[idx+9]
        }
        idx += 10
        
        # FWD weights
        fwd_weights = {
            'prob_goal': x[idx],
            'expected_goals': x[idx+1],
            'prob_assist': x[idx+2],
            'threat': x[idx+3],
            'form': x[idx+4],
            'fixture_difficulty': x[idx+5],
            'value_ratio': x[idx+6],
            'goal_involvement': x[idx+7],
            'minutes': x[idx+8]
        }
        
        return ScoringWeights(
            gk_weights=gk_weights,
            def_weights=def_weights,
            mid_weights=mid_weights,
            fwd_weights=fwd_weights
        )
        
    def objective_function(self, x: np.ndarray) -> float:
        """Objective function for optimization (to minimize)
        
        Args:
            x: Weight vector
            
        Returns:
            Negative correlation with actual points (to minimize)
        """
        # Create weights from vector
        weights = self._weights_from_vector(x)
        
        # Create scorer with these weights
        scorer = RuleBasedScorer(weights)
        
        # Score all players
        scored_data = scorer.score_all_players(self.data)
        
        # Calculate correlation with actual points
        # Use future points if available (next gameweek performance)
        if 'future_points' in scored_data.columns:
            correlation = scored_data['model_score'].corr(scored_data['future_points'])
        else:
            # Use total points as proxy
            correlation = scored_data['model_score'].corr(scored_data['total_points'])
            
        # Return negative correlation (we want to maximize correlation)
        return -correlation if pd.notna(correlation) else 0
        
    def optimize(self, iterations: int = 100, seed: int = 42) -> ScoringWeights:
        """Optimize weights using differential evolution
        
        Args:
            iterations: Number of iterations
            seed: Random seed
            
        Returns:
            Optimized weights
        """
        logger.info(f"Starting weight optimization with {iterations} iterations")
        
        # Get bounds
        bounds = self._create_weight_bounds()
        
        # Run optimization
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=iterations,
            seed=seed,
            workers=1,  # Use single worker for reproducibility
            disp=True,
            tol=0.001
        )
        
        # Get best weights
        self.best_weights = self._weights_from_vector(result.x)
        self.best_score = -result.fun
        
        logger.info(f"Optimization complete. Best correlation: {self.best_score:.4f}")
        
        return self.best_weights


def test_rule_based_scorer():
    """Test function for rule-based scorer"""
    # Create sample player data
    players_data = pd.DataFrame([
        {
            'player_name': 'Haaland', 'position': 'FWD', 'team_name': 'Man City',
            'price': 15.0, 'total_points': 200, 'form': 8.5,
            'prob_goal': 0.65, 'prob_assist': 0.25, 'expected_goals': 0.8,
            'threat': 150, 'fixture_difficulty': 2, 'value_ratio': 13.3,
            'goal_involvement': 0.5, 'minutes': 90, 'chance_of_playing_next_round': 100
        },
        {
            'player_name': 'Salah', 'position': 'MID', 'team_name': 'Liverpool',
            'price': 13.0, 'total_points': 180, 'form': 7.2,
            'prob_goal': 0.45, 'prob_assist': 0.35, 'expected_goals': 0.5,
            'expected_assists': 0.3, 'creativity': 80, 'threat': 120,
            'fixture_difficulty': 3, 'value_ratio': 13.8, 'ict_per_90': 12.5,
            'minutes': 85, 'chance_of_playing_next_round': 100
        },
        {
            'player_name': 'Alexander-Arnold', 'position': 'DEF', 'team_name': 'Liverpool',
            'price': 8.0, 'total_points': 150, 'form': 6.5,
            'prob_clean_sheet': 0.45, 'prob_goal': 0.05, 'prob_assist': 0.20,
            'expected_assists': 0.15, 'threat': 40, 'fixture_difficulty': 3,
            'value_ratio': 18.75, 'team_elo': 1780, 'minutes': 90,
            'chance_of_playing_next_round': 100
        }
    ])
    
    # Test scorer
    scorer = RuleBasedScorer()
    
    # Score all players
    scored_players = scorer.score_all_players(players_data)
    print("Scored Players:")
    print(scored_players[['player_name', 'position', 'price', 'model_score', 'position_rank']])
    
    # Get top players
    top_mids = scorer.get_top_players(players_data, position='MID', top_n=5)
    print("\nTop Midfielders:")
    print(top_mids)
    
    # Test weight saving/loading
    scorer.save_weights('weights_test.json')
    scorer.load_weights('weights_test.json')
    
    return scored_players


if __name__ == "__main__":
    # Run test
    test_rule_based_scorer()