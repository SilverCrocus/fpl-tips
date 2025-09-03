"""
Advanced FPL Team Optimizer
Combines ML predictions with Mixed-Integer Linear Programming for optimal team selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import pulp
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from team optimization"""
    players: List[Dict]
    total_cost: float
    remaining_budget: float
    expected_points: float
    captain_id: int
    vice_captain_id: int
    formation: str
    optimization_time: float
    solver_status: str


class FeatureEngineer:
    """Advanced feature engineering for FPL predictions"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for ML models"""
        df = df.copy()
        
        # Momentum features
        if 'form' in df.columns and 'total_points' in df.columns:
            df['momentum'] = df['form'] / (df['total_points'] / df.get('gameweek', 20) + 1)
            df['form_percentile'] = df.groupby('position')['form'].rank(pct=True)
        
        # Value features
        if 'price' in df.columns and 'total_points' in df.columns:
            df['points_per_million'] = df['total_points'] / df['price']
            df['value_percentile'] = df['points_per_million'].rank(pct=True)
        
        # Fixture-weighted features
        if all(col in df.columns for col in ['fixture_difficulty', 'fixture_diff_next2', 'fixture_diff_next5']):
            # Lower is better for fixtures, so invert
            df['fixture_score'] = (6 - df['fixture_difficulty']) * 0.5 + \
                                  (6 - df['fixture_diff_next2']) * 0.3 + \
                                  (6 - df['fixture_diff_next5']) * 0.2
        
        # Position-specific features
        if 'position' in df.columns:
            # Goalkeepers
            if 'saves' in df.columns and 'minutes' in df.columns:
                df.loc[df['position'] == 'GK', 'saves_per_90'] = \
                    (df['saves'] / df['minutes']) * 90 
            
            # Defenders
            if all(col in df.columns for col in ['clean_sheets', 'goals_conceded']):
                df.loc[df['position'] == 'DEF', 'defensive_rating'] = \
                    df['clean_sheets'] * 4 - df['goals_conceded'] * 0.5
            
            # Midfielders & Forwards
            if all(col in df.columns for col in ['goals_scored', 'assists']):
                df['goal_involvement'] = df['goals_scored'] + df['assists']
                df['goal_involvement_rate'] = df['goal_involvement'] / (df['minutes'] / 90 + 1)
        
        # ICT composite features
        if all(col in df.columns for col in ['influence', 'creativity', 'threat']):
            df['ict_index'] = df['influence'] + df['creativity'] + df['threat']
            df['ict_per_million'] = df['ict_index'] / df['price']
        
        # Betting market insights
        if 'prob_goal' in df.columns and 'prob_assist' in df.columns:
            df['expected_involvement'] = df['prob_goal'] + df['prob_assist'] * 0.5
            
            # Position-adjusted probabilities
            position_weights = {'FWD': 1.2, 'MID': 1.0, 'DEF': 0.8, 'GK': 0.3}
            for pos, weight in position_weights.items():
                mask = df['position'] == pos
                df.loc[mask, 'position_adjusted_prob'] = df.loc[mask, 'expected_involvement'] * weight
        
        # Team strength features
        if 'team_elo' in df.columns and 'opponent_elo' in df.columns:
            df['elo_advantage'] = df['team_elo'] - df['opponent_elo']
            df['win_probability'] = 1 / (1 + 10 ** (-df['elo_advantage'] / 400))
        
        # Availability risk adjustment
        if 'chance_of_playing_next_round' in df.columns:
            df['availability_factor'] = df['chance_of_playing_next_round'] / 100
        else:
            df['availability_factor'] = 1.0
        
        # Historical consistency
        if 'total_points' in df.columns and 'gameweek' in df.columns:
            df['points_per_game'] = df['total_points'] / df.get('gameweek', 1).clip(lower=1)
            if 'form' in df.columns:
                df['consistency_score'] = df['form'] / (df['points_per_game'] + 1)
        
        return df
    
    def get_position_features(self, position: str) -> List[str]:
        """Get relevant features for each position"""
        base_features = [
            'form', 'total_points', 'minutes', 'price',
            'fixture_score', 'momentum', 'value_percentile',
            'availability_factor', 'elo_advantage'
        ]
        
        position_specific = {
            'GK': ['saves_per_90', 'prob_clean_sheet', 'clean_sheets'],
            'DEF': ['prob_clean_sheet', 'defensive_rating', 'expected_assists', 'threat'],
            'MID': ['expected_goals', 'expected_assists', 'creativity', 'ict_index', 'goal_involvement_rate'],
            'FWD': ['expected_goals', 'threat', 'shots_in_box', 'goal_involvement', 'position_adjusted_prob']
        }
        
        all_features = base_features + position_specific.get(position, [])
        return [f for f in all_features if f in self.available_features]
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit scaler and transform features"""
        df_engineered = self.engineer_features(df)
        self.available_features = df_engineered.columns.tolist()
        
        # Select numeric features only
        numeric_features = df_engineered.select_dtypes(include=[np.number]).columns
        return self.scaler.fit_transform(df_engineered[numeric_features])
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler"""
        df_engineered = self.engineer_features(df)
        numeric_features = df_engineered.select_dtypes(include=[np.number]).columns
        return self.scaler.transform(df_engineered[numeric_features])


class EnsemblePredictor:
    """Ensemble ML model for predicting player points"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        
    def build_models(self):
        """Initialize ensemble models"""
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42,
                verbose=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        
        # Initial equal weights
        self.weights = {name: 1/len(self.models) for name in self.models}
    
    def train(self, historical_data: pd.DataFrame, target_col: str = 'actual_points'):
        """Train ensemble models on historical data"""
        logger.info("Training ensemble models...")
        
        # Engineer features
        X = self.feature_engineer.fit_transform(historical_data)
        y = historical_data[target_col].values
        
        # Build models if not already built
        if not self.models:
            self.build_models()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        model_scores = {name: [] for name in self.models}
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train each model
            for name, model in self.models.items():
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                
                # Calculate validation score (negative MAE)
                mae = np.mean(np.abs(val_pred - y_val))
                model_scores[name].append(-mae)
        
        # Update weights based on validation performance
        avg_scores = {name: np.mean(scores) for name, scores in model_scores.items()}
        total_score = sum(np.exp(score) for score in avg_scores.values())
        self.weights = {name: np.exp(score)/total_score for name, score in avg_scores.items()}
        
        # Final training on all data
        for name, model in self.models.items():
            model.fit(X, y)
            logger.info(f"Trained {name} with weight {self.weights[name]:.3f}")
        
        self.is_trained = True
    
    def predict(self, players_df: pd.DataFrame, horizon: str = 'next_gw') -> np.ndarray:
        """Predict expected points for players"""
        if not self.is_trained:
            logger.warning("Models not trained, using default scoring")
            # Fallback to simple heuristic
            return self._heuristic_prediction(players_df, horizon)
        
        # Engineer features
        X = self.feature_engineer.transform(players_df)
        
        # Get predictions from each model
        predictions = []
        weights_list = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights_list.append(self.weights[name])
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
        
        if not predictions:
            return self._heuristic_prediction(players_df, horizon)
        
        # Weighted ensemble average
        ensemble_pred = np.average(predictions, weights=weights_list, axis=0)
        
        # Apply horizon multiplier
        if horizon == 'next_5_gws':
            ensemble_pred *= 5
        elif horizon == 'next_3_gws':
            ensemble_pred *= 3
        
        # Apply availability factor
        if 'availability_factor' in players_df.columns:
            ensemble_pred *= players_df['availability_factor'].values
        
        return ensemble_pred
    
    def _heuristic_prediction(self, players_df: pd.DataFrame, horizon: str) -> np.ndarray:
        """Simple heuristic prediction when ML models unavailable"""
        base_score = np.zeros(len(players_df))
        
        # Use form as primary indicator
        if 'form' in players_df.columns:
            base_score += players_df['form'].fillna(0).values * 1.5
        
        # Add total points contribution
        if 'total_points' in players_df.columns and 'gameweek' in players_df.columns:
            ppg = players_df['total_points'] / players_df['gameweek'].clip(lower=1)
            base_score += ppg.fillna(0).values
        
        # Fixture difficulty adjustment
        if 'fixture_difficulty' in players_df.columns:
            fixture_adj = (6 - players_df['fixture_difficulty'].fillna(3)) / 3
            base_score *= fixture_adj
        
        # Betting odds boost
        if 'prob_goal' in players_df.columns:
            base_score += players_df['prob_goal'].fillna(0).values * 5
        if 'prob_assist' in players_df.columns:
            base_score += players_df['prob_assist'].fillna(0).values * 3
        
        # Horizon adjustment
        multiplier = {'next_gw': 1, 'next_3_gws': 3, 'next_5_gws': 5}.get(horizon, 1)
        
        return base_score * multiplier


class MILPOptimizer:
    """Mixed-Integer Linear Programming optimizer for team selection"""
    
    def __init__(self):
        self.problem = None
        self.variables = {}
        self.solution = None
        
    def build_model(self, players_df: pd.DataFrame, budget: float = 100.0,
                   current_team: Optional[List[int]] = None,
                   free_transfers: int = 1) -> LpProblem:
        """Build MILP model for team selection"""
        
        # Create problem
        self.problem = LpProblem("FPL_Team_Selection", LpMaximize)
        
        players = players_df.index.tolist()
        n_players = len(players)
        
        # Decision variables
        self.variables['select'] = LpVariable.dicts('select', players, cat='Binary')
        self.variables['captain'] = LpVariable.dicts('captain', players, cat='Binary')
        self.variables['vice_captain'] = LpVariable.dicts('vice_captain', players, cat='Binary')
        self.variables['bench'] = LpVariable.dicts('bench', players, cat='Binary')
        
        # Transfer variables if current team provided
        if current_team:
            self.variables['transfer_out'] = LpVariable.dicts('out', current_team, cat='Binary')
            self.variables['transfer_in'] = LpVariable.dicts('in', players, cat='Binary')
        
        # Objective function: Maximize expected points
        expected_points = players_df['expected_points'].to_dict()
        
        # Base points + captain bonus (2x for captain)
        objective = lpSum([
            expected_points[i] * self.variables['select'][i] +
            expected_points[i] * self.variables['captain'][i]  # Captain gets double
            for i in players
        ])
        
        # Subtract bench points (they don't score unless substituted)
        objective -= lpSum([
            expected_points[i] * self.variables['bench'][i] * 0.9  # Small penalty for benching
            for i in players
        ])
        
        self.problem += objective
        
        # Add constraints
        self._add_constraints(players_df, budget, current_team, free_transfers)
        
        return self.problem
    
    def _add_constraints(self, players_df: pd.DataFrame, budget: float,
                         current_team: Optional[List[int]], free_transfers: int):
        """Add all FPL constraints to the model"""
        
        players = players_df.index.tolist()
        
        # 1. Budget constraint
        prices = players_df['price'].to_dict()
        self.problem += lpSum([
            prices[i] * self.variables['select'][i] for i in players
        ]) <= budget, "Budget"
        
        # 2. Squad size: exactly 15 players
        self.problem += lpSum([
            self.variables['select'][i] for i in players
        ]) == 15, "Squad_Size"
        
        # 3. Position constraints
        for position, (min_count, max_count) in [
            ('GK', (2, 2)), ('DEF', (5, 5)), ('MID', (5, 5)), ('FWD', (3, 3))
        ]:
            position_players = players_df[players_df['position'] == position].index.tolist()
            self.problem += lpSum([
                self.variables['select'][i] for i in position_players
            ]) == max_count, f"Position_{position}"
        
        # 4. Team constraint: max 3 players from same team
        teams = players_df['team_name'].unique()
        for team in teams:
            team_players = players_df[players_df['team_name'] == team].index.tolist()
            self.problem += lpSum([
                self.variables['select'][i] for i in team_players
            ]) <= 3, f"Team_{team}_Limit"
        
        # 5. Captain constraints
        self.problem += lpSum([
            self.variables['captain'][i] for i in players
        ]) == 1, "One_Captain"
        
        self.problem += lpSum([
            self.variables['vice_captain'][i] for i in players
        ]) == 1, "One_Vice_Captain"
        
        # Captain and vice must be selected players
        for i in players:
            self.problem += self.variables['captain'][i] <= self.variables['select'][i], f"Captain_Selected_{i}"
            self.problem += self.variables['vice_captain'][i] <= self.variables['select'][i], f"Vice_Selected_{i}"
            
        # Captain and vice must be different
        for i in players:
            self.problem += self.variables['captain'][i] + self.variables['vice_captain'][i] <= 1, f"Different_Captains_{i}"
        
        # 6. Bench constraints: exactly 4 bench players
        self.problem += lpSum([
            self.variables['bench'][i] for i in players
        ]) == 4, "Bench_Size"
        
        # Bench players must be selected
        for i in players:
            self.problem += self.variables['bench'][i] <= self.variables['select'][i], f"Bench_Selected_{i}"
        
        # 7. Valid formation for starting 11
        # Starting XI = selected - bench
        starters = {i: self.variables['select'][i] - self.variables['bench'][i] for i in players}
        
        # At least 1 GK must start
        gk_players = players_df[players_df['position'] == 'GK'].index.tolist()
        self.problem += lpSum([starters[i] for i in gk_players]) >= 1, "Min_GK_Start"
        
        # At least 3 DEF must start
        def_players = players_df[players_df['position'] == 'DEF'].index.tolist()
        self.problem += lpSum([starters[i] for i in def_players]) >= 3, "Min_DEF_Start"
        
        # At least 2 MID must start  
        mid_players = players_df[players_df['position'] == 'MID'].index.tolist()
        self.problem += lpSum([starters[i] for i in mid_players]) >= 2, "Min_MID_Start"
        
        # At least 1 FWD must start
        fwd_players = players_df[players_df['position'] == 'FWD'].index.tolist()
        self.problem += lpSum([starters[i] for i in fwd_players]) >= 1, "Min_FWD_Start"
        
        # Exactly 11 starters
        self.problem += lpSum([starters[i] for i in players]) == 11, "Starting_11"
        
        # 8. Transfer constraints if applicable
        if current_team:
            # Number of transfers
            num_transfers = lpSum([self.variables['transfer_out'][i] for i in current_team])
            
            # Transfer cost (4 points per transfer beyond free transfers)
            transfer_cost = 4 * (num_transfers - free_transfers)
            
            # Add transfer cost to objective (as negative points)
            # This would need to be incorporated into the objective function
    
    def solve(self, time_limit: int = 60) -> bool:
        """Solve the optimization problem"""
        if not self.problem:
            raise ValueError("Model not built. Call build_model first.")
        
        # Solve with time limit
        self.problem.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0))
        
        # Check if solution found
        return LpStatus[self.problem.status] == 'Optimal'
    
    def get_solution(self, players_df: pd.DataFrame) -> OptimizationResult:
        """Extract solution from solved model"""
        if not self.problem or LpStatus[self.problem.status] != 'Optimal':
            raise ValueError("No optimal solution available")
        
        selected_players = []
        captain_id = None
        vice_captain_id = None
        bench_players = []
        
        for i in players_df.index:
            if value(self.variables['select'][i]) == 1:
                player_data = players_df.loc[i].to_dict()
                player_data['is_captain'] = value(self.variables['captain'][i]) == 1
                player_data['is_vice_captain'] = value(self.variables['vice_captain'][i]) == 1
                player_data['is_bench'] = value(self.variables['bench'][i]) == 1
                
                if player_data['is_captain']:
                    captain_id = int(player_data['player_id'])
                if player_data['is_vice_captain']:
                    vice_captain_id = int(player_data['player_id'])
                
                selected_players.append(player_data)
        
        # Calculate total cost
        total_cost = sum(p['price'] for p in selected_players)
        
        # Calculate expected points
        starting_xi = [p for p in selected_players if not p['is_bench']]
        expected_points = sum(p['expected_points'] * (2 if p['is_captain'] else 1) 
                             for p in starting_xi)
        
        # Determine formation
        positions = {'DEF': 0, 'MID': 0, 'FWD': 0}
        for p in starting_xi:
            if p['position'] in positions:
                positions[p['position']] += 1
        formation = f"{positions['DEF']}-{positions['MID']}-{positions['FWD']}"
        
        return OptimizationResult(
            players=selected_players,
            total_cost=total_cost,
            remaining_budget=100.0 - total_cost,
            expected_points=expected_points,
            captain_id=captain_id,
            vice_captain_id=vice_captain_id,
            formation=formation,
            optimization_time=0.0,  # Would track this in solve()
            solver_status=LpStatus[self.problem.status]
        )


class FPLOptimizer:
    """Main optimizer combining ML predictions with MILP optimization"""
    
    def __init__(self):
        self.predictor = EnsemblePredictor()
        self.milp_optimizer = MILPOptimizer()
        self.feature_engineer = FeatureEngineer()
        
    def optimize_team(self, players_df: pd.DataFrame, 
                     strategy: str = 'balanced',
                     budget: float = 100.0,
                     current_team: Optional[List[int]] = None,
                     free_transfers: int = 1) -> OptimizationResult:
        """Complete optimization pipeline"""
        
        logger.info(f"Starting optimization with strategy: {strategy}")
        
        # Step 1: Feature engineering
        players_df = self.feature_engineer.engineer_features(players_df)
        
        # Step 2: Get ML predictions
        horizon_map = {
            'short_term': 'next_gw',
            'balanced': 'next_3_gws', 
            'long_term': 'next_5_gws'
        }
        horizon = horizon_map.get(strategy, 'next_3_gws')
        
        predictions = self.predictor.predict(players_df, horizon=horizon)
        players_df['expected_points'] = predictions
        
        # Step 3: Apply strategy adjustments
        players_df = self._apply_strategy(players_df, strategy)
        
        # Step 4: Filter candidates for efficiency
        players_df = self._filter_candidates(players_df, budget)
        
        # Step 5: Build and solve MILP model
        self.milp_optimizer.build_model(players_df, budget, current_team, free_transfers)
        
        if not self.milp_optimizer.solve(time_limit=30):
            logger.warning("MILP optimization failed, falling back to greedy approach")
            return self._greedy_selection(players_df, budget)
        
        # Step 6: Extract and return solution
        result = self.milp_optimizer.get_solution(players_df)
        logger.info(f"Optimization complete: {result.expected_points:.1f} expected points")
        
        return result
    
    def _apply_strategy(self, players_df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Apply strategy-specific adjustments to predictions"""
        df = players_df.copy()
        
        if strategy == 'differential':
            # Boost low-ownership high-performers
            if 'selected_by_percent' in df.columns:
                differential_boost = (100 - df['selected_by_percent']) / 100 * 0.2
                df['expected_points'] *= (1 + differential_boost)
        
        elif strategy == 'template':
            # Boost high-ownership players (safety)
            if 'selected_by_percent' in df.columns:
                template_boost = df['selected_by_percent'] / 100 * 0.1
                df['expected_points'] *= (1 + template_boost)
        
        elif strategy == 'fixture_focus':
            # Extra weight on fixtures
            if 'fixture_score' in df.columns:
                fixture_boost = df['fixture_score'] / 10 * 0.3
                df['expected_points'] *= (1 + fixture_boost)
        
        return df
    
    def _filter_candidates(self, players_df: pd.DataFrame, budget: float,
                          top_n: int = 200) -> pd.DataFrame:
        """Pre-filter candidates for computational efficiency"""
        
        # Remove injured/suspended players
        if 'chance_of_playing_next_round' in players_df.columns:
            players_df = players_df[players_df['chance_of_playing_next_round'] > 0]
        
        # Remove players way over budget
        max_player_budget = budget * 0.15  # No single player > 15% of budget
        players_df = players_df[players_df['price'] <= max_player_budget]
        
        # Keep top N by expected points per position
        filtered = []
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_players = players_df[players_df['position'] == position]
            # Keep more of positions we need more of
            n_keep = {'GK': 10, 'DEF': 40, 'MID': 40, 'FWD': 25}.get(position, 20)
            top_pos = pos_players.nlargest(n_keep, 'expected_points')
            filtered.append(top_pos)
        
        return pd.concat(filtered).reset_index(drop=True)
    
    def _greedy_selection(self, players_df: pd.DataFrame, budget: float) -> OptimizationResult:
        """Fallback greedy algorithm if MILP fails"""
        selected = []
        spent = 0.0
        team_counts = {}
        
        positions_needed = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
        # Sort by value (expected points per million)
        players_df['value_score'] = players_df['expected_points'] / players_df['price']
        
        for position, count in positions_needed.items():
            pos_players = players_df[
                (players_df['position'] == position) &
                (~players_df['player_id'].isin([p['player_id'] for p in selected]))
            ].sort_values('value_score', ascending=False)
            
            added = 0
            for _, player in pos_players.iterrows():
                if added >= count:
                    break
                
                team = player['team_name']
                if team_counts.get(team, 0) >= 3:
                    continue
                
                if spent + player['price'] > budget:
                    continue
                
                selected.append(player.to_dict())
                spent += player['price']
                team_counts[team] = team_counts.get(team, 0) + 1
                added += 1
        
        # Pick captain (highest expected points)
        captain = max(selected, key=lambda x: x['expected_points'])
        captain_id = int(captain['player_id'])
        
        # Pick vice captain (second highest)
        vice = max([p for p in selected if p['player_id'] != captain_id],
                  key=lambda x: x['expected_points'])
        vice_captain_id = int(vice['player_id'])
        
        return OptimizationResult(
            players=selected,
            total_cost=spent,
            remaining_budget=budget - spent,
            expected_points=sum(p['expected_points'] for p in selected),
            captain_id=captain_id,
            vice_captain_id=vice_captain_id,
            formation="4-4-2",  # Default formation
            optimization_time=0.0,
            solver_status="Greedy"
        )
    
    def optimize_transfers(self, current_team_df: pd.DataFrame,
                          all_players_df: pd.DataFrame,
                          free_transfers: int = 1,
                          max_hits: int = 1,
                          wildcard: bool = False) -> Dict[str, Any]:
        """Optimize transfers for existing team"""
        
        if wildcard:
            # Complete rebuild
            return self.optimize_team(all_players_df, strategy='balanced')
        
        # Calculate expected point gains from transfers
        current_ids = current_team_df['player_id'].tolist()
        
        # Get predictions for all players
        all_players_df = self.feature_engineer.engineer_features(all_players_df)
        predictions = self.predictor.predict(all_players_df, horizon='next_gw')
        all_players_df['expected_points'] = predictions
        
        # Find best transfer candidates
        transfer_candidates = []
        
        for _, current_player in current_team_df.iterrows():
            position = current_player['position']
            current_points = current_player.get('expected_points', 0)
            
            # Find potential replacements
            replacements = all_players_df[
                (all_players_df['position'] == position) &
                (~all_players_df['player_id'].isin(current_ids)) &
                (all_players_df['expected_points'] > current_points)
            ]
            
            for _, replacement in replacements.iterrows():
                gain = replacement['expected_points'] - current_points
                cost_diff = replacement['price'] - current_player['price']
                
                transfer_candidates.append({
                    'out': current_player['player_id'],
                    'out_name': current_player['player_name'],
                    'in': replacement['player_id'],
                    'in_name': replacement['player_name'],
                    'gain': gain,
                    'cost': cost_diff
                })
        
        # Sort by gain minus hit cost
        hit_cost = 4
        for t in transfer_candidates:
            t['net_gain'] = t['gain'] - (0 if free_transfers > 0 else hit_cost)
        
        transfer_candidates.sort(key=lambda x: x['net_gain'], reverse=True)
        
        # Select best transfers within constraints
        recommended_transfers = []
        budget_change = 0
        transfers_used = 0
        
        for transfer in transfer_candidates:
            if transfers_used >= free_transfers + max_hits:
                break
            
            if budget_change + transfer['cost'] > 0:  # Can't exceed budget
                continue
            
            recommended_transfers.append(transfer)
            budget_change += transfer['cost']
            transfers_used += 1
        
        return {
            'transfers': recommended_transfers,
            'expected_gain': sum(t['net_gain'] for t in recommended_transfers),
            'hits_taken': max(0, transfers_used - free_transfers),
            'cost': max(0, transfers_used - free_transfers) * hit_cost
        }


def create_optimizer(train_on_historical: bool = False) -> FPLOptimizer:
    """Factory function to create configured optimizer"""
    optimizer = FPLOptimizer()
    
    if train_on_historical:
        # Load historical data and train models
        # This would load from database or file
        pass
    
    return optimizer


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    
    from src.data.data_merger import DataMerger
    
    # Load data
    merger = DataMerger()
    players_df = merger.get_latest_data()
    
    if not players_df.empty:
        # Create and run optimizer
        optimizer = create_optimizer()
        result = optimizer.optimize_team(players_df, strategy='balanced')
        
        # Display results
        print(f"\nOptimal Team (Expected Points: {result.expected_points:.1f})")
        print(f"Formation: {result.formation}")
        print(f"Budget Used: £{result.total_cost:.1f}m")
        print(f"Remaining: £{result.remaining_budget:.1f}m")
        
        print("\n--- Starting XI ---")
        for p in [p for p in result.players if not p.get('is_bench', False)]:
            captain_mark = " (C)" if p.get('is_captain') else " (VC)" if p.get('is_vice_captain') else ""
            print(f"{p['position']:3} {p['player_name']:20} {p['team_name']:15} £{p['price']:.1f}m{captain_mark}")
        
        print("\n--- Bench ---")
        for p in [p for p in result.players if p.get('is_bench', False)]:
            print(f"{p['position']:3} {p['player_name']:20} {p['team_name']:15} £{p['price']:.1f}m")
    
    merger.close()