"""
Data Merger Module
Combines FPL, odds, and Elo data into unified dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import sqlite3

logger = logging.getLogger(__name__)


class DataMerger:
    """Merges data from multiple sources into unified dataset"""
    
    def __init__(self, db_path: str = "data/fpl_data.db"):
        """Initialize data merger
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize SQLite database with tables"""
        self.conn = sqlite3.connect(self.db_path, isolation_level='DEFERRED')
        
        # Create tables
        queries = [
            """
            CREATE TABLE IF NOT EXISTS player_gameweek_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                gameweek INTEGER NOT NULL,
                season TEXT NOT NULL,
                -- Player info
                player_name TEXT,
                position TEXT,
                team INTEGER,
                team_name TEXT,
                team_short TEXT,
                price REAL,
                price_change REAL,
                -- FPL stats
                total_points INTEGER,
                gameweek_points INTEGER,
                minutes INTEGER,
                goals_scored INTEGER,
                assists INTEGER,
                clean_sheets INTEGER,
                goals_conceded INTEGER,
                saves INTEGER,
                bonus INTEGER,
                bps INTEGER,
                -- Advanced stats
                influence REAL,
                creativity REAL,
                threat REAL,
                ict_index REAL,
                expected_goals REAL,
                expected_assists REAL,
                expected_goal_involvements REAL,
                expected_goals_conceded REAL,
                -- Form and selection
                form REAL,
                selected_by_percent REAL,
                transfers_in INTEGER,
                transfers_out INTEGER,
                -- Betting odds
                odds_goal REAL,
                odds_assist REAL,
                odds_clean_sheet REAL,
                prob_goal REAL,
                prob_assist REAL,
                prob_clean_sheet REAL,
                -- Team context
                team_elo REAL,
                opponent_elo REAL,
                elo_diff REAL,
                fixture_difficulty INTEGER,
                fixture_diff_next2 REAL,
                fixture_diff_next5 REAL,
                easy_fixtures_next5 INTEGER,
                hard_fixtures_next5 INTEGER,
                is_home INTEGER,
                -- Derived features
                value_ratio REAL,
                points_per_million REAL,
                recent_form_5 REAL,
                ownership_delta REAL,
                goal_involvement REAL,
                ict_per_90 REAL,
                goals_vs_xg REAL,
                assists_vs_xa REAL,
                is_available INTEGER,
                is_goalkeeper INTEGER,
                is_defender INTEGER,
                is_midfielder INTEGER,
                is_forward INTEGER,
                -- Quality flags
                data_quality REAL,
                has_odds_data INTEGER,
                has_elo_data INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_id, gameweek, season)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_player_gameweek 
            ON player_gameweek_stats(player_id, gameweek, season)
            """,
            """
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                short_name TEXT,
                elo_rating REAL,
                updated_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS fixtures (
                id INTEGER PRIMARY KEY,
                gameweek INTEGER,
                season TEXT,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_score INTEGER,
                away_score INTEGER,
                kickoff_time TIMESTAMP,
                finished INTEGER,
                home_difficulty INTEGER,
                away_difficulty INTEGER
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS elo_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER,
                gameweek INTEGER,
                season TEXT,
                elo_rating REAL,
                UNIQUE(team_id, gameweek, season)
            )
            """
        ]
        
        for query in queries:
            self.conn.execute(query)
        self.conn.commit()
        
        logger.info(f"Database initialized at {self.db_path}")
        
    def merge_fpl_and_odds(self, fpl_data: pd.DataFrame, 
                          odds_data: pd.DataFrame) -> pd.DataFrame:
        """Merge FPL data with betting odds
        
        Args:
            fpl_data: DataFrame with FPL player data
            odds_data: DataFrame with betting odds
            
        Returns:
            Merged DataFrame
        """
        # Ensure player_id exists in both datasets
        if 'player_id' not in odds_data.columns and 'player' in odds_data.columns:
            # Try to match by player name if ID not available
            player_mapping = dict(zip(fpl_data['web_name'], fpl_data['id']))
            odds_data['player_id'] = odds_data['player'].map(player_mapping)
            
        # Select columns from odds_data that exist (handle both old and new naming)
        odds_columns = ['player_id'] if 'player_id' in odds_data.columns else []
        
        # Handle real odds from PlayerPropsCollector
        if 'odds_goal_anytime' in odds_data.columns:
            odds_data['odds_goal'] = odds_data['odds_goal_anytime']
        if 'odds_goal' in odds_data.columns:
            odds_columns.append('odds_goal')
        if 'prob_goal' in odds_data.columns:
            odds_columns.append('prob_goal')
        if 'odds_assist' in odds_data.columns:
            odds_columns.append('odds_assist')
        if 'prob_assist' in odds_data.columns:
            odds_columns.append('prob_assist')
        if 'is_real_odds' in odds_data.columns:
            odds_columns.append('is_real_odds')
        
        # Merge on player_id
        if 'player_id' in odds_data.columns:
            # If odds has player_id, merge and drop the duplicate
            merge_cols = [col for col in odds_columns if col != 'player_id']
            merged = fpl_data.merge(
                odds_data[['player_id'] + merge_cols],
                left_on='id',
                right_on='player_id',
                how='left',
                suffixes=('', '_odds')
            )
            # Drop the duplicate player_id column from odds
            merged = merged.drop('player_id', axis=1)
        else:
            # No player_id column in odds
            merged = fpl_data
        
        # Mark players with real odds vs those without
        # DO NOT fill NaN values - we want to exclude players without real odds
        if 'is_real_odds' in merged.columns:
            merged['has_real_odds'] = merged['is_real_odds'].fillna(False)
        elif 'odds_goal' in merged.columns:
            # If no real odds column, check if odds_goal exists and is not null
            merged['has_real_odds'] = merged['odds_goal'].notna()
        else:
            # No odds data at all
            merged['has_real_odds'] = False
        
        logger.info(f"Merged FPL and odds data: {len(merged)} records")
        return merged
        
    def add_elo_ratings(self, data: pd.DataFrame, 
                       elo_data: pd.DataFrame) -> pd.DataFrame:
        """Add Elo ratings to player data
        
        Args:
            data: Player data DataFrame
            elo_data: Elo ratings DataFrame
            
        Returns:
            DataFrame with Elo ratings
        """
        # Create team Elo mapping
        if 'team_name' in elo_data.columns and 'elo_rating' in elo_data.columns:
            team_elo_map = dict(zip(elo_data['team_name'], elo_data['elo_rating']))
            
            # Add team Elo
            data['team_elo'] = data['team_name'].map(team_elo_map)
            
            # For opponent Elo, we need fixture information
            # This would be enhanced with actual fixture data
            data['opponent_elo'] = data['team_elo'].mean()  # Placeholder
            
        else:
            # Use default Elo ratings
            logger.warning("Elo data not available, using default values")
            data['team_elo'] = 1500
            data['opponent_elo'] = 1500
            
        logger.info(f"Added Elo ratings to {len(data)} records")
        return data
        
    def add_fixture_difficulty(self, data: pd.DataFrame, 
                               fixtures_data: pd.DataFrame,
                               current_gameweek: int) -> pd.DataFrame:
        """Add fixture difficulty ratings for upcoming games
        
        Args:
            data: Player data DataFrame
            fixtures_data: Fixtures DataFrame with difficulty ratings
            current_gameweek: Current gameweek number
            
        Returns:
            DataFrame with fixture difficulty columns
        """
        # Create team ID mapping from player data
        team_mapping = {}
        if 'team' in data.columns and 'team_name' in data.columns:
            team_data = data[['team', 'team_name']].drop_duplicates()
            team_mapping = dict(zip(team_data['team_name'], team_data['team']))
        
        # Initialize difficulty columns
        data['fixture_difficulty'] = 3  # Default medium
        data['fixture_diff_next2'] = 3.0
        data['fixture_diff_next5'] = 3.0
        data['easy_fixtures_next5'] = 0  # Count of easy fixtures (diff <= 2)
        data['hard_fixtures_next5'] = 0  # Count of hard fixtures (diff >= 4)
        
        # Process fixtures for each team
        for team_name, team_id in team_mapping.items():
            # Get team's upcoming fixtures
            team_fixtures_home = fixtures_data[
                (fixtures_data['team_h'] == team_id) & 
                (fixtures_data['event'] >= current_gameweek) &
                (fixtures_data['event'] <= current_gameweek + 5)
            ].copy()
            
            team_fixtures_away = fixtures_data[
                (fixtures_data['team_a'] == team_id) & 
                (fixtures_data['event'] >= current_gameweek) &
                (fixtures_data['event'] <= current_gameweek + 5)
            ].copy()
            
            # Add difficulty for home games
            team_fixtures_home['difficulty'] = team_fixtures_home['team_h_difficulty']
            team_fixtures_home['is_home'] = 1
            
            # Add difficulty for away games
            team_fixtures_away['difficulty'] = team_fixtures_away['team_a_difficulty']
            team_fixtures_away['is_home'] = 0
            
            # Combine home and away fixtures
            all_fixtures = pd.concat([team_fixtures_home, team_fixtures_away])
            all_fixtures = all_fixtures.sort_values('event')
            
            if not all_fixtures.empty:
                # Get difficulty ratings
                difficulties = all_fixtures['difficulty'].values
                
                # Current gameweek difficulty
                current_gw_fixtures = all_fixtures[all_fixtures['event'] == current_gameweek]
                if not current_gw_fixtures.empty:
                    current_diff = current_gw_fixtures['difficulty'].iloc[0]
                    is_home = current_gw_fixtures['is_home'].iloc[0]
                    data.loc[data['team_name'] == team_name, 'fixture_difficulty'] = int(current_diff)
                    data.loc[data['team_name'] == team_name, 'is_home'] = int(is_home)
                
                # Next 2 gameweeks average
                if len(difficulties) >= 2:
                    data.loc[data['team_name'] == team_name, 'fixture_diff_next2'] = float(np.mean(difficulties[:2]))
                elif len(difficulties) > 0:
                    data.loc[data['team_name'] == team_name, 'fixture_diff_next2'] = float(difficulties[0])
                
                # Next 5 gameweeks average
                if len(difficulties) > 0:
                    data.loc[data['team_name'] == team_name, 'fixture_diff_next5'] = float(np.mean(difficulties[:5]))
                    # Count easy and hard fixtures
                    easy_count = sum(1 for d in difficulties[:5] if d <= 2)
                    hard_count = sum(1 for d in difficulties[:5] if d >= 4)
                    data.loc[data['team_name'] == team_name, 'easy_fixtures_next5'] = int(easy_count)
                    data.loc[data['team_name'] == team_name, 'hard_fixtures_next5'] = int(hard_count)
        
        logger.info(f"Added fixture difficulty for {len(team_mapping)} teams")
        return data
        
    def calculate_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features for modeling
        
        Args:
            data: DataFrame with merged data
            
        Returns:
            DataFrame with additional features
        """
        # Ensure numeric columns are actually numeric
        numeric_cols = ['total_points', 'price', 'points_per_game', 'form', 'minutes', 
                       'goals_scored', 'assists', 'expected_goals', 'expected_assists',
                       'ict_index', 'selected_by_percent']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Value metrics - safe division with minimum price
        safe_price = data['price'].clip(lower=0.1)  # Ensure minimum price of 0.1
        data['value_ratio'] = data['total_points'] / safe_price
        if 'points_per_game' in data.columns:
            data['points_per_million'] = data['points_per_game'] / safe_price
        else:
            data['points_per_million'] = data['total_points'] / safe_price
        
        # Form metrics (would be better with historical data)
        data['recent_form_5'] = data['form'].fillna(0)
        
        # Ownership delta (change in ownership)
        data['ownership_delta'] = data.groupby('id')['selected_by_percent'].diff()
        
        # Fixture difficulty from Elo difference
        data['elo_diff'] = data['team_elo'] - data['opponent_elo']
        # Convert pd.cut categorical to regular integers to avoid Series type in database
        fixture_diff_categorical = pd.cut(
            data['elo_diff'],
            bins=[-np.inf, -100, -50, 50, 100, np.inf],
            labels=[5, 4, 3, 2, 1]  # 5 = very hard, 1 = very easy
        )
        # Convert to regular integers, not categorical
        data['fixture_difficulty'] = fixture_diff_categorical.astype('int64')
        
        # Position-specific features
        data['is_goalkeeper'] = (data['position'] == 'GK').astype(int)
        data['is_defender'] = (data['position'] == 'DEF').astype(int)
        data['is_midfielder'] = (data['position'] == 'MID').astype(int)
        data['is_forward'] = (data['position'] == 'FWD').astype(int)
        
        # Clean sheet probability - only use if real data available
        # NO ESTIMATION - leave as NaN if not available from real odds
            
        # Goal involvement rate - safe division
        minutes_played = data['minutes'].fillna(0).clip(lower=1)  # At least 1 minute
        games_played = (minutes_played / 90).clip(lower=0.1)  # At least 0.1 games
        data['goal_involvement'] = (
            data['goals_scored'].fillna(0) + data['assists'].fillna(0)
        ) / games_played
        
        # ICT per 90 - safe division
        data['ict_per_90'] = data['ict_index'].fillna(0) / games_played
        
        # Expected vs actual performance
        data['goals_vs_xg'] = data['goals_scored'] - data['expected_goals'].fillna(0)
        data['assists_vs_xa'] = data['assists'] - data['expected_assists'].fillna(0)
        
        # Availability flag
        data['is_available'] = (
            data['chance_of_playing_next_round'].fillna(100) > 75
        ).astype(int)
        
        # Data quality score
        data['data_quality'] = self._calculate_data_quality(data)
        
        logger.info(f"Calculated {len(data.columns)} total features")
        return data
        
    def _calculate_data_quality(self, data: pd.DataFrame) -> pd.Series:
        """Calculate data quality score for each record
        
        Args:
            data: DataFrame with player data
            
        Returns:
            Series with quality scores (0-1)
        """
        quality_score = pd.Series(1.0, index=data.index)
        
        # Check for missing critical fields
        critical_fields = ['price', 'total_points', 'minutes', 'position']
        for field in critical_fields:
            if field in data.columns:
                quality_score -= 0.2 * data[field].isna()
                
        # Check for odds data
        if 'odds_goal' in data.columns:
            quality_score += 0.1 * data['odds_goal'].notna()
        else:
            quality_score -= 0.1
            
        # Check for Elo data
        if 'team_elo' in data.columns:
            quality_score += 0.1 * (data['team_elo'] != 1500)  # Not default
        else:
            quality_score -= 0.1
            
        # Ensure score is between 0 and 1
        quality_score = quality_score.clip(0, 1)
        
        return quality_score
        
    def create_unified_dataset(self, fpl_data: pd.DataFrame,
                             odds_data: Optional[pd.DataFrame] = None,
                             elo_data: Optional[pd.DataFrame] = None,
                             fixtures_data: Optional[pd.DataFrame] = None,
                             gameweek: int = 1,
                             season: str = "2024-25") -> pd.DataFrame:
        """Create unified dataset from all sources
        
        Args:
            fpl_data: FPL player data
            odds_data: Betting odds data
            elo_data: Team Elo ratings
            fixtures_data: Fixtures with difficulty ratings
            gameweek: Gameweek number
            season: Season identifier
            
        Returns:
            Unified DataFrame
        """
        # Start with FPL data
        unified = fpl_data.copy()
        
        # Add gameweek and season
        unified['gameweek'] = gameweek
        unified['season'] = season
        
        # Merge odds data if available
        if odds_data is not None and not odds_data.empty:
            unified = self.merge_fpl_and_odds(unified, odds_data)
            unified['has_odds_data'] = 1
        else:
            unified['has_odds_data'] = 0
            # Add placeholder columns
            for col in ['odds_goal', 'odds_assist', 'prob_goal', 'prob_assist']:
                unified[col] = np.nan
                
        # Add Elo ratings if available
        if elo_data is not None and not elo_data.empty:
            unified = self.add_elo_ratings(unified, elo_data)
            unified['has_elo_data'] = 1
        else:
            unified['has_elo_data'] = 0
            unified['team_elo'] = 1500
            unified['opponent_elo'] = 1500
            
        # Add fixture difficulty if available
        if fixtures_data is not None and not fixtures_data.empty:
            unified = self.add_fixture_difficulty(unified, fixtures_data, gameweek)
        else:
            # Default to medium difficulty if no fixture data
            unified['fixture_difficulty'] = 3
            unified['fixture_diff_next2'] = 3.0  # Average of next 2 games
            unified['fixture_diff_next5'] = 3.0  # Average of next 5 games
            unified['opponent_team'] = 'Unknown'
            unified['is_home'] = 0
            
        # Calculate derived features
        unified = self.calculate_derived_features(unified)
        
        # Rename columns for consistency
        column_mapping = {
            'id': 'player_id',
            'web_name': 'player_name',
            'points_per_game': 'avg_points',
            'event_points': 'gameweek_points'
        }
        
        unified = unified.rename(columns=column_mapping)
        
        # Add missing columns that database expects
        if 'gameweek_points' not in unified.columns and 'event_points' in unified.columns:
            unified['gameweek_points'] = unified['event_points']
        elif 'gameweek_points' not in unified.columns:
            unified['gameweek_points'] = 0
            
        if 'team_short' not in unified.columns:
            unified['team_short'] = unified['team_name'].str[:3].str.upper() if 'team_name' in unified.columns else ''
            
        # Add position flags
        if 'position' in unified.columns:
            unified['is_goalkeeper'] = (unified['position'] == 'GK').astype(int)
            unified['is_defender'] = (unified['position'] == 'DEF').astype(int)
            unified['is_midfielder'] = (unified['position'] == 'MID').astype(int)
            unified['is_forward'] = (unified['position'] == 'FWD').astype(int)
        else:
            unified['is_goalkeeper'] = 0
            unified['is_defender'] = 0
            unified['is_midfielder'] = 0
            unified['is_forward'] = 0
            
        # Add expected_goals_conceded if missing
        if 'expected_goals_conceded' not in unified.columns:
            unified['expected_goals_conceded'] = 0.0
            
        # Add elo_diff
        if 'team_elo' in unified.columns and 'opponent_elo' in unified.columns:
            unified['elo_diff'] = unified['team_elo'] - unified['opponent_elo']
        else:
            unified['elo_diff'] = 0.0
            
        # Add odds columns if missing
        if 'odds_assist' not in unified.columns:
            unified['odds_assist'] = np.nan
        if 'prob_assist' not in unified.columns:
            unified['prob_assist'] = np.nan
        if 'odds_clean_sheet' not in unified.columns:
            unified['odds_clean_sheet'] = np.nan
        if 'prob_clean_sheet' not in unified.columns:
            unified['prob_clean_sheet'] = np.nan
        
        # Select final columns
        final_columns = [
            'player_id', 'gameweek', 'season', 'player_name', 'position',
            'team', 'team_name', 'team_short', 'price', 'price_change', 'total_points', 'gameweek_points',
            'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
            'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
            'form', 'selected_by_percent', 'transfers_in', 'transfers_out',
            'odds_goal', 'odds_assist', 'odds_clean_sheet', 'prob_goal', 'prob_assist', 'prob_clean_sheet',
            'team_elo', 'opponent_elo', 'elo_diff', 'fixture_difficulty', 'fixture_diff_next2',
            'fixture_diff_next5', 'easy_fixtures_next5', 'hard_fixtures_next5',
            'value_ratio', 'points_per_million', 'recent_form_5', 'ownership_delta',
            'goal_involvement', 'ict_per_90', 'goals_vs_xg', 'assists_vs_xa',
            'is_available', 'is_goalkeeper', 'is_defender', 'is_midfielder', 'is_forward',
            'data_quality', 'has_odds_data', 'has_elo_data', 'is_home'
        ]
        
        # Keep only available columns
        available_columns = [col for col in final_columns if col in unified.columns]
        unified = unified[available_columns]
        
        logger.info(f"Created unified dataset: {len(unified)} records, {len(unified.columns)} features")
        return unified
        
    def _clean_dataframe_for_db(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame to ensure all values are database-compatible
        
        Args:
            data: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = data.copy()
        
        # Ensure no Series or complex objects in cells
        for col in df_clean.columns:
            try:
                # Convert categorical to regular types (from pd.cut, etc.)
                if pd.api.types.is_categorical_dtype(df_clean[col]):
                    # Check if categories are numeric
                    if pd.api.types.is_numeric_dtype(df_clean[col].cat.categories):
                        df_clean[col] = df_clean[col].astype('int64')
                    else:
                        df_clean[col] = df_clean[col].astype('object')
                        
                # Check for any Series values in the column
                if df_clean[col].apply(lambda x: isinstance(x, pd.Series)).any():
                    df_clean[col] = df_clean[col].apply(
                        lambda x: x.iloc[0] if isinstance(x, pd.Series) and len(x) > 0 else x
                    )
                    
                # Convert numpy types
                if df_clean[col].dtype.name.startswith('int'):
                    df_clean[col] = df_clean[col].astype(int, errors='ignore')
                elif df_clean[col].dtype.name.startswith('float'):
                    df_clean[col] = df_clean[col].astype(float, errors='ignore')
                    
            except Exception as e:
                logger.debug(f"Could not clean column {col}: {e}")
                pass
                
        return df_clean
    
    def save_to_database(self, data: pd.DataFrame, table: str = "player_gameweek_stats"):
        """Save data to SQLite database with transaction management
        
        Args:
            data: DataFrame to save
            table: Table name
        """
        # Clean data for database compatibility
        data = self._clean_dataframe_for_db(data)
        
        # Make sure we have the key columns
        if 'gameweek' not in data.columns or 'season' not in data.columns:
            logger.warning("Missing gameweek or season columns, cannot save to database")
            return
            
        # Check for duplicate column names
        if data.columns.duplicated().any():
            logger.warning(f"Duplicate columns found: {data.columns[data.columns.duplicated()].tolist()}")
            # Drop duplicate columns
            data = data.loc[:, ~data.columns.duplicated(keep='first')]
            logger.info(f"Data shape after removing duplicate columns: {data.shape}")
        
        # Remove any duplicate player_id records (keep first)
        if 'player_id' in data.columns:
            before_count = len(data)
            data = data.drop_duplicates(subset=['player_id', 'gameweek', 'season'], keep='first')
            if before_count != len(data):
                logger.warning(f"Removed {before_count - len(data)} duplicate player records")
            logger.info(f"Saving {len(data)} unique records to database")
            
        try:
            # Ensure no pending transactions
            try:
                self.conn.commit()
            except:
                pass
                
            # Delete existing records for this gameweek/season to avoid duplicates
            if 'player_id' in data.columns and not data.empty:
                gameweek = data['gameweek'].iloc[0]
                season = data['season'].iloc[0]
                
                # Ensure gameweek is an integer for proper comparison
                if isinstance(gameweek, (np.integer, np.floating)):
                    gameweek = int(gameweek)
                elif isinstance(gameweek, str) and gameweek.isdigit():
                    gameweek = int(gameweek)
                
                # Use a cursor for operations
                cursor = self.conn.cursor()
                
                # Check how many records exist for this gameweek/season
                count_query = f"SELECT COUNT(*) FROM {table} WHERE gameweek = ? AND season = ?"
                cursor.execute(count_query, (gameweek, season))
                existing_count = cursor.fetchone()[0]
                
                if existing_count > 0:
                    logger.debug(f"Found {existing_count} existing records for gameweek={gameweek}, season={season}")
                
                # Always try to delete, even if count returns 0
                delete_query = f"""
                    DELETE FROM {table}
                    WHERE gameweek = ? AND season = ?
                """
                cursor.execute(delete_query, (gameweek, season))
                actual_deleted = cursor.rowcount
                self.conn.commit()
                
                if actual_deleted > 0:
                    logger.info(f"Deleted {actual_deleted} existing records for gameweek {gameweek}, season {season}")
                else:
                    logger.info(f"No existing records deleted for gameweek {gameweek}, season {season}")
                    
                # Double-check after delete
                cursor.execute(count_query, (gameweek, season))
                remaining_count = cursor.fetchone()[0]
                if remaining_count > 0:
                    logger.warning(f"WARNING: {remaining_count} records still exist after delete!")
                
                # Get table columns (excluding auto-generated ones)
                cursor.execute(f"PRAGMA table_info({table})")
                db_columns = [col[1] for col in cursor.fetchall() if col[1] not in ['id', 'created_at']]
                
                # Filter dataframe to only include columns that exist in database
                insert_data = data[[col for col in db_columns if col in data.columns]].copy()
                
                # Replace NaN values with None for SQLite
                insert_data = insert_data.where(pd.notnull(insert_data), None)
                
                # Build insert query
                placeholders = ','.join(['?' for _ in insert_data.columns])
                columns = ','.join(insert_data.columns)
                insert_query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                
                # Insert all records
                cursor.executemany(insert_query, insert_data.values.tolist())
                rows_inserted = cursor.rowcount
                
                # Commit the insert
                self.conn.commit()
                cursor.close()
                
                logger.info(f"Saved {rows_inserted} records to {table}")
        except Exception as e:
            logger.error(f"Database error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            # Try to rollback if possible
            try:
                self.conn.rollback()
            except:
                pass
            raise
            
    def _update_existing_records(self, data: pd.DataFrame, table: str):
        """Update existing records in database
        
        Args:
            data: DataFrame with updated data
            table: Table name (must be a valid table)
        """
        # Whitelist of allowed table names to prevent SQL injection
        allowed_tables = ['player_gameweek_stats', 'player_info', 'fixtures', 'teams']
        if table not in allowed_tables:
            raise ValueError(f"Invalid table name: {table}. Must be one of: {allowed_tables}")
        
        # Clean data before update
        data = self._clean_dataframe_for_db(data)
            
        for _, row in data.iterrows():
            # Build update query with parameterized columns
            update_cols = [f"{col} = ?" for col in data.columns if col not in ['player_id', 'gameweek', 'season']]
            update_query = f"""
                UPDATE {table}
                SET {', '.join(update_cols)}
                WHERE player_id = ? AND gameweek = ? AND season = ?
            """
            
            # Get values - ensure scalar values not Series
            values = []
            for col in data.columns:
                if col not in ['player_id', 'gameweek', 'season']:
                    val = row[col]
                    # Convert Series to scalar if needed
                    if isinstance(val, pd.Series):
                        val = val.iloc[0] if len(val) > 0 else None
                    # Handle numpy types
                    elif hasattr(val, 'item'):  # numpy scalar
                        val = val.item()
                    # Handle numpy.int64, numpy.float64, etc
                    elif type(val).__module__ == 'numpy':
                        val = val.item() if hasattr(val, 'item') else float(val)
                    # Convert pandas NA to None for SQLite
                    elif pd.isna(val):
                        val = None
                    # Ensure not a complex type
                    elif not isinstance(val, (str, int, float, bool, type(None))):
                        val = str(val)  # Convert to string as last resort
                    values.append(val)
            values.extend([row['player_id'], row['gameweek'], row['season']])
            
            self.conn.execute(update_query, values)
            
        # Commit is handled by the caller within transaction
        logger.info(f"Updated {len(data)} records in {table}")
        
    def load_from_database(self, gameweek: Optional[int] = None,
                          season: Optional[str] = None) -> pd.DataFrame:
        """Load data from database
        
        Args:
            gameweek: Specific gameweek (None for all)
            season: Specific season (None for all)
            
        Returns:
            DataFrame with loaded data
        """
        query = "SELECT * FROM player_gameweek_stats WHERE 1=1"
        params = []
        
        if gameweek:
            query += " AND gameweek = ?"
            params.append(gameweek)
            
        if season:
            query += " AND season = ?"
            params.append(season)
            
        data = pd.read_sql_query(query, self.conn, params=params)
        logger.info(f"Loaded {len(data)} records from database")
        return data
        
    def get_latest_data(self, top_n: int = 100) -> pd.DataFrame:
        """Get latest player data sorted by total points
        
        Args:
            top_n: Number of top players to return
            
        Returns:
            DataFrame with top players
        """
        query = """
            SELECT * FROM player_gameweek_stats
            WHERE (player_id, gameweek) IN (
                SELECT player_id, MAX(gameweek) 
                FROM player_gameweek_stats
                GROUP BY player_id
            )
            ORDER BY total_points DESC
            LIMIT ?
        """
        
        data = pd.read_sql_query(query, self.conn, params=[top_n])
        return data
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def test_data_merger():
    """Test function for data merger"""
    # Create sample data
    fpl_data = pd.DataFrame([
        {
            'id': 1, 'web_name': 'Haaland', 'position': 'FWD',
            'team_name': 'Man City', 'price': 15.0, 'total_points': 200,
            'minutes': 2000, 'goals_scored': 25, 'assists': 5,
            'expected_goals': 22.5, 'expected_assists': 6.2,
            'ict_index': 450.2, 'form': 8.5, 'selected_by_percent': 65.2,
            'chance_of_playing_next_round': 100
        },
        {
            'id': 2, 'web_name': 'Salah', 'position': 'MID',
            'team_name': 'Liverpool', 'price': 13.0, 'total_points': 180,
            'minutes': 2100, 'goals_scored': 15, 'assists': 10,
            'expected_goals': 14.3, 'expected_assists': 8.7,
            'ict_index': 380.5, 'form': 7.2, 'selected_by_percent': 45.8,
            'chance_of_playing_next_round': 100
        }
    ])
    
    odds_data = pd.DataFrame([
        {
            'player': 'Haaland', 'player_id': 1,
            'odds': {'anytime_scorer': 1.50, 'assist': 3.50},
            'probabilities': {'goal': 0.625, 'assist': 0.25}
        },
        {
            'player': 'Salah', 'player_id': 2,
            'odds': {'anytime_scorer': 2.10, 'assist': 2.80},
            'probabilities': {'goal': 0.45, 'assist': 0.33}
        }
    ])
    
    elo_data = pd.DataFrame([
        {'team_name': 'Man City', 'elo_rating': 1850},
        {'team_name': 'Liverpool', 'elo_rating': 1780}
    ])
    
    # Test merger
    merger = DataMerger()
    
    # Create unified dataset
    unified = merger.create_unified_dataset(
        fpl_data, odds_data, elo_data,
        gameweek=10, season="2024-25"
    )
    
    print("Unified Dataset Shape:", unified.shape)
    print("\nSample Features:")
    print(unified[['player_name', 'position', 'price', 'total_points', 
                   'prob_goal', 'team_elo', 'value_ratio', 'data_quality']].head())
    
    # Save to database
    merger.save_to_database(unified)
    
    # Load back from database
    loaded = merger.load_from_database(gameweek=10, season="2024-25")
    print(f"\nLoaded {len(loaded)} records from database")
    
    merger.close()
    return unified


if __name__ == "__main__":
    # Run test
    test_data_merger()