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
        self.conn = sqlite3.connect(self.db_path)
        
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
        merged = fpl_data.merge(
            odds_data[odds_columns],
            left_on='id',
            right_on='player_id',
            how='left',
            suffixes=('', '_odds')
        )
        
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
        data['fixture_difficulty'] = pd.cut(
            data['elo_diff'],
            bins=[-np.inf, -100, -50, 50, 100, np.inf],
            labels=[5, 4, 3, 2, 1]  # 5 = very hard, 1 = very easy
        ).astype(int)
        
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
                             gameweek: int = 1,
                             season: str = "2024-25") -> pd.DataFrame:
        """Create unified dataset from all sources
        
        Args:
            fpl_data: FPL player data
            odds_data: Betting odds data
            elo_data: Team Elo ratings
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
        
        # Select final columns
        final_columns = [
            'player_id', 'gameweek', 'season', 'player_name', 'position',
            'team_name', 'price', 'price_change', 'total_points', 'gameweek_points',
            'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
            'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'expected_goals', 'expected_assists', 'expected_goal_involvements',
            'form', 'selected_by_percent', 'transfers_in', 'transfers_out',
            'odds_goal', 'odds_assist', 'prob_goal', 'prob_assist',
            'team_elo', 'opponent_elo', 'fixture_difficulty',
            'value_ratio', 'points_per_million', 'recent_form_5', 'ownership_delta',
            'goal_involvement', 'ict_per_90', 'goals_vs_xg', 'assists_vs_xa',
            'is_available', 'data_quality', 'has_odds_data', 'has_elo_data'
        ]
        
        # Keep only available columns
        available_columns = [col for col in final_columns if col in unified.columns]
        unified = unified[available_columns]
        
        logger.info(f"Created unified dataset: {len(unified)} records, {len(unified.columns)} features")
        return unified
        
    def save_to_database(self, data: pd.DataFrame, table: str = "player_gameweek_stats"):
        """Save data to SQLite database with transaction management
        
        Args:
            data: DataFrame to save
            table: Table name
        """
        # Start transaction
        self.conn.execute("BEGIN TRANSACTION")
        
        try:
            data.to_sql(
                table,
                self.conn,
                if_exists='append',
                index=False
            )
            self.conn.commit()
            logger.info(f"Saved {len(data)} records to {table}")
        except sqlite3.IntegrityError as e:
            logger.warning(f"Integrity error (likely duplicates): {e}")
            # Rollback the failed insert
            self.conn.rollback()
            # Try updating existing records with new transaction
            self.conn.execute("BEGIN TRANSACTION")
            try:
                self._update_existing_records(data, table)
                self.conn.commit()
            except Exception as update_e:
                logger.error(f"Failed to update records: {update_e}")
                self.conn.rollback()
                raise
        except Exception as e:
            logger.error(f"Database error: {e}")
            self.conn.rollback()
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