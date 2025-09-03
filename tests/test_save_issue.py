#!/usr/bin/env python
"""Test script to reproduce and fix the save_to_database issue"""

import asyncio
import pandas as pd
import sqlite3
from pathlib import Path
import logging
from src.data.fpl_api import FPLAPICollector
from src.data.odds_api import PlayerPropsCollector
from src.data.data_merger import DataMerger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_save_issue():
    """Test the save_to_database method with actual data"""
    
    # Get actual data
    async with FPLAPICollector() as fpl_collector:
        bootstrap = await fpl_collector.get_bootstrap_data()
        gameweek = 3  # Current gameweek
        gw_data = await fpl_collector.get_gameweek_data(gameweek)
        players_df = gw_data['players']
        fixtures_df = gw_data['fixtures']
    
    # Get odds data
    async with PlayerPropsCollector() as odds_collector:
        player_odds = await odds_collector.get_all_player_props_for_gameweek()
        if not player_odds.empty:
            player_odds = odds_collector.match_players_to_fpl(player_odds, players_df)
    
    # Create merger and unified dataset
    merger = DataMerger()
    
    unified_data = merger.create_unified_dataset(
        players_df,
        player_odds if not player_odds.empty else None,
        None,
        fixtures_df,
        gameweek=gameweek,
        season="2025-26"
    )
    
    logger.info(f"Unified data shape before save: {unified_data.shape}")
    logger.info(f"Unique player_id values: {unified_data['player_id'].nunique()}")
    
    # Check for duplicates
    dup_check = unified_data.duplicated(subset=['player_id', 'gameweek', 'season'])
    if dup_check.any():
        logger.warning(f"Found {dup_check.sum()} duplicates before save!")
    
    # Test the _clean_dataframe_for_db method
    cleaned_data = merger._clean_dataframe_for_db(unified_data)
    logger.info(f"Cleaned data shape: {cleaned_data.shape}")
    
    # Check for duplicates after cleaning
    dup_check2 = cleaned_data.duplicated(subset=['player_id', 'gameweek', 'season'])
    if dup_check2.any():
        logger.warning(f"Found {dup_check2.sum()} duplicates after cleaning!")
    
    # Now test the problematic code from save_to_database
    test_data = cleaned_data.copy()
    
    # This is the problematic code from lines 609-614
    for col in test_data.columns:
        first_val = test_data[col].iloc[0] if len(test_data) > 0 else None
        if isinstance(first_val, pd.Series):
            logger.warning(f"Column '{col}' contains Series objects")
            test_data[col] = test_data[col].apply(lambda x: x.iloc[0] if isinstance(x, pd.Series) else x)
    
    logger.info(f"Data shape after Series extraction: {test_data.shape}")
    
    # Check for duplicates after Series extraction
    dup_check3 = test_data.duplicated(subset=['player_id', 'gameweek', 'season'])
    if dup_check3.any():
        logger.warning(f"Found {dup_check3.sum()} duplicates after Series extraction!")
        # Show some examples
        dup_ids = test_data[dup_check3]['player_id'].unique()[:5]
        for pid in dup_ids:
            dup_records = test_data[test_data['player_id'] == pid]
            logger.warning(f"Player {pid} has {len(dup_records)} records")
    
    # Test actual save
    try:
        # First, manually delete existing records
        conn = sqlite3.connect("data/fpl_data.db")
        conn.execute("DELETE FROM player_gameweek_stats WHERE gameweek = 3 AND season = '2025-26'")
        conn.commit()
        logger.info("Manually deleted existing records")
        
        # Now try to insert
        test_data.to_sql(
            'player_gameweek_stats',
            conn,
            if_exists='append',
            index=False
        )
        logger.info("Successfully saved data!")
        conn.close()
    except Exception as e:
        logger.error(f"Save failed: {e}")
    
    merger.close()

if __name__ == "__main__":
    asyncio.run(test_save_issue())