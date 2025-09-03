#!/usr/bin/env python
"""Test the exact query being used"""

import sqlite3
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

conn = sqlite3.connect("data/fpl_data.db")
cursor = conn.cursor()

gameweek = 3
season = "2025-26"
table = "player_gameweek_stats"

# Test the count query
count_query = f"SELECT COUNT(*) FROM {table} WHERE gameweek = ? AND season = ?"
logger.info(f"Executing: {count_query} with params: ({gameweek}, {season})")
cursor.execute(count_query, (gameweek, season))
result = cursor.fetchone()[0]
logger.info(f"Result: {result}")

# Test with different parameter types
logger.info("\nTesting with different parameter types:")

# As integers
cursor.execute(count_query, (3, "2025-26"))
logger.info(f"With int 3: {cursor.fetchone()[0]}")

# As strings
cursor.execute(count_query, ("3", "2025-26"))
logger.info(f"With string '3': {cursor.fetchone()[0]}")

# Check what types we're actually passing
import asyncio
from src.data.fpl_api import FPLAPICollector
from src.data.data_merger import DataMerger

async def check_types():
    async with FPLAPICollector() as fpl_collector:
        bootstrap = await fpl_collector.get_bootstrap_data()
        gw_data = await fpl_collector.get_gameweek_data(3)
        players_df = gw_data['players']
        
    merger = DataMerger()
    unified_data = merger.create_unified_dataset(
        players_df,
        None,
        None,
        None,
        gameweek=3,
        season="2025-26"
    )
    
    # Check types in the dataframe
    logger.info("\nDataframe types:")
    logger.info(f"gameweek type: {type(unified_data['gameweek'].iloc[0])}")
    logger.info(f"gameweek value: {unified_data['gameweek'].iloc[0]}")
    logger.info(f"season type: {type(unified_data['season'].iloc[0])}")
    logger.info(f"season value: {unified_data['season'].iloc[0]}")
    
    merger.close()

asyncio.run(check_types())
conn.close()