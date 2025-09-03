#!/usr/bin/env python
"""Test transaction handling in save_to_database"""

import sqlite3
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a test dataframe similar to what we're trying to save
test_data = pd.DataFrame({
    'player_id': [1, 2, 3],
    'gameweek': [3, 3, 3],
    'season': ['2025-26', '2025-26', '2025-26'],
    'player_name': ['Player1', 'Player2', 'Player3']
})

# Connect to database
conn = sqlite3.connect("data/fpl_data.db")

# Method 1: Using execute with explicit transaction
logger.info("Method 1: Using explicit transaction with execute")
try:
    conn.execute("BEGIN TRANSACTION")
    
    # Delete existing
    result = conn.execute("DELETE FROM player_gameweek_stats WHERE gameweek = 3 AND season = '2025-26'")
    logger.info(f"Deleted {result.rowcount} rows")
    
    # Try to insert using to_sql
    test_data.to_sql('player_gameweek_stats', conn, if_exists='append', index=False)
    
    conn.commit()
    logger.info("Method 1: Success!")
except Exception as e:
    logger.error(f"Method 1 failed: {e}")
    conn.rollback()

# Method 2: Using isolation_level = None (autocommit off)
logger.info("\nMethod 2: Using isolation_level")
conn2 = sqlite3.connect("data/fpl_data.db", isolation_level=None)
try:
    conn2.execute("BEGIN")
    
    # Delete existing
    result = conn2.execute("DELETE FROM player_gameweek_stats WHERE gameweek = 3 AND season = '2025-26'")
    logger.info(f"Deleted {result.rowcount} rows")
    
    # Try to insert using to_sql
    test_data.to_sql('player_gameweek_stats', conn2, if_exists='append', index=False)
    
    conn2.execute("COMMIT")
    logger.info("Method 2: Success!")
except Exception as e:
    logger.error(f"Method 2 failed: {e}")
    conn2.execute("ROLLBACK")

# Method 3: Using cursor for delete, then to_sql
logger.info("\nMethod 3: Using cursor for delete")
conn3 = sqlite3.connect("data/fpl_data.db")
try:
    cursor = conn3.cursor()
    cursor.execute("BEGIN TRANSACTION")
    
    # Delete existing
    cursor.execute("DELETE FROM player_gameweek_stats WHERE gameweek = 3 AND season = '2025-26'")
    logger.info(f"Deleted {cursor.rowcount} rows")
    
    # Commit the delete first
    conn3.commit()
    
    # Now insert
    test_data.to_sql('player_gameweek_stats', conn3, if_exists='append', index=False)
    
    logger.info("Method 3: Success!")
except Exception as e:
    logger.error(f"Method 3 failed: {e}")
    conn3.rollback()

conn.close()
conn2.close()
conn3.close()