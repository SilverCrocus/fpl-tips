#!/usr/bin/env python
"""Test connection state issue"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a connection like DataMerger does
db_path = "data/fpl_data.db"
conn = sqlite3.connect(db_path)

# Try the same operations
cursor = conn.cursor()

# Count records
count_query = "SELECT COUNT(*) FROM player_gameweek_stats WHERE gameweek = ? AND season = ?"
cursor.execute(count_query, (3, "2025-26"))
count1 = cursor.fetchone()[0]
logger.info(f"Initial count: {count1}")

# Try delete
delete_query = "DELETE FROM player_gameweek_stats WHERE gameweek = ? AND season = ?"
cursor.execute(delete_query, (3, "2025-26"))
deleted = cursor.rowcount
logger.info(f"Deleted (rowcount): {deleted}")

# Count again before commit
cursor.execute(count_query, (3, "2025-26"))
count2 = cursor.fetchone()[0]
logger.info(f"Count after delete (before commit): {count2}")

# Commit
conn.commit()

# Count again after commit
cursor.execute(count_query, (3, "2025-26"))
count3 = cursor.fetchone()[0]
logger.info(f"Count after commit: {count3}")

# Now test with a new cursor
cursor2 = conn.cursor()
cursor2.execute(count_query, (3, "2025-26"))
count4 = cursor2.fetchone()[0]
logger.info(f"Count with new cursor: {count4}")

conn.close()