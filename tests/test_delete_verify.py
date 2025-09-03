#!/usr/bin/env python
"""Verify the delete is working"""

import sqlite3
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conn = sqlite3.connect("data/fpl_data.db")

# Count before delete
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM player_gameweek_stats WHERE gameweek = 3 AND season = '2025-26'")
count_before = cursor.fetchone()[0]
logger.info(f"Records before delete: {count_before}")

# Delete
cursor.execute("DELETE FROM player_gameweek_stats WHERE gameweek = 3 AND season = '2025-26'")
logger.info(f"Deleted {cursor.rowcount} rows")

# Count after delete
cursor.execute("SELECT COUNT(*) FROM player_gameweek_stats WHERE gameweek = 3 AND season = '2025-26'")
count_after = cursor.fetchone()[0]
logger.info(f"Records after delete: {count_after}")

# Commit
conn.commit()

# Count after commit
cursor.execute("SELECT COUNT(*) FROM player_gameweek_stats WHERE gameweek = 3 AND season = '2025-26'")
count_final = cursor.fetchone()[0]
logger.info(f"Records after commit: {count_final}")

conn.close()