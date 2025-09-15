#!/usr/bin/env python
"""
Migration Script for Data Pipeline Improvements
Applies critical fixes to the FPL data pipeline
"""

import sys
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def backup_file(file_path: Path) -> Path:
    """Create backup of original file

    Args:
        file_path: Path to file to backup

    Returns:
        Path to backup file
    """
    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
    if file_path.exists():
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    else:
        logger.warning(f"File not found: {file_path}")
        return None


def apply_data_merger_fixes():
    """Apply critical fixes to data_merger.py"""
    logger.info("Applying data_merger.py fixes...")

    merger_path = Path("src/data/data_merger.py")
    backup_file(merger_path)

    # Read the original file
    with open(merger_path, 'r') as f:
        content = f.read()

    # Fix 1: Database isolation level
    content = content.replace(
        'self.conn = sqlite3.connect(self.db_path, isolation_level="DEFERRED")',
        'self.conn = sqlite3.connect(self.db_path, isolation_level=None)  # Use autocommit for explicit transaction control'
    )

    # Fix 2: Rename fixture_difficulty overwriting
    # Line 405 issue - don't overwrite original fixture_difficulty
    content = content.replace(
        'data["fixture_difficulty"] = fixture_diff_categorical.astype("int64")',
        '# Create new column for Elo-based difficulty instead of overwriting\n        data["fixture_difficulty_elo"] = fixture_diff_categorical.astype("int64")'
    )

    # Fix 3: Clearer NaN handling for availability
    old_availability = """        # NaN means no injury concerns (100% available), >= 75% is considered available
        data["is_available"] = (
            data["chance_of_playing_next_round"].isna() |
            (data["chance_of_playing_next_round"] >= 75)
        ).astype(int)"""

    new_availability = """        # Clear availability logic: NaN = no injury data (100% available), >= 75% = likely to play
        data["is_available"] = np.where(
            data["chance_of_playing_next_round"].isna(),
            1,  # No injury data = available
            np.where(
                data["chance_of_playing_next_round"] >= 75,
                1,  # 75% or higher = available
                0   # Less than 75% = not available
            )
        ).astype(int)"""

    content = content.replace(old_availability, new_availability)

    # Fix 4: Rename goal_involvement to goal_involvement_rate for clarity
    content = content.replace(
        'data["goal_involvement"] = (',
        'data["goal_involvement_rate"] = (  # Rate per game, not total'
    )

    # Fix 5: Add transaction management to save_to_database
    # Find the save_to_database method and wrap critical section
    old_delete_insert = """                # Delete existing records for this gameweek/season to avoid duplicates
            if "player_id" in data.columns and not data.empty:
                gameweek = data["gameweek"].iloc[0]
                season = data["season"].iloc[0]

                # Ensure gameweek is an integer for proper comparison
                if isinstance(gameweek, (np.integer, np.floating)):
                    gameweek = int(gameweek)
                elif isinstance(gameweek, str) and gameweek.isdigit():
                    gameweek = int(gameweek)

                # Use a cursor for operations
                cursor = self.conn.cursor()"""

    new_delete_insert = """                # Delete existing records for this gameweek/season to avoid duplicates
            if "player_id" in data.columns and not data.empty:
                gameweek = data["gameweek"].iloc[0]
                season = data["season"].iloc[0]

                # Ensure gameweek is an integer for proper comparison
                if isinstance(gameweek, (np.integer, np.floating)):
                    gameweek = int(gameweek)
                elif isinstance(gameweek, str) and gameweek.isdigit():
                    gameweek = int(gameweek)

                # Start explicit transaction for atomic operation
                self.conn.execute("BEGIN IMMEDIATE")
                cursor = self.conn.cursor()"""

    content = content.replace(old_delete_insert, new_delete_insert)

    # Add commit after successful insert
    old_insert_end = """                logger.info(f"Saved {rows_inserted} records to {table}")
        except Exception as e:
            logger.error(f"Database error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            # Rollback on error
            self.conn.rollback()
            raise"""

    new_insert_end = """                logger.info(f"Saved {rows_inserted} records to {table}")

                # Verify the insert worked
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE gameweek = ? AND season = ?", (gameweek, season))
                final_count = cursor.fetchone()[0]
                if final_count != len(data):
                    self.conn.execute("ROLLBACK")
                    raise ValueError(f"Insert verification failed: expected {len(data)}, got {final_count}")

                # Commit the transaction
                self.conn.execute("COMMIT")
        except Exception as e:
            logger.error(f"Database error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            # Rollback on error
            try:
                self.conn.execute("ROLLBACK")
            except:
                pass  # Rollback might fail if connection is broken
            raise"""

    content = content.replace(old_insert_end, new_insert_end)

    # Fix 6: Add numpy import if missing
    if "import numpy as np" not in content:
        content = content.replace(
            "import pandas as pd",
            "import pandas as pd\nimport numpy as np"
        )

    # Write the fixed content
    with open(merger_path, 'w') as f:
        f.write(content)

    logger.info("data_merger.py fixes applied successfully")


def update_database_schema():
    """Update database schema to support new columns"""
    logger.info("Updating database schema...")

    import sqlite3
    from pathlib import Path

    db_path = Path("data/fpl_data.db")
    if not db_path.exists():
        logger.info("Database doesn't exist yet, will be created with new schema")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if new columns exist
        cursor.execute("PRAGMA table_info(player_gameweek_stats)")
        columns = [col[1] for col in cursor.fetchall()]

        # Add new columns if they don't exist
        new_columns = [
            ("fixture_difficulty_original", "INTEGER"),
            ("fixture_difficulty_elo", "INTEGER"),
            ("goal_involvement_rate", "REAL"),
            ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        ]

        for col_name, col_type in new_columns:
            if col_name not in columns:
                try:
                    cursor.execute(f"ALTER TABLE player_gameweek_stats ADD COLUMN {col_name} {col_type}")
                    logger.info(f"Added column: {col_name}")
                except sqlite3.OperationalError:
                    logger.warning(f"Column {col_name} might already exist")

        # Rename goal_involvement to goal_involvement_rate if it exists
        if "goal_involvement" in columns and "goal_involvement_rate" not in columns:
            # SQLite doesn't support RENAME COLUMN in older versions
            # We'll need to recreate the table or use a workaround
            logger.info("Note: 'goal_involvement' column should be interpreted as 'goal_involvement_rate'")

        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        logger.info("Enabled WAL mode for database")

        conn.commit()
        logger.info("Database schema updated successfully")

    except Exception as e:
        logger.error(f"Error updating database: {e}")
        conn.rollback()
    finally:
        conn.close()


def create_validation_tests():
    """Create test file for data validation"""
    logger.info("Creating validation tests...")

    test_content = '''"""
Tests for Data Pipeline Validation
"""

import pytest
import pandas as pd
import numpy as np
from src.data.data_validator import DataValidator, ValidationRule


class TestDataValidator:
    """Test data validation functionality"""

    def setup_method(self):
        """Set up test validator"""
        self.validator = DataValidator()

    def test_validate_fpl_data_success(self):
        """Test successful FPL data validation"""
        data = pd.DataFrame({
            'id': [1, 2],
            'web_name': ['Player A', 'Player B'],
            'position': ['FWD', 'MID'],
            'team': [1, 2],
            'price': [10.5, 8.0],
            'total_points': [100, 80],
            'minutes': [900, 720],
            'chance_of_playing_next_round': [100, 75]
        })

        result = self.validator.validate_fpl_data(data)
        assert result.passed
        assert len(result.errors) == 0

    def test_validate_fpl_data_missing_columns(self):
        """Test FPL data validation with missing columns"""
        data = pd.DataFrame({
            'id': [1, 2],
            'web_name': ['Player A', 'Player B']
        })

        result = self.validator.validate_fpl_data(data)
        assert not result.passed
        assert len(result.errors) > 0
        assert 'Missing required columns' in result.errors[0]

    def test_validate_odds_data_probability_range(self):
        """Test odds data validation with probability out of range"""
        data = pd.DataFrame({
            'player_id': [1, 2],
            'prob_goal': [0.5, 1.5],  # Second value out of range
            'odds_goal': [2.0, 3.0]
        })

        result = self.validator.validate_odds_data(data)
        assert len(result.errors) > 0 or len(result.warnings) > 0

    def test_validate_merged_data_position_flags(self):
        """Test merged data validation with position flag consistency"""
        data = pd.DataFrame({
            'player_id': [1, 2],
            'player_name': ['A', 'B'],
            'position': ['FWD', 'MID'],
            'price': [10.0, 8.0],
            'total_points': [100, 80],
            'gameweek': [1, 1],
            'season': ['2024-25', '2024-25'],
            'is_goalkeeper': [0, 0],
            'is_defender': [0, 0],
            'is_midfielder': [0, 1],
            'is_forward': [1, 1]  # Wrong for player 2
        })

        result = self.validator.validate_merged_data(data)
        assert len(result.errors) > 0
        assert 'invalid position flags' in result.errors[0]

    def test_custom_validation_rule(self):
        """Test custom validation rule application"""
        data = pd.DataFrame({
            'player_id': [1, 2],
            'total_points': [100, 500]  # Second value unusually high
        })

        rule = ValidationRule(
            column='total_points',
            rule_type='range',
            params={'min': 0, 'max': 400},
            severity='warning'
        )

        result = self.validator.validate_pipeline_stage(
            data, 'custom', [rule]
        )

        assert len(result.warnings) > 0

    def test_data_consistency_check(self):
        """Test consistency check between datasets"""
        df1 = pd.DataFrame({
            'player_id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        })

        df2 = pd.DataFrame({
            'player_id': [1, 2, 4],  # Player 3 missing, 4 extra
            'team': ['X', 'Y', 'Z']
        })

        is_consistent, issues = self.validator.check_data_consistency(
            df1, df2, ['player_id']
        )

        assert not is_consistent
        assert len(issues) == 2  # One for df1, one for df2

    def test_nan_handling_in_validation(self):
        """Test that NaN values are handled correctly"""
        data = pd.DataFrame({
            'player_id': [1, 2, 3],
            'price': [10.0, np.nan, 8.0],
            'chance_of_playing_next_round': [100, np.nan, 75]
        })

        # Should not fail on NaN values in optional fields
        result = self.validator.validate_pipeline_stage(data, 'test')
        # NaN in price might cause warning but not error for non-required validation
        assert result.data_quality_score < 1.0  # Quality affected by NaN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

    test_path = Path("tests/test_data_validation.py")
    test_path.parent.mkdir(exist_ok=True)

    with open(test_path, 'w') as f:
        f.write(test_content)

    logger.info(f"Created test file: {test_path}")


def main():
    """Run all migration steps"""
    logger.info("Starting data pipeline migration...")

    try:
        # Step 1: Apply code fixes
        apply_data_merger_fixes()

        # Step 2: Update database schema
        update_database_schema()

        # Step 3: Create validation tests
        create_validation_tests()

        logger.info("\n" + "="*50)
        logger.info("Migration completed successfully!")
        logger.info("="*50)

        logger.info("\nNext steps:")
        logger.info("1. Review the changes in src/data/data_merger.py")
        logger.info("2. Run tests: uv run pytest tests/test_data_validation.py")
        logger.info("3. Test the pipeline: uv run python -m src.main fetch-data")
        logger.info("4. Consider using data_merger_fixed.py for new implementations")
        logger.info("5. Integrate data_validator.py into your pipeline")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        logger.info("\nTo restore backups:")
        logger.info("  cp src/data/data_merger.py.backup src/data/data_merger.py")
        sys.exit(1)


if __name__ == "__main__":
    main()