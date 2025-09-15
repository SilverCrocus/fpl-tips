"""
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
            'player_name': ['A', 'B', 'C'],
            'position': ['FWD', 'MID', 'DEF'],
            'price': [10.0, np.nan, 8.0],  # NaN in critical field
            'total_points': [100, 80, 60],
            'gameweek': [1, 1, 1],
            'season': ['2024-25', '2024-25', '2024-25'],
            'chance_of_playing_next_round': [100, np.nan, 75]
        })

        # Should reduce quality score for NaN in critical fields
        result = self.validator.validate_merged_data(data)
        # NaN in price should affect quality score
        assert result.data_quality_score < 1.0  # Quality affected by NaN in critical field


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
