"""
Data Validation Module
Ensures data quality and consistency throughout the FPL pipeline
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Defines a validation rule for a column"""
    column: str
    rule_type: str  # 'required', 'type', 'range', 'values', 'regex'
    params: Dict[str, Any] = field(default_factory=dict)
    severity: str = 'error'  # 'error', 'warning', 'info'
    message: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation check"""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    data_quality_score: float = 1.0
    row_issues: Dict[int, List[str]] = field(default_factory=dict)


class DataValidator:
    """Validates FPL data at various pipeline stages"""

    # Define validation schemas for different data types
    FPL_PLAYER_SCHEMA = {
        'required_columns': [
            'id', 'web_name', 'position', 'team', 'price',
            'total_points', 'minutes', 'chance_of_playing_next_round'
        ],
        'column_types': {
            'id': 'int',
            'price': 'float',
            'total_points': 'int',
            'minutes': 'int',
            'goals_scored': 'int',
            'assists': 'int'
        },
        'value_ranges': {
            'price': (3.5, 20.0),
            'total_points': (0, 500),
            'minutes': (0, 3420),  # 38 games * 90 minutes
            'selected_by_percent': (0, 100),
            'chance_of_playing_next_round': (0, 100),
            'form': (0, 10)
        },
        'allowed_values': {
            'position': ['GK', 'DEF', 'MID', 'FWD']
        }
    }

    ODDS_DATA_SCHEMA = {
        'required_columns': ['player_id', 'prob_goal'],
        'column_types': {
            'player_id': 'int',
            'odds_goal': 'float',
            'prob_goal': 'float'
        },
        'value_ranges': {
            'prob_goal': (0, 1),
            'prob_assist': (0, 1),
            'prob_clean_sheet': (0, 1),
            'odds_goal': (1.01, 1000)
        }
    }

    MERGED_DATA_SCHEMA = {
        'required_columns': [
            'player_id', 'player_name', 'position', 'price',
            'total_points', 'gameweek', 'season'
        ],
        'column_types': {
            'player_id': 'int',
            'gameweek': 'int',
            'price': 'float',
            'total_points': 'int',
            'is_available': 'int'
        },
        'value_ranges': {
            'gameweek': (1, 38),
            'data_quality': (0, 1),
            'goal_involvement_rate': (0, 10),  # Goals+assists per game
            'ict_per_90': (0, 1000)
        },
        'allowed_values': {
            'is_available': [0, 1],
            'is_goalkeeper': [0, 1],
            'is_defender': [0, 1],
            'is_midfielder': [0, 1],
            'is_forward': [0, 1]
        }
    }

    def __init__(self):
        """Initialize validator with predefined rules"""
        self.validation_history = []

    def validate_fpl_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate FPL API data

        Args:
            df: DataFrame with FPL player data

        Returns:
            ValidationResult with details
        """
        logger.info("Validating FPL data...")
        return self._validate_against_schema(df, self.FPL_PLAYER_SCHEMA, "FPL Data")

    def validate_odds_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate odds/betting data

        Args:
            df: DataFrame with odds data

        Returns:
            ValidationResult with details
        """
        logger.info("Validating odds data...")
        result = self._validate_against_schema(df, self.ODDS_DATA_SCHEMA, "Odds Data")

        # Additional odds-specific validations
        if not result.errors:
            # Check odds-probability consistency
            if 'odds_goal' in df.columns and 'prob_goal' in df.columns:
                # Implied probability should roughly match given probability
                implied_prob = 1 / df['odds_goal']
                prob_diff = (implied_prob - df['prob_goal']).abs()
                inconsistent = df[prob_diff > 0.1]
                if not inconsistent.empty:
                    result.warnings.append(
                        f"Found {len(inconsistent)} records with inconsistent odds/probability"
                    )

        return result

    def validate_merged_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate merged/unified data

        Args:
            df: DataFrame with merged data

        Returns:
            ValidationResult with details
        """
        logger.info("Validating merged data...")
        result = self._validate_against_schema(df, self.MERGED_DATA_SCHEMA, "Merged Data")

        # Additional merged data validations
        if not result.errors:
            # Check position flag consistency
            position_flags = ['is_goalkeeper', 'is_defender', 'is_midfielder', 'is_forward']
            if all(col in df.columns for col in position_flags):
                flag_sum = df[position_flags].sum(axis=1)
                invalid_flags = df[flag_sum != 1]
                if not invalid_flags.empty:
                    result.errors.append(
                        f"Found {len(invalid_flags)} records with invalid position flags"
                    )
                    for idx in invalid_flags.index[:5]:
                        result.row_issues[idx] = [
                            f"Position flags sum to {flag_sum[idx]}, should be 1"
                        ]

            # Check data quality distribution
            if 'data_quality' in df.columns:
                low_quality = df[df['data_quality'] < 0.5]
                if len(low_quality) > len(df) * 0.2:
                    result.warnings.append(
                        f"{len(low_quality)}/{len(df)} records have low data quality (<0.5)"
                    )

        return result

    def _validate_against_schema(
        self, df: pd.DataFrame, schema: Dict[str, Any], data_name: str
    ) -> ValidationResult:
        """Validate DataFrame against a schema

        Args:
            df: DataFrame to validate
            schema: Validation schema
            data_name: Name for logging

        Returns:
            ValidationResult
        """
        result = ValidationResult(passed=True)

        # Check if DataFrame is empty
        if df.empty:
            result.errors.append(f"{data_name} is empty")
            result.passed = False
            return result

        # Check required columns
        if 'required_columns' in schema:
            missing = set(schema['required_columns']) - set(df.columns)
            if missing:
                result.errors.append(f"Missing required columns: {missing}")
                result.passed = False

        # Check column types
        if 'column_types' in schema:
            for col, expected_type in schema['column_types'].items():
                if col in df.columns:
                    actual_type = df[col].dtype
                    if expected_type == 'int' and not pd.api.types.is_integer_dtype(actual_type):
                        if not df[col].isna().all():  # Only error if column has data
                            result.warnings.append(
                                f"Column '{col}' should be int, got {actual_type}"
                            )
                    elif expected_type == 'float' and not pd.api.types.is_float_dtype(actual_type):
                        if not df[col].isna().all():
                            result.warnings.append(
                                f"Column '{col}' should be float, got {actual_type}"
                            )

        # Check value ranges
        if 'value_ranges' in schema:
            for col, (min_val, max_val) in schema['value_ranges'].items():
                if col in df.columns:
                    # Ignore NaN values in range check
                    valid_data = df[df[col].notna()]
                    if not valid_data.empty:
                        out_of_range = valid_data[
                            (valid_data[col] < min_val) | (valid_data[col] > max_val)
                        ]
                        if not out_of_range.empty:
                            pct = len(out_of_range) / len(valid_data) * 100
                            msg = (
                                f"Column '{col}' has {len(out_of_range)} values "
                                f"({pct:.1f}%) outside range [{min_val}, {max_val}]"
                            )
                            if pct > 10:
                                result.errors.append(msg)
                                result.passed = False
                            else:
                                result.warnings.append(msg)

                            # Add specific row issues (first 5)
                            for idx in out_of_range.index[:5]:
                                if idx not in result.row_issues:
                                    result.row_issues[idx] = []
                                result.row_issues[idx].append(
                                    f"{col}={out_of_range.loc[idx, col]} outside [{min_val}, {max_val}]"
                                )

        # Check allowed values
        if 'allowed_values' in schema:
            for col, allowed in schema['allowed_values'].items():
                if col in df.columns:
                    invalid = df[~df[col].isin(allowed + [None, np.nan])]
                    if not invalid.empty:
                        unique_invalid = invalid[col].unique()
                        result.errors.append(
                            f"Column '{col}' has invalid values: {unique_invalid}"
                        )
                        result.passed = False

        # Calculate data quality score
        result.data_quality_score = self._calculate_quality_score(df, result)

        # Log summary
        if result.passed:
            logger.info(f"{data_name} validation passed (quality score: {result.data_quality_score:.2f})")
        else:
            logger.error(f"{data_name} validation failed with {len(result.errors)} errors")

        return result

    def _calculate_quality_score(self, df: pd.DataFrame, result: ValidationResult) -> float:
        """Calculate overall data quality score

        Args:
            df: DataFrame being validated
            result: Current validation result

        Returns:
            Quality score between 0 and 1
        """
        score = 1.0

        # Penalize for errors (heavy penalty)
        score -= len(result.errors) * 0.2

        # Penalize for warnings (light penalty)
        score -= len(result.warnings) * 0.05

        # Penalize for missing data
        missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns))
        score -= missing_pct * 0.3

        # Penalize for row issues
        if result.row_issues:
            row_issue_pct = len(result.row_issues) / len(df)
            score -= row_issue_pct * 0.2

        return max(0, min(1, score))

    def validate_pipeline_stage(
        self, df: pd.DataFrame, stage: str, custom_rules: Optional[List[ValidationRule]] = None
    ) -> ValidationResult:
        """Validate data at a specific pipeline stage

        Args:
            df: DataFrame to validate
            stage: Pipeline stage name
            custom_rules: Additional custom validation rules

        Returns:
            ValidationResult
        """
        logger.info(f"Validating pipeline stage: {stage}")

        # Select appropriate schema based on stage
        if stage == 'fpl_fetch':
            result = self.validate_fpl_data(df)
        elif stage == 'odds_fetch':
            result = self.validate_odds_data(df)
        elif stage == 'merge':
            result = self.validate_merged_data(df)
        else:
            result = ValidationResult(passed=True)

        # Apply custom rules
        if custom_rules:
            for rule in custom_rules:
                self._apply_custom_rule(df, rule, result)

        # Store in history
        self.validation_history.append({
            'stage': stage,
            'timestamp': pd.Timestamp.now(),
            'passed': result.passed,
            'quality_score': result.data_quality_score,
            'errors': len(result.errors),
            'warnings': len(result.warnings)
        })

        return result

    def _apply_custom_rule(self, df: pd.DataFrame, rule: ValidationRule, result: ValidationResult):
        """Apply a custom validation rule

        Args:
            df: DataFrame to validate
            rule: Custom validation rule
            result: ValidationResult to update
        """
        if rule.column not in df.columns:
            if rule.rule_type == 'required':
                msg = rule.message or f"Required column '{rule.column}' is missing"
                if rule.severity == 'error':
                    result.errors.append(msg)
                    result.passed = False
                else:
                    result.warnings.append(msg)
            return

        col_data = df[rule.column]

        if rule.rule_type == 'type':
            expected_type = rule.params.get('type')
            if not self._check_type(col_data, expected_type):
                msg = rule.message or f"Column '{rule.column}' has wrong type"
                if rule.severity == 'error':
                    result.errors.append(msg)
                else:
                    result.warnings.append(msg)

        elif rule.rule_type == 'range':
            min_val = rule.params.get('min', -np.inf)
            max_val = rule.params.get('max', np.inf)
            out_of_range = col_data[(col_data < min_val) | (col_data > max_val)]
            if not out_of_range.empty:
                msg = rule.message or f"Column '{rule.column}' has {len(out_of_range)} values out of range"
                if rule.severity == 'error':
                    result.errors.append(msg)
                else:
                    result.warnings.append(msg)

        elif rule.rule_type == 'values':
            allowed = rule.params.get('allowed', [])
            invalid = col_data[~col_data.isin(allowed)]
            if not invalid.empty:
                msg = rule.message or f"Column '{rule.column}' has invalid values"
                if rule.severity == 'error':
                    result.errors.append(msg)
                else:
                    result.warnings.append(msg)

    def _check_type(self, series: pd.Series, expected_type: str) -> bool:
        """Check if series matches expected type

        Args:
            series: Pandas series to check
            expected_type: Expected type name

        Returns:
            True if type matches
        """
        if expected_type == 'int':
            return pd.api.types.is_integer_dtype(series)
        elif expected_type == 'float':
            return pd.api.types.is_float_dtype(series)
        elif expected_type == 'str':
            return pd.api.types.is_string_dtype(series)
        elif expected_type == 'bool':
            return pd.api.types.is_bool_dtype(series)
        else:
            return True

    def get_validation_report(self) -> pd.DataFrame:
        """Get summary report of all validations

        Returns:
            DataFrame with validation history
        """
        if not self.validation_history:
            return pd.DataFrame()

        return pd.DataFrame(self.validation_history)

    def check_data_consistency(
        self, df1: pd.DataFrame, df2: pd.DataFrame, join_cols: List[str]
    ) -> Tuple[bool, List[str]]:
        """Check consistency between two datasets

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            join_cols: Columns to join on

        Returns:
            Tuple of (is_consistent, list of issues)
        """
        issues = []

        # Check if join columns exist
        missing_df1 = set(join_cols) - set(df1.columns)
        missing_df2 = set(join_cols) - set(df2.columns)

        if missing_df1:
            issues.append(f"Join columns missing in df1: {missing_df1}")
        if missing_df2:
            issues.append(f"Join columns missing in df2: {missing_df2}")

        if issues:
            return False, issues

        # Check for orphaned records
        merged = df1.merge(df2, on=join_cols, how='outer', indicator=True)
        left_only = merged[merged['_merge'] == 'left_only']
        right_only = merged[merged['_merge'] == 'right_only']

        if not left_only.empty:
            issues.append(f"Found {len(left_only)} records in df1 not in df2")
        if not right_only.empty:
            issues.append(f"Found {len(right_only)} records in df2 not in df1")

        return len(issues) == 0, issues


# Example usage
if __name__ == "__main__":
    # Create validator
    validator = DataValidator()

    # Test with sample data
    test_data = pd.DataFrame({
        'player_id': [1, 2, 3, 4],
        'player_name': ['Player A', 'Player B', 'Player C', 'Player D'],
        'position': ['FWD', 'MID', 'DEF', 'GK'],
        'price': [10.5, 8.0, 5.5, 4.5],
        'total_points': [100, 80, 60, 40],
        'gameweek': [1, 1, 1, 1],
        'season': ['2024-25'] * 4,
        'prob_goal': [0.6, 0.4, 0.1, 0.01],
        'chance_of_playing_next_round': [100, 75, 50, None],  # Test NaN
        'is_available': [1, 1, 0, 1],
        'data_quality': [0.9, 0.8, 0.4, 0.7]
    })

    # Validate as merged data
    result = validator.validate_merged_data(test_data)

    print(f"Validation passed: {result.passed}")
    print(f"Data quality score: {result.data_quality_score:.2f}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result.row_issues:
        print("\nRow issues:")
        for idx, issues in list(result.row_issues.items())[:5]:
            print(f"  Row {idx}: {', '.join(issues)}")

    # Test custom rule
    custom_rule = ValidationRule(
        column='total_points',
        rule_type='range',
        params={'min': 0, 'max': 300},
        severity='warning',
        message='Total points seems unusually high'
    )

    result2 = validator.validate_pipeline_stage(
        test_data, 'custom_check', [custom_rule]
    )

    # Get validation report
    report = validator.get_validation_report()
    print("\nValidation History:")
    print(report)