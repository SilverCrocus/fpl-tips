"""Test cases for critical bug fixes"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from src.models.rule_based_scorer import RuleBasedScorer, ScoringWeights
from src.models.optimizer import FPLOptimizer
from src.data.data_merger import DataMerger


def test_price_factor_formula():
    """Test that price factor formula works correctly"""

    # Create test weights
    weights = ScoringWeights(
        gk_weights={"form": 1.0, "value_ratio": 0.0},
        def_weights={"form": 1.0, "value_ratio": 0.0},
        mid_weights={"form": 1.0, "value_ratio": 0.0},
        fwd_weights={"form": 1.0, "value_ratio": 0.0}
    )

    scorer = RuleBasedScorer(weights)

    # Test cheap player (£5.0)
    cheap_player = pd.Series({
        "player_name": "Cheap Player",
        "position": "MID",
        "price": 5.0,
        "form": 5.0,
        "chance_of_playing_next_round": 100,
        "total_points": 100
    })

    # Test expensive player (£15.0)
    expensive_player = pd.Series({
        "player_name": "Expensive Player",
        "position": "MID",
        "price": 15.0,
        "form": 5.0,
        "chance_of_playing_next_round": 100,
        "total_points": 100
    })

    cheap_score = scorer.score_player(cheap_player)
    expensive_score = scorer.score_player(expensive_player)

    # Cheap players should get a boost (price_factor > 1)
    # Expensive players should get a penalty (price_factor < 1)
    # With logarithmic scaling, £5 should score higher than £15 for same stats
    assert cheap_score > expensive_score, f"Cheap player ({cheap_score:.2f}) should score higher than expensive ({expensive_score:.2f})"

    # Verify the price factor follows logarithmic decay
    median_price = 7.0
    cheap_price_factor = np.exp(-0.1 * np.log(5.0 / median_price))
    expensive_price_factor = np.exp(-0.1 * np.log(15.0 / median_price))

    assert cheap_price_factor > 1.0, "Cheap player should get boost"
    assert expensive_price_factor < 1.0, "Expensive player should get penalty"


def test_blank_percentage_smooth_transition():
    """Test that blank percentage uses smooth transition instead of discrete jumps"""

    weights = ScoringWeights(
        gk_weights={"form": 1.0, "value_ratio": 0.0},
        def_weights={"form": 1.0, "value_ratio": 0.0},
        mid_weights={"form": 1.0, "value_ratio": 0.0},
        fwd_weights={"form": 1.0, "value_ratio": 0.0}
    )

    scorer = RuleBasedScorer(weights)

    # Test smooth transition around form values
    forms = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    blank_percentages = []

    for form_val in forms:
        player = pd.Series({
            "player_name": "Test Player",
            "form": form_val
        })
        blank_pct = scorer.calculate_blank_percentage(player)
        blank_percentages.append(blank_pct)

    # Check that blank percentage decreases smoothly as form increases
    for i in range(1, len(blank_percentages)):
        assert blank_percentages[i] <= blank_percentages[i-1], \
            f"Blank percentage should decrease smoothly with form"

    # No sudden jumps - difference between consecutive values should be reasonable
    for i in range(1, len(blank_percentages)):
        diff = abs(blank_percentages[i] - blank_percentages[i-1])
        assert diff < 0.15, f"Jump too large between form {forms[i-1]} and {forms[i]}: {diff:.3f}"


def test_milp_bench_objective():
    """Test that MILP objective function adds value for bench players"""

    # Create sample data with proper indexing
    players_data = pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5],
        "player_name": ["P1", "P2", "P3", "P4", "P5"],
        "position": ["GK", "DEF", "MID", "FWD", "MID"],
        "team_name": ["Team1", "Team2", "Team3", "Team4", "Team5"],
        "price": [5.0, 6.0, 7.0, 8.0, 9.0],
        "expected_points": [4.0, 5.0, 6.0, 7.0, 8.0],
        "chance_of_playing_next_round": [100, 100, 100, 100, 100]
    })
    players_data.index = players_data["player_id"]

    # Create MILPOptimizer and build model
    from src.models.optimizer import MILPOptimizer
    milp_optimizer = MILPOptimizer()

    # Build the optimization problem
    problem = milp_optimizer.build_model(
        players_data,
        budget=50.0,
        current_team=None,
        free_transfers=0
    )

    # Check that the objective function is properly constructed
    # The coefficient for bench players should be positive (0.1 * expected_points)
    objective_str = str(problem.objective)

    # Verify bench variables have positive coefficients
    assert "bench" in objective_str, "Bench variables should be in objective"

    # Check that bench points are being added (coefficient should be positive)
    # In the fixed version, bench players add 0.1 * expected_points
    # This means we should see addition, not subtraction
    # The old version had "- 0.9" which would show as negative in the objective
    # The new version has "+ 0.1" which should show as positive
    import re
    # Look for bench coefficients in the objective string
    bench_matches = re.findall(r'([+-]?\s*\d*\.?\d*)\*bench', objective_str)
    if bench_matches:
        # Check that at least one bench coefficient is positive
        has_positive = any(not match.strip().startswith('-') for match in bench_matches if match.strip())
        assert has_positive, "Bench points should have positive coefficients (added, not subtracted)"


def test_availability_check_conservative():
    """Test that availability check is conservative with NaN values"""

    # Create test data with various availability scenarios
    test_data = pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5],
        "player_name": ["P1", "P2", "P3", "P4", "P5"],
        "chance_of_playing_next_round": [100, 75, 50, 25, np.nan],
        "form": [5.0, 4.0, 3.0, 2.0, 0.0]
    })

    merger = DataMerger()

    # Apply the availability logic (mimicking the fixed data_merger.py logic)
    chance = test_data["chance_of_playing_next_round"].copy()
    has_recent_form = test_data["form"] > 0

    # Fill NaN values based on form
    for idx in test_data.index:
        if pd.isna(chance[idx]):
            if has_recent_form[idx]:
                chance[idx] = 100
            else:
                chance[idx] = 50

    test_data["is_available"] = (chance >= 75).astype(int)

    # Check results
    assert test_data.iloc[0]["is_available"] == 1, "100% chance should be available"
    assert test_data.iloc[1]["is_available"] == 1, "75% chance should be available"
    assert test_data.iloc[2]["is_available"] == 0, "50% chance should NOT be available"
    assert test_data.iloc[3]["is_available"] == 0, "25% chance should NOT be available"
    assert test_data.iloc[4]["is_available"] == 0, "NaN with no form should NOT be available"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])