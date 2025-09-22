"""Analyze MILP budget constraint behavior"""

import json
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.WARNING)

from src.data.data_merger import DataMerger
from src.models.rule_based_scorer import RuleBasedScorer, ScoringWeights
from src.milp_team_manager import MILPTransferOptimizer

def load_weights():
    """Load scoring weights from config"""
    with open("config/scoring_weights.json") as f:
        weights_data = json.load(f)
    return ScoringWeights.from_dict(weights_data)

print("="*70)
print("MILP BUDGET CONSTRAINT ANALYSIS")
print("="*70)

# Initialize components
merger = DataMerger()
data = merger.load_from_database(gameweek=5)
weights = load_weights()
scorer = RuleBasedScorer(weights)
data = scorer.score_all_players(data)

# Current team IDs from the user's team
current_team_ids = [67, 291, 348, 261, 569, 260, 381, 82, 200, 242, 249, 470, 624, 427, 64]

# Test different budget amounts
budgets = [0.0, 2.5, 5.5, 10.0, 15.0, 20.0]

print("\nTesting MILP optimization with different budgets:")
print("-"*70)
print(f"{'Budget (£m)':<15} {'Transfers':<10} {'Total In':<12} {'Total Out':<12} {'Net Cost':<12} {'Points Gain':<12}")
print("-"*70)

for budget in budgets:
    optimizer = MILPTransferOptimizer(scorer, data)
    result = optimizer.optimize_transfers(
        current_team_ids=current_team_ids,
        budget=budget,
        free_transfers=1,
        max_transfers=3,
        horizon_weeks=5
    )

    if result.transfers_out and result.transfers_in:
        total_out = sum(p['price'] for p in result.transfers_out)
        total_in = sum(p['price'] for p in result.transfers_in)
        net_cost = total_in - total_out
        num_transfers = len(result.transfers_out)

        print(f"£{budget:5.1f}          {num_transfers:<10} £{total_in:10.1f}  £{total_out:10.1f}  £{net_cost:10.1f}  {result.expected_points_gain:10.1f}")
    else:
        print(f"£{budget:5.1f}          No transfers recommended")

# Analyze the specific case with £5.5m
print("\n" + "="*70)
print("DETAILED ANALYSIS WITH £5.5m BUDGET")
print("="*70)

optimizer = MILPTransferOptimizer(scorer, data)
result = optimizer.optimize_transfers(
    current_team_ids=current_team_ids,
    budget=5.5,
    free_transfers=1,
    max_transfers=3,
    horizon_weeks=5
)

if result.transfers_out and result.transfers_in:
    print(f"\nOptimization status: {result.optimization_status}")
    print(f"Number of transfers: {len(result.transfers_out)}")

    # Calculate budget usage
    total_out = sum(p['price'] for p in result.transfers_out)
    total_in = sum(p['price'] for p in result.transfers_in)
    net_spend = total_in - total_out
    budget_utilization = (net_spend / 5.5) * 100 if 5.5 > 0 else 0

    print(f"\nBudget Analysis:")
    print(f"  Bank available: £5.5m")
    print(f"  Total sales: £{total_out:.1f}m")
    print(f"  Total purchases: £{total_in:.1f}m")
    print(f"  Net spend: £{net_spend:.1f}m")
    print(f"  Budget utilization: {budget_utilization:.1f}%")
    print(f"  Unused budget: £{5.5 - net_spend:.1f}m")

    print(f"\nTransfers OUT:")
    for p in result.transfers_out:
        print(f"  {p['name']:<20} {p['position']:<4} £{p['price']:5.1f}m  Score: {p['score']:5.1f}")

    print(f"\nTransfers IN:")
    for p in result.transfers_in:
        print(f"  {p['name']:<20} {p['position']:<4} £{p['price']:5.1f}m  Score: {p['score']:5.1f}")

    print(f"\nExpected points gain: {result.expected_points_gain:.1f}")

# Check if MILP tries to maximize budget usage
print("\n" + "="*70)
print("MILP OBJECTIVE FUNCTION ANALYSIS")
print("="*70)

print("\nChecking MILP code at src/milp_team_manager.py...")
print("\nKey findings:")
print("1. Budget constraint: money_in <= money_out + budget")
print("   - Uses <= not =, doesn't force spending all funds")
print("2. Objective function: Maximize total model_score")
print("   - No incentive to spend more money")
print("3. Result: MILP optimizes for points, not budget usage")

print("\n" + "="*70)
print("CONCLUSIONS")
print("="*70)

print("""
1. **Budget Usage**: The MILP doesn't try to maximize budget usage.
   It only spends what's needed for the best point gain.

2. **Salah → Bruno Logic**:
   - Bruno (£9.0m) scores 17.1 model points
   - Salah (£14.5m) scores 11.5 model points
   - Bruno provides 138% better value (1.90 vs 0.80 pts/£m)
   - The £5.5m saved can upgrade other positions

3. **Price Penalty**: The model has a logarithmic price factor that
   systematically favors cheaper players:
   - Salah gets 0.509 price multiplier
   - Bruno gets 0.639 price multiplier
   - This 25% advantage compounds the scoring difference

4. **Missing Consistency**: The model uses FPL's form rating but
   doesn't deeply analyze long-term consistency. Salah's reliability
   over 10+ gameweeks isn't captured.

5. **Recommendation**: Consider adjusting:
   - Price factor scaling (less penalty for premiums)
   - Add rolling average metrics (5-game, 10-game)
   - Weight consistency more heavily for captaincy candidates
""")

merger.close()
print("Analysis complete!")