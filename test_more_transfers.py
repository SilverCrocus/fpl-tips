"""Test MILP with different transfer limits to see budget usage"""

import json
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)

from src.data.data_merger import DataMerger
from src.models.rule_based_scorer import RuleBasedScorer, ScoringWeights
from src.milp_team_manager import MILPTransferOptimizer

def load_weights():
    with open("config/scoring_weights.json") as f:
        weights_data = json.load(f)
    return ScoringWeights.from_dict(weights_data)

print("="*70)
print("TESTING BUDGET USAGE WITH DIFFERENT TRANSFER LIMITS")
print("="*70)

# Initialize
merger = DataMerger()
data = merger.load_from_database(gameweek=5)
weights = load_weights()
scorer = RuleBasedScorer(weights)
data = scorer.score_all_players(data)

# Your current team
current_team_ids = [67, 291, 348, 261, 569, 260, 381, 82, 200, 242, 249, 470, 624, 427, 64]

# Test with different numbers of allowed transfers
for max_transfers in [3, 4, 5, 6]:
    print(f"\n{'='*70}")
    print(f"Testing with {max_transfers} transfers allowed")
    print('-'*70)

    optimizer = MILPTransferOptimizer(scorer, data)
    result = optimizer.optimize_transfers(
        current_team_ids=current_team_ids,
        budget=5.5,
        free_transfers=1,
        max_transfers=max_transfers,
        horizon_weeks=5
    )

    if result.transfers_out and result.transfers_in:
        total_out = sum(p['price'] for p in result.transfers_out)
        total_in = sum(p['price'] for p in result.transfers_in)
        net_spend = total_in - total_out
        unspent = 5.5 - net_spend

        print(f"Transfers made: {len(result.transfers_out)}")
        print(f"Money OUT: £{total_out:.1f}m")
        print(f"Money IN: £{total_in:.1f}m")
        print(f"Net spend: £{net_spend:.1f}m")
        print(f"Unspent: £{unspent:.1f}m")
        print(f"Points gain: {result.expected_points_gain:.1f}")

        if len(result.transfers_out) > 3:
            print("\nExtra transfers beyond 3:")
            for i in range(3, len(result.transfers_out)):
                print(f"  OUT: {result.transfers_out[i]['name']} (£{result.transfers_out[i]['price']}m)")
                print(f"  IN:  {result.transfers_in[i]['name']} (£{result.transfers_in[i]['price']}m)")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

print("""
If the model uses more budget with more transfers allowed, it shows:
1. The penalty IS working to encourage spending
2. With 3-transfer limit, there may not be good 4th option to use £5.5m on

If it still doesn't spend with more transfers:
1. The current team might already have the best value players
2. Available upgrades might not justify their cost
""")

merger.close()
print("\nTest complete!")