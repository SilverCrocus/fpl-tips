"""Test MILP optimizer with unspent money penalty fix"""

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
print("TESTING MILP WITH UNSPENT MONEY PENALTY")
print("="*70)

# Initialize
merger = DataMerger()
data = merger.load_from_database(gameweek=5)
weights = load_weights()
scorer = RuleBasedScorer(weights)
data = scorer.score_all_players(data)

# Your current team
current_team_ids = [67, 291, 348, 261, 569, 260, 381, 82, 200, 242, 249, 470, 624, 427, 64]

# Run optimization with £5.5m bank
optimizer = MILPTransferOptimizer(scorer, data)
result = optimizer.optimize_transfers(
    current_team_ids=current_team_ids,
    budget=5.5,
    free_transfers=1,
    max_transfers=3,
    horizon_weeks=5
)

print(f"\nOptimization Status: {result.optimization_status}")

if result.transfers_out and result.transfers_in:
    print("\n" + "="*70)
    print("TRANSFERS WITH UNSPENT PENALTY FIX")
    print("="*70)

    print("\nTransfers OUT:")
    total_out = 0
    for p in result.transfers_out:
        print(f"  {p['name']:<20} [{p['position']}] £{p['price']:5.1f}m  Score: {p['score']:5.1f}")
        total_out += p['price']

    print("\nTransfers IN:")
    total_in = 0
    for p in result.transfers_in:
        print(f"  {p['name']:<20} [{p['position']}] £{p['price']:5.1f}m  Score: {p['score']:5.1f}")
        total_in += p['price']

    print("\n" + "-"*70)
    print("BUDGET ANALYSIS")
    print("-"*70)

    net_spend = total_in - total_out
    unspent = 5.5 - net_spend

    print(f"Bank available: £5.5m")
    print(f"Total sales: £{total_out:.1f}m")
    print(f"Total purchases: £{total_in:.1f}m")
    print(f"Net spend: £{net_spend:.1f}m")
    print(f"Money left unspent: £{unspent:.1f}m")

    if unspent < 0.5:
        print(f"✅ EXCELLENT! Model is using budget efficiently (only £{unspent:.1f}m unused)")
    elif unspent < 2.0:
        print(f"✅ GOOD! Most of the budget is being used (£{unspent:.1f}m unused)")
    else:
        print(f"⚠️ Still leaving £{unspent:.1f}m unspent - may need stronger penalty")

    print(f"\nExpected points gain: {result.expected_points_gain:.1f}")

    print("\n" + "="*70)
    print("STRATEGY EXPLANATION")
    print("="*70)

    # Check if we're doing Salah swap
    salah_out = any('Salah' in p['name'] for p in result.transfers_out)
    bruno_in = any('Bruno' in p['name'] or 'Fernandes' in p['name'] for p in result.transfers_in)

    if salah_out and bruno_in:
        print("\n✅ Salah → Bruno swap detected")
        print("The model is now considering what to do with the freed £5.5m:")

        # Find the upgrade
        if len(result.transfers_out) > 1:
            other_outs = [p for p in result.transfers_out if 'Salah' not in p['name']]
            other_ins = [p for p in result.transfers_in if 'Bruno' not in p['name'] and 'Fernandes' not in p['name']]

            if other_outs and other_ins:
                print("\n🔄 Additional upgrades with freed money:")
                for out_p, in_p in zip(other_outs, other_ins):
                    upgrade_cost = in_p['price'] - out_p['price']
                    points_gain = in_p['score'] - out_p['score']
                    print(f"  {out_p['name']} → {in_p['name']}: +£{upgrade_cost:.1f}m for +{points_gain:.1f} points")

                print("\n💡 Smart strategy!")
                print(f"Instead of: Salah alone")
                print(f"You get: Bruno + upgraded player(s)")
                print(f"Net result: Better overall squad with money working for you")
    else:
        print("\nTransfer strategy optimized for maximum points with budget usage")
else:
    print("\nNo transfers recommended")

merger.close()
print("\nTest complete!")