"""Test transfer recommendation logic with bank balance consideration"""

import sys
import logging
import json

# Configure logging
logging.basicConfig(level=logging.WARNING)

from src.data.data_merger import DataMerger
from src.models.rule_based_scorer import RuleBasedScorer, ScoringWeights
from src.milp_team_manager import MILPTransferOptimizer

# Load scoring weights
def load_weights():
    with open("config/scoring_weights.json") as f:
        weights_data = json.load(f)
    return ScoringWeights.from_dict(weights_data)

# Initialize components
print("Initializing components...")
merger = DataMerger()
data = merger.load_from_database(gameweek=5)  # Load only gameweek 5 data
weights = load_weights()
scorer = RuleBasedScorer(weights)
data = scorer.score_all_players(data)

# Current team IDs from the user's team
current_team_ids = [67, 291, 348, 261, 569, 260, 381, 82, 200, 242, 249, 470, 624, 427, 64]

# Map IDs to player names for clarity
id_to_name = {
    67: "Petrović", 291: "Tarkowski", 348: "Rodon", 261: "Richards", 569: "Romero",
    260: "Guéhi", 381: "M.Salah", 82: "Semenyo", 200: "Anthony", 242: "Dewsbury-Hall",
    249: "João Pedro", 470: "Dúbravka", 624: "Bowen", 427: "Reijnders", 64: "Watkins"
}

print("\nRunning MILP optimization with £5.5m bank...")
optimizer = MILPTransferOptimizer(scorer, data)
result = optimizer.optimize_transfers(
    current_team_ids=current_team_ids,
    budget=5.5,  # £5.5m in bank
    free_transfers=1,
    max_transfers=3,
    horizon_weeks=5
)

print(f"\nOptimization Status: {result.optimization_status}")

if result.transfers_out and result.transfers_in:
    print("\n" + "="*50)
    print("TRANSFERS RECOMMENDED BY MILP")
    print("="*50)

    print("\nTransfers OUT:")
    for p in result.transfers_out:
        orig_name = id_to_name.get(p['id'], "Unknown")
        print(f"  {p['name']} ({orig_name}) [{p['position']}] - £{p['price']}m - Score: {p['score']:.1f}")

    print("\nTransfers IN:")
    for p in result.transfers_in:
        print(f"  {p['name']} [{p['position']}] - £{p['price']}m - Score: {p['score']:.1f}")

    # Analyze affordability
    total_out = sum(p['price'] for p in result.transfers_out)
    total_in = sum(p['price'] for p in result.transfers_in)

    print("\n" + "-"*50)
    print("AFFORDABILITY ANALYSIS")
    print("-"*50)
    print(f"Bank balance: £5.5m")
    print(f"Total sale value: £{total_out:.1f}m")
    print(f"Total funds available: £{5.5 + total_out:.1f}m")
    print(f"Total purchase cost: £{total_in:.1f}m")
    print(f"✅ Can afford all transfers: {(5.5 + total_out) >= total_in}")

    print("\n" + "-"*50)
    print("EXPECTED OUTCOME")
    print("-"*50)
    print(f"Points gain: {result.expected_points_gain:.1f}")
    print(f"New team value: £{result.new_team_value:.1f}m")

    # Check if Watkins -> Haaland was considered
    print("\n" + "-"*50)
    print("SPECIFIC TRANSFER CHECK")
    print("-"*50)

    # Find Haaland in data
    haaland = data[data['player_name'].str.contains('Haaland', case=False, na=False)]
    if not haaland.empty:
        haaland_row = haaland.iloc[0]
        print(f"Haaland price: £{haaland_row['price']}m")
        print(f"Watkins price: £8.8m")
        print(f"Watkins + Bank = £8.8m + £5.5m = £14.3m")
        print(f"Can afford Watkins -> Haaland: {(8.8 + 5.5) >= haaland_row['price']}")

        # Check if Watkins was transferred out
        watkins_out = any(p['id'] == 64 for p in result.transfers_out)
        print(f"Watkins transferred out: {watkins_out}")

        # Check if Haaland was transferred in
        haaland_in = any('Haaland' in p['name'] for p in result.transfers_in)
        print(f"Haaland transferred in: {haaland_in}")
else:
    print("\nNo transfers recommended")

merger.close()
print("\nTest complete!")