"""Debug MILP optimization failure"""

import sys
import logging
import json

# Configure logging to see more details
logging.basicConfig(level=logging.INFO)

from src.data.data_merger import DataMerger
from src.models.rule_based_scorer import RuleBasedScorer, ScoringWeights

# Load scoring weights
def load_weights():
    with open("config/scoring_weights.json") as f:
        weights_data = json.load(f)
    return ScoringWeights.from_dict(weights_data)

# Initialize components
print("Initializing components...")
merger = DataMerger()
data = merger.load_from_database()
weights = load_weights()
scorer = RuleBasedScorer(weights)
data = scorer.score_all_players(data)

# Current team IDs from the user's team
current_team_ids = [67, 291, 348, 261, 569, 260, 381, 82, 200, 242, 249, 470, 624, 427, 64]

# Check current team details
print("\n" + "="*50)
print("CURRENT TEAM ANALYSIS")
print("="*50)

current_team = data[data['player_id'].isin(current_team_ids)]
print(f"Players found in database: {len(current_team)}/{len(current_team_ids)}")

if len(current_team) < len(current_team_ids):
    missing = set(current_team_ids) - set(current_team['player_id'].values)
    print(f"Missing player IDs: {missing}")

# Check team composition
position_counts = current_team['position'].value_counts()
print("\nCurrent team composition:")
for pos in ['GK', 'DEF', 'MID', 'FWD']:
    count = position_counts.get(pos, 0)
    print(f"  {pos}: {count}")

# Check team value and constraints
total_value = current_team['price'].sum()
print(f"\nTotal team value: £{total_value:.1f}m")
print(f"Bank: £5.5m")
print(f"Total budget: £{total_value + 5.5:.1f}m")

# Check for specific players
print("\nKey players in team:")
for pid, expected_name in [(381, "Salah"), (64, "Watkins"), (249, "João Pedro")]:
    player = current_team[current_team['player_id'] == pid]
    if not player.empty:
        p = player.iloc[0]
        print(f"  {expected_name}: £{p['price']}m, Score: {p['model_score']:.1f}")
    else:
        print(f"  {expected_name} (ID {pid}): NOT FOUND")

# Check for Haaland in database
haaland = data[data['player_name'].str.contains('Haaland', case=False, na=False)]
if not haaland.empty:
    h = haaland.iloc[0]
    print(f"\nHaaland available: £{h['price']}m, Score: {h['model_score']:.1f}")
else:
    print("\nHaaland NOT found in database!")

# Check available players with good scores
print("\nTop 10 available players not in team:")
available = data[~data['player_id'].isin(current_team_ids)].nlargest(10, 'model_score')
for _, p in available.iterrows():
    print(f"  {p['player_name']} ({p['position']}): £{p['price']}m, Score: {p['model_score']:.1f}")

# Check if we can afford Watkins -> Haaland swap
if not haaland.empty:
    watkins = current_team[current_team['player_id'] == 64]
    if not watkins.empty:
        w = watkins.iloc[0]
        h = haaland.iloc[0]
        funds_after_sale = 5.5 + w['price']
        print(f"\n" + "-"*50)
        print("WATKINS -> HAALAND FEASIBILITY")
        print("-"*50)
        print(f"Sell Watkins: +£{w['price']}m")
        print(f"Current bank: £5.5m")
        print(f"Total funds: £{funds_after_sale:.1f}m")
        print(f"Haaland cost: £{h['price']}m")
        print(f"Affordable: {funds_after_sale >= h['price']}")
        print(f"Score gain: {h['model_score'] - w['model_score']:.1f}")

merger.close()
print("\nDebug complete!")