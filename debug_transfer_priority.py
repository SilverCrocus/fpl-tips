#!/usr/bin/env python3
"""Debug script to analyze transfer recommendation priority issues"""
import pandas as pd
import numpy as np
from src.models.rule_based_scorer import RuleBasedScorer
from src.my_team import TeamAnalyzer, MyTeam

# Create test data focusing on the injury issue
test_data = pd.DataFrame([
    {
        'player_id': 1,
        'player_name': 'Dúbravka',
        'position': 'GK',
        'team_name': 'Newcastle',
        'price': 4.5,
        'total_points': 50,
        'form': 3.2,
        'prob_goal': np.nan,
        'chance_of_playing_next_round': 100,  # Fit
        'is_available': 1,
        'model_score': 5.0,
        'has_real_odds': True
    },
    {
        'player_id': 2, 
        'player_name': 'Petrović',
        'position': 'GK',
        'team_name': 'Chelsea',
        'price': 4.0,
        'total_points': 35,
        'form': 2.8,
        'prob_goal': np.nan,
        'chance_of_playing_next_round': 100,  # Fit
        'is_available': 1,
        'model_score': 3.0,
        'has_real_odds': True
    },
    {
        'player_id': 3,
        'player_name': 'Saka',
        'position': 'MID', 
        'team_name': 'Arsenal',
        'price': 10.0,
        'total_points': 120,
        'form': 6.5,
        'prob_goal': 0.25,
        'chance_of_playing_next_round': 50,  # INJURED - 50% chance
        'is_available': 0,  # Unavailable
        'model_score': 8.0,  # Good player when fit
        'has_real_odds': True
    },
    {
        'player_id': 4,
        'player_name': 'Haaland',
        'position': 'FWD',
        'team_name': 'Man City', 
        'price': 15.0,
        'total_points': 180,
        'form': 8.2,
        'prob_goal': 0.65,
        'chance_of_playing_next_round': 100,  # Fit
        'is_available': 1,
        'model_score': 12.0,
        'has_real_odds': True
    },
    # Alternative players for transfers
    {
        'player_id': 5,
        'player_name': 'Alternative GK',
        'position': 'GK',
        'team_name': 'Brighton',
        'price': 4.5,
        'total_points': 45,
        'form': 4.0,
        'prob_goal': np.nan,
        'chance_of_playing_next_round': 100,
        'is_available': 1,
        'model_score': 6.0,  # Better than current GKs
        'has_real_odds': True
    },
    {
        'player_id': 6,
        'player_name': 'Alternative MID',
        'position': 'MID',
        'team_name': 'Liverpool',
        'price': 9.5,
        'total_points': 100,
        'form': 7.0,
        'prob_goal': 0.30,
        'chance_of_playing_next_round': 100,
        'is_available': 1,
        'model_score': 10.0,  # Good replacement
        'has_real_odds': True
    }
])

print("=== DEBUGGING TRANSFER RECOMMENDATION PRIORITY ===\n")

# Create scorer and analyzer
scorer = RuleBasedScorer()
analyzer = TeamAnalyzer(scorer, test_data)

# Create team with all test players
my_team = MyTeam(
    players=[1, 2, 3, 4],  # Include Saka (injured)
    captain=3,  # Saka is captain
    vice_captain=4,
    bank=2.0,
    free_transfers=2
)

print("Team players:")
team_data = test_data[test_data['player_id'].isin(my_team.players)]
for _, player in team_data.iterrows():
    availability = "INJURED/UNAVAILABLE" if player['chance_of_playing_next_round'] < 75 or player['is_available'] == 0 else "Available"
    print(f"  {player['player_name']} ({player['position']}): {availability} - Score: {player['model_score']}")

print(f"\nAnalyzing transfer logic...")

# Manually test the transfer recommendation logic
print("\n1. Testing _get_transfer_priority method:")

def debug_transfer_priority(player_out_row, player_in_row):
    """Debug version of _get_transfer_priority"""
    print(f"\nTransfer: {player_out_row['player_name']} -> {player_in_row['player_name']}")
    
    # Urgent: Player unavailable or very poor form
    if player_out_row.get('is_available', 1) == 0:
        print(f"  URGENT (Priority 1): Player unavailable (is_available={player_out_row.get('is_available')})")
        return 1
        
    if player_out_row.get('chance_of_playing_next_round', 100) < 75:
        print(f"  URGENT (Priority 1): Low playing chance ({player_out_row.get('chance_of_playing_next_round')}%)")
        return 1
        
    if player_out_row.get('form', 5) < 1:
        print(f"  URGENT (Priority 1): Very poor form ({player_out_row.get('form')})")
        return 1
        
    # Recommended: Significant score improvement
    score_diff = player_in_row.get('model_score', 0) - player_out_row.get('model_score', 0)
    if score_diff > 3:
        print(f"  RECOMMENDED (Priority 2): Significant score improvement ({score_diff:.2f})")
        return 2
        
    # Optional: Minor improvements
    print(f"  OPTIONAL (Priority 3): Minor improvement ({score_diff:.2f})")
    return 3

# Test each potential transfer
team_players = test_data[test_data['player_id'].isin(my_team.players)]
non_team_players = test_data[~test_data['player_id'].isin(my_team.players)]

print("Testing priority for each potential transfer:")
for _, player_out in team_players.iterrows():
    # Find potential replacements
    same_position = non_team_players[non_team_players['position'] == player_out['position']]
    
    for _, player_in in same_position.iterrows():
        priority = debug_transfer_priority(player_out, player_in)

print(f"\n2. Testing actual get_transfer_recommendations method:")
recommendations = analyzer.get_transfer_recommendations(my_team, num_transfers=3)

print(f"Found {len(recommendations)} recommendations:")
for i, rec in enumerate(recommendations):
    print(f"\n{i+1}. Transfer: {rec.player_out['name']} -> {rec.player_in['name']}")
    print(f"   Reason: {rec.reason}")
    print(f"   Priority: {rec.priority} (1=urgent, 2=recommended, 3=optional)")
    print(f"   Score improvement: {rec.score_improvement:.2f}")
    print(f"   Net cost: £{rec.net_cost:.1f}m")

# Debug the sorting logic
print(f"\n3. Checking sorting logic (line 238 in my_team.py):")
print("Sort key: (-x.priority, -x.score_improvement)")
print("This means: Higher priority first, then higher score improvement")

if recommendations:
    print("Sorting test:")
    for rec in recommendations:
        sort_key = (-rec.priority, -rec.score_improvement)
        print(f"  {rec.player_out['name']}: priority={rec.priority}, improvement={rec.score_improvement:.2f}, sort_key={sort_key}")