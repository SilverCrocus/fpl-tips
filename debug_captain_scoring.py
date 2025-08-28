#!/usr/bin/env python3
"""Debug script to analyze captain scoring issues"""
import pandas as pd
import numpy as np
from src.models.rule_based_scorer import RuleBasedScorer
from src.my_team import TeamAnalyzer, MyTeam

# Create test data that mimics the real issue
test_data = pd.DataFrame([
    {
        'player_id': 1,
        'player_name': 'Dúbravka',
        'position': 'GK',
        'team_name': 'Newcastle',
        'price': 4.5,
        'total_points': 50,
        'form': 3.2,
        'prob_goal': np.nan,  # Goalkeepers don't have goal probability
        'prob_assist': np.nan,
        'fixture_difficulty': 2.5,
        'chance_of_playing_next_round': 100,
        'expected_goals': np.nan,
        'expected_assists': np.nan,
        'minutes': 90,
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
        'prob_goal': np.nan,  # Goalkeepers don't have goal probability 
        'prob_assist': np.nan,
        'fixture_difficulty': 3.0,
        'chance_of_playing_next_round': 100,
        'expected_goals': np.nan,
        'expected_assists': np.nan,
        'minutes': 90,
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
        'prob_assist': 0.35,
        'fixture_difficulty': 2.0,
        'chance_of_playing_next_round': 50,  # INJURED - 50% chance
        'expected_goals': 0.3,
        'expected_assists': 0.25,
        'minutes': 85,
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
        'prob_assist': 0.15,
        'fixture_difficulty': 2.0,
        'chance_of_playing_next_round': 100,
        'expected_goals': 0.75,
        'expected_assists': 0.12,
        'minutes': 90,
        'has_real_odds': True
    }
])

print("=== DEBUGGING CAPTAIN SCORING ISSUES ===\n")

# 1. Test captain scoring from my_team.py
print("1. Testing Captain Analysis from my_team.py (_analyze_captain method)")
print("-" * 60)

# Simulate the captain analysis function
def debug_captain_analysis(team_data):
    captain_scores = []
    for _, player in team_data.iterrows():
        score = 0
        
        print(f"\nPlayer: {player['player_name']} ({player['position']})")
        
        if 'prob_goal' in player:
            prob_goal_val = player['prob_goal']
            if pd.isna(prob_goal_val):
                print(f"  prob_goal: NaN -> contribution = NaN * 5 = NaN")
                score += np.nan * 5  # This creates NaN!
            else:
                contribution = prob_goal_val * 5
                score += contribution
                print(f"  prob_goal: {prob_goal_val} -> contribution = {contribution}")
        
        if 'form' in player:
            form_val = player['form']
            contribution = form_val * 0.5
            score += contribution
            print(f"  form: {form_val} -> contribution = {contribution}")
        
        if 'fixture_difficulty' in player:
            fix_diff_val = player['fixture_difficulty']
            contribution = (5 - fix_diff_val) * 0.3
            score += contribution
            print(f"  fixture_difficulty: {fix_diff_val} -> contribution = {contribution}")
        
        print(f"  FINAL SCORE: {score}")
        
        captain_scores.append({
            'player': player['player_name'],
            'score': score,
            'position': player['position']
        })
    
    # Sort by score - NaN values will be sorted first/last depending on implementation
    captain_scores = sorted(captain_scores, key=lambda x: x['score'] if pd.notna(x['score']) else -999, reverse=True)
    return captain_scores

captain_scores = debug_captain_analysis(test_data)

print(f"\nCaptain Scores (sorted):")
for i, cap in enumerate(captain_scores):
    print(f"{i+1}. {cap['player']} ({cap['position']}): {cap['score']}")

# 2. Test transfer recommendations
print(f"\n\n2. Testing Transfer Recommendations")
print("-" * 60)

# Create scorer and analyzer
scorer = RuleBasedScorer()
analyzer = TeamAnalyzer(scorer, test_data)

# Create a fake team with the injured player
my_team = MyTeam(
    players=[1, 2, 3, 4],  # Include Saka (injured)
    captain=3,  # Saka is captain
    vice_captain=4,
    bank=1.5,
    free_transfers=1
)

# Get transfer recommendations
print("Getting transfer recommendations...")
recommendations = analyzer.get_transfer_recommendations(my_team, num_transfers=3)

print(f"Found {len(recommendations)} recommendations:")
for i, rec in enumerate(recommendations):
    print(f"\n{i+1}. Transfer: {rec.player_out['name']} -> {rec.player_in['name']}")
    print(f"   Reason: {rec.reason}")
    print(f"   Priority: {rec.priority} (1=urgent, 2=recommended, 3=optional)")
    print(f"   Score improvement: {rec.score_improvement:.2f}")

# 3. Test rule-based scorer for GKs specifically
print(f"\n\n3. Testing Rule-Based Scorer for Goalkeepers")
print("-" * 60)

gk_data = test_data[test_data['position'] == 'GK']
print("GK data before scoring:")
for _, gk in gk_data.iterrows():
    print(f"{gk['player_name']}: prob_goal={gk['prob_goal']}, form={gk['form']}")

scored_gks = scorer.score_all_players(gk_data)
print(f"\nGK data after scoring:")
if not scored_gks.empty:
    for _, gk in scored_gks.iterrows():
        print(f"{gk['player_name']}: model_score={gk['model_score']:.2f}")
else:
    print("No GKs in scored data - they may have been filtered out!")