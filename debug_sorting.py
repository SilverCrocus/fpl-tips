#!/usr/bin/env python3
"""Debug script to analyze sorting issues with NaN values"""
import pandas as pd
import numpy as np

# Test data with NaN values
captain_scores = [
    {'player': 'Dúbravka', 'score': np.nan},
    {'player': 'Petrović', 'score': np.nan},
    {'player': 'Saka', 'score': 5.4},
    {'player': 'Haaland', 'score': 8.25}
]

print("=== DEBUGGING SORTING WITH NaN VALUES ===\n")

print("Original data:")
for item in captain_scores:
    print(f"  {item['player']}: {item['score']}")

print(f"\nSorting with reverse=True (as done in my_team.py line 162):")
sorted_scores = sorted(captain_scores, key=lambda x: x['score'], reverse=True)
for i, item in enumerate(sorted_scores):
    print(f"  {i+1}. {item['player']}: {item['score']}")

print(f"\nThis is the BUG! NaN values are sorted first when reverse=True")
print("In Python, NaN comparisons always return False, so:")
print(f"  np.nan > 8.25 = {np.nan > 8.25}")  
print(f"  np.nan < 8.25 = {np.nan < 8.25}")
print(f"  np.nan == np.nan = {np.nan == np.nan}")

# Test main.py sorting with pandas
print(f"\n=== TESTING MAIN.PY CAPTAIN SORTING ===")

# Simulate main.py captain analysis (lines 508-534)
candidates = pd.DataFrame([
    {'player_name': 'Dúbravka', 'prob_goal': np.nan, 'form': 3.2, 'minutes': 90, 'position': 'GK'},
    {'player_name': 'Petrović', 'prob_goal': np.nan, 'form': 2.8, 'minutes': 90, 'position': 'GK'},
    {'player_name': 'Saka', 'prob_goal': 0.25, 'form': 6.5, 'minutes': 85, 'position': 'MID'},
    {'player_name': 'Haaland', 'prob_goal': 0.65, 'form': 8.2, 'minutes': 90, 'position': 'FWD'}
])

# Apply the exact logic from main.py lines 508-531
captain_score = pd.Series(0.0, index=candidates.index)

if 'prob_goal' in candidates.columns:
    captain_score = captain_score + candidates['prob_goal'].fillna(0) * 5.0  # This fixes NaN!

if 'form' in candidates.columns:
    captain_score = captain_score + candidates['form'].fillna(0) * 0.3

candidates['captain_score'] = captain_score

# Get top 10 captain choices (line 534)
top_captains = candidates.nlargest(10, 'captain_score')

print("Main.py results (uses fillna(0) - correct!):")
for i, (_, player) in enumerate(top_captains.iterrows(), 1):
    print(f"  {i}. {player['player_name']} ({player['position']}): {player['captain_score']:.2f}")

print(f"\n=== COMPARISON: my_team.py vs main.py ===")
print("my_team.py (_analyze_captain): Uses raw NaN values -> NaN scores -> wrong sorting")
print("main.py (captain_analysis): Uses fillna(0) -> numeric scores -> correct sorting")