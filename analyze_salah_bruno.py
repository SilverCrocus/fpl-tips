"""Detailed analysis of Salah vs Bruno Fernandes recommendation"""

import json
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.WARNING)

from src.data.data_merger import DataMerger
from src.models.rule_based_scorer import RuleBasedScorer, ScoringWeights

def load_weights():
    """Load scoring weights from config"""
    with open("config/scoring_weights.json") as f:
        weights_data = json.load(f)
    return ScoringWeights.from_dict(weights_data)

print("="*60)
print("SALAH vs BRUNO FERNANDES: DETAILED ANALYSIS")
print("="*60)

# Initialize components
merger = DataMerger()
data = merger.load_from_database(gameweek=5)
weights = load_weights()
scorer = RuleBasedScorer(weights)

# Score all players
data = scorer.score_all_players(data)

# Find Salah and Bruno
salah = data[data['player_name'].str.contains('Salah', case=False, na=False)]
bruno = data[data['player_name'].str.contains('B.Fernandes|Bruno', case=False, na=False)]

if salah.empty:
    print("Salah not found in data!")
if bruno.empty:
    print("Bruno Fernandes not found in data!")

if not salah.empty and not bruno.empty:
    s = salah.iloc[0]
    b = bruno.iloc[0]

    print("\n" + "="*60)
    print("PLAYER COMPARISON")
    print("="*60)

    # Basic stats
    print(f"\n{'Metric':<30} {'Salah':>12} {'Bruno':>12}")
    print("-"*60)
    print(f"{'Price (£m)':<30} {s['price']:>12.1f} {b['price']:>12.1f}")
    print(f"{'Model Score':<30} {s['model_score']:>12.1f} {b['model_score']:>12.1f}")
    print(f"{'Points per Million':<30} {s['model_score']/s['price']:>12.2f} {b['model_score']/b['price']:>12.2f}")
    print(f"{'Total Points (Season)':<30} {s['total_points']:>12.0f} {b['total_points']:>12.0f}")
    print(f"{'Form Rating':<30} {s.get('form', 0):>12.1f} {b.get('form', 0):>12.1f}")
    print(f"{'Minutes Played':<30} {s['minutes']:>12.0f} {b['minutes']:>12.0f}")

    # Performance metrics
    print(f"\n{'Goals Scored':<30} {s['goals_scored']:>12.0f} {b['goals_scored']:>12.0f}")
    print(f"{'Assists':<30} {s['assists']:>12.0f} {b['assists']:>12.0f}")
    print(f"{'Expected Goals (xG)':<30} {s.get('expected_goals', 0):>12.2f} {b.get('expected_goals', 0):>12.2f}")
    print(f"{'Expected Assists (xA)':<30} {s.get('expected_assists', 0):>12.2f} {b.get('expected_assists', 0):>12.2f}")

    # Probability metrics
    s_goal_prob = s.get('prob_goal') if pd.notna(s.get('prob_goal')) else 0
    b_goal_prob = b.get('prob_goal') if pd.notna(b.get('prob_goal')) else 0
    s_assist_prob = s.get('prob_assist') if pd.notna(s.get('prob_assist')) else 0
    b_assist_prob = b.get('prob_assist') if pd.notna(b.get('prob_assist')) else 0

    print(f"\n{'Goal Probability':<30} {s_goal_prob:>12.2f} {b_goal_prob:>12.2f}")
    print(f"{'Assist Probability':<30} {s_assist_prob:>12.2f} {b_assist_prob:>12.2f}")

    # Ownership and transfers
    print(f"\n{'Selected by %':<30} {s.get('selected_by_percent', 0):>12.1f} {b.get('selected_by_percent', 0):>12.1f}")
    print(f"{'Transfers In (GW)':<30} {s.get('transfers_in', 0):>12.0f} {b.get('transfers_in', 0):>12.0f}")

    # Fixture difficulty
    print(f"\n{'Fixture Diff (Next 2)':<30} {s.get('fixture_diff_next2', 0):>12.1f} {b.get('fixture_diff_next2', 0):>12.1f}")
    print(f"{'Fixture Diff (Next 5)':<30} {s.get('fixture_diff_next5', 0):>12.1f} {b.get('fixture_diff_next5', 0):>12.1f}")

    print("\n" + "="*60)
    print("SCORING BREAKDOWN (MID weights)")
    print("="*60)

    # Calculate component scores for each
    mid_weights = weights.mid_weights

    print(f"\n{'Component':<30} {'Weight':>8} {'Salah':>12} {'Bruno':>12}")
    print("-"*60)

    # Form component
    form_weight = mid_weights.get('form', 0)
    form_score_s = s.get('form', 0) * form_weight
    form_score_b = b.get('form', 0) * form_weight
    print(f"{'Form * weight':<30} {form_weight:>8.1f} {form_score_s:>12.2f} {form_score_b:>12.2f}")

    # Goal probability component
    goal_weight = mid_weights.get('goal_probability', 0)
    goal_score_s = (s.get('prob_goal') if pd.notna(s.get('prob_goal')) else 0) * goal_weight
    goal_score_b = (b.get('prob_goal') if pd.notna(b.get('prob_goal')) else 0) * goal_weight
    print(f"{'Goal Prob * weight':<30} {goal_weight:>8.1f} {goal_score_s:>12.2f} {goal_score_b:>12.2f}")

    # Assist probability component
    assist_weight = mid_weights.get('assist_probability', 0)
    assist_score_s = (s.get('prob_assist') if pd.notna(s.get('prob_assist')) else 0) * assist_weight
    assist_score_b = (b.get('prob_assist') if pd.notna(b.get('prob_assist')) else 0) * assist_weight
    print(f"{'Assist Prob * weight':<30} {assist_weight:>8.1f} {assist_score_s:>12.2f} {assist_score_b:>12.2f}")

    # xG component
    xg_weight = mid_weights.get('expected_goals', 0)
    xg_score_s = s.get('expected_goals', 0) * xg_weight
    xg_score_b = b.get('expected_goals', 0) * xg_weight
    print(f"{'xG * weight':<30} {xg_weight:>8.1f} {xg_score_s:>12.2f} {xg_score_b:>12.2f}")

    # xA component
    xa_weight = mid_weights.get('expected_assists', 0)
    xa_score_s = s.get('expected_assists', 0) * xa_weight
    xa_score_b = b.get('expected_assists', 0) * xa_weight
    print(f"{'xA * weight':<30} {xa_weight:>8.1f} {xa_score_s:>12.2f} {xa_score_b:>12.2f}")

    # Creativity component
    creativity_weight = mid_weights.get('creativity', 0)
    creativity_score_s = s.get('creativity', 0) * creativity_weight
    creativity_score_b = b.get('creativity', 0) * creativity_weight
    print(f"{'Creativity * weight':<30} {creativity_weight:>8.3f} {creativity_score_s:>12.2f} {creativity_score_b:>12.2f}")

    print("\n" + "="*60)
    print("PRICE FACTOR ANALYSIS")
    print("="*60)

    # Price factor calculation (from line 195 of rule_based_scorer.py)
    min_price = 4.0
    max_price = 15.0

    # Calculate logarithmic price factors
    salah_price_factor = 0.5 + 0.5 * (1 - np.log(s['price'] - min_price + 1) / np.log(max_price - min_price + 1))
    bruno_price_factor = 0.5 + 0.5 * (1 - np.log(b['price'] - min_price + 1) / np.log(max_price - min_price + 1))

    print(f"\nPrice Factor (favors cheaper):")
    print(f"  Salah (£{s['price']}m): {salah_price_factor:.3f}")
    print(f"  Bruno (£{b['price']}m): {bruno_price_factor:.3f}")
    print(f"  Difference: {bruno_price_factor - salah_price_factor:.3f} advantage to Bruno")

    # Calculate impact on final score
    base_score_s = form_score_s + goal_score_s + assist_score_s + xg_score_s + xa_score_s + creativity_score_s
    base_score_b = form_score_b + goal_score_b + assist_score_b + xg_score_b + xa_score_b + creativity_score_b

    print(f"\nBase score (before price factor):")
    print(f"  Salah: {base_score_s:.2f}")
    print(f"  Bruno: {base_score_b:.2f}")

    print(f"\nFinal score (after price factor):")
    print(f"  Salah: {base_score_s:.2f} * {salah_price_factor:.3f} = {base_score_s * salah_price_factor:.2f}")
    print(f"  Bruno: {base_score_b:.2f} * {bruno_price_factor:.3f} = {base_score_b * bruno_price_factor:.2f}")

    print("\n" + "="*60)
    print("CONSISTENCY ANALYSIS")
    print("="*60)

    # Check if we have gameweek data
    gw_cols = [col for col in data.columns if col.startswith('gw') and col.endswith('_points')]
    if gw_cols:
        print(f"\nFound gameweek data columns: {gw_cols}")
        print("\nRecent gameweek points:")
        for col in sorted(gw_cols)[-5:]:  # Last 5 gameweeks
            if col in s.index and col in b.index:
                print(f"  {col}: Salah={s[col]:.0f}, Bruno={b[col]:.0f}")
    else:
        print("\nNo gameweek-specific point columns found")
        print("Using form rating as consistency proxy")

    print("\n" + "="*60)
    print("WHY BRUNO OVER SALAH?")
    print("="*60)

    score_diff = b['model_score'] - s['model_score']
    if score_diff > 0:
        print(f"\n✅ Bruno scores {score_diff:.1f} points higher")
        print("\nKey factors:")

        if b.get('form', 0) > s.get('form', 0):
            print(f"  • Better recent form ({b.get('form', 0):.1f} vs {s.get('form', 0):.1f})")

        if bruno_price_factor > salah_price_factor:
            print(f"  • Price advantage (£{b['price']}m vs £{s['price']}m)")
            print(f"    Price factor bonus: {(bruno_price_factor - salah_price_factor):.3f}")

        b_goal = b.get('prob_goal') if pd.notna(b.get('prob_goal')) else 0
        s_goal = s.get('prob_goal') if pd.notna(s.get('prob_goal')) else 0
        if b_goal > s_goal:
            print(f"  • Higher goal probability ({b_goal:.2f} vs {s_goal:.2f})")

        if b.get('fixture_diff_next5', 0) < s.get('fixture_diff_next5', 0):
            print(f"  • Easier fixtures ({b.get('fixture_diff_next5', 0):.1f} vs {s.get('fixture_diff_next5', 0):.1f})")

        # Value calculation
        print(f"\n💰 Value comparison:")
        print(f"  Salah: {s['model_score']/s['price']:.2f} pts/£m")
        print(f"  Bruno: {b['model_score']/b['price']:.2f} pts/£m")
        print(f"  Bruno provides {(b['model_score']/b['price']) / (s['model_score']/s['price']) - 1:.1%} better value")

        # Budget freed up
        print(f"\n💵 Budget impact:")
        print(f"  Selling Salah frees up £{s['price'] - b['price']:.1f}m")
        print(f"  This allows upgrading other positions")

    else:
        print(f"\n❌ Salah actually scores {-score_diff:.1f} points higher")
        print("The recommendation may be driven by budget optimization")

merger.close()
print("\nAnalysis complete!")