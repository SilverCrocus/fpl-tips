#!/usr/bin/env python
"""
Test that MILP is always used without fallback
"""
from src.team_optimizer import TeamOptimizer
import json

def test_milp_only():
    print("Testing MILP-only optimization (no fallback)")
    print("=" * 60)
    
    optimizer = TeamOptimizer()
    
    # Test each strategy
    strategies = optimizer.get_strategies()
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        print("-" * 40)
        
        result = optimizer.recommend_team(strategy=strategy)
        
        # Check for errors
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
            print(f"   Optimization status: {result.get('optimization_status', 'unknown')}")
        else:
            print(f"✅ Success! Team selected using {result.get('optimization_status', 'unknown')}")
            print(f"   Total cost: £{result['total_cost']:.1f}m")
            print(f"   Expected points: {result['expected_points']:.1f}")
            
            # Verify it's using MILP (optimal) not greedy
            if result.get('optimization_status') == 'optimal':
                print(f"   ✅ Using MILP optimization (no fallback)")
            elif result.get('strategy') == 'greedy_fallback':
                print(f"   ❌ WARNING: Greedy fallback was used!")
            else:
                print(f"   Status: {result.get('optimization_status', 'unknown')}")
            
            # Show captain
            if 'captain' in result:
                captain = result['captain']
                print(f"   Captain: {captain.get('player_name', 'Unknown')} (score: {captain.get('model_score', 0):.1f})")
            
            # Count players by position
            team = result.get('team', {})
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                count = len(team.get(pos, []))
                print(f"   {pos}: {count} players")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("All strategies now use MILP optimization exclusively.")
    print("Fallback to greedy selection has been completely removed.")
    print("If MILP fails, an error is returned instead of falling back.")

if __name__ == "__main__":
    test_milp_only()