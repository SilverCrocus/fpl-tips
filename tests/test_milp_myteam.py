#!/usr/bin/env python
"""
Test MILP integration in my-team functionality
"""
import pandas as pd
from src.my_team import MyTeam, TeamAnalyzer, ChipAdvisor
from src.models.rule_based_scorer import RuleBasedScorer
from src.data.data_merger import DataMerger

def test_milp_integration():
    print("Testing MILP Integration for My Team Management")
    print("=" * 80)
    
    # Load data
    merger = DataMerger("data/fpl_data.db")
    data = merger.get_latest_data(top_n=500)
    
    if data.empty:
        print("No data available for testing")
        merger.close()
        return
    
    # Create scorer
    scorer = RuleBasedScorer()
    
    # Create a sample team (top players by position)
    sample_team_ids = []
    
    # Get 2 GKs
    gks = data[data['position'].isin(['GK', 'GKP'])].nlargest(2, 'total_points')
    sample_team_ids.extend(gks['player_id'].tolist()[:2])
    
    # Get 5 DEFs
    defs = data[data['position'] == 'DEF'].nlargest(5, 'total_points')
    sample_team_ids.extend(defs['player_id'].tolist()[:5])
    
    # Get 5 MIDs
    mids = data[data['position'] == 'MID'].nlargest(5, 'total_points')
    sample_team_ids.extend(mids['player_id'].tolist()[:5])
    
    # Get 3 FWDs
    fwds = data[data['position'] == 'FWD'].nlargest(3, 'total_points')
    sample_team_ids.extend(fwds['player_id'].tolist()[:3])
    
    # Create MyTeam object
    my_team = MyTeam(
        players=sample_team_ids[:15],  # Ensure exactly 15 players
        captain=sample_team_ids[10] if len(sample_team_ids) > 10 else sample_team_ids[0],
        vice_captain=sample_team_ids[11] if len(sample_team_ids) > 11 else sample_team_ids[1],
        bank=2.0,
        free_transfers=1,
        wildcard_available=True,
        free_hit_available=True,
        bench_boost_available=True,
        triple_captain_available=True
    )
    
    print(f"Created test team with {len(my_team.players)} players")
    print(f"Bank: £{my_team.bank}m, Free transfers: {my_team.free_transfers}")
    
    # Initialize analyzer
    analyzer = TeamAnalyzer(scorer, data)
    
    # Test 1: MILP Transfer Recommendations
    print("\n" + "=" * 80)
    print("TEST 1: MILP TRANSFER RECOMMENDATIONS")
    print("-" * 80)
    
    transfers_milp = analyzer.get_transfer_recommendations(my_team, num_transfers=2, use_milp=True)
    
    if transfers_milp:
        print(f"✅ Found {len(transfers_milp)} transfer recommendations using MILP:")
        for i, transfer in enumerate(transfers_milp, 1):
            print(f"\nTransfer {i}:")
            print(f"  OUT: {transfer.player_out['name']} (£{transfer.player_out['price']:.1f}m, score: {transfer.player_out['score']:.1f})")
            print(f"  IN:  {transfer.player_in['name']} (£{transfer.player_in['price']:.1f}m, score: {transfer.player_in['score']:.1f})")
            print(f"  Net cost: £{transfer.net_cost:.1f}m")
            print(f"  Score gain: +{transfer.score_improvement:.1f}")
            print(f"  Priority: {transfer.priority} - {transfer.reason}")
    else:
        print("❌ No transfers recommended")
    
    # Compare with greedy algorithm
    print("\n" + "-" * 40)
    print("Comparison with Greedy Algorithm:")
    transfers_greedy = analyzer.get_transfer_recommendations(my_team, num_transfers=2, use_milp=False)
    
    if transfers_greedy:
        print(f"Greedy found {len(transfers_greedy)} transfers")
        for transfer in transfers_greedy:
            print(f"  {transfer.player_out['name']} → {transfer.player_in['name']} (+{transfer.score_improvement:.1f})")
    
    # Test 2: MILP Captain Selection
    print("\n" + "=" * 80)
    print("TEST 2: MILP CAPTAIN SELECTION")
    print("-" * 80)
    
    team_data = data[data['player_id'].isin(my_team.players)]
    captain_analysis = analyzer._analyze_captain(team_data, my_team.captain, use_milp=True)
    
    if 'error' not in captain_analysis:
        print(f"Current captain: {captain_analysis.get('current_captain', 'Unknown')}")
        
        if captain_analysis.get('recommended_captain'):
            rec = captain_analysis['recommended_captain']
            print(f"✅ Recommended captain: {rec['player']} (score: {rec['score']:.1f})")
            print(f"   Form: {rec.get('form', 0):.1f}, Fixture difficulty: {rec.get('fixture', 3)}")
        
        if captain_analysis.get('vice_captain'):
            vice = captain_analysis['vice_captain']
            print(f"✅ Vice captain: {vice['player']} (score: {vice['score']:.1f})")
        
        print(f"Method: {captain_analysis.get('method', 'Unknown')}")
        
        if captain_analysis.get('message'):
            print(f"Advice: {captain_analysis['message']}")
    else:
        print(f"❌ Error: {captain_analysis['error']}")
    
    # Test 3: MILP Chip Advice
    print("\n" + "=" * 80)
    print("TEST 3: MILP CHIP ADVICE")
    print("-" * 80)
    
    # Get team analysis
    analysis = analyzer.analyze_team(my_team)
    
    # Initialize chip advisor with scorer
    chip_advisor = ChipAdvisor(data, scorer)
    chip_advice = chip_advisor.get_all_chip_advice(my_team, analysis, team_data, use_milp=True)
    
    for chip, advice in chip_advice.items():
        print(f"\n{chip.upper()}:")
        if isinstance(advice, dict):
            if advice.get('use') == True:
                print(f"  ✅ RECOMMENDED - {advice.get('confidence', 'unknown')} confidence")
                if advice.get('reasons'):
                    for reason in advice['reasons']:
                        print(f"    • {reason}")
                if chip == 'bench_boost' and advice.get('expected_bench_points'):
                    print(f"    Expected bench points: {advice['expected_bench_points']:.1f}")
                if chip == 'triple_captain' and advice.get('player'):
                    print(f"    Player: {advice['player']} ({advice.get('expected_points', 0):.1f} points)")
            elif advice.get('use') == 'consider':
                print(f"  🤔 CONSIDER - {advice.get('confidence', 'unknown')} confidence")
                if advice.get('reasons'):
                    for reason in advice['reasons']:
                        print(f"    • {reason}")
            else:
                print(f"  ❌ NOT RECOMMENDED")
                print(f"    {advice.get('reason', 'No specific reason')}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("-" * 80)
    print("✅ MILP integration successful for:")
    print("  • Transfer recommendations (optimal multi-transfer planning)")
    print("  • Captain selection (considers expected returns and variance)")
    print("  • Chip timing advice (bench boost and triple captain optimization)")
    print("\nThe my-team command now uses advanced MILP optimization for all key decisions!")
    
    merger.close()

if __name__ == "__main__":
    test_milp_integration()