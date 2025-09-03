"""
Strategic Chip Timing Module
Implements FPL best practices for chip usage timing
"""
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ChipStrategy:
    """Strategic rules for chip usage"""
    
    # Triple Captain: ONLY in DGW with premium captain
    TRIPLE_CAPTAIN_MIN_DGW_SCORE = 25  # Minimum expected score for TC in DGW
    TRIPLE_CAPTAIN_MIN_SGW_SCORE = 35  # Almost impossible threshold for SGW
    
    # Bench Boost: Best in big DGW
    BENCH_BOOST_MIN_BENCH_SCORE = 20  # Minimum expected bench points
    BENCH_BOOST_DGW_PREFERRED = True
    
    # Wildcard: Reactive or setup for DGW/BGW
    WILDCARD_MIN_ISSUES = 4  # Minimum team problems to trigger
    
    # Free Hit: Primarily for BGW
    FREE_HIT_MIN_PLAYING = 8  # Minimum players needed without FH


class StrategicChipAdvisor:
    """
    Strategic chip advisor that considers:
    - Double Gameweeks (DGW) 
    - Blank Gameweeks (BGW)
    - Long-term planning
    - FPL community best practices
    """
    
    def __init__(self, data: pd.DataFrame, scorer=None):
        self.data = data
        self.scorer = scorer
        self.strategy = ChipStrategy()
    
    def detect_gameweek_type(self, team_data: pd.DataFrame) -> Dict:
        """
        Detect if current/upcoming gameweeks are DGW or BGW
        
        Returns:
            Dict with gameweek type info
        """
        # Check for DGW indicators
        is_dgw = False
        dgw_players = []
        
        # Look for players with multiple fixtures (simplified check)
        # In real implementation, you'd check fixture lists
        if 'is_dgw' in team_data.columns:
            is_dgw = team_data['is_dgw'].any()
            dgw_players = team_data[team_data['is_dgw'] == True]['player_name'].tolist()
        
        # Check for BGW indicators (teams with no fixtures)
        is_bgw = False
        playing_count = len(team_data)
        
        if 'has_fixture' in team_data.columns:
            playing_count = team_data['has_fixture'].sum()
            is_bgw = playing_count < 11  # Less than a full team
        
        return {
            'is_dgw': is_dgw,
            'dgw_players': dgw_players,
            'is_bgw': is_bgw,
            'playing_count': playing_count,
            'gameweek_type': 'DGW' if is_dgw else ('BGW' if is_bgw else 'Regular')
        }
    
    def evaluate_triple_captain_strategic(self, 
                                         team_data: pd.DataFrame,
                                         captain_analysis: Dict) -> Dict:
        """
        Strategic Triple Captain evaluation
        
        Key Rules:
        1. STRONG PREFERENCE for DGW with premium captain
        2. ONLY consider SGW if exceptional circumstances
        3. Consider future DGW opportunities
        """
        # Get gameweek type
        gw_info = self.detect_gameweek_type(team_data)
        
        # Get best captain option
        if captain_analysis and captain_analysis.get('recommended_captain'):
            captain = captain_analysis['recommended_captain']
            captain_score = captain.get('score', 0)
            captain_name = captain.get('player', 'Unknown')
        else:
            return {
                'use': False,
                'reason': 'No clear captain option',
                'strategic_advice': 'Save for Double Gameweek'
            }
        
        # Check if captain is premium (Haaland, Salah, etc)
        premium_captains = ['Haaland', 'Salah', 'Palmer', 'Saka', 'Son']
        is_premium_captain = any(name in captain_name for name in premium_captains)
        
        reasons = []
        strategic_score = 0
        
        # RULE 1: Double Gameweek is STRONGLY preferred
        if gw_info['is_dgw']:
            reasons.append("🎯 DOUBLE GAMEWEEK - Players play TWICE!")
            strategic_score += 10
            
            if captain_name in gw_info.get('dgw_players', []):
                reasons.append(f"✅ {captain_name} plays twice this week")
                strategic_score += 5
                
                # Expected points are doubled for DGW
                expected_points = captain_score * 2 * 3  # 2 games * 3 (TC multiplier)
                
                if is_premium_captain:
                    reasons.append(f"⭐ Premium captain in DGW - OPTIMAL timing!")
                    strategic_score += 5
                
                if expected_points >= self.strategy.TRIPLE_CAPTAIN_MIN_DGW_SCORE:
                    return {
                        'use': True,
                        'confidence': 'VERY HIGH',
                        'player': captain_name,
                        'expected_points': expected_points,
                        'reasons': reasons,
                        'strategic_advice': '🎯 PERFECT TIMING - DGW with strong captain!'
                    }
        
        # RULE 2: Single Gameweek - VERY HIGH bar
        elif not gw_info['is_bgw']:
            # Only consider if truly exceptional
            if is_premium_captain and captain_score >= 15:
                reasons.append(f"Premium captain with high score ({captain_score:.1f})")
                
                # Check for dream fixture (e.g., Haaland vs bottom team at home)
                if 'fixture_difficulty' in team_data.columns:
                    captain_fixture = team_data[team_data['player_name'] == captain_name]
                    if not captain_fixture.empty:
                        difficulty = captain_fixture.iloc[0].get('fixture_difficulty', 3)
                        if difficulty <= 1.5:
                            reasons.append(f"Dream fixture (difficulty: {difficulty})")
                            strategic_score += 2
                
                # Still discourage unless truly exceptional
                if captain_score >= self.strategy.TRIPLE_CAPTAIN_MIN_SGW_SCORE:
                    return {
                        'use': 'consider',
                        'confidence': 'low',
                        'player': captain_name,
                        'expected_points': captain_score * 3,
                        'reasons': reasons,
                        'strategic_advice': '⚠️ Consider waiting for DGW unless emergency'
                    }
        
        # DEFAULT: Save for better opportunity
        return {
            'use': False,
            'confidence': 'high',
            'reasons': ['Save Triple Captain for Double Gameweek'],
            'strategic_advice': f'💡 Wait for DGW when {captain_name if is_premium_captain else "Haaland/Salah"} plays twice',
            'best_use': 'Double Gameweek with premium captain (2x points potential)'
        }
    
    def evaluate_bench_boost_strategic(self, team_data: pd.DataFrame) -> Dict:
        """
        Strategic Bench Boost evaluation
        
        Key Rules:
        1. Best in DGW when bench players have 2 games
        2. Need strong bench (not just fodder)
        3. Consider if used after Wildcard setup
        """
        gw_info = self.detect_gameweek_type(team_data)
        
        # Calculate bench strength (simplified)
        if self.scorer:
            scored_data = self.scorer.score_all_players(self.data)
            team_scored = scored_data[scored_data['player_id'].isin(
                team_data['player_id'].tolist()
            )]
        else:
            team_scored = team_data
        
        # Identify likely bench (lowest 4 by score)
        score_col = 'model_score' if 'model_score' in team_scored.columns else 'total_points'
        bench_players = team_scored.nsmallest(4, score_col)
        bench_score = bench_players[score_col].sum() if score_col in bench_players.columns else 0
        
        reasons = []
        
        # RULE 1: Double Gameweek strongly preferred
        if gw_info['is_dgw']:
            reasons.append("🎯 DOUBLE GAMEWEEK - Bench plays twice!")
            bench_score *= 2  # Double the expected score
            
            # Count how many bench players have DGW
            if 'is_dgw' in bench_players.columns:
                dgw_bench_count = bench_players['is_dgw'].sum()
                if dgw_bench_count >= 3:
                    reasons.append(f"✅ {dgw_bench_count}/4 bench players have DGW")
                    
                    if bench_score >= self.strategy.BENCH_BOOST_MIN_BENCH_SCORE:
                        return {
                            'use': True,
                            'confidence': 'high',
                            'expected_bench_points': bench_score,
                            'reasons': reasons,
                            'bench_players': bench_players['player_name'].tolist(),
                            'strategic_advice': '🎯 OPTIMAL - Strong bench in DGW!'
                        }
        
        # RULE 2: Regular gameweek - need exceptional bench
        elif bench_score >= self.strategy.BENCH_BOOST_MIN_BENCH_SCORE * 1.5:
            reasons.append(f"Very strong bench ({bench_score:.1f} expected)")
            return {
                'use': 'consider',
                'confidence': 'medium',
                'expected_bench_points': bench_score,
                'reasons': reasons,
                'strategic_advice': '💭 Good bench, but DGW would be better'
            }
        
        # DEFAULT: Save for better opportunity
        return {
            'use': False,
            'reasons': ['Save for Double Gameweek'],
            'expected_bench_points': bench_score,
            'strategic_advice': '💡 Wait for DGW after using Wildcard to set up strong bench',
            'best_use': 'Big DGW after Wildcard preparation'
        }
    
    def evaluate_wildcard_strategic(self, 
                                   team_issues: List[Dict],
                                   upcoming_dgw_bgw: bool = False) -> Dict:
        """
        Strategic Wildcard evaluation
        
        Key Rules:
        1. Reactive: Multiple injuries/issues
        2. Proactive: Setup for DGW/BGW
        3. Consider if team value is dropping
        """
        issue_count = len([i for i in team_issues if i.get('severity') in ['high', 'medium']])
        
        reasons = []
        
        # RULE 1: Reactive - too many fires
        if issue_count >= self.strategy.WILDCARD_MIN_ISSUES:
            reasons.append(f"🔥 {issue_count} significant team issues")
            return {
                'use': True,
                'confidence': 'high',
                'reasons': reasons,
                'strategic_advice': 'Team needs major rebuild',
                'type': 'reactive'
            }
        
        # RULE 2: Proactive - DGW/BGW preparation
        if upcoming_dgw_bgw:
            reasons.append("📅 Major DGW/BGW coming - need team setup")
            return {
                'use': 'consider',
                'confidence': 'medium',
                'reasons': reasons,
                'strategic_advice': 'Consider using to prepare for fixture swing',
                'type': 'proactive'
            }
        
        # DEFAULT: Hold
        return {
            'use': False,
            'reasons': ['Team in reasonable shape'],
            'strategic_advice': 'Save for emergency or DGW/BGW preparation'
        }
    
    def evaluate_free_hit_strategic(self, team_data: pd.DataFrame) -> Dict:
        """
        Strategic Free Hit evaluation
        
        Key Rules:
        1. PRIMARY: Blank Gameweek rescue
        2. SECONDARY: Small DGW opportunity
        3. EMERGENCY: Multiple injuries in one GW
        """
        gw_info = self.detect_gameweek_type(team_data)
        
        reasons = []
        
        # RULE 1: Blank Gameweek - primary use case
        if gw_info['is_bgw']:
            playing_count = gw_info['playing_count']
            
            if playing_count < self.strategy.FREE_HIT_MIN_PLAYING:
                reasons.append(f"⚠️ Only {playing_count} players have fixtures!")
                return {
                    'use': True,
                    'confidence': 'high',
                    'reasons': reasons,
                    'strategic_advice': '🎯 BLANK GAMEWEEK - Perfect for Free Hit!',
                    'playing_without_fh': playing_count
                }
        
        # RULE 2: Small DGW opportunity
        if gw_info['is_dgw'] and len(gw_info.get('dgw_players', [])) < 5:
            reasons.append("Small DGW - could load up on DGW players")
            return {
                'use': 'consider',
                'confidence': 'low',
                'reasons': reasons,
                'strategic_advice': 'Consider if you want to target DGW players'
            }
        
        # DEFAULT: Save for BGW
        return {
            'use': False,
            'reasons': ['Save for Blank Gameweek'],
            'strategic_advice': '💡 Best used when many teams blank (FA Cup, etc.)',
            'best_use': 'Blank Gameweek with <8 playing players'
        }
    
    def get_comprehensive_chip_strategy(self,
                                       team_data: pd.DataFrame,
                                       team_issues: List[Dict],
                                       captain_analysis: Dict,
                                       available_chips: Dict) -> Dict:
        """
        Get strategic advice for all chips
        
        Returns comprehensive strategy considering:
        - Current gameweek type
        - Future opportunities
        - FPL best practices
        """
        advice = {}
        
        if available_chips.get('triple_captain'):
            advice['triple_captain'] = self.evaluate_triple_captain_strategic(
                team_data, captain_analysis
            )
        
        if available_chips.get('bench_boost'):
            advice['bench_boost'] = self.evaluate_bench_boost_strategic(team_data)
        
        if available_chips.get('wildcard'):
            advice['wildcard'] = self.evaluate_wildcard_strategic(team_issues)
        
        if available_chips.get('free_hit'):
            advice['free_hit'] = self.evaluate_free_hit_strategic(team_data)
        
        # Add general strategic advice
        advice['strategic_summary'] = self._get_strategic_summary(advice)
        
        return advice
    
    def _get_strategic_summary(self, chip_advice: Dict) -> str:
        """Generate strategic summary"""
        
        # Check if any chips are recommended for immediate use
        immediate_use = [
            chip for chip, advice in chip_advice.items()
            if advice.get('use') == True
        ]
        
        if immediate_use:
            return f"✅ Use {', '.join(immediate_use)} this week based on strategic analysis"
        else:
            return "💎 HOLD all chips - wait for optimal opportunities (DGW/BGW)"