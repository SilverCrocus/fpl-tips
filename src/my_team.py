"""
My Team Management Module
Handles personalized team analysis and transfer recommendations
"""
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class MyTeam:
    """Represents user's current FPL team"""
    players: List[int]  # Player IDs
    captain: int
    vice_captain: int
    bank: float
    free_transfers: int = 1
    wildcard_available: bool = True
    free_hit_available: bool = True
    bench_boost_available: bool = True
    triple_captain_available: bool = True
    

@dataclass
class TransferRecommendation:
    """Transfer recommendation"""
    player_out: Dict  # Player to remove
    player_in: Dict   # Player to bring in
    net_cost: float
    score_improvement: float
    reason: str
    priority: int  # 1=urgent, 2=recommended, 3=optional


class TeamAnalyzer:
    """Analyze user's team and provide recommendations"""
    
    def __init__(self, scorer, data: pd.DataFrame):
        self.scorer = scorer
        self.data = data
        self.fixture_conflicts = []  # Track fixture conflicts
        
    def analyze_team(self, my_team: MyTeam) -> Dict:
        """Comprehensive team analysis
        
        Args:
            my_team: User's current team
            
        Returns:
            Analysis dictionary with insights
        """
        analysis = {}
        
        # Get team players data
        team_data = self.data[self.data['player_id'].isin(my_team.players)]
        
        # Overall metrics
        analysis['total_team_value'] = team_data['price'].sum()
        analysis['total_expected_points'] = team_data.get('expected_points', team_data['total_points']).sum()
        analysis['average_ownership'] = team_data['selected_by_percent'].mean()
        
        # Position breakdown
        position_counts = team_data['position'].value_counts()
        analysis['position_breakdown'] = position_counts.to_dict()
        
        # Team distribution
        team_counts = team_data['team_name'].value_counts()
        analysis['team_distribution'] = team_counts.to_dict()
        
        # Identify weaknesses
        analysis['weaknesses'] = self._identify_weaknesses(team_data)
        
        # Fixture analysis
        analysis['fixture_analysis'] = self._analyze_fixtures(team_data)
        
        # Captain analysis
        analysis['captain_analysis'] = self._analyze_captain(team_data, my_team.captain)
        
        return analysis
        
    def get_post_transfer_lineup(self, team_data: pd.DataFrame, transfers: List[TransferRecommendation]) -> Dict:
        """Get lineup suggestion after applying recommended transfers
        
        Args:
            team_data: Current team DataFrame
            transfers: List of transfer recommendations to apply
            
        Returns:
            Dictionary with post-transfer lineup
        """
        # Create a copy of team data
        new_team_data = team_data.copy()
        
        # Apply transfers
        for transfer in transfers:
            # Remove player out
            new_team_data = new_team_data[new_team_data['player_id'] != transfer.player_out['id']]
            
            # Add player in
            # Get full data for the new player
            new_player_data = self.data[self.data['player_id'] == transfer.player_in['id']]
            if not new_player_data.empty:
                new_team_data = pd.concat([new_team_data, new_player_data], ignore_index=True)
        
        # Generate lineup for the new team
        return self.get_lineup_suggestion(new_team_data)
    
    def get_lineup_suggestion(self, team_data: pd.DataFrame) -> Dict:
        """Suggest optimal starting 11 and bench order
        
        Args:
            team_data: DataFrame with team players
            
        Returns:
            Dictionary with lineup suggestion
        """
        # Score all players for lineup selection
        team_data = team_data.copy()
        
        # Add lineup score based on form, availability, and fixtures
        team_data['lineup_score'] = 0
        
        # Base score from form and expected points
        if 'form' in team_data.columns:
            team_data['lineup_score'] += team_data['form'] * 2
        if 'expected_points' in team_data.columns:
            team_data['lineup_score'] += team_data['expected_points']
        elif 'total_points' in team_data.columns:
            team_data['lineup_score'] += team_data['total_points'] / 10
            
        # Penalize for difficult fixtures
        if 'fixture_difficulty' in team_data.columns:
            team_data['lineup_score'] += (5 - team_data['fixture_difficulty'])
            
        # Heavy penalty for injured/doubtful players
        if 'chance_of_playing_next_round' in team_data.columns:
            team_data.loc[team_data['chance_of_playing_next_round'] < 75, 'lineup_score'] *= 0.3
        if 'is_available' in team_data.columns:
            team_data.loc[team_data['is_available'] == 0, 'lineup_score'] *= 0.1
            
        # Separate by position
        gkp = team_data[team_data['position'].isin(['GK', 'GKP'])].sort_values('lineup_score', ascending=False)
        defs = team_data[team_data['position'] == 'DEF'].sort_values('lineup_score', ascending=False)
        mids = team_data[team_data['position'] == 'MID'].sort_values('lineup_score', ascending=False)
        fwds = team_data[team_data['position'] == 'FWD'].sort_values('lineup_score', ascending=False)
        
        # Determine best formation based on available players and scores
        formations = []
        
        # Try common formations
        for def_count, mid_count, fwd_count in [(3,4,3), (3,5,2), (4,3,3), (4,4,2), (4,5,1), (5,3,2), (5,4,1)]:
            if len(defs) >= def_count and len(mids) >= mid_count and len(fwds) >= fwd_count:
                # Calculate formation score
                formation_score = (
                    gkp.head(1)['lineup_score'].sum() +
                    defs.head(def_count)['lineup_score'].sum() + 
                    mids.head(mid_count)['lineup_score'].sum() + 
                    fwds.head(fwd_count)['lineup_score'].sum()
                )
                formations.append({
                    'formation': f"{def_count}-{mid_count}-{fwd_count}",
                    'score': formation_score,
                    'def': def_count,
                    'mid': mid_count,
                    'fwd': fwd_count
                })
        
        if not formations:
            return {'status': 'insufficient_players'}
            
        # Pick best formation
        best_formation = max(formations, key=lambda x: x['score'])
        
        # Select starting 11
        starting_11 = []
        bench = []
        
        # Always play 1 goalkeeper
        if len(gkp) > 0:
            starting_11.append(gkp.iloc[0].to_dict())
            for _, player in gkp.iloc[1:].iterrows():
                bench.append(player.to_dict())
            
        # Add defenders
        for _, player in defs.head(best_formation['def']).iterrows():
            starting_11.append(player.to_dict())
        for _, player in defs.iloc[best_formation['def']:].iterrows():
            bench.append(player.to_dict())
        
        # Add midfielders
        for _, player in mids.head(best_formation['mid']).iterrows():
            starting_11.append(player.to_dict())
        for _, player in mids.iloc[best_formation['mid']:].iterrows():
            bench.append(player.to_dict())
        
        # Add forwards
        for _, player in fwds.head(best_formation['fwd']).iterrows():
            starting_11.append(player.to_dict())
        for _, player in fwds.iloc[best_formation['fwd']:].iterrows():
            bench.append(player.to_dict())
        
        # Sort bench by score (best first for auto-sub priority)
        if len(bench) > 0:
            bench = sorted(bench, key=lambda x: x.get('lineup_score', 0), reverse=True)
        
        return {
            'formation': best_formation['formation'],
            'formation_score': best_formation['score'],
            'starting_11': [
                {
                    'name': p.get('player_name', 'Unknown'),
                    'position': p.get('position', 'Unknown'),
                    'score': round(p.get('lineup_score', 0), 1)
                } for p in starting_11
            ],
            'bench': [
                {
                    'name': p.get('player_name', 'Unknown'),
                    'position': p.get('position', 'Unknown'),
                    'score': round(p.get('lineup_score', 0), 1)
                } for p in bench
            ]
        }
        
    def _identify_weaknesses(self, team_data: pd.DataFrame) -> List[Dict]:
        """Identify team weaknesses"""
        weaknesses = []
        
        # Check for injured/unavailable players
        if 'is_available' in team_data.columns:
            unavailable = team_data[team_data['is_available'] == 0]
            for _, player in unavailable.iterrows():
                weaknesses.append({
                    'type': 'unavailable',
                    'player': player['player_name'],
                    'severity': 'high'
                })
        
        # Check for players with poor form
        poor_form = team_data[team_data['form'] < 2.0]
        for _, player in poor_form.iterrows():
            weaknesses.append({
                'type': 'poor_form',
                'player': player['player_name'],
                'form': player['form'],
                'severity': 'medium'
            })
        
        # Check for players with bad fixtures
        if 'fixture_diff_next5' in team_data.columns:
            hard_fixtures = team_data[team_data['fixture_diff_next5'] > 3.5]
            for _, player in hard_fixtures.iterrows():
                weaknesses.append({
                    'type': 'hard_fixtures',
                    'player': player['player_name'],
                    'difficulty': player['fixture_diff_next5'],
                    'severity': 'low'
                })
        
        return weaknesses
        
    def _analyze_fixtures(self, team_data: pd.DataFrame) -> Dict:
        """Analyze team's upcoming fixtures"""
        if 'fixture_diff_next5' not in team_data.columns:
            return {'status': 'no_fixture_data'}
            
        return {
            'average_difficulty_next5': team_data['fixture_diff_next5'].mean(),
            'easiest_fixtures': team_data.nsmallest(3, 'fixture_diff_next5')[
                ['player_name', 'team_name', 'fixture_diff_next5']
            ].to_dict('records'),
            'hardest_fixtures': team_data.nlargest(3, 'fixture_diff_next5')[
                ['player_name', 'team_name', 'fixture_diff_next5']
            ].to_dict('records')
        }
        
    def _analyze_captain(self, team_data: pd.DataFrame, captain_id: int) -> Dict:
        """Analyze captain choice"""
        captain_data = team_data[team_data['player_id'] == captain_id]
        
        if captain_data.empty:
            return {'status': 'captain_not_found'}
            
        captain = captain_data.iloc[0]
        
        # Score all team players for captaincy (excluding goalkeepers)
        captain_scores = []
        for _, player in team_data.iterrows():
            # Skip goalkeepers - they should never be captained
            if player.get('position') == 'GK' or player.get('position') == 'GKP':
                continue
                
            score = 0
            # Use .get() with default 0 to avoid NaN issues
            score += player.get('prob_goal', 0) * 5
            score += player.get('form', 0) * 0.5
            
            # Fixture difficulty (lower is better, so invert)
            if 'fixture_difficulty' in player and pd.notna(player['fixture_difficulty']):
                score += (5 - player['fixture_difficulty']) * 0.3
            
            # Skip players with availability issues
            if player.get('chance_of_playing_next_round', 100) < 75:
                score *= 0.5  # Heavily penalize injured/doubtful players
                
            captain_scores.append({
                'player': player['player_name'],
                'score': score
            })
        
        # Sort by score, handling NaN properly
        captain_scores = sorted(captain_scores, 
                               key=lambda x: x['score'] if pd.notna(x['score']) else -999, 
                               reverse=True)
        
        return {
            'current_captain': captain['player_name'],
            'captain_rank': next((i+1 for i, p in enumerate(captain_scores) 
                                 if p['player'] == captain['player_name']), None),
            'recommended_captain': captain_scores[0] if captain_scores else None,
            'top_3_options': captain_scores[:3]
        }
    
    def detect_fixture_conflicts(self, team_data: pd.DataFrame, transfers: List[TransferRecommendation] = None) -> List[Dict]:
        """Detect when players in team or transfers face each other
        
        Args:
            team_data: Current team DataFrame  
            transfers: List of transfer recommendations to check
            
        Returns:
            List of fixture conflicts
        """
        conflicts = []
        
        # Get teams in current squad
        team_players = {}
        if 'team_name' in team_data.columns:
            for _, player in team_data.iterrows():
                team = player.get('team_name', 'Unknown')
                if team not in team_players:
                    team_players[team] = []
                team_players[team].append(player.get('player_name', 'Unknown'))
        
        # Add transfer players
        if transfers:
            for transfer in transfers:
                # Player coming in
                in_team = transfer.player_in.get('team_name', 'Unknown')
                if in_team not in team_players:
                    team_players[in_team] = []
                team_players[in_team].append(transfer.player_in.get('web_name', 'Unknown'))
        
        # Check for opposing fixtures (simplified check based on fixture difficulty)
        # In a full implementation, we'd check actual fixtures
        all_teams = list(team_players.keys())
        
        for i, team1 in enumerate(all_teams):
            for team2 in all_teams[i+1:]:
                # Get average fixture difficulties
                team1_players_df = team_data[team_data['team_name'] == team1] if 'team_name' in team_data.columns else pd.DataFrame()
                team2_players_df = team_data[team_data['team_name'] == team2] if 'team_name' in team_data.columns else pd.DataFrame()
                
                if not team1_players_df.empty and not team2_players_df.empty:
                    team1_diff = team1_players_df['fixture_difficulty'].mean() if 'fixture_difficulty' in team1_players_df.columns else 3
                    team2_diff = team2_players_df['fixture_difficulty'].mean() if 'fixture_difficulty' in team2_players_df.columns else 3
                    
                    # If one team has easy fixtures (<=2) and other has hard (>=4), they might be playing each other
                    if (team1_diff <= 2 and team2_diff >= 4) or (team1_diff >= 4 and team2_diff <= 2):
                        conflicts.append({
                            'team1': team1,
                            'team1_players': team_players[team1],
                            'team2': team2, 
                            'team2_players': team_players[team2],
                            'type': 'opposing_fixture',
                            'severity': 'warning',
                            'message': f"⚠️ Players from {team1} and {team2} may face each other - one's success hurts the other's points!"
                        })
        
        return conflicts
        
    def get_transfer_recommendations(self, my_team: MyTeam, 
                                    num_transfers: int = 2) -> List[TransferRecommendation]:
        """Get transfer recommendations
        
        Args:
            my_team: User's current team
            num_transfers: Number of transfers to recommend
            
        Returns:
            List of transfer recommendations
        """
        recommendations = []
        
        # Score all players
        all_scores = self.scorer.score_all_players(self.data)
        
        # Get team players
        team_data = all_scores[all_scores['player_id'].isin(my_team.players)]
        non_team_data = all_scores[~all_scores['player_id'].isin(my_team.players)]
        
        # Prioritize injured/unavailable players first
        unavailable_players = team_data[
            (team_data['is_available'] == 0) if 'is_available' in team_data.columns else False
        ]
        if 'chance_of_playing_next_round' in team_data.columns:
            doubtful_players = team_data[team_data['chance_of_playing_next_round'] < 75]
            unavailable_players = pd.concat([unavailable_players, doubtful_players]).drop_duplicates()
        
        # Then find worst performers
        worst_performers = team_data.nsmallest(5, 'model_score')
        
        # Combine unavailable and worst players (remove duplicates)
        players_to_consider = pd.concat([unavailable_players, worst_performers]).drop_duplicates()
        worst_players = players_to_consider.sort_values('model_score')
        
        for _, player_out in worst_players.iterrows():
            # Find best replacement in same position within budget
            budget = player_out['price'] + my_team.bank
            
            replacements = non_team_data[
                (non_team_data['position'] == player_out['position']) &
                (non_team_data['price'] <= budget)
            ]
            
            if replacements.empty:
                continue
                
            # Get best replacement
            best_replacement = replacements.nlargest(1, 'model_score').iloc[0]
            
            # Calculate improvement
            score_improvement = best_replacement['model_score'] - player_out['model_score']
            net_cost = best_replacement['price'] - player_out['price']
            
            # Determine reason and priority
            reason = self._get_transfer_reason(player_out, best_replacement)
            priority = self._get_transfer_priority(player_out, best_replacement)
            
            recommendations.append(TransferRecommendation(
                player_out={
                    'id': player_out['player_id'],
                    'name': player_out['player_name'],
                    'price': player_out['price'],
                    'score': player_out['model_score']
                },
                player_in={
                    'id': best_replacement['player_id'],
                    'name': best_replacement['player_name'],
                    'price': best_replacement['price'],
                    'score': best_replacement['model_score']
                },
                net_cost=net_cost,
                score_improvement=score_improvement,
                reason=reason,
                priority=priority
            ))
            
        # Sort by priority (1=urgent first) and score improvement
        # Priority 1 (urgent) should come first, so use ascending sort
        recommendations.sort(key=lambda x: (x.priority, -x.score_improvement))
        
        return recommendations[:num_transfers]
        
    def _get_transfer_reason(self, player_out: pd.Series, player_in: pd.Series) -> str:
        """Generate transfer reason"""
        reasons = []
        
        # Check injury/availability
        if player_out.get('is_available', 1) == 0:
            reasons.append("Injured/unavailable")
        
        # Check chance of playing
        chance_of_playing = player_out.get('chance_of_playing_next_round', 100)
        if chance_of_playing < 75:
            if chance_of_playing == 0:
                reasons.append("OUT - Not playing")
            elif chance_of_playing <= 25:
                reasons.append(f"DOUBTFUL - {chance_of_playing}% chance")
            elif chance_of_playing <= 50:
                reasons.append(f"50/50 - {chance_of_playing}% chance")
            else:
                reasons.append(f"75% doubt - {chance_of_playing}% chance")
            
        # Check form difference
        form_diff = player_in.get('form', 0) - player_out.get('form', 0)
        if form_diff > 3:
            reasons.append(f"Much better form (+{form_diff:.1f})")
        elif form_diff > 1:
            reasons.append(f"Better form (+{form_diff:.1f})")
            
        # Check fixture difficulty
        if 'fixture_diff_next5' in player_out and 'fixture_diff_next5' in player_in:
            fix_diff = player_out['fixture_diff_next5'] - player_in['fixture_diff_next5']
            if fix_diff > 1:
                reasons.append(f"Better fixtures (diff: {fix_diff:.1f})")
                
        # Check goal probability
        if 'prob_goal' in player_out and 'prob_goal' in player_in:
            goal_diff = player_in['prob_goal'] - player_out['prob_goal']
            if goal_diff > 0.2:
                reasons.append(f"Higher goal threat (+{goal_diff*100:.0f}%)")
                
        return " | ".join(reasons) if reasons else "Better overall value"
        
    def _get_transfer_priority(self, player_out: pd.Series, player_in: pd.Series) -> int:
        """Determine transfer priority (1=urgent, 2=recommended, 3=optional)"""
        
        # Urgent: Player unavailable, injured, or very poor form
        if player_out.get('is_available', 1) == 0:
            return 1
        
        # Check chance of playing (injured/doubtful)
        chance_of_playing = player_out.get('chance_of_playing_next_round', 100)
        if chance_of_playing < 75:  # Less than 75% chance is concerning
            return 1
        
        # Very poor form is also urgent
        if player_out.get('form', 5) < 1:
            return 1
            
        # Recommended: Significant score improvement or poor form
        score_diff = player_in.get('model_score', 0) - player_out.get('model_score', 0)
        if score_diff > 3:
            return 2
        
        # Poor but not terrible form
        if player_out.get('form', 5) < 3:
            return 2
            
        # Optional: Minor improvements
        return 3
        

class ChipAdvisor:
    """Advise on chip usage (Wildcard, Free Hit, etc.)"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def get_all_chip_advice(self, my_team: MyTeam, analysis: Dict, team_data: pd.DataFrame) -> Dict:
        """Get advice for all available chips
        
        Returns:
            Dictionary with advice for each chip
        """
        advice = {}
        
        if my_team.wildcard_available:
            advice['wildcard'] = self.should_use_wildcard(my_team, analysis)
            
        if my_team.free_hit_available:
            advice['free_hit'] = self.should_use_free_hit(my_team, analysis, team_data)
            
        if my_team.bench_boost_available:
            advice['bench_boost'] = self.should_use_bench_boost(team_data)
            
        if my_team.triple_captain_available:
            advice['triple_captain'] = self.should_use_triple_captain(analysis)
            
        return advice
    
    def should_use_wildcard(self, my_team: MyTeam, analysis: Dict) -> Dict:
        """Determine if wildcard should be used
        
        Args:
            my_team: User's current team
            analysis: Team analysis results
            
        Returns:
            Advice dictionary
        """
        if not my_team.wildcard_available:
            return {'use': False, 'reason': 'Wildcard not available'}
            
        reasons_to_use = []
        score = 0
        
        # Check team value vs template
        if analysis.get('total_team_value', 100) < 95:
            reasons_to_use.append("Team value significantly below template")
            score += 2
            
        # Check number of weaknesses
        weaknesses = analysis.get('weaknesses', [])
        high_severity = sum(1 for w in weaknesses if w.get('severity') == 'high')
        if high_severity >= 3:
            reasons_to_use.append(f"{high_severity} urgent issues in team")
            score += 3
            
        # Check fixture swing
        avg_difficulty = analysis.get('fixture_analysis', {}).get('average_difficulty_next5', 3)
        if avg_difficulty > 3.8:
            reasons_to_use.append("Very difficult fixtures ahead")
            score += 2
            
        # Decision
        if score >= 5:
            return {
                'use': True,
                'confidence': 'high',
                'reasons': reasons_to_use
            }
        elif score >= 3:
            return {
                'use': 'consider',
                'confidence': 'medium',
                'reasons': reasons_to_use
            }
        else:
            return {
                'use': False,
                'confidence': 'low',
                'reasons': ['Team is in reasonable shape']
            }
    
    def should_use_free_hit(self, my_team: MyTeam, analysis: Dict, team_data: pd.DataFrame) -> Dict:
        """Determine if free hit should be used
        
        Free Hit is best for:
        - Blank gameweeks (many teams not playing)
        - Very difficult fixtures for most of your team
        - When many players are injured/unavailable temporarily
        """
        if not my_team.free_hit_available:
            return {'use': False, 'reason': 'Free Hit not available'}
            
        reasons = []
        score = 0
        
        # Check for multiple injuries/unavailable
        weaknesses = analysis.get('weaknesses', [])
        unavailable_count = sum(1 for w in weaknesses if w.get('type') == 'unavailable')
        if unavailable_count >= 4:
            reasons.append(f"{unavailable_count} players unavailable")
            score += 3
            
        # Check fixture difficulty
        avg_difficulty = analysis.get('fixture_analysis', {}).get('average_difficulty_next5', 3)
        if avg_difficulty > 4.0:
            reasons.append("Extremely difficult fixtures")
            score += 2
            
        # Check if too many players from same team (blank gameweek indicator)
        team_distribution = analysis.get('team_distribution', {})
        max_from_team = max(team_distribution.values()) if team_distribution else 0
        if max_from_team >= 5:
            reasons.append(f"Too many players from one team ({max_from_team})")
            score += 2
            
        if score >= 4:
            return {'use': True, 'confidence': 'medium', 'reasons': reasons}
        else:
            return {'use': False, 'confidence': 'low', 'reasons': ['Save for blank/difficult gameweek']}
    
    def should_use_bench_boost(self, team_data: pd.DataFrame) -> Dict:
        """Determine if bench boost should be used
        
        Bench Boost is best when:
        - Bench players have easy fixtures
        - Double gameweek (players play twice)
        - All bench players are fit and starting
        """
        # Separate bench players
        if len(team_data) < 15:
            return {'use': False, 'reason': 'Incomplete team'}
            
        # Get bench players (marked as benched)
        bench_players = team_data[team_data.get('is_benched', 0) == 1] if 'is_benched' in team_data.columns else team_data.tail(4)
        
        # If no bench players found, fall back to last 4 players
        if bench_players.empty or len(bench_players) < 4:
            bench_players = team_data.tail(4)
        
        # Check bench fixture difficulty
        avg_bench_difficulty = bench_players['fixture_difficulty'].mean() if 'fixture_difficulty' in bench_players.columns else 3
        
        # Check bench availability
        bench_available = (bench_players.get('is_available', 1) == 1).all() if 'is_available' in bench_players.columns else True
        
        # Calculate bench expected points
        if 'expected_points' in bench_players.columns:
            bench_expected = bench_players['expected_points'].sum()
        elif 'model_score' in bench_players.columns:
            bench_expected = bench_players['model_score'].sum()
        else:
            bench_expected = 12  # Default assumption
        
        reasons = []
        
        if avg_bench_difficulty <= 2.5:
            reasons.append(f"Easy bench fixtures (avg: {avg_bench_difficulty:.1f})")
            
        if not bench_available:
            return {'use': False, 'confidence': 'low', 'reasons': ['Bench players unavailable']}
            
        if bench_expected >= 20:  # Good bench boost potential
            reasons.append(f"High bench potential ({bench_expected:.0f} pts expected)")
            return {'use': True, 'confidence': 'high', 'reasons': reasons}
        elif bench_expected >= 15:
            return {'use': 'consider', 'confidence': 'medium', 'reasons': reasons}
        else:
            return {'use': False, 'confidence': 'low', 'reasons': ['Low bench scoring potential']}
    
    def should_use_triple_captain(self, analysis: Dict) -> Dict:
        """Determine if triple captain should be used
        
        Triple Captain is best for:
        - Premium captain with very high expected points
        - Easy fixture for captain
        - Double gameweek for captain
        """
        captain_analysis = analysis.get('captain_analysis', {})
        
        if not captain_analysis or not captain_analysis.get('recommended_captain'):
            return {'use': False, 'reason': 'No clear captain choice'}
            
        top_captain = captain_analysis.get('recommended_captain', {})
        captain_score = top_captain.get('score', 0)
        
        reasons = []
        
        # Very high captain score threshold
        if captain_score >= 10:
            reasons.append(f"Exceptional captain choice (score: {captain_score:.1f})")
            return {'use': True, 'confidence': 'high', 'reasons': reasons}
        elif captain_score >= 8:
            reasons.append(f"Strong captain choice (score: {captain_score:.1f})")
            return {'use': 'consider', 'confidence': 'medium', 'reasons': reasons}
        else:
            return {'use': False, 'confidence': 'low', 'reasons': ['Save for premium captain opportunity']}