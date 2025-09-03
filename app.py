"""
FPL Team Builder - Interactive GUI Application
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import plotly.graph_objects as go
from datetime import datetime
import os

# Import our modules
from src.data.data_merger import DataMerger
from src.models.rule_based_scorer import RuleBasedScorer
from src.my_team import TeamAnalyzer, MyTeam
from src.team_optimizer import TeamOptimizer

# Page config
st.set_page_config(
    page_title="FPL Team Builder",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #38003c;
        color: white;
    }
    .stButton > button:hover {
        background-color: #2d002e;
        color: white;
    }
    .player-card {
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .budget-info {
        font-size: 1.2em;
        font-weight: bold;
        padding: 10px;
        background-color: #00ff87;
        color: #38003c;
        border-radius: 5px;
        text-align: center;
    }
    .team-valid {
        background-color: #00ff87;
        color: #38003c;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .team-invalid {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'team' not in st.session_state:
    st.session_state.team = {
        'GKP': [],
        'DEF': [],
        'MID': [],
        'FWD': []
    }

if 'budget' not in st.session_state:
    st.session_state.budget = 100.0

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'player_data' not in st.session_state:
    st.session_state.player_data = None

def load_data():
    """Load player data from database"""
    try:
        merger = DataMerger("data/fpl_data.db")
        data = merger.load_from_database()
        merger.close()
        
        if not data.empty:
            # Add display columns
            data['display_name'] = data['player_name'] + ' (' + data['team_name'].str[:3] + ')' + ' - £' + data['price'].astype(str) + 'm'
            data['short_name'] = data['player_name'].str[:15]
            return data
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_team_players():
    """Get all players currently in the team"""
    all_players = []
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        all_players.extend(st.session_state.team[pos])
    return all_players

def calculate_team_cost():
    """Calculate total cost of current team"""
    total = 0
    for player_id in get_team_players():
        player = st.session_state.player_data[st.session_state.player_data['player_id'] == player_id].iloc[0]
        total += player['price']
    return total

def get_remaining_budget():
    """Get remaining budget"""
    return st.session_state.budget - calculate_team_cost()

def validate_team():
    """Validate team according to FPL rules"""
    issues = []
    
    # Check formation
    if len(st.session_state.team['GKP']) != 2:
        issues.append(f"Need exactly 2 goalkeepers (have {len(st.session_state.team['GKP'])})")
    if len(st.session_state.team['DEF']) != 5:
        issues.append(f"Need exactly 5 defenders (have {len(st.session_state.team['DEF'])})")
    if len(st.session_state.team['MID']) != 5:
        issues.append(f"Need exactly 5 midfielders (have {len(st.session_state.team['MID'])})")
    if len(st.session_state.team['FWD']) != 3:
        issues.append(f"Need exactly 3 forwards (have {len(st.session_state.team['FWD'])})")
    
    # Check budget
    if calculate_team_cost() > st.session_state.budget:
        issues.append(f"Team exceeds budget")
    
    # Check max 3 players per team
    team_counts = {}
    for player_id in get_team_players():
        player = st.session_state.player_data[st.session_state.player_data['player_id'] == player_id].iloc[0]
        team = player['team_name']
        team_counts[team] = team_counts.get(team, 0) + 1
        
    for team, count in team_counts.items():
        if count > 3:
            issues.append(f"Too many players from {team} (max 3, have {count})")
    
    return len(issues) == 0, issues

def display_team_formation():
    """Display team in formation layout"""
    st.markdown("### 👥 Your Team")
    
    # Create formation display
    positions = ['GKP', 'DEF', 'MID', 'FWD']
    position_names = {
        'GKP': 'Goalkeepers',
        'DEF': 'Defenders', 
        'MID': 'Midfielders',
        'FWD': 'Forwards'
    }
    
    for pos in positions:
        st.markdown(f"**{position_names[pos]}**")
        
        # Get players in this position
        player_ids = st.session_state.team[pos]
        
        # Determine number of columns based on position
        if pos == 'GKP':
            cols = st.columns(2)
            max_players = 2
        elif pos == 'DEF':
            cols = st.columns(5)
            max_players = 5
        elif pos == 'MID':
            cols = st.columns(5)
            max_players = 5
        else:  # FWD
            cols = st.columns(3)
            max_players = 3
        
        # Display players or empty slots
        for i in range(max_players):
            with cols[i]:
                if i < len(player_ids):
                    player = st.session_state.player_data[
                        st.session_state.player_data['player_id'] == player_ids[i]
                    ].iloc[0]
                    
                    # Player card
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #38003c 0%, #00ff87 100%); 
                               padding: 10px; border-radius: 10px; text-align: center; height: 120px;'>
                        <div style='color: white; font-weight: bold;'>{player['short_name']}</div>
                        <div style='color: #f0f0f0; font-size: 0.9em;'>{player['team_name'][:3]}</div>
                        <div style='color: #00ff87; font-weight: bold;'>£{player['price']:.1f}m</div>
                        <div style='color: #f0f0f0; font-size: 0.8em;'>{int(player.get('total_points', 0))} pts</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Remove button
                    if st.button(f"❌ Remove", key=f"remove_{pos}_{i}"):
                        st.session_state.team[pos].remove(player_ids[i])
                        st.rerun()
                else:
                    # Empty slot
                    st.markdown(f"""
                    <div style='background: #e0e0e0; padding: 10px; border-radius: 10px; 
                               text-align: center; height: 120px; border: 2px dashed #999;'>
                        <div style='color: #666; margin-top: 40px;'>Empty Slot</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write("")  # Space for button alignment

def main():
    st.title("⚽ FPL Team Builder")
    st.markdown("Build your Fantasy Premier League team with an interactive interface")
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading player data..."):
            data = load_data()
            if data is not None:
                st.session_state.player_data = data
                st.session_state.data_loaded = True
            else:
                st.error("Failed to load data. Please run 'python fpl.py fetch-data' first.")
                return
    
    # Sidebar for player selection
    with st.sidebar:
        st.header("🎯 Player Selection")
        
        # Budget display
        remaining = get_remaining_budget()
        if remaining >= 0:
            st.markdown(f"""
            <div class='budget-info'>
                💰 Budget Remaining: £{remaining:.1f}m
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: #ff4444; color: white; padding: 10px; 
                       border-radius: 5px; text-align: center; font-weight: bold;'>
                ⚠️ Over Budget: £{abs(remaining):.1f}m
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Position selector
        position = st.selectbox(
            "Select Position",
            options=['GKP', 'DEF', 'MID', 'FWD'],
            format_func=lambda x: {
                'GKP': '🥅 Goalkeeper',
                'DEF': '🛡️ Defender',
                'MID': '⚡ Midfielder',
                'FWD': '⚔️ Forward'
            }[x]
        )
        
        # Filter players by position
        position_players = st.session_state.player_data[
            st.session_state.player_data['position'] == position
        ].copy()
        
        # Remove already selected players
        selected_ids = get_team_players()
        position_players = position_players[~position_players['player_id'].isin(selected_ids)]
        
        # Don't filter by budget here - let the user see all players and show warnings instead
        
        # Additional filters
        st.markdown("### 🔍 Filters")
        
        # Team filter
        teams = ['All Teams'] + sorted(position_players['team_name'].unique().tolist())
        selected_team = st.selectbox("Team", teams)
        if selected_team != 'All Teams':
            position_players = position_players[position_players['team_name'] == selected_team]
        
        # Price filter
        if not position_players.empty:
            min_price = float(position_players['price'].min())
            max_price = float(position_players['price'].max())
            
            # Ensure max_price is at least min_price to avoid slider errors
            if max_price < min_price:
                max_price = min_price
            
            # Set default values for slider based on budget
            if remaining > 0:
                default_max = min(max_price, remaining)
                # Ensure default_max is not less than min_price
                if default_max < min_price:
                    default_max = min_price
            else:
                default_max = max_price
            
            price_range = st.slider(
                "Price Range (£m)",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, default_max),
                step=0.5
            )
            position_players = position_players[
                (position_players['price'] >= price_range[0]) & 
                (position_players['price'] <= price_range[1])
            ]
        
        # Sort options
        sort_by = st.selectbox(
            "Sort By",
            options=['total_points', 'price', 'form', 'selected_by_percent'],
            format_func=lambda x: {
                'total_points': 'Total Points',
                'price': 'Price',
                'form': 'Form',
                'selected_by_percent': 'Ownership %'
            }[x]
        )
        position_players = position_players.sort_values(sort_by, ascending=False)
        
        # Search box
        search = st.text_input("🔎 Search Player Name")
        if search:
            position_players = position_players[
                position_players['player_name'].str.contains(search, case=False, na=False)
            ]
        
        st.markdown("---")
        
        # Player dropdown
        if not position_players.empty:
            st.markdown("### 👤 Select Player")
            
            # Check position limits
            position_limits = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            current_count = len(st.session_state.team[position])
            
            if current_count >= position_limits[position]:
                st.warning(f"Already have {position_limits[position]} {position}s")
            else:
                # Create display options with stats
                position_players['select_display'] = (
                    position_players['player_name'] + ' (' + 
                    position_players['team_name'].str[:3] + ') - £' + 
                    position_players['price'].astype(str) + 'm - ' +
                    position_players['total_points'].astype(int).astype(str) + ' pts'
                )
                
                selected_player = st.selectbox(
                    "Choose a player",
                    options=position_players['player_id'].tolist(),
                    format_func=lambda x: position_players[
                        position_players['player_id'] == x
                    ]['select_display'].iloc[0],
                    key='player_select'
                )
                
                if selected_player:
                    player_info = position_players[position_players['player_id'] == selected_player].iloc[0]
                    
                    # Display player stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Points", int(player_info['total_points']))
                        st.metric("Form", f"{player_info.get('form', 0):.1f}")
                    with col2:
                        st.metric("Price", f"£{player_info['price']:.1f}m")
                        st.metric("Selected", f"{player_info.get('selected_by_percent', 0):.1f}%")
                    
                    # Add button
                    if st.button("➕ Add to Team", type="primary", use_container_width=True):
                        # Check budget
                        if player_info['price'] <= remaining:
                            st.session_state.team[position].append(selected_player)
                            st.success(f"Added {player_info['player_name']} to team!")
                            st.rerun()
                        else:
                            st.error("Not enough budget!")
        else:
            st.info("No players available with current filters")
    
    # Main area - Team display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_team_formation()
    
    with col2:
        st.markdown("### 📊 Team Stats")
        
        # Team size
        team_size = len(get_team_players())
        st.metric("Players", f"{team_size}/15")
        
        # Budget
        spent = calculate_team_cost()
        st.metric("Money Spent", f"£{spent:.1f}m")
        st.metric("Money Remaining", f"£{get_remaining_budget():.1f}m")
        
        # Validation
        is_valid, issues = validate_team()
        
        st.markdown("### ✅ Validation")
        if is_valid and team_size == 15:
            st.markdown("<div class='team-valid'>✅ Team is Valid!</div>", unsafe_allow_html=True)
        else:
            if issues:
                for issue in issues:
                    st.error(issue)
        
        st.markdown("---")
        
        # Action buttons
        st.markdown("### 🎮 Actions")
        
        # Team Recommendation Section
        st.markdown("### 🤖 AI Team Recommendation")
        
        # Strategy selector
        optimizer = TeamOptimizer()
        strategies = optimizer.get_strategies()
        strategy_descriptions = {
            'balanced': 'Balanced - Best for next 3 gameweeks',
            'short_term': 'Short-term - Focus on next gameweek',
            'long_term': 'Long-term - Plan for next 5 gameweeks',
            'differential': 'Differential - Low ownership gems',
            'template': 'Template - Follow the crowd'
        }
        
        selected_strategy = st.selectbox(
            "Select Strategy",
            strategies,
            format_func=lambda x: strategy_descriptions[x]
        )
        
        if st.button("🎯 Get Fresh Team Recommendation", use_container_width=True):
            with st.spinner("Optimizing team selection..."):
                try:
                    # Get recommendation
                    result = optimizer.recommend_team(strategy=selected_strategy)
                    
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        # Clear current team
                        st.session_state.team = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
                        
                        # Add recommended players
                        for position in ['GKP', 'DEF', 'MID', 'FWD']:
                            for player in result['team'][position]:
                                st.session_state.team[position].append(player['player_id'])
                        
                        # Show success message with details
                        st.success(f"✅ Team recommended using '{selected_strategy}' strategy!")
                        st.info(f"💰 Total Cost: £{result['total_cost']:.1f}m")
                        st.info(f"📊 Expected Points: {result['expected_points']:.1f}")
                        if result['captain']:
                            st.info(f"©️ Captain: {result['captain']['player_name']}")
                        
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error generating recommendation: {str(e)}")
        
        st.markdown("---")
        
        if st.button("🔄 Reset Team", use_container_width=True):
            st.session_state.team = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
            st.rerun()
        
        # Save/Load section
        st.markdown("### 💾 Save/Load")
        
        # Save team
        if team_size > 0:
            team_name = st.text_input("Team Name", value="my_team")
            if st.button("💾 Save Team", use_container_width=True):
                # Create teams directory
                teams_dir = Path("teams")
                teams_dir.mkdir(exist_ok=True)
                
                # Save team data
                team_data = {
                    'player_ids': get_team_players(),
                    'formation': st.session_state.team,
                    'budget': st.session_state.budget,
                    'cost': calculate_team_cost(),
                    'saved_at': datetime.now().isoformat()
                }
                
                filename = teams_dir / f"{team_name}.json"
                with open(filename, 'w') as f:
                    json.dump(team_data, f, indent=2)
                
                st.success(f"Team saved to {filename}")
        
        # Load team
        st.markdown("**Load Existing Team**")
        teams_dir = Path("teams")
        if teams_dir.exists():
            team_files = list(teams_dir.glob("*.json"))
            if team_files:
                team_names = [f.stem for f in team_files]
                selected_team = st.selectbox("Select Team", team_names)
                
                if st.button("📂 Load Team", use_container_width=True):
                    with open(teams_dir / f"{selected_team}.json", 'r') as f:
                        team_data = json.load(f)
                    
                    if 'formation' in team_data:
                        st.session_state.team = team_data['formation']
                    else:
                        # Convert from player_ids list
                        st.session_state.team = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
                        for pid in team_data['player_ids']:
                            player = st.session_state.player_data[
                                st.session_state.player_data['player_id'] == pid
                            ]
                            if not player.empty:
                                pos = player.iloc[0]['position']
                                st.session_state.team[pos].append(pid)
                    
                    st.success(f"Loaded team: {selected_team}")
                    st.rerun()
        
        # Export for CLI
        if is_valid and team_size == 15:
            st.markdown("### 🔗 Use with CLI")
            player_ids_str = ','.join(str(pid) for pid in get_team_players())
            st.code(f"python fpl.py my-team -p '{player_ids_str}'", language="bash")

if __name__ == "__main__":
    main()