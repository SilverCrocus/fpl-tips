"""
FPL Team Builder - Advanced GUI Application with Enhanced Analytics
Run with: streamlit run app_advanced.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional

# Import our modules
from src.data.data_merger import DataMerger
from src.data.fpl_api import FPLApiClient
from src.models.rule_based_scorer import RuleBasedScorer
from src.models.backtester import Backtester
from src.my_team import TeamAnalyzer, MyTeam
from src.team_optimizer import TeamOptimizer
from src.transfer_strategy import TransferStrategy
from src.chip_strategy import ChipStrategy, ChipType

# Page config with wide layout for advanced features
st.set_page_config(
    page_title="FPL Team Builder - Advanced",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for advanced interface
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #38003c;
        color: white;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #2d002e;
        color: white;
        transform: scale(1.02);
    }
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .player-card {
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
        color: #38003c;
        font-weight: bold;
    }
    .advanced-header {
        background: linear-gradient(90deg, #38003c 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .tab-content {
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stats-table {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with advanced features
if 'team' not in st.session_state:
    st.session_state.team = {
        'GK': [],
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

if 'team_history' not in st.session_state:
    st.session_state.team_history = []

if 'saved_teams' not in st.session_state:
    st.session_state.saved_teams = {}

if 'comparison_players' not in st.session_state:
    st.session_state.comparison_players = []

def load_data():
    """Load player data from database with caching"""
    try:
        merger = DataMerger("data/fpl_data.db")
        data = merger.load_from_database()
        merger.close()
        
        if not data.empty:
            st.session_state.player_data = data
            st.session_state.data_loaded = True
            return data
        else:
            st.error("No data found in database. Please run 'python -m src.main fetch-data' first.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def get_team_players():
    """Get all players in current team"""
    players = []
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        players.extend(st.session_state.team[position])
    return players

def calculate_team_cost():
    """Calculate total team cost"""
    total = 0
    for player_id in get_team_players():
        player = st.session_state.player_data[st.session_state.player_data['id'] == player_id].iloc[0]
        total += player['now_cost'] / 10  # Convert to millions
    return total

def get_remaining_budget():
    """Get remaining budget"""
    return st.session_state.budget - calculate_team_cost()

def validate_team():
    """Validate team constraints"""
    team = st.session_state.team
    issues = []
    
    # Check squad size
    total_players = sum(len(players) for players in team.values())
    if total_players != 15:
        issues.append(f"Squad must have exactly 15 players (currently {total_players})")
    
    # Check position limits
    if len(team['GK']) != 2:
        issues.append(f"Must have exactly 2 GK (currently {len(team['GK'])})")
    if len(team['DEF']) != 5:
        issues.append(f"Must have exactly 5 DEF (currently {len(team['DEF'])})")
    if len(team['MID']) != 5:
        issues.append(f"Must have exactly 5 MID (currently {len(team['MID'])})")
    if len(team['FWD']) != 3:
        issues.append(f"Must have exactly 3 FWD (currently {len(team['FWD'])})")
    
    # Check budget
    if calculate_team_cost() > st.session_state.budget:
        issues.append(f"Team exceeds budget (£{calculate_team_cost():.1f}m > £{st.session_state.budget}m)")
    
    # Check team limits (max 3 per team)
    team_counts = {}
    for player_id in get_team_players():
        player = st.session_state.player_data[st.session_state.player_data['id'] == player_id].iloc[0]
        team_name = player['team_name']
        team_counts[team_name] = team_counts.get(team_name, 0) + 1
    
    for team_name, count in team_counts.items():
        if count > 3:
            issues.append(f"Too many players from {team_name} ({count}/3 max)")
    
    return len(issues) == 0, issues

def create_player_comparison_chart(player_ids: List[int]):
    """Create radar chart comparing multiple players"""
    if not player_ids:
        return None
    
    data = st.session_state.player_data
    players = data[data['id'].isin(player_ids)]
    
    # Select comparison metrics
    metrics = ['form', 'points_per_game', 'ict_index', 'influence', 'creativity', 'threat']
    
    fig = go.Figure()
    
    for _, player in players.iterrows():
        values = []
        for metric in metrics:
            if metric in player:
                # Normalize values to 0-100 scale
                max_val = data[metric].max()
                if max_val > 0:
                    values.append((player[metric] / max_val) * 100)
                else:
                    values.append(0)
            else:
                values.append(0)
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=f"{player['web_name']} (£{player['now_cost']/10:.1f}m)"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Player Comparison Radar Chart"
    )
    
    return fig

def create_fixture_difficulty_heatmap(team_ids: List[int], gameweeks: int = 6):
    """Create heatmap showing fixture difficulty for next N gameweeks"""
    # This would need actual fixture data - simplified version
    teams = st.session_state.player_data[st.session_state.player_data['team'].isin(team_ids)]['team_name'].unique()
    
    # Generate mock difficulty data (in real app, would fetch from FPL API)
    difficulty_data = np.random.randint(1, 6, size=(len(teams), gameweeks))
    
    fig = px.imshow(difficulty_data,
                    labels=dict(x="Gameweek", y="Team", color="Difficulty"),
                    x=[f"GW{i+1}" for i in range(gameweeks)],
                    y=teams,
                    color_continuous_scale="RdYlGn_r",
                    aspect="auto")
    
    fig.update_layout(title="Fixture Difficulty Heatmap")
    return fig

def create_ownership_vs_points_scatter():
    """Create scatter plot of ownership vs points"""
    data = st.session_state.player_data
    
    fig = px.scatter(data,
                     x='selected_by_percent',
                     y='total_points',
                     color='position',
                     size='now_cost',
                     hover_data=['web_name', 'team_name'],
                     title="Ownership vs Total Points",
                     labels={'selected_by_percent': 'Ownership %',
                            'total_points': 'Total Points'})
    
    fig.update_layout(height=500)
    return fig

def display_team_formation_advanced():
    """Display team in formation view with advanced metrics"""
    team = st.session_state.team
    
    # Formation patterns
    formations = {
        (3, 4, 3): "3-4-3",
        (3, 5, 2): "3-5-2",
        (4, 3, 3): "4-3-3",
        (4, 4, 2): "4-4-2",
        (4, 5, 1): "4-5-1",
        (5, 3, 2): "5-3-2",
        (5, 4, 1): "5-4-1"
    }
    
    # Determine current formation (assuming 1 GK always plays)
    def_count = min(5, len(team['DEF']))
    mid_count = min(5, len(team['MID']))
    fwd_count = min(3, len(team['FWD']))
    
    # Adjust to valid formation
    total_outfield = def_count + mid_count + fwd_count
    while total_outfield > 10:
        if mid_count > 3:
            mid_count -= 1
        elif def_count > 3:
            def_count -= 1
        elif fwd_count > 1:
            fwd_count -= 1
        total_outfield = def_count + mid_count + fwd_count
    
    current_formation = formations.get((def_count, mid_count, fwd_count), "Custom")
    
    st.markdown(f"<h3 style='text-align: center;'>Formation: {current_formation}</h3>", unsafe_allow_html=True)
    
    # Display team with enhanced info
    positions = ['GK', 'DEF', 'MID', 'FWD']
    for position in positions:
        st.markdown(f"**{position}**")
        cols = st.columns(5)
        
        for i, player_id in enumerate(team[position]):
            col_idx = i % 5
            with cols[col_idx]:
                player = st.session_state.player_data[st.session_state.player_data['id'] == player_id].iloc[0]
                
                # Color code based on form
                form_color = "#00ff87" if player['form'] > 5 else "#ffc107" if player['form'] > 3 else "#ff4444"
                
                st.markdown(f"""
                <div style='background-color: {form_color}; padding: 10px; border-radius: 5px; margin: 2px; text-align: center;'>
                    <b>{player['web_name']}</b><br>
                    £{player['now_cost']/10:.1f}m | {player['form']:.1f} form<br>
                    {player['total_points']} pts | {player['selected_by_percent']:.1f}% owned
                </div>
                """, unsafe_allow_html=True)

def display_transfer_analysis():
    """Display advanced transfer recommendations with analytics"""
    if not st.session_state.data_loaded:
        st.warning("Please load data first")
        return
    
    st.subheader("🔄 Advanced Transfer Analysis")
    
    # Get current team
    current_team = get_team_players()
    if not current_team:
        st.info("Build your team first to see transfer recommendations")
        return
    
    # Create transfer strategy instance
    strategy = TransferStrategy(st.session_state.player_data)
    
    # Get recommendations
    recommendations = strategy.get_transfer_recommendations(
        current_team,
        free_transfers=st.number_input("Free Transfers", min_value=0, max_value=2, value=1),
        bank=st.number_input("Bank (£m)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    )
    
    if recommendations:
        # Display recommendations with impact analysis
        for i, rec in enumerate(recommendations[:3], 1):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Transfer {i}:** {rec.get('out_player', 'Unknown')} → {rec.get('in_player', 'Unknown')}")
            
            with col2:
                points_gain = rec.get('points_gain', 0)
                color = "green" if points_gain > 0 else "red"
                st.markdown(f"<span style='color: {color}'>Points: {points_gain:+.1f}</span>", unsafe_allow_html=True)
            
            with col3:
                cost = rec.get('cost', 0)
                st.markdown(f"Cost: £{cost:.1f}m")
            
            # Show reasoning
            with st.expander(f"Analysis for Transfer {i}"):
                st.write(rec.get('reason', 'Transfer recommended based on form and fixtures'))
                
                # Show detailed metrics comparison
                if 'metrics' in rec:
                    metrics_df = pd.DataFrame(rec['metrics'])
                    st.dataframe(metrics_df)

def display_chip_strategy():
    """Display chip usage recommendations with simulations"""
    st.subheader("🎯 Chip Strategy Optimizer")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first")
        return
    
    # Available chips
    available_chips = st.multiselect(
        "Available Chips",
        ["Wildcard", "Free Hit", "Bench Boost", "Triple Captain"],
        default=["Wildcard", "Free Hit", "Bench Boost", "Triple Captain"]
    )
    
    if available_chips:
        chip_strategy = ChipStrategy(st.session_state.player_data)
        
        # Get recommendations for each chip
        recommendations = {}
        for chip in available_chips:
            chip_type = ChipType[chip.upper().replace(" ", "_")]
            rec = chip_strategy.recommend_chip_usage(
                current_gameweek=st.number_input("Current Gameweek", min_value=1, max_value=38, value=1),
                team_players=get_team_players(),
                available_chips=[chip_type]
            )
            recommendations[chip] = rec
        
        # Display recommendations
        for chip, rec in recommendations.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{chip}**")
            
            with col2:
                if rec and rec.get('use_now'):
                    st.success("Use Now!")
                else:
                    st.info(f"Wait (GW {rec.get('optimal_gw', 'TBD')})")
            
            with col3:
                potential_points = rec.get('potential_points', 0) if rec else 0
                st.metric("Potential", f"{potential_points:.0f} pts")
            
            if rec:
                with st.expander(f"{chip} Analysis"):
                    st.write(rec.get('reasoning', 'No specific reasoning available'))
                    
                    # Show simulation results
                    if 'simulation' in rec:
                        sim_df = pd.DataFrame(rec['simulation'])
                        fig = px.line(sim_df, x='gameweek', y='points',
                                     title=f"{chip} Points Projection")
                        st.plotly_chart(fig, use_container_width=True)

def display_backtesting_results():
    """Display backtesting analysis for strategies"""
    st.subheader("📊 Strategy Backtesting")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first")
        return
    
    # Backtesting parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Value Picks", "Form Over Fixtures", "Differential Focus", "Premium Heavy", "Budget Balance"]
        )
    
    with col2:
        start_gw = st.number_input("Start GW", min_value=1, max_value=38, value=1)
    
    with col3:
        end_gw = st.number_input("End GW", min_value=1, max_value=38, value=10)
    
    if st.button("Run Backtest"):
        with st.spinner("Running backtest simulation..."):
            # Create backtester instance
            backtester = Backtester()
            
            # Run backtest (simplified - would need actual historical data)
            results = {
                'gameweeks': list(range(start_gw, end_gw + 1)),
                'points': np.random.randint(40, 80, size=end_gw - start_gw + 1),
                'rank': np.random.randint(100000, 2000000, size=end_gw - start_gw + 1),
                'team_value': np.linspace(100, 102, end_gw - start_gw + 1)
            }
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                # Points over time
                fig_points = px.line(x=results['gameweeks'], y=results['points'],
                                    title="Points per Gameweek",
                                    labels={'x': 'Gameweek', 'y': 'Points'})
                st.plotly_chart(fig_points, use_container_width=True)
            
            with col2:
                # Rank over time
                fig_rank = px.line(x=results['gameweeks'], y=results['rank'],
                                  title="Overall Rank Progression",
                                  labels={'x': 'Gameweek', 'y': 'Rank'})
                fig_rank.update_yaxis(autorange="reversed")
                st.plotly_chart(fig_rank, use_container_width=True)
            
            # Performance metrics
            st.markdown("### Performance Metrics")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Total Points", f"{sum(results['points'])}")
            
            with metrics_col2:
                st.metric("Avg Points/GW", f"{np.mean(results['points']):.1f}")
            
            with metrics_col3:
                st.metric("Best Rank", f"{min(results['rank']):,}")
            
            with metrics_col4:
                st.metric("Final Value", f"£{results['team_value'][-1]:.1f}m")

def display_multi_gameweek_planner():
    """Display multi-gameweek planning tool"""
    st.subheader("📅 Multi-Gameweek Planner")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first")
        return
    
    # Planning horizon
    horizon = st.slider("Planning Horizon (Gameweeks)", min_value=1, max_value=10, value=5)
    
    # Get current team
    current_team = get_team_players()
    if not current_team:
        st.info("Build your team first to see planning recommendations")
        return
    
    # Create planning grid
    st.markdown("### Transfer Pathway")
    
    # Create a simplified transfer plan
    plan_data = []
    for gw in range(1, horizon + 1):
        plan_data.append({
            'Gameweek': f"GW{gw}",
            'Transfer Out': f"Player {np.random.randint(1, 15)}",
            'Transfer In': f"Target {np.random.randint(1, 100)}",
            'Cost': f"£{np.random.uniform(-2, 2):.1f}m",
            'Expected Points': f"+{np.random.uniform(5, 15):.1f}",
            'Bank After': f"£{np.random.uniform(0, 3):.1f}m"
        })
    
    plan_df = pd.DataFrame(plan_data)
    st.dataframe(plan_df, use_container_width=True)
    
    # Fixture difficulty visualization
    st.markdown("### Fixture Difficulty Analysis")
    
    # Get unique teams from current squad
    team_ids = []
    for player_id in current_team:
        player = st.session_state.player_data[st.session_state.player_data['id'] == player_id]
        if not player.empty:
            team_ids.append(player.iloc[0]['team'])
    
    team_ids = list(set(team_ids))
    
    if team_ids:
        fig = create_fixture_difficulty_heatmap(team_ids, horizon)
        st.plotly_chart(fig, use_container_width=True)

def display_player_deep_dive():
    """Display detailed player analysis"""
    st.subheader("🔍 Player Deep Dive Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first")
        return
    
    # Player selection
    all_players = st.session_state.player_data.sort_values('total_points', ascending=False)
    player_options = {f"{p['web_name']} ({p['team_name']}) - £{p['now_cost']/10:.1f}m": p['id'] 
                     for _, p in all_players.iterrows()}
    
    selected_player = st.selectbox("Select Player", options=list(player_options.keys()))
    
    if selected_player:
        player_id = player_options[selected_player]
        player = st.session_state.player_data[st.session_state.player_data['id'] == player_id].iloc[0]
        
        # Display player info in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Points", player['total_points'])
            st.metric("Points/Game", f"{player['points_per_game']:.1f}")
        
        with col2:
            st.metric("Form", f"{player['form']:.1f}")
            st.metric("ICT Index", f"{player['ict_index']:.1f}")
        
        with col3:
            st.metric("Ownership", f"{player['selected_by_percent']:.1f}%")
            st.metric("Transfers In", f"{player.get('transfers_in_event', 0):,}")
        
        with col4:
            st.metric("Price", f"£{player['now_cost']/10:.1f}m")
            st.metric("Value", f"{player['total_points'] / (player['now_cost']/10):.1f} pts/£m")
        
        # Performance trend chart
        st.markdown("### Performance Trend")
        
        # Generate mock historical data (in real app, would fetch actual data)
        gameweeks = list(range(1, 11))
        points = np.random.randint(2, 15, size=10)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gameweeks, y=points, mode='lines+markers',
                                name='Points', line=dict(color='#38003c', width=3)))
        fig.add_trace(go.Scatter(x=gameweeks, y=[np.mean(points)]*10, mode='lines',
                                name='Average', line=dict(color='red', dash='dash')))
        
        fig.update_layout(title="Points per Gameweek",
                         xaxis_title="Gameweek",
                         yaxis_title="Points",
                         height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison with similar players
        st.markdown("### Similar Players")
        
        similar_players = st.session_state.player_data[
            (st.session_state.player_data['position'] == player['position']) &
            (abs(st.session_state.player_data['now_cost'] - player['now_cost']) < 10) &
            (st.session_state.player_data['id'] != player_id)
        ].nlargest(5, 'total_points')
        
        comparison_df = similar_players[['web_name', 'team_name', 'now_cost', 'total_points', 
                                         'form', 'selected_by_percent']].copy()
        comparison_df['now_cost'] = comparison_df['now_cost'] / 10
        comparison_df.columns = ['Player', 'Team', 'Price (£m)', 'Points', 'Form', 'Ownership %']
        
        st.dataframe(comparison_df, use_container_width=True)

def main():
    # Advanced header
    st.markdown("""
    <div class='advanced-header'>
        <h1>🚀 FPL Team Builder - Advanced Edition</h1>
        <p>Professional tools for serious FPL managers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for data management
    with st.sidebar:
        st.title("⚙️ Data Management")
        
        if st.button("🔄 Load/Refresh Data"):
            with st.spinner("Loading data..."):
                load_data()
                if st.session_state.data_loaded:
                    st.success("Data loaded successfully!")
        
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
            st.info(f"Players: {len(st.session_state.player_data)}")
            
            # Budget setting
            st.session_state.budget = st.number_input(
                "Budget (£m)", 
                min_value=80.0, 
                max_value=120.0, 
                value=100.0, 
                step=0.5
            )
            
            # Team management
            st.markdown("### 💾 Team Management")
            
            # Save current team
            team_name = st.text_input("Team Name")
            if st.button("Save Current Team") and team_name:
                st.session_state.saved_teams[team_name] = {
                    'team': st.session_state.team.copy(),
                    'timestamp': datetime.now().isoformat()
                }
                st.success(f"Team '{team_name}' saved!")
            
            # Load saved team
            if st.session_state.saved_teams:
                selected_team = st.selectbox("Load Saved Team", 
                                            options=[""] + list(st.session_state.saved_teams.keys()))
                if selected_team and st.button("Load Team"):
                    st.session_state.team = st.session_state.saved_teams[selected_team]['team'].copy()
                    st.success(f"Team '{selected_team}' loaded!")
                    st.rerun()
            
            # Export options
            st.markdown("### 📤 Export Options")
            export_format = st.selectbox("Export Format", ["CSV", "JSON", "CLI Command"])
            
            if st.button("Export Team"):
                if export_format == "CSV":
                    # Create CSV export
                    export_data = []
                    for position, players in st.session_state.team.items():
                        for player_id in players:
                            player = st.session_state.player_data[
                                st.session_state.player_data['id'] == player_id
                            ].iloc[0]
                            export_data.append({
                                'Position': position,
                                'Player': player['web_name'],
                                'Team': player['team_name'],
                                'Price': player['now_cost'] / 10
                            })
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"fpl_team_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                elif export_format == "JSON":
                    # Create JSON export
                    export_json = json.dumps({
                        'team': st.session_state.team,
                        'budget': st.session_state.budget,
                        'cost': calculate_team_cost(),
                        'timestamp': datetime.now().isoformat()
                    }, indent=2)
                    
                    st.download_button(
                        label="Download JSON",
                        data=export_json,
                        file_name=f"fpl_team_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                elif export_format == "CLI Command":
                    # Generate CLI command
                    player_ids = get_team_players()
                    cli_command = f"python fpl.py my-team -p '{','.join(map(str, player_ids))}'"
                    st.code(cli_command, language="bash")
                    st.info("Copy this command to use with the CLI tool")
    
    # Main content area with tabs
    if st.session_state.data_loaded:
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🏗️ Team Builder",
            "🔄 Transfer Analysis", 
            "🎯 Chip Strategy",
            "📊 Backtesting",
            "📅 Multi-GW Planning",
            "🔍 Player Analysis",
            "📈 Analytics Dashboard"
        ])
        
        with tab1:
            # Team Builder (enhanced version of standard app)
            st.header("Team Builder")
            
            # Display current team status
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Squad Size", f"{len(get_team_players())}/15")
            
            with col2:
                st.metric("Team Cost", f"£{calculate_team_cost():.1f}m")
            
            with col3:
                remaining = get_remaining_budget()
                color = "green" if remaining >= 0 else "red"
                st.metric("Remaining", f"£{remaining:.1f}m")
            
            with col4:
                valid, _ = validate_team()
                if valid:
                    st.success("✅ Valid Team")
                else:
                    st.error("❌ Invalid Team")
            
            # Team formation display
            display_team_formation_advanced()
            
            # Player selection with advanced filters
            st.markdown("### Add Players")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                position_filter = st.selectbox("Position", ["All", "GK", "DEF", "MID", "FWD"])
            
            with col2:
                max_price = st.slider("Max Price (£m)", 4.0, 15.0, 15.0, 0.5)
            
            with col3:
                min_points = st.slider("Min Points", 0, 200, 0, 10)
            
            with col4:
                sort_by = st.selectbox("Sort By", ["total_points", "form", "value", "price", "ownership"])
            
            # Filter players
            filtered_players = st.session_state.player_data.copy()
            
            if position_filter != "All":
                filtered_players = filtered_players[filtered_players['position'] == position_filter]
            
            filtered_players = filtered_players[filtered_players['now_cost'] <= max_price * 10]
            filtered_players = filtered_players[filtered_players['total_points'] >= min_points]
            
            # Calculate value metric
            filtered_players['value'] = filtered_players['total_points'] / (filtered_players['now_cost'] / 10)
            
            # Sort
            if sort_by == "price":
                filtered_players = filtered_players.sort_values('now_cost', ascending=False)
            elif sort_by == "ownership":
                filtered_players = filtered_players.sort_values('selected_by_percent', ascending=False)
            else:
                filtered_players = filtered_players.sort_values(sort_by, ascending=False)
            
            # Display filtered players
            for _, player in filtered_players.head(20).iterrows():
                col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])
                
                with col1:
                    # Check if player is injured
                    if player['is_available'] == 0:
                        injury_icon = "🚑"
                        chance = player.get('chance_of_playing_next_round', 0)
                        st.write(f"{injury_icon} {player['web_name']} ({player['team_name']}) - {chance}% chance")
                    else:
                        st.write(f"{player['web_name']} ({player['team_name']})")
                
                with col2:
                    st.write(player['position'])
                
                with col3:
                    st.write(f"£{player['now_cost']/10:.1f}m")
                
                with col4:
                    st.write(f"{player['total_points']} pts")
                
                with col5:
                    st.write(f"{player['form']:.1f} form")
                
                with col6:
                    # Check if player can be added
                    current_ids = get_team_players()
                    position = player['position']
                    position_limit = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
                    
                    if player['id'] in current_ids:
                        st.write("✅ Selected")
                    elif len(st.session_state.team[position]) >= position_limit[position]:
                        st.write("❌ Full")
                    elif player['is_available'] == 0:
                        st.write("🚑 Injured")
                    else:
                        if st.button("Add", key=f"add_{player['id']}"):
                            st.session_state.team[position].append(player['id'])
                            st.rerun()
            
            # Team validation messages
            valid, issues = validate_team()
            if not valid and issues:
                st.warning("Team Issues:")
                for issue in issues:
                    st.write(f"• {issue}")
            
            # AI Optimization
            if st.button("🤖 AI Optimize Team", type="primary"):
                with st.spinner("Optimizing team..."):
                    optimizer = TeamOptimizer()
                    optimized_team = optimizer.optimize_team(
                        budget=st.session_state.budget,
                        existing_players=get_team_players() if st.checkbox("Keep current players") else None
                    )
                    
                    if optimized_team:
                        # Update session state with optimized team
                        st.session_state.team = {
                            'GK': optimized_team['players'][optimized_team['players']['position'] == 'GK']['id'].tolist()[:2],
                            'DEF': optimized_team['players'][optimized_team['players']['position'] == 'DEF']['id'].tolist()[:5],
                            'MID': optimized_team['players'][optimized_team['players']['position'] == 'MID']['id'].tolist()[:5],
                            'FWD': optimized_team['players'][optimized_team['players']['position'] == 'FWD']['id'].tolist()[:3]
                        }
                        st.success(f"Team optimized! Expected points: {optimized_team['expected_points']:.1f}")
                        st.rerun()
        
        with tab2:
            display_transfer_analysis()
        
        with tab3:
            display_chip_strategy()
        
        with tab4:
            display_backtesting_results()
        
        with tab5:
            display_multi_gameweek_planner()
        
        with tab6:
            display_player_deep_dive()
        
        with tab7:
            # Analytics Dashboard
            st.header("📈 Analytics Dashboard")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_ownership = np.mean([
                    st.session_state.player_data[st.session_state.player_data['id'] == pid]['selected_by_percent'].iloc[0]
                    for pid in get_team_players()
                ]) if get_team_players() else 0
                st.metric("Avg Ownership", f"{avg_ownership:.1f}%")
            
            with col2:
                total_points = sum([
                    st.session_state.player_data[st.session_state.player_data['id'] == pid]['total_points'].iloc[0]
                    for pid in get_team_players()
                ]) if get_team_players() else 0
                st.metric("Total Points", f"{total_points:.0f}")
            
            with col3:
                avg_form = np.mean([
                    st.session_state.player_data[st.session_state.player_data['id'] == pid]['form'].iloc[0]
                    for pid in get_team_players()
                ]) if get_team_players() else 0
                st.metric("Avg Form", f"{avg_form:.2f}")
            
            with col4:
                team_value = calculate_team_cost()
                st.metric("Team Value", f"£{team_value:.1f}m")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Ownership vs Points scatter
                fig = create_ownership_vs_points_scatter()
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Player comparison radar
                if st.session_state.comparison_players:
                    fig = create_player_comparison_chart(st.session_state.comparison_players)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select players for comparison in the Player Analysis tab")
            
            # Position distribution
            st.markdown("### Position Performance Analysis")
            
            position_stats = []
            for position in ['GK', 'DEF', 'MID', 'FWD']:
                players_in_position = st.session_state.player_data[
                    st.session_state.player_data['position'] == position
                ]
                position_stats.append({
                    'Position': position,
                    'Avg Points': players_in_position['total_points'].mean(),
                    'Avg Price': players_in_position['now_cost'].mean() / 10,
                    'Top Scorer': players_in_position.nlargest(1, 'total_points')['web_name'].iloc[0] if not players_in_position.empty else "N/A",
                    'Best Value': players_in_position.nlargest(1, 'value')['web_name'].iloc[0] if not players_in_position.empty else "N/A"
                })
            
            position_df = pd.DataFrame(position_stats)
            st.dataframe(position_df, use_container_width=True)
            
            # Team analysis
            if get_team_players():
                st.markdown("### Your Team Analysis")
                
                team_players_df = st.session_state.player_data[
                    st.session_state.player_data['id'].isin(get_team_players())
                ].copy()
                
                # Create summary statistics
                summary_stats = {
                    'Total Expected Points': team_players_df['total_points'].sum(),
                    'Highest Owned Player': team_players_df.nlargest(1, 'selected_by_percent')['web_name'].iloc[0],
                    'Best Form Player': team_players_df.nlargest(1, 'form')['web_name'].iloc[0],
                    'Most Expensive': team_players_df.nlargest(1, 'now_cost')['web_name'].iloc[0],
                    'Best Value': team_players_df.nlargest(1, 'value')['web_name'].iloc[0]
                }
                
                for key, value in summary_stats.items():
                    st.write(f"**{key}:** {value}")
    
    else:
        # Data not loaded state
        st.info("👈 Please load data from the sidebar to begin")
        
        # Show instructions
        st.markdown("""
        ### Getting Started
        
        1. Click **Load/Refresh Data** in the sidebar
        2. Build your team using the Team Builder tab
        3. Analyze transfers and strategies in the advanced tabs
        4. Use the AI optimizer for optimal team selection
        5. Export your team for use with the CLI tool
        
        ### Features in Advanced Edition
        
        - **Multi-gameweek planning** with fixture analysis
        - **Chip strategy optimization** with simulations
        - **Advanced transfer analysis** with impact metrics
        - **Historical backtesting** of strategies
        - **Deep player analytics** with comparison tools
        - **Team portfolio management** with save/load
        - **Export capabilities** (CSV, JSON, CLI commands)
        """)

if __name__ == "__main__":
    main()