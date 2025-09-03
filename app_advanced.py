"""
FPL Team Builder - Advanced GUI with Analysis Features
Run with: streamlit run app_advanced.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import our modules
from src.data.data_merger import DataMerger
from src.models.rule_based_scorer import RuleBasedScorer
from src.my_team import TeamAnalyzer, MyTeam, TransferSuggestion

# Page config
st.set_page_config(
    page_title="FPL Manager Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 0rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #38003c;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 5px;
        gap: 10px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff87;
        color: #38003c;
    }
    .player-card-small {
        background: linear-gradient(135deg, #38003c 0%, #00ff87 100%);
        padding: 8px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-size: 0.9em;
        min-height: 100px;
    }
    .metric-card {
        background: #f0f0f0;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'team' not in st.session_state:
    st.session_state.team = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
if 'budget' not in st.session_state:
    st.session_state.budget = 100.0
if 'captain' not in st.session_state:
    st.session_state.captain = None
if 'vice_captain' not in st.session_state:
    st.session_state.vice_captain = None
if 'bench' not in st.session_state:
    st.session_state.bench = []
if 'starting_11' not in st.session_state:
    st.session_state.starting_11 = []

@st.cache_data
def load_player_data():
    """Load and cache player data"""
    try:
        merger = DataMerger("data/fpl_data.db")
        data = merger.load_from_database()
        merger.close()
        if not data.empty:
            data['display_name'] = data['player_name'] + ' (' + data['team_name'].str[:3] + ')'
            return data
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_pitch_visualization(team_data):
    """Create an interactive pitch visualization with the team"""
    fig = go.Figure()
    
    # Draw pitch
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100,
                  line=dict(color="white", width=2),
                  fillcolor="green", opacity=0.3)
    
    # Penalty areas
    fig.add_shape(type="rect", x0=30, y0=0, x1=70, y1=18,
                  line=dict(color="white", width=2))
    fig.add_shape(type="rect", x0=30, y0=82, x1=70, y1=100,
                  line=dict(color="white", width=2))
    
    # Center circle
    fig.add_shape(type="circle", x0=40, y0=40, x1=60, y1=60,
                  line=dict(color="white", width=2))
    
    # Center line
    fig.add_shape(type="line", x0=0, y0=50, x1=100, y1=50,
                  line=dict(color="white", width=2))
    
    # Position coordinates
    positions = {
        'GK': [(50, 5)],
        'DEF': [(20, 25), (35, 25), (50, 25), (65, 25), (80, 25)],
        'MID': [(20, 50), (35, 50), (50, 50), (65, 50), (80, 50)],
        'FWD': [(35, 75), (50, 75), (65, 75)]
    }
    
    # Add players to pitch
    for pos, coords in positions.items():
        players = st.session_state.team[pos]
        for i, (x, y) in enumerate(coords[:len(players)]):
            if i < len(players):
                player = team_data[team_data['player_id'] == players[i]].iloc[0]
                
                # Determine color based on captain/vice captain
                color = 'gold' if players[i] == st.session_state.captain else \
                        'silver' if players[i] == st.session_state.vice_captain else 'white'
                
                # Add player marker
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=30, color=color, line=dict(color='black', width=2)),
                    text=player['player_name'].split()[-1][:10],
                    textposition='bottom center',
                    name=player['player_name'],
                    hovertemplate=f"<b>{player['player_name']}</b><br>" +
                                  f"Team: {player['team_name']}<br>" +
                                  f"Price: £{player['price']:.1f}m<br>" +
                                  f"Points: {int(player.get('total_points', 0))}<br>" +
                                  "<extra></extra>"
                ))
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False, range=[0, 100]),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='darkgreen',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def get_team_stats(team_data):
    """Calculate team statistics"""
    team_players = []
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        for pid in st.session_state.team[pos]:
            player = team_data[team_data['player_id'] == pid]
            if not player.empty:
                team_players.append(player.iloc[0])
    
    if not team_players:
        return {}
    
    df = pd.DataFrame(team_players)
    
    stats = {
        'total_cost': df['price'].sum(),
        'total_points': df['total_points'].sum(),
        'avg_points': df['total_points'].mean(),
        'avg_ownership': df['selected_by_percent'].mean(),
        'team_distribution': df['team_name'].value_counts().to_dict()
    }
    
    return stats

def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("⚽ FPL Manager Pro")
        st.markdown("Complete Fantasy Premier League Management System")
    
    # Load data
    data = load_player_data()
    if data.empty:
        st.error("No data available. Please run 'python fpl.py fetch-data' first.")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏗️ Team Builder",
        "📊 Team Analysis", 
        "🔄 Transfers",
        "©️ Captain Selection",
        "📈 Performance"
    ])
    
    with tab1:
        # Team Builder Tab
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("⚽ Your Squad")
            
            # Visual pitch
            if len([p for pos in st.session_state.team.values() for p in pos]) > 0:
                fig = create_pitch_visualization(data)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Add players to see them on the pitch")
            
            # Team formation selector
            st.subheader("📋 Formation & Lineup")
            formation = st.select_slider(
                "Select Formation",
                options=["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"],
                value="4-4-2"
            )
            
        with col2:
            st.subheader("➕ Add Players")
            
            # Budget display
            team_cost = sum([
                data[data['player_id'] == pid]['price'].iloc[0]
                for pos in st.session_state.team.values()
                for pid in pos
                if not data[data['player_id'] == pid].empty
            ])
            remaining = st.session_state.budget - team_cost
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.metric("💰 Budget Used", f"£{team_cost:.1f}m")
            with col_b2:
                if remaining >= 0:
                    st.metric("💵 Remaining", f"£{remaining:.1f}m")
                else:
                    st.metric("⚠️ Over Budget", f"£{abs(remaining):.1f}m", delta_color="inverse")
            
            # Position selector with icons
            position = st.radio(
                "Select Position",
                options=['GK', 'DEF', 'MID', 'FWD'],
                format_func=lambda x: {
                    'GK': '🥅 Goalkeeper',
                    'DEF': '🛡️ Defender',
                    'MID': '⚡ Midfielder',
                    'FWD': '⚔️ Forward'
                }[x],
                horizontal=True
            )
            
            # Position limits check
            limits = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            current = len(st.session_state.team[position])
            st.progress(current / limits[position], text=f"{current}/{limits[position]} {position}s selected")
            
            if current < limits[position]:
                # Filter available players
                available = data[data['position'] == position].copy()
                selected_ids = [pid for pos in st.session_state.team.values() for pid in pos]
                available = available[~available['player_id'].isin(selected_ids)]
                available = available[available['price'] <= remaining]
                
                # Advanced filters in expander
                with st.expander("🔍 Advanced Filters"):
                    col_f1, col_f2 = st.columns(2)
                    
                    with col_f1:
                        teams = ['All'] + sorted(available['team_name'].unique().tolist())
                        team_filter = st.selectbox("Team", teams)
                        if team_filter != 'All':
                            available = available[available['team_name'] == team_filter]
                    
                    with col_f2:
                        min_points = st.number_input("Min Points", value=0)
                        available = available[available['total_points'] >= min_points]
                    
                    price_range = st.slider(
                        "Price Range (£m)",
                        min_value=float(available['price'].min()) if not available.empty else 4.0,
                        max_value=float(available['price'].max()) if not available.empty else 15.0,
                        value=(4.0, remaining)
                    )
                    available = available[
                        (available['price'] >= price_range[0]) & 
                        (available['price'] <= price_range[1])
                    ]
                
                # Sort options
                sort_col1, sort_col2 = st.columns([2, 1])
                with sort_col1:
                    sort_by = st.selectbox(
                        "Sort by",
                        ['total_points', 'price', 'form', 'selected_by_percent'],
                        format_func=lambda x: {
                            'total_points': '📊 Points',
                            'price': '💰 Price',
                            'form': '📈 Form',
                            'selected_by_percent': '👥 Ownership'
                        }[x]
                    )
                with sort_col2:
                    order = st.radio("Order", ['⬇️ High to Low', '⬆️ Low to High'], horizontal=True)
                
                available = available.sort_values(sort_by, ascending=(order == '⬆️ Low to High'))
                
                # Player selection dropdown
                if not available.empty:
                    available['display_full'] = (
                        available['player_name'] + ' | ' +
                        available['team_name'].str[:3] + ' | £' +
                        available['price'].astype(str) + 'm | ' +
                        available['total_points'].astype(int).astype(str) + 'pts'
                    )
                    
                    selected = st.selectbox(
                        "Select Player",
                        options=available['player_id'].tolist(),
                        format_func=lambda x: available[available['player_id'] == x]['display_full'].iloc[0]
                    )
                    
                    if selected:
                        player = available[available['player_id'] == selected].iloc[0]
                        
                        # Player stats card
                        st.markdown("---")
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric("Points", int(player['total_points']))
                        with col_s2:
                            st.metric("Form", f"{player.get('form', 0):.1f}")
                        with col_s3:
                            st.metric("Selected", f"{player.get('selected_by_percent', 0):.1f}%")
                        
                        if st.button(f"➕ Add {player['player_name']}", type="primary", use_container_width=True):
                            st.session_state.team[position].append(selected)
                            st.success(f"Added {player['player_name']}!")
                            st.rerun()
                else:
                    st.warning("No players available with current filters")
            else:
                st.info(f"Maximum {limits[position]} {position}s reached")
            
            # Team actions
            st.markdown("---")
            st.subheader("🎮 Actions")
            
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                if st.button("🔄 Clear Team", use_container_width=True):
                    st.session_state.team = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
                    st.session_state.captain = None
                    st.session_state.vice_captain = None
                    st.rerun()
            
            with col_a2:
                if st.button("💾 Save Team", use_container_width=True):
                    # Save logic here
                    st.success("Team saved!")
    
    with tab2:
        # Team Analysis Tab
        st.subheader("📊 Team Analysis")
        
        if any(st.session_state.team.values()):
            stats = get_team_stats(data)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cost", f"£{stats.get('total_cost', 0):.1f}m")
            with col2:
                st.metric("Total Points", int(stats.get('total_points', 0)))
            with col3:
                st.metric("Avg Points", f"{stats.get('avg_points', 0):.1f}")
            with col4:
                st.metric("Avg Ownership", f"{stats.get('avg_ownership', 0):.1f}%")
            
            # Team distribution chart
            if stats.get('team_distribution'):
                fig = px.bar(
                    x=list(stats['team_distribution'].keys()),
                    y=list(stats['team_distribution'].values()),
                    labels={'x': 'Team', 'y': 'Number of Players'},
                    title="Players per Team"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Position breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Position Breakdown")
                for pos in ['GK', 'DEF', 'MID', 'FWD']:
                    players_in_pos = st.session_state.team[pos]
                    if players_in_pos:
                        st.write(f"**{pos}:**")
                        for pid in players_in_pos:
                            p = data[data['player_id'] == pid]
                            if not p.empty:
                                p = p.iloc[0]
                                st.write(f"• {p['player_name']} ({p['team_name'][:3]}) - £{p['price']:.1f}m")
            
            with col2:
                st.markdown("### Top Performers")
                team_players = []
                for pos in ['GK', 'DEF', 'MID', 'FWD']:
                    for pid in st.session_state.team[pos]:
                        p = data[data['player_id'] == pid]
                        if not p.empty:
                            team_players.append(p.iloc[0])
                
                if team_players:
                    df = pd.DataFrame(team_players).nlargest(5, 'total_points')
                    for _, player in df.iterrows():
                        st.write(f"• {player['player_name']}: {int(player['total_points'])} pts")
        else:
            st.info("Build your team first to see analysis")
    
    with tab3:
        # Transfer Recommendations Tab
        st.subheader("🔄 Transfer Recommendations")
        
        if any(st.session_state.team.values()):
            # Get current team
            team_players = []
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                for pid in st.session_state.team[pos]:
                    p = data[data['player_id'] == pid]
                    if not p.empty:
                        team_players.append(p.iloc[0])
            
            if team_players:
                df_team = pd.DataFrame(team_players)
                
                # Find underperformers
                st.markdown("### 📉 Consider Removing")
                underperformers = df_team.nsmallest(3, 'form')
                for _, player in underperformers.iterrows():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{player['player_name']}** ({player['team_name'][:3]})")
                    with col2:
                        st.write(f"Form: {player.get('form', 0):.1f}")
                    with col3:
                        st.write(f"£{player['price']:.1f}m")
                
                st.markdown("### 📈 Consider Adding")
                # Find best players not in team
                not_in_team = data[~data['player_id'].isin([p['player_id'] for p in team_players])]
                top_players = not_in_team.nlargest(5, 'form')
                
                for _, player in top_players.iterrows():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{player['player_name']}** ({player['team_name'][:3]})")
                    with col2:
                        st.write(f"Form: {player.get('form', 0):.1f}")
                    with col3:
                        st.write(f"£{player['price']:.1f}m")
        else:
            st.info("Build your team first to see transfer recommendations")
    
    with tab4:
        # Captain Selection Tab
        st.subheader("©️ Captain & Vice-Captain Selection")
        
        if any(st.session_state.team.values()):
            all_players = []
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                for pid in st.session_state.team[pos]:
                    p = data[data['player_id'] == pid]
                    if not p.empty:
                        all_players.append(p.iloc[0])
            
            if all_players:
                df = pd.DataFrame(all_players)
                
                # Captain selector
                captain_options = df['player_id'].tolist()
                captain_display = df.set_index('player_id')['display_name'].to_dict()
                
                st.markdown("### 👑 Captain (2x Points)")
                captain = st.selectbox(
                    "Select Captain",
                    options=[None] + captain_options,
                    format_func=lambda x: "Not Selected" if x is None else captain_display.get(x, "Unknown")
                )
                st.session_state.captain = captain
                
                # Vice Captain selector
                st.markdown("### 🥈 Vice-Captain")
                vc_options = [pid for pid in captain_options if pid != captain]
                vice_captain = st.selectbox(
                    "Select Vice-Captain",
                    options=[None] + vc_options,
                    format_func=lambda x: "Not Selected" if x is None else captain_display.get(x, "Unknown")
                )
                st.session_state.vice_captain = vice_captain
                
                # Captain recommendations
                st.markdown("### 💡 Captain Recommendations")
                top_scorers = df.nlargest(3, 'total_points')
                for i, (_, player) in enumerate(top_scorers.iterrows(), 1):
                    st.write(f"{i}. **{player['player_name']}** - {int(player['total_points'])} pts, Form: {player.get('form', 0):.1f}")
        else:
            st.info("Build your team first to select captain")
    
    with tab5:
        # Performance Tab
        st.subheader("📈 Team Performance Tracking")
        
        # This would connect to historical data
        st.info("Performance tracking will be available after saving your team and playing gameweeks")
        
        # Mock performance chart
        if any(st.session_state.team.values()):
            # Create sample data
            gameweeks = list(range(1, 11))
            points = np.random.randint(30, 80, size=10).cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=gameweeks, y=points,
                mode='lines+markers',
                name='Team Points',
                line=dict(color='#38003c', width=3),
                marker=dict(size=8, color='#00ff87')
            ))
            
            fig.update_layout(
                title="Season Performance (Sample)",
                xaxis_title="Gameweek",
                yaxis_title="Total Points",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()