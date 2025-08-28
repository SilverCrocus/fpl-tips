#!/usr/bin/env python
"""
Main FPL Recommender CLI
Run with: uv run python -m src.main
"""
import asyncio
import functools
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from pathlib import Path
import logging
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import json

# Import our modules
from src.data.fpl_api import FPLAPICollector
from src.data.odds_api import PlayerPropsCollector
from src.data.data_merger import DataMerger
from src.models.rule_based_scorer import RuleBasedScorer
from src.models.backtester import Backtester

# Setup
load_dotenv()
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """FPL Transfer & Strategy Recommender"""
    pass


def async_command(f):
    """Decorator to properly handle async Click commands"""
    import functools
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

@cli.command(name='fetch-data')
@click.option('--gameweek', '-gw', type=int, help='Gameweek number')
@async_command
async def fetch_data(gameweek):
    """Fetch latest data from FPL and Odds APIs"""
    console.print(Panel.fit("[bold cyan]FPL Data Fetcher[/bold cyan]"))
    
    with console.status("[bold green]Fetching FPL data..."):
        async with FPLAPICollector() as fpl_collector:
            # Get bootstrap data
            bootstrap = await fpl_collector.get_bootstrap_data()
            
            # Get current gameweek if not specified
            if not gameweek:
                for event in bootstrap['events']:
                    if event['is_current']:
                        gameweek = event['id']
                        break
            
            console.print(f"[green]✓[/green] Fetching data for Gameweek {gameweek}")
            
            # Get gameweek data
            gw_data = await fpl_collector.get_gameweek_data(gameweek)
            players_df = gw_data['players']
            fixtures_df = gw_data['fixtures']
            
            console.print(f"[green]✓[/green] Found {len(players_df)} players")
            console.print(f"[green]✓[/green] Found {len(fixtures_df)} fixtures")
    
    # Fetch REAL player prop odds from US bookmakers
    with console.status("[bold green]Fetching real player goal scorer odds..."):
        async with PlayerPropsCollector() as odds_collector:
            try:
                # Get all player goal scorer odds for current gameweek
                player_odds = await odds_collector.get_all_player_props_for_gameweek()
                
                if not player_odds.empty:
                    # Show summary of odds data
                    num_matches = player_odds['match_id'].nunique()
                    num_players = len(player_odds)
                    bookmakers = player_odds['bookmaker_title'].unique()
                    
                    console.print(f"[green]✓[/green] Retrieved REAL player odds:")
                    console.print(f"  • {num_players} player odds across {num_matches} matches")
                    console.print(f"  • Bookmakers: {', '.join(bookmakers[:3])}")
                    
                    # Match player names to FPL data
                    player_odds = odds_collector.match_players_to_fpl(player_odds, players_df)
                    console.print(f"[green]✓[/green] Matched odds to FPL player data")
                    
                    # Show how many players we matched
                    if 'has_real_odds' in player_odds.columns:
                        matched_count = player_odds[player_odds['has_real_odds'] == True]['player_id'].nunique()
                        console.print(f"  • {matched_count} FPL players have real bookmaker odds")
                else:
                    console.print(f"[red]✗[/red] No player odds available - cannot provide accurate recommendations")
                    console.print("[yellow]Note: This system requires real bookmaker odds for accuracy[/yellow]")
                    player_odds = None
                    
            except Exception as e:
                console.print(f"[red]Error fetching odds: {e}[/red]")
                console.print("[red]Please ensure ODDS_API_KEY is set in your .env file[/red]")
                raise
    
    # Merge data
    console.print("[bold]Merging data sources...[/bold]")
    merger = DataMerger()
    
    # Fetch additional fixtures for next 5 gameweeks
    all_fixtures = []
    async with FPLAPICollector() as fpl_collector:
        for gw in range(gameweek, min(gameweek + 5, 39)):  # Get next 5 GWs or until end of season
            gw_fixtures = await fpl_collector.get_fixtures(gw)
            if gw_fixtures:
                fixtures_df_temp = fpl_collector.process_fixtures_data(gw_fixtures)
                fixtures_df_temp['event'] = gw
                all_fixtures.append(fixtures_df_temp)
    
    # Combine all fixtures
    if all_fixtures:
        extended_fixtures_df = pd.concat(all_fixtures, ignore_index=True)
    else:
        extended_fixtures_df = fixtures_df
    
    unified_data = merger.create_unified_dataset(
        players_df,
        player_odds if not player_odds.empty else None,
        None,  # Elo data would go here
        extended_fixtures_df,  # Pass fixtures data
        gameweek=gameweek,
        season="2025-26"
    )
    
    # Save to database
    merger.save_to_database(unified_data)
    merger.close()
    
    console.print(f"[green]✓[/green] Saved {len(unified_data)} player records to database")


@cli.command()
@click.option('--position', '-p', type=click.Choice(['GK', 'DEF', 'MID', 'FWD']), 
              help='Filter by position')
@click.option('--max-price', '-mp', type=float, help='Maximum price filter')
@click.option('--top', '-t', type=int, default=10, help='Number of players to show')
def recommend(position, max_price, top):
    """Get player recommendations"""
    console.print(Panel.fit("[bold cyan]FPL Player Recommendations[/bold cyan]"))
    
    # Load latest data
    merger = DataMerger()
    data = merger.get_latest_data(top_n=500)
    
    if data.empty:
        console.print("[red]No data found. Run 'fetch-data' first.[/red]")
        return
    
    # Score players (only those with real odds)
    scorer = RuleBasedScorer()
    scored = scorer.score_all_players(data)
    
    if scored.empty:
        console.print("[red]No players with real odds found![/red]")
        console.print("Please run 'fetch-data' to get latest odds data.")
        merger.close()
        return
    
    # Filter
    if position:
        scored = scored[scored['position'] == position]
    if max_price:
        scored = scored[scored['price'] <= max_price]
    
    # Get top players
    top_players = scored.head(top)
    
    # Create table
    table = Table(title=f"Top {top} Player Recommendations")
    table.add_column("Rank", style="cyan", justify="center")
    table.add_column("Player", style="magenta")
    table.add_column("Team", style="yellow")
    table.add_column("Pos", style="green", justify="center")
    table.add_column("Price", style="red", justify="right")
    table.add_column("Score", style="bold cyan", justify="right")
    table.add_column("Form", justify="right")
    table.add_column("Points", justify="right")
    
    for idx, (_, player) in enumerate(top_players.iterrows(), 1):
        table.add_row(
            str(idx),
            player['player_name'],
            player.get('team_name', 'N/A'),
            player['position'],
            f"£{player['price']:.1f}",
            f"{player['model_score']:.1f}",
            f"{player.get('form', 0):.1f}",
            str(int(player.get('total_points', 0)))
        )
    
    console.print(table)
    merger.close()


@cli.command()
@click.option('--budget', '-b', type=float, default=100.0, help='Team budget')
def build_team(budget):
    """Build optimal team within budget"""
    console.print(Panel.fit("[bold cyan]FPL Team Builder[/bold cyan]"))
    
    # Load data
    merger = DataMerger()
    data = merger.get_latest_data(top_n=500)
    
    if data.empty:
        console.print("[red]No data found. Run 'fetch-data' first.[/red]")
        return
    
    # Score players
    scorer = RuleBasedScorer()
    scored = scorer.score_all_players(data)
    
    # Build team with constraints
    team = build_optimal_team(scored, budget)
    
    # Display team
    display_team(team, console)
    merger.close()


def build_optimal_team(players_df, budget):
    """Build optimal team respecting FPL constraints with improved algorithm
    
    Uses a value-based approach that balances high scorers with budget players
    """
    if 'model_score' not in players_df.columns:
        return {'players': [], 'total_cost': 0, 'remaining_budget': budget}
    
    selected = []
    spent = 0.0
    
    # Position requirements
    positions_needed = {
        'GK': 2,
        'DEF': 5,
        'MID': 5,
        'FWD': 3
    }
    
    team_counts = {}
    
    # Calculate value score (points per million) for better selection
    players_df = players_df.copy()
    safe_price = players_df['price'].clip(lower=0.1)
    players_df['value_score'] = players_df['model_score'] / safe_price
    
    # Strategy: Mix high-scoring premiums with value picks
    # Reserve budget for key positions
    min_prices_per_pos = {'GK': 4.0, 'DEF': 4.0, 'MID': 4.5, 'FWD': 4.5}
    
    # First pass: Get 1-2 premium players per position (high score, regardless of price)
    premiums_per_pos = {'GK': 1, 'DEF': 2, 'MID': 2, 'FWD': 1}
    
    for position, premium_count in premiums_per_pos.items():
        pos_players = players_df[
            players_df['position'] == position
        ].sort_values('model_score', ascending=False)
        
        added = 0
        for _, player in pos_players.iterrows():
            if added >= premium_count:
                break
                
            team = player['team_name']
            if team_counts.get(team, 0) >= 3:
                continue
            
            # For premiums, allow up to 15% of budget per player
            if player['price'] > budget * 0.15:
                continue
                
            if spent + player['price'] > budget * 0.85:  # Reserve 15% for bench
                continue
            
            selected.append({
                'player_id': player['player_id'],
                'player_name': player['player_name'],
                'position': player['position'],
                'team': team,
                'price': player['price'],
                'score': player['model_score'],
                'total_points': player.get('total_points', 0)
            })
            
            spent += player['price']
            team_counts[team] = team_counts.get(team, 0) + 1
            added += 1
    
    # Second pass: Fill remaining slots with best value players
    for position, count in positions_needed.items():
        current_count = sum(1 for p in selected if p['position'] == position)
        
        if current_count < count:
            # Get value players (good score per million)
            pos_players = players_df[
                (players_df['position'] == position) &
                (~players_df['player_id'].isin([p['player_id'] for p in selected]))
            ].sort_values('value_score', ascending=False)
            
            for _, player in pos_players.iterrows():
                if current_count >= count:
                    break
                    
                team = player['team_name']
                if team_counts.get(team, 0) >= 3:
                    continue
                
                if spent + player['price'] > budget:
                    continue
                
                selected.append({
                    'player_id': player['player_id'],
                    'player_name': player['player_name'],
                    'position': player['position'],
                    'team': team,
                    'price': player['price'],
                    'score': player['model_score'],
                    'total_points': player.get('total_points', 0)
                })
                
                spent += player['price']
                team_counts[team] = team_counts.get(team, 0) + 1
                current_count += 1
    
    # Final pass: If still missing players, get cheapest valid options
    for position, count in positions_needed.items():
        current_count = sum(1 for p in selected if p['position'] == position)
        
        if current_count < count:
            remaining_players = players_df[
                (players_df['position'] == position) &
                (~players_df['player_id'].isin([p['player_id'] for p in selected]))
            ].sort_values('price')
            
            for _, player in remaining_players.iterrows():
                if current_count >= count:
                    break
                    
                team = player['team_name']
                if team_counts.get(team, 0) >= 3:
                    continue
                
                if spent + player['price'] > budget:
                    # If we can't afford anyone, we have a problem
                    logger.warning(f"Cannot complete team within budget for {position}")
                    break
                
                selected.append({
                    'player_id': player['player_id'],
                    'player_name': player['player_name'],
                    'position': player['position'],
                    'team': team,
                    'price': player['price'],
                    'score': player['model_score'],
                    'total_points': player.get('total_points', 0)
                })
                
                spent += player['price']
                team_counts[team] = team_counts.get(team, 0) + 1
                current_count += 1
    
    return {
        'players': selected,
        'total_cost': spent,
        'remaining_budget': budget - spent
    }


def display_team(team, console):
    """Display team in a nice format"""
    # Group by position
    positions = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
    
    for player in team['players']:
        positions[player['position']].append(player)
    
    # Create table
    table = Table(title="Your Optimal FPL Team")
    table.add_column("Position", style="cyan")
    table.add_column("Player", style="magenta")
    table.add_column("Team", style="yellow")
    table.add_column("Price", style="red", justify="right")
    table.add_column("Score", style="bold green", justify="right")
    
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        for player in positions[pos]:
            table.add_row(
                pos,
                player['player_name'],
                player['team'],
                f"£{player['price']:.1f}",
                f"{player['score']:.1f}"
            )
    
    console.print(table)
    console.print(f"\n[bold]Total Cost:[/bold] £{team['total_cost']:.1f}")
    console.print(f"[bold]Remaining:[/bold] £{team['remaining_budget']:.1f}")


@cli.command()
@click.option('--min-odds', type=float, default=3.0, help='Minimum goal odds (lower = more likely to score)')
@click.option('--max-ownership', type=float, default=5.0, help='Maximum ownership %')
@click.option('--top', '-n', type=int, default=10, help='Number of differentials to show')
def differentials(min_odds, max_ownership, top):
    """Find differential players with high goal probability but low ownership"""
    console.print(Panel.fit("[bold cyan]FPL Differential Finder[/bold cyan]"))
    
    # Load data
    merger = DataMerger()
    data = merger.load_from_database()
    
    if data.empty:
        console.print("[red]No data found. Please run 'fetch-data' first.[/red]")
        return
        
    # Filter for differentials
    differentials_df = data[
        (data['prob_goal'] > (1 / min_odds)) &  # Good goal probability
        (data['selected_by_percent'] < max_ownership) &  # Low ownership
        (data['chance_of_playing_next_round'] >= 75)  # Likely to play
    ]
    
    if differentials_df.empty:
        console.print(f"[yellow]No differentials found with odds < {min_odds} and ownership < {max_ownership}%[/yellow]")
        return
        
    # Score and sort
    scorer = RuleBasedScorer()
    differentials_df['model_score'] = differentials_df.apply(scorer.score_player, axis=1)
    differentials_df = differentials_df.sort_values('model_score', ascending=False).head(top)
    
    # Display results
    table = Table(title=f"Top {top} Differential Players")
    table.add_column("Player", style="cyan")
    table.add_column("Team", style="yellow")
    table.add_column("Pos", style="green")
    table.add_column("Price", justify="right")
    table.add_column("Own%", justify="right")
    table.add_column("Goal%", justify="right")
    table.add_column("Score", justify="right")
    
    for _, player in differentials_df.iterrows():
        table.add_row(
            player['player_name'],
            player['team_name'],
            player['position'],
            f"£{player['price']:.1f}",
            f"{player['selected_by_percent']:.1f}%",
            f"{player['prob_goal']*100:.1f}%",
            f"{player['model_score']:.1f}"
        )
    
    console.print(table)
    merger.close()


@cli.command()
@click.option('--gameweek', '-gw', type=int, help='Gameweek to analyze')
def captain(gameweek):
    """Recommend best captain choices based on goal probability"""
    console.print(Panel.fit("[bold cyan]FPL Captain Selector[/bold cyan]"))
    
    # Load data for specific gameweek or latest
    merger = DataMerger()
    data = merger.load_from_database(gameweek=gameweek)
    
    # If no specific gameweek, get the latest gameweek for each player
    if gameweek is None and not data.empty:
        latest_gw = data.groupby('player_id')['gameweek'].max().reset_index()
        latest_gw.columns = ['player_id', 'max_gameweek']
        data = data.merge(latest_gw, on='player_id')
        data = data[data['gameweek'] == data['max_gameweek']]
        data = data.drop('max_gameweek', axis=1)
    
    if data.empty:
        console.print("[red]No data found. Please run 'fetch-data' first.[/red]")
        return
        
    # Filter for likely starters
    # Use is_available column or default to all players if not available
    if 'is_available' in data.columns:
        candidates = data[
            (data['is_available'] == 1) &
            (data['minutes'] > 60)  # Regular starters
        ].copy()  # Use copy to avoid SettingWithCopyWarning
    else:
        # Fallback: just use regular starters based on minutes
        candidates = data[data['minutes'] > 60].copy()
    
    # Calculate captain score (emphasizing goal probability)
    # Check which columns are available and build score accordingly
    captain_score = pd.Series(0.0, index=candidates.index)
    
    if 'prob_goal' in candidates.columns:
        captain_score = captain_score + candidates['prob_goal'].fillna(0) * 5.0  # Heavy weight on goals
    
    if 'prob_assist' in candidates.columns:
        # Convert to float to avoid downcasting warning
        prob_assist_values = candidates['prob_assist'].astype(float).fillna(0)
        captain_score = captain_score + prob_assist_values * 2.0
    
    if 'form' in candidates.columns:
        captain_score = captain_score + candidates['form'].fillna(0) * 0.3
    
    if 'expected_goals' in candidates.columns:
        captain_score = captain_score + candidates['expected_goals'].fillna(0) * 2.0
        
    if 'expected_assists' in candidates.columns:
        captain_score = captain_score + candidates['expected_assists'].fillna(0) * 1.5
    
    # Add total points as a factor if no odds data
    if 'prob_goal' not in candidates.columns and 'total_points' in candidates.columns:
        captain_score = captain_score + candidates['total_points'].fillna(0) * 0.1
    
    candidates['captain_score'] = captain_score
    
    # Get top 10 captain choices
    top_captains = candidates.nlargest(10, 'captain_score')
    
    # Display results
    table = Table(title="Top Captain Choices")
    table.add_column("Rank", style="bold")
    table.add_column("Player", style="cyan")
    table.add_column("Team", style="yellow")
    table.add_column("vs", style="white")
    table.add_column("Goal%", justify="right", style="green")
    table.add_column("xPts", justify="right")
    table.add_column("Form", justify="right")
    table.add_column("Score", justify="right")
    
    for i, (_, player) in enumerate(top_captains.iterrows(), 1):
        goal_prob = player.get('prob_goal', 0)
        form_val = player.get('form', 0)
        
        table.add_row(
            str(i),
            player['player_name'],
            player['team_name'],
            player.get('opponent_team', 'TBD'),
            f"{goal_prob*100:.1f}%" if goal_prob > 0 else "N/A",
            f"{player.get('expected_goals', 0):.1f}",
            f"{form_val:.1f}",
            f"{player['captain_score']:.1f}"
        )
    
    console.print(table)
    
    # Add captaincy tips
    console.print("\n[bold]Captain Tips:[/bold]")
    if not top_captains.empty:
        top_pick = top_captains.iloc[0]
        goal_prob = top_pick.get('prob_goal', 0)
        
        if goal_prob > 0:
            console.print(f"• Top pick: [cyan]{top_pick['player_name']}[/cyan] with {goal_prob*100:.1f}% goal probability")
            
            if goal_prob > 0.5:
                console.print(f"• [green]Strong captaincy choice - over 50% chance to score![/green]")
            elif goal_prob > 0.35:
                console.print(f"• [yellow]Good captaincy choice - decent goal probability[/yellow]")
            else:
                console.print(f"• [red]Consider alternative options - low goal probability this week[/red]")
        else:
            console.print(f"• Top pick: [cyan]{top_pick['player_name']}[/cyan] (based on form and expected stats)")
            console.print("• [yellow]No betting odds available - using historical performance[/yellow]")
        
    merger.close()


@cli.command()
@click.option('--season', '-s', default='2024-25', help='Season to backtest')
@click.option('--start-gw', type=int, default=1, help='Starting gameweek')
@click.option('--end-gw', type=int, default=38, help='Ending gameweek')
def backtest(season, start_gw, end_gw):
    """Run backtest on historical data"""
    console.print(Panel.fit("[bold cyan]FPL Backtest Runner[/bold cyan]"))
    
    # Load historical data
    merger = DataMerger()
    data = merger.load_from_database(season=season)
    
    if data.empty:
        console.print(f"[red]No data found for season {season}[/red]")
        return
    
    # Create model and backtester
    scorer = RuleBasedScorer()
    backtester = Backtester(data, scorer)
    
    # Run backtest
    with console.status("[bold green]Running backtest..."):
        result = backtester.run_backtest(season, start_gw, end_gw)
    
    # Display results
    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Total Points", str(result.total_points))
    
    # Handle average calculation safely
    if result.gameweek_points:
        avg_points = sum(result.gameweek_points) / len(result.gameweek_points)
        table.add_row("Average GW Points", f"{avg_points:.1f}")
    else:
        table.add_row("Average GW Points", "N/A")
    
    # Display team value correctly (it's actually the full team value, not just bank)
    if result.final_team_value > 50:  # Reasonable team value
        table.add_row("Final Team Value", f"£{result.final_team_value:.1f}m")
    else:
        table.add_row("Remaining Budget", f"£{result.final_team_value:.1f}m")
        
    table.add_row("Transfers Made", str(result.transfers_made))
    table.add_row("Captain Points", str(result.captain_points))
    table.add_row("Estimated Rank", f"Top {100-result.rank_percentile:.0f}%")
    
    console.print(table)
    
    # Add warning about missing data
    if result.total_points <= 0:
        console.print("\n[yellow]⚠️  Warning: Negative or zero points detected![/yellow]")
        console.print("[yellow]This usually means:[/yellow]")
        console.print("• [dim]The gameweek_points data is missing (NULL) in the database[/dim]")
        console.print("• [dim]You're using forecast data instead of historical data[/dim]")
        console.print("• [dim]To properly backtest, you need historical data with actual points[/dim]")
        console.print("\n[cyan]For current season predictions, use:[/cyan]")
        console.print("[green]  uv run fpl.py recommend[/green] - Get player recommendations")
        console.print("[green]  uv run fpl.py build-team[/green] - Build optimal team")
        console.print("[green]  uv run fpl.py captain[/green] - Get captain picks")
    
    merger.close()


@cli.command()
@click.argument('search_term')
@click.option('--team', '-t', help='Filter by team name')
@click.option('--position', '-p', type=click.Choice(['GK', 'DEF', 'MID', 'FWD']), help='Filter by position')
@click.option('--max-price', '-m', type=float, help='Maximum price')
def search(search_term, team, position, max_price):
    """Search for players and get their IDs
    
    Examples:
        python src/main.py search Haaland
        python src/main.py search Salah --team Liverpool
        python src/main.py search Son --position MID
    """
    console.print(Panel.fit("[bold cyan]Player Search[/bold cyan]"))
    
    # Initialize database
    merger = DataMerger("data/fpl_data.db")
    
    # Load latest data
    data = merger.load_from_database()
    if data.empty:
        console.print("[red]No data available. Please run 'fetch-data' first.[/red]")
        merger.close()
        return
    
    # Search for players - handle accented characters
    import unicodedata
    
    def normalize_text(text):
        """Remove accents and special characters for matching"""
        if pd.isna(text):
            return ""
        # Remove accents
        nfd = unicodedata.normalize('NFD', str(text))
        without_accents = ''.join(ch for ch in nfd if unicodedata.category(ch) != 'Mn')
        return without_accents.lower()
    
    search_normalized = normalize_text(search_term)
    
    # Add normalized column for searching
    data['player_name_normalized'] = data['player_name'].apply(normalize_text)
    
    # Search both original and normalized names
    matches = data[
        data['player_name'].str.lower().str.contains(search_term.lower(), na=False) |
        data['player_name_normalized'].str.contains(search_normalized, na=False)
    ]
    
    # Apply filters
    if team:
        matches = matches[matches['team_name'].str.lower().str.contains(team.lower(), na=False)]
    if position:
        matches = matches[matches['position'] == position]
    if max_price:
        matches = matches[matches['price'] <= max_price]
    
    if matches.empty:
        console.print(f"[yellow]No players found matching '{search_term}'[/yellow]")
        merger.close()
        return
    
    # Sort by total points
    matches = matches.sort_values('total_points', ascending=False)
    
    # Create table
    table = Table(title=f"Players matching '{search_term}'")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Player", style="green")
    table.add_column("Team", style="blue")
    table.add_column("Pos", justify="center")
    table.add_column("Price", style="yellow", justify="right")
    table.add_column("Points", justify="right")
    table.add_column("Form", justify="right")
    
    for _, player in matches.head(10).iterrows():
        table.add_row(
            str(player['player_id']),
            player['player_name'],
            player.get('team_name', 'Unknown'),
            player['position'],
            f"£{player['price']:.1f}",
            str(int(player.get('total_points', 0))),
            f"{player.get('form', 0):.1f}"
        )
    
    console.print(table)
    console.print(f"\n[dim]Showing top {min(10, len(matches))} of {len(matches)} matches[/dim]")
    console.print("\n[cyan]Use these IDs with:[/cyan] python src/main.py my-team -p 'ID1,ID2,ID3,...'")
    merger.close()


@cli.command()
@click.option('--player-ids', '-p', help='Comma-separated player IDs (e.g., "1,15,234")')
@click.option('--player-names', '-n', help='Comma-separated player names (will auto-find IDs)')
@click.option('--transfers', '-t', type=int, help='Number of free transfers available')
@click.option('--bank', '-b', type=float, help='Money in bank')
@click.option('--wildcard', '--wc', is_flag=True, help='Wildcard available')
@click.option('--free-hit', '--fh', is_flag=True, help='Free Hit available')
@click.option('--bench-boost', '--bb', is_flag=True, help='Bench Boost available')
@click.option('--triple-captain', '--tc', is_flag=True, help='Triple Captain available')
@click.option('--gameweek', '-gw', type=int, help='Gameweek to analyze')
@click.option('--no-interactive', is_flag=True, help='Disable interactive prompts')
def my_team(player_ids, player_names, transfers, bank, wildcard, free_hit, bench_boost, triple_captain, gameweek, no_interactive):
    """Analyze your team and get personalized recommendations
    
    Examples:
        # Using player IDs (find with 'search' command):
        python src/main.py my-team -p '1,15,234,567'
        
        # Using player names (auto-searches):
        python src/main.py my-team -n 'Haaland,Salah,Saka,Rice'
        
        # With transfers and bank:
        python src/main.py my-team -p '1,15,234' -t 2 -b 1.5
    """
    console.print(Panel.fit("[bold cyan]My Team Analyzer[/bold cyan]"))
    
    # Initialize database first
    merger = DataMerger("data/fpl_data.db")
    
    # Handle player names if provided
    if player_names and not player_ids:
        console.print("[dim]Finding player IDs from names...[/dim]")
        names_list = [n.strip() for n in player_names.split(',')]
        
        # Load data to search
        all_data = merger.load_from_database()
        if all_data.empty:
            console.print("[red]No data available. Please run 'fetch-data' first.[/red]")
            merger.close()
            return
        
        # Import for handling accented characters
        import unicodedata
        
        def normalize_text(text):
            """Remove accents and special characters for matching"""
            if pd.isna(text):
                return ""
            # Remove accents
            nfd = unicodedata.normalize('NFD', str(text))
            without_accents = ''.join(ch for ch in nfd if unicodedata.category(ch) != 'Mn')
            return without_accents.lower()
        
        # Add normalized column for searching
        all_data['player_name_normalized'] = all_data['player_name'].apply(normalize_text)
        
        found_ids = []
        not_found = []
        
        for name in names_list:
            name_lower = name.lower()
            name_normalized = normalize_text(name)
            
            # Try exact match first (both original and normalized)
            exact_match = all_data[
                (all_data['player_name'].str.lower() == name_lower) |
                (all_data['player_name_normalized'] == name_normalized)
            ]
            
            if not exact_match.empty:
                found_ids.append(str(exact_match.iloc[0]['player_id']))
                console.print(f"✓ Found: {name} → {exact_match.iloc[0]['player_name']} (ID {exact_match.iloc[0]['player_id']})")
            else:
                # Try partial match (both original and normalized)
                partial_match = all_data[
                    all_data['player_name'].str.lower().str.contains(name_lower, na=False) |
                    all_data['player_name_normalized'].str.contains(name_normalized, na=False)
                ]
                if not partial_match.empty:
                    # Take the highest scoring player
                    best_match = partial_match.nlargest(1, 'total_points').iloc[0]
                    found_ids.append(str(best_match['player_id']))
                    console.print(f"✓ Found: {name} → {best_match['player_name']} (ID {best_match['player_id']})")
                else:
                    not_found.append(name)
        
        if not_found:
            console.print(f"\n[yellow]Could not find: {', '.join(not_found)}[/yellow]")
            console.print("[dim]Use 'search' command to find exact player names[/dim]")
            
        if found_ids:
            player_ids = ','.join(found_ids)
            console.print(f"\n[green]Using player IDs: {player_ids}[/green]")
        else:
            console.print("[red]No players found![/red]")
            merger.close()
            return
    
    # Parse player IDs
    if not player_ids:
        console.print("[red]Please provide either player IDs (-p) or player names (-n)[/red]")
        console.print("\n[yellow]Examples:[/yellow]")
        console.print("  python src/main.py search Haaland  # Find player IDs")
        console.print("  python src/main.py my-team -p '1,15,234'  # Use IDs")  
        console.print("  python src/main.py my-team -n 'Haaland,Salah'  # Use names")
        merger.close()
        return
    
    try:
        team_ids = [int(pid.strip()) for pid in player_ids.split(',')]
    except ValueError:
        console.print("[red]Invalid player IDs format. Use comma-separated numbers.[/red]")
        merger.close()
        return
        
    if len(team_ids) != 15:
        console.print(f"[yellow]Warning: You entered {len(team_ids)} players (expected 15)[/yellow]")
    
    # Load data (merger already initialized above)
    data = merger.load_from_database(gameweek=gameweek)
    
    if data.empty:
        console.print("[red]No data found. Please run 'fetch-data' first.[/red]")
        merger.close()
        return
    
    # Create team object
    from src.my_team import MyTeam, TeamAnalyzer, ChipAdvisor
    
    # Interactive mode to get user inputs if not provided
    if not no_interactive:
        if transfers is None:
            console.print("\n[bold cyan]📝 Team Configuration[/bold cyan]")
            console.print("[dim]Press Enter for default values[/dim]\n")
            
            # Free transfers
            transfers_input = console.input("How many free transfers do you have? [default: 1]: ")
            transfers = int(transfers_input) if transfers_input else 1
        
        if bank is None:
            # Bank balance
            bank_input = console.input("How much money in the bank (£m)? [default: 0.0]: ")
            bank = float(bank_input) if bank_input else 0.0
            
        if not any([wildcard, free_hit, bench_boost, triple_captain]):
            # Power-ups/Chips
            console.print("\n[bold cyan]Available Power-ups/Chips:[/bold cyan]")
            wc_input = console.input("Do you have Wildcard available? (y/n) [default: y]: ")
            wildcard = wc_input.lower() != 'n' if wc_input else True
            
            fh_input = console.input("Do you have Free Hit available? (y/n) [default: y]: ")
            free_hit = fh_input.lower() != 'n' if fh_input else True
            
            bb_input = console.input("Do you have Bench Boost available? (y/n) [default: y]: ")
            bench_boost = bb_input.lower() != 'n' if bb_input else True
            
            tc_input = console.input("Do you have Triple Captain available? (y/n) [default: y]: ")
            triple_captain = tc_input.lower() != 'n' if tc_input else True
            console.print()
    
    # Set defaults if still not provided
    if transfers is None:
        transfers = 1
    if bank is None:
        bank = 0.0
    if wildcard is None:
        wildcard = True
    if free_hit is None:
        free_hit = True
    if bench_boost is None:
        bench_boost = True
    if triple_captain is None:
        triple_captain = True
    
    my_team = MyTeam(
        players=team_ids[:15],  # Ensure max 15
        captain=team_ids[0] if team_ids else 0,  # Default first player as captain
        vice_captain=team_ids[1] if len(team_ids) > 1 else 0,
        bank=bank,
        free_transfers=transfers,
        wildcard_available=wildcard,  # Fixed: Don't invert
        free_hit_available=free_hit,
        bench_boost_available=bench_boost,
        triple_captain_available=triple_captain
    )
    
    # Analyze team
    scorer = RuleBasedScorer()
    analyzer = TeamAnalyzer(scorer, data)
    analysis = analyzer.analyze_team(my_team)
    
    # Display team overview
    table = Table(title="Team Overview")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    team_value = analysis.get('total_team_value', 0)
    table.add_row("Total Team Value", f"£{team_value:.1f}m")
    table.add_row("Bank", f"£{bank:.1f}m")
    table.add_row("Free Transfers", str(transfers))
    
    # Position breakdown
    positions = analysis.get('position_breakdown', {})
    # FPL uses 'GKP' for goalkeepers, not 'GK'
    pos_str = f"GK:{positions.get('GKP',0)} DEF:{positions.get('DEF',0)} MID:{positions.get('MID',0)} FWD:{positions.get('FWD',0)}"
    table.add_row("Formation", pos_str)
    
    console.print(table)
    
    # Show weaknesses
    weaknesses = analysis.get('weaknesses', [])
    if weaknesses:
        console.print("\n[bold red]⚠️  Team Issues:[/bold red]")
        for w in weaknesses[:5]:  # Show top 5 issues
            if w['type'] == 'unavailable':
                console.print(f"• {w['player']} - [red]Unavailable/Injured[/red]")
            elif w['type'] == 'poor_form':
                console.print(f"• {w['player']} - Poor form ({w['form']:.1f})")
            elif w['type'] == 'hard_fixtures':
                console.print(f"• {w['player']} - Difficult fixtures ({w['difficulty']:.1f})")
    
    # Get transfer recommendations - use actual number of free transfers
    recommendations = analyzer.get_transfer_recommendations(my_team, num_transfers=transfers+2)
    
    if recommendations:
        console.print("\n[bold green]📈 Transfer Recommendations:[/bold green]")
        
        for i, rec in enumerate(recommendations, 1):
            if i <= transfers:
                console.print(f"\n[green]Transfer {i} (FREE):[/green]")
            else:
                console.print(f"\n[yellow]Transfer {i} (-4 pts):[/yellow]")
            
            console.print(f"OUT: {rec.player_out['name']} (£{rec.player_out['price']:.1f}m)")
            console.print(f"IN:  {rec.player_in['name']} (£{rec.player_in['price']:.1f}m)")
            console.print(f"Net cost: £{rec.net_cost:+.1f}m")
            console.print(f"Score improvement: {rec.score_improvement:+.1f}")
            console.print(f"Reason: {rec.reason}")
            
            if rec.priority == 1:
                console.print("[red]Priority: URGENT[/red]")
            elif rec.priority == 2:
                console.print("[yellow]Priority: Recommended[/yellow]")
            else:
                console.print("[dim]Priority: Optional[/dim]")
    
    # Current lineup suggestion
    # Get team players data
    team_data = data[data['player_id'].isin(my_team.players)]
    lineup = analyzer.get_lineup_suggestion(team_data)
    if lineup and lineup.get('starting_11'):
        console.print("\n[bold green]⚽ Current Best Lineup:[/bold green]")
        console.print(f"Formation: [yellow]{lineup['formation']}[/yellow]")
        
        # Starting 11
        console.print("\n[bold]Starting XI:[/bold]")
        current_pos = None
        
        for player in lineup['starting_11']:
            pos = player['position']
            if pos != current_pos:
                current_pos = pos
                pos_display = 'GK' if pos == 'GKP' else pos
                console.print(f"\n[cyan]{pos_display}:[/cyan]")
            
            # Highlight injured/doubtful players
            player_row = team_data[team_data['player_name'] == player['name']]
            if not player_row.empty:
                p = player_row.iloc[0]
                status = ""
                if p.get('is_available', 1) == 0:
                    status = " [red]⚠️ INJURED[/red]"
                elif p.get('chance_of_playing_next_round', 100) < 75:
                    chance = p.get('chance_of_playing_next_round', 0)
                    status = f" [yellow]({chance}% chance)[/yellow]"
                
                console.print(f"  • {player['name']} ({player['score']:.1f}){status}")
            else:
                console.print(f"  • {player['name']} ({player['score']:.1f})")
        
        # Bench
        if lineup.get('bench'):
            console.print("\n[bold]Bench (in priority order):[/bold]")
            for i, player in enumerate(lineup['bench'], 1):
                pos_display = 'GK' if player['position'] == 'GKP' else player['position']
                console.print(f"  {i}. {player['name']} ({pos_display}, {player['score']:.1f})")
    
    # Post-transfer lineup (if transfers recommended)
    if recommendations and transfers > 0:
        # Take only free transfers
        free_transfers = recommendations[:transfers]
        post_transfer_lineup = analyzer.get_post_transfer_lineup(team_data, free_transfers)
        
        if post_transfer_lineup and post_transfer_lineup.get('starting_11'):
            console.print("\n[bold cyan]🔄 Post-Transfer Lineup (after free transfers):[/bold cyan]")
            console.print(f"Formation: [yellow]{post_transfer_lineup['formation']}[/yellow]")
            
            # Show changes
            console.print("\n[dim]Applied transfers:[/dim]")
            for i, transfer in enumerate(free_transfers, 1):
                console.print(f"  {i}. {transfer.player_out['name']} → {transfer.player_in['name']}")
            
            console.print("\n[bold]New Starting XI:[/bold]")
            current_pos = None
            
            for player in post_transfer_lineup['starting_11']:
                pos = player['position']
                if pos != current_pos:
                    current_pos = pos
                    pos_display = 'GK' if pos == 'GKP' else pos
                    console.print(f"\n[cyan]{pos_display}:[/cyan]")
                
                # Highlight new players
                is_new = any(t.player_in['name'] == player['name'] for t in free_transfers)
                if is_new:
                    console.print(f"  • {player['name']} ({player['score']:.1f}) [green]✨ NEW[/green]")
                else:
                    console.print(f"  • {player['name']} ({player['score']:.1f})")
            
            # New bench
            if post_transfer_lineup.get('bench'):
                console.print("\n[bold]New Bench:[/bold]")
                for i, player in enumerate(post_transfer_lineup['bench'], 1):
                    pos_display = 'GK' if player['position'] == 'GKP' else player['position']
                    is_new = any(t.player_in['name'] == player['name'] for t in free_transfers)
                    new_tag = " [green]✨ NEW[/green]" if is_new else ""
                    console.print(f"  {i}. {player['name']} ({pos_display}, {player['score']:.1f}){new_tag}")
    
    # Captain recommendation
    captain_analysis = analysis.get('captain_analysis', {})
    if captain_analysis and captain_analysis.get('top_3_options'):
        console.print("\n[bold cyan]👑 Captain Options:[/bold cyan]")
        for i, opt in enumerate(captain_analysis['top_3_options'], 1):
            console.print(f"{i}. {opt['player']} (score: {opt['score']:.1f})")
    
    # Comprehensive chip advice
    chip_advisor = ChipAdvisor(data)
    all_chip_advice = chip_advisor.get_all_chip_advice(my_team, analysis, team_data)
    
    chips_to_use = []
    chips_to_consider = []
    
    for chip_name, advice in all_chip_advice.items():
        if advice.get('use') == True:
            chips_to_use.append((chip_name, advice))
        elif advice.get('use') == 'consider':
            chips_to_consider.append((chip_name, advice))
    
    # ALWAYS show chip recommendations section
    console.print("\n[bold magenta]🎯 Power-up/Chip Recommendations:[/bold magenta]")
    
    # Show available chips
    available_chips = []
    if my_team.wildcard_available: available_chips.append("Wildcard")
    if my_team.free_hit_available: available_chips.append("Free Hit")
    if my_team.bench_boost_available: available_chips.append("Bench Boost")
    if my_team.triple_captain_available: available_chips.append("Triple Captain")
    
    if available_chips:
        console.print(f"[dim]Available: {', '.join(available_chips)}[/dim]")
    else:
        console.print("[dim]No chips available[/dim]")
    
    if chips_to_use or chips_to_consider:
        
        # Urgent chip recommendations
        if chips_to_use:
            for chip_name, advice in chips_to_use:
                chip_emoji = {
                    'wildcard': '🃏',
                    'free_hit': '💨',
                    'bench_boost': '📈',
                    'triple_captain': '👑'
                }.get(chip_name, '🎯')
                
                chip_display = chip_name.replace('_', ' ').title()
                console.print(f"\n[bold red]{chip_emoji} USE {chip_display.upper()} THIS WEEK[/bold red]")
                for reason in advice.get('reasons', []):
                    console.print(f"  • {reason}")
        
        # Consider using chips
        if chips_to_consider:
            for chip_name, advice in chips_to_consider:
                chip_emoji = {
                    'wildcard': '🃏',
                    'free_hit': '💨', 
                    'bench_boost': '📈',
                    'triple_captain': '👑'
                }.get(chip_name, '🎯')
                
                chip_display = chip_name.replace('_', ' ').title()
                console.print(f"\n[yellow]{chip_emoji} Consider {chip_display}[/yellow]")
                for reason in advice.get('reasons', []):
                    console.print(f"  • {reason}")
    else:
        # No chips to use or consider - give hold recommendation
        console.print("\n[green]✅ Recommendation: HOLD all chips for now[/green]")
        console.print("[dim]Save your power-ups for:[/dim]")
        console.print("  • Double gameweeks (players play twice)")
        console.print("  • Blank gameweeks (use Free Hit)")
        console.print("  • Easy fixture runs for premium captains (Triple Captain)")
        console.print("  • When bench has favorable fixtures (Bench Boost)")
    
    merger.close()


@cli.command()
def status():
    """Check system status and API limits"""
    console.print(Panel.fit("[bold cyan]System Status[/bold cyan]"))
    
    # Check database
    merger = DataMerger()
    data = merger.load_from_database()
    
    table = Table(title="Status Report")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    # Database status
    if not data.empty:
        latest_gw = data['gameweek'].max()
        player_count = data['player_id'].nunique()
        table.add_row(
            "Database",
            "✓ Connected",
            f"{len(data)} records, {player_count} players, GW{latest_gw}"
        )
    else:
        table.add_row("Database", "✗ Empty", "Run 'fetch-data' to populate")
    
    # Check .env file
    env_path = Path('.env')
    if env_path.exists():
        table.add_row(".env file", "✓ Found", "API keys configured")
    else:
        table.add_row(".env file", "✗ Missing", "Copy .env.example to .env")
    
    # Check cache directory
    cache_path = Path('cache')
    if cache_path.exists():
        cache_files = list(cache_path.rglob('*.json'))
        table.add_row("Cache", "✓ Active", f"{len(cache_files)} cached files")
    else:
        table.add_row("Cache", "✗ Missing", "Will be created on first run")
    
    console.print(table)
    
    # Show quick tips
    tips = Panel.fit(
        "[bold]Quick Start:[/bold]\n\n"
        "1. Run [cyan]uv run fpl.py fetch-data[/cyan] to get latest data\n"
        "2. Run [cyan]uv run fpl.py recommend[/cyan] for player suggestions\n"
        "3. Run [cyan]uv run fpl.py build-team[/cyan] to build optimal team\n"
        "4. Run [cyan]uv run fpl.py backtest[/cyan] to test on historical data",
        title="Getting Started"
    )
    console.print(tips)
    
    merger.close()


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    cli()