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
    
    unified_data = merger.create_unified_dataset(
        players_df,
        player_odds if not player_odds.empty else None,
        None,  # Elo data would go here
        gameweek=gameweek,
        season="2024-25"
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
    
    # Load data
    merger = DataMerger()
    data = merger.load_from_database(gameweek=gameweek)
    
    if data.empty:
        console.print("[red]No data found. Please run 'fetch-data' first.[/red]")
        return
        
    # Filter for likely starters
    candidates = data[
        (data['chance_of_playing_next_round'] >= 90) &
        (data['minutes'] > 60)  # Regular starters
    ]
    
    # Calculate captain score (emphasizing goal probability)
    candidates['captain_score'] = (
        candidates['prob_goal'] * 5.0 +  # Heavy weight on goals
        candidates['prob_assist'] * 2.0 +
        candidates['form'] * 0.3 +
        candidates['expected_points'] * 0.1
    )
    
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
        table.add_row(
            str(i),
            player['player_name'],
            player['team_name'],
            player.get('opponent_team', 'TBD'),
            f"{player['prob_goal']*100:.1f}%",
            f"{player.get('expected_points', 0):.1f}",
            f"{player['form']:.1f}",
            f"{player['captain_score']:.1f}"
        )
    
    console.print(table)
    
    # Add captaincy tips
    console.print("\n[bold]Captain Tips:[/bold]")
    top_pick = top_captains.iloc[0]
    console.print(f"• Top pick: [cyan]{top_pick['player_name']}[/cyan] with {top_pick['prob_goal']*100:.1f}% goal probability")
    
    if top_pick['prob_goal'] > 0.5:
        console.print(f"• [green]Strong captaincy choice - over 50% chance to score![/green]")
    elif top_pick['prob_goal'] > 0.35:
        console.print(f"• [yellow]Good captaincy choice - decent goal probability[/yellow]")
    else:
        console.print(f"• [red]Consider alternative options - low goal probability this week[/red]")
        
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
    table.add_row("Average GW Points", f"{sum(result.gameweek_points)/len(result.gameweek_points):.1f}")
    table.add_row("Final Team Value", f"£{result.final_team_value:.1f}")
    table.add_row("Transfers Made", str(result.transfers_made))
    table.add_row("Captain Points", str(result.captain_points))
    table.add_row("Estimated Rank", f"Top {100-result.rank_percentile:.0f}%")
    
    console.print(table)
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