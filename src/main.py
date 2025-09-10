#!/usr/bin/env python
"""
Main FPL Recommender CLI
Run with: uv run python -m src.main
"""
import asyncio
import functools
import json
import logging
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.data.data_merger import DataMerger

# Import our modules
from src.data.fpl_api import FPLAPICollector
from src.data.odds_api import PlayerPropsCollector
from src.models.backtester import Backtester
from src.models.rule_based_scorer import RuleBasedScorer, ScoringWeights

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

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def _load_scoring_weights() -> ScoringWeights:
    """Load scoring weights from config file"""
    weights_file = "config/scoring_weights.json"
    try:
        with open(weights_file) as f:
            weights_data = json.load(f)
        return ScoringWeights.from_dict(weights_data)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scoring weights config not found at {weights_file}. Create the config file with position-specific weights."
        )


@cli.command(name="fetch-data")
@click.option("--gameweek", "-gw", type=int, help="Gameweek number")
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
                for event in bootstrap["events"]:
                    if event["is_current"]:
                        gameweek = event["id"]
                        break

            console.print(f"[green]✓[/green] Fetching data for Gameweek {gameweek}")

            # Get gameweek data
            gw_data = await fpl_collector.get_gameweek_data(gameweek)
            players_df = gw_data["players"]
            fixtures_df = gw_data["fixtures"]

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
                    num_matches = player_odds["match_id"].nunique()
                    num_players = len(player_odds)
                    bookmakers = player_odds["bookmaker_title"].unique()

                    console.print("[green]✓[/green] Retrieved REAL player odds:")
                    console.print(f"  • {num_players} player odds across {num_matches} matches")
                    console.print(f"  • Bookmakers: {', '.join(bookmakers[:3])}")

                    # Match player names to FPL data
                    player_odds = odds_collector.match_players_to_fpl(player_odds, players_df)
                    console.print("[green]✓[/green] Matched odds to FPL player data")

                    # Show how many players we matched
                    if "has_real_odds" in player_odds.columns:
                        matched_count = player_odds[player_odds["has_real_odds"]][
                            "player_id"
                        ].nunique()
                        console.print(f"  • {matched_count} FPL players have real bookmaker odds")
                else:
                    console.print(
                        "[red]✗[/red] No player odds available - cannot provide accurate recommendations"
                    )
                    console.print(
                        "[yellow]Note: This system requires real bookmaker odds for accuracy[/yellow]"
                    )
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
                fixtures_df_temp["event"] = gw
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
        season="2025-26",
    )

    # Save to database
    merger.save_to_database(unified_data)
    merger.close()

    console.print(f"[green]✓[/green] Saved {len(unified_data)} player records to database")


@cli.command()
@click.option(
    "--position", "-p", type=click.Choice(["GK", "DEF", "MID", "FWD"]), help="Filter by position"
)
@click.option("--max-price", "-mp", type=float, help="Maximum price filter")
@click.option("--top", "-t", type=int, default=10, help="Number of players to show")
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
    weights = _load_scoring_weights()
    scorer = RuleBasedScorer(weights)
    scored = scorer.score_all_players(data)

    if scored.empty:
        console.print("[red]No players with real odds found![/red]")
        console.print("Please run 'fetch-data' to get latest odds data.")
        merger.close()
        return

    # Filter
    if position:
        scored = scored[scored["position"] == position]
    if max_price:
        scored = scored[scored["price"] <= max_price]

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
            player["player_name"],
            player.get("team_name", "N/A"),
            player["position"],
            f"£{player['price']:.1f}",
            f"{player['model_score']:.1f}",
            f"{player.get('form', 0):.1f}",
            str(int(player.get("total_points", 0))),
        )

    console.print(table)
    merger.close()


@cli.command(name="quick-build")
@click.option("--budget", "-b", type=float, default=100.0, help="Team budget")
def quick_build_team(budget):
    """Quickly build optimal team within budget"""
    console.print(Panel.fit("[bold cyan]FPL Team Builder[/bold cyan]"))

    # Load data
    merger = DataMerger()
    data = merger.get_latest_data(top_n=500)

    if data.empty:
        console.print("[red]No data found. Run 'fetch-data' first.[/red]")
        return

    # Score players
    weights = _load_scoring_weights()
    scorer = RuleBasedScorer(weights)
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
    if "model_score" not in players_df.columns:
        return {"players": [], "total_cost": 0, "remaining_budget": budget}

    selected = []
    spent = 0.0

    # Position requirements
    positions_needed = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    team_counts = {}

    # Calculate value score (points per million) for better selection
    players_df = players_df.copy()
    safe_price = players_df["price"].clip(lower=0.1)
    players_df["value_score"] = players_df["model_score"] / safe_price

    # Strategy: Mix high-scoring premiums with value picks

    # First pass: Get 1-2 premium players per position (high score, regardless of price)
    premiums_per_pos = {"GK": 1, "DEF": 2, "MID": 2, "FWD": 1}

    for position, premium_count in premiums_per_pos.items():
        pos_players = players_df[players_df["position"] == position].sort_values(
            "model_score", ascending=False
        )

        added = 0
        for _, player in pos_players.iterrows():
            if added >= premium_count:
                break

            team = player["team_name"]
            if team in team_counts and team_counts[team] >= 3:
                continue

            # For premiums, allow up to 15% of budget per player
            if player["price"] > budget * 0.15:
                continue

            if spent + player["price"] > budget * 0.85:  # Reserve 15% for bench
                continue

            selected.append(
                {
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": player["position"],
                    "team": team,
                    "price": player["price"],
                    "score": player["model_score"],
                    "total_points": player["total_points"] if "total_points" in player else 0,
                }
            )

            spent += player["price"]
            team_counts[team] = team_counts[team] + 1 if team in team_counts else 1
            added += 1

    # Second pass: Fill remaining slots with best value players
    for position, count in positions_needed.items():
        current_count = sum(1 for p in selected if p["position"] == position)

        if current_count < count:
            # Get value players (good score per million)
            pos_players = players_df[
                (players_df["position"] == position)
                & (~players_df["player_id"].isin([p["player_id"] for p in selected]))
            ].sort_values("value_score", ascending=False)

            for _, player in pos_players.iterrows():
                if current_count >= count:
                    break

                team = player["team_name"]
                if team in team_counts and team_counts[team] >= 3:
                    continue

                if spent + player["price"] > budget:
                    continue

                selected.append(
                    {
                        "player_id": player["player_id"],
                        "player_name": player["player_name"],
                        "position": player["position"],
                        "team": team,
                        "price": player["price"],
                        "score": player["model_score"],
                        "total_points": player["total_points"] if "total_points" in player else 0,
                    }
                )

                spent += player["price"]
                team_counts[team] = team_counts[team] + 1 if team in team_counts else 1
                current_count += 1

    # Final pass: If still missing players, get cheapest valid options
    for position, count in positions_needed.items():
        current_count = sum(1 for p in selected if p["position"] == position)

        if current_count < count:
            remaining_players = players_df[
                (players_df["position"] == position)
                & (~players_df["player_id"].isin([p["player_id"] for p in selected]))
            ].sort_values("price")

            for _, player in remaining_players.iterrows():
                if current_count >= count:
                    break

                team = player["team_name"]
                if team in team_counts and team_counts[team] >= 3:
                    continue

                if spent + player["price"] > budget:
                    # If we can't afford anyone, we have a problem
                    logger.warning(f"Cannot complete team within budget for {position}")
                    break

                selected.append(
                    {
                        "player_id": player["player_id"],
                        "player_name": player["player_name"],
                        "position": player["position"],
                        "team": team,
                        "price": player["price"],
                        "score": player["model_score"],
                        "total_points": player["total_points"] if "total_points" in player else 0,
                    }
                )

                spent += player["price"]
                team_counts[team] = team_counts[team] + 1 if team in team_counts else 1
                current_count += 1

    return {"players": selected, "total_cost": spent, "remaining_budget": budget - spent}


def display_team(team, console):
    """Display team in a nice format"""
    # Group by position
    positions = {"GK": [], "DEF": [], "MID": [], "FWD": []}

    for player in team["players"]:
        positions[player["position"]].append(player)

    # Create table
    table = Table(title="Your Optimal FPL Team")
    table.add_column("Position", style="cyan")
    table.add_column("Player", style="magenta")
    table.add_column("Team", style="yellow")
    table.add_column("Price", style="red", justify="right")
    table.add_column("Score", style="bold green", justify="right")

    for pos in ["GK", "DEF", "MID", "FWD"]:
        for player in positions[pos]:
            table.add_row(
                pos,
                player["player_name"],
                player["team"],
                f"£{player['price']:.1f}",
                f"{player['score']:.1f}",
            )

    console.print(table)
    console.print(f"\n[bold]Total Cost:[/bold] £{team['total_cost']:.1f}")
    console.print(f"[bold]Remaining:[/bold] £{team['remaining_budget']:.1f}")


@cli.command()
@click.option(
    "--min-odds", type=float, default=3.0, help="Minimum goal odds (lower = more likely to score)"
)
@click.option("--max-ownership", type=float, default=5.0, help="Maximum ownership %")
@click.option("--top", "-n", type=int, default=10, help="Number of differentials to show")
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
        (data["prob_goal"] > (1 / min_odds))  # Good goal probability
        & (data["selected_by_percent"] < max_ownership)  # Low ownership
        & (data["chance_of_playing_next_round"] >= 75)  # Likely to play
    ]

    if differentials_df.empty:
        console.print(
            f"[yellow]No differentials found with odds < {min_odds} and ownership < {max_ownership}%[/yellow]"
        )
        return

    # Score and sort
    weights = _load_scoring_weights()
    scorer = RuleBasedScorer(weights)
    differentials_df["model_score"] = differentials_df.apply(scorer.score_player, axis=1)
    differentials_df = differentials_df.sort_values("model_score", ascending=False).head(top)

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
            player["player_name"],
            player["team_name"],
            player["position"],
            f"£{player['price']:.1f}",
            f"{player['selected_by_percent']:.1f}%",
            f"{player['prob_goal']*100:.1f}%",
            f"{player['model_score']:.1f}",
        )

    console.print(table)
    merger.close()


@cli.command()
@click.option("--gameweek", "-gw", type=int, help="Gameweek to analyze")
def captain(gameweek):
    """Recommend best captain choices based on goal probability"""
    console.print(Panel.fit("[bold cyan]FPL Captain Selector[/bold cyan]"))

    # Load data for specific gameweek or latest
    merger = DataMerger()
    data = merger.load_from_database(gameweek=gameweek)

    # If no specific gameweek, get the latest gameweek for each player
    if gameweek is None and not data.empty:
        latest_gw = data.groupby("player_id")["gameweek"].max().reset_index()
        latest_gw.columns = ["player_id", "max_gameweek"]
        data = data.merge(latest_gw, on="player_id")
        data = data[data["gameweek"] == data["max_gameweek"]]
        data = data.drop("max_gameweek", axis=1)

    if data.empty:
        console.print("[red]No data found. Please run 'fetch-data' first.[/red]")
        return

    # Filter for likely starters
    # Use is_available column or default to all players if not available
    if "is_available" in data.columns:
        candidates = data[
            (data["is_available"] == 1) & (data["minutes"] > 60)  # Regular starters
        ].copy()  # Use copy to avoid SettingWithCopyWarning
    else:
        # Fallback: just use regular starters based on minutes
        candidates = data[data["minutes"] > 60].copy()

    # Calculate captain score (emphasizing goal probability)
    # Check which columns are available and build score accordingly
    captain_score = pd.Series(0.0, index=candidates.index)

    if "prob_goal" in candidates.columns:
        # Require real odds data for captain selection
        if "prob_goal" not in candidates.columns:
            raise ValueError("Cannot select captain without goal probability data from bookmakers")
        captain_score = captain_score + candidates["prob_goal"] * 5.0  # Heavy weight on goals

    if "prob_assist" in candidates.columns:
        # Convert to float to avoid downcasting warning
        if "prob_assist" in candidates.columns:
            prob_assist_values = candidates["prob_assist"].astype(float)
        else:
            prob_assist_values = pd.Series(0, index=candidates.index)
        captain_score = captain_score + prob_assist_values * 2.0

    if "form" in candidates.columns:
        if "form" not in candidates.columns:
            raise ValueError("Missing form data for captain selection")
        captain_score = captain_score + candidates["form"] * 0.3

    if "expected_goals" in candidates.columns:
        if "expected_goals" in candidates.columns:
            captain_score = captain_score + candidates["expected_goals"] * 2.0

    if "expected_assists" in candidates.columns:
        if "expected_assists" in candidates.columns:
            captain_score = captain_score + candidates["expected_assists"] * 1.5

    # Add total points as a factor if no odds data
    if "prob_goal" not in candidates.columns and "total_points" in candidates.columns:
        if "total_points" not in candidates.columns:
            raise ValueError("Missing total_points data for captain selection")
        captain_score = captain_score + candidates["total_points"] * 0.1

    candidates["captain_score"] = captain_score

    # Get top 10 captain choices
    top_captains = candidates.nlargest(10, "captain_score")

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
        goal_prob = player["prob_goal"] if "prob_goal" in player else 0
        form_val = player["form"] if "form" in player else 0

        table.add_row(
            str(i),
            player["player_name"],
            player["team_name"],
            player["opponent_team"] if "opponent_team" in player else "TBD",
            f"{goal_prob*100:.1f}%" if goal_prob > 0 else "N/A",
            f"{player['expected_goals']:.1f}" if "expected_goals" in player else "N/A",
            f"{form_val:.1f}",
            f"{player['captain_score']:.1f}",
        )

    console.print(table)

    # Add captaincy tips
    console.print("\n[bold]Captain Tips:[/bold]")
    if not top_captains.empty:
        top_pick = top_captains.iloc[0]
        goal_prob = top_pick["prob_goal"] if "prob_goal" in top_pick else 0

        if goal_prob > 0:
            console.print(
                f"• Top pick: [cyan]{top_pick['player_name']}[/cyan] with {goal_prob*100:.1f}% goal probability"
            )

            if goal_prob > 0.5:
                console.print(
                    "• [green]Strong captaincy choice - over 50% chance to score![/green]"
                )
            elif goal_prob > 0.35:
                console.print("• [yellow]Good captaincy choice - decent goal probability[/yellow]")
            else:
                console.print(
                    "• [red]Consider alternative options - low goal probability this week[/red]"
                )
        else:
            console.print(
                f"• Top pick: [cyan]{top_pick['player_name']}[/cyan] (based on form and expected stats)"
            )
            console.print(
                "• [yellow]No betting odds available - using historical performance[/yellow]"
            )

    merger.close()


@cli.command()
@click.option("--season", "-s", default="2024-25", help="Season to backtest")
@click.option("--start-gw", type=int, default=1, help="Starting gameweek")
@click.option("--end-gw", type=int, default=38, help="Ending gameweek")
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
    weights = _load_scoring_weights()
    scorer = RuleBasedScorer(weights)
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
        console.print(
            "• [dim]To properly backtest, you need historical data with actual points[/dim]"
        )
        console.print("\n[cyan]For current season predictions, use:[/cyan]")
        console.print("[green]  uv run fpl.py recommend[/green] - Get player recommendations")
        console.print("[green]  uv run fpl.py build-team[/green] - Build optimal team")
        console.print("[green]  uv run fpl.py captain[/green] - Get captain picks")

    merger.close()


@cli.command()
@click.argument("search_term")
@click.option("--team", "-t", help="Filter by team name")
@click.option(
    "--position", "-p", type=click.Choice(["GK", "DEF", "MID", "FWD"]), help="Filter by position"
)
@click.option("--max-price", "-m", type=float, help="Maximum price")
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
        nfd = unicodedata.normalize("NFD", str(text))
        without_accents = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        return without_accents.lower()

    search_normalized = normalize_text(search_term)

    # Add normalized column for searching
    data["player_name_normalized"] = data["player_name"].apply(normalize_text)

    # Search both original and normalized names
    matches = data[
        data["player_name"].str.lower().str.contains(search_term.lower(), na=False)
        | data["player_name_normalized"].str.contains(search_normalized, na=False)
    ]

    # Apply filters
    if team:
        matches = matches[matches["team_name"].str.lower().str.contains(team.lower(), na=False)]
    if position:
        matches = matches[matches["position"] == position]
    if max_price:
        matches = matches[matches["price"] <= max_price]

    if matches.empty:
        console.print(f"[yellow]No players found matching '{search_term}'[/yellow]")
        merger.close()
        return

    # Sort by total points
    matches = matches.sort_values("total_points", ascending=False)

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
            str(player["player_id"]),
            player["player_name"],
            player["team_name"] if "team_name" in player else "Unknown",
            player["position"],
            f"£{player['price']:.1f}",
            str(int(player["total_points"])) if "total_points" in player else "N/A",
            f"{player['form']:.1f}" if "form" in player else "N/A",
        )

    console.print(table)
    console.print(f"\n[dim]Showing top {min(10, len(matches))} of {len(matches)} matches[/dim]")
    console.print(
        "\n[cyan]Use these IDs with:[/cyan] python src/main.py my-team -p 'ID1,ID2,ID3,...'"
    )
    merger.close()


@cli.command()
@click.option("--player-ids", "-p", help='Comma-separated player IDs (e.g., "1,15,234")')
@click.option("--player-names", "-n", help="Comma-separated player names (will auto-find IDs)")
@click.option("--transfers", "-t", type=int, help="Number of free transfers available")
@click.option("--bank", "-b", type=float, help="Money in bank")
@click.option("--wildcard", "--wc", is_flag=True, help="Wildcard available")
@click.option("--free-hit", "--fh", is_flag=True, help="Free Hit available")
@click.option("--bench-boost", "--bb", is_flag=True, help="Bench Boost available")
@click.option("--triple-captain", "--tc", is_flag=True, help="Triple Captain available")
@click.option("--gameweek", "-gw", type=int, help="Gameweek to analyze")
@click.option("--no-interactive", is_flag=True, help="Disable interactive prompts")
def my_team(
    player_ids,
    player_names,
    transfers,
    bank,
    wildcard,
    free_hit,
    bench_boost,
    triple_captain,
    gameweek,
    no_interactive,
):
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
        names_list = [n.strip() for n in player_names.split(",")]

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
            nfd = unicodedata.normalize("NFD", str(text))
            without_accents = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
            return without_accents.lower()

        # Add normalized column for searching
        all_data["player_name_normalized"] = all_data["player_name"].apply(normalize_text)

        found_ids = []
        not_found = []

        for name in names_list:
            name_lower = name.lower()
            name_normalized = normalize_text(name)

            # Try exact match first (both original and normalized)
            exact_match = all_data[
                (all_data["player_name"].str.lower() == name_lower)
                | (all_data["player_name_normalized"] == name_normalized)
            ]

            if not exact_match.empty:
                found_ids.append(str(exact_match.iloc[0]["player_id"]))
                console.print(
                    f"✓ Found: {name} → {exact_match.iloc[0]['player_name']} (ID {exact_match.iloc[0]['player_id']})"
                )
            else:
                # Try partial match (both original and normalized)
                partial_match = all_data[
                    all_data["player_name"].str.lower().str.contains(name_lower, na=False)
                    | all_data["player_name_normalized"].str.contains(name_normalized, na=False)
                ]
                if not partial_match.empty:
                    # Take the highest scoring player
                    best_match = partial_match.nlargest(1, "total_points").iloc[0]
                    found_ids.append(str(best_match["player_id"]))
                    console.print(
                        f"✓ Found: {name} → {best_match['player_name']} (ID {best_match['player_id']})"
                    )
                else:
                    not_found.append(name)

        if not_found:
            console.print(f"\n[yellow]Could not find: {', '.join(not_found)}[/yellow]")
            console.print("[dim]Use 'search' command to find exact player names[/dim]")

        if found_ids:
            player_ids = ",".join(found_ids)
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
        team_ids = [int(pid.strip()) for pid in player_ids.split(",")]
    except ValueError:
        console.print("[red]Invalid player IDs format. Use comma-separated numbers.[/red]")
        merger.close()
        return

    if len(team_ids) != 15:
        console.print(
            f"[yellow]Warning: You entered {len(team_ids)} players (expected 15)[/yellow]"
        )

    # Load data (merger already initialized above)
    data = merger.load_from_database(gameweek=gameweek)

    if data.empty:
        console.print("[red]No data found. Please run 'fetch-data' first.[/red]")
        merger.close()
        return

    # Create team object
    from src.my_team import ChipAdvisor, MyTeam, TeamAnalyzer

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
            wildcard = wc_input.lower() != "n" if wc_input else True

            fh_input = console.input("Do you have Free Hit available? (y/n) [default: y]: ")
            free_hit = fh_input.lower() != "n" if fh_input else True

            bb_input = console.input("Do you have Bench Boost available? (y/n) [default: y]: ")
            bench_boost = bb_input.lower() != "n" if bb_input else True

            tc_input = console.input("Do you have Triple Captain available? (y/n) [default: y]: ")
            triple_captain = tc_input.lower() != "n" if tc_input else True
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
        triple_captain_available=triple_captain,
    )

    # Analyze team
    weights = _load_scoring_weights()
    scorer = RuleBasedScorer(weights)
    analyzer = TeamAnalyzer(scorer, data)
    analysis = analyzer.analyze_team(my_team)

    # Display team overview
    table = Table(title="Team Overview")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    team_value = analysis.get("total_team_value", 0)
    table.add_row("Total Team Value", f"£{team_value:.1f}m")
    table.add_row("Bank", f"£{bank:.1f}m")
    table.add_row("Free Transfers", str(transfers))

    # Position breakdown
    positions = analysis.get("position_breakdown", {})
    pos_str = f"GK:{positions.get('GK',0)} DEF:{positions.get('DEF',0)} MID:{positions.get('MID',0)} FWD:{positions.get('FWD',0)}"
    table.add_row("Formation", pos_str)

    console.print(table)

    # Show weaknesses
    weaknesses = analysis.get("weaknesses", [])
    if weaknesses:
        console.print("\n[bold red]⚠️  Team Issues:[/bold red]")
        for w in weaknesses[:5]:  # Show top 5 issues
            if w["type"] == "unavailable":
                console.print(f"• {w['player']} - [red]Unavailable/Injured[/red]")
            elif w["type"] == "poor_form":
                console.print(f"• {w['player']} - Poor form ({w['form']:.1f})")
            elif w["type"] == "hard_fixtures":
                console.print(f"• {w['player']} - Difficult fixtures ({w['difficulty']:.1f})")

    # Get team players data first (needed for conflict detection)
    team_data = data[data["player_id"].isin(my_team.players)]

    # Get transfer recommendations - use actual number of free transfers
    recommendations = analyzer.get_transfer_recommendations(my_team, num_transfers=transfers + 2)

    # Check for fixture conflicts
    if recommendations:
        conflicts = analyzer.detect_fixture_conflicts(
            team_data, recommendations[: my_team.free_transfers]
        )

        if conflicts:
            console.print("\n[bold red]⚠️ Fixture Conflict Warning:[/bold red]")
            for conflict in conflicts:
                console.print(f"[yellow]{conflict['message']}[/yellow]")
                console.print(
                    f"  • {conflict['team1']}: {', '.join(conflict['team1_players'][:2])}"
                )
                console.print(
                    f"  • {conflict['team2']}: {', '.join(conflict['team2_players'][:2])}"
                )
            console.print(
                "[dim]Tip: Having attackers vs defenders from opposing teams means one's success hurts the other's clean sheet points![/dim]"
            )

    if recommendations:
        console.print("\n[bold green]📈 Transfer Recommendations:[/bold green]")

        # Track running budget for affordability check
        running_bank = bank

        for i, rec in enumerate(recommendations, 1):
            if i <= transfers:
                console.print(f"\n[green]Transfer {i} (FREE):[/green]")
            else:
                console.print(f"\n[yellow]Transfer {i} (-4 pts):[/yellow]")

            # Check affordability
            if rec.net_cost > running_bank:
                console.print("[bold red]⚠️ WARNING: This transfer is NOT affordable![/bold red]")
                console.print(
                    f"[red]Required: £{rec.net_cost:.1f}m | Available: £{running_bank:.1f}m[/red]"
                )
                console.print("[red]Skipping this transfer...[/red]\n")
                continue

            console.print(f"OUT: {rec.player_out['name']} (£{rec.player_out['price']:.1f}m)")
            console.print(f"IN:  {rec.player_in['name']} (£{rec.player_in['price']:.1f}m)")
            console.print(f"Net cost: £{rec.net_cost:+.1f}m")
            console.print(f"Score improvement: {rec.score_improvement:+.1f}")
            console.print(f"Reason: {rec.reason}")

            # Update running bank
            running_bank -= rec.net_cost

            if rec.priority == 1:
                console.print("[red]Priority: URGENT[/red]")
            elif rec.priority == 2:
                console.print("[yellow]Priority: Recommended[/yellow]")
            else:
                console.print("[dim]Priority: Optional[/dim]")

            # Show strategic hit evaluation for transfers beyond free
            if i > transfers and hasattr(rec, "hit_evaluation") and rec.hit_evaluation:
                eval_result = rec.hit_evaluation

                # Color-code based on recommendation
                if eval_result["recommendation"] == "AVOID":
                    color = "red"
                    emoji = "🚫"
                elif eval_result["recommendation"] == "CONSIDER":
                    color = "yellow"
                    emoji = "🤔"
                else:  # TAKE
                    color = "green"
                    emoji = "✅"

                console.print(
                    f"\n[{color}]Hit Evaluation: {emoji} {eval_result['recommendation']} (Confidence: {eval_result['confidence']})[/{color}]"
                )
                console.print(
                    f"[dim]Expected gain: {eval_result.get('adjusted_gain', eval_result.get('point_gain', 0)):.1f} pts[/dim]"
                )
                console.print(f"[dim]{eval_result['strategic_advice']}[/dim]")

                # Show key modifiers if present
                if eval_result.get("modifiers"):
                    mods = eval_result["modifiers"]
                    if mods.get("is_fire"):
                        console.print("[red]  • 🔥 Replacing injured/suspended player[/red]")
                    if mods.get("is_captain_candidate"):
                        console.print("[cyan]  • 👑 Captain candidate[/cyan]")
                    if mods.get("is_dgw_player"):
                        console.print("[green]  • 🎯 Double Gameweek player[/green]")
                    if mods.get("has_fixture_swing"):
                        console.print("[green]  • 📈 Favorable fixture swing[/green]")

    # Current lineup suggestion
    lineup = analyzer.get_lineup_suggestion(team_data)
    if lineup and lineup.get("starting_11"):
        console.print("\n[bold green]⚽ Current Best Lineup:[/bold green]")
        console.print(f"Formation: [yellow]{lineup['formation']}[/yellow]")

        # Starting 11
        console.print("\n[bold]Starting XI:[/bold]")
        current_pos = None

        for player in lineup["starting_11"]:
            pos = player["position"]
            if pos != current_pos:
                current_pos = pos
                pos_display = pos
                console.print(f"\n[cyan]{pos_display}:[/cyan]")

            # Highlight injured/doubtful players
            player_row = team_data[team_data["player_name"] == player["name"]]
            if not player_row.empty:
                p = player_row.iloc[0]
                status = ""
                if p.get("is_available", 1) == 0:
                    status = " [red]⚠️ INJURED[/red]"
                elif p.get("chance_of_playing_next_round", 100) < 75:
                    chance = p.get("chance_of_playing_next_round", 0)
                    status = f" [yellow]({chance}% chance)[/yellow]"

                console.print(f"  • {player['name']} ({player['score']:.1f}){status}")
            else:
                console.print(f"  • {player['name']} ({player['score']:.1f})")

        # Bench
        if lineup.get("bench"):
            console.print("\n[bold]Bench (in priority order):[/bold]")
            for i, player in enumerate(lineup["bench"], 1):
                pos_display = player["position"]
                console.print(f"  {i}. {player['name']} ({pos_display}, {player['score']:.1f})")

    # Post-transfer lineup (if transfers recommended)
    if recommendations and transfers > 0:
        # Take only free transfers
        free_transfers = recommendations[:transfers]
        post_transfer_lineup = analyzer.get_post_transfer_lineup(team_data, free_transfers)

        if post_transfer_lineup and post_transfer_lineup.get("starting_11"):
            console.print(
                "\n[bold cyan]🔄 Post-Transfer Lineup (after free transfers):[/bold cyan]"
            )
            console.print(f"Formation: [yellow]{post_transfer_lineup['formation']}[/yellow]")

            # Show changes
            console.print("\n[dim]Applied transfers:[/dim]")
            for i, transfer in enumerate(free_transfers, 1):
                console.print(
                    f"  {i}. {transfer.player_out['name']} → {transfer.player_in['name']}"
                )

            console.print("\n[bold]New Starting XI:[/bold]")
            current_pos = None

            for player in post_transfer_lineup["starting_11"]:
                pos = player["position"]
                if pos != current_pos:
                    current_pos = pos
                    pos_display = pos
                    console.print(f"\n[cyan]{pos_display}:[/cyan]")

                # Highlight new players
                is_new = any(t.player_in["name"] == player["name"] for t in free_transfers)
                if is_new:
                    console.print(
                        f"  • {player['name']} ({player['score']:.1f}) [green]✨ NEW[/green]"
                    )
                else:
                    console.print(f"  • {player['name']} ({player['score']:.1f})")

            # New bench
            if post_transfer_lineup.get("bench"):
                console.print("\n[bold]New Bench:[/bold]")
                for i, player in enumerate(post_transfer_lineup["bench"], 1):
                    pos_display = player["position"]
                    is_new = any(t.player_in["name"] == player["name"] for t in free_transfers)
                    new_tag = " [green]✨ NEW[/green]" if is_new else ""
                    console.print(
                        f"  {i}. {player['name']} ({pos_display}, {player['score']:.1f}){new_tag}"
                    )

    # Captain recommendation (current team)
    captain_analysis = analysis.get("captain_analysis", {})
    if captain_analysis and captain_analysis.get("top_3_options"):
        console.print("\n[bold cyan]👑 Captain Options:[/bold cyan]")
        for i, opt in enumerate(captain_analysis["top_3_options"], 1):
            console.print(f"{i}. {opt['player']} (score: {opt['score']:.1f})")

    # Post-transfer captain analysis
    if recommendations and transfers > 0:
        free_transfers = recommendations[:transfers]
        post_captain_analysis = analyzer.get_post_transfer_captain_analysis(
            team_data, free_transfers, my_team.captain
        )

        if post_captain_analysis and post_captain_analysis.get("top_3_options"):
            console.print("\n[bold cyan]🔄 Post-Transfer Captain Options:[/bold cyan]")
            for i, opt in enumerate(post_captain_analysis["top_3_options"], 1):
                # Check if this player is a new transfer
                is_new = any(t.player_in["name"] == opt["player"] for t in free_transfers)
                new_indicator = " [green]✨ NEW[/green]" if is_new else ""
                console.print(f"{i}. {opt['player']} (score: {opt['score']:.1f}){new_indicator}")

    # Comprehensive chip advice
    chip_advisor = ChipAdvisor(data)
    all_chip_advice = chip_advisor.get_all_chip_advice(my_team, analysis, team_data)

    chips_to_use = []
    chips_to_consider = []

    for chip_name, advice in all_chip_advice.items():
        # Skip non-chip entries like 'strategic_summary'
        if chip_name == "strategic_summary" or not isinstance(advice, dict):
            continue
        if advice.get("use") is True:
            chips_to_use.append((chip_name, advice))
        elif advice.get("use") == "consider":
            chips_to_consider.append((chip_name, advice))

    # ALWAYS show chip recommendations section
    console.print("\n[bold magenta]🎯 Power-up/Chip Recommendations:[/bold magenta]")

    # Show available chips
    available_chips = []
    if my_team.wildcard_available:
        available_chips.append("Wildcard")
    if my_team.free_hit_available:
        available_chips.append("Free Hit")
    if my_team.bench_boost_available:
        available_chips.append("Bench Boost")
    if my_team.triple_captain_available:
        available_chips.append("Triple Captain")

    if available_chips:
        console.print(f"[dim]Available: {', '.join(available_chips)}[/dim]")
    else:
        console.print("[dim]No chips available[/dim]")

    if chips_to_use or chips_to_consider:

        # Urgent chip recommendations
        if chips_to_use:
            for chip_name, advice in chips_to_use:
                chip_emoji = {
                    "wildcard": "🃏",
                    "free_hit": "💨",
                    "bench_boost": "📈",
                    "triple_captain": "👑",
                }.get(chip_name, "🎯")

                chip_display = chip_name.replace("_", " ").title()
                console.print(
                    f"\n[bold red]{chip_emoji} USE {chip_display.upper()} THIS WEEK[/bold red]"
                )

                # Special display for triple captain - show the player
                if chip_name == "triple_captain" and advice.get("player"):
                    console.print(f"  [bold cyan]→ Captain: {advice['player']}[/bold cyan]")
                    if advice.get("expected_points"):
                        console.print(
                            f"  [cyan]→ Expected points: {advice['expected_points']:.1f}[/cyan]"
                        )

                # Special display for bench boost - show expected bench points
                elif chip_name == "bench_boost" and advice.get("expected_bench_points"):
                    console.print(
                        f"  [cyan]→ Expected bench points: {advice['expected_bench_points']:.1f}[/cyan]"
                    )

                # Display reasons
                for reason in advice.get("reasons", []):
                    console.print(f"  • {reason}")

                # Display strategic advice if available
                if advice.get("strategic_advice"):
                    console.print(f"  [bold yellow]{advice['strategic_advice']}[/bold yellow]")

        # Consider using chips
        if chips_to_consider:
            for chip_name, advice in chips_to_consider:
                chip_emoji = {
                    "wildcard": "🃏",
                    "free_hit": "💨",
                    "bench_boost": "📈",
                    "triple_captain": "👑",
                }.get(chip_name, "🎯")

                chip_display = chip_name.replace("_", " ").title()
                console.print(f"\n[yellow]{chip_emoji} Consider {chip_display}[/yellow]")

                # Special display for triple captain - show the player
                if chip_name == "triple_captain" and advice.get("player"):
                    console.print(f"  [cyan]→ Captain: {advice['player']}[/cyan]")
                    if advice.get("expected_points"):
                        console.print(
                            f"  [dim]→ Expected points: {advice['expected_points']:.1f}[/dim]"
                        )

                # Special display for bench boost - show expected bench points
                elif chip_name == "bench_boost" and advice.get("expected_bench_points"):
                    console.print(
                        f"  [dim]→ Expected bench points: {advice['expected_bench_points']:.1f}[/dim]"
                    )

                # Display reasons
                for reason in advice.get("reasons", []):
                    console.print(f"  • {reason}")

                # Display strategic advice if available
                if advice.get("strategic_advice"):
                    console.print(f"  [bold yellow]{advice['strategic_advice']}[/bold yellow]")
    else:
        # Check for strategic summary
        strategic_summary = all_chip_advice.get("strategic_summary")
        if strategic_summary:
            console.print(f"\n[bold cyan]{strategic_summary}[/bold cyan]")
        else:
            # No chips to use or consider - give hold recommendation
            console.print("\n[green]✅ Recommendation: HOLD all chips for now[/green]")

        console.print("\n[dim]💡 Strategic chip timing tips:[/dim]")
        console.print(
            "  • [bold]Triple Captain[/bold]: SAVE for Double Gameweek with premium captain"
        )
        console.print("  • [bold]Bench Boost[/bold]: SAVE for Double Gameweek after Wildcard")
        console.print("  • [bold]Free Hit[/bold]: SAVE for Blank Gameweek (<8 players)")
        console.print("  • [bold]Wildcard[/bold]: Use when 4+ issues or to prepare for DGW/BGW")

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
        latest_gw = data["gameweek"].max()
        player_count = data["player_id"].nunique()
        table.add_row(
            "Database", "✓ Connected", f"{len(data)} records, {player_count} players, GW{latest_gw}"
        )
    else:
        table.add_row("Database", "✗ Empty", "Run 'fetch-data' to populate")

    # Check .env file
    env_path = Path(".env")
    if env_path.exists():
        table.add_row(".env file", "✓ Found", "API keys configured")
    else:
        table.add_row(".env file", "✗ Missing", "Copy .env.example to .env")

    # Check cache directory
    cache_path = Path("cache")
    if cache_path.exists():
        cache_files = list(cache_path.rglob("*.json"))
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
        title="Getting Started",
    )
    console.print(tips)

    merger.close()


@cli.command()
@click.option("--load-ids", "-l", help="Load existing team from comma-separated player IDs")
@click.option("--budget", "-b", type=float, default=100.0, help="Team budget (default: 100.0)")
def build_team(load_ids, budget):
    """Interactive team builder with user-friendly interface

    Build your FPL team using an intuitive selection interface:
    • Visual team formation display
    • Search and filter players
    • Track budget in real-time
    • Validate team requirements

    Examples:
        # Build new team from scratch
        python fpl.py build-team

        # Edit existing team
        python fpl.py build-team -l '1,15,234,567,890...'

        # Build with custom budget
        python fpl.py build-team -b 98.5
    """
    from src.team_builder import TeamBuilder, load_existing_team

    console.print(
        Panel.fit(
            "[bold cyan]⚽ Interactive Team Builder[/bold cyan]\n"
            "[dim]Build your FPL team with an intuitive interface[/dim]",
            border_style="cyan",
        )
    )

    # Initialize database
    merger = DataMerger("data/fpl_data.db")

    # Load player data
    data = merger.load_from_database()
    if data.empty:
        console.print("[red]No data available. Please run 'fetch-data' first.[/red]")
        merger.close()
        return

    # Initialize team builder
    builder = TeamBuilder(data)
    builder.team.budget = budget

    # Load existing team if provided
    if load_ids:
        try:
            team_ids = [int(pid.strip()) for pid in load_ids.split(",")]
            builder.team = load_existing_team(data, team_ids)
            console.print(f"[green]Loaded {len(team_ids)} players from existing team[/green]")
        except ValueError:
            console.print("[red]Invalid player IDs format[/red]")
            merger.close()
            return

    # Build team interactively
    team = builder.build_team_interactive()

    if team:
        # Display final team
        console.print("\n" + "=" * 80)
        console.print("[bold green]✓ Team Built Successfully![/bold green]")
        console.print("=" * 80)

        # Show player IDs for use with other commands
        player_ids_str = ",".join(str(pid) for pid in team.player_ids)
        console.print(f"\n[bold]Player IDs:[/bold] {player_ids_str}")
        console.print(f"[bold]Captain:[/bold] Player ID {team.captain}")
        console.print(f"[bold]Vice-Captain:[/bold] Player ID {team.vice_captain}")

        # Offer to analyze team
        console.print("\n[bold]Next steps:[/bold]")
        console.print(
            f"• Analyze this team: [cyan]python fpl.py my-team -p '{player_ids_str}'[/cyan]"
        )
        console.print(
            f"• Get transfer recommendations: [cyan]python fpl.py recommend -p '{player_ids_str}'[/cyan]"
        )

        # Save team to file for easy reuse
        if Confirm.ask("\nSave team to file for easy access?"):
            team_data = {
                "player_ids": team.player_ids,
                "captain": team.captain,
                "vice_captain": team.vice_captain,
                "formation": {
                    "goalkeepers": [p["player_id"] for p in team.goalkeepers],
                    "defenders": [p["player_id"] for p in team.defenders],
                    "midfielders": [p["player_id"] for p in team.midfielders],
                    "forwards": [p["player_id"] for p in team.forwards],
                },
            }

            import json
            from pathlib import Path

            # Create teams directory if it doesn't exist
            teams_dir = Path("teams")
            teams_dir.mkdir(exist_ok=True)

            # Save team
            team_name = Prompt.ask("Team name", default="my_team")
            filename = teams_dir / f"{team_name}.json"

            with open(filename, "w") as f:
                json.dump(team_data, f, indent=2)

            console.print(f"[green]✓ Team saved to {filename}[/green]")
            console.print(f"Load it later with: [cyan]python fpl.py load-team {team_name}[/cyan]")
    else:
        console.print("[yellow]Team building cancelled[/yellow]")

    merger.close()


@cli.command()
@click.argument("team_name")
def load_team(team_name):
    """Load a saved team and display options

    Args:
        team_name: Name of saved team file (without .json extension)

    Example:
        python fpl.py load-team my_team
    """
    from pathlib import Path

    teams_dir = Path("teams")
    filename = teams_dir / f"{team_name}.json"

    if not filename.exists():
        console.print(f"[red]Team file not found: {filename}[/red]")

        # Show available teams
        if teams_dir.exists():
            team_files = list(teams_dir.glob("*.json"))
            if team_files:
                console.print("\n[bold]Available teams:[/bold]")
                for tf in team_files:
                    console.print(f"  • {tf.stem}")
        return

    # Load team data
    with open(filename) as f:
        team_data = json.load(f)

    player_ids_str = ",".join(str(pid) for pid in team_data["player_ids"])

    console.print(
        Panel.fit(f"[bold green]Loaded Team: {team_name}[/bold green]", border_style="green")
    )
    console.print(f"\n[bold]Player IDs:[/bold] {player_ids_str}")
    console.print(f"[bold]Captain:[/bold] Player ID {team_data.get('captain', 'Not set')}")
    console.print(
        f"[bold]Vice-Captain:[/bold] Player ID {team_data.get('vice_captain', 'Not set')}"
    )

    # Show formation
    if "formation" in team_data:
        console.print("\n[bold]Formation:[/bold]")
        console.print(f"  GK: {len(team_data['formation']['GK'])} players")
        console.print(f"  DEF: {len(team_data['formation']['DEF'])} players")
        console.print(f"  MID: {len(team_data['formation']['MID'])} players")
        console.print(f"  FWD: {len(team_data['formation']['FWD'])} players")

    console.print("\n[bold]Commands to use with this team:[/bold]")
    console.print(f"• Edit team: [cyan]python fpl.py build-team -l '{player_ids_str}'[/cyan]")
    console.print(f"• Analyze team: [cyan]python fpl.py my-team -p '{player_ids_str}'[/cyan]")
    console.print(
        f"• Get recommendations: [cyan]python fpl.py recommend -p '{player_ids_str}'[/cyan]"
    )


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    cli()
