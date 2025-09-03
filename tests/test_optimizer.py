#!/usr/bin/env python
"""
Test script for the FPL Optimizer
Demonstrates optimization with different strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.optimizer import FPLOptimizer, OptimizationResult
from src.data.data_merger import DataMerger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import time

console = Console()


def display_optimization_result(result: OptimizationResult, strategy: str):
    """Display optimization results in a nice format"""
    
    # Header panel
    console.print(Panel.fit(
        f"[bold cyan]Optimization Result - {strategy.upper()} Strategy[/bold cyan]\n"
        f"Expected Points: [bold yellow]{result.expected_points:.1f}[/bold yellow] | "
        f"Formation: [bold green]{result.formation}[/bold green] | "
        f"Budget: [bold red]£{result.total_cost:.1f}m[/bold red] / £100.0m",
        border_style="cyan"
    ))
    
    # Starting XI
    console.print("\n[bold]Starting XI:[/bold]")
    starting_xi_table = Table(box=box.ROUNDED)
    starting_xi_table.add_column("Pos", style="cyan", width=5)
    starting_xi_table.add_column("Player", style="white", width=22)
    starting_xi_table.add_column("Team", style="yellow", width=12)
    starting_xi_table.add_column("Price", style="green", width=8)
    starting_xi_table.add_column("Exp Pts", style="magenta", width=8)
    starting_xi_table.add_column("Role", style="bold cyan", width=8)
    
    starting_players = [p for p in result.players if not p.get('is_bench', False)]
    starting_players.sort(key=lambda x: ['GK', 'DEF', 'MID', 'FWD'].index(x['position']))
    
    for player in starting_players:
        role = ""
        if player.get('is_captain'):
            role = "(C)"
        elif player.get('is_vice_captain'):
            role = "(VC)"
        
        starting_xi_table.add_row(
            player['position'],
            player['player_name'][:22],
            player.get('team_name', 'N/A')[:12],
            f"£{player['price']:.1f}m",
            f"{player.get('expected_points', 0):.1f}",
            role
        )
    
    console.print(starting_xi_table)
    
    # Bench
    console.print("\n[bold]Bench:[/bold]")
    bench_table = Table(box=box.SIMPLE)
    bench_table.add_column("Pos", style="dim", width=5)
    bench_table.add_column("Player", style="dim white", width=22)
    bench_table.add_column("Team", style="dim yellow", width=12)
    bench_table.add_column("Price", style="dim green", width=8)
    
    bench_players = [p for p in result.players if p.get('is_bench', False)]
    for player in bench_players:
        bench_table.add_row(
            player['position'],
            player['player_name'][:22],
            player.get('team_name', 'N/A')[:12],
            f"£{player['price']:.1f}m"
        )
    
    console.print(bench_table)
    
    # Team composition
    console.print("\n[bold]Team Composition:[/bold]")
    team_counts = {}
    for player in result.players:
        team = player.get('team_name', 'Unknown')
        team_counts[team] = team_counts.get(team, 0) + 1
    
    comp_str = " | ".join([f"{team}: {count}" for team, count in 
                           sorted(team_counts.items(), key=lambda x: x[1], reverse=True)[:5]])
    console.print(f"  {comp_str}")
    
    # Stats
    console.print(f"\n[bold]Optimization Stats:[/bold]")
    console.print(f"  • Solver Status: {result.solver_status}")
    console.print(f"  • Optimization Time: {result.optimization_time:.2f}s")
    console.print(f"  • Remaining Budget: £{result.remaining_budget:.2f}m")


def test_strategies():
    """Test different optimization strategies"""
    
    console.print(Panel.fit(
        "[bold cyan]FPL Advanced Optimizer Testing[/bold cyan]\n"
        "[dim]Testing multiple optimization strategies with ML predictions and MILP[/dim]",
        border_style="cyan"
    ))
    
    # Load data
    console.print("\n[bold]Loading player data...[/bold]")
    merger = DataMerger()
    players_df = merger.get_latest_data(top_n=500)
    
    if players_df.empty:
        console.print("[red]No data found. Please run 'fetch-data' first.[/red]")
        merger.close()
        return
    
    console.print(f"[green]✓[/green] Loaded {len(players_df)} players")
    
    # Filter to players with real odds for accuracy
    if 'has_real_odds' in players_df.columns:
        players_df = players_df[players_df['has_real_odds'] == True].copy()
        console.print(f"[green]✓[/green] Filtered to {len(players_df)} players with real odds")
    
    # Initialize optimizer
    console.print("\n[bold]Initializing optimizer...[/bold]")
    optimizer = FPLOptimizer()
    
    # Test different strategies
    strategies = ['balanced', 'short_term', 'long_term', 'differential']
    results = {}
    
    for strategy in strategies:
        console.print(f"\n[bold yellow]Testing {strategy.upper()} strategy...[/bold yellow]")
        
        try:
            start_time = time.time()
            result = optimizer.optimize_team(
                players_df.copy(),
                strategy=strategy,
                budget=100.0
            )
            result.optimization_time = time.time() - start_time
            
            results[strategy] = result
            console.print(f"[green]✓[/green] {strategy} optimization complete in {result.optimization_time:.2f}s")
            
        except Exception as e:
            console.print(f"[red]✗[/red] {strategy} optimization failed: {e}")
            continue
    
    # Display all results
    console.print("\n" + "="*80)
    console.print("[bold cyan]OPTIMIZATION RESULTS[/bold cyan]")
    console.print("="*80)
    
    for strategy, result in results.items():
        console.print(f"\n[bold]━━━ {strategy.upper()} STRATEGY ━━━[/bold]")
        display_optimization_result(result, strategy)
    
    # Compare strategies
    if len(results) > 1:
        console.print("\n" + "="*80)
        console.print("[bold cyan]STRATEGY COMPARISON[/bold cyan]")
        console.print("="*80)
        
        comparison_table = Table(box=box.ROUNDED)
        comparison_table.add_column("Strategy", style="cyan")
        comparison_table.add_column("Expected Points", style="yellow", justify="right")
        comparison_table.add_column("Budget Used", style="green", justify="right")
        comparison_table.add_column("Formation", style="magenta")
        comparison_table.add_column("Time (s)", style="dim", justify="right")
        
        for strategy, result in results.items():
            comparison_table.add_row(
                strategy.upper(),
                f"{result.expected_points:.1f}",
                f"£{result.total_cost:.1f}m",
                result.formation,
                f"{result.optimization_time:.2f}"
            )
        
        console.print(comparison_table)
        
        # Find best strategy
        best_strategy = max(results.items(), key=lambda x: x[1].expected_points)
        console.print(f"\n[bold green]✨ Best Strategy: {best_strategy[0].upper()} "
                     f"with {best_strategy[1].expected_points:.1f} expected points[/bold green]")
    
    merger.close()


def test_transfer_optimization():
    """Test transfer optimization for existing team"""
    
    console.print(Panel.fit(
        "[bold cyan]Transfer Optimization Testing[/bold cyan]",
        border_style="cyan"
    ))
    
    # Load data
    merger = DataMerger()
    players_df = merger.get_latest_data(top_n=500)
    
    if players_df.empty:
        console.print("[red]No data found.[/red]")
        merger.close()
        return
    
    # Create a sample current team (would normally load from user data)
    # For demo, use a balanced team first
    optimizer = FPLOptimizer()
    initial_team = optimizer.optimize_team(players_df.copy(), strategy='balanced', budget=95.0)
    
    # Simulate current team DataFrame
    current_team_df = pd.DataFrame(initial_team.players[:15])  # Just the 15 players
    
    console.print("[bold]Current Team:[/bold]")
    for player in current_team_df.itertuples():
        console.print(f"  {player.position:3} {player.player_name:20} £{player.price:.1f}m")
    
    # Test transfer recommendations
    console.print("\n[bold]Optimizing Transfers...[/bold]")
    
    transfer_result = optimizer.optimize_transfers(
        current_team_df,
        players_df,
        free_transfers=2,
        max_hits=1,
        wildcard=False
    )
    
    console.print("\n[bold]Recommended Transfers:[/bold]")
    if transfer_result['transfers']:
        transfer_table = Table(box=box.ROUNDED)
        transfer_table.add_column("OUT", style="red")
        transfer_table.add_column("IN", style="green")
        transfer_table.add_column("Gain", style="yellow", justify="right")
        transfer_table.add_column("Cost", style="cyan", justify="right")
        
        for transfer in transfer_result['transfers']:
            transfer_table.add_row(
                transfer['out_name'][:20],
                transfer['in_name'][:20],
                f"+{transfer['gain']:.1f}",
                f"£{transfer['cost']:.1f}m"
            )
        
        console.print(transfer_table)
        
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Expected Point Gain: [green]+{transfer_result['expected_gain']:.1f}[/green]")
        console.print(f"  Hits Taken: [yellow]{transfer_result['hits_taken']}[/yellow]")
        console.print(f"  Hit Cost: [red]-{transfer_result['cost']} pts[/red]")
        console.print(f"  Net Gain: [bold cyan]+{transfer_result['expected_gain'] - transfer_result['cost']:.1f} pts[/bold cyan]")
    else:
        console.print("[yellow]No beneficial transfers found[/yellow]")
    
    merger.close()


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FPL Optimizer')
    parser.add_argument('--mode', choices=['strategies', 'transfers', 'both'], 
                       default='strategies',
                       help='Test mode: strategies, transfers, or both')
    
    args = parser.parse_args()
    
    if args.mode in ['strategies', 'both']:
        test_strategies()
    
    if args.mode in ['transfers', 'both']:
        console.print("\n" + "="*80 + "\n")
        test_transfer_optimization()


if __name__ == "__main__":
    main()