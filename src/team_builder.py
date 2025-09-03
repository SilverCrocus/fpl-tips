"""
Interactive Team Builder Module
Provides a user-friendly interface for building and managing FPL teams
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.layout import Layout
from rich.columns import Columns
from rich.text import Text
from rich.live import Live
from rich import box
import logging

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class TeamSelection:
    """Represents a team being built"""
    goalkeepers: List[Dict] = field(default_factory=list)
    defenders: List[Dict] = field(default_factory=list)
    midfielders: List[Dict] = field(default_factory=list)
    forwards: List[Dict] = field(default_factory=list)
    captain: Optional[int] = None
    vice_captain: Optional[int] = None
    budget: float = 100.0
    
    @property
    def all_players(self) -> List[Dict]:
        """Get all selected players"""
        return self.goalkeepers + self.defenders + self.midfielders + self.forwards
    
    @property
    def player_ids(self) -> List[int]:
        """Get all player IDs"""
        return [p['player_id'] for p in self.all_players]
    
    @property
    def total_cost(self) -> float:
        """Calculate total team cost"""
        return sum(p['price'] for p in self.all_players)
    
    @property
    def remaining_budget(self) -> float:
        """Calculate remaining budget"""
        return self.budget - self.total_cost
    
    @property
    def team_size(self) -> int:
        """Get current team size"""
        return len(self.all_players)
    
    def is_valid(self) -> Tuple[bool, List[str]]:
        """Check if team is valid according to FPL rules
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check team size
        if self.team_size != 15:
            issues.append(f"Team must have 15 players (currently {self.team_size})")
        
        # Check formation rules
        if len(self.goalkeepers) != 2:
            issues.append(f"Must have exactly 2 goalkeepers (currently {len(self.goalkeepers)})")
        if len(self.defenders) != 5:
            issues.append(f"Must have exactly 5 defenders (currently {len(self.defenders)})")
        if len(self.midfielders) != 5:
            issues.append(f"Must have exactly 5 midfielders (currently {len(self.midfielders)})")
        if len(self.forwards) != 3:
            issues.append(f"Must have exactly 3 forwards (currently {len(self.forwards)})")
        
        # Check budget
        if self.total_cost > self.budget:
            issues.append(f"Team exceeds budget: £{self.total_cost:.1f}m > £{self.budget:.1f}m")
        
        # Check max players per team (max 3)
        team_counts = {}
        for player in self.all_players:
            team = player['team_name']
            team_counts[team] = team_counts.get(team, 0) + 1
            if team_counts[team] > 3:
                issues.append(f"Too many players from {team} (max 3)")
        
        return len(issues) == 0, issues


class TeamBuilder:
    """Interactive team builder interface"""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize team builder with player data
        
        Args:
            data: DataFrame with player data
        """
        self.data = data
        self.team = TeamSelection()
        
        # Position requirements
        self.position_limits = {
            'GK': (2, 2),  # min, max
            'DEF': (5, 5),
            'MID': (5, 5),
            'FWD': (3, 3)
        }
        
    def build_team_interactive(self) -> Optional[TeamSelection]:
        """Main interactive team building interface
        
        Returns:
            Completed team selection or None if cancelled
        """
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]⚽ FPL Team Builder[/bold cyan]\n"
            "[dim]Build your Fantasy Premier League team with an intuitive interface[/dim]",
            border_style="cyan"
        ))
        
        # Show initial instructions
        console.print("\n[bold]Team Requirements:[/bold]")
        console.print("• 2 Goalkeepers")
        console.print("• 5 Defenders") 
        console.print("• 5 Midfielders")
        console.print("• 3 Forwards")
        console.print("• Maximum 3 players from same team")
        console.print("• Total budget: £100.0m\n")
        
        # Main building loop
        while True:
            self._display_current_team()
            
            # Check if team is complete
            if self.team.team_size == 15:
                is_valid, issues = self.team.is_valid()
                if is_valid:
                    console.print("\n[green]✓ Team is complete and valid![/green]")
                    if Confirm.ask("Finalize this team?"):
                        return self._finalize_team()
                else:
                    console.print("\n[yellow]Team issues:[/yellow]")
                    for issue in issues:
                        console.print(f"  • {issue}")
            
            # Show menu options
            console.print("\n[bold]Actions:[/bold]")
            options = []
            
            # Add position options based on what's needed
            for pos, (min_req, max_req) in self.position_limits.items():
                current = self._get_position_count(pos)
                if current < max_req:
                    options.append(f"Add {self._get_position_name(pos)}")
            
            if self.team.team_size > 0:
                options.append("Remove player")
                options.append("View team stats")
            
            options.extend(["Reset team", "Cancel"])
            
            # Display options
            for i, option in enumerate(options, 1):
                console.print(f"  [{i}] {option}")
            
            # Get user choice
            choice = IntPrompt.ask("\nSelect action", choices=[str(i) for i in range(1, len(options) + 1)])
            action = options[choice - 1]
            
            # Handle action
            if action.startswith("Add "):
                position = action.replace("Add ", "").replace("Goalkeeper", "GK").replace("Defender", "DEF").replace("Midfielder", "MID").replace("Forward", "FWD")
                self._add_player(position)
            elif action == "Remove player":
                self._remove_player()
            elif action == "View team stats":
                self._show_team_stats()
            elif action == "Reset team":
                if Confirm.ask("Reset entire team?"):
                    self.team = TeamSelection()
                    console.print("[yellow]Team reset[/yellow]")
            elif action == "Cancel":
                if Confirm.ask("Cancel team building?"):
                    return None
    
    def _display_current_team(self):
        """Display current team in a visual formation layout"""
        console.print("\n" + "=" * 80)
        console.print(f"[bold]Current Team[/bold] ({self.team.team_size}/15 players)")
        console.print(f"Budget: £{self.team.remaining_budget:.1f}m remaining (£{self.team.total_cost:.1f}m spent)")
        console.print("=" * 80)
        
        # Create visual formation
        formation_lines = []
        
        # Goalkeepers
        gk_line = self._format_position_line(self.team.goalkeepers, 2, "GK")
        formation_lines.append(gk_line)
        
        # Defenders
        def_line = self._format_position_line(self.team.defenders, 5, "DEF")
        formation_lines.append(def_line)
        
        # Midfielders
        mid_line = self._format_position_line(self.team.midfielders, 5, "MID")
        formation_lines.append(mid_line)
        
        # Forwards
        fwd_line = self._format_position_line(self.team.forwards, 3, "FWD")
        formation_lines.append(fwd_line)
        
        # Display formation
        for line in formation_lines:
            console.print(line, justify="center")
            console.print()
    
    def _format_position_line(self, players: List[Dict], max_count: int, position: str) -> str:
        """Format a line of players for visual display"""
        line_parts = []
        
        for i in range(max_count):
            if i < len(players):
                player = players[i]
                # Format: Name (Team) £Xm
                player_str = f"{player['player_name'][:12]:^12}\n{player['team_name'][:3]} £{player['price']:.1f}m"
                if player.get('player_id') == self.team.captain:
                    player_str = f"[bold cyan]©{player_str}[/bold cyan]"
                elif player.get('player_id') == self.team.vice_captain:
                    player_str = f"[cyan]ⓥ{player_str}[/cyan]"
                line_parts.append(f"[{player_str}]")
            else:
                line_parts.append(f"[dim][  Empty {position}  ][/dim]")
        
        return "    ".join(line_parts)
    
    def _add_player(self, position: str):
        """Add a player to the team"""
        console.clear()
        console.print(Panel.fit(f"[bold]Add {self._get_position_name(position)}[/bold]", border_style="cyan"))
        
        # Filter players by position
        position_players = self.data[self.data['position'] == position].copy()
        
        # Remove already selected players
        position_players = position_players[~position_players['player_id'].isin(self.team.player_ids)]
        
        # Apply budget constraint
        max_price = self.team.remaining_budget
        position_players = position_players[position_players['price'] <= max_price]
        
        if position_players.empty:
            console.print("[red]No affordable players available for this position![/red]")
            Prompt.ask("Press Enter to continue")
            return
        
        # Offer filter options
        console.print("\n[bold]Filter Options:[/bold]")
        console.print("[1] Show all players")
        console.print("[2] Filter by team")
        console.print("[3] Filter by price range")
        console.print("[4] Search by name")
        console.print("[5] Show top performers")
        
        filter_choice = IntPrompt.ask("Select filter", choices=["1", "2", "3", "4", "5"], default=1)
        
        if filter_choice == 2:
            # Filter by team
            teams = sorted(position_players['team_name'].unique())
            console.print("\n[bold]Teams:[/bold]")
            for i, team in enumerate(teams, 1):
                console.print(f"[{i}] {team}")
            team_choice = IntPrompt.ask("Select team", choices=[str(i) for i in range(1, len(teams) + 1)])
            selected_team = teams[team_choice - 1]
            position_players = position_players[position_players['team_name'] == selected_team]
            
        elif filter_choice == 3:
            # Filter by price range
            min_price = float(Prompt.ask("Minimum price (£m)", default="4.0"))
            max_price_input = float(Prompt.ask(f"Maximum price (£m)", default=str(max_price)))
            position_players = position_players[
                (position_players['price'] >= min_price) & 
                (position_players['price'] <= min_price_input)
            ]
            
        elif filter_choice == 4:
            # Search by name
            search_term = Prompt.ask("Enter player name (partial match)").lower()
            position_players = position_players[
                position_players['player_name'].str.lower().str.contains(search_term, na=False)
            ]
            
        elif filter_choice == 5:
            # Show top performers
            position_players = position_players.nlargest(20, 'total_points')
        
        if position_players.empty:
            console.print("[red]No players match your criteria![/red]")
            Prompt.ask("Press Enter to continue")
            return
        
        # Sort by total points by default
        position_players = position_players.sort_values('total_points', ascending=False)
        
        # Display players in pages
        page_size = 15
        page = 0
        total_pages = (len(position_players) - 1) // page_size + 1
        
        while True:
            console.clear()
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, len(position_players))
            page_players = position_players.iloc[start_idx:end_idx]
            
            # Create player table
            table = Table(title=f"Select {self._get_position_name(position)} (Page {page + 1}/{total_pages})", 
                         box=box.ROUNDED)
            table.add_column("#", style="cyan", width=3)
            table.add_column("Player", style="white", width=20)
            table.add_column("Team", style="dim", width=4)
            table.add_column("Price", style="green", width=6)
            table.add_column("Points", style="yellow", width=6)
            table.add_column("Form", style="magenta", width=5)
            table.add_column("Selected", style="blue", width=8)
            table.add_column("Fixtures", style="dim", width=20)
            
            for i, (_, player) in enumerate(page_players.iterrows(), 1):
                # Get fixture difficulty info
                fixtures = []
                for col in ['fixture_diff_next2', 'fixture_diff_next5']:
                    if col in player and not pd.isna(player[col]):
                        fixtures.append(f"{col[-1]}GW: {player[col]:.1f}")
                fixture_str = " | ".join(fixtures) if fixtures else "N/A"
                
                table.add_row(
                    str(i),
                    player['player_name'][:20],
                    player['team_name'][:3],
                    f"£{player['price']:.1f}",
                    str(int(player.get('total_points', 0))),
                    f"{player.get('form', 0):.1f}",
                    f"{player.get('selected_by_percent', 0):.1f}%",
                    fixture_str
                )
            
            console.print(table)
            console.print(f"\nRemaining budget: £{self.team.remaining_budget:.1f}m")
            
            # Navigation options
            console.print("\n[bold]Options:[/bold]")
            console.print("• Enter number (1-15) to select player")
            if page > 0:
                console.print("• [P] Previous page")
            if page < total_pages - 1:
                console.print("• [N] Next page")
            console.print("• [B] Back to team")
            
            choice = Prompt.ask("Your choice").upper()
            
            if choice == 'B':
                return
            elif choice == 'P' and page > 0:
                page -= 1
            elif choice == 'N' and page < total_pages - 1:
                page += 1
            elif choice.isdigit():
                player_idx = int(choice) - 1
                if 0 <= player_idx < len(page_players):
                    selected_player = page_players.iloc[player_idx]
                    self._add_player_to_team(selected_player, position)
                    console.print(f"[green]✓ Added {selected_player['player_name']} to team![/green]")
                    Prompt.ask("Press Enter to continue")
                    return
    
    def _add_player_to_team(self, player: pd.Series, position: str):
        """Add selected player to appropriate position list"""
        player_dict = {
            'player_id': int(player['player_id']),
            'player_name': player['player_name'],
            'team_name': player['team_name'],
            'price': player['price'],
            'total_points': player.get('total_points', 0),
            'form': player.get('form', 0),
            'position': position
        }
        
        if position == 'GK':
            self.team.goalkeepers.append(player_dict)
        elif position == 'DEF':
            self.team.defenders.append(player_dict)
        elif position == 'MID':
            self.team.midfielders.append(player_dict)
        elif position == 'FWD':
            self.team.forwards.append(player_dict)
    
    def _remove_player(self):
        """Remove a player from the team"""
        if not self.team.all_players:
            console.print("[red]No players to remove![/red]")
            return
        
        console.clear()
        console.print(Panel.fit("[bold]Remove Player[/bold]", border_style="red"))
        
        # Display all players with numbers
        table = Table(title="Select player to remove", box=box.ROUNDED)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Position", style="dim", width=8)
        table.add_column("Player", style="white", width=20)
        table.add_column("Team", style="dim", width=4)
        table.add_column("Price", style="green", width=6)
        
        all_players = []
        for pos_name, players in [
            ("GK", self.team.goalkeepers),
            ("DEF", self.team.defenders),
            ("MID", self.team.midfielders),
            ("FWD", self.team.forwards)
        ]:
            for player in players:
                all_players.append((pos_name, player))
        
        for i, (pos, player) in enumerate(all_players, 1):
            table.add_row(
                str(i),
                pos,
                player['player_name'],
                player['team_name'][:3],
                f"£{player['price']:.1f}"
            )
        
        console.print(table)
        
        choice = IntPrompt.ask("Select player to remove (0 to cancel)", 
                               choices=[str(i) for i in range(len(all_players) + 1)])
        
        if choice == 0:
            return
        
        pos_name, player = all_players[choice - 1]
        
        # Remove from appropriate list
        if pos_name == "GK":
            self.team.goalkeepers.remove(player)
        elif pos_name == "DEF":
            self.team.defenders.remove(player)
        elif pos_name == "MID":
            self.team.midfielders.remove(player)
        elif pos_name == "FWD":
            self.team.forwards.remove(player)
        
        console.print(f"[yellow]Removed {player['player_name']}[/yellow]")
        Prompt.ask("Press Enter to continue")
    
    def _show_team_stats(self):
        """Display detailed team statistics"""
        console.clear()
        console.print(Panel.fit("[bold]Team Statistics[/bold]", border_style="cyan"))
        
        if not self.team.all_players:
            console.print("[red]No players in team![/red]")
            Prompt.ask("Press Enter to continue")
            return
        
        # Team composition
        console.print("\n[bold]Team Composition:[/bold]")
        team_counts = {}
        for player in self.team.all_players:
            team = player['team_name']
            team_counts[team] = team_counts.get(team, 0) + 1
        
        for team, count in sorted(team_counts.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {team}: {count} player{'s' if count > 1 else ''}")
        
        # Value distribution
        console.print("\n[bold]Value Distribution:[/bold]")
        console.print(f"  Total cost: £{self.team.total_cost:.1f}m")
        console.print(f"  Remaining: £{self.team.remaining_budget:.1f}m")
        console.print(f"  Average player cost: £{self.team.total_cost / max(1, self.team.team_size):.1f}m")
        
        # Points summary
        total_points = sum(p.get('total_points', 0) for p in self.team.all_players)
        console.print(f"\n[bold]Points Summary:[/bold]")
        console.print(f"  Total points: {total_points}")
        console.print(f"  Average points: {total_points / max(1, self.team.team_size):.1f}")
        
        # Top performers
        console.print("\n[bold]Top Performers:[/bold]")
        sorted_players = sorted(self.team.all_players, 
                              key=lambda x: x.get('total_points', 0), 
                              reverse=True)[:5]
        for player in sorted_players:
            console.print(f"  {player['player_name']}: {player.get('total_points', 0)} pts")
        
        Prompt.ask("\nPress Enter to continue")
    
    def _finalize_team(self) -> TeamSelection:
        """Finalize team selection with captain choices"""
        console.clear()
        console.print(Panel.fit("[bold green]Finalize Team[/bold green]", border_style="green"))
        
        # Select captain
        console.print("\n[bold]Select Captain (2x points):[/bold]")
        all_players = self.team.all_players
        
        table = Table(box=box.ROUNDED)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Player", style="white", width=20)
        table.add_column("Position", style="dim", width=4)
        table.add_column("Points", style="yellow", width=6)
        
        for i, player in enumerate(all_players, 1):
            table.add_row(
                str(i),
                player['player_name'],
                player['position'],
                str(int(player.get('total_points', 0)))
            )
        
        console.print(table)
        
        captain_choice = IntPrompt.ask("Select captain", 
                                      choices=[str(i) for i in range(1, len(all_players) + 1)])
        self.team.captain = all_players[captain_choice - 1]['player_id']
        
        # Select vice-captain
        console.print(f"\n[bold]Select Vice-Captain:[/bold]")
        console.print(f"[dim]Captain: {all_players[captain_choice - 1]['player_name']}[/dim]")
        
        vc_choice = IntPrompt.ask("Select vice-captain (different from captain)", 
                                 choices=[str(i) for i in range(1, len(all_players) + 1) 
                                        if i != captain_choice])
        
        # Adjust index if needed
        vc_idx = int(vc_choice) - 1
        self.team.vice_captain = all_players[vc_idx]['player_id']
        
        console.print(f"\n[green]✓ Team finalized![/green]")
        console.print(f"Captain: {all_players[captain_choice - 1]['player_name']}")
        console.print(f"Vice-Captain: {all_players[vc_idx]['player_name']}")
        
        return self.team
    
    def _get_position_count(self, position: str) -> int:
        """Get current count for a position"""
        if position == 'GK':
            return len(self.team.goalkeepers)
        elif position == 'DEF':
            return len(self.team.defenders)
        elif position == 'MID':
            return len(self.team.midfielders)
        elif position == 'FWD':
            return len(self.team.forwards)
        return 0
    
    def _get_position_name(self, position: str) -> str:
        """Get full position name"""
        return {
            'GK': 'Goalkeeper',
            'DEF': 'Defender',
            'MID': 'Midfielder',
            'FWD': 'Forward'
        }.get(position, position)


def load_existing_team(data: pd.DataFrame, player_ids: List[int]) -> TeamSelection:
    """Load an existing team from player IDs
    
    Args:
        data: Player data DataFrame
        player_ids: List of player IDs
        
    Returns:
        TeamSelection object with loaded team
    """
    team = TeamSelection()
    
    for player_id in player_ids:
        player_data = data[data['player_id'] == player_id]
        if player_data.empty:
            continue
            
        player = player_data.iloc[0]
        player_dict = {
            'player_id': int(player['player_id']),
            'player_name': player['player_name'],
            'team_name': player['team_name'],
            'price': player['price'],
            'total_points': player.get('total_points', 0),
            'form': player.get('form', 0),
            'position': player['position']
        }
        
        position = player['position']
        if position == 'GK':
            team.goalkeepers.append(player_dict)
        elif position == 'DEF':
            team.defenders.append(player_dict)
        elif position == 'MID':
            team.midfielders.append(player_dict)
        elif position == 'FWD':
            team.forwards.append(player_dict)
    
    return team