"""
Team Optimizer module for recommending optimal FPL teams
Uses the existing rule-based scorer with MILP optimization
"""

import json
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pulp

from src.data.data_merger import DataMerger
from src.models.rule_based_scorer import RuleBasedScorer, ScoringWeights

warnings.filterwarnings("ignore")


@dataclass
class TeamConstraints:
    """FPL team constraints"""

    budget: float = 100.0
    min_budget_spend: float = 98.0  # Research shows ~99m is optimal
    squad_size: int = 15
    positions: dict[str, int] = None
    max_per_team: int = 3

    def __post_init__(self):
        if self.positions is None:
            self.positions = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


class TeamOptimizer:
    """Optimizes FPL team selection using rule-based scoring and MILP"""

    def __init__(self, db_path: str = "data/fpl_data.db"):
        self.db_path = db_path
        self.constraints = TeamConstraints()
        # Load scoring weights from config
        weights = self._load_scoring_weights()
        self.scorer = RuleBasedScorer(weights)
        # Don't keep persistent connection - create fresh for each request
        self.merger = None

    def _load_scoring_weights(self):
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
        except Exception as e:
            raise ValueError(f"Error loading scoring weights: {e}")

    def get_player_scores(self, gameweek: Optional[int] = None) -> pd.DataFrame:
        """Get players with their scores from the rule-based scorer"""
        # Create fresh merger instance
        merger = DataMerger(self.db_path)
        # Load latest data
        data = merger.get_latest_data(top_n=500)
        merger.close()

        if data.empty:
            return pd.DataFrame()

        # Filter out unavailable players (injured/suspended)
        if "is_available" in data.columns:
            available_count = len(data)
            data = data[data["is_available"] == 1]
            if available_count > len(data):
                print(f"Filtered out {available_count - len(data)} unavailable players")

        # Score players using the existing system
        scored = self.scorer.score_all_players(data)

        # Filter for only players with real odds (like existing system does)
        if "has_real_odds" in scored.columns:
            scored = scored[scored["has_real_odds"]]

        # Handle missing values
        numeric_columns = scored.select_dtypes(include=[np.number]).columns
        scored[numeric_columns] = scored[numeric_columns].fillna(0)

        return scored

    def optimize_team_milp(self, df: pd.DataFrame, strategy: str = "balanced") -> dict:
        """Use Mixed Integer Linear Programming to select optimal team"""

        if df.empty:
            return {"error": "No players available for optimization"}

        # Apply strategy-specific adjustments to scores
        df = self._apply_strategy(df, strategy)

        # Create optimization problem
        prob = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)

        # Decision variables - binary for each player
        player_vars = {}
        for idx, row in df.iterrows():
            player_vars[row["player_id"]] = pulp.LpVariable(
                f"player_{row['player_id']}", cat="Binary"
            )

        # Objective: Maximize total team score
        prob += (
            pulp.lpSum(
                [player_vars[row["player_id"]] * row["model_score"] for idx, row in df.iterrows()]
            ),
            "Total_Team_Score",
        )

        # Constraint 1: Budget (must be between min_spend and max)
        total_cost = pulp.lpSum(
            [player_vars[row["player_id"]] * row["price"] for idx, row in df.iterrows()]
        )
        prob += total_cost <= self.constraints.budget, "Max_Budget"
        prob += total_cost >= self.constraints.min_budget_spend, "Min_Budget"

        # Constraint 2: Squad size
        prob += (
            pulp.lpSum([player_vars[row["player_id"]] for idx, row in df.iterrows()])
            == self.constraints.squad_size,
            "Squad_Size",
        )

        # Constraint 3: Position requirements
        for position, count in self.constraints.positions.items():
            pos_players = df[df["position"] == position]

            prob += (
                pulp.lpSum([player_vars[row["player_id"]] for idx, row in pos_players.iterrows()])
                == count,
                f"Position_{position}",
            )

        # Constraint 4: Max players per team
        for team in df["team"].unique():
            prob += (
                pulp.lpSum(
                    [
                        player_vars[row["player_id"]]
                        for idx, row in df[df["team"] == team].iterrows()
                    ]
                )
                <= self.constraints.max_per_team,
                f"Team_{team}_Limit",
            )

        # Solve the problem with longer timeout for robustness
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=120)  # Increased from 30 to 120 seconds
        status = prob.solve(solver)

        if status != pulp.LpStatusOptimal:
            return {
                "error": f"MILP optimization failed with status: {pulp.LpStatus[status]}",
                "optimization_status": "failed",
            }

        # Extract selected team
        selected_players = []
        for idx, row in df.iterrows():
            if player_vars[row["player_id"]].varValue == 1:
                selected_players.append(row.to_dict())

        if len(selected_players) != self.constraints.squad_size:
            return {
                "error": f"MILP selected {len(selected_players)} players instead of {self.constraints.squad_size}",
                "optimization_status": "incomplete",
            }

        selected_df = pd.DataFrame(selected_players)

        # Calculate team metrics
        total_cost = selected_df["price"].sum()
        total_score = selected_df["model_score"].sum()

        # Select captain (highest score)
        captain = selected_df.nlargest(1, "model_score").iloc[0]

        # Format team by position
        gk_players = selected_df[selected_df["position"] == "GK"]

        team_by_position = {
            "GK": gk_players.to_dict("records"),
            "DEF": selected_df[selected_df["position"] == "DEF"].to_dict("records"),
            "MID": selected_df[selected_df["position"] == "MID"].to_dict("records"),
            "FWD": selected_df[selected_df["position"] == "FWD"].to_dict("records"),
        }

        return {
            "team": team_by_position,
            "total_cost": total_cost,
            "expected_points": total_score,
            "captain": captain.to_dict(),
            "strategy": strategy,
            "optimization_status": "optimal",
        }

    def _apply_strategy(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Apply strategy-specific score adjustments"""
        df = df.copy()

        if strategy == "differential":
            # Boost low-ownership players
            if "selected_by_percent" in df.columns:
                ownership_factor = (100 - df["selected_by_percent"]) / 100
                df["model_score"] = df["model_score"] * (1 + ownership_factor * 0.3)

        elif strategy == "template":
            # Boost high-ownership players
            if "selected_by_percent" in df.columns:
                ownership_factor = df["selected_by_percent"] / 100
                df["model_score"] = df["model_score"] * (1 + ownership_factor * 0.2)

        elif strategy == "short_term":
            # Focus more on current form and fixtures
            if "fixture_difficulty" in df.columns:
                fixture_factor = (6 - df["fixture_difficulty"]) / 5  # Invert and normalize
                df["model_score"] = df["model_score"] * (1 + fixture_factor * 0.2)

        elif strategy == "long_term":
            # Focus on consistency and upcoming fixtures
            if "fixture_diff_next5" in df.columns:
                fixture_factor = (6 - df["fixture_diff_next5"]) / 5
                df["model_score"] = df["model_score"] * (1 + fixture_factor * 0.15)

        # Ensure premium players aren't undervalued
        # Research shows Haaland and similar premiums are essential
        premium_threshold = 12.0  # Players above £12m
        premium_mask = df["price"] >= premium_threshold
        df.loc[premium_mask, "model_score"] = df.loc[premium_mask, "model_score"] * 1.1

        return df

    def _greedy_team_selection(self, df: pd.DataFrame) -> dict:
        """Greedy team selection as fallback"""
        df = df.copy()

        # Sort by score to price ratio but ensure premiums are included
        df["selection_score"] = df["model_score"]

        # Boost premium players to ensure they're selected
        premium_mask = df["price"] >= 12.0
        df.loc[premium_mask, "selection_score"] = df.loc[premium_mask, "selection_score"] * 2

        df = df.sort_values("selection_score", ascending=False)

        selected = []
        budget_left = self.constraints.budget
        team_counts = {}
        position_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}

        # First pass - select high-scoring players
        for idx, player in df.iterrows():
            pos = player["position"]

            # Check constraints
            if player["price"] > budget_left:
                continue
            if position_counts[pos] >= self.constraints.positions[pos]:
                continue
            if team_counts.get(player["team"], 0) >= self.constraints.max_per_team:
                continue

            # Add player
            selected.append(player.to_dict())
            budget_left -= player["price"]
            position_counts[pos] += 1
            team_counts[player["team"]] = team_counts.get(player["team"], 0) + 1

            # Check if team is complete
            if len(selected) == self.constraints.squad_size:
                break

        # If we didn't get a full team, fill with cheapest players
        if len(selected) < self.constraints.squad_size:
            df_remaining = df.sort_values("price")
            for idx, player in df_remaining.iterrows():
                if any(p["player_id"] == player["player_id"] for p in selected):
                    continue

                pos = player["position"]

                if position_counts[pos] >= self.constraints.positions[pos]:
                    continue
                if team_counts.get(player["team"], 0) >= self.constraints.max_per_team:
                    continue

                selected.append(player.to_dict())
                budget_left -= player["price"]
                position_counts[pos] += 1
                team_counts[player["team"]] = team_counts.get(player["team"], 0) + 1

                if len(selected) == self.constraints.squad_size:
                    break

        if len(selected) < self.constraints.squad_size:
            return {"error": f"Could only select {len(selected)} players"}

        selected_df = pd.DataFrame(selected)

        # Format result
        gk_players = selected_df[selected_df["position"] == "GK"]

        team_by_position = {
            "GK": gk_players.to_dict("records"),
            "DEF": selected_df[selected_df["position"] == "DEF"].to_dict("records"),
            "MID": selected_df[selected_df["position"] == "MID"].to_dict("records"),
            "FWD": selected_df[selected_df["position"] == "FWD"].to_dict("records"),
        }

        captain = selected_df.nlargest(1, "model_score").iloc[0] if len(selected_df) > 0 else None

        return {
            "team": team_by_position,
            "total_cost": selected_df["price"].sum(),
            "expected_points": selected_df["model_score"].sum(),
            "captain": captain.to_dict() if captain is not None else None,
            "strategy": "greedy_fallback",
            "optimization_status": "fallback",
        }

    def recommend_team(self, strategy: str = "balanced", gameweek: Optional[int] = None) -> dict:
        """Main method to recommend an optimal team"""
        try:
            # Get scored players
            df = self.get_player_scores(gameweek)

            if df.empty:
                return {"error": "No player data available"}

            print(f"Optimizing team with {len(df)} players...")
            print(
                f"Top scorer: {df.nlargest(1, 'model_score')[['player_name', 'model_score', 'price']].to_dict('records')[0]}"
            )

            # Optimize team selection
            result = self.optimize_team_milp(df, strategy)

            # Log budget usage but don't fall back
            if result.get("total_cost", 0) < 95.0:
                print(f"Warning: Only spending £{result.get('total_cost', 0):.1f}m of £100m budget")

            return result

        except Exception as e:
            print(f"Error in team recommendation: {e}")
            return {"error": f"Optimization failed: {str(e)}"}

    def get_strategies(self) -> list[str]:
        """Get available optimization strategies"""
        return ["balanced", "short_term", "long_term", "differential", "template"]
