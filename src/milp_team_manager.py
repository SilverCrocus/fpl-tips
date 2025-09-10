"""
MILP-based Team Manager
Uses Mixed Integer Linear Programming for optimal transfer, captain, and chip decisions
"""

import logging
from dataclasses import dataclass

import pandas as pd
import pulp

logger = logging.getLogger(__name__)


@dataclass
class MILPTransferResult:
    """Result from MILP transfer optimization"""

    transfers_out: list[dict]
    transfers_in: list[dict]
    total_cost: float
    expected_points_gain: float
    new_team_value: float
    optimization_status: str


class MILPTransferOptimizer:
    """
    MILP-based transfer optimizer that considers:
    - Transfer costs (-4 points per transfer beyond free transfers)
    - Budget constraints
    - Team composition rules
    - Expected points over multiple gameweeks
    """

    def __init__(self, scorer, data: pd.DataFrame):
        self.scorer = scorer
        self.data = data

    def optimize_transfers(
        self,
        current_team_ids: list[int],
        budget: float,
        free_transfers: int = 1,
        max_transfers: int = 3,
        horizon_weeks: int = 5,
    ) -> MILPTransferResult:
        """
        Find optimal transfers using MILP

        Args:
            current_team_ids: List of current player IDs
            budget: Available bank money
            free_transfers: Number of free transfers available
            max_transfers: Maximum transfers to consider
            horizon_weeks: Gameweeks to optimize over

        Returns:
            MILPTransferResult with optimal transfers
        """
        # Score all players
        all_players = self.scorer.score_all_players(self.data)

        # Filter to players with real odds
        if "has_real_odds" in all_players.columns:
            all_players = all_players[all_players["has_real_odds"]]

        # Get current team and available players
        current_team = all_players[all_players["player_id"].isin(current_team_ids)]
        available_players = all_players[~all_players["player_id"].isin(current_team_ids)]

        # Pre-filter available players to those realistically affordable
        # Maximum money available = budget + max sell price from team
        max_sell_price = (
            current_team.nlargest(1, "price")["price"].iloc[0] if not current_team.empty else 0
        )
        max_available_budget = budget + max_sell_price

        # Filter out players that are impossible to afford even with best case scenario
        available_players = available_players[available_players["price"] <= max_available_budget]

        # Create the optimization problem
        prob = pulp.LpProblem("Transfer_Optimization", pulp.LpMaximize)

        # Decision variables
        # Keep/sell current players (binary)
        keep_vars = {}
        for _, player in current_team.iterrows():
            keep_vars[player["player_id"]] = pulp.LpVariable(
                f"keep_{player['player_id']}", cat="Binary"
            )

        # Buy new players (binary)
        buy_vars = {}
        for _, player in available_players.iterrows():
            buy_vars[player["player_id"]] = pulp.LpVariable(
                f"buy_{player['player_id']}", cat="Binary"
            )

        # Calculate expected points over horizon
        # Account for form trends and fixture difficulty
        def calculate_expected_points(player, horizon_weeks):
            base_score = player["model_score"]

            # Adjust for fixture difficulty over horizon
            if "fixture_diff_next5" in player:
                fixture_factor = (5 - player["fixture_diff_next5"]) / 5
                base_score *= 1 + fixture_factor * 0.2

            # Project over horizon weeks
            return base_score * horizon_weeks

        # Calculate transfer cost
        num_transfers = pulp.lpSum([1 - keep_vars[pid] for pid in current_team_ids])

        transfer_cost = 0
        if free_transfers < max_transfers:
            # Cost is -4 points per transfer beyond free transfers
            transfer_cost = (num_transfers - free_transfers) * 4

        # Objective: Maximize expected points minus transfer costs
        team_score = 0

        # Points from kept players
        for _, player in current_team.iterrows():
            expected_pts = calculate_expected_points(player, horizon_weeks)
            team_score += keep_vars[player["player_id"]] * expected_pts

        # Points from bought players
        for _, player in available_players.iterrows():
            expected_pts = calculate_expected_points(player, horizon_weeks)
            team_score += buy_vars[player["player_id"]] * expected_pts

        # Set objective (maximize points - transfer cost)
        prob += team_score - transfer_cost, "Total_Expected_Points"

        # Constraints

        # 1. Squad size must remain 15
        prob += (
            pulp.lpSum([keep_vars[pid] for pid in current_team_ids])
            + pulp.lpSum([buy_vars[pid] for pid in available_players["player_id"]])
            == 15,
            "Squad_Size",
        )

        # 2. Budget constraint
        money_out = pulp.lpSum(
            [
                (1 - keep_vars[player["player_id"]]) * player["price"]
                for _, player in current_team.iterrows()
            ]
        )

        money_in = pulp.lpSum(
            [
                buy_vars[player["player_id"]] * player["price"]
                for _, player in available_players.iterrows()
            ]
        )

        prob += money_in <= money_out + budget, "Budget_Constraint"

        # 3. Position constraints (must maintain valid formation)
        positions = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

        for position, required_count in positions.items():
            current_pos = current_team[current_team["position"] == position]
            available_pos = available_players[available_players["position"] == position]

            count = pulp.lpSum(
                [keep_vars[player["player_id"]] for _, player in current_pos.iterrows()]
            ) + pulp.lpSum(
                [buy_vars[player["player_id"]] for _, player in available_pos.iterrows()]
            )

            prob += count == required_count, f"Position_{position}"

        # 4. Max 3 players per team
        all_teams = pd.concat([current_team, available_players])["team"].unique()

        for team in all_teams:
            current_team_players = current_team[current_team["team"] == team]
            available_team_players = available_players[available_players["team"] == team]

            team_count = pulp.lpSum(
                [keep_vars[player["player_id"]] for _, player in current_team_players.iterrows()]
            ) + pulp.lpSum(
                [buy_vars[player["player_id"]] for _, player in available_team_players.iterrows()]
            )

            prob += team_count <= 3, f"Team_{team}_Limit"

        # 5. Transfer limit
        prob += num_transfers <= max_transfers, "Transfer_Limit"

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
        status = prob.solve(solver)

        if status != pulp.LpStatusOptimal:
            return MILPTransferResult(
                transfers_out=[],
                transfers_in=[],
                total_cost=0,
                expected_points_gain=0,
                new_team_value=current_team["price"].sum(),
                optimization_status=f"Failed: {pulp.LpStatus[status]}",
            )

        # Extract results
        transfers_out = []
        transfers_in = []

        # Find sold players
        for _, player in current_team.iterrows():
            if keep_vars[player["player_id"]].varValue == 0:
                transfers_out.append(
                    {
                        "id": player["player_id"],
                        "name": player["player_name"],
                        "team": player.get("team_name", player.get("team", "Unknown")),
                        "position": player["position"],
                        "price": player["price"],
                        "score": player["model_score"],
                    }
                )

        # Find bought players
        for _, player in available_players.iterrows():
            if buy_vars[player["player_id"]].varValue == 1:
                transfers_in.append(
                    {
                        "id": player["player_id"],
                        "name": player["player_name"],
                        "team": player.get("team_name", player.get("team", "Unknown")),
                        "position": player["position"],
                        "price": player["price"],
                        "score": player["model_score"],
                    }
                )

        # Calculate metrics
        money_out = sum(p["price"] for p in transfers_out)
        money_in = sum(p["price"] for p in transfers_in)
        total_cost = money_in - money_out

        old_score = sum(p["score"] for p in transfers_out)
        new_score = sum(p["score"] for p in transfers_in)
        points_gain = new_score - old_score

        # New team value
        kept_value = current_team[
            current_team["player_id"].isin(
                [
                    pid
                    for pid in current_team_ids
                    if keep_vars.get(pid) and keep_vars[pid].varValue == 1
                ]
            )
        ]["price"].sum()
        bought_value = sum(p["price"] for p in transfers_in)
        new_team_value = kept_value + bought_value

        # Sort transfers to ensure affordability order
        # Pair up transfers and sort by net cost (cheapest/profitable first)
        paired_transfers = []
        for i in range(min(len(transfers_out), len(transfers_in))):
            net_cost = transfers_in[i]["price"] - transfers_out[i]["price"]
            paired_transfers.append((transfers_out[i], transfers_in[i], net_cost))

        # Sort by net cost (ascending - cheapest or profitable first)
        paired_transfers.sort(key=lambda x: x[2])

        # Validate transfer affordability in sequence
        sorted_transfers_out = []
        sorted_transfers_in = []
        running_budget = budget

        for out_player, in_player, net_cost in paired_transfers:
            # Check if this individual transfer is affordable
            if running_budget >= net_cost:
                sorted_transfers_out.append(out_player)
                sorted_transfers_in.append(in_player)
                running_budget -= net_cost
            else:
                # Skip unaffordable transfers when done individually
                # Log warning about transfer that can't be done in isolation
                import logging

                logging.warning(
                    f"Transfer {out_player['name']} → {in_player['name']} "
                    f"requires £{net_cost:.1f}m but only £{running_budget:.1f}m available"
                )

        # If no transfers are affordable individually, return empty result
        if not sorted_transfers_out:
            return MILPTransferResult(
                transfers_out=[],
                transfers_in=[],
                total_cost=0,
                expected_points_gain=0,
                new_team_value=current_team["price"].sum(),
                optimization_status="No affordable transfers found",
            )

        return MILPTransferResult(
            transfers_out=sorted_transfers_out,
            transfers_in=sorted_transfers_in,
            total_cost=total_cost,
            expected_points_gain=points_gain,
            new_team_value=new_team_value,
            optimization_status="optimal",
        )


class MILPCaptainSelector:
    """
    MILP-based captain selection
    Considers expected returns, variance, and effective ownership
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def select_captain_and_vice(
        self, team_ids: list[int], consider_effective_ownership: bool = True
    ) -> dict:
        """
        Select optimal captain and vice-captain using MILP

        Args:
            team_ids: List of player IDs in team
            consider_effective_ownership: Whether to consider EO in optimization

        Returns:
            Dictionary with captain and vice-captain recommendations
        """
        # Get team data
        team_data = self.data[self.data["player_id"].isin(team_ids)]

        # Filter out goalkeepers (they should never be captain)
        team_data = team_data[team_data["position"] != "GK"]

        if team_data.empty:
            return {"error": "No valid captain options"}

        # Create optimization problem
        prob = pulp.LpProblem("Captain_Selection", pulp.LpMaximize)

        # Decision variables (binary for each player)
        captain_vars = {}
        vice_vars = {}

        for _, player in team_data.iterrows():
            captain_vars[player["player_id"]] = pulp.LpVariable(
                f"captain_{player['player_id']}", cat="Binary"
            )
            vice_vars[player["player_id"]] = pulp.LpVariable(
                f"vice_{player['player_id']}", cat="Binary"
            )

        # Calculate expected captain points
        def calculate_captain_score(player):
            """Calculate expected points for captaincy"""
            score = 0

            # Base expected points (captain gets double)
            base_points = player.get("model_score", 0) * 2
            score += base_points

            # Consider form heavily
            if "form" in player:
                score += player["form"] * 1.5

            # Goal probability is crucial for captain
            if "prob_goal" in player and pd.notna(player["prob_goal"]):
                score += player["prob_goal"] * 10

            # Fixture difficulty (lower is better)
            if "fixture_difficulty" in player and pd.notna(player["fixture_difficulty"]):
                score += (5 - player["fixture_difficulty"]) * 2

            # Penalty for injury doubt
            if "chance_of_playing_next_round" in player:
                if player["chance_of_playing_next_round"] < 100:
                    score *= player["chance_of_playing_next_round"] / 100

            # Effective ownership consideration
            if consider_effective_ownership and "selected_by_percent" in player:
                # Higher ownership means less differential
                ownership = player["selected_by_percent"]
                if ownership > 50:  # Very high ownership
                    score *= 0.95  # Small penalty for template picks
                elif ownership < 10:  # Differential
                    score *= 1.1  # Bonus for differential captain

            return score

        # Objective: Maximize expected captain returns
        total_score = pulp.lpSum(
            [
                captain_vars[player["player_id"]] * calculate_captain_score(player) * 1.0
                + vice_vars[player["player_id"]]
                * calculate_captain_score(player)
                * 0.3  # Vice worth 30%
                for _, player in team_data.iterrows()
            ]
        )

        prob += total_score, "Captain_Score"

        # Constraints

        # 1. Exactly one captain
        prob += (
            pulp.lpSum([captain_vars[pid] for pid in team_data["player_id"]]) == 1,
            "One_Captain",
        )

        # 2. Exactly one vice-captain
        prob += (
            pulp.lpSum([vice_vars[pid] for pid in team_data["player_id"]]) == 1,
            "One_Vice_Captain",
        )

        # 3. Captain and vice must be different players
        for pid in team_data["player_id"]:
            prob += captain_vars[pid] + vice_vars[pid] <= 1, f"Different_{pid}"

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=0)
        status = prob.solve(solver)

        if status != pulp.LpStatusOptimal:
            # Fallback to highest scorer
            best = team_data.nlargest(2, "model_score")
            if len(best) >= 2:
                return {
                    "captain": {
                        "name": best.iloc[0]["player_name"],
                        "score": calculate_captain_score(best.iloc[0]),
                    },
                    "vice_captain": {
                        "name": best.iloc[1]["player_name"],
                        "score": calculate_captain_score(best.iloc[1]),
                    },
                    "method": "fallback",
                }

        # Extract results
        captain = None
        vice_captain = None

        for _, player in team_data.iterrows():
            if captain_vars[player["player_id"]].varValue == 1:
                captain = {
                    "id": player["player_id"],
                    "name": player["player_name"],
                    "team": player.get("team_name", player.get("team", "Unknown")),
                    "score": calculate_captain_score(player),
                    "form": player.get("form", 0),
                    "fixture_difficulty": player.get("fixture_difficulty", 3),
                }
            if vice_vars[player["player_id"]].varValue == 1:
                vice_captain = {
                    "id": player["player_id"],
                    "name": player["player_name"],
                    "team": player.get("team_name", player.get("team", "Unknown")),
                    "score": calculate_captain_score(player),
                    "form": player.get("form", 0),
                    "fixture_difficulty": player.get("fixture_difficulty", 3),
                }

        return {
            "captain": captain,
            "vice_captain": vice_captain,
            "method": "MILP",
            "optimization_status": "optimal",
        }


class MILPChipAdvisor:
    """
    MILP-based chip timing advisor
    Optimizes when to play chips for maximum benefit
    """

    def __init__(self, scorer, data: pd.DataFrame):
        self.scorer = scorer
        self.data = data

    def optimize_wildcard_team(self, budget: float = 100.0) -> dict:
        """
        Optimize complete team rebuild using wildcard (uses existing TeamOptimizer)

        Args:
            budget: Total budget available

        Returns:
            Optimal wildcard team
        """
        from src.team_optimizer import TeamOptimizer

        optimizer = TeamOptimizer()
        result = optimizer.recommend_team(strategy="balanced")

        if "error" not in result:
            result["chip_type"] = "wildcard"
            result["recommendation"] = "Use wildcard to rebuild with this optimal team"

        return result

    def evaluate_bench_boost(self, team_ids: list[int]) -> dict:
        """
        Evaluate if bench boost should be used this week

        Args:
            team_ids: All 15 player IDs

        Returns:
            Bench boost recommendation
        """
        # Score all players first
        scored_data = self.scorer.score_all_players(self.data)
        team_data = scored_data[scored_data["player_id"].isin(team_ids)]

        # Identify likely bench players (lowest scoring per position after filling starting 11)
        # Assume 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD in starting 11
        # So bench is typically: 1 GK, 1-2 DEF, 1-2 MID, 0-1 FWD

        gks = team_data[team_data["position"] == "GK"].nsmallest(1, "model_score")
        defs = team_data[team_data["position"] == "DEF"].nsmallest(2, "model_score")
        mids = team_data[team_data["position"] == "MID"].nsmallest(2, "model_score")
        fwds = team_data[team_data["position"] == "FWD"].nsmallest(1, "model_score")

        # Likely bench (bottom 4 players)
        likely_bench = pd.concat([gks, defs, mids, fwds]).nsmallest(4, "model_score")

        if likely_bench.empty:
            return {"use": False, "reason": "Could not identify bench players"}

        # Calculate bench expected points
        bench_score = likely_bench["model_score"].sum()
        avg_bench_score = bench_score / 4

        # Check bench fixture difficulty
        if "fixture_difficulty" in likely_bench.columns:
            avg_difficulty = likely_bench["fixture_difficulty"].mean()
        else:
            avg_difficulty = 3.0

        # Decision logic
        reasons = []
        confidence = 0

        if avg_bench_score > 5:  # Strong bench expected points
            reasons.append(
                f"Strong bench scoring potential ({bench_score:.1f} total points expected)"
            )
            confidence += 3

        if avg_difficulty < 2.5:  # Easy fixtures
            reasons.append(f"Favorable bench fixtures (avg difficulty: {avg_difficulty:.1f})")
            confidence += 2

        # Check for DGW (Double Gameweek) - simplified check
        if "is_dgw" in team_data.columns and team_data["is_dgw"].any():
            reasons.append("Double gameweek - bench players play twice!")
            confidence += 5

        # Make recommendation
        if confidence >= 5:
            return {
                "use": True,
                "confidence": "high",
                "expected_bench_points": bench_score,
                "reasons": reasons,
                "bench_players": [
                    {"name": p["player_name"], "expected_points": p["model_score"]}
                    for _, p in likely_bench.iterrows()
                ],
            }
        elif confidence >= 3:
            return {
                "use": "consider",
                "confidence": "medium",
                "expected_bench_points": bench_score,
                "reasons": reasons,
            }
        else:
            return {
                "use": False,
                "confidence": "low",
                "reason": "Bench not strong enough this week",
                "expected_bench_points": bench_score,
            }

    def evaluate_triple_captain(self, team_ids: list[int]) -> dict:
        """
        Evaluate if triple captain should be used

        Args:
            team_ids: Player IDs in team

        Returns:
            Triple captain recommendation
        """
        # Score all players first
        scored_data = self.scorer.score_all_players(self.data)
        team_data = scored_data[scored_data["player_id"].isin(team_ids)]

        # Find best captain option
        non_gk = team_data[team_data["position"] != "GK"]
        if non_gk.empty:
            return {"use": False, "reason": "No valid captain options"}

        best_player = non_gk.nlargest(1, "model_score").iloc[0]

        # Calculate expected triple captain return
        expected_return = best_player["model_score"] * 3  # Triple points

        # Decision factors
        reasons = []
        confidence = 0

        # Check if it's a premium player with high ceiling
        if best_player["price"] > 11:  # Premium player
            reasons.append(f"{best_player['player_name']} is a premium player")
            confidence += 1

        # Check form
        if "form" in best_player and best_player["form"] > 8:
            reasons.append(f"Exceptional form ({best_player['form']:.1f})")
            confidence += 3

        # Check fixture
        if "fixture_difficulty" in best_player and best_player["fixture_difficulty"] <= 2:
            reasons.append(
                f"Very favorable fixture (difficulty: {best_player['fixture_difficulty']})"
            )
            confidence += 3

        # Check for DGW
        if "is_dgw" in best_player and best_player["is_dgw"]:
            reasons.append("Double gameweek - plays twice!")
            confidence += 5

        # Check expected return threshold
        if expected_return > 30:  # Very high expected return
            reasons.append(f"Exceptional expected return ({expected_return:.1f} points)")
            confidence += 3
        elif expected_return > 20:
            reasons.append(f"Strong expected return ({expected_return:.1f} points)")
            confidence += 1

        # Make recommendation
        if confidence >= 6:
            return {
                "use": True,
                "player": best_player["player_name"],
                "expected_points": expected_return,
                "confidence": "high",
                "reasons": reasons,
            }
        elif confidence >= 4:
            return {
                "use": "consider",
                "player": best_player["player_name"],
                "expected_points": expected_return,
                "confidence": "medium",
                "reasons": reasons,
            }
        else:
            return {
                "use": False,
                "confidence": "low",
                "reason": "Wait for better opportunity (DGW or exceptional fixture)",
                "best_option": best_player["player_name"],
                "expected_points": expected_return,
            }
