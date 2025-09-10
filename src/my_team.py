"""
My Team Management Module
Handles personalized team analysis and transfer recommendations
Now uses MILP optimization for transfers, captain selection, and chip timing
With strategic chip timing based on FPL best practices
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

# Import strategic chip advisor
from .chip_strategy import StrategicChipAdvisor

# Import MILP components
from .milp_team_manager import MILPCaptainSelector, MILPChipAdvisor, MILPTransferOptimizer

# Import strategic transfer evaluator
from .transfer_strategy import StrategicTransferEvaluator

logger = logging.getLogger(__name__)


@dataclass
class MyTeam:
    """Represents user's current FPL team"""

    players: list[int]  # Player IDs
    captain: int
    vice_captain: int
    bank: float
    free_transfers: int = 1
    wildcard_available: bool = True
    free_hit_available: bool = True
    bench_boost_available: bool = True
    triple_captain_available: bool = True


@dataclass
class TransferRecommendation:
    """Transfer recommendation"""

    player_out: dict  # Player to remove
    player_in: dict  # Player to bring in
    net_cost: float
    score_improvement: float
    reason: str
    priority: int  # 1=urgent, 2=recommended, 3=optional
    hit_evaluation: Optional[dict] = None  # Strategic hit evaluation


class TeamAnalyzer:
    """Analyze user's team and provide recommendations"""

    def __init__(self, scorer, data: pd.DataFrame):
        self.scorer = scorer
        self.data = data
        self.fixture_conflicts = []  # Track fixture conflicts

    def analyze_team(self, my_team: MyTeam) -> dict:
        """Comprehensive team analysis

        Args:
            my_team: User's current team

        Returns:
            Analysis dictionary with insights
        """
        analysis = {}

        # Get team players data
        team_data = self.data[self.data["player_id"].isin(my_team.players)]

        # Overall metrics
        analysis["total_team_value"] = team_data["price"].sum()
        analysis["total_expected_points"] = team_data.get(
            "expected_points", team_data["total_points"]
        ).sum()
        analysis["average_ownership"] = team_data["selected_by_percent"].mean()

        # Position breakdown
        position_counts = team_data["position"].value_counts()
        analysis["position_breakdown"] = position_counts.to_dict()

        # Team distribution
        team_counts = team_data["team_name"].value_counts()
        analysis["team_distribution"] = team_counts.to_dict()

        # Identify weaknesses
        analysis["weaknesses"] = self._identify_weaknesses(team_data)

        # Fixture analysis
        analysis["fixture_analysis"] = self._analyze_fixtures(team_data)

        # Captain analysis
        analysis["captain_analysis"] = self._analyze_captain(team_data, my_team.captain)

        return analysis

    def get_post_transfer_lineup(
        self, team_data: pd.DataFrame, transfers: list[TransferRecommendation]
    ) -> dict:
        """Get lineup suggestion after applying recommended transfers

        Args:
            team_data: Current team DataFrame
            transfers: List of transfer recommendations to apply

        Returns:
            Dictionary with post-transfer lineup
        """
        # Create a copy of team data
        new_team_data = team_data.copy()

        # Apply transfers
        for transfer in transfers:
            # Remove player out
            new_team_data = new_team_data[new_team_data["player_id"] != transfer.player_out["id"]]

            # Add player in
            # Get full data for the new player
            new_player_data = self.data[self.data["player_id"] == transfer.player_in["id"]]
            if not new_player_data.empty:
                new_team_data = pd.concat([new_team_data, new_player_data], ignore_index=True)

        # Generate lineup for the new team
        return self.get_lineup_suggestion(new_team_data)

    def get_lineup_suggestion(self, team_data: pd.DataFrame) -> dict:
        """Suggest optimal starting 11 and bench order

        Args:
            team_data: DataFrame with team players

        Returns:
            Dictionary with lineup suggestion
        """
        # Score all players for lineup selection
        team_data = team_data.copy()

        # Add lineup score based on form, availability, and fixtures
        team_data["lineup_score"] = 0

        # Base score from form and expected points
        if "form" in team_data.columns:
            team_data["lineup_score"] += team_data["form"] * 2
        if "expected_points" in team_data.columns:
            team_data["lineup_score"] += team_data["expected_points"]
        elif "total_points" in team_data.columns:
            team_data["lineup_score"] += team_data["total_points"] / 10

        # Penalize for difficult fixtures
        if "fixture_difficulty" in team_data.columns:
            team_data["lineup_score"] += 5 - team_data["fixture_difficulty"]

        # Heavy penalty for injured/doubtful players
        if "chance_of_playing_next_round" in team_data.columns:
            team_data.loc[team_data["chance_of_playing_next_round"] < 75, "lineup_score"] *= 0.3
        if "is_available" in team_data.columns:
            team_data.loc[team_data["is_available"] == 0, "lineup_score"] *= 0.1

        # Separate by position
        gkp = team_data[team_data["position"] == "GK"].sort_values(
            "lineup_score", ascending=False
        )
        defs = team_data[team_data["position"] == "DEF"].sort_values(
            "lineup_score", ascending=False
        )
        mids = team_data[team_data["position"] == "MID"].sort_values(
            "lineup_score", ascending=False
        )
        fwds = team_data[team_data["position"] == "FWD"].sort_values(
            "lineup_score", ascending=False
        )

        # Determine best formation based on available players and scores
        formations = []

        # Try common formations
        for def_count, mid_count, fwd_count in [
            (3, 4, 3),
            (3, 5, 2),
            (4, 3, 3),
            (4, 4, 2),
            (4, 5, 1),
            (5, 3, 2),
            (5, 4, 1),
        ]:
            if len(defs) >= def_count and len(mids) >= mid_count and len(fwds) >= fwd_count:
                # Calculate formation score
                formation_score = (
                    gkp.head(1)["lineup_score"].sum()
                    + defs.head(def_count)["lineup_score"].sum()
                    + mids.head(mid_count)["lineup_score"].sum()
                    + fwds.head(fwd_count)["lineup_score"].sum()
                )
                formations.append(
                    {
                        "formation": f"{def_count}-{mid_count}-{fwd_count}",
                        "score": formation_score,
                        "def": def_count,
                        "mid": mid_count,
                        "fwd": fwd_count,
                    }
                )

        if not formations:
            return {"status": "insufficient_players"}

        # Pick best formation
        best_formation = max(formations, key=lambda x: x["score"])

        # Select starting 11
        starting_11 = []
        bench = []

        # Always play 1 goalkeeper
        if len(gkp) > 0:
            starting_11.append(gkp.iloc[0].to_dict())
            for _, player in gkp.iloc[1:].iterrows():
                bench.append(player.to_dict())

        # Add defenders
        for _, player in defs.head(best_formation["def"]).iterrows():
            starting_11.append(player.to_dict())
        for _, player in defs.iloc[best_formation["def"] :].iterrows():
            bench.append(player.to_dict())

        # Add midfielders
        for _, player in mids.head(best_formation["mid"]).iterrows():
            starting_11.append(player.to_dict())
        for _, player in mids.iloc[best_formation["mid"] :].iterrows():
            bench.append(player.to_dict())

        # Add forwards
        for _, player in fwds.head(best_formation["fwd"]).iterrows():
            starting_11.append(player.to_dict())
        for _, player in fwds.iloc[best_formation["fwd"] :].iterrows():
            bench.append(player.to_dict())

        # Sort bench by score (best first for auto-sub priority)
        if len(bench) > 0:
            bench = sorted(bench, key=lambda x: x.get("lineup_score", 0), reverse=True)

        return {
            "formation": best_formation["formation"],
            "formation_score": best_formation["score"],
            "starting_11": [
                {
                    "name": p.get("player_name", "Unknown"),
                    "position": p.get("position", "Unknown"),
                    "score": round(p.get("lineup_score", 0), 1),
                }
                for p in starting_11
            ],
            "bench": [
                {
                    "name": p.get("player_name", "Unknown"),
                    "position": p.get("position", "Unknown"),
                    "score": round(p.get("lineup_score", 0), 1),
                }
                for p in bench
            ],
        }

    def _identify_weaknesses(self, team_data: pd.DataFrame) -> list[dict]:
        """Identify team weaknesses"""
        weaknesses = []

        # Check for injured/unavailable players
        if "is_available" in team_data.columns:
            unavailable = team_data[team_data["is_available"] == 0]
            for _, player in unavailable.iterrows():
                weaknesses.append(
                    {"type": "unavailable", "player": player["player_name"], "severity": "high"}
                )

        # Check for players with poor form
        poor_form = team_data[team_data["form"] < 2.0]
        for _, player in poor_form.iterrows():
            weaknesses.append(
                {
                    "type": "poor_form",
                    "player": player["player_name"],
                    "form": player["form"],
                    "severity": "medium",
                }
            )

        # Check for players with bad fixtures
        if "fixture_diff_next5" in team_data.columns:
            hard_fixtures = team_data[team_data["fixture_diff_next5"] > 3.5]
            for _, player in hard_fixtures.iterrows():
                weaknesses.append(
                    {
                        "type": "hard_fixtures",
                        "player": player["player_name"],
                        "difficulty": player["fixture_diff_next5"],
                        "severity": "low",
                    }
                )

        return weaknesses

    def _analyze_fixtures(self, team_data: pd.DataFrame) -> dict:
        """Analyze team's upcoming fixtures"""
        if "fixture_diff_next5" not in team_data.columns:
            return {"status": "no_fixture_data"}

        return {
            "average_difficulty_next5": team_data["fixture_diff_next5"].mean(),
            "easiest_fixtures": team_data.nsmallest(3, "fixture_diff_next5")[
                ["player_name", "team_name", "fixture_diff_next5"]
            ].to_dict("records"),
            "hardest_fixtures": team_data.nlargest(3, "fixture_diff_next5")[
                ["player_name", "team_name", "fixture_diff_next5"]
            ].to_dict("records"),
        }

    def _analyze_captain(
        self, team_data: pd.DataFrame, captain_id: int, use_milp: bool = True
    ) -> dict:
        """Analyze captain choice using MILP optimization

        Args:
            team_data: DataFrame with team players
            captain_id: Current captain player ID
            use_milp: Whether to use MILP optimization

        Returns:
            Captain analysis dictionary
        """
        if use_milp:
            # Use MILP captain selector
            milp_captain = MILPCaptainSelector(team_data)

            # Get optimal captain and vice-captain
            team_ids = team_data["player_id"].tolist()
            result = milp_captain.select_captain_and_vice(
                team_ids=team_ids, consider_effective_ownership=True
            )

            if "error" not in result and result.get("captain"):
                # Get current captain data
                captain_data = team_data[team_data["player_id"] == captain_id]
                current_captain_name = (
                    captain_data.iloc[0]["player_name"] if not captain_data.empty else "Unknown"
                )

                # Check if current captain is the recommended one
                is_optimal = False
                if result["captain"].get("id") == captain_id:
                    is_optimal = True

                # Create top_3_options format for display
                top_options = [
                    {"player": result["captain"]["name"], "score": result["captain"]["score"]}
                ]

                if result.get("vice_captain"):
                    top_options.append(
                        {
                            "player": result["vice_captain"]["name"],
                            "score": result["vice_captain"]["score"],
                        }
                    )

                # Add third option if available in team data
                team_scored = team_data.copy()
                if "model_score" in team_scored.columns:
                    # Exclude captain and vice from third option
                    exclude_ids = [
                        result["captain"].get("id"),
                        result["vice_captain"].get("id") if result.get("vice_captain") else None,
                    ]
                    other_options = team_scored[
                        ~team_scored["player_id"].isin([id for id in exclude_ids if id])
                    ]
                    if not other_options.empty:
                        third = other_options.nlargest(1, "model_score").iloc[0]
                        top_options.append(
                            {"player": third["player_name"], "score": third["model_score"]}
                        )

                return {
                    "current_captain": current_captain_name,
                    "is_optimal": is_optimal,
                    "recommended_captain": {
                        "player": result["captain"]["name"],
                        "score": result["captain"]["score"],
                        "form": result["captain"].get("form", 0),
                        "fixture": result["captain"].get("fixture_difficulty", 3),
                    },
                    "vice_captain": (
                        {
                            "player": result["vice_captain"]["name"],
                            "score": result["vice_captain"]["score"],
                        }
                        if result.get("vice_captain")
                        else None
                    ),
                    "top_3_options": top_options,
                    "method": "MILP optimization",
                    "message": (
                        "Optimal"
                        if is_optimal
                        else f"Consider switching to {result['captain']['name']}"
                    ),
                }

        # Fallback to original method if MILP fails or is disabled
        captain_data = team_data[team_data["player_id"] == captain_id]

        if captain_data.empty:
            return {"status": "captain_not_found"}

        captain = captain_data.iloc[0]

        # Score all team players for captaincy (excluding goalkeepers)
        captain_scores = []
        for _, player in team_data.iterrows():
            # Skip goalkeepers - they should never be captained
            if player.get("position") == "GK":
                continue

            score = 0
            # Properly handle NaN values
            prob_goal = player.get("prob_goal", 0)
            if pd.notna(prob_goal):
                score += prob_goal * 5

            form = player.get("form", 0)
            if pd.notna(form):
                score += form * 0.5

            # Fixture difficulty (lower is better, so invert)
            fixture_diff = player.get("fixture_difficulty")
            if pd.notna(fixture_diff):
                score += (5 - fixture_diff) * 0.3

            # Skip players with availability issues
            if player.get("chance_of_playing_next_round", 100) < 75:
                score *= 0.5  # Heavily penalize injured/doubtful players

            captain_scores.append({"player": player["player_name"], "score": score})

        # Sort by score, handling NaN properly
        captain_scores = sorted(
            captain_scores, key=lambda x: x["score"] if pd.notna(x["score"]) else -999, reverse=True
        )

        return {
            "current_captain": captain["player_name"],
            "captain_rank": next(
                (
                    i + 1
                    for i, p in enumerate(captain_scores)
                    if p["player"] == captain["player_name"]
                ),
                None,
            ),
            "recommended_captain": captain_scores[0] if captain_scores else None,
            "top_3_options": captain_scores[:3],
            "method": "heuristic scoring",
        }

    def get_post_transfer_captain_analysis(
        self,
        team_data: pd.DataFrame,
        transfers: list[TransferRecommendation],
        current_captain_id: int,
    ) -> dict:
        """Analyze captain options for post-transfer team

        Args:
            team_data: Current team data
            transfers: List of transfer recommendations to apply
            current_captain_id: Current captain player ID

        Returns:
            Dictionary with captain analysis for post-transfer team
        """
        # Create new team data by applying transfers
        new_team_data = team_data.copy()

        # Apply each transfer
        for transfer in transfers:
            # Remove transferred out player
            new_team_data = new_team_data[new_team_data["player_id"] != transfer.player_out["id"]]

            # Add transferred in player
            new_player_data = self.data[self.data["player_id"] == transfer.player_in["id"]]
            if not new_player_data.empty:
                new_team_data = pd.concat([new_team_data, new_player_data], ignore_index=True)

        # Check if current captain is still in the team
        captain_in_new_team = current_captain_id in new_team_data["player_id"].values

        # If captain was transferred out, use the best non-GK player as new captain
        if not captain_in_new_team:
            non_gk_data = new_team_data[new_team_data["position"] != "GK"]
            if not non_gk_data.empty:
                # Use model_score to find best player for captaincy
                best_idx = non_gk_data["model_score"].idxmax()
                current_captain_id = non_gk_data.loc[best_idx, "player_id"]

        # Get captain data
        captain = new_team_data[new_team_data["player_id"] == current_captain_id]
        if captain.empty:
            return {"error": "Could not find captain in post-transfer team"}

        # Use the fixed _analyze_captain method (pass the captain_id, not the data)
        return self._analyze_captain(new_team_data, current_captain_id)

    def detect_fixture_conflicts(
        self, team_data: pd.DataFrame, transfers: list[TransferRecommendation] = None
    ) -> list[dict]:
        """Detect when players in team or transfers face each other

        Args:
            team_data: Current team DataFrame
            transfers: List of transfer recommendations to check

        Returns:
            List of fixture conflicts
        """
        conflicts = []

        # Get teams in current squad
        team_players = {}
        if "team_name" in team_data.columns:
            for _, player in team_data.iterrows():
                team = player.get("team_name", "Unknown")
                if team not in team_players:
                    team_players[team] = []
                team_players[team].append(player.get("player_name", "Unknown"))

        # Add transfer players
        if transfers:
            for transfer in transfers:
                # Player coming in
                in_team = transfer.player_in.get("team_name", "Unknown")
                if in_team not in team_players:
                    team_players[in_team] = []
                team_players[in_team].append(transfer.player_in.get("web_name", "Unknown"))

        # Check for opposing fixtures (simplified check based on fixture difficulty)
        # In a full implementation, we'd check actual fixtures
        all_teams = list(team_players.keys())

        for i, team1 in enumerate(all_teams):
            for team2 in all_teams[i + 1 :]:
                # Get average fixture difficulties
                team1_players_df = (
                    team_data[team_data["team_name"] == team1]
                    if "team_name" in team_data.columns
                    else pd.DataFrame()
                )
                team2_players_df = (
                    team_data[team_data["team_name"] == team2]
                    if "team_name" in team_data.columns
                    else pd.DataFrame()
                )

                if not team1_players_df.empty and not team2_players_df.empty:
                    team1_diff = (
                        team1_players_df["fixture_difficulty"].mean()
                        if "fixture_difficulty" in team1_players_df.columns
                        else 3
                    )
                    team2_diff = (
                        team2_players_df["fixture_difficulty"].mean()
                        if "fixture_difficulty" in team2_players_df.columns
                        else 3
                    )

                    # If one team has easy fixtures (<=2) and other has hard (>=4), they might be playing each other
                    if (team1_diff <= 2 and team2_diff >= 4) or (
                        team1_diff >= 4 and team2_diff <= 2
                    ):
                        conflicts.append(
                            {
                                "team1": team1,
                                "team1_players": team_players[team1],
                                "team2": team2,
                                "team2_players": team_players[team2],
                                "type": "opposing_fixture",
                                "severity": "warning",
                                "message": f"⚠️ Players from {team1} and {team2} may face each other - one's success hurts the other's points!",
                            }
                        )

        return conflicts

    def get_transfer_recommendations(
        self, my_team: MyTeam, num_transfers: int = 2, use_milp: bool = True
    ) -> list[TransferRecommendation]:
        """Get transfer recommendations using MILP optimization

        Args:
            my_team: User's current team
            num_transfers: Number of transfers to recommend
            use_milp: Whether to use MILP (True) or fallback to greedy (False)

        Returns:
            List of transfer recommendations
        """
        if use_milp:
            # Use MILP optimizer
            milp_optimizer = MILPTransferOptimizer(self.scorer, self.data)

            result = milp_optimizer.optimize_transfers(
                current_team_ids=my_team.players,
                budget=my_team.bank,
                free_transfers=my_team.free_transfers,
                max_transfers=num_transfers,
                horizon_weeks=5,
            )

            if result.optimization_status == "optimal" and result.transfers_out:
                # Convert MILP result to TransferRecommendation format
                recommendations = []

                # Initialize strategic evaluator
                strategic_evaluator = StrategicTransferEvaluator(self.data, self.scorer)

                for i, (player_out, player_in) in enumerate(
                    zip(result.transfers_out, result.transfers_in)
                ):
                    # Get full player data for reason generation
                    out_data = self.data[self.data["player_id"] == player_out["id"]]
                    in_data = self.data[self.data["player_id"] == player_in["id"]]

                    if not out_data.empty and not in_data.empty:
                        reason = self._get_transfer_reason(out_data.iloc[0], in_data.iloc[0])
                        priority = self._get_transfer_priority(out_data.iloc[0], in_data.iloc[0])

                        # Strategic evaluation for hit decisions
                        is_free_transfer = i < my_team.free_transfers
                        hit_evaluation = strategic_evaluator.evaluate_transfer_hit(
                            out_data.iloc[0], in_data.iloc[0], is_free_transfer
                        )
                    else:
                        reason = (
                            f"Upgrade: {player_in['score']:.1f} vs {player_out['score']:.1f} score"
                        )
                        priority = 2
                        hit_evaluation = None

                    recommendations.append(
                        TransferRecommendation(
                            player_out=player_out,
                            player_in=player_in,
                            net_cost=player_in["price"] - player_out["price"],
                            score_improvement=player_in["score"] - player_out["score"],
                            reason=reason,
                            priority=priority,
                            hit_evaluation=hit_evaluation,
                        )
                    )

                return recommendations[:num_transfers]
            else:
                logger.warning(
                    f"MILP optimization failed: {result.optimization_status}, falling back to greedy"
                )
                use_milp = False

        if not use_milp:
            # Fallback to original greedy algorithm
            recommendations = []

            # Score all players
            all_scores = self.scorer.score_all_players(self.data)

            # Get team players
            team_data = all_scores[all_scores["player_id"].isin(my_team.players)]
            non_team_data = all_scores[~all_scores["player_id"].isin(my_team.players)]

            # Initialize strategic evaluator
            strategic_evaluator = StrategicTransferEvaluator(self.data, self.scorer)

            # Track already recommended players to avoid duplicates
            recommended_player_ids = set()

            # Prioritize injured/unavailable players first
            unavailable_players = team_data[
                (team_data["is_available"] == 0) if "is_available" in team_data.columns else False
            ]
            if "chance_of_playing_next_round" in team_data.columns:
                doubtful_players = team_data[team_data["chance_of_playing_next_round"] < 75]
                unavailable_players = pd.concat(
                    [unavailable_players, doubtful_players]
                ).drop_duplicates()

            # Then find worst performers
            worst_performers = team_data.nsmallest(5, "model_score")

            # Combine unavailable and worst players (remove duplicates)
            players_to_consider = pd.concat(
                [unavailable_players, worst_performers]
            ).drop_duplicates()
            worst_players = players_to_consider.sort_values("model_score")

            for transfer_idx, (_, player_out) in enumerate(worst_players.iterrows()):
                # Find best replacement in same position within budget
                # IMPORTANT: Use actual available budget for this specific transfer
                available_budget = player_out["price"] + my_team.bank

                # Exclude already recommended players from search
                available_replacements = non_team_data[
                    ~non_team_data["player_id"].isin(recommended_player_ids)
                ]

                # Filter to affordable replacements only
                replacements = available_replacements[
                    (available_replacements["position"] == player_out["position"])
                    & (available_replacements["price"] <= available_budget)
                ]

                if replacements.empty:
                    continue

                # Get best replacement
                best_replacement = replacements.nlargest(1, "model_score").iloc[0]

                # Add to recommended players set
                recommended_player_ids.add(best_replacement["player_id"])

                # Calculate improvement
                score_improvement = best_replacement["model_score"] - player_out["model_score"]
                net_cost = best_replacement["price"] - player_out["price"]

                # Determine reason and priority
                reason = self._get_transfer_reason(player_out, best_replacement)
                priority = self._get_transfer_priority(player_out, best_replacement)

                # Strategic evaluation for hit decisions
                is_free_transfer = transfer_idx < my_team.free_transfers
                hit_evaluation = strategic_evaluator.evaluate_transfer_hit(
                    player_out, best_replacement, is_free_transfer
                )

                recommendations.append(
                    TransferRecommendation(
                        player_out={
                            "id": player_out["player_id"],
                            "name": player_out["player_name"],
                            "price": player_out["price"],
                            "score": player_out["model_score"],
                        },
                        player_in={
                            "id": best_replacement["player_id"],
                            "name": best_replacement["player_name"],
                            "price": best_replacement["price"],
                            "score": best_replacement["model_score"],
                        },
                        net_cost=net_cost,
                        score_improvement=score_improvement,
                        reason=reason,
                        priority=priority,
                        hit_evaluation=hit_evaluation,
                    )
                )

            # Sort by priority (1=urgent first) and score improvement
            # Priority 1 (urgent) should come first, so use ascending sort
            recommendations.sort(key=lambda x: (x.priority, -x.score_improvement))

            return recommendations[:num_transfers]

    def _get_transfer_reason(self, player_out: pd.Series, player_in: pd.Series) -> str:
        """Generate transfer reason"""
        reasons = []

        # Check injury/availability
        if player_out.get("is_available", 1) == 0:
            reasons.append("Injured/unavailable")

        # Check chance of playing
        chance_of_playing = player_out.get("chance_of_playing_next_round", 100)
        if chance_of_playing < 75:
            if chance_of_playing == 0:
                reasons.append("OUT - Not playing")
            elif chance_of_playing <= 25:
                reasons.append(f"DOUBTFUL - {chance_of_playing}% chance")
            elif chance_of_playing <= 50:
                reasons.append(f"50/50 - {chance_of_playing}% chance")
            else:
                reasons.append(f"75% doubt - {chance_of_playing}% chance")

        # Check form difference
        form_diff = player_in.get("form", 0) - player_out.get("form", 0)
        if form_diff > 3:
            reasons.append(f"Much better form (+{form_diff:.1f})")
        elif form_diff > 1:
            reasons.append(f"Better form (+{form_diff:.1f})")

        # Check fixture difficulty
        if "fixture_diff_next5" in player_out and "fixture_diff_next5" in player_in:
            fix_diff = player_out["fixture_diff_next5"] - player_in["fixture_diff_next5"]
            if fix_diff > 1:
                reasons.append(f"Better fixtures (diff: {fix_diff:.1f})")

        # Check goal probability
        if "prob_goal" in player_out and "prob_goal" in player_in:
            goal_diff = player_in["prob_goal"] - player_out["prob_goal"]
            if goal_diff > 0.2:
                reasons.append(f"Higher goal threat (+{goal_diff*100:.0f}%)")

        return " | ".join(reasons) if reasons else "Better overall value"

    def _get_transfer_priority(self, player_out: pd.Series, player_in: pd.Series) -> int:
        """Determine transfer priority (1=urgent, 2=recommended, 3=optional)"""

        # Urgent: Player unavailable, injured, or very poor form
        if player_out.get("is_available", 1) == 0:
            return 1

        # Check chance of playing (injured/doubtful)
        chance_of_playing = player_out.get("chance_of_playing_next_round", 100)
        if chance_of_playing < 75:  # Less than 75% chance is concerning
            return 1

        # Very poor form is also urgent
        if player_out.get("form", 5) < 1:
            return 1

        # Recommended: Significant score improvement or poor form
        score_diff = player_in.get("model_score", 0) - player_out.get("model_score", 0)
        if score_diff > 3:
            return 2

        # Poor but not terrible form
        if player_out.get("form", 5) < 3:
            return 2

        # Optional: Minor improvements
        return 3


class ChipAdvisor:
    """Advise on chip usage (Wildcard, Free Hit, etc.) using MILP optimization"""

    def __init__(self, data: pd.DataFrame, scorer=None):
        self.data = data
        self.scorer = scorer  # Needed for MILP wildcard optimization

    def get_all_chip_advice(
        self,
        my_team: MyTeam,
        analysis: dict,
        team_data: pd.DataFrame,
        use_milp: bool = True,
        use_strategic: bool = True,
    ) -> dict:
        """Get advice for all available chips using strategic FPL best practices

        Args:
            my_team: User's team
            analysis: Team analysis results
            team_data: DataFrame with team players
            use_milp: Whether to use MILP optimization
            use_strategic: Whether to use strategic DGW/BGW aware logic

        Returns:
            Dictionary with advice for each chip
        """
        # PRIORITY: Use strategic advisor for DGW/BGW aware recommendations
        if use_strategic:
            strategic_advisor = StrategicChipAdvisor(self.data, self.scorer)

            # Get captain analysis for Triple Captain evaluation
            analyzer = TeamAnalyzer(self.scorer, self.data)
            captain_analysis = analyzer._analyze_captain(team_data, my_team.captain, use_milp=False)

            # Get team issues for Wildcard evaluation
            team_issues = analysis.get("weaknesses", [])

            # Build available chips dict
            available_chips = {
                "triple_captain": my_team.triple_captain_available,
                "bench_boost": my_team.bench_boost_available,
                "wildcard": my_team.wildcard_available,
                "free_hit": my_team.free_hit_available,
            }

            # Get comprehensive strategic advice
            return strategic_advisor.get_comprehensive_chip_strategy(
                team_data=team_data,
                team_issues=team_issues,
                captain_analysis=captain_analysis,
                available_chips=available_chips,
            )

        # Fallback to MILP or original methods if strategic is disabled
        advice = {}

        if use_milp and self.scorer:
            # Use MILP chip advisor
            milp_advisor = MILPChipAdvisor(self.scorer, self.data)

            if my_team.wildcard_available:
                # Wildcard uses full team optimization
                advice["wildcard"] = self.should_use_wildcard(my_team, analysis)
                if advice["wildcard"].get("use") is True:
                    # Get optimal wildcard team
                    advice["wildcard"]["optimal_team"] = milp_advisor.optimize_wildcard_team()

            if my_team.bench_boost_available:
                # Use MILP to evaluate bench strength
                team_ids = team_data["player_id"].tolist()
                advice["bench_boost"] = milp_advisor.evaluate_bench_boost(team_ids)

            if my_team.triple_captain_available:
                # Use MILP to evaluate triple captain timing
                team_ids = team_data["player_id"].tolist()
                advice["triple_captain"] = milp_advisor.evaluate_triple_captain(team_ids)

            if my_team.free_hit_available:
                # Free hit still uses heuristic (could extend MILP for this)
                advice["free_hit"] = self.should_use_free_hit(my_team, analysis, team_data)
        else:
            # Fallback to original methods
            if my_team.wildcard_available:
                advice["wildcard"] = self.should_use_wildcard(my_team, analysis)

            if my_team.free_hit_available:
                advice["free_hit"] = self.should_use_free_hit(my_team, analysis, team_data)

            if my_team.bench_boost_available:
                advice["bench_boost"] = self.should_use_bench_boost(team_data)

            if my_team.triple_captain_available:
                advice["triple_captain"] = self.should_use_triple_captain(analysis)

        return advice

    def should_use_wildcard(self, my_team: MyTeam, analysis: dict) -> dict:
        """Determine if wildcard should be used

        Args:
            my_team: User's current team
            analysis: Team analysis results

        Returns:
            Advice dictionary
        """
        if not my_team.wildcard_available:
            return {"use": False, "reason": "Wildcard not available"}

        reasons_to_use = []
        score = 0

        # Check team value vs template
        if analysis.get("total_team_value", 100) < 95:
            reasons_to_use.append("Team value significantly below template")
            score += 2

        # Check number of weaknesses
        weaknesses = analysis.get("weaknesses", [])
        high_severity = sum(1 for w in weaknesses if w.get("severity") == "high")
        if high_severity >= 3:
            reasons_to_use.append(f"{high_severity} urgent issues in team")
            score += 3

        # Check fixture swing
        avg_difficulty = analysis.get("fixture_analysis", {}).get("average_difficulty_next5", 3)
        if avg_difficulty > 3.8:
            reasons_to_use.append("Very difficult fixtures ahead")
            score += 2

        # Decision
        if score >= 5:
            return {"use": True, "confidence": "high", "reasons": reasons_to_use}
        elif score >= 3:
            return {"use": "consider", "confidence": "medium", "reasons": reasons_to_use}
        else:
            return {"use": False, "confidence": "low", "reasons": ["Team is in reasonable shape"]}

    def should_use_free_hit(self, my_team: MyTeam, analysis: dict, team_data: pd.DataFrame) -> dict:
        """Determine if free hit should be used

        Free Hit is best for:
        - Blank gameweeks (many teams not playing)
        - Very difficult fixtures for most of your team
        - When many players are injured/unavailable temporarily
        """
        if not my_team.free_hit_available:
            return {"use": False, "reason": "Free Hit not available"}

        reasons = []
        score = 0

        # Check for multiple injuries/unavailable
        weaknesses = analysis.get("weaknesses", [])
        unavailable_count = sum(1 for w in weaknesses if w.get("type") == "unavailable")
        if unavailable_count >= 4:
            reasons.append(f"{unavailable_count} players unavailable")
            score += 3

        # Check fixture difficulty
        avg_difficulty = analysis.get("fixture_analysis", {}).get("average_difficulty_next5", 3)
        if avg_difficulty > 4.0:
            reasons.append("Extremely difficult fixtures")
            score += 2

        # Check if too many players from same team (blank gameweek indicator)
        team_distribution = analysis.get("team_distribution", {})
        max_from_team = max(team_distribution.values()) if team_distribution else 0
        if max_from_team >= 5:
            reasons.append(f"Too many players from one team ({max_from_team})")
            score += 2

        if score >= 4:
            return {"use": True, "confidence": "medium", "reasons": reasons}
        else:
            return {
                "use": False,
                "confidence": "low",
                "reasons": ["Save for blank/difficult gameweek"],
            }

    def should_use_bench_boost(self, team_data: pd.DataFrame) -> dict:
        """Determine if bench boost should be used

        Bench Boost is best when:
        - Bench players have easy fixtures
        - Double gameweek (players play twice)
        - All bench players are fit and starting
        """
        # Separate bench players
        if len(team_data) < 15:
            return {"use": False, "reason": "Incomplete team"}

        # Get bench players (marked as benched)
        bench_players = (
            team_data[team_data.get("is_benched", 0) == 1]
            if "is_benched" in team_data.columns
            else team_data.tail(4)
        )

        # If no bench players found, fall back to last 4 players
        if bench_players.empty or len(bench_players) < 4:
            bench_players = team_data.tail(4)

        # Check bench fixture difficulty
        avg_bench_difficulty = (
            bench_players["fixture_difficulty"].mean()
            if "fixture_difficulty" in bench_players.columns
            else 3
        )

        # Check bench availability
        bench_available = (
            (bench_players.get("is_available", 1) == 1).all()
            if "is_available" in bench_players.columns
            else True
        )

        # Calculate bench expected points
        if "expected_points" in bench_players.columns:
            bench_expected = bench_players["expected_points"].sum()
        elif "model_score" in bench_players.columns:
            bench_expected = bench_players["model_score"].sum()
        else:
            bench_expected = 12  # Default assumption

        reasons = []

        if avg_bench_difficulty <= 2.5:
            reasons.append(f"Easy bench fixtures (avg: {avg_bench_difficulty:.1f})")

        if not bench_available:
            return {"use": False, "confidence": "low", "reasons": ["Bench players unavailable"]}

        if bench_expected >= 20:  # Good bench boost potential
            reasons.append(f"High bench potential ({bench_expected:.0f} pts expected)")
            return {"use": True, "confidence": "high", "reasons": reasons}
        elif bench_expected >= 15:
            return {"use": "consider", "confidence": "medium", "reasons": reasons}
        else:
            return {"use": False, "confidence": "low", "reasons": ["Low bench scoring potential"]}

    def should_use_triple_captain(self, analysis: dict) -> dict:
        """Determine if triple captain should be used

        Triple Captain is best for:
        - Premium captain with very high expected points
        - Easy fixture for captain
        - Double gameweek for captain
        """
        captain_analysis = analysis.get("captain_analysis", {})

        if not captain_analysis or not captain_analysis.get("recommended_captain"):
            return {"use": False, "reason": "No clear captain choice"}

        top_captain = captain_analysis.get("recommended_captain", {})
        captain_score = top_captain.get("score", 0)

        reasons = []

        # Very high captain score threshold
        if captain_score >= 10:
            reasons.append(f"Exceptional captain choice (score: {captain_score:.1f})")
            return {
                "use": True,
                "confidence": "high",
                "reasons": reasons,
                "player": top_captain.get("player", "Unknown"),
                "expected_points": captain_score * 3,  # Triple captain multiplier
            }
        elif captain_score >= 8:
            reasons.append(f"Strong captain choice (score: {captain_score:.1f})")
            return {
                "use": "consider",
                "confidence": "medium",
                "reasons": reasons,
                "player": top_captain.get("player", "Unknown"),
                "expected_points": captain_score * 3,  # Triple captain multiplier
            }
        else:
            return {
                "use": False,
                "confidence": "low",
                "reasons": ["Save for premium captain opportunity"],
            }
