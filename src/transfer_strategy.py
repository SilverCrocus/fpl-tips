"""
Strategic Transfer Hit Evaluator
Implements FPL best practices for evaluating -4 point hits
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class TransferStrategy:
    """Strategic rules for transfer hits"""

    # Break-even thresholds
    BREAK_EVEN_THRESHOLD = 4.0  # Minimum point gain to justify hit
    CONFIDENCE_THRESHOLD = 6.0  # Strong confidence in hit payoff

    # Strategic modifiers
    CAPTAIN_MULTIPLIER = 2.0  # Captaincy potential doubles impact
    DGW_MULTIPLIER = 2.0  # Double gameweek doubles points
    INJURY_URGENCY = 10.0  # High urgency for injured/suspended players

    # Horizon for payback calculation
    SHORT_TERM_WEEKS = 2  # Hits should pay back within 2 GWs
    FIXTURE_SWING_WEEKS = 4  # Consider fixture swings over 4 GWs

    # Premium player thresholds
    PREMIUM_PRICE = 11.0  # Players above this are premium
    CAPTAIN_SCORE_THRESHOLD = 8.0  # Expected score for captaincy candidate


class StrategicTransferEvaluator:
    """
    Strategic transfer evaluator that considers:
    - Expected point gains vs -4 cost
    - Short-term payback periods
    - Strategic factors (injuries, captaincy, fixtures)
    - FPL community best practices
    """

    def __init__(self, data: pd.DataFrame, scorer=None):
        self.data = data
        self.scorer = scorer
        self.strategy = TransferStrategy()

    def convert_model_score_to_expected_points(self, model_score: float, position: str) -> float:
        """
        Convert model score to expected FPL points

        Model scores are relative rankings (0-100 scale)
        Expected points should be realistic FPL predictions
        """
        # Position-specific conversion factors based on typical FPL scoring
        conversion_factors = {
            "GK": 0.04,  # GKs typically score 2-6 points
            "DEF": 0.05,  # Defenders typically score 2-8 points
            "MID": 0.06,  # Midfielders typically score 2-12 points
            "FWD": 0.055,  # Forwards typically score 2-10 points
        }

        base_points = {
            "GK": 2.0,  # Playing points + average saves
            "DEF": 2.0,  # Playing points
            "MID": 2.0,  # Playing points
            "FWD": 2.0,  # Playing points
        }

        factor = conversion_factors.get(position, 0.05)
        base = base_points.get(position, 2.0)

        # Convert model score to expected points
        expected_points = base + (model_score * factor)

        # Cap at realistic maximums
        max_points = {"GK": 12, "DEF": 15, "MID": 20, "FWD": 18}
        expected_points = min(expected_points, max_points.get(position, 15))

        return expected_points

    def calculate_point_gain(
        self, player_out: pd.Series, player_in: pd.Series, horizon_weeks: int = None
    ) -> float:
        """
        Calculate expected point gain from transfer

        Args:
            player_out: Player being transferred out
            player_in: Player being transferred in
            horizon_weeks: Number of weeks to consider

        Returns:
            Expected point gain over horizon
        """
        if horizon_weeks is None:
            horizon_weeks = self.strategy.SHORT_TERM_WEEKS

        # Get expected points for each player
        # Check if we have dictionary data (from MILP) or Series data
        if isinstance(player_out, dict):
            out_score = player_out.get("score", 0)
            out_position = player_out.get("position", "MID")
        else:
            out_score = player_out.get("model_score", player_out.get("score", 0))
            out_position = player_out.get("position", "MID")

        if isinstance(player_in, dict):
            in_score = player_in.get("score", 0)
            in_position = player_in.get("position", "MID")
        else:
            in_score = player_in.get("model_score", player_in.get("score", 0))
            in_position = player_in.get("position", "MID")

        # If scores are very low or zero, use a minimum baseline
        if out_score == 0:
            out_score = 20.0  # Minimum expected score
        if in_score == 0:
            in_score = 30.0  # Slightly higher for replacement

        out_points = self.convert_model_score_to_expected_points(out_score, out_position)
        in_points = self.convert_model_score_to_expected_points(in_score, in_position)

        # Check for injuries/suspensions (0 expected points)
        status = player_out.get("status") if not isinstance(player_out, dict) else None
        if status in ["injured", "suspended", "unavailable"]:
            out_points = 0

        # Also check availability flags
        if not isinstance(player_out, dict):
            if player_out.get("is_available") == 0:
                out_points = 0
            elif player_out.get("chance_of_playing_next_round", 100) < 50:
                out_points *= 0.5  # Reduce expected points for doubtful players

        # Calculate gain over horizon
        point_gain = (in_points - out_points) * horizon_weeks

        return point_gain

    def evaluate_strategic_modifiers(self, player_out: pd.Series, player_in: pd.Series) -> dict:
        """
        Evaluate strategic factors that modify transfer value

        Returns dict with strategic modifiers and their impact
        """
        modifiers = {
            "is_fire": False,
            "is_captain_candidate": False,
            "has_fixture_swing": False,
            "is_dgw_player": False,
            "price_dynamics": 0,
            "total_modifier": 0,
        }

        # Handle both dict and Series data
        # 1. Fire check - injured/suspended/dropped
        if not isinstance(player_out, dict):
            status = player_out.get("status")
            is_available = player_out.get("is_available", 1)
            chance_playing = player_out.get("chance_of_playing_next_round", 100)

            if status in ["injured", "suspended", "unavailable"] or is_available == 0:
                modifiers["is_fire"] = True
                modifiers["fire_urgency"] = self.strategy.INJURY_URGENCY
                modifiers["total_modifier"] += self.strategy.INJURY_URGENCY
            elif chance_playing < 50:
                modifiers["is_fire"] = True
                modifiers["fire_urgency"] = self.strategy.INJURY_URGENCY * 0.5
                modifiers["total_modifier"] += self.strategy.INJURY_URGENCY * 0.5

        # 2. Captaincy potential check
        if isinstance(player_in, dict):
            in_price = player_in.get("price", 0)
        else:
            in_price = player_in.get("price", 0)

        # Determine if player is premium captain material
        if in_price >= self.strategy.PREMIUM_PRICE:
            modifiers["is_captain_candidate"] = True
            modifiers["captain_boost"] = self.strategy.CAPTAIN_MULTIPLIER
            modifiers["total_modifier"] += 2.0  # Bonus for captain potential

        # 3. Fixture swing check
        if not isinstance(player_out, dict) and not isinstance(player_in, dict):
            out_fixture_difficulty = player_out.get("fixture_difficulty", 3)
            in_fixture_difficulty = player_in.get("fixture_difficulty", 3)

            # Significant fixture swing (e.g., difficulty 5 to 2)
            fixture_diff = out_fixture_difficulty - in_fixture_difficulty
            if fixture_diff >= 2:
                modifiers["has_fixture_swing"] = True
                modifiers["fixture_swing_value"] = fixture_diff
                modifiers["total_modifier"] += fixture_diff * 0.5

        # 4. Double gameweek check
        if not isinstance(player_in, dict):
            if player_in.get("is_dgw", False):
                modifiers["is_dgw_player"] = True
                modifiers["dgw_multiplier"] = self.strategy.DGW_MULTIPLIER
                modifiers["total_modifier"] += 3.0  # Strong DGW bonus

        # 5. Price dynamics (secondary consideration)
        if not isinstance(player_in, dict) and not isinstance(player_out, dict):
            price_rise_probability = player_in.get("price_rise_probability", 0)
            price_fall_probability = player_out.get("price_fall_probability", 0)

            if price_rise_probability > 0.7 or price_fall_probability > 0.7:
                modifiers["price_dynamics"] = 1.0
                modifiers["total_modifier"] += 0.5

        return modifiers

    def evaluate_transfer_hit(
        self, player_out: pd.Series, player_in: pd.Series, is_free_transfer: bool = False
    ) -> dict:
        """
        Comprehensive evaluation of whether a -4 hit is worth it

        Returns:
            Dict with recommendation, confidence, and reasoning
        """
        # Skip evaluation for free transfers
        if is_free_transfer:
            return {
                "recommendation": "TAKE",
                "confidence": "FREE",
                "point_gain": 0,
                "reasoning": "Free transfer - no hit required",
                "strategic_advice": "Free transfer available",
            }

        # Calculate base point gain
        point_gain = self.calculate_point_gain(player_out, player_in)

        # Get strategic modifiers
        modifiers = self.evaluate_strategic_modifiers(player_out, player_in)

        # Adjust point gain with modifiers
        adjusted_gain = point_gain + modifiers["total_modifier"]

        # Build reasoning
        reasons = []

        # Rule 1: Break-Even Check (The Gatekeeper)
        if adjusted_gain < self.strategy.BREAK_EVEN_THRESHOLD:
            recommendation = "AVOID"
            confidence = "high"
            reasons.append(f"❌ Expected gain ({adjusted_gain:.1f}) below break-even threshold")
            strategic_advice = "🚫 Hit not justified - save transfer"

        # Rule 2: Confidence Check
        elif adjusted_gain < self.strategy.CONFIDENCE_THRESHOLD:
            recommendation = "CONSIDER"
            confidence = "medium"
            reasons.append(f"⚠️ Marginal gain ({adjusted_gain:.1f}) - risky hit")
            strategic_advice = "🤔 Optional - only if no better use of transfer"

        # Rule 3: Strong Recommendation
        else:
            recommendation = "TAKE"
            confidence = "high"
            reasons.append(f"✅ Strong gain ({adjusted_gain:.1f}) justifies hit")
            strategic_advice = "👍 Hit recommended - clear upgrade"

        # Add modifier reasoning
        if modifiers["is_fire"]:
            reasons.append("🔥 Replacing injured/suspended player")
            if recommendation == "CONSIDER":
                recommendation = "TAKE"  # Upgrade for fires

        if modifiers["is_captain_candidate"]:
            reasons.append(f"👑 Captain candidate (price: £{player_in['price']:.1f}m)")

        if modifiers["has_fixture_swing"]:
            reasons.append(
                f"📈 Favorable fixture swing (diff: {modifiers['fixture_swing_value']:.1f})"
            )

        if modifiers["is_dgw_player"]:
            reasons.append("🎯 Double Gameweek player")
            if recommendation != "AVOID":
                recommendation = "TAKE"  # Strong preference for DGW

        if modifiers["price_dynamics"] > 0:
            reasons.append("💰 Favorable price movements expected")

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "point_gain": point_gain,
            "adjusted_gain": adjusted_gain,
            "modifiers": modifiers,
            "reasons": reasons,
            "strategic_advice": strategic_advice,
            "payback_weeks": self._calculate_payback_period(point_gain),
        }

    def _calculate_payback_period(self, point_gain_per_week: float) -> float:
        """Calculate how many weeks to pay back the -4 hit"""
        if point_gain_per_week <= 0:
            return float("inf")  # Never pays back

        # -4 hit needs to be recovered
        payback_weeks = 4.0 / point_gain_per_week
        return round(payback_weeks, 1)

    def evaluate_multiple_hits(
        self, transfers: list[tuple[pd.Series, pd.Series]], free_transfers: int = 1
    ) -> dict:
        """
        Evaluate multiple transfer hits together

        Args:
            transfers: List of (player_out, player_in) tuples
            free_transfers: Number of free transfers available

        Returns:
            Comprehensive evaluation of transfer strategy
        """
        evaluations = []
        total_hits = max(0, len(transfers) - free_transfers)
        total_cost = total_hits * 4
        total_gain = 0

        for i, (out_player, in_player) in enumerate(transfers):
            is_free = i < free_transfers
            eval_result = self.evaluate_transfer_hit(out_player, in_player, is_free)
            evaluations.append(eval_result)
            total_gain += eval_result.get("adjusted_gain", 0)

        # Overall recommendation
        if total_gain < total_cost:
            overall_recommendation = "AVOID_HITS"
            overall_advice = f"🚫 Total cost (-{total_cost}) exceeds gain ({total_gain:.1f})"
        elif total_gain < total_cost + 4:
            overall_recommendation = "LIMIT_HITS"
            overall_advice = "⚠️ Marginal benefit - consider fewer hits"
        else:
            overall_recommendation = "PROCEED"
            overall_advice = f"✅ Hits justified - net gain of {total_gain - total_cost:.1f} pts"

        return {
            "transfers": evaluations,
            "total_hits": total_hits,
            "total_cost": total_cost,
            "total_gain": total_gain,
            "net_benefit": total_gain - total_cost,
            "overall_recommendation": overall_recommendation,
            "overall_advice": overall_advice,
        }

    def get_hit_recommendation_summary(self, evaluation: dict) -> str:
        """
        Generate user-friendly summary of hit recommendation
        """
        rec = evaluation["recommendation"]
        confidence = evaluation["confidence"]
        gain = evaluation["adjusted_gain"]
        advice = evaluation["strategic_advice"]

        if rec == "AVOID":
            emoji = "🚫"
            action = "AVOID HIT"
        elif rec == "CONSIDER":
            emoji = "🤔"
            action = "CONSIDER"
        else:  # TAKE
            emoji = "✅"
            action = "TAKE HIT"

        summary = f"{emoji} {action} (Confidence: {confidence})\n"
        summary += f"   Expected gain: {gain:.1f} pts\n"
        summary += f"   {advice}"

        return summary
