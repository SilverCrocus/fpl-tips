"""
Centralized captain scoring module for consistent scoring across the application
"""
import pandas as pd


def calculate_captain_score(player, consider_effective_ownership: bool = True) -> float:
    """Calculate captain-specific score for a player

    This function provides a consistent captain scoring method used throughout the application.
    It considers multiple factors beyond base model score to determine captain suitability.

    Args:
        player: Player data (Series or dict)
        consider_effective_ownership: Whether to factor in ownership differential

    Returns:
        Captain score for the player
    """
    score = 0

    # Base expected points (captain gets double)
    base_points = player.get("model_score", 0) * 2
    score += base_points

    # Consider form heavily (recent performance indicator)
    if "form" in player:
        score += float(player["form"]) * 1.5

    # Goal probability is crucial for captain (main source of points)
    if "prob_goal" in player and pd.notna(player.get("prob_goal")):
        score += float(player["prob_goal"]) * 10

    # Fixture difficulty (lower is better, scale 1-5)
    if "fixture_difficulty" in player and pd.notna(player.get("fixture_difficulty")):
        score += (5 - float(player["fixture_difficulty"])) * 2

    # Penalty for injury doubt
    if "chance_of_playing_next_round" in player:
        chance = float(player.get("chance_of_playing_next_round", 100))
        if chance < 100:
            score *= chance / 100

    # Effective ownership consideration for differential captains
    if consider_effective_ownership and "selected_by_percent" in player:
        ownership = float(player.get("selected_by_percent", 0))
        if ownership > 50:  # Very high ownership
            score *= 0.95  # Small penalty for template picks
        elif ownership < 10:  # Differential captain
            score *= 1.1  # Bonus for differential captain

    return score


def get_captain_recommendations(
    team_data: pd.DataFrame,
    top_n: int = 3,
    consider_effective_ownership: bool = True
) -> list[dict]:
    """Get top captain recommendations for a team

    Args:
        team_data: DataFrame with team players
        top_n: Number of captain options to return
        consider_effective_ownership: Whether to consider ownership in scoring

    Returns:
        List of captain recommendations with scores
    """
    # Filter out goalkeepers (they should never be captain)
    candidates = team_data[team_data["position"] != "GK"].copy()

    if candidates.empty:
        return []

    # Calculate captain scores for all candidates
    candidates["captain_score"] = candidates.apply(
        lambda row: calculate_captain_score(row, consider_effective_ownership),
        axis=1
    )

    # Get top N candidates
    top_candidates = candidates.nlargest(top_n, "captain_score")

    # Format results
    recommendations = []
    for _, player in top_candidates.iterrows():
        recommendations.append({
            "player_id": player.get("player_id"),
            "player_name": player.get("player_name"),
            "team": player.get("team_name", player.get("team", "Unknown")),
            "position": player.get("position"),
            "captain_score": player["captain_score"],
            "base_score": player.get("model_score", 0),
            "form": player.get("form", 0),
            "fixture_difficulty": player.get("fixture_difficulty", 3),
            "prob_goal": player.get("prob_goal", 0),
            "ownership": player.get("selected_by_percent", 0)
        })

    return recommendations