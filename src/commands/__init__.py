"""
FPL CLI Commands Package
Split into logical command groups for better organization
"""

from .analysis_commands import captain, differentials, recommend, search
from .backtest_commands import backtest
from .data_commands import fetch_data, status
from .team_commands import build_team, load_team, my_team

__all__ = [
    "fetch_data",
    "status",
    "build_team",
    "my_team",
    "load_team",
    "recommend",
    "differentials",
    "captain",
    "search",
    "backtest",
]
