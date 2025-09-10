# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management
- This project uses `uv` for package management (lightning-fast Python package installer)
- To run any Python file: `uv run filename.py` (not `python filename.py`)
- To add dependencies: `uv add package-name` (not `pip install`)
- To add dev dependencies: `uv add --dev package-name`
- To sync dependencies: `uv sync`

## Common Development Commands

### Running the Application
```bash
# GUI Application
uv run python run_gui.py                  # Standard GUI
uv run python run_gui.py --advanced      # Advanced GUI with analysis

# CLI Commands
uv run python -m src.main status         # Check system status
uv run python -m src.main fetch-data     # Fetch latest FPL and odds data
uv run python -m src.main recommend      # Get player recommendations
uv run python -m src.main build-team     # Interactive team builder
uv run python -m src.main backtest       # Run backtesting
```

### Testing
```bash
uv run pytest                             # Run all tests
uv run pytest tests/test_optimizer.py    # Run specific test file
uv run pytest -v                          # Verbose output
```

### Code Quality
```bash
uv run black src/                         # Format code
uv run ruff src/                          # Lint code
```

## Project Architecture

This is a Fantasy Premier League (FPL) recommendation system that combines:
- Real-time FPL API data
- TAB betting odds from The Odds API
- Machine learning models for player scoring
- Mixed Integer Linear Programming (MILP) for team optimization

### Key Components

1. **Data Collection** (`src/data/`)
   - `fpl_api.py`: Fetches official FPL data (players, fixtures, gameweeks)
   - `odds_api.py`: Gets player goal scorer odds from TAB bookmaker
   - `data_merger.py`: Unifies FPL and odds data
   - `name_matcher.py`: Fuzzy matching between FPL and bookmaker player names

2. **Models** (`src/models/`)
   - `rule_based_scorer.py`: Position-specific scoring formulas
   - `optimizer.py`: MILP optimization for team selection
   - `backtester.py`: Historical performance validation

3. **Strategy Components** (`src/`)
   - `team_optimizer.py`: Optimal team building with constraints
   - `transfer_strategy.py`: Transfer recommendations
   - `chip_strategy.py`: Timing for chips (Triple Captain, Bench Boost, etc.)
   - `my_team.py`: Team analysis and management

4. **User Interfaces**
   - `app.py`: Streamlit GUI for interactive team selection
   - `src/main.py`: CLI interface with Click commands

### Data Flow
1. FPL API → Player stats, fixtures, gameweek data
2. TAB Odds API → Goal scorer probabilities 
3. Data merger → Combined dataset with predictions
4. Scorer model → Player ratings
5. MILP optimizer → Optimal team selection
6. GUI/CLI → User recommendations

## Important Technical Details

### API Keys and Rate Limits
- The Odds API key is configured: `18405dde82249ca0a31950d7819767c7`
- Quota: 20,000 requests total
- Optimization: Only fetches TAB odds (saves 90% of API calls vs all bookmakers)

### Caching Strategy
- API responses cached in `cache/` directory with timestamps
- Cache TTL: 24 hours (configurable in `config.yaml`)
- SQLite database for persistent storage: `data/fpl_data.db`

### MILP Optimization Constraints
- Budget: £100M (configurable)
- Formation: Valid FPL formations (e.g., 3-5-2, 4-4-2)
- Max 3 players per team
- Position limits: 2 GK, 5 DEF, 5 MID, 3 FWD

### Position-Specific Scoring Weights
- **GK**: Clean sheets (2.0x), saves (0.5x)
- **DEF**: Clean sheets (2.5x), assists (1.5x)
- **MID**: Goals (3.0x), xG (2.5x), creativity
- **FWD**: Goals (3.5x), xG (3.0x), goal involvement

## File Organization
- `teams/`: Saved team configurations (JSON)
- `cache/`: API response caching
- `data/`: SQLite database
- `logs/`: Application logs
- `tests/`: Test files using pytest

## Environment Variables
Required in `.env`:
- `ODDS_API_KEY`: The Odds API key for fetching betting data
- Other optional configurations can override `config.yaml` settings
- Do not add default values as fallback actually try to fix the error dont have fallback logic or backwards compatibility