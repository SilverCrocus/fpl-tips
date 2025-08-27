# FPL Transfer & Strategy Recommender 🚀

A data-driven Fantasy Premier League (FPL) tool that provides weekly transfer recommendations, captaincy choices, and chip timing strategies using machine learning and real-time betting odds.

## Features ✨

- **Real-time Data Integration**: Fetches latest FPL player stats and TAB betting odds
- **ML-Powered Predictions**: Rule-based scorer with position-specific formulas optimized through backtesting
- **Smart Recommendations**: Transfer suggestions, captain picks, and chip timing
- **TAB Odds Integration**: Uses only TAB bookmaker odds via The Odds API to save requests
- **Fast Package Management**: Uses `uv` for lightning-fast dependency management

## Quick Start 🏃‍♂️

### Prerequisites

1. Install `uv` (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone <your-repo>
cd fpltips
```

3. Install dependencies with `uv`:
```bash
uv sync
```

4. Set up your environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Your Odds API key is already configured: `18405dde82249ca0a31950d7819767c7`

## Usage with uv 🎮

With `uv`, you can run files directly:

### 1. Check System Status
```bash
uv run fpl.py status
```

### 2. Fetch Latest Data
```bash
# Fetch current gameweek data with real odds
uv run fpl.py fetch-data

# Fetch specific gameweek
uv run fpl.py fetch-data --gameweek 10
```

### 3. Get Player Recommendations
```bash
# Top 10 overall players
uv run fpl.py recommend

# Top midfielders under £10m
uv run fpl.py recommend --position MID --max-price 10.0 --top 5

# Top forwards
uv run fpl.py recommend --position FWD --top 15
```

### 4. Build Optimal Team
```bash
# Build best team with £100m budget
uv run fpl.py build-team

# Build team with custom budget
uv run fpl.py build-team --budget 98.5
```

### 5. Run Backtest
```bash
# Backtest on current season
uv run fpl.py backtest

# Backtest specific gameweek range
uv run fpl.py backtest --start-gw 5 --end-gw 20
```

### 6. Test Odds API Connection
```bash
uv run test_odds_api.py
```

## Development Commands 🛠️

### Add new dependencies:
```bash
uv add package-name
```

### Add dev dependencies:
```bash
uv add --dev package-name
```

### Run tests:
```bash
uv run pytest
```

### Format code:
```bash
uv run black src/
uv run ruff src/
```

### Start Jupyter notebook:
```bash
uv run jupyter notebook
```

## Project Structure 📁

```
fpltips/
├── src/
│   ├── data/
│   │   ├── fpl_api.py       # FPL API client
│   │   ├── odds_api.py      # The Odds API client
│   │   └── data_merger.py   # Data unification
│   ├── models/
│   │   ├── rule_based_scorer.py  # Position-specific scoring
│   │   └── backtester.py         # Historical validation
│   ├── recommender/          # Transfer & captain logic
│   └── main.py               # CLI interface
├── data/                     # SQLite database
├── cache/                    # API response caching
├── logs/                     # Application logs
├── .env                      # Your API keys
├── pyproject.toml           # uv project config
└── README.md                # You are here!
```

## API Usage Tracking 📊

The Odds API usage is tracked automatically:
- **Bookmaker**: TAB only (or bet365 if TAB unavailable)
- **Request Optimization**: Only fetching from one bookmaker saves API calls
- **Total quota**: 20,000 requests
- **Current remaining**: Check console output when fetching

Monitor usage in the console output when fetching odds data. Using only TAB reduces API usage by 90% compared to fetching all bookmakers.

## Scoring Model 🎯

The rule-based scorer uses position-specific weights:

### Goalkeepers (GK)
- Clean sheet probability (2.0x)
- Saves per game (0.5x)
- Team strength (Elo rating)

### Defenders (DEF)
- Clean sheet probability (2.5x)
- Goal/assist probability
- Expected assists (1.5x)

### Midfielders (MID)
- Goal probability (3.0x)
- Expected goals (2.5x)
- Creativity & threat indices

### Forwards (FWD)
- Goal probability (3.5x)
- Expected goals (3.0x)
- Goal involvement rate

## Advanced Features 🔬

### Custom Weights
Edit weights in `src/models/rule_based_scorer.py` or optimize using historical data:
```python
from src.models.rule_based_scorer import WeightOptimizer
optimizer = WeightOptimizer(historical_data)
optimal_weights = optimizer.optimize(iterations=100)
```

### Database Queries
Access the SQLite database directly:
```bash
sqlite3 data/fpl_data.db
.tables
SELECT * FROM player_gameweek_stats LIMIT 10;
```

## Troubleshooting 🔧

### Issue: "No data found"
**Solution**: Run `uv run fpl.py fetch-data` first

### Issue: API rate limit
**Solution**: Use `--use-mock-odds` flag or wait for rate limit reset

### Issue: Package installation fails
**Solution**: Make sure you're using Python 3.9+ and run `uv sync`

## Coming Soon 🚀

- [ ] LinUCB contextual bandit for adaptive learning
- [ ] Web interface with Streamlit
- [ ] Automated chip timing recommendations
- [ ] Weekly email notifications
- [ ] Historical performance tracking

## License 📄

MIT License - See LICENSE file for details

## Support 💬

For issues or questions, please open a GitHub issue or contact the maintainer.

---

## 🚀 Quick Command Reference

```bash
# Show all available commands
uv run help.py

# Check system status
uv run fpl.py status

# Fetch latest data
uv run fpl.py fetch-data

# Get recommendations
uv run fpl.py recommend

# Build optimal team
uv run fpl.py build-team

# Test Odds API
uv run test_odds_api.py
```

Remember: With `uv`, you just type `uv run filename.py` - no need for `python`!

---

Built with ❤️ for FPL managers who love data-driven decisions!