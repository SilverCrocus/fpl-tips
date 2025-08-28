# Backtest Issue Analysis & Solution

## Problem Summary
Your backtest is showing -4 points and £0.2 team value because:
1. **Missing Historical Data**: Only GW1-2 data exists, no GW3-38
2. **NULL Points**: The `gameweek_points` column is NULL (games haven't been played yet)
3. **Display Bug**: Team value showing bank remainder instead of total value

## Root Cause
You're fetching **forecast data** (for upcoming games) not **historical data** (with actual points).

### Database Status:
```sql
-- Your current data:
Gameweeks available: 1, 2
gameweek_points: NULL for all players
Price data: Valid (£4.0-14.5 range)
Players: 707 total
```

## Solutions

### Solution 1: Use for Predictions (Recommended)
Since you have current season data with betting odds, use it for predictions:

```bash
# Get recommendations for upcoming gameweek
uv run fpl.py recommend --position FWD --top 10

# Build optimal team
uv run fpl.py build-team --budget 100

# Get captain picks
uv run fpl.py captain

# Find differentials
uv run fpl.py differentials
```

### Solution 2: Add Historical Data Fetcher
To properly backtest, you'd need to:

1. Create a historical data fetcher that gets past season data WITH points
2. The FPL API provides this at: `/bootstrap-static` for past seasons
3. Store it with actual `gameweek_points` populated

Example implementation:
```python
# In src/data/fpl_api.py
async def fetch_historical_season(self, season_id: str = "2023-24"):
    """Fetch completed season data with actual points"""
    # Past season endpoints available at:
    # https://fantasy.premierleague.com/api/event/{gw}/live/
    # Contains actual points scored
    pass
```

### Solution 3: Quick Fix Applied ✅
I've fixed the display issues:
- Safe division handling (no crash on empty gameweek_points)
- Clear warning message when historical data is missing
- Proper labeling of team value vs remaining budget

## What You Should Do Now

### For Immediate Use:
```bash
# Fetch latest data
uv run fpl.py fetch-data --gameweek 3

# Use the system for predictions
uv run fpl.py recommend
uv run fpl.py build-team
uv run fpl.py captain
```

### For Backtesting:
Wait until more gameweeks are played, or implement historical data fetching.

## Understanding Your System's Strengths

Your system is designed for **forward-looking predictions** using:
- Real-time FPL stats
- Actual bookmaker odds (not estimates!)
- Smart scoring algorithm

It's most valuable for:
1. **Weekly transfer decisions** - Who to bring in/out
2. **Captain selection** - Based on real goal probabilities
3. **Finding differentials** - Low ownership, high potential

## Current Data Pipeline
```
FPL API → Current player stats
   ↓
Odds API → Real goal probabilities  
   ↓
Merger → Combined dataset
   ↓
Scorer → Predictions (not backtesting)
```

## Next Steps
1. **Use for predictions** - That's what it's built for
2. **Track performance** - Compare your recommendations to actual outcomes weekly
3. **Consider adding** - Historical fetcher if you really need backtesting

---

*Remember: Your system uses REAL betting odds, which is its unique value. Focus on using it for upcoming gameweeks rather than historical analysis.*