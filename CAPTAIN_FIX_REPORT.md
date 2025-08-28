# Captain Command Fix Report

## Problem
The `captain` command was failing with:
```
KeyError: 'chance_of_playing_next_round'
```

## Root Cause
The database schema doesn't include the `chance_of_playing_next_round` column, which was being referenced directly in the captain selection filter.

## Fixes Applied

### 1. Missing Column Handling
- Replaced `chance_of_playing_next_round` with existing `is_available` column
- Added fallback logic if columns don't exist

### 2. Duplicate Players Issue
- Added logic to get only the latest gameweek data for each player when no specific gameweek is specified
- This prevents Haaland appearing 3 times (once for each gameweek)

### 3. Safe Column Access
- Added checks for all potentially missing columns (`prob_goal`, `prob_assist`, `form`, etc.)
- Used `.get()` method with defaults for safe access

### 4. Pandas Warnings Fixed
- Used `.copy()` to avoid `SettingWithCopyWarning`
- Proper type conversion to avoid `FutureWarning` about downcasting

## Result
✅ Captain command now works correctly:
- Shows top 10 captain choices
- Haaland correctly identified as top pick with 54.3% goal probability
- No duplicates, warnings, or errors

## Usage
```bash
# Get captain picks for current gameweek
uv run fpl.py captain

# Get captain picks for specific gameweek
uv run fpl.py captain --gameweek 3
```

## Top Captain Recommendations (GW3)
1. **Haaland** (Man City) - 54.3% goal probability ⭐
2. **Richarlison** (Spurs) - 44.3% goal probability
3. **João Pedro** (Chelsea) - 42.0% goal probability

---

*Fixed: August 28, 2025*