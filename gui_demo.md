# FPL Team Builder GUI - Features Demo

## 🚀 Quick Start

```bash
# Standard version
python run_gui.py

# Advanced version with more features
python run_gui.py --advanced
```

## 📸 Features Overview

### 1. **Player Selection Interface**
- **Dropdown menus** to select players
- **Real-time filtering** by:
  - Team (Arsenal, Chelsea, Liverpool, etc.)
  - Price range (slider)
  - Minimum points
  - Form rating
  - Ownership percentage
- **Sort options** for easy player comparison
- **Player stats display** before adding

### 2. **Visual Team Display**
- **Formation view** showing your squad layout
- **Interactive pitch** visualization (Advanced version)
- **Color coding** for captain/vice-captain
- **Player cards** with key info:
  - Name and team
  - Price
  - Total points
  - Form rating

### 3. **Budget Management**
- **Live budget tracking** as you add/remove players
- **Remaining budget** displayed prominently
- **Over-budget warnings** in red
- **Price filtering** to only show affordable players

### 4. **Team Validation**
- **Automatic rule checking**:
  - ✅ Exactly 15 players
  - ✅ 2 GK, 5 DEF, 5 MID, 3 FWD
  - ✅ Max 3 players per team
  - ✅ Within budget
- **Visual indicators** for valid/invalid team
- **Clear error messages** for rule violations

### 5. **Advanced Features** (app_advanced.py)

#### Tabs Interface:
1. **Team Builder** - Main team selection
2. **Team Analysis** - Stats and charts
3. **Transfers** - Recommendations for changes
4. **Captain Selection** - Choose C and VC
5. **Performance** - Track team over time

#### Extra Features:
- **Formation selector** (3-4-3, 4-4-2, etc.)
- **Team distribution charts**
- **Top performers list**
- **Transfer suggestions** based on form
- **Captain recommendations**
- **Performance tracking graphs**

### 6. **Save/Load System**
- **Save teams** to JSON files
- **Load previous teams** from dropdown
- **Multiple team slots** for different strategies
- **Export player IDs** for CLI commands

## 🎮 How to Use

### Building a Team:
1. **Select position** (GK/DEF/MID/FWD)
2. **Apply filters** to narrow choices
3. **Choose player** from dropdown
4. **Click "Add to Team"**
5. **Repeat** until 15 players selected
6. **Save team** when complete

### Managing Your Team:
- **Remove players**: Click ❌ button on player card
- **Clear team**: Reset all selections
- **Check validation**: See green/red status
- **Track budget**: Monitor spending in real-time

### Using Filters:
- **Team filter**: Show only players from specific team
- **Price slider**: Set min/max price range
- **Points filter**: Minimum points threshold
- **Sort options**: Order by points, price, form, etc.
- **Search box**: Find players by name

## 🌟 Key Advantages Over CLI

| Feature | CLI | GUI |
|---------|-----|-----|
| Player Selection | Type IDs manually | Click from dropdown |
| Budget Tracking | Calculate manually | Live updates |
| Team Display | Text list | Visual formation |
| Filtering | Limited | Multiple filter options |
| Validation | Run command | Instant feedback |
| Stats & Charts | Text output | Interactive graphs |

## 💡 Tips

1. **Start with expensive players** - Add premium players first while budget allows
2. **Use filters extensively** - Narrow choices for faster selection
3. **Check team distribution** - Avoid too many players from one team
4. **Save multiple teams** - Compare different strategies
5. **Use Advanced version** for full analysis features

## 🔧 Troubleshooting

- **No data?** Run `python fpl.py fetch-data` first
- **Port in use?** Close other Streamlit apps or use different port
- **Slow loading?** Clear browser cache and reload
- **Missing players?** Check filters aren't too restrictive

## 🎯 Next Steps

After building your team in the GUI:

1. **Analyze**: Review team stats and charts
2. **Get recommendations**: Check transfer suggestions
3. **Export**: Get player IDs for CLI commands
4. **Track**: Monitor performance over gameweeks

Enjoy your new interactive FPL team builder! 🚀