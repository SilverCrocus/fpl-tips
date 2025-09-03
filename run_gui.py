#!/usr/bin/env python
"""
FPL GUI Launcher
Automatically launches the Streamlit GUI application
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app"""
    print("🚀 Launching FPL Team Builder GUI...")
    print("-" * 50)
    print("The app will open in your browser automatically.")
    print("If it doesn't, navigate to: http://localhost:8501")
    print("-" * 50)
    print("\nPress Ctrl+C to stop the application\n")
    
    # Check which app to run
    if '--advanced' in sys.argv:
        app_file = 'app_advanced.py'
        print("Running ADVANCED version with full features...")
    else:
        app_file = 'app.py'
        print("Running STANDARD version...")
        print("Tip: Use 'python run_gui.py --advanced' for more features\n")
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not installed!")
        print("Install with: pip install streamlit")
        sys.exit(1)
    
    # Check if data exists
    if not Path("data/fpl_data.db").exists():
        print("⚠️  Warning: No data found!")
        print("Run 'python fpl.py fetch-data' first to get player data\n")
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_file,
            '--theme.primaryColor=#38003c',
            '--theme.backgroundColor=#ffffff',
            '--theme.secondaryBackgroundColor=#f0f0f0',
            '--theme.textColor=#262730'
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Application stopped")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()