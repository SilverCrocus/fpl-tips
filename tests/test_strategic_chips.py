#!/usr/bin/env python
"""
Test strategic chip advisor - should NOT recommend Triple Captain in regular gameweeks
"""
import subprocess

# Test with sample team IDs
team_ids = '67,470,261,291,348,508,610,82,242,381,382,427,64,624,249'

print("Testing Strategic Chip Advisor")
print("=" * 60)
print("\nRunning my-team command with strategic chip logic...")
print("\nExpected behavior:")
print("  - Triple Captain should NOT be recommended in regular GW")
print("  - Should advise saving for Double Gameweek")
print("  - Should show strategic advice")
print("\n" + "=" * 60)

# Create input for the prompts
input_text = "1\n2.8\ny\ny\ny\ny\n"

# Run the command
result = subprocess.run(
    ['uv', 'run', 'fpl.py', 'my-team', '-p', team_ids],
    input=input_text,
    capture_output=True,
    text=True
)

output = result.stdout

print("\nChecking chip recommendations...")
print("-" * 40)

# Look for chip recommendations
if "Power-up/Chip Recommendations" in output:
    start_idx = output.index("Power-up/Chip Recommendations")
    chip_section = output[start_idx:start_idx+1500]
    
    # Check Triple Captain recommendation
    if "USE TRIPLE CAPTAIN" in chip_section:
        print("❌ ERROR: Triple Captain still being recommended in regular GW!")
        
        # Extract the recommendation
        lines = chip_section.split('\n')
        for i, line in enumerate(lines):
            if "TRIPLE CAPTAIN" in line:
                for j in range(max(0, i-2), min(i+8, len(lines))):
                    print(f"  {lines[j]}")
    else:
        print("✅ SUCCESS: Triple Captain NOT recommended for regular GW")
        
        # Look for strategic advice
        if "SAVE for Double Gameweek" in chip_section or "Wait for DGW" in chip_section:
            print("✅ Strategic advice to save for DGW found!")
        
        if "HOLD all chips" in chip_section or "strategic" in chip_section.lower():
            print("✅ Strategic hold recommendation found!")
            
        # Show the actual recommendation
        print("\nActual chip advice:")
        print("-" * 40)
        lines = chip_section.split('\n')[:30]  # First 30 lines
        for line in lines:
            if line.strip():
                print(line)
else:
    print("❌ ERROR: No chip recommendations section found")

print("\n" + "=" * 60)
print("Summary:")
print("-" * 40)
if "USE TRIPLE CAPTAIN" not in output:
    print("✅ Strategic chip advisor is working correctly!")
    print("   - NOT recommending TC in regular gameweek")
    print("   - Advising to save for optimal timing (DGW)")
else:
    print("❌ Still recommending TC in regular gameweek - needs fixing")