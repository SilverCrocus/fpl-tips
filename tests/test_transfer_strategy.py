#!/usr/bin/env python
"""
Test strategic transfer hit evaluator
"""
import subprocess

# Test with sample team IDs
team_ids = '67,470,261,291,348,508,610,82,242,381,382,427,64,624,249'

print("Testing Strategic Transfer Hit Evaluator")
print("=" * 60)
print("\nRunning my-team command with strategic transfer logic...")
print("\nExpected behavior:")
print("  - FREE transfers should not show hit evaluation")
print("  - -4 pt transfers should show strategic evaluation")
print("  - AVOID for low point gains (<4 pts)")
print("  - CONSIDER for marginal gains (4-6 pts)")
print("  - TAKE for strong gains (>6 pts)")
print("\n" + "=" * 60)

# Create input for the prompts (1 free transfer, bank: 2.8m, yes to all chips)
input_text = "1\n2.8\ny\ny\ny\ny\n"

# Run the command with 3 transfers (1 free + 2 hits)
result = subprocess.run(
    ['uv', 'run', 'fpl.py', 'my-team', '-p', team_ids, '-t', '3'],
    input=input_text,
    capture_output=True,
    text=True
)

output = result.stdout

print("\nChecking transfer recommendations...")
print("-" * 40)

# Look for transfer recommendations
if "Transfer Recommendations" in output:
    start_idx = output.index("Transfer Recommendations")
    transfer_section = output[start_idx:start_idx+3000]
    
    # Check for hit evaluations
    if "Hit Evaluation" in transfer_section:
        print("✅ Strategic hit evaluation found!")
        
        # Extract the evaluations
        lines = transfer_section.split('\n')
        transfer_num = 0
        for i, line in enumerate(lines):
            if "Transfer" in line and ("FREE" in line or "-4 pts" in line):
                transfer_num += 1
                print(f"\n{line}")
                
                # Look for hit evaluation in next few lines
                for j in range(i+1, min(i+20, len(lines))):
                    if "Hit Evaluation" in lines[j]:
                        # Print evaluation details
                        for k in range(j, min(j+10, len(lines))):
                            if lines[k].strip():
                                print(f"  {lines[k]}")
                            if "fixture swing" in lines[k].lower():
                                break
                        break
    else:
        print("⚠️ No hit evaluations found - checking if all transfers are free...")
        
        free_count = transfer_section.count("(FREE)")
        hit_count = transfer_section.count("(-4 pts)")
        
        print(f"  Free transfers: {free_count}")
        print(f"  Hit transfers: {hit_count}")
        
        if hit_count == 0:
            print("  ℹ️ All transfers are free - no hit evaluation needed")
        else:
            print("  ❌ ERROR: Hit transfers found but no evaluation shown")
else:
    print("❌ ERROR: No transfer recommendations section found")

print("\n" + "=" * 60)
print("Summary:")
print("-" * 40)

if "Hit Evaluation" in output:
    # Count different recommendations
    avoid_count = output.count("AVOID")
    consider_count = output.count("CONSIDER") 
    take_count = output.count("TAKE")
    
    print("✅ Strategic transfer evaluator is working!")
    print(f"   - AVOID recommendations: {avoid_count}")
    print(f"   - CONSIDER recommendations: {consider_count}")
    print(f"   - TAKE recommendations: {take_count}")
    
    if "Expected gain:" in output:
        print("✅ Shows expected point gains")
    if "🔥" in output:
        print("✅ Detects injured/suspended players")
    if "👑" in output:
        print("✅ Identifies captain candidates")
    if "📈" in output:
        print("✅ Considers fixture swings")
else:
    print("ℹ️ No hit evaluations shown (possibly all free transfers)")

# Show any errors
if result.stderr:
    print("\n⚠️ Errors encountered:")
    print(result.stderr)