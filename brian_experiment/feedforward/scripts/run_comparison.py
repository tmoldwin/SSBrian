#!/usr/bin/env python3
"""
Comparison script to run both recurrent and feedforward simulations.
This script demonstrates how to easily switch between the two modes.
"""

import os
import sys
import subprocess
import time

def run_simulation(mode="recurrent"):
    """
    Run the direct input simulation in the specified mode.
    
    Args:
        mode (str): Either "recurrent" or "feedforward"
    """
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING {mode.upper()} SIMULATION")
    print(f"{'='*60}")
    
    # Read the main file
    with open('brian_experiment/feedforward/single_run.py', 'r') as f:
        content = f.read()
    
    # Modify the DISABLE_RECURRENT_CONNECTIONS setting
    if mode == "feedforward":
        content = content.replace('DISABLE_RECURRENT_CONNECTIONS = False', 'DISABLE_RECURRENT_CONNECTIONS = True')
        content = content.replace('DISABLE_RECURRENT_CONNECTIONS = True', 'DISABLE_RECURRENT_CONNECTIONS = True')
    else:  # recurrent
        content = content.replace('DISABLE_RECURRENT_CONNECTIONS = True', 'DISABLE_RECURRENT_CONNECTIONS = False')
        content = content.replace('DISABLE_RECURRENT_CONNECTIONS = False', 'DISABLE_RECURRENT_CONNECTIONS = False')
    
    # Write back the modified content
    with open('brian_experiment/feedforward/single_run.py', 'w') as f:
        f.write(content)
    
    # Run the simulation
    start_time = time.time()
    result = subprocess.run([sys.executable, 'brian_experiment/feedforward/single_run.py'], 
                          capture_output=True, text=True)
    end_time = time.time()
    
    print(f"[TIME] Simulation completed in {end_time - start_time:.1f} seconds")
    
    if result.returncode == 0:
        print("✅ Simulation completed successfully!")
        # Extract key metrics from output
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if "Average Firing Rates:" in line or "Final network sparsity:" in line or "Final network selectivity:" in line:
                print(f"   {line.strip()}")
    else:
        print("❌ Simulation failed!")
        print("Error output:")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    """Run both recurrent and feedforward simulations for comparison."""
    print("🧠 Direct Input Network Comparison: Recurrent vs Feedforward")
    print("="*60)
    
    # Run recurrent simulation
    recurrent_success = run_simulation("recurrent")
    
    # Wait a moment between simulations
    time.sleep(2)
    
    # Run feedforward simulation
    feedforward_success = run_simulation("feedforward")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Recurrent simulation: {'✅ SUCCESS' if recurrent_success else '❌ FAILED'}")
    print(f"Feedforward simulation: {'✅ SUCCESS' if feedforward_success else '❌ FAILED'}")
    
    if recurrent_success and feedforward_success:
        print("\n🎉 Both simulations completed successfully!")
        print("📁 Check the Figures/DI_Sims/ directory for results:")
        print("   - direct_input_run_stim100/ (recurrent)")
        print("   - direct_input_FF_* (feedforward)")
        print("\n💡 Key differences to observe:")
        print("   - Firing rates (feedforward typically higher)")
        print("   - Sparsity (feedforward typically lower)")
        print("   - Selectivity (feedforward typically lower)")
        print("   - Network dynamics (recurrent shows more complex patterns)")

if __name__ == "__main__":
    main() 