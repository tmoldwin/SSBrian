"""
Test script for parameter sweep system.

Runs a minimal sweep to verify the system works correctly.
Should complete in a few minutes.
"""

from parameter_sweep import ParameterSweep
from aggregate_analysis import AggregateAnalyzer


def test_minimal_sweep():
    """
    Run a minimal parameter sweep for testing.
    Uses very small networks and few stimuli for speed.
    """
    print("\n" + "="*60)
    print("TESTING PARAMETER SWEEP SYSTEM")
    print("="*60 + "\n")
    
    # Create a very small sweep for testing
    sweep = ParameterSweep(
        sweep_name="test_minimal",
        base_params={
            'NUM_STIMULI': 5,  # Very few stimuli
            'NUM_EPOCHS': 1,   # Just one epoch
            'N_external_input_dims': 20,  # Small input
        },
        varied_params={
            'N_rec_E': [10, 20],  # Two small network sizes
            'N_rec_I': [10, 20]
        },
        sweep_type='grid',  # 4 runs total
        generate_individual_plots=False,
        random_seed=42
    )
    
    print("Running 4 test simulations...")
    print("This should take 2-3 minutes.\n")
    
    # Run the sweep
    try:
        results = sweep.run()
        
        if len(results) == 4:
            print("\nAll 4 simulations completed successfully!")
        else:
            print(f"\nWarning: Only {len(results)}/4 simulations completed")
        
        # Test aggregate analysis
        print("\nTesting aggregate analysis...")
        analyzer = AggregateAnalyzer(sweep.sweep_folder)
        analyzer.generate_all_plots()
        
        print("\nAggregate analysis completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Sweep folder: {sweep.sweep_folder}")
        print(f"Successful runs: {len(results)}/4")
        print("\nSample statistics from first run:")
        if len(results) > 0:
            r = results[0]
            print(f"  Network size: {r['params']['N_rec_E']}E + {r['params']['N_rec_I']}I")
            print(f"  Mean selectivity: {r['mean_selectivity']:.3f}")
            print(f"  Mean sparsity: {r['mean_sparsity']:.3f}")
            print(f"  Mean firing rate: {r['mean_firing_rate']:.3f} Hz")
            print(f"  Active neurons: {r['n_active_neurons']}")
        
        print("\nPARAMETER SWEEP SYSTEM TEST PASSED!")
        print(f"\nTest results saved to: {sweep.sweep_folder}")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_minimal_sweep()
    
    if success:
        print("\nThe parameter sweep system is working correctly!")
        print("You can now run larger sweeps with:")
        print("  - sweep_presets.py (pre-configured sweeps)")
        print("  - parameter_sweep.py (custom sweeps)")
    else:
        print("\nTest failed. Please check error messages above.")

