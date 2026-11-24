"""
Example: Custom Parameter Sweep

This is a template you can copy and modify for your own experiments.
Shows how to create a custom sweep, run it, and analyze results.
"""

from brian_experiment.feedforward.sweeps.parameter_sweep import ParameterSweep
from brian_experiment.shared.analysis.aggregate_analysis import AggregateAnalyzer


def my_custom_sweep():
    """
    Example: Investigate how selectivity changes with network size and input strength.
    
    Hypothesis: Larger networks with stronger inputs will have higher selectivity.
    """
    
    sweep = ParameterSweep(
        # Give your sweep a descriptive name
        sweep_name="network_size_vs_input_strength",
        
        # Base parameters (fixed across all runs)
        base_params={
            'NUM_STIMULI': 50,      # Moderate number of stimuli
            'NUM_EPOCHS': 2,         # Two epochs to see some dynamics
            'N_external_input_dims': 100,  # Fixed input dimension
            'stimulus_alpha': 0.5,   # Medium sparsity
        },
        
        # Parameters to vary
        varied_params={
            'N_rec_E': [50, 100, 150, 200],  # Network size
            'N_rec_I': [50, 100, 150, 200],  # Keep E and I equal
            'input_amplitude_scale': [0.5, 1.0, 2.0]  # Input strength
        },
        
        # Sweep type
        sweep_type='grid',  # All combinations: 4 sizes × 3 input strengths = 12 runs
        
        # Options
        generate_individual_plots=False,  # Skip individual plots for speed
        random_seed=42  # For reproducibility
    )
    
    return sweep


def run_and_analyze():
    """Run the sweep and perform analysis"""
    
    # Create the sweep
    sweep = my_custom_sweep()
    
    print(f"\nThis sweep will run {len(sweep.param_combinations)} simulations")
    print(f"Estimated time: ~{len(sweep.param_combinations) * 2} minutes")
    
    # Confirm before running
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Run all simulations
    print("\nRunning simulations...")
    results = sweep.run()
    
    # Analyze results
    print("\nGenerating aggregate analysis plots...")
    analyzer = AggregateAnalyzer(sweep.sweep_folder)
    analyzer.generate_all_plots()
    
    # Custom analysis: Check our hypothesis
    print("\n" + "="*60)
    print("HYPOTHESIS TEST RESULTS")
    print("="*60)
    
    # Group results by network size
    import numpy as np
    network_sizes = sorted(set([r['params']['N_rec_E'] for r in results]))
    
    print("\nMean selectivity by network size:")
    for size in network_sizes:
        size_results = [r for r in results if r['params']['N_rec_E'] == size]
        mean_sel = np.mean([r['mean_selectivity'] for r in size_results])
        std_sel = np.std([r['mean_selectivity'] for r in size_results])
        print(f"  {size:3d} neurons: {mean_sel:.3f} ± {std_sel:.3f}")
    
    # Group by input strength
    input_scales = sorted(set([r['params']['input_amplitude_scale'] for r in results]))
    
    print("\nMean selectivity by input strength:")
    for scale in input_scales:
        scale_results = [r for r in results if r['params']['input_amplitude_scale'] == scale]
        mean_sel = np.mean([r['mean_selectivity'] for r in scale_results])
        std_sel = np.std([r['mean_selectivity'] for r in scale_results])
        print(f"  {scale:.1f}x: {mean_sel:.3f} ± {std_sel:.3f}")
    
    print("\n" + "="*60)
    print(f"\nResults saved to: {sweep.sweep_folder}")
    print(f"Plots saved to: {analyzer.analysis_folder}")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    To run this example:
    
    1. Open this file
    2. Modify the parameters in my_custom_sweep() to test your hypothesis
    3. Run: python example_custom_sweep.py
    4. Check the generated plots in Sweeps/<sweep_name>/AggregateAnalysis/
    
    Tips:
    - Start with small parameter ranges to test
    - Use NUM_STIMULI=20 for quick testing
    - Increase to NUM_STIMULI=100 for publication-quality results
    - Use 'list' sweep_type for paired parameters
    - Use 'grid' sweep_type for all combinations
    """
    run_and_analyze()

