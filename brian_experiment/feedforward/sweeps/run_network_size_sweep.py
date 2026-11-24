"""
Network Size Sweep: 100 to 500 neurons

Runs parameter sweep varying network size from 100 to 500 neurons.
Generates analysis plots showing how selectivity scales with network size.
"""

from brian_experiment.feedforward.sweeps.parameter_sweep import ParameterSweep
from brian_experiment.shared.analysis.aggregate_analysis import AggregateAnalyzer

def main():
    """Run network size sweep from 100 to 500 neurons"""
    
    print("\n" + "="*60)
    print("NETWORK SIZE SWEEP: 100 TO 500 NEURONS")
    print("="*60)
    print("\nThis will run 25 simulations (5 parameter combinations × 5 runs each):")
    print("  - Network sizes: 100, 200, 300, 400, 500 neurons")
    print("  - E and I neurons kept equal")
    print("  - 50 stimuli per run")
    print("  - 2 epochs per run")
    print("  - 5 runs per parameter combination (for averaging)")
    print("\nEstimated time: ~100-150 minutes")
    print("="*60 + "\n")
    
    # Create parameter sweep
    sweep = ParameterSweep(
        sweep_name="network_size_100_to_500",
        base_params={
            'NUM_STIMULI': 50,
            'NUM_EPOCHS': 2,
            'N_external_input_dims': 100,
            'stimulus_alpha': 0.5,
            'disable_recurrent': False,
        },
        varied_params={
            'N_rec_E': [100, 200, 300, 400, 500],
            'N_rec_I': [100, 200, 300, 400, 500]
        },
        sweep_type='list',  # Paired: (100,100), (200,200), etc.
        generate_individual_plots=False,
        random_seed=42,
        n_runs_per_param=5  # Run each parameter combination 5 times for averaging
    )
    
    # Run the sweep
    print("Starting simulations...\n")
    results = sweep.run()
    
    if len(results) == 0:
        print("\nNo simulations completed successfully.")
        print("Check error messages above.")
        return
    
    print(f"\n{len(results)}/25 simulations completed successfully!")
    
    # Generate aggregate analysis
    print("\nGenerating aggregate analysis plots...")
    analyzer = AggregateAnalyzer(sweep.sweep_folder)
    analyzer.generate_all_plots()
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print("\nSelectivity by Network Size:")
    print("-" * 60)
    
    results_sorted = sorted(results, key=lambda x: x['params']['N_rec_E'] + x['params']['N_rec_I'])
    for r in results_sorted:
        size = r['params']['N_rec_E'] + r['params']['N_rec_I']
        sel = r['mean_selectivity']
        std = r['std_selectivity']
        print(f"  {size:4d} neurons: {sel:.4f} ± {std:.4f}")
    
    print("\n" + "="*60)
    print("PLOTS SAVED TO:")
    print(f"  {analyzer.analysis_folder}/")
    print("\nKey plots:")
    print("  - summary_dashboard.png")
    print("  - mean_selectivity_vs_N_rec_E.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

