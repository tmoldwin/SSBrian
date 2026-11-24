"""
Parameter Sweep Presets

Pre-configured parameter sweeps for common experiments.
Modify these or use as templates for custom sweeps.

Usage:
    from sweep_presets import run_preset
    run_preset('network_size')  # Run one of the presets below
"""

from brian_experiment.feedforward.sweeps.parameter_sweep import ParameterSweep


def network_size_sweep(quick_test=False):
    """
    Sweep over network size (N_rec_E and N_rec_I).
    Tests how network size affects selectivity and sparsity.
    """
    if quick_test:
        n_values = [50, 100]
        n_stimuli = 20
        n_epochs = 1
    else:
        n_values = [50, 100, 150, 200, 300]
        n_stimuli = 100
        n_epochs = 2
    
    sweep = ParameterSweep(
        sweep_name="network_size_sweep",
        base_params={
            'NUM_STIMULI': n_stimuli,
            'NUM_EPOCHS': n_epochs,
            'N_external_input_dims': 100,
        },
        varied_params={
            'N_rec_E': n_values,
            'N_rec_I': n_values
        },
        sweep_type='grid',
        generate_individual_plots=False
    )
    return sweep


def stimulus_alpha_sweep(quick_test=False):
    """
    Sweep over Dirichlet alpha parameter.
    Tests how stimulus sparsity affects network responses.
    """
    if quick_test:
        alphas = [0.1, 0.5, 1.0]
        n_stimuli = 20
    else:
        alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        n_stimuli = 100
    
    sweep = ParameterSweep(
        sweep_name="stimulus_alpha_sweep",
        base_params={
            'NUM_STIMULI': n_stimuli,
            'NUM_EPOCHS': 2,
        },
        varied_params={
            'stimulus_alpha': alphas
        },
        sweep_type='list',
        generate_individual_plots=False
    )
    return sweep


def inhibitory_weight_sweep(quick_test=False):
    """
    Sweep over inhibitory weight scale.
    Tests E/I balance effects.
    """
    if quick_test:
        scales = [3, 5, 7]
        n_stimuli = 20
    else:
        scales = [1, 2, 3, 4, 5, 6, 7, 8, 10]
        n_stimuli = 100
    
    sweep = ParameterSweep(
        sweep_name="inhibitory_weight_sweep",
        base_params={
            'NUM_STIMULI': n_stimuli,
            'NUM_EPOCHS': 2,
        },
        varied_params={
            'inhibitory_weight_scale': scales
        },
        sweep_type='list',
        generate_individual_plots=False
    )
    return sweep


def input_dimensions_sweep(quick_test=False):
    """
    Sweep over number of external input dimensions.
    Tests how input diversity affects responses.
    """
    if quick_test:
        dims = [50, 100, 150]
        n_stimuli = 20
    else:
        dims = [25, 50, 75, 100, 150, 200]
        n_stimuli = 100
    
    sweep = ParameterSweep(
        sweep_name="input_dimensions_sweep",
        base_params={
            'NUM_STIMULI': n_stimuli,
            'NUM_EPOCHS': 2,
        },
        varied_params={
            'N_external_input_dims': dims
        },
        sweep_type='list',
        generate_individual_plots=False
    )
    return sweep


def feedforward_vs_recurrent(quick_test=False):
    """
    Compare feedforward vs. recurrent networks across different sizes.
    """
    if quick_test:
        n_values = [50, 100]
        n_stimuli = 20
    else:
        n_values = [50, 100, 150, 200]
        n_stimuli = 100
    
    sweep = ParameterSweep(
        sweep_name="feedforward_vs_recurrent",
        base_params={
            'NUM_STIMULI': n_stimuli,
            'NUM_EPOCHS': 2,
        },
        varied_params={
            'N_rec_E': n_values,
            'N_rec_I': n_values,
            'disable_recurrent': [True, False]
        },
        sweep_type='grid',
        generate_individual_plots=False
    )
    return sweep


def epoch_evolution_sweep(quick_test=False):
    """
    Sweep over number of epochs to see temporal evolution.
    """
    if quick_test:
        epochs = [1, 2, 3]
        n_stimuli = 20
    else:
        epochs = [1, 2, 3, 5, 10]
        n_stimuli = 100
    
    sweep = ParameterSweep(
        sweep_name="epoch_evolution_sweep",
        base_params={
            'NUM_STIMULI': n_stimuli,
        },
        varied_params={
            'NUM_EPOCHS': epochs
        },
        sweep_type='list',
        generate_individual_plots=False
    )
    return sweep


def comprehensive_architecture_sweep(quick_test=False):
    """
    Comprehensive sweep over multiple architectural parameters.
    WARNING: This can take a long time!
    """
    if quick_test:
        sweep = ParameterSweep(
            sweep_name="comprehensive_architecture_quick",
            base_params={
                'NUM_STIMULI': 20,
                'NUM_EPOCHS': 1,
            },
            varied_params={
                'N_rec_E': [50, 100],
                'N_rec_I': [50, 100],
                'N_external_input_dims': [50, 100],
                'inhibitory_weight_scale': [3, 5]
            },
            sweep_type='grid',
            generate_individual_plots=False
        )
    else:
        sweep = ParameterSweep(
            sweep_name="comprehensive_architecture_full",
            base_params={
                'NUM_STIMULI': 100,
                'NUM_EPOCHS': 2,
            },
            varied_params={
                'N_rec_E': [50, 100, 150],
                'N_rec_I': [50, 100, 150],
                'N_external_input_dims': [50, 100, 150],
                'inhibitory_weight_scale': [3, 5, 7]
            },
            sweep_type='grid',
            generate_individual_plots=False
        )
    return sweep


def input_amplitude_sweep(quick_test=False):
    """
    Sweep over input amplitude scale.
    Tests how input strength affects network responses.
    """
    if quick_test:
        scales = [0.5, 1.0, 2.0]
        n_stimuli = 20
    else:
        scales = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        n_stimuli = 100
    
    sweep = ParameterSweep(
        sweep_name="input_amplitude_sweep",
        base_params={
            'NUM_STIMULI': n_stimuli,
            'NUM_EPOCHS': 2,
        },
        varied_params={
            'input_amplitude_scale': scales
        },
        sweep_type='list',
        generate_individual_plots=False
    )
    return sweep


# Dictionary of all presets
PRESETS = {
    'network_size': network_size_sweep,
    'stimulus_alpha': stimulus_alpha_sweep,
    'inhibitory_weight': inhibitory_weight_sweep,
    'input_dimensions': input_dimensions_sweep,
    'ff_vs_rec': feedforward_vs_recurrent,
    'epoch_evolution': epoch_evolution_sweep,
    'comprehensive': comprehensive_architecture_sweep,
    'input_amplitude': input_amplitude_sweep,
}


def run_preset(preset_name, quick_test=False):
    """
    Run a preset parameter sweep.
    
    Parameters:
    -----------
    preset_name : str
        Name of preset from PRESETS dict
    quick_test : bool
        If True, run a quick test version with fewer parameters
    
    Returns:
    --------
    results : list
        List of summary statistics from all runs
    """
    if preset_name not in PRESETS:
        print(f"Error: Unknown preset '{preset_name}'")
        print(f"Available presets: {list(PRESETS.keys())}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Running preset: {preset_name}")
    print(f"Quick test mode: {quick_test}")
    print(f"{'='*60}\n")
    
    # Create sweep
    sweep_func = PRESETS[preset_name]
    sweep = sweep_func(quick_test=quick_test)
    
    # Run sweep
    results = sweep.run()
    
    # Automatically run aggregate analysis
    print("\nRunning aggregate analysis...")
    from aggregate_analysis import AggregateAnalyzer
    analyzer = AggregateAnalyzer(sweep.sweep_folder)
    analyzer.generate_all_plots()
    
    print(f"\n{'='*60}")
    print(f"PRESET COMPLETE: {preset_name}")
    print(f"Results saved to: {sweep.sweep_folder}")
    print(f"{'='*60}\n")
    
    return results


def list_presets():
    """Print information about available presets"""
    print("\n" + "="*60)
    print("AVAILABLE PARAMETER SWEEP PRESETS")
    print("="*60 + "\n")
    
    for name, func in PRESETS.items():
        print(f"{name}:")
        print(f"  {func.__doc__.strip().split('.')[0]}")
        print()
    
    print("Usage:")
    print("  from sweep_presets import run_preset")
    print("  run_preset('network_size', quick_test=True)")
    print("\nOr from command line:")
    print("  python sweep_presets.py <preset_name> [--quick]")
    print()


if __name__ == "__main__":
    """Command-line interface for running presets"""
    import sys
    
    if len(sys.argv) < 2:
        list_presets()
        sys.exit(0)
    
    preset_name = sys.argv[1]
    quick_test = '--quick' in sys.argv or '-q' in sys.argv
    
    if preset_name == 'list':
        list_presets()
    else:
        run_preset(preset_name, quick_test=quick_test)

