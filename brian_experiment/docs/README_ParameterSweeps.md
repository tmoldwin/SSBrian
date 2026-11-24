# Parameter Sweep System

A comprehensive system for running multiple simulations with different parameter configurations and analyzing aggregate results.

## Overview

The parameter sweep system consists of three main modules:

1. **`parameter_sweep.py`** - Executes multiple simulations with varying parameters
2. **`aggregate_analysis.py`** - Analyzes and visualizes results across runs
3. **`sweep_presets.py`** - Pre-configured common sweeps

## Quick Start

### Option 1: Use a Preset

The easiest way to get started is with a preset configuration:

```python
from sweep_presets import run_preset

# Run a quick test of network size sweep
run_preset('network_size', quick_test=True)

# Run full network size sweep
run_preset('network_size', quick_test=False)
```

Available presets:
- `network_size` - Vary N_rec_E and N_rec_I
- `stimulus_alpha` - Vary Dirichlet alpha parameter
- `inhibitory_weight` - Vary E/I balance
- `input_dimensions` - Vary number of input dimensions
- `ff_vs_rec` - Compare feedforward vs recurrent
- `epoch_evolution` - Vary number of epochs
- `input_amplitude` - Vary input strength
- `comprehensive` - Multi-parameter sweep (slow!)

### Option 2: Custom Sweep

Create your own parameter sweep:

```python
from parameter_sweep import ParameterSweep

sweep = ParameterSweep(
    sweep_name="my_custom_sweep",
    base_params={
        'NUM_STIMULI': 50,  # Use fewer stimuli for speed
        'NUM_EPOCHS': 2
    },
    varied_params={
        'N_rec_E': [50, 100, 150, 200],
        'stimulus_alpha': [0.1, 0.5, 1.0]
    },
    sweep_type='grid',  # All combinations
    generate_individual_plots=False  # Skip individual plots for speed
)

# Run the sweep
results = sweep.run()

# Analyze results
from aggregate_analysis import AggregateAnalyzer
analyzer = AggregateAnalyzer(sweep.sweep_folder)
analyzer.generate_all_plots()
```

## Sweep Types

### Grid Sweep
All combinations of parameters (Cartesian product):

```python
varied_params={
    'N_rec_E': [50, 100],
    'N_rec_I': [50, 100]
}
# Results in 4 runs: (50,50), (50,100), (100,50), (100,100)
```

### List Sweep
Paired parameter values (all lists must be same length):

```python
varied_params={
    'N_rec_E': [50, 100, 150],
    'N_rec_I': [50, 100, 150]
}
# Results in 3 runs: (50,50), (100,100), (150,150)
```

### Random Sweep
Random sampling from parameter ranges:

```python
sweep = ParameterSweep(
    sweep_name="random_sweep",
    varied_params={
        'N_rec_E': [50, 200],  # Will sample uniformly between 50 and 200
        'stimulus_alpha': [0.01, 10.0]
    },
    sweep_type='random',
    random_samples=20  # Number of random combinations
)
```

## Available Parameters

All parameters from `feedforward/single_run.py` can be varied:

### Network Architecture
- `N_rec_E` - Number of excitatory recurrent neurons
- `N_rec_I` - Number of inhibitory recurrent neurons
- `N_external_input_dims` - Number of input dimensions
- `NUM_STIMULI` - Number of unique stimuli
- `NUM_EPOCHS` - Number of training epochs

### Simulation Timing (in ms)
- `stim_duration` - Duration of each stimulus presentation
- `break_time` - Break between stimuli
- `start_time` - Initial delay before first stimulus

### Weight & Input Scaling
- `base_weight_scale` - Base scale for recurrent weights
- `weight_scale_method` - Scaling method: 'k' or 'sqrt_k'
- `inhibitory_weight_scale` - Additional scale for inhibitory weights
- `input_amplitude_scale` - Scale for input amplitudes

### Stimulus Generation
- `stimulus_distribution` - Distribution type: 'dirichlet', 'normal', 'uniform'
- `stimulus_alpha` - Dirichlet alpha parameter (lower = more sparse)

### Input Mode
- `probabilistic_input` - Boolean: probabilistic vs deterministic input
- `unitary_current_nA` - Current amplitude for probabilistic input (in nA)

### Network Type
- `disable_recurrent` - Boolean: True for feedforward, False for recurrent

## Folder Structure

```
Sweeps/
└── sweep_name_TIMESTAMP/
    ├── sweep_metadata.pkl          # Sweep configuration
    ├── sweep_info.txt              # Human-readable info
    ├── aggregate_stats.pkl         # Summary statistics from all runs
    ├── run_0000/
    │   ├── Data/
    │   │   └── simulation_data.pkl # Full simulation data
    │   └── summary_stats.pkl       # Quick-access statistics
    ├── run_0001/
    │   └── ...
    └── AggregateAnalysis/          # Aggregate plots (created by analyzer)
        ├── summary_dashboard.png
        ├── mean_selectivity_vs_N_rec_E.png
        ├── selectivity_distribution_evolution_N_rec_E.png
        └── ...
```

## Output Plots

The aggregate analyzer generates several types of plots:

### 1. Summary Dashboard
6-panel overview showing how key metrics vary with parameters:
- Mean selectivity
- Mean sparsity
- Mean firing rate
- Number of active neurons
- Standard deviation of selectivity
- Standard deviation of firing rate

### 2. Individual Metric Plots
Scatter plots with error bars showing:
- Individual run results (blue points)
- Mean ± std for each parameter value (red line with error bars)

### 3. Distribution Evolution
Shows how distributions (e.g., selectivity distribution) change across parameter values.

### 4. 2D Heatmaps
For sweeps with 2+ varied parameters, shows metrics as heatmaps.

## Command-Line Usage

### Run a preset:
```bash
cd brian_experiment
python sweep_presets.py network_size --quick
```

### Analyze existing sweep:
```bash
python aggregate_analysis.py Sweeps/network_size_sweep_20231015_143022
```

### List available presets:
```bash
python sweep_presets.py list
```

## Performance Tips

1. **Reduce stimuli for testing**: Use `NUM_STIMULI: 20` instead of 100
2. **Disable individual plots**: Set `generate_individual_plots=False` (default)
3. **Use quick_test mode**: Most presets have a `quick_test=True` option
4. **Start small**: Test with 2-3 parameter values before full sweep
5. **Use list sweep**: Faster than grid sweep for exploring trends

## Example Workflows

### Workflow 1: Explore network size effects
```python
from sweep_presets import run_preset

# Quick test (5 minutes)
run_preset('network_size', quick_test=True)

# Full sweep (30-60 minutes)
run_preset('network_size', quick_test=False)
```

### Workflow 2: Custom E/I balance study
```python
from parameter_sweep import ParameterSweep
from aggregate_analysis import AggregateAnalyzer

# Create sweep
sweep = ParameterSweep(
    sweep_name="ei_balance_study",
    base_params={'NUM_STIMULI': 50, 'NUM_EPOCHS': 2},
    varied_params={
        'inhibitory_weight_scale': [1, 2, 3, 4, 5, 6, 7, 8],
        'N_rec_I': [50, 100, 150]  # Also vary I neuron count
    },
    sweep_type='grid'
)

# Run
results = sweep.run()

# Analyze
analyzer = AggregateAnalyzer(sweep.sweep_folder)
analyzer.generate_all_plots()

# Custom analysis
import matplotlib.pyplot as plt
analyzer.plot_metric_vs_two_parameters(
    'inhibitory_weight_scale', 'N_rec_I', 'mean_selectivity',
    title='Selectivity: E/I Balance Study'
)
```

### Workflow 3: Compare architectures
```python
from sweep_presets import run_preset

# Compare feedforward vs recurrent across sizes
results = run_preset('ff_vs_rec', quick_test=False)

# Results will show whether recurrence helps or hurts selectivity
# at different network sizes
```

## Extending the System

### Add a new preset:

Edit `sweep_presets.py`:

```python
def my_new_sweep(quick_test=False):
    """Description of what this sweep does."""
    sweep = ParameterSweep(
        sweep_name="my_new_sweep",
        base_params={...},
        varied_params={...},
        sweep_type='grid'
    )
    return sweep

# Add to PRESETS dict
PRESETS['my_sweep'] = my_new_sweep
```

### Add custom analysis:

```python
from aggregate_analysis import AggregateAnalyzer

analyzer = AggregateAnalyzer(sweep_folder)

# Access raw data
for run in analyzer.all_stats:
    print(run['params'], run['mean_selectivity'])

# Create custom plot
import matplotlib.pyplot as plt
param_vals = analyzer.get_varied_param_values('N_rec_E')
metric_vals = analyzer.get_metric_values('mean_selectivity')
plt.plot(param_vals, metric_vals, 'o-')
plt.xlabel('Number of E Neurons')
plt.ylabel('Mean Selectivity')
plt.savefig('my_custom_plot.png')
```

## Troubleshooting

### Out of memory
- Reduce `NUM_STIMULI`
- Reduce network size
- Run fewer parameter combinations at once

### Slow simulations
- Use `quick_test=True` for presets
- Reduce `NUM_EPOCHS`
- Reduce `NUM_STIMULI`
- Use 'list' instead of 'grid' sweep type

### Missing plots
- Check that `aggregate_analysis.py` was run
- Look for error messages in console output
- Verify sweep completed successfully

### Load existing sweep
```python
from aggregate_analysis import AggregateAnalyzer

# Just analysis, no re-simulation
analyzer = AggregateAnalyzer('Sweeps/my_sweep_20231015_143022')
analyzer.generate_all_plots()
```

## Best Practices

1. **Always start with quick_test=True** to verify setup
2. **Use descriptive sweep names** that include what you're varying
3. **Save sweep_folder path** for later analysis
4. **Document parameter choices** in a notebook or script
5. **Keep base_params reasonable** (don't make everything vary)
6. **Check summary dashboard first** before diving into details
7. **Use version control** for sweep configuration scripts

## Integration with Existing Code

The parameter sweep system is fully compatible with existing analysis tools:

- All run folders have standard structure (Data/, Figures/)
- Individual runs can be analyzed with `generate_plots.py`
- Sweep results can be loaded and re-analyzed anytime
- No changes needed to core simulation code

## Questions?

See the docstrings in each module for more details:
- `parameter_sweep.py` - Main sweep runner
- `aggregate_analysis.py` - Analysis and visualization
- `sweep_presets.py` - Preset configurations

