# Plasticity Demo System

This document describes the new plasticity demo system that allows you to run simulations with synaptic plasticity and analyze weight changes over time.

## Overview

The plasticity demo system consists of three main components:

1. **PlasticDirectInputNetwork** - A network class that extends DirectInputNetwork with plasticity support
2. **feedforward/plastic_demo.py** - A demo script that runs a simple plasticity experiment
3. **feedforward/weight_evolution_analysis.py** - Analysis and plotting tools for visualizing weight changes

## Files Created

### 1. `feedforward/plastic_network.py`
A new network class that supports synaptic plasticity:
- Extends the existing `DirectInputNetwork` class
- Supports plasticity kernels from `plasticity_kernels.py`
- Monitors weight changes over time at configurable intervals
- Compatible with all existing DirectInputNetwork features

**Key Features:**
- Pass `plasticity_config` dictionary to specify which connections have plasticity
- Weight monitoring with configurable time resolution (`weight_monitoring_dt`)
- Methods to extract weight evolution and weight matrices at specific times

### 2. `feedforward/plastic_demo.py`
A demonstration script showing how to use plasticity:
- **Configuration**: Single long stimulus (10 seconds) to observe weight evolution
- **Small network**: 50E + 50I neurons for faster simulation
- **Plasticity types**: Supports STDP, Hebbian, Anti-Hebbian, and mixed rules
- **Output**: Generates comprehensive analysis plots

**Key Configuration Variables:**
```python
N_REC_E = 50                      # Excitatory neurons
N_REC_I = 50                      # Inhibitory neurons
N_EXTERNAL_INPUT_DIMS = 100       # Input dimensions
SIMULATION_DURATION = 10000       # ms (10 seconds)
PLASTICITY_TYPE = 'stdp'          # 'stdp', 'hebbian', 'mixed', 'none'
WEIGHT_MONITORING_DT = 100 * ms   # Record weights every 100ms
```

### 3. `feedforward/weight_evolution_analysis.py`
Analysis and plotting tools for weight evolution:
- **plot_weight_evolution()**: Multi-panel plot showing weight traces, statistics over time
- **plot_weight_distribution_evolution()**: Histograms showing weight distribution changes
- **plot_weight_matrices_timeline()**: Weight matrices at different time points
- **plot_weight_change_statistics()**: Comprehensive statistics about weight changes
- **plot_synapse_trajectories()**: Individual synapse trajectories

## How to Use

### Basic Usage

1. Run the plasticity demo:
```bash
cd brian_experiment
python feedforward/plastic_demo.py
```

2. Results will be saved to `Runs/plasticity_demo/Figures/`

### Customizing the Demo

Edit the configuration variables at the top of `feedforward/plastic_demo.py`:

```python
# Change plasticity type
PLASTICITY_TYPE = 'hebbian'  # or 'stdp', 'mixed', 'none'

# Adjust network size
N_REC_E = 100
N_REC_I = 100

# Change simulation duration
SIMULATION_DURATION = 20000  # 20 seconds

# Adjust weight monitoring resolution
WEIGHT_MONITORING_DT = 50 * ms  # Record every 50ms
```

### Using in Your Own Simulations

```python
from brian_experiment.feedforward.plastic_network import PlasticDirectInputNetwork
from plasticity_kernels import STDPKernel, HebbianKernel

# Configure plasticity
stdp_kernel = STDPKernel(
    A_pre=0.01, 
    A_post=-0.0105, 
    tau_pre=20*ms, 
    tau_post=20*ms,
    learning_rate=0.1,
    weight_bounds=(0, 0.5*nA)
)

plasticity_config = {
    'E->E': stdp_kernel,  # STDP on E->E connections
    'E->I': None,         # No plasticity on other connections
    'I->E': None,
    'I->I': None
}

# Create network
network = PlasticDirectInputNetwork(
    N_rec_E, N_rec_I, N_external_input_dims,
    weight_rec, input_weight_scale_eff, lognormal_params_in,
    is_feedforward=False,
    plasticity_config=plasticity_config,
    weight_monitoring_dt=100*ms
)

# Run simulation (same as DirectInputNetwork)
network.run_simulation(
    T, input_vectors, stims_by_time, stim_info, 0*nA,
    monitoring_level='basic',
    unitary_current=UNITARY_CURRENT,
    num_epochs=1
)

# Analyze weight evolution
times, weights = network.get_weight_evolution('E->all')
print(f"Initial mean weight: {weights[0, :].mean():.4f} nA")
print(f"Final mean weight: {weights[-1, :].mean():.4f} nA")
```

### Analyzing Weight Changes

```python
import weight_evolution_analysis as wea

# Generate comprehensive weight evolution plots
fig1 = wea.plot_weight_evolution(network, 'E->all')
fig1.savefig('weight_evolution.png')

# Plot weight distribution changes
fig2 = wea.plot_weight_distribution_evolution(network, 'E->all')
fig2.savefig('weight_distribution.png')

# Show weight matrices at different times
fig3 = wea.plot_weight_matrices_timeline(network, 'E->all')
fig3.savefig('weight_matrices.png')

# Detailed change statistics
fig4 = wea.plot_weight_change_statistics(network, 'E->all')
fig4.savefig('weight_statistics.png')
```

## Plasticity Rules Available

### STDP (Spike-Timing Dependent Plasticity)
- Classic STDP with separate LTP and LTD windows
- Parameters: `A_pre`, `A_post`, `tau_pre`, `tau_post`

### Hebbian
- Symmetric potentiation when neurons fire together
- Parameters: `A_hebb`, `tau_hebb`

### Anti-Hebbian
- Symmetric depression when neurons fire together
- Parameters: `A_anti`, `tau_anti`

### Custom
- Define your own plasticity rule with a custom function
- See `feedforward/example_plasticity.py` for examples

## Output Files

After running the demo, you'll find:

```
Runs/plasticity_demo/
├── Figures/
│   ├── weight_evolution.png          # Weight traces over time
│   ├── weight_distribution.png       # Distribution changes
│   ├── weight_matrices_timeline.png  # Matrix snapshots
│   ├── raster_plot.png              # Spike raster
│   └── voltage_traces.png           # Mean voltage traces
└── run_parameters.txt               # Simulation parameters (if enabled)
```

## Technical Details

### Weight Monitoring
- Weights are recorded at intervals specified by `weight_monitoring_dt`
- Uses Brian2's `StateMonitor` with custom `dt` parameter
- Minimal overhead - only records weights, not full state

### Connection Types
The `plasticity_config` dictionary can specify plasticity for:
- `'E->E'`: Excitatory to excitatory
- `'E->I'`: Excitatory to inhibitory  
- `'I->E'`: Inhibitory to excitatory
- `'I->I'`: Inhibitory to inhibitory
- `'input->E'`: Input to excitatory (not yet implemented)
- `'input->I'`: Input to inhibitory (not yet implemented)

Note: Currently, the network creates two synapse groups:
- `rec_synapses_E`: E->all (both E and I targets)
- `rec_synapses_I`: I->all (both E and I targets)

The plasticity specified for `'E->E'` applies to all connections from E neurons,
and plasticity for `'I->E'` applies to all connections from I neurons.

### Performance Considerations
- Weight monitoring adds minimal overhead (~5% for dt=100ms)
- Use larger `weight_monitoring_dt` for longer simulations
- Consider smaller networks for exploratory runs

## Future Enhancements

Potential improvements:
1. **Input plasticity**: Add plasticity for input->network connections
2. **Homeostatic plasticity**: Add weight normalization and homeostatic rules
3. **Metaplasticity**: Learning rate that depends on activity history
4. **More kernels**: BCM rule, triplet STDP, etc.
5. **Analysis tools**: More sophisticated weight change analysis

## Troubleshooting

### Simulation runs but weights don't change
- Check that `learning_rate` is not too small
- Verify neurons are actually spiking during simulation
- Ensure `weight_bounds` aren't too restrictive

### Weights explode or go to zero
- Reduce `learning_rate`
- Add appropriate `weight_bounds`
- Check that A_pre and A_post have appropriate signs and magnitudes

### Memory issues
- Increase `weight_monitoring_dt` to record less frequently
- Reduce network size
- Use `monitoring_level='minimal'` to reduce other monitors

## References

- Original plasticity kernel implementation: `plasticity_kernels.py`
- Example usage: `example_plasticity.py`
- Base network class: `feedforward/network.py`

