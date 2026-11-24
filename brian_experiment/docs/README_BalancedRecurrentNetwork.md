# Balanced Recurrent Network Simulation

This module implements a balanced recurrent network for generating spontaneous chaotic dynamics, similar to those described in Szymbalinski et al. and related balanced network literature.

## Overview

Unlike the `DirectInputNetwork` which requires external feed-forward input to drive activity, the `BalancedRecurrentNetwork` generates spontaneous activity through recurrent connectivity alone. The network exhibits chaotic dynamics when properly balanced between excitation and inhibition.

## Key Differences from DirectInputNetwork

### DirectInputNetwork (Feed-Forward + Recurrent)
- Requires external input vectors/currents to drive activity
- Inhibition must cancel both feed-forward input AND recurrent excitation
- Typically lower recurrent connection strengths
- Activity is stimulus-locked

### BalancedRecurrentNetwork (Recurrent Only)
- No external input (or minimal constant input)
- Inhibition only needs to balance recurrent excitation
- Stronger recurrent connections for sustained activity
- Spontaneous, asynchronous irregular (AI) activity
- Initial state determines trajectory

## Files

- `recurrent/network.py` - Network class implementation
- `recurrent/single_run.py` - Main simulation script with configuration

## Network Architecture

- **Neurons**: Leaky Integrate-and-Fire (LIF) with separate E and I populations
- **Connectivity**: All-to-all recurrent connections (E→E, E→I, I→E, I→I)
- **Dale's Law**: Excitatory neurons have positive weights, inhibitory neurons have negative weights
- **No External Input**: Activity is self-sustained through recurrent connections

## Configuration Parameters

### Network Size
```python
N_REC_E = 200  # Excitatory neurons
N_REC_I = 50   # Inhibitory neurons
```

### Weight Scaling
```python
BASE_WEIGHT_SCALE = 3.0  # Higher than feed-forward networks
WEIGHT_SCALE_METHOD = 'sqrt_k'  # Scales as 1/√k
INHIBITORY_WEIGHT_SCALE = 4.0  # Critical for E-I balance
```

**Key Parameter**: `INHIBITORY_WEIGHT_SCALE` determines the E-I balance ratio. For chaotic dynamics:
- Too low (< 3): Runaway excitation, network saturates
- Optimal (3-5): Balanced chaotic dynamics, asynchronous irregular state
- Too high (> 6): Inhibition dominates, network goes silent

### Initial State

The initial state "kickstarts" the dynamics. Three options:

1. **Random Voltage** (recommended for quick chaos onset)
```python
INITIAL_STATE = {
    'type': 'random_voltage',
    'v_min': -70*mV,  # Below threshold
    'v_max': -50*mV   # Near threshold
}
```

2. **Random Current Injection** (for smoother onset)
```python
INITIAL_STATE = {
    'type': 'random_current',
    'I_mean': 1*nA,
    'I_std': 0.5*nA,
    'duration': 50*ms  # Duration of initial current
}
```

3. **Subset Spike Initialization**
```python
INITIAL_STATE = {
    'type': 'subset_spikes',
    'fraction': 0.1  # 10% of neurons start above threshold
}
```

## Expected Dynamics

### Chaotic Balanced State (Good Parameters)
- **Firing rates**: E ~5-15 Hz, I ~15-30 Hz (I fires ~2-4x more than E)
- **CV of ISI**: ~0.8-1.2 (irregular spiking)
- **Population synchrony**: Low (asynchronous)
- **Current balance**: |I_inh| ≈ I_exc (tracked balance)

### Common Issues

**Network Goes Silent**
- Inhibition too strong → Reduce `INHIBITORY_WEIGHT_SCALE`
- Recurrent weights too weak → Increase `BASE_WEIGHT_SCALE`
- Initial state too far from threshold → Increase `v_max` in random_voltage

**Runaway Excitation**
- Inhibition too weak → Increase `INHIBITORY_WEIGHT_SCALE`
- Recurrent weights too strong → Decrease `BASE_WEIGHT_SCALE`

**Synchronous Oscillations**
- Network too small → Increase neuron count
- Weights too uniform → Use higher lognormal variance
- Synaptic timescale too slow → Decrease `tau_syn` (currently 1ms)

## Usage

### Basic Usage
```bash
python brian_experiment/recurrent/single_run.py
```

### Customization
Edit the configuration section at the top of `recurrent/single_run.py`:
```python
N_REC_E = 200
N_REC_I = 50
SIMULATION_DURATION = 5000  # ms
BASE_WEIGHT_SCALE = 3.0
INHIBITORY_WEIGHT_SCALE = 4.0
```

## Output

Results are saved to `Runs/balanced_recurrent_E{N_E}_I{N_I}_T{duration}/{timestamp}/`:

1. **run_parameters.txt** - Complete parameter log
2. **simulation_data.pkl** - Raw data for analysis
3. **weight_matrices.png** - Weight matrix structure
4. **spontaneous_activity_full.png** - Full simulation plots
5. **spontaneous_activity_1s.png** - Detailed 1-second view

### Plots Include
- **Raster plot**: Spike times (red=E, blue=I)
- **Population firing rates**: Average rates over time for E and I
- **Mean membrane potential**: Population average voltage dynamics

## Statistics Computed

The simulation automatically computes:
- Average firing rates (E and I populations)
- Coefficient of variation of inter-spike intervals (ISI)
- Population rate variability (synchrony measure)
- Mean synaptic currents and E-I balance ratio

## Theoretical Background

### Balanced Networks
In balanced networks (van Vreeswijk & Sompolinsky 1996, 1998):
- Strong recurrent excitation is canceled by strong recurrent inhibition
- Neurons operate in a fluctuation-driven regime
- Activity is asynchronous and irregular (AI state)
- Chaos emerges from the high-dimensional dynamics

### E-I Balance Condition
For balance: `g × I_rate × w_I ≈ E_rate × w_E`
where:
- `g` is the inhibitory weight scale factor
- `I_rate`, `E_rate` are firing rates
- `w_I`, `w_E` are average connection weights

### Comparison to Szymbalinski/Rajan Papers
These papers typically study:
- Spontaneous activity patterns
- Selectivity emergence without training
- Robustness of network representations
- Scaling properties with network size

This implementation provides the substrate for such analyses.

## Tips for Parameter Tuning

1. **Start with recommended parameters** (defaults in the script)
2. **Check firing rates first**: 
   - E should be 5-20 Hz
   - I should be 2-4x higher than E
3. **Check CV**: Should be 0.8-1.2 for irregular spiking
4. **Check balance ratio**: |I_current|/E_current should be 0.8-1.2
5. **Adjust `INHIBITORY_WEIGHT_SCALE`** as the main tuning knob

## References

- van Vreeswijk & Sompolinsky (1996). "Chaos in neuronal networks with balanced excitatory and inhibitory activity"
- Renart et al. (2010). "The asynchronous state in cortical circuits"
- Sompolinsky et al. (1988). "Chaos in random neural networks"
- Brunel (2000). "Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons"

## Future Extensions

Potential additions:
- Structured connectivity (not all-to-all)
- Plasticity rules for spontaneous activity
- Multiple attractors/fixed points
- External perturbations to study response dynamics
- Spike-frequency adaptation

