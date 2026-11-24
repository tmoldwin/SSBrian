# DirectInputNetwork - Brian Simulation with Direct Input Vectors

## Overview

The `DirectInputNetwork` is a new Brian simulation that replaces the Poisson neurons from the original `HaimNetwork` with direct input vector presentation. Instead of changing Poisson neuron firing rates, input vectors are presented directly as weighted currents to the recurrent neurons.

## Key Features

### 🔄 **Direct Input Mechanism**
- **No Poisson Neurons**: Completely removed PoissonGroup and related synapses
- **Weighted Input Vectors**: Each neuron receives weighted input from all input dimensions
- **Real-time Current Injection**: Input vectors are converted to currents via `I_input = W_input × input_vector`

### ⚖️ **Sign-Dependent Weights**
- **Excitatory Inputs**: Positive weights for excitatory input dimensions
- **Inhibitory Inputs**: Negative weights for inhibitory input dimensions  
- **Random Initialization**: Weights are randomly generated with proper signs

### 🧠 **Same Recurrent Dynamics**
- **Preserved Structure**: Maintains the same recurrent connectivity as HaimNetwork
- **Same Neuron Model**: Uses identical LIF neurons with synaptic dynamics
- **Compatible Analysis**: Works with existing spike analysis and selectivity metrics

## Files Created

1. **`feedforward/network.py`** - Main network class implementation
2. **`feedforward/single_run.py`** - Full simulation script with comprehensive plotting
3. **`simple_direct_test.py`** - Simplified test script for basic functionality
4. **`test_direct_input.py`** - Unit test for debugging

## Usage

### Basic Usage

```python
from brian_experiment.feedforward import network as din
import weight_matrices as wm
import numpy as np
from brian2 import *

# Network parameters
N_rec_E, N_rec_I = 100, 100  # Recurrent neurons
N_input_E, N_input_I = 100, 100  # Input dimensions

# Create recurrent weight matrix
weight_rec = 0.001 * wm.weight_matrix_factory(
    {'recurrent': [N_rec_E, N_rec_I]}, 
    {'exponential': 1}
)

# Create network
network = din.DirectInputNetwork(
    N_rec_E, N_rec_I, N_input_E, N_input_I, 
    weight_rec, input_weight_scale=0.8
)

# Create input vectors [N_input_total x time_steps]
input_vectors = np.random.rand(N_input_E + N_input_I, 1000) * 10.0

# Run simulation
network.run_simulation(
    T=1000*ms,
    input_vectors=input_vectors,
    stims_by_time=stimulus_labels,
    stim_info=stimulus_info,
    current_injection=0*nA,
    monitoring_level='basic'
)

# Analyze results
network.analyze_spikes()
```

### Running the Full Simulation

```bash
# Full simulation with comprehensive analysis
python brian_experiment/feedforward/single_run.py

# Simple test with basic functionality
python brian_experiment/simple_direct_test.py
```

## Network Architecture

```
Input Vector (N_input dims) 
        ↓
   Weight Matrix W_input [N_rec × N_input]
        ↓  
   I_input = W_input × input_vector
        ↓
Recurrent Neurons (N_rec_E + N_rec_I)
        ↓
   Recurrent Connectivity W_rec
        ↓
    Spike Output
```

## Key Differences from HaimNetwork

| Aspect | HaimNetwork | DirectInputNetwork |
|--------|-------------|-------------------|
| **Input Method** | PoissonGroup rates | Direct current injection |
| **Input Weights** | Fixed Poisson→Rec weights | Learnable input weights |
| **Input Control** | Rate changes | Vector values |
| **Stimulation** | Poisson spike trains | Deterministic currents |
| **Input Dimensions** | Fixed Poisson neurons | Flexible input dimensions |

## Input Vector Format

Input vectors should be shaped as `[N_input_total × time_steps]` where:
- **Rows**: Input dimensions (first N_input_E are excitatory, rest inhibitory)
- **Columns**: Time steps (1ms resolution)
- **Values**: Input amplitudes (will be weighted and injected as current)

## Configuration Options

### Network Sizes
```python
run_configs = {
    'short': {
        'N_rec_E': 10, 'N_rec_I': 10,
        'N_input_E': 15, 'N_input_I': 10,
        'num_epochs': 5, 'stim_list': np.arange(4)
    },
    'test': {
        'N_rec_E': 100, 'N_rec_I': 100,
        'N_input_E': 100, 'N_input_I': 100,
        'num_epochs': 5, 'stim_list': np.arange(3)
    }
}
```

### Monitoring Levels
- **`'minimal'`**: Only spike data (fastest)
- **`'basic'`**: Spikes + voltage from subset of neurons
- **`'full'`**: All data including currents (slowest)

## Generated Plots

The full simulation creates four comprehensive figures:

1. **Figure 1**: Weight matrices and input patterns
2. **Figure 2**: Network dynamics and spiking activity  
3. **Figure 3**: Sparsity and selectivity evolution over time
4. **Figure 4**: Selectivity comparison between epochs

## Performance Optimizations

- **Pre-computed Input Changes**: Only updates when input vectors change
- **Efficient Current Injection**: Vectorized weight matrix operations
- **Configurable Monitoring**: Trade-off between data and speed
- **Progress Tracking**: Epoch-based progress reporting

## Example Output

```
🎉 Simulation completed successfully!
📁 Results saved to: figures/direct_input_run_20250717_150059
📊 Final network sparsity: 0.123 ± 0.045
📊 Final network selectivity: 0.456 ± 0.123
```

## Validation

The implementation has been tested with:
- ✅ Network initialization and weight generation
- ✅ Simulation execution with various input patterns
- ✅ Spike analysis and selectivity calculations
- ✅ Plot generation and data saving
- ✅ Multiple network configurations

## Next Steps

1. **Parameter Exploration**: Test different input weight scales and amplitudes
2. **Learning Rules**: Add plasticity to input weights
3. **Complex Stimuli**: Experiment with more sophisticated input patterns
4. **Comparative Analysis**: Compare with original HaimNetwork results

## Troubleshooting

If you encounter issues:
1. Run `simple_direct_test.py` first to verify basic functionality
2. Check input vector shapes match network dimensions
3. Ensure sufficient input amplitude for network activation
4. Try different monitoring levels if memory issues occur

The DirectInputNetwork provides a flexible alternative to Poisson-driven networks while maintaining compatibility with existing analysis tools. 