## 🚀 Parameter Sweep System - Quick Start

I've created a complete parameter sweep system that lets you run multiple simulations with different parameters and analyze aggregate results!

---

## 📦 What Was Created

### Core Modules
1. **`parameter_sweep.py`** - Main sweep runner (400+ lines)
   - Flexible parameter specification
   - Grid, list, and random sampling
   - Organized data storage
   - Progress tracking

2. **`aggregate_analysis.py`** - Analysis & visualization (400+ lines)
   - Summary dashboards
   - Metric vs parameter plots
   - Distribution evolution plots
   - 2D heatmaps

3. **`sweep_presets.py`** - Pre-configured sweeps (300+ lines)
   - 8 ready-to-use configurations
   - Quick test modes
   - Easy to extend

### Documentation & Examples
4. **`README_ParameterSweeps.md`** - Complete documentation
5. **`example_custom_sweep.py`** - Template for custom sweeps
6. **`test_parameter_sweep.py`** - Test script

---

## 🎯 Getting Started (3 Steps)

### Step 1: Test the System (2-3 minutes)
```bash
cd brian_experiment
python test_parameter_sweep.py
```
This runs 4 tiny simulations to verify everything works.

### Step 2: Run a Quick Preset (5-10 minutes)
```python
from sweep_presets import run_preset
run_preset('network_size', quick_test=True)
```
This automatically runs simulations AND generates plots!

### Step 3: Check Results
Results are saved to `Sweeps/<sweep_name>_<timestamp>/`
- Individual run data in `run_XXXX/Data/`
- Aggregate plots in `AggregateAnalysis/`
- Summary dashboard shows all key metrics

---

## 📊 Example Use Cases

### Use Case 1: How does network size affect selectivity?
```python
from sweep_presets import run_preset
run_preset('network_size')
```
**Output:** Plots showing selectivity, sparsity, and firing rate vs. network size

### Use Case 2: What's the optimal E/I balance?
```python
run_preset('inhibitory_weight')
```
**Output:** Plots showing how inhibitory weight scaling affects network behavior

### Use Case 3: Custom experiment
```python
from parameter_sweep import ParameterSweep

sweep = ParameterSweep(
    sweep_name="my_experiment",
    base_params={'NUM_STIMULI': 50},
    varied_params={
        'N_rec_E': [50, 100, 150],
        'stimulus_alpha': [0.1, 0.5, 1.0, 5.0]
    },
    sweep_type='grid'
)
results = sweep.run()
```

---

## 🎨 Generated Plots

The system automatically creates:

### 1. Summary Dashboard (6 panels)
- Mean selectivity vs parameter
- Mean sparsity vs parameter  
- Mean firing rate vs parameter
- Active neurons vs parameter
- Std selectivity vs parameter
- Std firing rate vs parameter

### 2. Distribution Evolution
Shows how distributions change with parameters:
- Selectivity distributions
- Firing rate distributions

### 3. 2D Heatmaps (for multi-parameter sweeps)
Shows interactions between parameters

### 4. Individual Metric Plots
Detailed plots for each metric with:
- Scatter points (individual runs)
- Mean ± std (error bars)
- Trend lines

---

## 🔧 All Tunable Parameters

You can vary ANY parameter from `feedforward/single_run.py`:

**Network Architecture:**
- `N_rec_E`, `N_rec_I` - Network size
- `N_external_input_dims` - Input dimensions
- `NUM_STIMULI` - Number of stimuli
- `NUM_EPOCHS` - Number of epochs

**Weight Scaling:**
- `base_weight_scale` - Recurrent weight scale
- `inhibitory_weight_scale` - E/I balance
- `input_amplitude_scale` - Input strength
- `weight_scale_method` - Scaling method ('k' or 'sqrt_k')

**Stimulus Parameters:**
- `stimulus_alpha` - Dirichlet sparsity
- `stimulus_distribution` - Distribution type

**Network Type:**
- `disable_recurrent` - Feedforward vs recurrent

**Timing:**
- `stim_duration`, `break_time`, `start_time`

**Input Mode:**
- `probabilistic_input` - Boolean
- `unitary_current_nA` - Current amplitude

---

## 📁 Folder Structure

```
Sweeps/
└── my_sweep_20241029_143022/
    ├── sweep_metadata.pkl           # Configuration
    ├── sweep_info.txt               # Human-readable summary
    ├── aggregate_stats.pkl          # All results
    ├── run_0000/
    │   ├── Data/
    │   │   └── simulation_data.pkl  # Full data
    │   └── summary_stats.pkl        # Quick stats
    ├── run_0001/
    │   └── ...
    └── AggregateAnalysis/
        ├── summary_dashboard.png    # Main overview
        ├── mean_selectivity_vs_N_rec_E.png
        ├── selectivity_distribution_evolution.png
        └── ... (many more plots)
```

---

## 💡 Pro Tips

### For Quick Testing
```python
base_params={
    'NUM_STIMULI': 20,  # Instead of 100
    'NUM_EPOCHS': 1     # Instead of 2
}
```

### For Paired Parameters
```python
# Keep E and I neurons equal
varied_params={
    'N_rec_E': [50, 100, 150],
    'N_rec_I': [50, 100, 150]
}
sweep_type='list'  # Paired: (50,50), (100,100), (150,150)
```

### For All Combinations
```python
varied_params={
    'N_rec_E': [50, 100],
    'N_rec_I': [50, 100]
}
sweep_type='grid'  # All combos: (50,50), (50,100), (100,50), (100,100)
```

### To Re-analyze Without Re-simulating
```python
from aggregate_analysis import AggregateAnalyzer
analyzer = AggregateAnalyzer('Sweeps/my_sweep_20241029_143022')
analyzer.generate_all_plots()
```

---

## 📋 Available Presets

Run with: `run_preset('<name>', quick_test=True/False)`

| Preset Name | What It Does | Parameters Varied |
|-------------|--------------|-------------------|
| `network_size` | Network size effects | N_rec_E, N_rec_I |
| `stimulus_alpha` | Stimulus sparsity | stimulus_alpha |
| `inhibitory_weight` | E/I balance | inhibitory_weight_scale |
| `input_dimensions` | Input diversity | N_external_input_dims |
| `ff_vs_rec` | Architecture comparison | disable_recurrent, N_rec |
| `epoch_evolution` | Temporal dynamics | NUM_EPOCHS |
| `input_amplitude` | Input strength | input_amplitude_scale |
| `comprehensive` | Multi-parameter | Multiple (slow!) |

---

## 🔬 Example Workflow

```python
# 1. Start with quick test
from sweep_presets import run_preset
run_preset('network_size', quick_test=True)
# ✓ Results in ~5 minutes, 2-3 network sizes

# 2. If it looks good, run full version
run_preset('network_size', quick_test=False)
# ✓ Results in ~30 minutes, 5+ network sizes

# 3. Examine plots
# Open: Sweeps/network_size_sweep_*/AggregateAnalysis/summary_dashboard.png

# 4. Custom follow-up analysis
from aggregate_analysis import AggregateAnalyzer
analyzer = AggregateAnalyzer('Sweeps/network_size_sweep_20241029_143022')

# Get raw data
selectivity_values = analyzer.get_metric_values('mean_selectivity')
network_sizes = analyzer.get_varied_param_values('N_rec_E')

# Create custom plot
import matplotlib.pyplot as plt
plt.plot(network_sizes, selectivity_values, 'o')
plt.xlabel('Network Size')
plt.ylabel('Selectivity')
plt.savefig('my_custom_analysis.png')
```

---

## 🐛 Troubleshooting

### "Out of memory"
→ Reduce `NUM_STIMULI` or network size

### "Too slow"
→ Use `quick_test=True` or reduce parameter ranges

### "Can't find module"
→ Make sure you're in `brian_experiment/` directory

### "Want to modify a run"
→ Copy `example_custom_sweep.py` and modify it

### "Need different plots"
→ Use `analyzer.plot_metric_vs_parameter()` for custom plots

---

## 📚 Full Documentation

See `README_ParameterSweeps.md` for:
- Detailed API reference
- Advanced usage examples
- Extending the system
- Performance optimization
- Integration with existing code

---

## ✅ Summary

You now have a complete system that:

✓ Runs multiple simulations with varying parameters  
✓ Stores data in an organized structure  
✓ Automatically generates aggregate analysis plots  
✓ Works with all parameters from `feedforward/single_run.py`
✓ Includes 8 pre-configured sweeps  
✓ Fully documented and tested  
✓ Easy to extend and customize

**Next Steps:**
1. Run `test_parameter_sweep.py` to verify setup
2. Try a preset with `quick_test=True`
3. Examine the generated plots
4. Create your own custom sweep!

**Questions?** Check `README_ParameterSweeps.md` or examine the docstrings in the code.

---

Enjoy exploring your parameter space! 🎉

