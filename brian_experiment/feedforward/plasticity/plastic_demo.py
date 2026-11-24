"""
Plasticity demo: Single long stimulus to observe weight changes.

This script runs a simple simulation with a single stimulus presented
continuously to observe how synaptic weights evolve under plasticity.
"""

import argparse
import json
import os
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy.ndimage import uniform_filter1d

# Import our modules
from brian_experiment.feedforward.plasticity.plastic_network import PlasticDirectInputNetwork
from brian_experiment.feedforward.plasticity.plasticity_kernels import (
    STDPKernel,
    HebbianKernel,
    AntiHebbianKernel,
)
from brian_experiment.shared.core import stimuli as stim
from brian_experiment.shared.core import weight_matrices as wm
from brian_experiment.shared.core import constants as consts

# Set backend
prefs.codegen.target = "numpy"

# ======================================================================================
#                                 CONFIGURATION
# ======================================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Network size (keep small for visualization)
N_REC_E = 50
N_REC_I = 50
N_EXTERNAL_INPUT_DIMS = 100

# Active configuration label (overridden when loading JSON)
PLASTICITY_CONFIG_FALLBACK = "plasticity_demo"
CURRENT_CONFIG_LABEL = PLASTICITY_CONFIG_FALLBACK

# Simulation timing - SINGLE LONG STIMULUS
SIMULATION_DURATION = 2000  # ms (2 seconds)
STIMULUS_ID = 0  # Single stimulus

# Weight scaling
BASE_WEIGHT_SCALE = 1.0
WEIGHT_SCALE_METHOD = 'sqrt_k'
INHIBITORY_WEIGHT_SCALE = 5.0
INPUT_AMPLITUDE_SCALE = 1.0

# Input generation
STIMULUS_DISTRIBUTION = 'dirichlet'
STIMULUS_ALPHA = 0.5
PROBABILISTIC_INPUT = True
UNITARY_CURRENT_NA = 1.0
UNITARY_CURRENT = UNITARY_CURRENT_NA * nA

# Weight monitoring
WEIGHT_MONITORING_DT_MS = 10  # ms
WEIGHT_MONITORING_DT = WEIGHT_MONITORING_DT_MS * ms  # Record weights every 10ms

# Plasticity configuration
PLASTICITY_TYPE = 'hebbian'  # Options: 'stdp', 'hebbian', 'mixed', 'none'


def _set_int(var_name, value):
    globals()[var_name] = int(value)


def _set_float(var_name, value):
    globals()[var_name] = float(value)


def _set_str(var_name, value):
    globals()[var_name] = str(value)


def _set_bool(var_name, value):
    if isinstance(value, str):
        value = value.strip().lower() in {"1", "true", "t", "yes", "y"}
    else:
        value = bool(value)
    globals()[var_name] = value


def _set_unitary_current(value):
    global UNITARY_CURRENT_NA, UNITARY_CURRENT
    UNITARY_CURRENT_NA = float(value)
    UNITARY_CURRENT = UNITARY_CURRENT_NA * nA


def _set_weight_monitoring_dt(value):
    global WEIGHT_MONITORING_DT_MS, WEIGHT_MONITORING_DT
    WEIGHT_MONITORING_DT_MS = float(value)
    WEIGHT_MONITORING_DT = WEIGHT_MONITORING_DT_MS * ms


CONFIG_SETTERS = {
    "random_seed": lambda v: _set_int("RANDOM_SEED", v),
    "n_rec_e": lambda v: _set_int("N_REC_E", v),
    "n_rec_i": lambda v: _set_int("N_REC_I", v),
    "n_external_input_dims": lambda v: _set_int("N_EXTERNAL_INPUT_DIMS", v),
    "simulation_duration": lambda v: _set_int("SIMULATION_DURATION", v),
    "stimulus_id": lambda v: _set_int("STIMULUS_ID", v),
    "base_weight_scale": lambda v: _set_float("BASE_WEIGHT_SCALE", v),
    "weight_scale_method": lambda v: _set_str("WEIGHT_SCALE_METHOD", v),
    "inhibitory_weight_scale": lambda v: _set_float("INHIBITORY_WEIGHT_SCALE", v),
    "input_amplitude_scale": lambda v: _set_float("INPUT_AMPLITUDE_SCALE", v),
    "stimulus_distribution": lambda v: _set_str("STIMULUS_DISTRIBUTION", v),
    "stimulus_alpha": lambda v: _set_float("STIMULUS_ALPHA", v),
    "probabilistic_input": lambda v: _set_bool("PROBABILISTIC_INPUT", v),
    "unitary_current_nA": _set_unitary_current,
    "weight_monitoring_dt_ms": _set_weight_monitoring_dt,
    "plasticity_type": lambda v: _set_str("PLASTICITY_TYPE", v),
}


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run the plasticity demo with optional JSON parameter overrides."
    )
    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Path to a JSON file containing run parameters. "
            "Defaults to feedforward/configs/runs/plasticity_demo.json if present."
        ),
    )
    return parser.parse_args()


def resolve_config_path(config_arg):
    if config_arg:
        candidate = Path(config_arg).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Config file not found: {candidate}")
        return candidate
    default_path = consts.FEEDFWD_RUN_CONFIG_DIR / "plasticity_demo.json"
    if default_path.exists():
        return default_path
    return None


def load_config_dict(config_path):
    if config_path is None:
        return {}, "defaults"
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data, config_path.stem


def apply_config_overrides(config_dict, label):
    global CURRENT_CONFIG_LABEL
    CURRENT_CONFIG_LABEL = label or CURRENT_CONFIG_LABEL
    if not config_dict:
        return
    print(f"\n>> Loading run parameters from JSON config '{CURRENT_CONFIG_LABEL}'")
    for key, value in config_dict.items():
        setter = CONFIG_SETTERS.get(key)
        if setter is None:
            print(f"   [skip] Unknown key '{key}'")
            continue
        setter(value)
        print(f"   {key}: {value}")

# ======================================================================================

def generate_constant_input_vector(N_external_input_dims, n_timesteps, 
                                   distribution='dirichlet', alpha=0.5, 
                                   probabilistic_input=False):
    """Generate constant input vector for entire simulation."""
    if distribution == 'dirichlet':
        alpha_vec = np.full(N_external_input_dims, alpha)
        pattern = np.random.dirichlet(alpha_vec) * 10.0  # Scale up
    elif distribution == 'normal':
        pattern = np.abs(np.random.normal(0, 1, N_external_input_dims))
    elif distribution == 'uniform':
        pattern = np.random.uniform(0, 1, N_external_input_dims)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Create input vectors for all time steps
    input_vectors = np.zeros((N_external_input_dims, n_timesteps))
    
    for t in range(n_timesteps):
        if probabilistic_input:
            # Sample from the pattern (Bernoulli process)
            input_vectors[:, t] = np.random.binomial(1, np.clip(pattern, 0, 1))
        else:
            # Use the pattern directly
            input_vectors[:, t] = pattern
    
    return input_vectors, pattern


def configure_plasticity(plasticity_type):
    """Configure plasticity rules."""
    if plasticity_type == 'none':
        return {}
    
    elif plasticity_type == 'stdp':
        stdp_kernel = STDPKernel(
            eta_pre=0.5, 
            eta_post=0.525, 
            tau_pre=20*ms, 
            tau_post=20*ms,
            weight_bounds=(0, 0.5*nA)
        )
        return {'E->E': stdp_kernel}
    
    elif plasticity_type == 'hebbian':
        # Hebbian on E->E connections (Mexican hat: in-sync strengthen, out-of-sync weaken)
        hebbian_kernel = HebbianKernel(
            A_hebb=0.01, 
            tau_hebb=20*ms,        # Narrow window for potentiation
            tau_hebb_wide=40*ms,   # Wide window for depression
            learning_rate=0.1,
            weight_bounds=(0, 0.5*nA)
        )
        return {'E->E': hebbian_kernel}
    
    elif plasticity_type == 'mixed':
        # Different rules for different connections
        stdp_kernel = STDPKernel(A_pre=0.01, A_post=-0.0105, tau_pre=20*ms, tau_post=20*ms, 
                                learning_rate=0.1, weight_bounds=(0, 0.5*nA))
        hebbian_kernel = HebbianKernel(A_hebb=0.01, tau_hebb=20*ms, tau_hebb_wide=40*ms,
                                      learning_rate=0.05, weight_bounds=(0, 0.5*nA))
        anti_hebbian_kernel = AntiHebbianKernel(A_anti=0.01, tau_anti=20*ms, 
                                               learning_rate=0.05, weight_bounds=(0, 0.5*nA))
        
        return {
            'E->E': stdp_kernel,
            'E->I': hebbian_kernel,
            'I->E': None,
            'I->I': anti_hebbian_kernel
        }
    
    else:
        raise ValueError(f"Unknown plasticity type: {plasticity_type}")


def main():
    """Main simulation function."""
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    print("="*70)
    print("PLASTICITY DEMO: Single Stimulus Weight Evolution")
    print("="*70)
    
    # Create output folder with structure: <rule>/<timestamp>/Figures
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = os.path.join(os.path.dirname(__file__), "Runs", "plasticity_demo")
    config_folder = os.path.join(base_folder, CURRENT_CONFIG_LABEL)
    save_folder = os.path.join(config_folder, PLASTICITY_TYPE, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    figures_folder = os.path.join(save_folder, "Figures")
    os.makedirs(figures_folder, exist_ok=True)
    
    # Calculate weight scaling
    k_input = N_EXTERNAL_INPUT_DIMS
    k_recurrent = N_REC_E + N_REC_I
    
    if WEIGHT_SCALE_METHOD == 'k':
        input_weight_scale_eff = BASE_WEIGHT_SCALE / k_input
        r_w_scale_eff = BASE_WEIGHT_SCALE / k_recurrent
    elif WEIGHT_SCALE_METHOD == 'sqrt_k':
        input_weight_scale_eff = BASE_WEIGHT_SCALE / np.sqrt(k_input) * INPUT_AMPLITUDE_SCALE
        r_w_scale_eff = BASE_WEIGHT_SCALE / np.sqrt(k_recurrent)
    else:
        raise ValueError(f"Unknown weight_scale_method: {WEIGHT_SCALE_METHOD}")
    
    print(f"\n>> Network Configuration:")
    print(f"   Neurons: {N_REC_E}E + {N_REC_I}I = {N_REC_E + N_REC_I} total")
    print(f"   Input dimensions: {N_EXTERNAL_INPUT_DIMS}")
    print(f"   Simulation duration: {SIMULATION_DURATION}ms")
    print(f"   Plasticity type: {PLASTICITY_TYPE}")
    print(f"   Weight monitoring every {WEIGHT_MONITORING_DT/ms}ms")
    print(f"   Config profile: {CURRENT_CONFIG_LABEL}")
    
    # Generate recurrent weight matrix
    lognormal_params_rec = [0.0, 0.5, (N_REC_E + N_REC_I, N_REC_E + N_REC_I)]
    weight_rec = r_w_scale_eff * wm.weight_matrix_factory(
        {'recurrent': [N_REC_E, N_REC_I]}, 
        {'lognormal': lognormal_params_rec}
    )
    
    # Apply inhibitory scaling
    weight_rec[N_REC_E:, :N_REC_E] *= INHIBITORY_WEIGHT_SCALE  # I->E
    weight_rec[N_REC_E:, N_REC_E:] *= INHIBITORY_WEIGHT_SCALE  # I->I
    
    print(f"\n>> Recurrent weight statistics:")
    print(f"   E->E: mean={np.mean(weight_rec[:N_REC_E, :N_REC_E]):.4f}, std={np.std(weight_rec[:N_REC_E, :N_REC_E]):.4f}")
    print(f"   I->E: mean={np.mean(weight_rec[:N_REC_E, N_REC_E:]):.4f}, std={np.std(weight_rec[:N_REC_E, N_REC_E:]):.4f}")
    
    # Generate input vectors (constant stimulus throughout)
    dt_ms = defaultclock.dt / ms
    n_timesteps = int(SIMULATION_DURATION / dt_ms)
    
    print(f"\n>> Generating constant input stimulus...")
    print(f"   Time steps: {n_timesteps}")
    print(f"   dt: {dt_ms}ms")
    
    input_vectors, stimulus_pattern = generate_constant_input_vector(
        N_EXTERNAL_INPUT_DIMS, 
        n_timesteps,
        distribution=STIMULUS_DISTRIBUTION,
        alpha=STIMULUS_ALPHA,
        probabilistic_input=PROBABILISTIC_INPUT
    )
    
    # Create stimulus info (single constant stimulus)
    stims_by_time = np.full(n_timesteps, STIMULUS_ID)
    stim_info = [(STIMULUS_ID, 0*ms, SIMULATION_DURATION*ms)]
    
    # Configure plasticity
    plasticity_config = configure_plasticity(PLASTICITY_TYPE)
    print(f"\n>> Plasticity configuration:")
    for conn_type, kernel in plasticity_config.items():
        if kernel is not None:
            print(f"   {conn_type}: {kernel.__class__.__name__}")
        else:
            print(f"   {conn_type}: No plasticity")
    
    # Create network with plasticity
    print(f"\n>> Creating network...")
    lognormal_params_in = [0.0, 0.5, (N_REC_E + N_REC_I, N_EXTERNAL_INPUT_DIMS)]
    network = PlasticDirectInputNetwork(
        N_REC_E, N_REC_I, N_EXTERNAL_INPUT_DIMS,
        weight_rec, input_weight_scale_eff, lognormal_params_in,
        is_feedforward=False,
        plasticity_config=plasticity_config,
        weight_monitoring_dt=WEIGHT_MONITORING_DT
    )
    
    # Check initial E->E weight distribution before simulation
    initial_E_weights = network.rec_synapses_E.w_syn / nA
    n_E_to_E = N_REC_E * N_REC_E  # Number of E->E connections
    initial_E_to_E_weights = initial_E_weights[:n_E_to_E]
    print(f"\n>> Initial E->E weight distribution (before simulation):")
    print(f"   Mean: {initial_E_to_E_weights.mean():.6f} nA")
    print(f"   Std: {initial_E_to_E_weights.std():.6f} nA")
    print(f"   Min: {initial_E_to_E_weights.min():.6f} nA")
    print(f"   Max: {initial_E_to_E_weights.max():.6f} nA")
    print(f"   Sample of 10 weights: {initial_E_to_E_weights[:10]}")
    
    # Run simulation
    print(f"\n>> Running simulation...")
    T = SIMULATION_DURATION * ms
    network.run_simulation(
        T, input_vectors, stims_by_time, stim_info, 0*nA,
        monitoring_level='basic',
        unitary_current=UNITARY_CURRENT,
        num_epochs=1
    )
    
    # Analyze results
    print(f"\n>> Analyzing results...")
    
    # Calculate firing rates
    total_time_s = T / second
    e_spikes = network.rec_spikes.count[:N_REC_E].sum()
    i_spikes = network.rec_spikes.count[N_REC_E:].sum()
    e_rate = e_spikes / (N_REC_E * total_time_s)
    i_rate = i_spikes / (N_REC_I * total_time_s)
    
    print(f"\n>> Firing rates:")
    print(f"   Excitatory: {e_rate:.2f} Hz")
    print(f"   Inhibitory: {i_rate:.2f} Hz")
    
    # Get weight evolution
    if 'E->E' in plasticity_config and plasticity_config['E->E'] is not None:
        times, weights = network.get_weight_evolution('E->all')
        
        # CRITICAL: Separate E->E from E->I connections
        # E->all includes: [E->E (50x50=2500), E->I (50x50=2500)] = 5000 total
        # Weights are stored as flattened: [source0->all_targets, source1->all_targets, ...]
        # For each source: first N_rec_E targets are E, next N_rec_I targets are I
        
        n_sources = N_REC_E
        n_targets_total = N_REC_E + N_REC_I
        n_E_targets = N_REC_E
        
        # Extract only E->E connections (plasticity is only on these)
        E_to_E_indices = []
        for src in range(n_sources):
            start_idx = src * n_targets_total
            end_idx = start_idx + n_E_targets
            E_to_E_indices.extend(range(start_idx, end_idx))
        
        weights_E_to_E = weights[E_to_E_indices, :]
        
        print(f"\n>> Weight evolution recorded:")
        print(f"   Total E->all synapses: {weights.shape[0]}")
        print(f"   E->E synapses (with plasticity): {weights_E_to_E.shape[0]}")
        print(f"   E->I synapses (no plasticity): {weights.shape[0] - weights_E_to_E.shape[0]}")
        print(f"   Time points: {weights.shape[1]}")
        print(f"\n   ALL E->all weights:")
        print(f"      Initial mean: {weights[:, 0].mean():.4f} nA")
        print(f"      Final mean: {weights[:, -1].mean():.4f} nA")
        print(f"      Change: {(weights[:, -1].mean() - weights[:, 0].mean()):.4f} nA")
        print(f"\n   E->E weights ONLY (with plasticity):")
        print(f"      Initial mean: {weights_E_to_E[:, 0].mean():.4f} nA")
        print(f"      Final mean: {weights_E_to_E[:, -1].mean():.4f} nA")
        print(f"      Change: {(weights_E_to_E[:, -1].mean() - weights_E_to_E[:, 0].mean()):.4f} nA")
        
        # Check weight variability at early timepoints (E->E only)
        print(f"\n>> E->E weight variability over time:")
        early_timepoints = [0, 1, 2, 5, 10, 20, 50, -1]  # indices into time array
        for t_idx in early_timepoints:
            if t_idx == -1 or t_idx < weights_E_to_E.shape[1]:
                t_val = times[t_idx] if t_idx >= 0 else times[-1]
                w_at_t = weights_E_to_E[:, t_idx]
                print(f"   t={t_val:6.1f}ms: mean={w_at_t.mean():.6f} nA, std={w_at_t.std():.6f} nA, min={w_at_t.min():.6f} nA, max={w_at_t.max():.6f} nA")
        
        # Print detailed sample of individual E->E synapse weight changes
        # Show BOTH potentiating and depressing synapses
        all_changes = weights_E_to_E[:, -1] - weights_E_to_E[:, 0]
        
        # Find potentiating and depressing synapses
        potentiating_idx = np.where(all_changes > 0.01)[0]  # significant potentiation
        depressing_idx = np.where(all_changes < -0.001)[0]  # any depression
        
        print(f"\n>> E->E synapse weight changes:")
        print(f"   Potentiating synapses: {len(potentiating_idx)}")
        print(f"   Depressing synapses: {len(depressing_idx)}")
        print(f"   Unchanged synapses: {weights_E_to_E.shape[0] - len(potentiating_idx) - len(depressing_idx)}")
        
        print(f"\n>> Sample of POTENTIATING E->E synapses:")
        print(f"   {'Synapse':<10} {'Initial (nA)':<15} {'Final (nA)':<15} {'Change (nA)':<15} {'Change (%)':<12}")
        print(f"   {'-'*70}")
        
        # Sample 10 potentiating synapses
        if len(potentiating_idx) > 0:
            n_samples = min(10, len(potentiating_idx))
            sample_indices = np.random.choice(potentiating_idx, n_samples, replace=False)
            sample_indices = np.sort(sample_indices)
            
            for idx in sample_indices:
                initial = weights_E_to_E[idx, 0]
                final = weights_E_to_E[idx, -1]
                change = final - initial
                pct_change = 100 * change / initial if initial > 0 else 0
                print(f"   {idx:<10} {initial:<15.6f} {final:<15.6f} {change:<15.6f} {pct_change:<12.2f}")
        
        print(f"\n>> Sample of DEPRESSING E->E synapses:")
        print(f"   {'Synapse':<10} {'Initial (nA)':<15} {'Final (nA)':<15} {'Change (nA)':<15} {'Change (%)':<12}")
        print(f"   {'-'*70}")
        
        # Sample 10 depressing synapses
        if len(depressing_idx) > 0:
            n_samples = min(10, len(depressing_idx))
            sample_indices = np.random.choice(depressing_idx, n_samples, replace=False)
            sample_indices = np.sort(sample_indices)
            
            for idx in sample_indices:
                initial = weights_E_to_E[idx, 0]
                final = weights_E_to_E[idx, -1]
                change = final - initial
                pct_change = 100 * change / initial if initial > 0 else 0
                print(f"   {idx:<10} {initial:<15.6f} {final:<15.6f} {change:<15.6f} {pct_change:<12.2f}")
        else:
            print(f"   No depressing synapses found!")
        
        # Print statistics about E->E weight changes
        all_changes = weights_E_to_E[:, -1] - weights_E_to_E[:, 0]
        print(f"\n>> E->E weight change statistics:")
        print(f"   Mean change: {all_changes.mean():.6f} nA")
        print(f"   Std of changes: {all_changes.std():.6f} nA")
        print(f"   Min change: {all_changes.min():.6f} nA")
        print(f"   Max change: {all_changes.max():.6f} nA")
        print(f"   Range: {all_changes.max() - all_changes.min():.6f} nA")
    
    # Generate plots
    print(f"\n>> Generating plots...")
    generate_plasticity_analysis_plots(network, figures_folder, PLASTICITY_TYPE, plasticity_config)
    
    print(f"\n>> Simulation complete!")
    print(f"   Results saved to: {save_folder}")
    print("="*70)


def generate_plasticity_analysis_plots(network, figures_folder, plasticity_type, plasticity_config=None):
    """Generate analysis plots for plasticity demo."""
    import weight_evolution_analysis as wea
    
    # 0. Plasticity kernel visualization
    if plasticity_config and plasticity_type != 'none':
        fig_kernel, ax_kernel = plt.subplots(figsize=(10, 6))
        for conn_type, kernel in plasticity_config.items():
            if kernel is not None:
                label = f"{conn_type}: {kernel.__class__.__name__}"
                kernel.visualize(delta_t_range=(-100, 100), n_points=1000, ax=ax_kernel, label=label)
        
        ax_kernel.set_title(f'Plasticity Kernel: {plasticity_type}', fontsize=14, fontweight='bold')
        ax_kernel.legend()
        fig_kernel.savefig(os.path.join(figures_folder, 'plasticity_kernel.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close(fig_kernel)
        print(f"   Saved: plasticity_kernel.png")
    
    # 1. Weight evolution over time
    if plasticity_type != 'none':
        fig1 = wea.plot_weight_evolution(network, 'E->all')
        fig1.savefig(os.path.join(figures_folder, 'weight_evolution.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"   Saved: weight_evolution.png")
        
        # 2. Weight distribution changes
        fig2 = wea.plot_weight_distribution_evolution(network, 'E->all')
        fig2.savefig(os.path.join(figures_folder, 'weight_distribution.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"   Saved: weight_distribution.png")
        
        # 3. Weight matrix visualization at different times
        fig3 = wea.plot_weight_matrices_timeline(network, 'E->all')
        fig3.savefig(os.path.join(figures_folder, 'weight_matrices_timeline.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"   Saved: weight_matrices_timeline.png")
    
    # 4. Raster plot
    fig4, ax = plt.subplots(figsize=(12, 6))
    spike_times = network.rec_spikes.t / ms
    spike_neurons = network.rec_spikes.i
    
    # Separate E and I
    e_mask = spike_neurons < network.N_rec_E
    i_mask = spike_neurons >= network.N_rec_E
    
    ax.scatter(spike_times[e_mask], spike_neurons[e_mask], s=1, c='red', 
              alpha=0.5, label='Excitatory', rasterized=True)
    ax.scatter(spike_times[i_mask], spike_neurons[i_mask], s=1, c='blue', 
              alpha=0.5, label='Inhibitory', rasterized=True)
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Neuron ID', fontsize=12)
    ax.set_title('Spike Raster Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(False)
    
    fig4.savefig(os.path.join(figures_folder, 'raster_plot.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"   Saved: raster_plot.png")
    
    # 5. Mean voltage traces
    if network.rec_voltage is not None:
        fig5, ax = plt.subplots(figsize=(12, 4))
        
        time_ms = network.rec_voltage.t / ms
        
        # Get E and I neuron voltage traces
        e_voltage = np.mean(network.rec_voltage.v[:network.N_rec_E, :], axis=0) / mV
        i_voltage = np.mean(network.rec_voltage.v[network.N_rec_E:, :], axis=0) / mV
        
        ax.plot(time_ms, e_voltage, 'r-', linewidth=1.5, alpha=0.7, label='Excitatory')
        ax.plot(time_ms, i_voltage, 'b-', linewidth=1.5, alpha=0.7, label='Inhibitory')
        
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Mean Membrane Potential (mV)', fontsize=12)
        ax.set_title('Mean Voltage Traces', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig5.savefig(os.path.join(figures_folder, 'voltage_traces.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close(fig5)
        print(f"   Saved: voltage_traces.png")
    
    # 6. Smoothed firing rates over time (population + individuals)
    fig6, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Compute firing rates in time bins
    spike_times = network.rec_spikes.t / ms
    spike_neurons = network.rec_spikes.i
    
    # Separate E and I
    e_mask = spike_neurons < network.N_rec_E
    i_mask = spike_neurons >= network.N_rec_E
    
    e_spike_times = spike_times[e_mask]
    i_spike_times = spike_times[i_mask]
    
    # Bin spikes
    bin_width = 50  # ms
    max_time = spike_times.max() if len(spike_times) > 0 else 1000
    bins = np.arange(0, max_time + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    e_counts, _ = np.histogram(e_spike_times, bins=bins)
    i_counts, _ = np.histogram(i_spike_times, bins=bins)
    
    # Convert to rates (Hz)
    e_rates = e_counts / (network.N_rec_E * (bin_width / 1000.0))
    i_rates = i_counts / (network.N_rec_I * (bin_width / 1000.0))
    
    # Smooth with moving average
    window_size = 5
    e_rates_smooth = uniform_filter1d(e_rates, size=window_size, mode='nearest')
    i_rates_smooth = uniform_filter1d(i_rates, size=window_size, mode='nearest')
    
    # Top panel: Population average
    ax1.plot(bin_centers, e_rates_smooth, 'r-', linewidth=2, alpha=0.8, label='Excitatory (avg)')
    ax1.plot(bin_centers, i_rates_smooth, 'b-', linewidth=2, alpha=0.8, label='Inhibitory (avg)')
    
    ax1.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax1.set_title('Smoothed Population Firing Rates', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Individual neurons
    # Sample a few neurons to show
    n_sample_e = min(5, network.N_rec_E)
    n_sample_i = min(5, network.N_rec_I)
    sample_e_neurons = np.random.choice(network.N_rec_E, n_sample_e, replace=False)
    sample_i_neurons = np.random.choice(network.N_rec_I, n_sample_i, replace=False) + network.N_rec_E
    
    for neuron_id in sample_e_neurons:
        neuron_spike_times = spike_times[spike_neurons == neuron_id]
        if len(neuron_spike_times) > 0:
            neuron_counts, _ = np.histogram(neuron_spike_times, bins=bins)
            neuron_rates = neuron_counts / (bin_width / 1000.0)
            neuron_rates_smooth = uniform_filter1d(neuron_rates, size=window_size, mode='nearest')
            ax2.plot(bin_centers, neuron_rates_smooth, 'r-', linewidth=1, alpha=0.4)
    
    for neuron_id in sample_i_neurons:
        neuron_spike_times = spike_times[spike_neurons == neuron_id]
        if len(neuron_spike_times) > 0:
            neuron_counts, _ = np.histogram(neuron_spike_times, bins=bins)
            neuron_rates = neuron_counts / (bin_width / 1000.0)
            neuron_rates_smooth = uniform_filter1d(neuron_rates, size=window_size, mode='nearest')
            ax2.plot(bin_centers, neuron_rates_smooth, 'b-', linewidth=1, alpha=0.4)
    
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax2.set_title(f'Individual Neurons ({n_sample_e}E + {n_sample_i}I)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig6.savefig(os.path.join(figures_folder, 'firing_rates.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig6)
    print(f"   Saved: firing_rates.png")
    
    # 7. Individual synapse weight trajectories
    if plasticity_type != 'none':
        times_wt, weights_wt = network.get_weight_evolution('E->all')
        
        # Extract E->E weights
        n_sources = network.N_rec_E
        n_targets_total = network.N_rec_E + network.N_rec_I
        n_E_targets = network.N_rec_E
        
        E_to_E_indices = []
        for src in range(n_sources):
            start_idx = src * n_targets_total
            end_idx = start_idx + n_E_targets
            E_to_E_indices.extend(range(start_idx, end_idx))
        
        weights_E_to_E = weights_wt[E_to_E_indices, :]
        
        # Find synapses with different behaviors
        all_changes = weights_E_to_E[:, -1] - weights_E_to_E[:, 0]
        
        # Categories: potentiation, stable, depression
        potentiating = np.where(all_changes > 0.001)[0]
        stable = np.where((all_changes >= -0.001) & (all_changes <= 0.001))[0]
        depressing = np.where(all_changes < -0.001)[0]
        
        fig7, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.flatten()
        
        categories = [
            (potentiating, 'Potentiating', 'red'),
            (stable, 'Stable', 'gray'),
            (depressing, 'Depressing', 'blue')
        ]
        
        for idx, (category, title, color) in enumerate(categories):
            ax = axes[idx]
            if len(category) > 0:
                # Sample up to 20 synapses from this category
                n_sample = min(20, len(category))
                sample_idx = np.random.choice(category, n_sample, replace=False)
                
                for syn_idx in sample_idx:
                    ax.plot(times_wt, weights_E_to_E[syn_idx, :], alpha=0.5, linewidth=0.8, color=color)
                
                ax.set_title(f'{title} (n={len(category)})', fontsize=11, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No synapses', ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{title} (n=0)', fontsize=11, fontweight='bold')
            
            ax.set_xlabel('Time (ms)', fontsize=10)
            ax.set_ylabel('Weight (nA)', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Summary panel
        ax_summary = axes[3]
        categories_summary = ['Potentiating', 'Stable', 'Depressing']
        counts = [len(potentiating), len(stable), len(depressing)]
        colors_bar = ['red', 'gray', 'blue']
        
        ax_summary.bar(categories_summary, counts, color=colors_bar, alpha=0.7, edgecolor='black')
        ax_summary.set_ylabel('Number of Synapses', fontsize=11)
        ax_summary.set_title('Distribution of Weight Changes', fontsize=12, fontweight='bold')
        ax_summary.grid(True, alpha=0.3, axis='y')
        
        # Add text annotations
        for i, (cat, count) in enumerate(zip(categories_summary, counts)):
            ax_summary.text(i, count + max(counts)*0.02, str(count), 
                          ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        fig7.savefig(os.path.join(figures_folder, 'individual_synapse_trajectories.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close(fig7)
        print(f"   Saved: individual_synapse_trajectories.png")
    
    # 8. Plasticity traces for sample synapses
    if plasticity_type != 'none' and hasattr(network, 'trace_sample_indices') and network.trace_sample_indices is not None:
        trace_narrow_monitor = network.weight_monitors.get('E_trace_narrow')
        trace_wide_monitor = network.weight_monitors.get('E_trace_wide')
        
        if trace_narrow_monitor is not None and trace_wide_monitor is not None:
            # Plot traces for 6 sample synapses
            n_synapses_to_plot = min(6, len(network.trace_sample_indices))
            fig8, axes = plt.subplots(3, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for idx in range(n_synapses_to_plot):
                ax = axes[idx]
                syn_idx = network.trace_sample_indices[idx]
                
                times_trace = trace_narrow_monitor.t / ms
                trace_narrow = trace_narrow_monitor.apost_narrow[idx]
                trace_wide = trace_wide_monitor.apost_wide[idx]
                
                # Compute effective plasticity signal
                # Note: wide trace is already scaled to 0.5*A_hebb at increment time
                dw_hebb = (trace_narrow - trace_wide)
                
                # Create twin axes for weight
                ax2 = ax.twinx()
                
                # Plot traces on left axis
                ax.plot(times_trace, trace_narrow, 'g-', linewidth=1.5, alpha=0.7, label='Narrow (A, τ=20ms)')
                ax.plot(times_trace, trace_wide, 'purple', linewidth=1.5, alpha=0.7, label='Wide (0.5A, τ=40ms)')
                ax.plot(times_trace, dw_hebb, 'k-', linewidth=2, alpha=0.8, label='Net (narrow - wide)')
                ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
                
                # Plot weight on right axis
                weight_times, weight_vals = network.get_weight_evolution('E->all')
                # Get this synapse's weight
                E_to_all_idx = syn_idx  # Since we're monitoring E synapses
                ax2.plot(weight_times, weight_vals[E_to_all_idx, :], 'r-', linewidth=1, alpha=0.5, label='Weight')
                
                ax.set_xlabel('Time (ms)', fontsize=10)
                ax.set_ylabel('Trace Value', fontsize=10, color='black')
                ax2.set_ylabel('Weight (nA)', fontsize=10, color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax.set_title(f'Synapse {syn_idx}', fontsize=11, fontweight='bold')
                ax.legend(loc='upper left', fontsize=8)
                ax2.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig8.savefig(os.path.join(figures_folder, 'plasticity_traces.png'), 
                        dpi=150, bbox_inches='tight')
            plt.close(fig8)
            print(f"   Saved: plasticity_traces.png")


if __name__ == "__main__":
    args = parse_cli_args()
    config_path = resolve_config_path(args.config)
    config_dict, config_label = load_config_dict(config_path)
    apply_config_overrides(config_dict, config_label)
    main()

