"""
Random search over E-to-I and I-to-E weight scales.
Explores random combinations of weight scales on a log scale.
Saves spike data and computes summary statistics.
Generates a grid of raster plots.
"""

import os
import pickle
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from brian2 import *

from brian_experiment.recurrent import network as brn
from brian_experiment.recurrent import utils as rd_utils
from brian_experiment.shared.core import weight_matrices as wm

# ======================================================================================
#                           PLOT GRID CONFIGURATION (NUMBER OF SIMULATIONS)
# ======================================================================================
# Specify the grid size for the plot. The total number of simulations will be:
# N_SIMULATIONS = PLOT_ROWS * PLOT_COLS
PLOT_ROWS = 3
PLOT_COLS = 3
N_RANDOM_COMBINATIONS = PLOT_ROWS * PLOT_COLS  # Total number of simulations

# ======================================================================================
#                                 RANDOM SEARCH CONFIGURATION
# ======================================================================================
# Network Architecture (fixed 50/50 ratio)
N_REC_E = 100
N_REC_I = 100

# Simulation Timing
SIMULATION_DURATION = 1000  # 1 second per run

# Weight Scaling
BASE_WEIGHT_SCALE = 10.0
WEIGHT_SCALE_METHOD = 'sqrt_k'

# Per-quadrant multiplicative factors (log-uniform sampling)
EE_QUAD_SCALE_MIN = 0.3
EE_QUAD_SCALE_MAX = 1.0
EI_QUAD_SCALE_MIN = 0.1
EI_QUAD_SCALE_MAX = 10.0
IE_QUAD_SCALE_MIN = 0.1
IE_QUAD_SCALE_MAX = 10.0
II_QUAD_SCALE_MIN = 0.1
II_QUAD_SCALE_MAX = 10.0

# Background drive
# May need adjustment after increasing BASE_WEIGHT_SCALE
BACKGROUND_CURRENT = 0.5 * nA  # Will adjust if firing rates are too high

# Initial State
INITIAL_STATE = {
    'type': 'random_voltage',
    'v_min': -70*mV,
    'v_max': -50*mV
}

# Monitoring
MONITORING_LEVEL = 'basic'

rd_utils.setup_numpy_device()

def sample_log_uniform(min_val, max_val, size=None):
    """Sample from a log-uniform distribution between min_val and max_val."""
    if min_val <= 0 or max_val <= 0:
        raise ValueError("Log-uniform bounds must be positive.")
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    return np.exp(np.random.uniform(log_min, log_max, size=size))

def generate_random_parameter_combinations(n_combinations):
    """Generate n random parameter combinations with quadrant scales."""
    combinations = []
    
    for _ in range(n_combinations):
        # Sample per-quadrant multiplicative factors (log-uniform)
        quadrant_scales = {
            'EE': sample_log_uniform(EE_QUAD_SCALE_MIN, EE_QUAD_SCALE_MAX),
            'EI': sample_log_uniform(EI_QUAD_SCALE_MIN, EI_QUAD_SCALE_MAX),
            'IE': sample_log_uniform(IE_QUAD_SCALE_MIN, IE_QUAD_SCALE_MAX),
            'II': sample_log_uniform(II_QUAD_SCALE_MIN, II_QUAD_SCALE_MAX),
        }
        
        combinations.append({
            'quadrant_scales': quadrant_scales
        })
    
    return combinations

def run_single_configuration(quadrant_scales, run_idx, base_folder):
    """Run a single configuration with quadrant scales and return results.
    
    Args:
        quadrant_scales: Dict with 'EE', 'EI', 'IE', 'II' scale factors
        run_idx: Index of this run
        base_folder: Base folder for saving results
    """
    
    # Set random seed for reproducibility (but vary by run_idx to get different weight matrices)
    np.random.seed(42 + run_idx)
    random.seed(42 + run_idx)
    
    # Generate weight matrix
    N_rec = N_REC_E + N_REC_I
    k_recurrent = N_rec
    
    if WEIGHT_SCALE_METHOD == 'k':
        r_w_scale_eff = BASE_WEIGHT_SCALE / k_recurrent
    elif WEIGHT_SCALE_METHOD == 'sqrt_k':
        r_w_scale_eff = BASE_WEIGHT_SCALE / np.sqrt(k_recurrent)
    else:
        raise ValueError(f"Unknown weight_scale_method: {WEIGHT_SCALE_METHOD}")
    
    # Generate lognormal weights with different sigma for different quadrants to increase diversity
    # EE: smaller sigma (less diversity), EI/IE/II: larger sigma (more diversity)
    lognormal_weights = np.zeros((N_rec, N_rec))
    
    # EE quadrant: smaller sigma for less diversity
    lognormal_weights[:N_REC_E, :N_REC_E] = np.random.lognormal(mean=0.0, sigma=0.3, size=(N_REC_E, N_REC_E))
    
    # EI quadrant: larger sigma for more diversity
    lognormal_weights[N_REC_E:, :N_REC_E] = np.random.lognormal(mean=0.0, sigma=0.8, size=(N_REC_I, N_REC_E))
    
    # IE quadrant: larger sigma for more diversity (even though E scales are narrow)
    lognormal_weights[:N_REC_E, N_REC_E:] = np.random.lognormal(mean=0.0, sigma=0.8, size=(N_REC_E, N_REC_I))
    
    # II quadrant: larger sigma for more diversity
    lognormal_weights[N_REC_E:, N_REC_E:] = np.random.lognormal(mean=0.0, sigma=0.8, size=(N_REC_I, N_REC_I))
    
    weight_rec = r_w_scale_eff * lognormal_weights
    
    # Apply Dale's law
    weight_rec[:, :N_REC_E] = np.abs(weight_rec[:, :N_REC_E])  # E sources: positive
    weight_rec[:, N_REC_E:] = -np.abs(weight_rec[:, N_REC_E:])  # I sources: negative
    
    # Apply per-quadrant multiplicative factors
    weight_rec[:N_REC_E, :N_REC_E] *= quadrant_scales['EE']
    weight_rec[N_REC_E:, :N_REC_E] *= quadrant_scales['EI']
    weight_rec[:N_REC_E, N_REC_E:] *= quadrant_scales['IE']
    weight_rec[N_REC_E:, N_REC_E:] *= quadrant_scales['II']
    
    # Record mean values per quadrant before shuffling (actual connectivity strengths)
    quadrant_means = {
        'EE': float(np.mean(weight_rec[:N_REC_E, :N_REC_E])),
        'EI': float(np.mean(weight_rec[N_REC_E:, :N_REC_E])),
        'IE': float(np.mean(weight_rec[:N_REC_E, N_REC_E:])),
        'II': float(np.mean(weight_rec[N_REC_E:, N_REC_E:]))
    }
    
    # Shuffle weights within each quadrant to remove row-wise structure
    # Extract each quadrant, shuffle its values, and put them back
    # EE quadrant: E targets from E sources
    ee_weights = weight_rec[:N_REC_E, :N_REC_E].flatten()
    np.random.shuffle(ee_weights)
    weight_rec[:N_REC_E, :N_REC_E] = ee_weights.reshape(N_REC_E, N_REC_E)
    
    # EI quadrant: I targets from E sources
    ei_weights = weight_rec[N_REC_E:, :N_REC_E].flatten()
    np.random.shuffle(ei_weights)
    weight_rec[N_REC_E:, :N_REC_E] = ei_weights.reshape(N_REC_I, N_REC_E)
    
    # IE quadrant: E targets from I sources
    ie_weights = weight_rec[:N_REC_E, N_REC_E:].flatten()
    np.random.shuffle(ie_weights)
    weight_rec[:N_REC_E, N_REC_E:] = ie_weights.reshape(N_REC_E, N_REC_I)
    
    # II quadrant: I targets from I sources
    ii_weights = weight_rec[N_REC_E:, N_REC_E:].flatten()
    np.random.shuffle(ii_weights)
    weight_rec[N_REC_E:, N_REC_E:] = ii_weights.reshape(N_REC_I, N_REC_I)
    
    # Initialize network and run simulation (suppress all verbose output)
    T = SIMULATION_DURATION * ms
    with rd_utils.suppress_stdout():
        network = brn.BalancedRecurrentNetwork(
            N_REC_E, 
            N_REC_I, 
            weight_rec, 
            initial_state_config=INITIAL_STATE,
            background_current=BACKGROUND_CURRENT
        )
        
        # Run simulation
        network.run_simulation(T, monitoring_level=MONITORING_LEVEL)
        
        # Analyze spikes (also suppress verbose output here)
        network.analyze_spikes()
    
    try:
        metrics = rd_utils.analyze_network_activity(network, weight_rec, N_REC_E, N_REC_I)
        print(
            f"Run {run_idx+1:02d}: EE={quadrant_means['EE']:.3f} EI={quadrant_means['EI']:.3f} "
            f"IE={quadrant_means['IE']:.3f} II={quadrant_means['II']:.3f} | "
            f"E={metrics['e_rate']:.2f}Hz I={metrics['i_rate']:.2f}Hz | "
            f"CVe={metrics['cv_e']:.3f} CVi={metrics['cv_i']:.3f}"
        )
        results = {
            'run_idx': run_idx,
            'quadrant_scales': quadrant_scales,
            'quadrant_means': quadrant_means,
            **metrics,
            'success': True
        }
        
    except Exception as e:
        print(f"Run {run_idx+1:02d}: ERROR - {e}")
        results = {
            'run_idx': run_idx,
            'quadrant_scales': quadrant_scales,
            'quadrant_means': quadrant_means,
            'e_rate': 0,
            'i_rate': 0,
            'cv_e': 0,
            'cv_i': 0,
            'rate_correlation': 0,
            'mean_pairwise_corr_e': 0,
            'mean_pairwise_corr_i': 0,
            'mean_e_weight': 0,
            'mean_i_weight': 0,
            'balance_ratio': 0,
            'spike_times_all': None,
            'spike_neurons_all': None,
            'success': False,
            'error_message': str(e),
        }
    
    return results

def create_raster_plot_grid(all_results, save_folder):
    """Create an adaptive grid of raster plots (square grid covering all runs)."""
    successful_results = [r for r in all_results if r.get('success', False) and r.get('spike_times_all') is not None]
    if len(successful_results) == 0:
        return
    
    n_plots = len(successful_results)
    # Use the configured grid size
    grid_rows = PLOT_ROWS
    grid_cols = PLOT_COLS
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2.5, grid_rows * 2.5))
    axes = np.atleast_2d(axes)
    fig.suptitle(f'Random Search: Raster Plots ({grid_rows}x{grid_cols} Grid)', fontsize=16, fontweight='bold', y=0.995)
    
    for idx in range(grid_rows * grid_cols):
        row = idx // grid_cols
        col = idx % grid_cols
        ax = axes[row, col]
        
        if idx >= n_plots:
            ax.axis('off')
            continue
        
        result = successful_results[idx]
        spike_times_all = result['spike_times_all']
        spike_neurons_all = result['spike_neurons_all']
        
        if spike_times_all is None or len(spike_times_all) == 0:
            ax.text(0.5, 0.5, "No spikes", ha="center", va="center", transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_xlim(0, SIMULATION_DURATION)
            ax.set_ylim(0, N_REC_E + N_REC_I)
            ax.axis('off')
            continue
        
        colors = np.where(spike_neurons_all < N_REC_E, 'red', 'blue')
        ax.scatter(spike_times_all, spike_neurons_all, c=colors, s=0.5, alpha=0.6, rasterized=True)
        ax.axhline(y=N_REC_E - 0.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        e_spikes = spike_times_all[spike_neurons_all < N_REC_E]
        i_spikes = spike_times_all[spike_neurons_all >= N_REC_E]
        e_rate = len(e_spikes) / (N_REC_E * SIMULATION_DURATION / 1000.0) if len(e_spikes) > 0 else 0.0
        i_rate = len(i_spikes) / (N_REC_I * SIMULATION_DURATION / 1000.0) if len(i_spikes) > 0 else 0.0
        
        quadrant_means = result.get('quadrant_means', {})
        if all(k in quadrant_means for k in ['EE', 'EI', 'IE', 'II']):
            title = (f"EE={quadrant_means['EE']:.3f}, EI={quadrant_means['EI']:.3f}, "
                     f"IE={quadrant_means['IE']:.3f}, II={quadrant_means['II']:.3f}\n"
                     f"E:{e_rate:.1f}Hz, I:{i_rate:.1f}Hz")
        else:
            title = f"E:{e_rate:.1f}Hz, I:{i_rate:.1f}Hz"
        ax.set_title(title, fontsize=7, pad=2)
        
        ax.set_xlim(0, SIMULATION_DURATION)
        ax.set_ylim(-0.5, N_REC_E + N_REC_I - 0.5)
        
        # Reconstruct weight matrix for inset visualization
        N_rec = N_REC_E + N_REC_I
        k_recurrent = N_rec
        if WEIGHT_SCALE_METHOD == 'k':
            r_w_scale_eff = BASE_WEIGHT_SCALE / k_recurrent
        elif WEIGHT_SCALE_METHOD == 'sqrt_k':
            r_w_scale_eff = BASE_WEIGHT_SCALE / np.sqrt(k_recurrent)
        else:
            r_w_scale_eff = BASE_WEIGHT_SCALE / np.sqrt(k_recurrent)
        
        quadrant_scales = result.get('quadrant_scales', {'EE': 1.0, 'EI': 1.0, 'IE': 1.0, 'II': 1.0})
        for key in ['EE', 'EI', 'IE', 'II']:
            if key not in quadrant_scales:
                quadrant_scales[key] = 1.0
        
        original_run_idx = result.get('run_idx', idx)
        np.random.seed(42 + original_run_idx)
        
        lognormal_weights = np.zeros((N_rec, N_rec))
        lognormal_weights[:N_REC_E, :N_REC_E] = np.random.lognormal(mean=0.0, sigma=0.3, size=(N_REC_E, N_REC_E))
        lognormal_weights[N_REC_E:, :N_REC_E] = np.random.lognormal(mean=0.0, sigma=0.8, size=(N_REC_I, N_REC_E))
        lognormal_weights[:N_REC_E, N_REC_E:] = np.random.lognormal(mean=0.0, sigma=0.8, size=(N_REC_E, N_REC_I))
        lognormal_weights[N_REC_E:, N_REC_E:] = np.random.lognormal(mean=0.0, sigma=0.8, size=(N_REC_I, N_REC_I))
        
        weight_rec = r_w_scale_eff * lognormal_weights
        weight_rec[:, :N_REC_E] = np.abs(weight_rec[:, :N_REC_E])
        weight_rec[:, N_REC_E:] = -np.abs(weight_rec[:, N_REC_E:])
        
        # Apply per-quadrant multiplicative factors
        weight_rec[:N_REC_E, :N_REC_E] *= quadrant_scales['EE']
        weight_rec[N_REC_E:, :N_REC_E] *= quadrant_scales['EI']
        weight_rec[:N_REC_E, N_REC_E:] *= quadrant_scales['IE']
        weight_rec[N_REC_E:, N_REC_E:] *= quadrant_scales['II']
        
        ee_weights = weight_rec[:N_REC_E, :N_REC_E].flatten()
        np.random.shuffle(ee_weights)
        weight_rec[:N_REC_E, :N_REC_E] = ee_weights.reshape(N_REC_E, N_REC_E)
        
        ei_weights = weight_rec[N_REC_E:, :N_REC_E].flatten()
        np.random.shuffle(ei_weights)
        weight_rec[N_REC_E:, :N_REC_E] = ei_weights.reshape(N_REC_I, N_REC_E)
        
        ie_weights = weight_rec[:N_REC_E, N_REC_E:].flatten()
        np.random.shuffle(ie_weights)
        weight_rec[:N_REC_E, N_REC_E:] = ie_weights.reshape(N_REC_E, N_REC_I)
        
        ii_weights = weight_rec[N_REC_E:, N_REC_E:].flatten()
        np.random.shuffle(ii_weights)
        weight_rec[N_REC_E:, N_REC_E:] = ii_weights.reshape(N_REC_I, N_REC_I)
        
        inset = ax.inset_axes([0.02, 0.02, 0.25, 0.25])
        abs_max = np.max(np.abs(weight_rec))
        inset.imshow(weight_rec, cmap='bwr', aspect='auto', vmin=-abs_max, vmax=abs_max, interpolation='nearest')
        inset.axhline(y=N_REC_E - 0.5, color='black', linestyle='-', linewidth=1)
        inset.axvline(x=N_REC_E - 0.5, color='black', linestyle='-', linewidth=1)
        inset.text(N_REC_E/2, N_REC_E/2, 'EE', ha='center', va='center', fontsize=6, color='white', weight='bold')
        inset.text(N_REC_E/2, N_REC_E + N_REC_I/2, 'EI', ha='center', va='center', fontsize=6, color='white', weight='bold')
        inset.text(N_REC_E + N_REC_I/2, N_REC_E/2, 'IE', ha='center', va='center', fontsize=6, color='white', weight='bold')
        inset.text(N_REC_E + N_REC_I/2, N_REC_E + N_REC_I/2, 'II', ha='center', va='center', fontsize=6, color='white', weight='bold')
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title('Weight Matrix', fontsize=5, pad=2)
        
        if row == grid_rows - 1:
            ax.set_xlabel('Time (ms)', fontsize=7)
        else:
            ax.set_xticks([])
        if col == 0:
            ax.set_ylabel('Neuron', fontsize=7)
        else:
            ax.set_yticks([])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row != grid_rows - 1:
            ax.spines['bottom'].set_visible(False)
        if col != 0:
            ax.spines['left'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    save_path = os.path.join(save_folder, "raster_plot_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Create folder for random search
    base_save_folder = os.path.join(os.path.dirname(__file__), "..", "Runs", "random_search")
    grid_folder_name = f"random_search_EtoI_ItoE_E{N_REC_E}_I{N_REC_I}"
    grid_base_folder = os.path.join(base_save_folder, grid_folder_name)
    
    # Move old runs to "old" folder
    if os.path.exists(grid_base_folder):
        old_folder = os.path.join(grid_base_folder, "old")
        os.makedirs(old_folder, exist_ok=True)
        for item in os.listdir(grid_base_folder):
            item_path = os.path.join(grid_base_folder, item)
            if os.path.isdir(item_path) and item != "old":
                if len(item) == 15 and item[8] == '_' and item.replace('_', '').isdigit():
                    old_item_path = os.path.join(old_folder, item)
                    if os.path.exists(old_item_path):
                        import shutil
                        counter = 1
                        while os.path.exists(old_item_path):
                            old_item_path = os.path.join(old_folder, f"{item}_{counter}")
                            counter += 1
                    os.rename(item_path, old_item_path)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(grid_base_folder, timestamp)
    figures_folder = os.path.join(run_folder, "Figures")
    pickles_folder = os.path.join(run_folder, "Pickles")
    
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(figures_folder, exist_ok=True)
    os.makedirs(pickles_folder, exist_ok=True)
    
    # Generate random parameter combinations (filtered a priori)
    np.random.seed(42)  # Seed for reproducibility of parameter generation
    parameter_combinations = generate_random_parameter_combinations(N_RANDOM_COMBINATIONS)
    
    # Run random search
    # Store summary statistics in memory, but save each run's full data to separate pickle
    all_results_summary = []  # Only statistics for plotting, not full spike data
    run_idx = 0
    total_runs = len(parameter_combinations)
    
    for combo in parameter_combinations:
        quadrant_scales = combo['quadrant_scales']
        results = run_single_configuration(
            quadrant_scales,
            run_idx,
            run_folder
        )
        
        # Save individual run to separate pickle file
        # Filename includes quadrant scales
        filename = (f"run_{run_idx:04d}_EE_{quadrant_scales['EE']:.4f}_EI_{quadrant_scales['EI']:.4f}_"
                   f"IE_{quadrant_scales['IE']:.4f}_II_{quadrant_scales['II']:.4f}.pkl")
        run_pickle_file = os.path.join(pickles_folder, filename)
        with open(run_pickle_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Store summary (without spike data) for quick plotting
        summary = {
            'run_idx': run_idx,
            'quadrant_scales': results.get('quadrant_scales'),
            'quadrant_means': results.get('quadrant_means'),
            'e_rate': results['e_rate'],
            'i_rate': results['i_rate'],
            'cv_e': results['cv_e'],
            'cv_i': results['cv_i'],
            'rate_correlation': results['rate_correlation'],
            'mean_pairwise_corr_e': results['mean_pairwise_corr_e'],
            'mean_pairwise_corr_i': results['mean_pairwise_corr_i'],
            'mean_e_weight': results['mean_e_weight'],
            'mean_i_weight': results['mean_i_weight'],
            'balance_ratio': results['balance_ratio'],
            'success': results['success'],
            'pickle_file': run_pickle_file  # Reference to full data
        }
        all_results_summary.append(summary)
        
        run_idx += 1
    
    # Save summary file for quick access
    summary_file = os.path.join(run_folder, "random_search_summary.pkl")
    with open(summary_file, 'wb') as f:
        pickle.dump(all_results_summary, f)
    
    # Create raster plot grid
    # Need to load full spike data for raster plots
    all_results_with_spikes = []
    for summary in all_results_summary:
        if summary.get('success', False):
            pickle_file = summary.get('pickle_file')
            if pickle_file and os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as f:
                    full_result = pickle.load(f)
                    all_results_with_spikes.append(full_result)
    
    create_raster_plot_grid(all_results_with_spikes, figures_folder)

