"""
Grid search over E-to-I and I-to-E weight scales.
Explores all combinations of weight scales on a log scale.
Saves spike data and computes summary statistics.
"""

import os
from datetime import datetime
import pickle
import random

import numpy as np
from brian2 import *

from brian_experiment.recurrent import network as brn
from brian_experiment.recurrent import utils as rd_utils
from brian_experiment.shared.core import weight_matrices as wm
from brian_experiment.shared.plotting import plot_grid_search_results as plotter

# ======================================================================================
#                                 GRID SEARCH CONFIGURATION
# ======================================================================================
# Network Architecture (fixed 50/50 ratio)
N_REC_E = 100
N_REC_I = 100

# Simulation Timing
SIMULATION_DURATION = 1000  # 1 second per run

# Weight Scaling
BASE_WEIGHT_SCALE = 15.0  # Base scale for recurrent weights
WEIGHT_SCALE_METHOD = 'sqrt_k'

# Grid search: All 4 connection types (E-to-E, E-to-I, I-to-E, I-to-I) on log scale
# 4D grid search with separate scales for each parameter

# Log scale values for each parameter (can be customized per connection type)
GRID_SCALES_E_TO_E = [1e-3, 1e-2, 1e-1, 1.0, 10]  
GRID_SCALES_E_TO_I = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
GRID_SCALES_I_TO_E = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
GRID_SCALES_I_TO_I = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

# Background drive
BACKGROUND_CURRENT = 0 * nA

# Initial State
INITIAL_STATE = {
    'type': 'random_voltage',
    'v_min': -70*mV,
    'v_max': -50*mV
}

# Monitoring
MONITORING_LEVEL = 'basic'

rd_utils.setup_numpy_device()

def run_single_configuration(e_to_e_scale, e_to_i_scale, i_to_e_scale, i_to_i_scale, run_idx, base_folder):
    """Run a single configuration and return results"""
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate weight matrix
    N_rec = N_REC_E + N_REC_I
    k_recurrent = N_rec
    
    if WEIGHT_SCALE_METHOD == 'k':
        r_w_scale_eff = BASE_WEIGHT_SCALE / k_recurrent
    elif WEIGHT_SCALE_METHOD == 'sqrt_k':
        r_w_scale_eff = BASE_WEIGHT_SCALE / np.sqrt(k_recurrent)
    else:
        raise ValueError(f"Unknown weight_scale_method: {WEIGHT_SCALE_METHOD}")
    
    # Generate lognormal weights
    lognormal_weights = np.random.lognormal(mean=0.0, sigma=0.5, size=(N_rec, N_rec))
    weight_rec = r_w_scale_eff * lognormal_weights
    
    # Apply Dale's law
    weight_rec[:, :N_REC_E] = np.abs(weight_rec[:, :N_REC_E])  # E sources: positive
    weight_rec[:, N_REC_E:] = -np.abs(weight_rec[:, N_REC_E:])  # I sources: negative
    
    # Apply weight scaling: vary all 4 connection types
    # weight_rec is [target, source] (rows = targets, columns = sources)
    # - Rows: target neurons (0 to N_REC_E = E, N_REC_E to end = I)
    # - Columns: source neurons (0 to N_REC_E = E, N_REC_E to end = I)
    weight_rec[:N_REC_E, :N_REC_E] *= e_to_e_scale  # E→E: E targets from E sources
    weight_rec[N_REC_E:, :N_REC_E] *= e_to_i_scale  # E→I: I targets from E sources
    weight_rec[:N_REC_E, N_REC_E:] *= i_to_e_scale  # I→E: E targets from I sources
    weight_rec[N_REC_E:, N_REC_E:] *= i_to_i_scale  # I→I: I targets from I sources
    
    # Initialize network
    T = SIMULATION_DURATION * ms
    network = brn.BalancedRecurrentNetwork(
        N_REC_E, 
        N_REC_I, 
        weight_rec, 
        initial_state_config=INITIAL_STATE,
        background_current=BACKGROUND_CURRENT,
        verbose=False,
    )
    
    try:
        # Run simulation
        network.run_simulation(T, monitoring_level=MONITORING_LEVEL)
        
        # Analyze spikes
        network.analyze_spikes()
        metrics = rd_utils.analyze_network_activity(network, weight_rec, N_REC_E, N_REC_I)
        results = {
            'e_to_e_scale': e_to_e_scale,
            'e_to_i_scale': e_to_i_scale,
            'i_to_e_scale': i_to_e_scale,
            'i_to_i_scale': i_to_i_scale,
            **metrics,
            'success': True
        }
        
    except Exception as e:
        results = {
            'e_to_e_scale': e_to_e_scale,
            'e_to_i_scale': e_to_i_scale,
            'i_to_e_scale': i_to_e_scale,
            'i_to_i_scale': i_to_i_scale,
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


if __name__ == "__main__":
    # Create folder for grid search
    base_save_folder = os.path.join(os.path.dirname(__file__), "..", "Runs")
    grid_folder_name = f"grid_search_EtoI_ItoE_E{N_REC_E}_I{N_REC_I}"
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
    
    # Run 4D grid search over all connection types
    # Store summary statistics in memory, but save each run's full data to separate pickle
    all_results_summary = []  # Only statistics for plotting, not full spike data
    run_idx = 0
    
    for e_to_e_scale in GRID_SCALES_E_TO_E:
        for e_to_i_scale in GRID_SCALES_E_TO_I:
            for i_to_e_scale in GRID_SCALES_I_TO_E:
                for i_to_i_scale in GRID_SCALES_I_TO_I:
                    results = run_single_configuration(
                        e_to_e_scale, e_to_i_scale, i_to_e_scale, i_to_i_scale, run_idx, run_folder
                    )
                    
                    # Save individual run to separate pickle file
                    # This includes full spike data: spike_times_all and spike_neurons_all
                    # Filename includes all 4 connection type scales
                    filename = (f"run_{run_idx:04d}_EtoE_{e_to_e_scale:.4f}_EtoI_{e_to_i_scale:.4f}_"
                               f"ItoE_{i_to_e_scale:.4f}_ItoI_{i_to_i_scale:.4f}.pkl")
                    run_pickle_file = os.path.join(pickles_folder, filename)
                    with open(run_pickle_file, 'wb') as f:
                        pickle.dump(results, f)
                    # Store summary (without spike data) for quick plotting
                    summary = {
                        'run_idx': run_idx,
                        'e_to_e_scale': results['e_to_e_scale'],
                        'e_to_i_scale': results['e_to_i_scale'],
                        'i_to_e_scale': results['i_to_e_scale'],
                        'i_to_i_scale': results['i_to_i_scale'],
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
                    
                    line = (
                        f"Run {run_idx+1:04d} | E={results['e_rate']:.2f}Hz | I={results['i_rate']:.2f}Hz | "
                        f"EE={results['e_to_e_scale']:.4g} EI={results['e_to_i_scale']:.4g} "
                        f"IE={results['i_to_e_scale']:.4g} II={results['i_to_i_scale']:.4g}"
                    )
                    if not results.get('success', True):
                        error_msg = results.get('error_message', 'failed')
                        line += f" | ERROR: {error_msg}"
                    print(line)
                    run_idx += 1
    
    # Save summary file for quick access
    summary_file = os.path.join(run_folder, "grid_search_summary.pkl")
    with open(summary_file, 'wb') as f:
        pickle.dump(all_results_summary, f)
    
    # Create summary plots (using summary data)
    plotter.create_summary_plots(all_results_summary, figures_folder)
    

