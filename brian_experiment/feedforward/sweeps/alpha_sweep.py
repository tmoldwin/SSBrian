import os
import numpy as np
import matplotlib.pyplot as plt
import random
from random import seed as random_seed
import re
from brian2 import *
from datetime import datetime

# --- Custom Module Imports ---
from brian_experiment.feedforward import network as din
from brian_experiment.shared.analysis import sparsity_and_selectivity as ss
from brian_experiment.shared.core import spike_analysis as sa
from brian_experiment.shared.core import spike_analysis_utils as sau
from brian_experiment.shared.core import stimuli as stim
from brian_experiment.shared.core import weight_matrices as wm

# ======================================================================================
#                            ALPHA SWEEP CONFIGURATION
# ======================================================================================
# --- Experiment Setup ---
# List of alpha values to test for the Dirichlet distribution of stimuli.
ALPHA_VALUES = [0.005, 0.05, 0.5, 5.0]
# The primary variable to change for experiments.
NUM_STIMULI = 100
NUM_EPOCHS = 1  # Keep it at 2 to see some evolution, but not too long for a sweep

# --- Network Architecture ---
N_REC_E = 100
N_REC_I = 100
N_EXTERNAL_INPUT_DIMS = 100

# --- Simulation Timing (in ms) ---
STIM_DURATION = 100
BREAK_TIME = 20
START_TIME = 100

# --- Weight & Input Scaling ---
BASE_WEIGHT_SCALE = 1
WEIGHT_SCALE_METHOD = 'sqrt_k'
INHIBITORY_WEIGHT_SCALE = 5
INPUT_AMPLITUDE_SCALE = 1.0

# --- Stimulus Generation ---
STIMULUS_DISTRIBUTION = 'dirichlet'

# --- Input Mode ---
PROBABILISTIC_INPUT = True
UNITARY_CURRENT = 1.0 * nA
# ======================================================================================

def setup_device():
    """Use normal CPU processing"""
    print("Using normal CPU processing")
    prefs.codegen.target = 'numpy'
    print("Using numpy backend to avoid compatibility issues")

def generate_direct_input_vectors(stims_by_time, N_external_input_dims, distribution='dirichlet', alpha=0.5, probabilistic_input=False):
    """
    Generate direct input vectors for each time step based on stimulus schedule.
    """
    input_vectors = np.zeros((N_external_input_dims, len(stims_by_time)))
    unique_stims = np.unique(stims_by_time)
    unique_stims = unique_stims[unique_stims != -1]
    stimulus_patterns = {}
    for stim_id in unique_stims:
        pattern = np.zeros(N_external_input_dims)
        if distribution == 'dirichlet':
            alpha_vec_E = np.full(N_external_input_dims, alpha)
            pattern_E = np.random.dirichlet(alpha_vec_E)
            pattern = pattern_E * 10.0
        elif distribution == 'normal':
            pattern = np.abs(np.random.normal(0, 1, N_external_input_dims))
        elif distribution == 'uniform':
            pattern = np.random.uniform(0, 1, N_external_input_dims)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        stimulus_patterns[stim_id] = pattern

    for t, stim_id in enumerate(stims_by_time):
        if stim_id >= 0:
            pattern = stimulus_patterns[stim_id]
            if probabilistic_input:
                input_vectors[:, t] = np.random.binomial(1, np.clip(pattern, 0, 1))
            else:
                input_vectors[:, t] = pattern
        else:
            input_vectors[:, t] = 0
    return input_vectors, stimulus_patterns

def run_simulation_for_alpha(stimulus_alpha, base_seed):
    """
    Runs a single simulation for a given stimulus alpha and returns sparsity/selectivity.
    This function is a headless version of the main script.
    """
    # Seed everything for reproducibility of this specific run
    np.random.seed(base_seed)
    random_seed(base_seed)
    
    print(f"\n{'='*20} Running for alpha = {stimulus_alpha} {'='*20}")

    stim_list = np.arange(NUM_STIMULI)
    
    # Calculate total simulation time
    epoch_time = (STIM_DURATION + BREAK_TIME) * len(stim_list) * ms
    T = START_TIME * ms + NUM_EPOCHS * epoch_time

    # Calculate scaling factors
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
    
    # Create weight matrices
    lognormal_params_rec = [0.0, 0.5, (N_REC_E + N_REC_I, N_REC_E + N_REC_I)]
    weight_rec = r_w_scale_eff * wm.weight_matrix_factory({'recurrent': [N_REC_E, N_REC_I]}, {'lognormal': lognormal_params_rec})
    weight_rec[N_REC_E:, :N_REC_E] *= INHIBITORY_WEIGHT_SCALE
    weight_rec[N_REC_E:, N_REC_E:] *= INHIBITORY_WEIGHT_SCALE
    
    # Generate stimuli and input vectors
    stims_by_time, stim_info = stim.create_stim_list(T, STIM_DURATION, stim_list, BREAK_TIME)
    if stims_by_time[0] == -1:
        stims_by_time[0] = stim_list[0]
        
    input_vectors, _ = generate_direct_input_vectors(
        stims_by_time, N_EXTERNAL_INPUT_DIMS, 
        distribution=STIMULUS_DISTRIBUTION,
        alpha=stimulus_alpha,
        probabilistic_input=PROBABILISTIC_INPUT
    )
    
    # Initialize and run network
    lognormal_params_in = [0.0, 0.5, (N_REC_E + N_REC_I, N_EXTERNAL_INPUT_DIMS)]
    direct_network = din.DirectInputNetwork(N_REC_E, N_REC_I, N_EXTERNAL_INPUT_DIMS, weight_rec, input_weight_scale_eff, lognormal_params_in)
    direct_network.run_simulation(T, input_vectors, stims_by_time, stim_info, 0*nA, monitoring_level='basic', unitary_current=UNITARY_CURRENT)
    
    # Analyze final epoch
    spike_matrix_rec = sau.spike_matrix_from_monitor(direct_network.rec_spikes, N_REC_E + N_REC_I, T, dt=defaultclock.dt)
    rate_matrices_arr, _, _ = sa.analyze_firing_rates_over_time(
        T, stims_by_time, spike_matrix_rec, NUM_EPOCHS, 
        dt=defaultclock.dt, start_time=START_TIME * ms
    )
    
    if not rate_matrices_arr:
        print(f"WARNING: No spikes recorded for alpha = {stimulus_alpha}. Returning empty results.")
        return np.array([]), np.array([])
        
    last_rate_matrix = rate_matrices_arr[-1]
    
    # We are interested in the E-population
    e_rate_matrix = last_rate_matrix[:N_REC_E, :]
    
    spi_vec, sli_vec = ss.create_vecs(e_rate_matrix)
    
    return np.array(spi_vec), np.array(sli_vec)

if __name__ == "__main__":
    setup_device()
    
    # --- Define save folder ---
    save_folder = os.path.join("Figures", "DI_Sims")
    os.makedirs(save_folder, exist_ok=True)
    print(f"Plot will be saved in: {save_folder}")

    results = {}
    # Use a fixed seed for the network, but the stimulus generation will be different
    # due to different alpha, which is what we want to test.
    simulation_seed = 42

    for alpha in ALPHA_VALUES:
        sparsity, selectivity = run_simulation_for_alpha(alpha, simulation_seed)
        results[alpha] = {'sparsity': sparsity, 'selectivity': selectivity}
        
        # --- Print Mean and Std Dev ---
        if sparsity.size > 0:
            print(f"  Alpha {alpha}: Sparsity    - Mean: {np.mean(sparsity):.3f}, Std: {np.std(sparsity):.3f}")
            print(f"  Alpha {alpha}: Selectivity - Mean: {np.mean(selectivity):.3f}, Std: {np.std(selectivity):.3f}")
        else:
            print(f"  Alpha {alpha}: No spikes recorded.")
        
    # --- Plotting Results ---
    print("\nPlotting final results...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    bins = np.linspace(0, 1, 51)
    
    # Plot Sparsity
    ax1.set_title('Sparsity Distribution of Excitatory Neurons')
    for alpha, data in results.items():
        if data['sparsity'].size > 0:
            ax1.hist(data['sparsity'], bins=bins, histtype='step', linewidth=2, density=True, label=f'Alpha = {alpha}')
    ax1.set_xlabel('Sparsity (SPI)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot Selectivity
    ax2.set_title('Selectivity Distribution of Excitatory Neurons')
    for alpha, data in results.items():
        if data['selectivity'].size > 0:
            ax2.hist(data['selectivity'], bins=bins, histtype='step', linewidth=2, density=True, label=f'Alpha = {alpha}')
    ax2.set_xlabel('Selectivity (SLI)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    fig.suptitle('Effect of Stimulus Alpha on Sparsity and Selectivity', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plot_filename = os.path.join(save_folder, 'alpha_sweep_distributions.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)
    
    print(f"\n🎉 Alpha sweep complete. Plot saved to {plot_filename}") 