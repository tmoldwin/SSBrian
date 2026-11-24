import argparse
import os
import re
import platform
import random as py_random
import pickle
import shutil

import numpy as np
from brian2 import *
from datetime import datetime

from brian_experiment.feedforward import network as din
from brian_experiment.feedforward.configs import config_manager as cfg_manager
from brian_experiment.shared.analysis import sparsity_and_selectivity as ss
from brian_experiment.shared.core import spike_analysis as sa
from brian_experiment.shared.core import spike_analysis_utils as sau
from brian_experiment.shared.core import stimuli as stim
from brian_experiment.shared.core import weight_matrices as wm
from brian_experiment.shared.plotting.generate_plots import generate_all_plots

# Ensure we keep access to Python's random module (Brian2 overrides the name)
random = py_random

# ======================================================================================
#                                 USER CONFIGURATION
# ======================================================================================
# --- Experiment Setup ---
# The primary variable to change for experiments.
NUM_STIMULI = 100  # Number of unique stimuli
# The number of stimuli above which heavy plots (e.g., detailed dynamics) will be skipped.
HEAVY_PLOT_STIMULUS_THRESHOLD = 10

# Active configuration label (overridden by JSON)
CURRENT_RUN_CONFIG_NAME = "random_baseline"

# --- Network Architecture ---
NUM_EPOCHS = 5
N_REC_E = 100
N_REC_I = 100
N_EXTERNAL_INPUT_DIMS = 500

# --- Simulation Timing (in ms) ---
STIM_DURATION = 250
BREAK_TIME = 20
START_TIME = 100

# --- Weight & Input Scaling ---
BASE_WEIGHT_SCALE = 1
WEIGHT_SCALE_METHOD = 'sqrt_k'  # Options: 'k' or 'sqrt_k'
INHIBITORY_WEIGHT_SCALE = 5
INPUT_AMPLITUDE_SCALE = 1.0

# --- Stimulus Generation ---
STIMULUS_DISTRIBUTION = 'dirichlet'
STIMULUS_ALPHA = 0.5

# --- Input Mode ---
# If True, input vectors are probabilities for unitary input; if False, they are current strengths
PROBABILISTIC_INPUT = True
UNITARY_CURRENT = 1.0 * nA

# --- Recurrent Connections ---
# Set to True to disable recurrent connections (feedforward only)
DISABLE_RECURRENT_CONNECTIONS = False

# --- Simulation Mode ---
# Set to True to skip simulation and load existing data for plot generation only
LOAD_EXISTING_DATA = False  # Set to True to load data and regenerate plots only

# --- Plot Configuration ---
# NOTE: Plots are now organized into modular analysis files:
#   network_structure.py → network_structure/           (weight matrices)
#   connectivity_tuning.py → connectivity_tuning/       (CCI functional analysis)
#   temporal_evolution.py → temporal_evolution/         (sparsity/selectivity over time)
#   voltage_current_monitors.py → voltage_current_monitors/  (live voltage/current)
#   correlation_analysis.py → correlation_analysis/     (all correlation analyses)
# All plots are generated automatically based on the plots_config in the main script

# Voltage plot subsampling parameters (instead of binary threshold)
MAX_STIMULI_TO_PLOT = 10  # Maximum stimuli to show in voltage plots (takes first N, not random)
MAX_NEURONS_VOLTAGE_TRACES = 20  # Maximum neurons to show in individual voltage traces
MAX_NEURONS_RASTER = 100  # Maximum neurons to show in raster plot
# Note: voltage plots will always be generated, but with subsampling if needed
    
# ======================================================================================

# Check platform and GPU availability before setting device
def setup_device():
    """Use normal CPU processing"""
    print("Using normal CPU processing")
    # Force pure Python backend to avoid Cython compatibility issues
    prefs.codegen.target = 'numpy'
    print("Using numpy backend to avoid compatibility issues")

# Setup device
setup_device()


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
    global UNITARY_CURRENT
    UNITARY_CURRENT = float(value) * nA


RUN_CONFIG_SETTERS = {
    "num_stimuli": lambda v: _set_int("NUM_STIMULI", v),
    "heavy_plot_stimulus_threshold": lambda v: _set_int("HEAVY_PLOT_STIMULUS_THRESHOLD", v),
    "num_epochs": lambda v: _set_int("NUM_EPOCHS", v),
    "n_rec_e": lambda v: _set_int("N_REC_E", v),
    "n_rec_i": lambda v: _set_int("N_REC_I", v),
    "n_external_input_dims": lambda v: _set_int("N_EXTERNAL_INPUT_DIMS", v),
    "stim_duration": lambda v: _set_int("STIM_DURATION", v),
    "break_time": lambda v: _set_int("BREAK_TIME", v),
    "start_time": lambda v: _set_int("START_TIME", v),
    "base_weight_scale": lambda v: _set_float("BASE_WEIGHT_SCALE", v),
    "weight_scale_method": lambda v: _set_str("WEIGHT_SCALE_METHOD", v),
    "inhibitory_weight_scale": lambda v: _set_float("INHIBITORY_WEIGHT_SCALE", v),
    "input_amplitude_scale": lambda v: _set_float("INPUT_AMPLITUDE_SCALE", v),
    "stimulus_distribution": lambda v: _set_str("STIMULUS_DISTRIBUTION", v),
    "stimulus_alpha": lambda v: _set_float("STIMULUS_ALPHA", v),
    "probabilistic_input": lambda v: _set_bool("PROBABILISTIC_INPUT", v),
    "unitary_current_nA": _set_unitary_current,
    "disable_recurrent": lambda v: _set_bool("DISABLE_RECURRENT_CONNECTIONS", v),
    "load_existing_data": lambda v: _set_bool("LOAD_EXISTING_DATA", v),
    "max_stimuli_to_plot": lambda v: _set_int("MAX_STIMULI_TO_PLOT", v),
    "max_neurons_voltage_traces": lambda v: _set_int("MAX_NEURONS_VOLTAGE_TRACES", v),
    "max_neurons_raster": lambda v: _set_int("MAX_NEURONS_RASTER", v),
}


DEFAULT_RUN_CONFIG = {
    "num_stimuli": NUM_STIMULI,
    "heavy_plot_stimulus_threshold": HEAVY_PLOT_STIMULUS_THRESHOLD,
    "num_epochs": NUM_EPOCHS,
    "n_rec_e": N_REC_E,
    "n_rec_i": N_REC_I,
    "n_external_input_dims": N_EXTERNAL_INPUT_DIMS,
    "stim_duration": STIM_DURATION,
    "break_time": BREAK_TIME,
    "start_time": START_TIME,
    "base_weight_scale": BASE_WEIGHT_SCALE,
    "weight_scale_method": WEIGHT_SCALE_METHOD,
    "inhibitory_weight_scale": INHIBITORY_WEIGHT_SCALE,
    "input_amplitude_scale": INPUT_AMPLITUDE_SCALE,
    "stimulus_distribution": STIMULUS_DISTRIBUTION,
    "stimulus_alpha": STIMULUS_ALPHA,
    "probabilistic_input": PROBABILISTIC_INPUT,
    "unitary_current_nA": UNITARY_CURRENT / nA,
    "disable_recurrent": DISABLE_RECURRENT_CONNECTIONS,
    "load_existing_data": LOAD_EXISTING_DATA,
    "max_stimuli_to_plot": MAX_STIMULI_TO_PLOT,
    "max_neurons_voltage_traces": MAX_NEURONS_VOLTAGE_TRACES,
    "max_neurons_raster": MAX_NEURONS_RASTER,
}


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run a feedforward DirectInput network simulation using JSON configs."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Name or path to the run config JSON (falls back to default random baseline).",
    )
    return parser.parse_args()


def apply_run_config_overrides(config_dict, label):
    global CURRENT_RUN_CONFIG_NAME
    if label:
        CURRENT_RUN_CONFIG_NAME = label

    for key, value in config_dict.items():
        setter = RUN_CONFIG_SETTERS.get(key)
        if setter is None:
            continue
        setter(value)


def save_run_parameters(config, run_type, fixed_params, folder_path):
    """Save run parameters to a text file in the timestamped folder"""
    param_file = os.path.join(folder_path, "run_parameters.txt")
    
    with open(param_file, 'w') as f:
        f.write(f"Direct Input RNN Simulation Run Parameters\n")
        f.write(f"{'='*45}\n")
        f.write(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration Type: {run_type}\n\n")
        
        f.write("Configuration Parameters:\n")
        f.write(f"{'='*25}\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nFixed Parameters:\n")
        f.write(f"{'='*17}\n")
        for key, value in fixed_params.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nDerived Parameters:\n")
        f.write(f"{'='*19}\n")
        f.write(f"Total neurons: {config['N_rec_E'] + config['N_rec_I']}\n")
        f.write(f"Total input dimensions: {config['N_external_input_dims']}\n")
        f.write(f"Stimuli count: {len(config['stim_list'])}\n")
        f.write(f"Epoch duration: {(fixed_params['stim_duration'] + fixed_params['break_time']) * len(config['stim_list'])} ms\n")
        f.write(f"Total simulation time: {fixed_params['start_time'] + config['num_epochs'] * (fixed_params['stim_duration'] + fixed_params['break_time']) * len(config['stim_list'])} ms\n")

def generate_direct_input_vectors(stims_by_time, N_external_input_dims, distribution='dirichlet', alpha=0.5, probabilistic_input=False):
    """
    Generate direct input vectors for each time step based on stimulus schedule.
    Now supports only excitatory external input dimensions.
    """
    input_vectors = np.zeros((N_external_input_dims, len(stims_by_time)))
    unique_stims = np.unique(stims_by_time)
    unique_stims = unique_stims[unique_stims != -1]
    stimulus_patterns = {}
    for stim_id in unique_stims:
        pattern = np.zeros(N_external_input_dims)
        if distribution == 'dirichlet':
            # Generate E input pattern
            alpha_vec_E = np.full(N_external_input_dims, alpha)
            pattern_E = np.random.dirichlet(alpha_vec_E)
            pattern = pattern_E * 10.0  # Scale up dirichlet values to make them stronger
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
                # Treat pattern values as probabilities and sample them
                input_vectors[:, t] = np.random.binomial(1, np.clip(pattern, 0, 1))
            else:
                input_vectors[:, t] = pattern
        else:
            input_vectors[:, t] = 0
    return input_vectors, stimulus_patterns

def save_simulation_data(data_dict, folder_path):
    """Save simulation data to pickle file in Data subfolder"""
    data_folder = os.path.join(folder_path, "Data")
    data_file = os.path.join(data_folder, "simulation_data.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"   Simulation data saved to: {data_file}")

def load_simulation_data(folder_path):
    """Load simulation data from pickle file in Data subfolder"""
    data_file = os.path.join(folder_path, "Data", "simulation_data.pkl")
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"   Simulation data loaded from: {data_file}")
        return data_dict
    else:
        print(f"   No simulation data found at: {data_file}")
        return None

# NETWORK ARCHITECTURE DESCRIPTION:
# - N_rec_E excitatory neurons and N_rec_I inhibitory neurons
# - N_external_input_dims excitatory input dimensions that can connect to both E and I neurons
# - Each excitatory input dimension provides direct current injection to target neurons
# - Recurrent connections exist between all neurons (E->E, E->I, I->E, I->I)
# - NO inhibitory external input dimensions (all external input is excitatory)
if __name__ == "__main__":
    args = parse_cli_args()
    loaded_config = cfg_manager.load_run_config(args.config, defaults=DEFAULT_RUN_CONFIG)
    apply_run_config_overrides(loaded_config.params, loaded_config.name)
    
    # Define a single random seed for reproducibility
    simulation_seed = 42
    np.random.seed(simulation_seed)
    random.seed(simulation_seed)

    #-----------------------------parameters from config--------------------------------
    num_epochs = NUM_EPOCHS
    N_rec_E = N_REC_E
    N_rec_I = N_REC_I
    N_external_input_dims = N_EXTERNAL_INPUT_DIMS
    stim_list = np.arange(NUM_STIMULI)
    
    # Create folder for this run using config-driven layout
    run_folder_path, _, _ = cfg_manager.prepare_single_run_folders(CURRENT_RUN_CONFIG_NAME)
    run_folder = str(run_folder_path)
    
    print(f"Creating directories for config '{CURRENT_RUN_CONFIG_NAME}': {run_folder}")
    
    # Assign parameters from the top configuration
    stim_duration = STIM_DURATION
    break_time = BREAK_TIME
    start_time = START_TIME
    base_weight_scale = BASE_WEIGHT_SCALE
    weight_scale_method = WEIGHT_SCALE_METHOD
    inhibitory_weight_scale = INHIBITORY_WEIGHT_SCALE
    input_amplitude_scale = INPUT_AMPLITUDE_SCALE
    stimulus_distribution = STIMULUS_DISTRIBUTION
    stimulus_alpha = STIMULUS_ALPHA
    probabilistic_input = PROBABILISTIC_INPUT
    unitary_current = UNITARY_CURRENT
    
    stimulus_scale_factor = input_amplitude_scale
    
    # Update save string to reflect recurrent connection setting
    if DISABLE_RECURRENT_CONNECTIONS:
        save_str = f'DirectInput_FF_NrE{N_rec_E}_NrI{N_rec_I}_NiE{N_external_input_dims}_stim{NUM_STIMULI}'
    else:
        save_str = f'DirectInput_NrE{N_rec_E}_NrI{N_rec_I}_NiE{N_external_input_dims}_stim{NUM_STIMULI}'
    save_str = save_str.replace('.', '__')
    save_str = re.sub(r'[^\w\-_]', '__', save_str)

    # Save run parameters
    fixed_params = {
        'stim_duration': stim_duration,
        'break_time': break_time,
        'start_time': start_time,
        'base_weight_scale': base_weight_scale,
        'weight_scale_method': weight_scale_method,
        'inhibitory_weight_scale': inhibitory_weight_scale,
        'input_amplitude_scale': input_amplitude_scale,
        'stimulus_distribution': stimulus_distribution,
        'stimulus_alpha': stimulus_alpha,
        'stimulus_scale_factor': stimulus_scale_factor,
        'probabilistic_input': probabilistic_input,
        'unitary_current_nA': unitary_current / nA,
        'NUM_STIMULI': NUM_STIMULI,
        'DISABLE_RECURRENT_CONNECTIONS': DISABLE_RECURRENT_CONNECTIONS
    }
    # Create a temporary config dict for the save function
    config = {
        'num_epochs': num_epochs, 'N_rec_E': N_rec_E, 'N_rec_I': N_rec_I,
        'N_external_input_dims': N_external_input_dims, 'stim_list': stim_list
    }
    save_run_parameters(config, CURRENT_RUN_CONFIG_NAME, fixed_params, run_folder)
    
    # Copy the config file to the run folder for record-keeping
    if loaded_config.path and loaded_config.path.exists():
        config_copy_path = os.path.join(run_folder, "config.json")
        shutil.copy2(loaded_config.path, config_copy_path)
        print(f"Config file copied to: {config_copy_path}")
    
    print(f"Run parameters saved to: {run_folder}")

    # Calculate total simulation time based on epochs
    epoch_time = (stim_duration + break_time) * len(stim_list) * ms
    T = start_time * ms + num_epochs * epoch_time

    #-----------------------------weight matrices--------------------------------
    # Calculate scaling factors based on chosen method
    k_input = N_external_input_dims  # Total input synapses per neuron
    k_recurrent = N_rec_E + N_rec_I  # Total recurrent synapses per neuron
    
    if weight_scale_method == 'k':
        input_weight_scale_eff = base_weight_scale / k_input
        r_w_scale_eff = base_weight_scale / k_recurrent
    elif weight_scale_method == 'sqrt_k':
        input_weight_scale_eff = base_weight_scale / np.sqrt(k_input) * input_amplitude_scale  # Apply input amplitude scaling
        r_w_scale_eff = base_weight_scale / np.sqrt(k_recurrent)
    else:
        raise ValueError(f"Unknown weight_scale_method: {weight_scale_method}")
    
    print(f"Weight scaling: {weight_scale_method} method with base_scale={base_weight_scale}")
    print(f"Inhibitory scaling factor: {inhibitory_weight_scale}")
    print(f"Input scale: {input_weight_scale_eff:.6f} (k_input={k_input})")
    print(f"Recurrent scale: {r_w_scale_eff:.6f} (k_recurrent={k_recurrent})")
    
    lognormal_params_rec = [0.0, 0.5, (N_rec_E + N_rec_I, N_rec_E + N_rec_I)]
    weight_rec = r_w_scale_eff * wm.weight_matrix_factory({'recurrent': [N_rec_E, N_rec_I]}, {'lognormal': lognormal_params_rec})
    
    # Check if recurrent connections should be disabled
    if DISABLE_RECURRENT_CONNECTIONS:
        print(f"\n>> DISABLING RECURRENT CONNECTIONS - Running in feedforward mode")
        weight_rec = np.zeros_like(weight_rec)
        print(f"Recurrent weight matrix set to zero - feedforward network only")
        is_feedforward = True
    else:
        # Apply inhibitory weight scaling to I->E and I->I connections
        # In balanced networks, inhibitory neurons typically have higher firing rates than excitatory neurons
        # to provide sufficient inhibition. The inhibitory_weight_scale should be tuned to achieve this balance.
        # Fix: Only scale the inhibitory connections (I->E and I->I), not E->I
        weight_rec[N_rec_E:, :N_rec_E] *= inhibitory_weight_scale  # I->E connections
        weight_rec[N_rec_E:, N_rec_E:] *= inhibitory_weight_scale  # I->I connections
        is_feedforward = False
    
    # Print weight statistics for debugging
    print(f"\nWeight matrix statistics after scaling:")
    print(f"E->E connections: mean={np.mean(weight_rec[:N_rec_E, :N_rec_E]):.4f}, std={np.std(weight_rec[:N_rec_E, :N_rec_E]):.4f}")
    print(f"E->I connections: mean={np.mean(weight_rec[N_rec_E:, :N_rec_E]):.4f}, std={np.std(weight_rec[N_rec_E:, :N_rec_E]):.4f}")
    print(f"I->E connections: mean={np.mean(weight_rec[:N_rec_E, N_rec_E:]):.4f}, std={np.std(weight_rec[:N_rec_E, N_rec_E:]):.4f}")
    print(f"I->I connections: mean={np.mean(weight_rec[N_rec_E:, N_rec_E:]):.4f}, std={np.std(weight_rec[N_rec_E:, N_rec_E:]):.4f}")

    #----------------------------external input generation --------------------------------
    external_input_recurrent = stim.step_stim(N_rec_I+N_rec_E, T, [], [], [], [])
    stims_by_time, stim_info = stim.create_stim_list(T, stim_duration, stim_list, break_time)
    
    # Generate direct input vectors instead of Poisson rates
    # Ensure the first time step in stims_by_time is a valid stimulus
    if stims_by_time[0] == -1:
        # Set to the first valid stimulus in stim_list
        stims_by_time[0] = stim_list[0]
    input_vectors, stimulus_patterns = generate_direct_input_vectors(
        stims_by_time, 
        N_external_input_dims, 
        distribution=stimulus_distribution,
        alpha=stimulus_alpha,
        probabilistic_input=probabilistic_input
    )

    #-----------------------------initializing the network--------------------------------
    lognormal_params_in = [0.0, 0.5, (N_rec_E + N_rec_I, N_external_input_dims)]
    direct_network = din.DirectInputNetwork(N_rec_E, N_rec_I, N_external_input_dims, weight_rec, input_weight_scale_eff, lognormal_params_in, is_feedforward)
    
    # Monitoring is always 'basic' now
    monitoring_level = 'basic'
    
    print(f"\n>> Starting simulation with {NUM_STIMULI} stimuli across {NUM_EPOCHS} epochs")
    print(f"   Total simulation time: {T/ms:.1f}ms")
    print(f"   Stimulus duration: {stim_duration}ms, Break time: {break_time}ms")
    print(f"   Network: {N_rec_E}E + {N_rec_I}I neurons, {N_external_input_dims} input dimensions")
    print(f"   Monitoring level: {monitoring_level}")
    print("="*60)
    
    # Check if we should load existing data or run simulation
    if LOAD_EXISTING_DATA:
        print(">> Loading existing simulation data...")
        data = load_simulation_data(run_folder)
        if data is not None:
            # Extract data from saved dictionary
            individual_rate_matrices = data['individual_rate_matrices'] 
            individual_sparsity = data['individual_sparsity']
            individual_selectivity = data['individual_selectivity']
            rate_matrices_arr = data['rate_matrices_arr']
            sparsity_vec_arr = data['sparsity_vec_arr']
            selectivity_vec_arr = data['selectivity_vec_arr']
            # Create a minimal network-like object with the essential attributes for plotting
            class MinimalNetwork:
                def __init__(self, neuron_order, weight_poisson, weight_rec):
                    self.neuron_order = neuron_order
                    self.weight_poisson = weight_poisson
                    self.weight_rec = weight_rec
            direct_network = MinimalNetwork(data.get('neuron_order'), data.get('weight_poisson'), data.get('weight_rec'))
            print("   Data loaded successfully - skipping simulation")
        else:
            print("   No existing data found - running simulation instead")
            LOAD_EXISTING_DATA = False  # Fall back to running simulation
    
    if not LOAD_EXISTING_DATA:
        print(">> Running simulation...")
        direct_network.run_simulation(T, input_vectors, stims_by_time, stim_info, 0*nA, monitoring_level=monitoring_level, unitary_current=unitary_current, num_epochs=num_epochs)
        try:
            direct_network.analyze_spikes()
        except Exception as e:
            print(f"   Warning: analyze_spikes() failed (memory issue): {e}")
            print("   Continuing without neuron ordering...")

        # Analyze how network responses evolve over time.
        print("Analyzing firing rates...")
        spike_matrix_rec = sau.spike_matrix_from_monitor(direct_network.rec_spikes, N_rec_E + N_rec_I, T, dt=defaultclock.dt)
        direct_network.spike_matrix_rec = spike_matrix_rec # Store for later use
        
        # Get both individual and cumulative epoch analyses
        (individual_rate_matrices, individual_sparsity, individual_selectivity,
         rate_matrices_arr, sparsity_vec_arr, selectivity_vec_arr) = sa.analyze_both_individual_and_cumulative_epochs(
            direct_network.rec_spikes, T, stims_by_time, num_epochs, N_rec_E + N_rec_I, 
            dt=defaultclock.dt, start_time=start_time * ms
        )
        
        # Save simulation data for future plot generation (only essential data, no Brian2 objects)
        simulation_data = {
            'individual_rate_matrices': individual_rate_matrices,
            'individual_sparsity': individual_sparsity, 
            'individual_selectivity': individual_selectivity,
            'rate_matrices_arr': rate_matrices_arr,
            'sparsity_vec_arr': sparsity_vec_arr,
            'selectivity_vec_arr': selectivity_vec_arr,
            'neuron_order': getattr(direct_network, 'neuron_order', None),
            'weight_poisson': getattr(direct_network, 'input_weights', None),
            'weight_rec': getattr(direct_network, 'weight_rec', None),
            'stim_info': stim_info,
            'T': T,
            'config': config,
            'input_vectors': getattr(direct_network, 'input_vectors', None),
            'stimulus_patterns': stimulus_patterns,
            'stim_type_vec': getattr(direct_network, 'stim_type_vec', None)
        }
        save_simulation_data(simulation_data, run_folder)

    # Check if any rate matrices were generated. If not, it means there were no spikes
    # or no stimuli in the analysis windows, so we should exit gracefully.
    if not rate_matrices_arr:
        print("WARNING: No rate matrices were generated. This likely means no spikes occurred during any analysis epoch.")
        # Create empty arrays to avoid crashing subsequent code
        num_neurons = N_rec_E + N_rec_I
        rate_matrices_arr = [np.zeros((num_neurons, len(stim_list)))]
        sparsity_vec_arr = np.zeros((config['num_epochs'], num_neurons))
        selectivity_vec_arr = np.zeros((config['num_epochs'], num_neurons))

    # Note: For long runs, direct_network.rate_matrix_rec is not available.
    # We use the last rate matrix from the analysis instead.
    last_rate_matrix = rate_matrices_arr[-1]
    spi_arr, sli_arr = ss.create_vecs(last_rate_matrix)
    sli_arr = np.array(sli_arr)
    
    # Select random indices for plotting
    random_indices = np.random.choice(selectivity_vec_arr.shape[0], min(10, selectivity_vec_arr.shape[0]), replace=False)

    #------------------------plotting rate matrix, selectivity and sparsity-----------------------------------------
    # Analyze how network responses evolve over time. For 'long' runs, this is done
    # in a memory-efficient way to avoid creating a massive spike matrix.
    # This block is now redundant as the analysis is always performed.

    # Check if any rate matrices were generated.
    if not rate_matrices_arr:
        print("WARNING: No rate matrices were generated. This likely means no spikes occurred during any analysis epoch.")
        # Create empty arrays to avoid crashing subsequent code
        num_neurons = N_rec_E + N_rec_I
        rate_matrices_arr = [np.zeros((num_neurons, len(stim_list)))]
        sparsity_vec_arr = np.zeros((config['num_epochs'], num_neurons))
        selectivity_vec_arr = np.zeros((config['num_epochs'], num_neurons))

    # Note: For long runs, direct_network.rate_matrix_rec is not available.
    # We use the last rate matrix from the analysis instead.
    last_rate_matrix = rate_matrices_arr[-1]
    spi_arr, sli_arr = ss.create_vecs(last_rate_matrix)
    sli_arr = np.array(sli_arr)
    
    # Select random indices for plotting
    random_indices = np.random.choice(selectivity_vec_arr.shape[0], min(10, selectivity_vec_arr.shape[0]), replace=False)

    #------------------------plotting figures-----------------------------------------
    # Prepare data and configuration for plotting
    plot_data = {
        'individual_rate_matrices': individual_rate_matrices,
        'individual_sparsity': individual_sparsity,
        'individual_selectivity': individual_selectivity,
        'rate_matrices_arr': rate_matrices_arr,
        'sparsity_vec_arr': sparsity_vec_arr,
        'selectivity_vec_arr': selectivity_vec_arr,
        'neuron_order': getattr(direct_network, 'neuron_order', None),
        'weight_poisson': getattr(direct_network, 'input_weights', None),
        'weight_rec': getattr(direct_network, 'weight_rec', None),
        'stim_info': stim_info,
        'T': T,
        'config': config
    }
    
    plot_params = {
        'MAX_STIMULI_TO_PLOT': MAX_STIMULI_TO_PLOT,
        'MAX_NEURONS_VOLTAGE_TRACES': MAX_NEURONS_VOLTAGE_TRACES,
        'MAX_NEURONS_RASTER': MAX_NEURONS_RASTER
    }
    
    # Use new modular plot generation system
    # This will create separate folders for each analysis type
    plots_config = {
        'network_structure': True,             # Weight matrices → network_structure/
        'connectivity_tuning': True,           # CCI analysis → connectivity_tuning/
        'temporal_evolution': True,            # Sparsity/selectivity evolution → temporal_evolution/
        'voltage_current_monitors': True,      # Voltage/current traces → voltage_current_monitors/
        'correlation_analysis': True,          # All correlation analyses → correlation_analysis/
    }
    
    # Generate all plots using modular system
    all_stats = generate_all_plots(
        run_folder=run_folder,
        network=direct_network if not LOAD_EXISTING_DATA else None,
        data=plot_data,
        config=config,
        plots_config=plots_config,
        plot_params=plot_params
    )

    # --- Final Summary ---
    try:
        final_sparsity = sparsity_vec_arr[-1, :]
        final_selectivity = sli_arr
        
        # Calculate and print average firing rates over the whole simulation
        total_time_s = T / second
        e_spikes = direct_network.rec_spikes.count[:N_rec_E].sum()
        i_spikes = direct_network.rec_spikes.count[N_rec_E:].sum()
        
        e_rate = e_spikes / (N_rec_E * total_time_s)
        i_rate = i_spikes / (N_rec_I * total_time_s)

        print(f"\n>> Average Firing Rates:")
        print(f"   Excitatory: {e_rate:.2f} Hz")
        print(f"   Inhibitory: {i_rate:.2f} Hz")

        print(f"\n>> Simulation completed successfully!")
        print(f"   Results saved to: {run_folder}")
        print(f"   Final network sparsity: {np.mean(final_sparsity):.3f} ± {np.std(final_sparsity):.3f}")
        print(f"   Final network selectivity: {np.mean(final_selectivity):.3f} ± {np.std(final_selectivity):.3f}")
    except Exception as e:
        print(f"   Error in final summary: {e}")
        print(f"   Results saved to: {run_folder}")