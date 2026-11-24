import os
import re
import haim_network as hn
import plots as pf
import matplotlib.pyplot as plt
import stimuli as stim
import weight_matrices as wm
from brian2 import *
import numpy as np
import spike_analysis as sa
import random
from constants import RANDOM_SEED, SAVE_FOLDER
import spike_analysis_utils as sau
import matplotlib.ticker as ticker
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import rate_analysis as ra
import sparsity_and_selectivity as ss
import spike_analysis as sa
import figures as figs
from datetime import datetime
import platform

# Check platform and GPU availability before setting device
def setup_device():
    """Use normal CPU processing"""
    print("Using normal CPU processing")
    # Force pure Python backend to avoid Cython compatibility issues
    prefs.codegen.target = 'numpy'
    print("Using numpy backend to avoid compatibility issues")

# Setup device
setup_device()

# Set global random seeds

# np.random.seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)

# Create timestamped folder for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder = os.path.join(SAVE_FOLDER, f"run_{timestamp}")
os.makedirs(run_folder, exist_ok=True)

# Update SAVE_FOLDER to point to timestamped folder
SAVE_FOLDER = run_folder

def save_run_parameters(config, run_type, fixed_params, folder_path):
    """Save run parameters to a text file in the timestamped folder"""
    param_file = os.path.join(folder_path, "run_parameters.txt")
    
    with open(param_file, 'w') as f:
        f.write(f"RNN Simulation Run Parameters\n")
        f.write(f"{'='*40}\n")
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
        f.write(f"Total poisson inputs: {config['N_poisson_E'] + config['N_poisson_I']}\n")
        f.write(f"Stimuli count: {len(config['stim_list'])}\n")
        f.write(f"Epoch duration: {(fixed_params['stim_duration'] + fixed_params['break_time']) * len(config['stim_list'])} ms\n")
        f.write(f"Total simulation time: {fixed_params['start_time'] + config['num_epochs'] * (fixed_params['stim_duration'] + fixed_params['break_time']) * len(config['stim_list'])} ms\n")

if __name__ == "__main__":
#-----------------------------parameters--------------------------------
    # Parameter configurations for different run lengths
    run_configs = {
        'short': {
            'num_epochs': 5,
            'N_rec_E': 10,
            'N_rec_I': 10,
            'N_poisson_E': 15,
            'N_poisson_I': 10,
            'stim_list': np.arange(4)
        },
        'medium': {
            'num_epochs': 10,
            'N_rec_E': 50,
            'N_rec_I': 50,
            'N_poisson_E': 75,
            'N_poisson_I': 50,
            'stim_list': np.arange(10)
        },
        'test': {
            'num_epochs': 5,
            'N_rec_E': 100,
            'N_rec_I': 100,
            'N_poisson_E': 100,
            'N_poisson_I': 100,
            'stim_list': np.arange(3)
        },
        'long': {
            'num_epochs': 5,
            'N_rec_E': 100,
            'N_rec_I': 100,
            'N_poisson_E': 100,
            'N_poisson_I': 100,
            'stim_list': np.arange(50)
        }
    }
    
    # Choose which configuration to use
    run_type = 'test'  # Options: 'short', 'medium', 'test', 'long'
    config = run_configs[run_type]
    
    # Extract parameters from config
    num_epochs = config['num_epochs']
    N_rec_E = config['N_rec_E']
    N_rec_I = config['N_rec_I']
    N_poisson_E = config['N_poisson_E']
    N_poisson_I = config['N_poisson_I']
    stim_list = config['stim_list']
    
    # Fixed parameters (same for all configurations)
    stim_duration = 100  # ms
    break_time = 20     # ms
    start_time = 100    # ms initial delay before first stimulus
    p_w_scale = 0.8      # Poisson weight scale
    poisson_amplitude_scale = 10.0  # Poisson input amplitude - increased for higher firing rates
    r_w_scale = 0.001   # Recurrent weight scale
    
    # Stimulus generation parameters
    stimulus_distribution = 'dirichlet'
    stimulus_alpha = 0.5  # Lower alpha creates more sparse patterns
    stimulus_scale_factor = poisson_amplitude_scale * 10  # Scale to appropriate firing rate range
    
    save_str = f'NrE{N_rec_E}_NrI{N_rec_I}_NpE{N_poisson_E}_NpI{N_poisson_I}_' \
               f'st_dur{stim_duration}_bt{break_time}_sl{stim_list}_' \
               f'pws{p_w_scale:.2f}_pas{poisson_amplitude_scale:.2f}_rws{r_w_scale:.2f}'
    save_str = save_str.replace('.', '__')
    save_str = re.sub(r'[^\w\-_]', '__', save_str)  # Replace any non-word characters with double underscores
  # Replace periods with double underscores

    # Save run parameters
    fixed_params = {
        'stim_duration': stim_duration,
        'break_time': break_time,
        'start_time': start_time,
        'p_w_scale': p_w_scale,
        'poisson_amplitude_scale': poisson_amplitude_scale,
        'r_w_scale': r_w_scale,
        'stimulus_distribution': stimulus_distribution,
        'stimulus_alpha': stimulus_alpha,
        'stimulus_scale_factor': stimulus_scale_factor
    }
    save_run_parameters(config, run_type, fixed_params, run_folder)
    
    print(f"Run parameters saved to: {run_folder}")

    # Calculate total simulation time based on epochs
    epoch_time = (stim_duration + break_time) * len(stim_list) * ms  # Time for one complete epoch
    T = start_time * ms + num_epochs * epoch_time
    time_intervals = T / num_epochs
#-----------------------------weight matrices--------------------------------
    weight_rec = r_w_scale*wm.weight_matrix_factory({'recurrent': [N_rec_E, N_rec_I]}, {'exponential': 1})
    weight_poisson = p_w_scale *wm.weight_matrix_factory({'poisson': [N_rec_E, N_rec_I, N_poisson_E, N_poisson_I]},
                                                                    {'exponential': 1})

#----------------------------external input for rec & poisson --------------------------------
    external_input_recurrent = stim.step_stim(N_rec_I+N_rec_E, T, [], [], [], [])
    stims_by_time, stim_info = stim.create_stim_list(T, stim_duration, stim_list, break_time)
    # Change from Weibull to Dirichlet distribution with alpha parameter
    stim_trials = stim.generate_stimuli_from_distribution(
        stims_by_time, 
        N_poisson_E + N_poisson_I, 
        distribution=stimulus_distribution,
        alpha=stimulus_alpha,  # Lower alpha creates more sparse patterns
        scale_factor=stimulus_scale_factor  # Scale to appropriate firing rate range
    )
    external_input_poisson = stim_trials

    # Extract individual stimulus patterns first
    unique_stims = np.unique(stims_by_time)
    unique_stims = unique_stims[unique_stims != -1]  # Remove -1 (no stimulus)
    unique_stims = np.sort(unique_stims)  # Sort to ensure consistent order
    
    # Create a matrix with just the unique patterns
    stim_patterns = {}
    stim_pattern_matrix = np.zeros((external_input_poisson.shape[0], len(unique_stims)))
    
    for i, stim in enumerate(unique_stims):
        t = np.where(stims_by_time == stim)[0][0]  # Get first occurrence
        pattern = external_input_poisson[:, t]
        stim_patterns[stim] = pattern
        stim_pattern_matrix[:, i] = pattern
    
    # Use the proper plotting function from the plots module
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    pf.plot_stimuli_imshow(stim_pattern_matrix, None, ax, title="Dirichlet Stimulus Patterns")
    plt.savefig(os.path.join(SAVE_FOLDER, 'dirichlet_stimuli_patterns.png'), bbox_inches='tight')
    plt.close()

#-----------------------------initializing the network--------------------------------
    haim_network = hn.HaimNetwork(N_rec_E, N_rec_I, N_poisson_E, N_poisson_I, weight_rec, weight_poisson)
    
    # Choose monitoring level based on run type for optimal performance
    if run_type == 'long':
        monitoring_level = 'minimal'  # Maximum speed for long runs
        print("Using minimal monitoring for maximum speed on long run")
    elif run_type in ['medium', 'test']:
        monitoring_level = 'basic'   # Good balance for medium runs
        print("Using basic monitoring for medium run")
    else:
        monitoring_level = 'full'    # Full data for short runs
        print("Using full monitoring for short run")
    
    haim_network.run_simulation(T, external_input_poisson, external_input_recurrent, stims_by_time, stim_info, 0, monitoring_level=monitoring_level)
    haim_network.analyze_spikes()

    #------------------------plotting rate matrix, selectivity and sparsity-----------------------------------------
    # Calculate number of intervals based on epochs
    num_intervals = num_epochs  # One analysis point per epoch
    spi_arr, sli_arr = ss.create_vecs(haim_network.rate_matrix_rec)
    sli_arr = np.array(sli_arr)
    
    # Analyze how network responses evolve over time
    rate_matrices_arr, sparsity_vec_arr, selectivity_vec_arr = sa.analyze_firing_rates_over_time(
        T, stims_by_time, haim_network.spike_matrix_rec, num_intervals, 
        dt=defaultclock.dt, start_time=start_time
    )
    
    # Select random indices for plotting
    selectivity_vec_arr = selectivity_vec_arr.T
    random_indices = np.random.choice(selectivity_vec_arr.shape[0], min(10, selectivity_vec_arr.shape[0]), replace=False)
    
    # # Plot vectors with proper epoch labeling
    # pf.plot_two_vector_arrays_in_panels_with_colors(
    #     sparsity_vec_arr.T,
    #     selectivity_vec_arr[random_indices, :],
    #     x_label="Number of Training Epochs",
    #     title="Network Sparsity and Selectivity Evolution"
    # )

    # #------------------------plotting selectivity -----------------------------------------
    # # selectivity indices before
    # ra.selectivity_stem_plot(rate_matrices_arr[2], selectivity_vec_arr[:, 2])
    # # selectivity indices after
    # ra.selectivity_stem_plot(haim_network.rate_matrix_rec, sli_arr)

    print("last epoch sli arr shape:", sli_arr.shape)
    print("first epoch sli arr shape:", selectivity_vec_arr[0].shape)

    #------------------------plotting figures-----------------------------------------
    figs.plotting_dashboard_figure_1(weight_poisson, weight_rec, save_folder=SAVE_FOLDER)
    
    # Skip the heavy dynamics and spiking plot for long runs
    if run_type != 'long':
        figs.plotting_dashboard_figure_2(haim_network, save_folder=SAVE_FOLDER)
    else:
        print("Skipping dynamics and spiking plot for long run (too memory intensive)")
    
    figs.plotting_dashboard_figure_3(haim_network, sparsity_vec_arr, selectivity_vec_arr, save_folder=SAVE_FOLDER)
    figs.plotting_dashboard_figure_4(rate_matrices_arr[2], selectivity_vec_arr[: ,2], haim_network.rate_matrix_rec, sli_arr, save_folder=SAVE_FOLDER)








