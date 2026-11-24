#!/usr/bin/env python3
"""
Test script for the new aggregate input correlation analysis.
This demonstrates how the new correlation analysis works compared to the old Hz-based approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the brian_experiment directory to the path
sys.path.append('brian_experiment')

import plots as pf
import haim_network as hn
import weight_matrices as wm
import stimuli as stim
from brian2 import *
from constants import SAVE_FOLDER

def test_correlation_analysis():
    """Test the new correlation analysis with a simple network simulation."""
    
    print("Setting up test network...")
    
    # Network parameters
    N_rec_E = 20
    N_rec_I = 20
    N_poisson_E = 30
    N_poisson_I = 20
    
    # Simulation parameters
    T = 2000 * ms
    stim_duration = 100  # ms
    break_time = 20     # ms
    start_time = 100    # ms
    num_epochs = 3
    
    # Weight scales
    p_w_scale = 0.8
    r_w_scale = 0.001
    poisson_amplitude_scale = 10.0
    
    # Create weight matrices
    weight_rec = r_w_scale * wm.weight_matrix_factory({'recurrent': [N_rec_E, N_rec_I]}, {'exponential': 1})
    weight_poisson = p_w_scale * wm.weight_matrix_factory({'poisson': [N_rec_E, N_rec_I, N_poisson_E, N_poisson_I]}, {'exponential': 1})
    
    # Create stimuli
    stim_list = np.arange(4)
    epoch_time = (stim_duration + break_time) * len(stim_list) * ms
    T = start_time * ms + num_epochs * epoch_time
    
    external_input_recurrent = stim.step_stim(N_rec_I + N_rec_E, T, [], [], [], [])
    stims_by_time, stim_info = stim.create_stim_list(T, stim_duration, stim_list, break_time)
    
    stim_trials = stim.generate_stimuli_from_distribution(
        stims_by_time, 
        N_poisson_E + N_poisson_I, 
        distribution='dirichlet',
        alpha=0.5,
        scale_factor=poisson_amplitude_scale * 10
    )
    external_input_poisson = stim_trials
    
    # Create and run network
    print("Creating and running network...")
    haim_network = hn.HaimNetwork(N_rec_E, N_rec_I, N_poisson_E, N_poisson_I, weight_rec, weight_poisson)
    haim_network.run_simulation(T, external_input_poisson, external_input_recurrent, stims_by_time, stim_info, 0)
    haim_network.analyze_spikes()
    
    # Create comparison plots
    print("Creating comparison plots...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # New correlation analysis (aggregate input)
    pf.plot_aggregate_input_correlation(ax, haim_network.spike_matrix_poiss, 
                                      haim_network.spike_matrix_rec, 
                                      haim_network.stim_type_vec, 
                                      haim_network.N_poisson_E, haim_network.N_poisson_I, 
                                      haim_network.N_rec_E, haim_network.N_rec_I, 
                                      window_size=50)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_FOLDER, 'aggregate_correlation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also test the convenience method
    print("Testing convenience method...")
    fig2 = haim_network.plot_aggregate_correlation(window_size=50)
    plt.savefig(os.path.join(SAVE_FOLDER, 'aggregate_correlation_standalone.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Test completed! Check the saved figures in the SAVE_FOLDER.")

if __name__ == "__main__":
    test_correlation_analysis() 