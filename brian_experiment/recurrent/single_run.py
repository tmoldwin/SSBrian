import os
import re
import random
from datetime import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
from brian2 import *

from brian_experiment.recurrent import network as brn
from brian_experiment.recurrent import utils as rd_utils
from brian_experiment.shared.core import spike_analysis_utils as sau
from brian_experiment.shared.core import weight_matrices as wm
from brian_experiment.shared.core.constants import RANDOM_SEED, SAVE_FOLDER

# ======================================================================================
#                                 USER CONFIGURATION
# ======================================================================================
# --- Network Architecture ---
N_REC_E = 200
N_REC_I = 200  # Equal to E for clearer interpretation

# --- Simulation Timing (in ms) ---
SIMULATION_DURATION = 5000  # 5 seconds of spontaneous activity

# --- Weight & Balance Parameters ---
# For chaotic balanced networks, we need stronger recurrent connections
BASE_WEIGHT_SCALE = 3.0  # Higher than feedforward networks
WEIGHT_SCALE_METHOD = 'sqrt_k'  # 'k' or 'sqrt_k'

# Critical parameter: E-I balance
# For chaotic dynamics, inhibition should roughly balance excitation
# The ratio I_strength/E_strength should be around 4-5 (Dale's law requires this)
INHIBITORY_WEIGHT_SCALE = 4.0

# --- Initial State Configuration ---
# Options:
#   {'type': 'random_voltage', 'v_min': -70*mV, 'v_max': -50*mV}
#   {'type': 'random_current', 'I_mean': 1*nA, 'I_std': 0.5*nA, 'duration': 50*ms}
#   {'type': 'subset_spikes', 'fraction': 0.1}
INITIAL_STATE = {
    'type': 'random_voltage',
    'v_min': -70*mV,
    'v_max': -50*mV
}

# --- Monitoring Level ---
MONITORING_LEVEL = 'basic'  # 'minimal', 'basic', or 'full'

# ======================================================================================

# Configure Brian backend (CPU/Numpy)
rd_utils.setup_numpy_device()

def save_run_parameters(folder_path, params):
    """Save run parameters to a text file"""
    param_file = os.path.join(folder_path, "run_parameters.txt")
    
    with open(param_file, 'w') as f:
        f.write(f"Balanced Recurrent Network Simulation\n")
        f.write(f"{'='*40}\n")
        f.write(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Network Configuration:\n")
        f.write(f"{'='*22}\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

def save_simulation_data(data_dict, folder_path):
    """Save simulation data to pickle file"""
    data_file = os.path.join(folder_path, "simulation_data.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Simulation data saved to: {data_file}")

def calculate_statistics(network, weight_rec):
    """Calculate and print network statistics using shared utilities."""
    metrics = rd_utils.analyze_network_activity(
        network, weight_rec, network.N_rec_E, network.N_rec_I
    )

    print("\n" + "=" * 60)
    print("NETWORK STATISTICS")
    print("=" * 60)

    e_rate = metrics["e_rate"]
    i_rate = metrics["i_rate"]
    print("\nFiring Rates:")
    print(f"  Excitatory: {e_rate:.2f} Hz")
    print(f"  Inhibitory: {i_rate:.2f} Hz")
    ratio = i_rate / e_rate if e_rate > 0 else np.inf
    print(f"  Ratio (I/E): {ratio:.2f}")

    print("\nCoefficient of Variation (ISI):")
    print(f"  Excitatory: {metrics['cv_e']:.3f}")
    print(f"  Inhibitory: {metrics['cv_i']:.3f}")

    print("\nCorrelations:")
    print(f"  E-I Rate Corr: {metrics['rate_correlation']:.3f}")
    print(f"  Mean Pairwise E: {metrics['mean_pairwise_corr_e']:.3f}")
    print(f"  Mean Pairwise I: {metrics['mean_pairwise_corr_i']:.3f}")

    print("\nSynaptic balance:")
    print(f"  Mean E weight: {metrics['mean_e_weight']:.4f}")
    print(f"  Mean I weight: {metrics['mean_i_weight']:.4f}")
    print(f"  Current balance (|I|/E): {metrics['balance_ratio']:.3f}")
    print("=" * 60)

def plot_weight_matrices(weight_rec, N_rec_E, save_folder):
    """Plot weight matrix structure"""
    print("\nGenerating weight matrix plots...")
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Full weight matrix
    im0 = axs[0, 0].imshow(weight_rec, cmap='RdBu_r', aspect='auto')
    axs[0, 0].set_title('Full Recurrent Weight Matrix')
    axs[0, 0].set_xlabel('Target Neuron')
    axs[0, 0].set_ylabel('Source Neuron')
    axs[0, 0].axhline(N_rec_E, color='white', linestyle='--', linewidth=2)
    axs[0, 0].axvline(N_rec_E, color='white', linestyle='--', linewidth=2)
    plt.colorbar(im0, ax=axs[0, 0])
    
    # E->E connections
    im1 = axs[0, 1].imshow(weight_rec[:N_rec_E, :N_rec_E], cmap='Reds', aspect='auto')
    axs[0, 1].set_title('E→E Connections')
    axs[0, 1].set_xlabel('Target E Neuron')
    axs[0, 1].set_ylabel('Source E Neuron')
    plt.colorbar(im1, ax=axs[0, 1])
    
    # I->E connections
    im2 = axs[1, 0].imshow(weight_rec[:N_rec_E, N_rec_E:], cmap='Blues_r', aspect='auto')
    axs[1, 0].set_title('I→E Connections')
    axs[1, 0].set_xlabel('Source I Neuron')
    axs[1, 0].set_ylabel('Target E Neuron')
    plt.colorbar(im2, ax=axs[1, 0])
    
    # Weight distribution histogram
    axs[1, 1].hist(weight_rec[:N_rec_E, :N_rec_E].flatten(), bins=50, alpha=0.7, 
                   color='red', label='E→all', density=True)
    axs[1, 1].hist(weight_rec[N_rec_E:, :].flatten(), bins=50, alpha=0.7, 
                   color='blue', label='I→all', density=True)
    axs[1, 1].set_xlabel('Weight Value')
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].set_title('Weight Distribution')
    axs[1, 1].legend()
    axs[1, 1].axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, "weight_matrices.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Weight matrices saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    simulation_seed = 42
    np.random.seed(simulation_seed)
    random.seed(simulation_seed)
    
    # Create folder for this run under "no_feedforward"
    base_save_folder = os.path.join(os.path.dirname(__file__), "Runs")
    run_folder_name = f"single_run_E{N_REC_E}_I{N_REC_I}_T{SIMULATION_DURATION}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(base_save_folder, run_folder_name, timestamp)
    figures_folder = os.path.join(run_folder, "Figures")
    
    print(f"Creating directory: {run_folder}")
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(figures_folder, exist_ok=True)
    
    # Save parameters
    params = {
        'N_rec_E': N_REC_E,
        'N_rec_I': N_REC_I,
        'simulation_duration_ms': SIMULATION_DURATION,
        'base_weight_scale': BASE_WEIGHT_SCALE,
        'weight_scale_method': WEIGHT_SCALE_METHOD,
        'inhibitory_weight_scale': INHIBITORY_WEIGHT_SCALE,
        'initial_state': INITIAL_STATE,
        'monitoring_level': MONITORING_LEVEL,
        'random_seed': simulation_seed
    }
    save_run_parameters(run_folder, params)
    
    # Calculate weight matrix
    N_rec = N_REC_E + N_REC_I
    k_recurrent = N_rec
    
    if WEIGHT_SCALE_METHOD == 'k':
        r_w_scale_eff = BASE_WEIGHT_SCALE / k_recurrent
    elif WEIGHT_SCALE_METHOD == 'sqrt_k':
        r_w_scale_eff = BASE_WEIGHT_SCALE / np.sqrt(k_recurrent)
    else:
        raise ValueError(f"Unknown weight_scale_method: {WEIGHT_SCALE_METHOD}")
    
    print(f"\n{'='*60}")
    print("BALANCED RECURRENT NETWORK SIMULATION")
    print(f"{'='*60}")
    print(f"Network: {N_REC_E}E + {N_REC_I}I neurons")
    print(f"Simulation duration: {SIMULATION_DURATION} ms")
    print(f"Weight scaling: {WEIGHT_SCALE_METHOD} method with base_scale={BASE_WEIGHT_SCALE}")
    print(f"  Effective recurrent scale: {r_w_scale_eff:.6f}")
    print(f"  Inhibitory weight scale: {INHIBITORY_WEIGHT_SCALE}x")
    print(f"Initial state: {INITIAL_STATE['type']}")
    print(f"{'='*60}\n")
    
    # Generate recurrent weight matrix
    lognormal_params_rec = [0.0, 0.5, (N_rec, N_rec)]
    weight_rec = r_w_scale_eff * wm.weight_matrix_factory(
        {'recurrent': [N_REC_E, N_REC_I]}, 
        {'lognormal': lognormal_params_rec}
    )
    
    # Apply inhibitory weight scaling for balanced dynamics
    # Scale I->E and I->I connections to balance excitation
    weight_rec[N_REC_E:, :N_REC_E] *= INHIBITORY_WEIGHT_SCALE  # I->E
    weight_rec[N_REC_E:, N_REC_E:] *= INHIBITORY_WEIGHT_SCALE  # I->I
    
    # Print weight statistics
    print(f"Weight matrix statistics:")
    print(f"  E->E: mean={np.mean(np.abs(weight_rec[:N_REC_E, :N_REC_E])):.4f}, "
          f"std={np.std(weight_rec[:N_REC_E, :N_REC_E]):.4f}")
    print(f"  E->I: mean={np.mean(np.abs(weight_rec[:N_REC_E, N_REC_E:])):.4f}, "
          f"std={np.std(weight_rec[:N_REC_E, N_REC_E:]):.4f}")
    print(f"  I->E: mean={np.mean(np.abs(weight_rec[N_REC_E:, :N_REC_E])):.4f}, "
          f"std={np.std(weight_rec[N_REC_E:, :N_REC_E]):.4f}")
    print(f"  I->I: mean={np.mean(np.abs(weight_rec[N_REC_E:, N_REC_E:])):.4f}, "
          f"std={np.std(weight_rec[N_REC_E:, N_REC_E:]):.4f}")
    
    # Calculate theoretical E-I balance ratio
    mean_E_strength = np.mean(np.abs(weight_rec[:N_REC_E, :]))
    mean_I_strength = np.mean(np.abs(weight_rec[N_REC_E:, :]))
    print(f"  Theoretical E-I balance ratio (|I|/E): {mean_I_strength/mean_E_strength:.2f}")
    print()
    
    # Plot weight matrices
    plot_weight_matrices(weight_rec, N_REC_E, figures_folder)
    
    # Initialize network
    T = SIMULATION_DURATION * ms
    network = brn.BalancedRecurrentNetwork(
        N_REC_E, 
        N_REC_I, 
        weight_rec, 
        initial_state_config=INITIAL_STATE
    )
    
    # Run simulation
    network.run_simulation(T, monitoring_level=MONITORING_LEVEL)
    
    # Analyze spikes
    print("\nAnalyzing spikes...")
    network.analyze_spikes()
    
    # Calculate and display statistics
    calculate_statistics(network, weight_rec)
    
    # Generate plots
    print("\nGenerating activity plots...")
    
    # Plot full simulation
    fig1 = network.plot_spontaneous_activity()
    save_path1 = os.path.join(figures_folder, "spontaneous_activity_full.png")
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"Full activity plot saved to: {save_path1}")
    plt.close()
    
    # Plot first 1 second (detailed view)
    fig2 = network.plot_spontaneous_activity(time_window=(0, 1000))
    save_path2 = os.path.join(figures_folder, "spontaneous_activity_1s.png")
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Detailed (1s) activity plot saved to: {save_path2}")
    plt.close()
    
    # Save simulation data
    simulation_data = {
        'spike_matrix': network.spike_matrix_rec,
        'smoothed_rates': network.smoothed_rates_rec,
        'smoothed_time_points': network.smoothed_time_points,
        'neuron_order': network.neuron_order,
        'weight_rec': network.weight_rec,
        'T': T,
        'N_rec_E': N_REC_E,
        'N_rec_I': N_REC_I,
        'params': params
    }
    save_simulation_data(simulation_data, run_folder)
    
    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {run_folder}")
    print(f"{'='*60}\n")

