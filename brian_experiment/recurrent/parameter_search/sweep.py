"""
Parameter sweep for balanced recurrent network to demonstrate critical E-I balance.

This script sweeps the E:I NEURON RATIO to show how different ratios of excitatory
to inhibitory neurons affect network dynamics and outgoing synaptic weights.
"""

import os
import re
import random
from datetime import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
from scipy.stats import gaussian_kde

from brian_experiment.recurrent import network as brn
from brian_experiment.recurrent import utils as rd_utils
from brian_experiment.shared.core import weight_matrices as wm

# ======================================================================================
#                                 SWEEP CONFIGURATION
# ======================================================================================
# Network Architecture (fixed 50/50 ratio)
N_REC_E = 200
N_REC_I = 200

# Simulation Timing (shorter for sweep)
SIMULATION_DURATION = 1000  # 1 second per run

# Weight Scaling
BASE_WEIGHT_SCALE = 15.0  # Base scale for recurrent weights
WEIGHT_SCALE_METHOD = 'sqrt_k'

# CRITICAL PARAMETER TO SWEEP: Individual connection type weight scales
# Base scale for all connections (kept constant for non-varied types)
BASE_CONNECTION_SCALE = 1.0

# Scales to sweep for the varied connection type
# Log scale from 10^-3 to 10^1, evenly spaced (4 values)
# log10 range: -3 to 1, step = 4/3 ≈ 1.33
log_min = -3
log_max = 1
n_values = 4
log_values = np.linspace(log_min, log_max, n_values)
VARIED_SCALES = [10**log_val for log_val in log_values]  # [0.001, 0.0215, 0.464, 10.0]

# Use same scales for all connection types (log scale sweep)
VARIED_SCALES_BY_TYPE = {
    'E_to_E': VARIED_SCALES,
    'E_to_I': VARIED_SCALES,
    'I_to_E': VARIED_SCALES,
    'I_to_I': VARIED_SCALES,
}

# Four sweep types: vary one connection type, keep others constant
SWEEP_TYPES = ['E_to_E', 'E_to_I', 'I_to_E', 'I_to_I']

# Background drive (constant external current to sustain activity)
BACKGROUND_CURRENT = 2.0 * nA  # Small constant drive

# Initial State (same for all)
INITIAL_STATE = {
    'type': 'random_voltage',
    'v_min': -70*mV,
    'v_max': -50*mV
}

# Monitoring
MONITORING_LEVEL = 'basic'

# ======================================================================================

rd_utils.setup_numpy_device()

def run_single_configuration(sweep_type, varied_scale, run_idx, base_folder):
    """Run a single configuration and return results
    
    Args:
        sweep_type: One of 'E_to_E', 'E_to_I', 'I_to_E', 'I_to_I'
        varied_scale: Scale factor for the varied connection type
        run_idx: Index of this run
        base_folder: Base folder for saving results
    """
    print(f"\n{'='*60}")
    # Get total number of runs for this sweep type
    total_runs = len(VARIED_SCALES_BY_TYPE[sweep_type])
    print(f"RUN {run_idx+1}/{total_runs}: {sweep_type} scale = {varied_scale}")
    print(f"{'='*60}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate weight matrix for this configuration
    N_rec = N_REC_E + N_REC_I
    k_recurrent = N_rec
    
    if WEIGHT_SCALE_METHOD == 'k':
        r_w_scale_eff = BASE_WEIGHT_SCALE / k_recurrent
    elif WEIGHT_SCALE_METHOD == 'sqrt_k':
        r_w_scale_eff = BASE_WEIGHT_SCALE / np.sqrt(k_recurrent)
    else:
        raise ValueError(f"Unknown weight_scale_method: {WEIGHT_SCALE_METHOD}")
    
    # Generate lognormal weights: shape = (target, source) = (N_rec, N_rec)
    lognormal_weights = np.random.lognormal(mean=0.0, sigma=0.5, size=(N_rec, N_rec))
    weight_rec = r_w_scale_eff * lognormal_weights
    
    # Apply Dale's law: E neurons have positive weights, I neurons have negative weights
    # Sources: columns 0 to N_REC_E-1 are E, columns N_REC_E to N_rec-1 are I
    weight_rec[:, :N_REC_E] = np.abs(weight_rec[:, :N_REC_E])  # E sources: positive
    weight_rec[:, N_REC_E:] = -np.abs(weight_rec[:, N_REC_E:])  # I sources: negative
    
    # Apply weight scaling: vary only the specified connection type, keep others at BASE_CONNECTION_SCALE
    # E→E: rows 0 to N_REC_E-1, cols 0 to N_REC_E-1
    # E→I: rows N_REC_E to end, cols 0 to N_REC_E-1
    # I→E: rows 0 to N_REC_E-1, cols N_REC_E to end
    # I→I: rows N_REC_E to end, cols N_REC_E to end
    
    if sweep_type == 'E_to_E':
        weight_rec[:N_REC_E, :N_REC_E] *= varied_scale  # Vary E→E
        weight_rec[N_REC_E:, :N_REC_E] *= BASE_CONNECTION_SCALE  # Keep E→I constant
        weight_rec[:N_REC_E, N_REC_E:] *= BASE_CONNECTION_SCALE  # Keep I→E constant
        weight_rec[N_REC_E:, N_REC_E:] *= BASE_CONNECTION_SCALE  # Keep I→I constant
    elif sweep_type == 'E_to_I':
        weight_rec[:N_REC_E, :N_REC_E] *= BASE_CONNECTION_SCALE  # Keep E→E constant
        weight_rec[N_REC_E:, :N_REC_E] *= varied_scale  # Vary E→I
        weight_rec[:N_REC_E, N_REC_E:] *= BASE_CONNECTION_SCALE  # Keep I→E constant
        weight_rec[N_REC_E:, N_REC_E:] *= BASE_CONNECTION_SCALE  # Keep I→I constant
    elif sweep_type == 'I_to_E':
        weight_rec[:N_REC_E, :N_REC_E] *= BASE_CONNECTION_SCALE  # Keep E→E constant
        weight_rec[N_REC_E:, :N_REC_E] *= BASE_CONNECTION_SCALE  # Keep E→I constant
        weight_rec[:N_REC_E, N_REC_E:] *= varied_scale  # Vary I→E
        weight_rec[N_REC_E:, N_REC_E:] *= BASE_CONNECTION_SCALE  # Keep I→I constant
    elif sweep_type == 'I_to_I':
        weight_rec[:N_REC_E, :N_REC_E] *= BASE_CONNECTION_SCALE  # Keep E→E constant
        weight_rec[N_REC_E:, :N_REC_E] *= BASE_CONNECTION_SCALE  # Keep E→I constant
        weight_rec[:N_REC_E, N_REC_E:] *= BASE_CONNECTION_SCALE  # Keep I→E constant
        weight_rec[N_REC_E:, N_REC_E:] *= varied_scale  # Vary I→I
    else:
        raise ValueError(f"Unknown sweep_type: {sweep_type}")
    
    # Initialize network
    T = SIMULATION_DURATION * ms
    network = brn.BalancedRecurrentNetwork(
        N_REC_E, 
        N_REC_I, 
        weight_rec, 
        initial_state_config=INITIAL_STATE,
        background_current=BACKGROUND_CURRENT
    )
    
    # Run simulation
    try:
        network.run_simulation(T, monitoring_level=MONITORING_LEVEL)
        network.analyze_spikes()
        metrics = rd_utils.analyze_network_activity(network, weight_rec, N_REC_E, N_REC_I)
        print(f"  E rate: {metrics['e_rate']:.2f} Hz, I rate: {metrics['i_rate']:.2f} Hz")
        print(f"  CV_E: {metrics['cv_e']:.3f}, CV_I: {metrics['cv_i']:.3f}")
        print(f"  Current balance ratio: {metrics['balance_ratio']:.3f}")
        print(f"  Mean E outgoing weight: {metrics['mean_e_weight']:.4f}")
        print(f"  Mean I outgoing weight: {metrics['mean_i_weight']:.4f}")

        results = {
            'sweep_type': sweep_type,
            'varied_scale': varied_scale,
            'varied_connection': sweep_type,
            **metrics,
            'network': network,
            'weight_rec': weight_rec,
            'spike_times': metrics.get('spike_times_all'),
            'spike_neurons': metrics.get('spike_neurons_all'),
            'N_rec_E': network.N_rec_E,
            'N_rec_I': network.N_rec_I,
            'success': True
        }
        
    except Exception as e:
        print(f"  ERROR: Simulation failed: {e}")
        results = {
            'sweep_type': sweep_type,
            'varied_scale': varied_scale,
            'varied_connection': sweep_type,
            'e_rate': 0,
            'i_rate': 0,
            'cv_e': 0,
            'cv_i': 0,
            'balance_ratio': 0,
            'mean_e_weight': 0,
            'mean_i_weight': 0,
            'network': None,
            'weight_rec': None,
            'success': False,
            'error_message': str(e),
        }
    
    return results

def plot_sweep_summary(all_results, save_folder, sweep_type):
    """Create summary plots of parameter sweep"""
    print(f"\nGenerating parameter sweep summary plots for {sweep_type}...")
    
    varied_scales = [r['varied_scale'] for r in all_results]
    e_rates = [r['e_rate'] for r in all_results]
    i_rates = [r['i_rate'] for r in all_results]
    cv_e = [r['cv_e'] for r in all_results]
    cv_i = [r['cv_i'] for r in all_results]
    balance_ratios = [r['balance_ratio'] for r in all_results]
    mean_e_weights = [r['mean_e_weight'] for r in all_results]
    mean_i_weights = [r['mean_i_weight'] for r in all_results]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Firing rates vs varied scale
    axs[0, 0].plot(varied_scales, e_rates, 'ro-', linewidth=2, markersize=8, label='E neurons')
    axs[0, 0].plot(varied_scales, i_rates, 'bo-', linewidth=2, markersize=8, label='I neurons')
    axs[0, 0].set_xlabel(f'{sweep_type} Weight Scale', fontsize=12)
    axs[0, 0].set_ylabel('Firing Rate (Hz)', fontsize=12)
    axs[0, 0].set_title(f'Firing Rates vs {sweep_type} Weight Scale', fontsize=13, fontweight='bold')
    axs[0, 0].legend(fontsize=11)
    axs[0, 0].grid(True, alpha=0.3)
    
    # Panel 2: CV vs varied scale
    axs[0, 1].plot(varied_scales, cv_e, 'ro-', linewidth=2, markersize=8, label='E neurons')
    axs[0, 1].plot(varied_scales, cv_i, 'bo-', linewidth=2, markersize=8, label='I neurons')
    axs[0, 1].axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='Irregular (CV~1)')
    axs[0, 1].set_xlabel(f'{sweep_type} Weight Scale', fontsize=12)
    axs[0, 1].set_ylabel('Coefficient of Variation', fontsize=12)
    axs[0, 1].set_title(f'Firing Irregularity vs {sweep_type} Weight Scale', fontsize=13, fontweight='bold')
    axs[0, 1].legend(fontsize=10)
    axs[0, 1].grid(True, alpha=0.3)
    
    # Panel 3: Mean outgoing weights vs varied scale
    axs[1, 0].plot(varied_scales, mean_e_weights, 'ro-', linewidth=2, markersize=8, label='From E sources')
    axs[1, 0].plot(varied_scales, np.abs(mean_i_weights), 'bo-', linewidth=2, markersize=8, label='From I sources (abs)')
    axs[1, 0].set_xlabel(f'{sweep_type} Weight Scale', fontsize=12)
    axs[1, 0].set_ylabel('Mean Outgoing Weight', fontsize=12)
    axs[1, 0].set_title(f'Outgoing Weights vs {sweep_type} Weight Scale', fontsize=13, fontweight='bold')
    axs[1, 0].legend(fontsize=10)
    axs[1, 0].grid(True, alpha=0.3)
    
    # Panel 4: Current balance ratio vs varied scale
    axs[1, 1].plot(varied_scales, balance_ratios, 'mo-', linewidth=2, markersize=8)
    axs[1, 1].axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Perfect balance')
    axs[1, 1].set_xlabel(f'{sweep_type} Weight Scale', fontsize=12)
    axs[1, 1].set_ylabel('|I_inh|/I_exc Current Ratio', fontsize=12)
    axs[1, 1].set_title(f'Current Balance vs {sweep_type} Weight Scale', fontsize=13, fontweight='bold')
    axs[1, 1].legend(fontsize=10)
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, f"parameter_sweep_summary_{sweep_type}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Summary plot saved to: {save_path}")
    plt.close()

def plot_all_runs_grid(all_results, save_folder, sweep_type):
    """Plot all runs in a comprehensive grid with all visualizations"""
    print(f"\nGenerating comprehensive grid for {sweep_type} sweep...")
    
    # Filter successful runs
    successful_results = [r for r in all_results if r['success'] and r['network'] is not None]
    
    if not successful_results:
        print("  No successful runs to plot")
        return
    
    n_runs = len(successful_results)
    # Columns: Weight Matrix, Histogram, Raster, Pop Rates, Mean Voltage
    n_cols = 5
    
    # First pass: collect all weight values to determine global axis limits
    all_e_to_e = []
    all_e_to_i = []
    all_i_to_e = []
    all_i_to_i = []
    for result in successful_results:
        weight_rec = result['weight_rec']
        network = result['network']
        # Four types of connections
        e_to_e = weight_rec[:network.N_rec_E, :network.N_rec_E].flatten()  # E→E: positive
        e_to_i = weight_rec[network.N_rec_E:, :network.N_rec_E].flatten()  # E→I: positive
        i_to_e = weight_rec[:network.N_rec_E, network.N_rec_E:].flatten()  # I→E: negative
        i_to_i = weight_rec[network.N_rec_E:, network.N_rec_E:].flatten()  # I→I: negative
        all_e_to_e.extend(e_to_e)
        all_e_to_i.extend(e_to_i)
        all_i_to_e.extend(i_to_e)
        all_i_to_i.extend(i_to_i)
    
    # Determine global axis limits for log scale histogram
    all_weights_combined = list(all_e_to_e) + list(all_e_to_i) + list(all_i_to_e) + list(all_i_to_i)
    if all_weights_combined:
        # For log scale, need absolute values and reasonable min/max
        abs_weights = [abs(w) for w in all_weights_combined if w != 0]
        if abs_weights:
            # Use log scale: find min and max in log space
            log_abs_weights = [np.log10(w) for w in abs_weights if w > 0]
            if log_abs_weights:
                global_x_min_log = 10**(min(log_abs_weights) - 0.5)  # Add padding in log space
                global_x_max_log = 10**(max(log_abs_weights) + 0.5)
            else:
                global_x_min_log, global_x_max_log = 0.001, 10.0
        else:
            global_x_min_log, global_x_max_log = 0.001, 10.0
        
        # For y-axis, compute max density across all four distributions
        # Need to evaluate KDE on log scale
        global_y_max = 0
        for weights in [all_e_to_e, all_e_to_i, all_i_to_e, all_i_to_i]:
            if len(weights) > 1:
                kde = gaussian_kde(weights)
                # Evaluate on log scale (but KDE is computed on linear scale)
                x_test_log = np.logspace(np.log10(max(global_x_min_log, 0.001)), 
                                        np.log10(global_x_max_log), 200)
                # For negative weights, use absolute value
                x_test_signed = np.concatenate([-x_test_log[::-1], x_test_log])
                density = kde(x_test_signed)
                global_y_max = max(global_y_max, np.max(density))
        global_y_max = global_y_max * 1.1  # Add 10% padding
    else:
        global_x_min_log, global_x_max_log = 0.001, 10.0
        global_y_max = 1.0
    
    fig = plt.figure(figsize=(24, 4*n_runs))
    gs = fig.add_gridspec(n_runs, n_cols, hspace=0.4, wspace=0.3)
    
    for idx, result in enumerate(successful_results):
        network = result['network']
        weight_rec = result['weight_rec']
        varied_scale = result['varied_scale']
        
        print(f"  Plotting run {idx+1}/{n_runs}: {sweep_type} scale={varied_scale}")
        
        # Column 0: Weight matrix (rows=target, cols=source)
        ax_weights = fig.add_subplot(gs[idx, 0])
        # Use log scale for colorbar to handle wide range of values
        abs_weights = np.abs(weight_rec[weight_rec != 0])
        if len(abs_weights) > 0:
            # Use log scale for colorbar: find min and max in log space
            log_abs_weights = np.log10(abs_weights[abs_weights > 0])
            if len(log_abs_weights) > 0:
                vmax_log = 10**(max(log_abs_weights) + 0.1)
                vmin_log = 10**(min(log_abs_weights) - 0.1)
                # For symmetric colormap, use the larger absolute value
                vmax = max(vmax_log, abs(vmin_log))
            else:
                vmax = np.max(np.abs(weight_rec))
        else:
            vmax = np.max(np.abs(weight_rec))
        # Use SymLogNorm for symmetric log scale colormap
        from matplotlib.colors import SymLogNorm
        im = ax_weights.imshow(weight_rec, cmap='RdBu_r', aspect='auto', 
                              norm=SymLogNorm(linthresh=0.01, linscale=0.5, vmin=-vmax, vmax=vmax))
        ax_weights.axhline(network.N_rec_E, color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax_weights.axvline(network.N_rec_E, color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax_weights.set_ylabel(f'{sweep_type}={varied_scale:.2f}\nTarget', fontsize=9, fontweight='bold')
        if idx == 0:
            ax_weights.set_title('Weight Matrix\n(rows=target, cols=source)', fontsize=10, fontweight='bold')
        if idx == n_runs - 1:
            ax_weights.set_xlabel('Source', fontsize=9)
        
        # Column 1: Histogram showing FOUR distributions: E→E, E→I, I→E, I→I
        ax_dist = fig.add_subplot(gs[idx, 1])
        
        # Extract four types of connections
        e_to_e_weights = weight_rec[:network.N_rec_E, :network.N_rec_E].flatten()  # E→E: positive
        e_to_i_weights = weight_rec[network.N_rec_E:, :network.N_rec_E].flatten()  # E→I: positive
        i_to_e_weights = weight_rec[:network.N_rec_E, network.N_rec_E:].flatten()  # I→E: negative
        i_to_i_weights = weight_rec[network.N_rec_E:, network.N_rec_E:].flatten()  # I→I: negative
        
        # Calculate means
        e_to_e_mean = np.mean(e_to_e_weights) if len(e_to_e_weights) > 0 else 0
        e_to_i_mean = np.mean(e_to_i_weights) if len(e_to_i_weights) > 0 else 0
        i_to_e_mean = np.mean(i_to_e_weights) if len(i_to_e_weights) > 0 else 0
        i_to_i_mean = np.mean(i_to_i_weights) if len(i_to_i_weights) > 0 else 0
        
        # Create x-axis for plotting on log scale
        # Need to handle both positive and negative weights
        # For log scale, we'll plot absolute values and use symmetric log scale
        all_weights_this_run = list(e_to_e_weights) + list(e_to_i_weights) + list(i_to_e_weights) + list(i_to_i_weights)
        abs_weights = [abs(w) for w in all_weights_this_run if w != 0]
        if abs_weights:
            log_min = np.log10(max(min(abs_weights), global_x_min_log))
            log_max = np.log10(min(max(abs_weights), global_x_max_log))
            x_plot_abs = np.logspace(log_min, log_max, 200)
            # Create symmetric x-axis: negative and positive
            x_plot = np.concatenate([-x_plot_abs[::-1], x_plot_abs])
        else:
            x_plot = np.linspace(-global_x_max_log, global_x_max_log, 200)
        
        # Plot FOUR separate KDE distributions with different colors/styles
        # E→E: red solid
        if len(e_to_e_weights) > 1:
            e_to_e_kde = gaussian_kde(e_to_e_weights)
            e_to_e_density = e_to_e_kde(x_plot)
            ax_dist.plot(x_plot, e_to_e_density, 'r-', linewidth=2, alpha=0.8, 
                        label=f'E→E: μ={e_to_e_mean:.4f}')
        
        # E→I: red dashed
        if len(e_to_i_weights) > 1:
            e_to_i_kde = gaussian_kde(e_to_i_weights)
            e_to_i_density = e_to_i_kde(x_plot)
            ax_dist.plot(x_plot, e_to_i_density, 'r--', linewidth=2, alpha=0.8, 
                        label=f'E→I: μ={e_to_i_mean:.4f}')
        
        # I→E: blue solid
        if len(i_to_e_weights) > 1:
            i_to_e_kde = gaussian_kde(i_to_e_weights)
            i_to_e_density = i_to_e_kde(x_plot)
            ax_dist.plot(x_plot, i_to_e_density, 'b-', linewidth=2, alpha=0.8, 
                        label=f'I→E: μ={i_to_e_mean:.4f}')
        
        # I→I: blue dashed
        if len(i_to_i_weights) > 1:
            i_to_i_kde = gaussian_kde(i_to_i_weights)
            i_to_i_density = i_to_i_kde(x_plot)
            ax_dist.plot(x_plot, i_to_i_density, 'b--', linewidth=2, alpha=0.8, 
                        label=f'I→I: μ={i_to_i_mean:.4f}')
        
        ax_dist.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax_dist.set_ylabel('Density', fontsize=9)
        # Use symmetric log scale for x-axis
        ax_dist.set_xscale('symlog', linthresh=0.01, linscale=0.5)
        ax_dist.set_xlim(-global_x_max_log, global_x_max_log)
        ax_dist.set_ylim(0, global_y_max)
        if idx == 0:
            ax_dist.set_title("Weight Distributions\n(E→E: red-, E→I: red--, I→E: blue-, I→I: blue--)", 
                            fontsize=10, fontweight='bold')
        ax_dist.legend(fontsize=6, loc='upper left', framealpha=0.9)
        if idx == n_runs - 1:
            ax_dist.set_xlabel('Weight Value (log scale)', fontsize=9)
        ax_dist.grid(True, alpha=0.2)
        
        # Column 2: Raster plot
        ax_raster = fig.add_subplot(gs[idx, 2])
        spike_times = network.rec_spikes.t / ms
        spike_neurons = network.rec_spikes.i
        colors = ['red' if n < network.N_rec_E else 'blue' for n in spike_neurons]
        ax_raster.scatter(spike_times, spike_neurons, s=0.5, c=colors, alpha=0.6, rasterized=True)
        ax_raster.set_ylabel('Neuron', fontsize=9)
        ax_raster.axhline(network.N_rec_E, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        if idx == 0:
            ax_raster.set_title('Raster Plot', fontsize=11, fontweight='bold')
        if idx == n_runs - 1:
            ax_raster.set_xlabel('Time (ms)', fontsize=9)
        
        # Column 3: Population rates
        ax_rates = fig.add_subplot(gs[idx, 3])
        if hasattr(network, 'smoothed_rates_rec'):
            time_points = network.smoothed_time_points * 1000
            rates_e = np.mean(network.smoothed_rates_rec[:network.N_rec_E, :], axis=0)
            rates_i = np.mean(network.smoothed_rates_rec[network.N_rec_E:, :], axis=0)
            ax_rates.plot(time_points, rates_e, 'r-', linewidth=1.5, label='E', alpha=0.8)
            ax_rates.plot(time_points, rates_i, 'b-', linewidth=1.5, label='I', alpha=0.8)
            ax_rates.set_ylabel(f'{result["e_rate"]:.1f}Hz', fontsize=9)
        if idx == 0:
            ax_rates.set_title('Population Rates', fontsize=11, fontweight='bold')
            ax_rates.legend(fontsize=8, loc='upper right')
        if idx == n_runs - 1:
            ax_rates.set_xlabel('Time (ms)', fontsize=9)
        ax_rates.grid(True, alpha=0.2)
        
        # Column 4: Mean voltage
        ax_voltage = fig.add_subplot(gs[idx, 4])
        if network.rec_voltage is not None:
            time_ms = network.rec_voltage.t / ms
            v_e = np.mean(network.rec_voltage.v[:network.N_rec_E, :], axis=0) / mV
            v_i = np.mean(network.rec_voltage.v[network.N_rec_E:, :], axis=0) / mV
            ax_voltage.plot(time_ms, v_e, 'r-', linewidth=1, alpha=0.8)
            ax_voltage.plot(time_ms, v_i, 'b-', linewidth=1, alpha=0.8)
            ax_voltage.axhline(network.v_thresh/mV, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            ax_voltage.set_ylabel(f'CV={result["cv_e"]:.2f}', fontsize=9)
        if idx == 0:
            ax_voltage.set_title('Mean Voltage', fontsize=11, fontweight='bold')
        if idx == n_runs - 1:
            ax_voltage.set_xlabel('Time (ms)', fontsize=9)
        ax_voltage.grid(True, alpha=0.2)
    
    first_config = successful_results[0]
    last_config = successful_results[-1]
    plt.suptitle(f'Parameter Sweep: {sweep_type} Weight Scale\n{n_runs} configurations from {sweep_type}={first_config["varied_scale"]:.2f} to {sweep_type}={last_config["varied_scale"]:.2f}',
                 fontsize=14, fontweight='bold', y=0.995)
    
    save_path = os.path.join(save_folder, f"all_runs_comprehensive_grid_{sweep_type}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comprehensive grid saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Create ONE folder for all four sweeps under "no_feedforward"
    base_save_folder = os.path.join(os.path.dirname(__file__), "..", "Runs")
    sweep_folder_name = f"weight_scale_sweep_all_connections_E{N_REC_E}_I{N_REC_I}"
    sweep_base_folder = os.path.join(base_save_folder, sweep_folder_name)
    
    # Move old runs to "old" folder
    if os.path.exists(sweep_base_folder):
        old_folder = os.path.join(sweep_base_folder, "old")
        os.makedirs(old_folder, exist_ok=True)
        # Move all existing timestamped folders to old folder
        for item in os.listdir(sweep_base_folder):
            item_path = os.path.join(sweep_base_folder, item)
            if os.path.isdir(item_path) and item != "old":
                # Check if it's a timestamp folder (format: YYYYMMDD_HHMMSS)
                if len(item) == 15 and item[8] == '_' and item.replace('_', '').isdigit():
                    old_item_path = os.path.join(old_folder, item)
                    if os.path.exists(old_item_path):
                        # If already exists in old, add a suffix
                        import shutil
                        counter = 1
                        while os.path.exists(old_item_path):
                            old_item_path = os.path.join(old_folder, f"{item}_{counter}")
                            counter += 1
                    os.rename(item_path, old_item_path)
                    print(f"Moved old run to: {old_item_path}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(sweep_base_folder, timestamp)
    figures_folder = os.path.join(run_folder, "Figures")
    
    print(f"\n{'='*60}")
    print(f"CREATING SWEEP DIRECTORY FOR ALL CONNECTION TYPES")
    print(f"{'='*60}")
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(figures_folder, exist_ok=True)
    
    # Run four separate sweeps, one for each connection type
    for sweep_type in SWEEP_TYPES:
        print(f"\n{'='*60}")
        print(f"BALANCED RECURRENT NETWORK - PARAMETER SWEEP: {sweep_type}")
        print(f"{'='*60}")
        print(f"Network: {N_REC_E}E + {N_REC_I}I neurons (50/50 ratio, fixed)")
        print(f"Simulation duration: {SIMULATION_DURATION} ms per run")
        print(f"Varying: {sweep_type} weight scale")
        print(f"Keeping constant: {[s for s in SWEEP_TYPES if s != sweep_type]} at scale {BASE_CONNECTION_SCALE}")
        varied_scales = VARIED_SCALES_BY_TYPE[sweep_type]
        print(f"Varied scales: {varied_scales}")
        print(f"Total runs: {len(varied_scales)}")
        print(f"{'='*60}\n")
        
        # Run sweep for this connection type
        all_results = []
        varied_scales = VARIED_SCALES_BY_TYPE[sweep_type]
        for idx, varied_scale in enumerate(varied_scales):
            results = run_single_configuration(
                sweep_type, varied_scale, idx, run_folder
            )
            all_results.append(results)
        
        # Save results
        print(f"\nSaving {sweep_type} sweep results...")
        results_file = os.path.join(run_folder, f"sweep_results_{sweep_type}.pkl")
        with open(results_file, 'wb') as f:
            # Save without network objects (too large), but keep spike data
            results_to_save = [{k: v for k, v in r.items() if k != 'network'} for r in all_results]
            pickle.dump(results_to_save, f)
        print(f"Results saved to: {results_file}")
        
        # Generate summary plots
        plot_sweep_summary(all_results, figures_folder, sweep_type)
        plot_all_runs_grid(all_results, figures_folder, sweep_type)
        
        # Print summary table
        print(f"\n{'='*100}")
        print(f"PARAMETER SWEEP SUMMARY: {sweep_type}")
        print(f"{'='*100}")
        print(f"{'Scale':<12} {'E Rate':<10} {'I Rate':<10} {'CV_E':<8} {'CV_I':<8} {'E Wt':<10} {'I Wt':<10}")
        print(f"{'-'*100}")
        for r in all_results:
            print(f"{r['varied_scale']:<12.2f} "
                  f"{r['e_rate']:<10.2f} {r['i_rate']:<10.2f} "
                  f"{r['cv_e']:<8.3f} {r['cv_i']:<8.3f} "
                  f"{r['mean_e_weight']:<10.4f} {r['mean_i_weight']:<10.4f}")
        print(f"{'='*100}")
        
        print(f"\n{'='*60}")
        print(f"SWEEP COMPLETE: {sweep_type}")
        print(f"{'='*60}\n")
    
    # Print comprehensive firing rate summary for all sweeps
    print(f"\n{'='*100}")
    print("FIRING RATE SUMMARY - ALL SWEEPS")
    print(f"{'='*100}")
    
    for sweep_type in SWEEP_TYPES:
        print(f"\n{sweep_type}:")
        print(f"{'Scale':<12} {'E Rate (Hz)':<15} {'I Rate (Hz)':<15} {'E/I Ratio':<12}")
        print(f"{'-'*60}")
        
        # Load results from pickle file
        results_file = os.path.join(run_folder, f"sweep_results_{sweep_type}.pkl")
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            for r in results:
                if r.get('success'):
                    scale = r.get('varied_scale', 0)
                    e_rate = r.get('e_rate', 0)
                    i_rate = r.get('i_rate', 0)
                    ratio = e_rate / i_rate if i_rate > 0 else 0
                    print(f"{scale:<12.4f} {e_rate:<15.2f} {i_rate:<15.2f} {ratio:<12.3f}")
        else:
            print(f"  Results file not found: {results_file}")
    
    print(f"\n{'='*100}")
    print(f"ALL SWEEPS COMPLETE")
    print(f"{'='*100}")
    print(f"Results saved to: {run_folder}")
    print(f"{'='*100}\n")

