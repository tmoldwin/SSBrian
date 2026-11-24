"""
Regenerate sweep figures with consistent histogram scaling across all sweeps.
Loads pickle files and regenerates plots with global axis limits.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys
from brian2 import *

# Import the plotting function
from balanced_recurrent_sweep import plot_all_runs_grid, plot_sweep_summary

def compute_global_limits(all_sweep_results):
    """Compute global axis limits across all sweeps"""
    all_e_to_e = []
    all_e_to_i = []
    all_i_to_e = []
    all_i_to_i = []
    
    for sweep_type, results in all_sweep_results.items():
        for result in results:
            if result.get('success') and 'weight_rec' in result:
                weight_rec = result['weight_rec']
                N_rec_E = result.get('N_rec_E', 200)  # Default if not saved
                N_rec_I = result.get('N_rec_I', 200)
                
                # Four types of connections
                e_to_e = weight_rec[:N_rec_E, :N_rec_E].flatten()
                e_to_i = weight_rec[N_rec_E:, :N_rec_E].flatten()
                i_to_e = weight_rec[:N_rec_E, N_rec_E:].flatten()
                i_to_i = weight_rec[N_rec_E:, N_rec_E:].flatten()
                
                all_e_to_e.extend(e_to_e)
                all_e_to_i.extend(e_to_i)
                all_i_to_e.extend(i_to_e)
                all_i_to_i.extend(i_to_i)
    
    # Compute global x limits (linear scale)
    all_weights_combined = list(all_e_to_e) + list(all_e_to_i) + list(all_i_to_e) + list(all_i_to_i)
    if all_weights_combined:
        global_x_min = min(all_weights_combined) * 1.1
        global_x_max = max(all_weights_combined) * 1.1
        
        # Compute global y max (max density across all distributions)
        global_y_max = 0
        for weights in [all_e_to_e, all_e_to_i, all_i_to_e, all_i_to_i]:
            if len(weights) > 1:
                kde = gaussian_kde(weights)
                x_test = np.linspace(global_x_min, global_x_max, 200)
                density = kde(x_test)
                global_y_max = max(global_y_max, np.max(density))
        global_y_max = global_y_max * 1.1
    else:
        global_x_min, global_x_max = -5, 5
        global_y_max = 1.0
    
    return global_x_min, global_x_max, global_y_max

def plot_all_runs_grid_with_global_limits(all_results, save_folder, sweep_type, global_x_min, global_x_max, global_y_max):
    """Plot all runs with global axis limits"""
    print(f"\nGenerating comprehensive grid for {sweep_type} sweep...")
    
    # Filter successful runs
    successful_results = [r for r in all_results if r.get('success') and 'weight_rec' in r]
    
    if not successful_results:
        print("  No successful runs to plot")
        return
    
    n_runs = len(successful_results)
    n_cols = 5
    
    fig = plt.figure(figsize=(24, 4*n_runs))
    gs = fig.add_gridspec(n_runs, n_cols, hspace=0.4, wspace=0.3)
    
    for idx, result in enumerate(successful_results):
        weight_rec = result['weight_rec']
        varied_scale = result['varied_scale']
        N_rec_E = result.get('N_rec_E', 200)
        N_rec_I = result.get('N_rec_I', 200)
        
        # Try to get network object if available, otherwise create dummy
        network = result.get('network', None)
        if network is None:
            # Create a simple object with just N_rec_E attribute
            class DummyNetwork:
                def __init__(self, N_rec_E, N_rec_I):
                    self.N_rec_E = N_rec_E
                    self.N_rec_I = N_rec_I
                    self.rec_spikes = None
                    self.rec_voltage = None
                    self.smoothed_rates_rec = None
                    self.smoothed_time_points = None
                    self.v_thresh = -50*mV
            network = DummyNetwork(N_rec_E, N_rec_I)
        
        print(f"  Plotting run {idx+1}/{n_runs}: {sweep_type} scale={varied_scale}")
        
        # Column 0: Weight matrix
        ax_weights = fig.add_subplot(gs[idx, 0])
        vmax = np.max(np.abs(weight_rec))
        im = ax_weights.imshow(weight_rec, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax_weights.axhline(N_rec_E, color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax_weights.axvline(N_rec_E, color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax_weights.set_ylabel(f'{sweep_type}={varied_scale:.2f}\nTarget', fontsize=9, fontweight='bold')
        if idx == 0:
            ax_weights.set_title('Weight Matrix\n(rows=target, cols=source)', fontsize=10, fontweight='bold')
        if idx == n_runs - 1:
            ax_weights.set_xlabel('Source', fontsize=9)
        
        # Column 1: Histogram with GLOBAL limits
        ax_dist = fig.add_subplot(gs[idx, 1])
        
        e_to_e_weights = weight_rec[:N_rec_E, :N_rec_E].flatten()
        e_to_i_weights = weight_rec[N_rec_E:, :N_rec_E].flatten()
        i_to_e_weights = weight_rec[:N_rec_E, N_rec_E:].flatten()
        i_to_i_weights = weight_rec[N_rec_E:, N_rec_E:].flatten()
        
        e_to_e_mean = np.mean(e_to_e_weights) if len(e_to_e_weights) > 0 else 0
        e_to_i_mean = np.mean(e_to_i_weights) if len(e_to_i_weights) > 0 else 0
        i_to_e_mean = np.mean(i_to_e_weights) if len(i_to_e_weights) > 0 else 0
        i_to_i_mean = np.mean(i_to_i_weights) if len(i_to_i_weights) > 0 else 0
        
        # Use GLOBAL x limits for consistent scaling
        x_plot = np.linspace(global_x_min, global_x_max, 200)
        
        # Plot four distributions
        if len(e_to_e_weights) > 1:
            e_to_e_kde = gaussian_kde(e_to_e_weights)
            e_to_e_density = e_to_e_kde(x_plot)
            ax_dist.plot(x_plot, e_to_e_density, 'r-', linewidth=2, alpha=0.8, 
                        label=f'E→E: μ={e_to_e_mean:.4f}')
        
        if len(e_to_i_weights) > 1:
            e_to_i_kde = gaussian_kde(e_to_i_weights)
            e_to_i_density = e_to_i_kde(x_plot)
            ax_dist.plot(x_plot, e_to_i_density, 'r--', linewidth=2, alpha=0.8, 
                        label=f'E→I: μ={e_to_i_mean:.4f}')
        
        if len(i_to_e_weights) > 1:
            i_to_e_kde = gaussian_kde(i_to_e_weights)
            i_to_e_density = i_to_e_kde(x_plot)
            ax_dist.plot(x_plot, i_to_e_density, 'b-', linewidth=2, alpha=0.8, 
                        label=f'I→E: μ={i_to_e_mean:.4f}')
        
        if len(i_to_i_weights) > 1:
            i_to_i_kde = gaussian_kde(i_to_i_weights)
            i_to_i_density = i_to_i_kde(x_plot)
            ax_dist.plot(x_plot, i_to_i_density, 'b--', linewidth=2, alpha=0.8, 
                        label=f'I→I: μ={i_to_i_mean:.4f}')
        
        ax_dist.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax_dist.set_ylabel('Density', fontsize=9)
        ax_dist.set_xlim(global_x_min, global_x_max)  # Use GLOBAL limits
        ax_dist.set_ylim(0, global_y_max)  # Use GLOBAL limits
        if idx == 0:
            ax_dist.set_title("Weight Distributions\n(E→E: red-, E→I: red--, I→E: blue-, I→I: blue--)", 
                            fontsize=10, fontweight='bold')
        ax_dist.legend(fontsize=6, loc='upper left', framealpha=0.9)
        if idx == n_runs - 1:
            ax_dist.set_xlabel('Weight Value', fontsize=9)
        ax_dist.grid(True, alpha=0.2)
        
        # Column 2: Raster plot (use saved spike data)
        ax_raster = fig.add_subplot(gs[idx, 2])
        spike_times = result.get('spike_times')
        spike_neurons = result.get('spike_neurons')
        if spike_times is not None and spike_neurons is not None:
            colors = ['red' if n < N_rec_E else 'blue' for n in spike_neurons]
            ax_raster.scatter(spike_times, spike_neurons, s=0.5, c=colors, alpha=0.6, rasterized=True)
        ax_raster.set_ylabel('Neuron', fontsize=9)
        ax_raster.axhline(N_rec_E, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        if idx == 0:
            ax_raster.set_title('Raster Plot', fontsize=11, fontweight='bold')
        if idx == n_runs - 1:
            ax_raster.set_xlabel('Time (ms)', fontsize=9)
        
        # Column 3: Population rates
        ax_rates = fig.add_subplot(gs[idx, 3])
        if network.smoothed_rates_rec is not None:
            time_points = network.smoothed_time_points * 1000
            rates_e = np.mean(network.smoothed_rates_rec[:N_rec_E, :], axis=0)
            rates_i = np.mean(network.smoothed_rates_rec[N_rec_E:, :], axis=0)
            ax_rates.plot(time_points, rates_e, 'r-', linewidth=1.5, label='E', alpha=0.8)
            ax_rates.plot(time_points, rates_i, 'b-', linewidth=1.5, label='I', alpha=0.8)
            ax_rates.set_ylabel(f'{result.get("e_rate", 0):.1f}Hz', fontsize=9)
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
            v_e = np.mean(network.rec_voltage.v[:N_rec_E, :], axis=0) / mV
            v_i = np.mean(network.rec_voltage.v[N_rec_E:, :], axis=0) / mV
            ax_voltage.plot(time_ms, v_e, 'r-', linewidth=1, alpha=0.8)
            ax_voltage.plot(time_ms, v_i, 'b-', linewidth=1, alpha=0.8)
            ax_voltage.axhline(network.v_thresh/mV, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            ax_voltage.set_ylabel(f'CV={result.get("cv_e", 0):.2f}', fontsize=9)
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python regenerate_sweep_figures.py <run_folder>")
        print("\nExample:")
        print("  python regenerate_sweep_figures.py ../Runs/weight_scale_sweep_all_connections_E200_I200/20251115_164538")
        sys.exit(1)
    
    run_folder = sys.argv[1]
    if not os.path.exists(run_folder):
        print(f"Error: Run folder not found: {run_folder}")
        sys.exit(1)
    
    figures_folder = os.path.join(run_folder, "Figures")
    
    # Load all pickle files
    SWEEP_TYPES = ['E_to_E', 'E_to_I', 'I_to_E', 'I_to_I']
    all_sweep_results = {}
    
    print(f"Loading pickle files from: {run_folder}")
    for sweep_type in SWEEP_TYPES:
        pickle_file = os.path.join(run_folder, f"sweep_results_{sweep_type}.pkl")
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                results = pickle.load(f)
            all_sweep_results[sweep_type] = results
            print(f"  Loaded {len(results)} results for {sweep_type}")
        else:
            print(f"  Warning: {pickle_file} not found")
    
    if not all_sweep_results:
        print("Error: No pickle files found!")
        sys.exit(1)
    
    # Compute global limits across ALL sweeps
    print("\nComputing global axis limits across all sweeps...")
    global_x_min, global_x_max, global_y_max = compute_global_limits(all_sweep_results)
    print(f"  Global x limits: [{global_x_min:.4f}, {global_x_max:.4f}]")
    print(f"  Global y max: {global_y_max:.4f}")
    
    # Regenerate plots for each sweep with global limits
    print("\nRegenerating plots with consistent scaling...")
    for sweep_type, results in all_sweep_results.items():
        plot_all_runs_grid_with_global_limits(results, figures_folder, sweep_type, 
                                             global_x_min, global_x_max, global_y_max)
        # Also regenerate summary plots
        plot_sweep_summary(results, figures_folder, sweep_type)
    
    print("\n" + "="*60)
    print("FIGURE REGENERATION COMPLETE")
    print("="*60)
    print(f"All figures saved to: {figures_folder}")

if __name__ == "__main__":
    main()

