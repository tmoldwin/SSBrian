"""
Plot Smooth Firing Rates by Population

Creates a figure for each population showing smooth firing rates for exemplar neurons
across different stimuli. Each row shows a different neuron, each column shows a 
different stimulus, with time aligned to stimulus onset (t=0).

Usage:
    python plot_smooth_firing_rates.py <run_folder>
    python plot_smooth_firing_rates.py --config <config_name>
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage
import os
import sys


def get_population_colors():
    """Return color mapping for populations."""
    return {
        'L1_E': '#E74C3C',  # Red
        'L1_I': '#3498DB',  # Blue  
        'L2_E': '#E67E22',  # Orange
        'L2_I': '#9B59B6',  # Purple
        'L3_E': '#27AE60',  # Green
        'L3_I': '#1ABC9C',  # Teal
        'L4_E': '#F39C12',  # Yellow-orange
        'L4_I': '#8E44AD',  # Dark purple
    }


def compute_smooth_firing_rate(spike_indices, spike_times, neuron_idx, t_start, t_end, 
                                dt_ms=1.0, window_ms=20.0):
    """
    Compute smoothed firing rate for a single neuron in a time window.
    
    Parameters
    ----------
    spike_indices : array
        Neuron indices for each spike
    spike_times : array
        Spike times in seconds
    neuron_idx : int
        Index of neuron to compute rate for
    t_start : float
        Start time in seconds
    t_end : float
        End time in seconds
    dt_ms : float
        Time resolution in ms
    window_ms : float
        Smoothing window in ms
        
    Returns
    -------
    time_axis : array
        Time points in ms relative to t_start
    smooth_rate : array
        Smoothed firing rate in Hz
    """
    # Convert to ms for internal calculations
    t_start_ms = t_start * 1000
    t_end_ms = t_end * 1000
    duration_ms = t_end_ms - t_start_ms
    
    n_bins = int(duration_ms / dt_ms)
    time_axis = np.arange(n_bins) * dt_ms  # Time relative to stimulus onset
    
    # Create spike train for this neuron in this window
    spike_train = np.zeros(n_bins)
    
    # Get spikes for this neuron
    neuron_mask = spike_indices == neuron_idx
    neuron_spike_times = spike_times[neuron_mask]  # in seconds
    
    # Filter to time window and convert to bins
    window_mask = (neuron_spike_times >= t_start) & (neuron_spike_times < t_end)
    window_spikes_ms = (neuron_spike_times[window_mask] - t_start) * 1000
    
    # Bin the spikes
    bin_indices = (window_spikes_ms / dt_ms).astype(int)
    bin_indices = bin_indices[bin_indices < n_bins]  # Safety check
    
    for idx in bin_indices:
        spike_train[idx] += 1
    
    # Smooth with Gaussian kernel
    window_bins = int(window_ms / dt_ms)
    sigma = window_bins / 4  # Standard deviation in bins
    
    if sigma > 0:
        smooth_rate = ndimage.gaussian_filter1d(spike_train, sigma)
        # Convert to Hz: spikes per bin -> spikes per second
        smooth_rate = smooth_rate / (dt_ms / 1000)
    else:
        smooth_rate = spike_train / (dt_ms / 1000)
    
    return time_axis, smooth_rate


def plot_smooth_firing_rates_grid(run_folder, n_neurons_per_pop=5, seed=42, window_ms=20.0):
    """
    Create smooth firing rate plots for each population.
    
    Parameters
    ----------
    run_folder : str
        Path to the run directory containing Data/simulation_data.pkl
    n_neurons_per_pop : int
        Number of exemplar neurons to show per population
    seed : int
        Random seed for reproducibility
    window_ms : float
        Smoothing window in milliseconds
    """
    data_path = os.path.join(run_folder, 'Data', 'simulation_data.pkl')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    print(f"Loading simulation data from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    spike_data = data.get('spike_data', {})
    stim_info = data.get('stim_info', [])
    config = data.get('config', {})
    population_configs = config.get('populations', [])
    
    if not spike_data:
        raise ValueError("No spike data found in the file.")
    
    if not stim_info:
        raise ValueError("No stimulus info found in the file.")
    
    # Get stimulus information
    # stim_info is a list of (stim_id, start_time, end_time) tuples
    # Times are in brian2 units, need to convert to seconds
    stim_list = []
    for stim_id, start, end in stim_info:
        # Handle brian2 quantities
        if hasattr(start, 'item'):
            start_s = float(start)  # Already in seconds if brian2 Quantity
        else:
            start_s = float(start) / 1000  # Assume ms if plain number
        if hasattr(end, 'item'):
            end_s = float(end)
        else:
            end_s = float(end) / 1000
        stim_list.append((int(stim_id), start_s, end_s))
    
    # Get unique stimuli (first presentation of each)
    unique_stims = {}
    for stim_id, start, end in stim_list:
        if stim_id not in unique_stims:
            unique_stims[stim_id] = (start, end)
    
    n_stimuli = len(unique_stims)
    stim_ids = sorted(unique_stims.keys())
    
    # Get stimulus duration from first stimulus
    stim_duration_s = stim_list[0][2] - stim_list[0][1]
    stim_duration_ms = stim_duration_s * 1000
    
    print(f"Found {n_stimuli} stimuli, duration: {stim_duration_ms:.1f} ms")
    
    # Get ordered population names
    pop_colors = get_population_colors()
    ordered_pops = [p['name'] for p in population_configs if p['name'] != 'Layer0']
    
    if not ordered_pops:
        ordered_pops = [k for k in spike_data.keys() if k != 'Layer0']
    
    np.random.seed(seed)
    
    # Create output folder
    figures_folder = os.path.join(run_folder, 'Figures')
    os.makedirs(figures_folder, exist_ok=True)
    
    # Generate plots for each population
    for pop_name in ordered_pops:
        if pop_name not in spike_data:
            print(f"   Skipping {pop_name}: no spike data")
            continue
        
        print(f"   Generating smooth firing rate plot for {pop_name}...")
        
        pop_spike_data = spike_data[pop_name]
        spike_times = np.array(pop_spike_data['times'])  # in seconds (brian2)
        spike_indices = np.array(pop_spike_data['indices'])
        spike_counts = np.array(pop_spike_data['count'])
        
        n_neurons = len(spike_counts)
        
        # Select active neurons (those with at least some spikes)
        active_neurons = np.where(spike_counts > 0)[0]
        if len(active_neurons) == 0:
            print(f"      No active neurons in {pop_name}, skipping")
            continue
        
        # Select exemplar neurons - prefer neurons with more spikes
        n_to_select = min(n_neurons_per_pop, len(active_neurons))
        
        # Sort by spike count and take a spread of neurons
        sorted_by_spikes = active_neurons[np.argsort(spike_counts[active_neurons])[::-1]]
        
        if len(sorted_by_spikes) >= n_to_select:
            # Take evenly spaced neurons from the sorted list
            indices = np.linspace(0, len(sorted_by_spikes) - 1, n_to_select, dtype=int)
            selected_neurons = sorted_by_spikes[indices]
        else:
            selected_neurons = sorted_by_spikes
        
        selected_neurons = np.sort(selected_neurons)
        
        # Create figure: rows = neurons, columns = stimuli
        fig, axes = plt.subplots(n_to_select, n_stimuli, 
                                  figsize=(2.5 * n_stimuli, 2 * n_to_select),
                                  squeeze=False, sharex=True, sharey='row')
        
        # Get population color
        pop_color = pop_colors.get(pop_name, '#333333')
        
        # Get stimulus colors
        stim_cmap = plt.cm.tab10
        stim_colors = {stim_id: stim_cmap(i % 10) for i, stim_id in enumerate(stim_ids)}
        
        # Find global y-max for consistent scaling
        global_max = 0
        all_rates = []
        
        # First pass: compute all rates and find max
        for row_idx, neuron_idx in enumerate(selected_neurons):
            neuron_rates = []
            for col_idx, stim_id in enumerate(stim_ids):
                # Get all presentations of this stimulus
                stim_presentations = [(s, e) for sid, s, e in stim_list if sid == stim_id]
                
                # Average smooth firing rate across presentations
                all_smooth_rates = []
                for start_s, end_s in stim_presentations:
                    time_axis, smooth_rate = compute_smooth_firing_rate(
                        spike_indices, spike_times, neuron_idx,
                        start_s, end_s, dt_ms=1.0, window_ms=window_ms
                    )
                    all_smooth_rates.append(smooth_rate)
                
                if all_smooth_rates:
                    # Ensure all arrays have same length (should be)
                    min_len = min(len(r) for r in all_smooth_rates)
                    all_smooth_rates = [r[:min_len] for r in all_smooth_rates]
                    mean_rate = np.mean(all_smooth_rates, axis=0)
                    std_rate = np.std(all_smooth_rates, axis=0)
                    time_axis = time_axis[:min_len]
                else:
                    mean_rate = np.zeros_like(time_axis)
                    std_rate = np.zeros_like(time_axis)
                
                neuron_rates.append((time_axis, mean_rate, std_rate))
                global_max = max(global_max, np.max(mean_rate + std_rate))
            
            all_rates.append(neuron_rates)
        
        # Second pass: plot
        for row_idx, neuron_idx in enumerate(selected_neurons):
            for col_idx, stim_id in enumerate(stim_ids):
                ax = axes[row_idx, col_idx]
                
                time_axis, mean_rate, std_rate = all_rates[row_idx][col_idx]
                
                # Plot mean with shaded error region
                stim_color = stim_colors[stim_id]
                ax.fill_between(time_axis, mean_rate - std_rate, mean_rate + std_rate,
                               alpha=0.3, color=stim_color, linewidth=0)
                ax.plot(time_axis, mean_rate, color=stim_color, linewidth=1.5)
                
                # Add stimulus period indicator (full duration)
                ax.axvspan(0, stim_duration_ms, alpha=0.1, color='gray')
                ax.axvline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
                
                # Labels
                if row_idx == 0:
                    ax.set_title(f'Stim {stim_id}', fontsize=10, fontweight='bold')
                
                if col_idx == 0:
                    ax.set_ylabel(f'N{neuron_idx}\n(Hz)', fontsize=9)
                
                if row_idx == n_to_select - 1:
                    ax.set_xlabel('Time (ms)', fontsize=9)
                
                ax.set_xlim(0, stim_duration_ms)
                ax.set_ylim(0, global_max * 1.1 if global_max > 0 else 1)
                ax.tick_params(axis='both', labelsize=7)
                ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Smooth Firing Rates: {pop_name}\n(window={window_ms:.0f}ms, {len(stim_list)//n_stimuli} epochs averaged)', 
                    fontsize=12, fontweight='bold', y=0.98, color=pop_color)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save in the population's temporal_evolution folder
        pop_folder = os.path.join(figures_folder, f'temporal_evolution_{pop_name}')
        os.makedirs(pop_folder, exist_ok=True)
        output_path = os.path.join(pop_folder, 'smooth_firing_rates.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"      Saved: {output_path}")
    
    return figures_folder


def plot_population_overlaid_rates(run_folder, window_ms=20.0, max_neurons_per_panel=50):
    """
    Create overlaid smooth firing rate plots for all populations.
    
    Each row is a population, each column is a stimulus. Each panel shows
    all neurons' traces overlaid (thin lines) with the average as a thicker line.
    
    Parameters
    ----------
    run_folder : str
        Path to the run directory containing Data/simulation_data.pkl
    window_ms : float
        Smoothing window in milliseconds
    max_neurons_per_panel : int
        Maximum number of individual neuron traces to show (for performance)
    """
    data_path = os.path.join(run_folder, 'Data', 'simulation_data.pkl')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    print(f"Loading simulation data from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    spike_data = data.get('spike_data', {})
    stim_info = data.get('stim_info', [])
    config = data.get('config', {})
    population_configs = config.get('populations', [])
    
    if not spike_data:
        raise ValueError("No spike data found in the file.")
    
    if not stim_info:
        raise ValueError("No stimulus info found in the file.")
    
    # Get stimulus information
    stim_list = []
    for stim_id, start, end in stim_info:
        if hasattr(start, 'item'):
            start_s = float(start)
        else:
            start_s = float(start) / 1000
        if hasattr(end, 'item'):
            end_s = float(end)
        else:
            end_s = float(end) / 1000
        stim_list.append((int(stim_id), start_s, end_s))
    
    # Get unique stimuli
    unique_stims = {}
    for stim_id, start, end in stim_list:
        if stim_id not in unique_stims:
            unique_stims[stim_id] = (start, end)
    
    n_stimuli = len(unique_stims)
    stim_ids = sorted(unique_stims.keys())
    
    # Get stimulus duration
    stim_duration_s = stim_list[0][2] - stim_list[0][1]
    stim_duration_ms = stim_duration_s * 1000
    
    print(f"Found {n_stimuli} stimuli, duration: {stim_duration_ms:.1f} ms")
    
    # Get ordered population names
    pop_colors = get_population_colors()
    ordered_pops = [p['name'] for p in population_configs if p['name'] != 'Layer0']
    
    if not ordered_pops:
        ordered_pops = [k for k in spike_data.keys() if k != 'Layer0']
    
    # Create output folder
    figures_folder = os.path.join(run_folder, 'Figures')
    os.makedirs(figures_folder, exist_ok=True)
    
    # Create single figure with all populations
    n_pops = len([p for p in ordered_pops if p in spike_data])
    
    if n_pops == 0:
        print("No populations with spike data found.")
        return
    
    fig, axes = plt.subplots(n_pops, n_stimuli, 
                              figsize=(2.5 * n_stimuli, 2 * n_pops),
                              squeeze=False, sharex=True, sharey='row')
    
    # Get stimulus colors
    stim_cmap = plt.cm.tab10
    stim_colors = {stim_id: stim_cmap(i % 10) for i, stim_id in enumerate(stim_ids)}
    
    pop_idx = 0
    for pop_name in ordered_pops:
        if pop_name not in spike_data:
            continue
        
        print(f"   Processing {pop_name}...")
        
        pop_spike_data = spike_data[pop_name]
        spike_times = np.array(pop_spike_data['times'])
        spike_indices = np.array(pop_spike_data['indices'])
        spike_counts = np.array(pop_spike_data['count'])
        
        n_neurons = len(spike_counts)
        active_neurons = np.where(spike_counts > 0)[0]
        
        if len(active_neurons) == 0:
            print(f"      No active neurons in {pop_name}, skipping")
            continue
        
        # Limit number of neurons to plot for performance
        n_to_plot = min(len(active_neurons), max_neurons_per_panel)
        if len(active_neurons) > max_neurons_per_panel:
            # Sample evenly across active neurons
            indices = np.linspace(0, len(active_neurons) - 1, n_to_plot, dtype=int)
            neurons_to_plot = active_neurons[indices]
        else:
            neurons_to_plot = active_neurons
        
        pop_color = pop_colors.get(pop_name, '#333333')
        
        # Compute rates for all neurons and stimuli
        all_neuron_rates = []  # List of lists: [neuron][stimulus] -> (time, rate)
        global_max = 0
        
        for neuron_idx in neurons_to_plot:
            neuron_rates = []
            for stim_id in stim_ids:
                # Get all presentations of this stimulus
                stim_presentations = [(s, e) for sid, s, e in stim_list if sid == stim_id]
                
                # Average smooth firing rate across presentations
                all_smooth_rates = []
                for start_s, end_s in stim_presentations:
                    time_axis, smooth_rate = compute_smooth_firing_rate(
                        spike_indices, spike_times, neuron_idx,
                        start_s, end_s, dt_ms=1.0, window_ms=window_ms
                    )
                    all_smooth_rates.append(smooth_rate)
                
                if all_smooth_rates:
                    min_len = min(len(r) for r in all_smooth_rates)
                    all_smooth_rates = [r[:min_len] for r in all_smooth_rates]
                    mean_rate = np.mean(all_smooth_rates, axis=0)
                    time_axis = time_axis[:min_len]
                else:
                    mean_rate = np.zeros(int(stim_duration_ms))
                    time_axis = np.arange(len(mean_rate))
                
                neuron_rates.append((time_axis, mean_rate))
                global_max = max(global_max, np.max(mean_rate))
            
            all_neuron_rates.append(neuron_rates)
        
        # Now plot for each stimulus
        for col_idx, stim_id in enumerate(stim_ids):
            ax = axes[pop_idx, col_idx]
            stim_color = stim_colors[stim_id]
            
            # Collect all rates for this stimulus across neurons
            rates_for_stim = []
            time_axis = None
            
            for neuron_rates in all_neuron_rates:
                time_axis, rate = neuron_rates[col_idx]
                rates_for_stim.append(rate)
            
            if rates_for_stim:
                # Ensure all have same length
                min_len = min(len(r) for r in rates_for_stim)
                rates_for_stim = [r[:min_len] for r in rates_for_stim]
                time_axis = time_axis[:min_len]
                
                # Plot individual neuron traces (thin, semi-transparent)
                for rate in rates_for_stim:
                    ax.plot(time_axis, rate, color=stim_color, linewidth=0.5, alpha=0.3)
                
                # Compute and plot average (thick line)
                avg_rate = np.mean(rates_for_stim, axis=0)
                ax.plot(time_axis, avg_rate, color=stim_color, linewidth=2.5, 
                       label='Average' if col_idx == 0 else '')
            
            # Add stimulus period indicator
            ax.axvspan(0, stim_duration_ms, alpha=0.05, color='gray')
            ax.axvline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            
            # Labels
            if pop_idx == 0:
                ax.set_title(f'Stim {stim_id}', fontsize=10, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel(f'{pop_name}\n(Hz)', fontsize=9, color=pop_color, fontweight='bold')
            
            if pop_idx == n_pops - 1:
                ax.set_xlabel('Time (ms)', fontsize=9)
            
            ax.set_xlim(0, stim_duration_ms)
            ax.set_ylim(0, global_max * 1.1 if global_max > 0 else 1)
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(True, alpha=0.3)
        
        pop_idx += 1
    
    fig.suptitle(f'Population Smooth Firing Rates (Overlaid)\n(window={window_ms:.0f}ms, {len(stim_list)//n_stimuli} epochs averaged)', 
                fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save in main Figures folder
    output_path = os.path.join(figures_folder, 'population_smooth_firing_rates_overlaid.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"      Saved: {output_path}")
    
    return figures_folder


def main():
    parser = argparse.ArgumentParser(
        description='Plot smooth firing rates for each population (layer-cell type)'
    )
    parser.add_argument(
        'run_folder',
        nargs='?',
        type=str,
        help='Path to run folder containing Data/simulation_data.pkl'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Config name (e.g., three_layer_no_skip). Will look in Runs/single_runs/<config>/latest'
    )
    parser.add_argument(
        '--neurons', '-n',
        type=int,
        default=5,
        help='Number of neurons to show per population (default: 5)'
    )
    parser.add_argument(
        '--window', '-w',
        type=float,
        default=20.0,
        help='Smoothing window in ms (default: 20.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for neuron selection (default: 42)'
    )
    parser.add_argument(
        '--overlaid',
        action='store_true',
        help='Generate overlaid population plot (all neurons in each panel)'
    )
    parser.add_argument(
        '--max-neurons',
        type=int,
        default=50,
        help='Max neurons to plot per panel in overlaid mode (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Resolve run folder path
    if args.config:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        run_folder = os.path.join(script_dir, 'Runs', 'single_runs', args.config, 'latest')
    elif args.run_folder:
        run_folder = args.run_folder
    else:
        parser.error("Must provide either run_folder or --config")
    
    if not os.path.exists(run_folder):
        print(f"Error: Run folder not found: {run_folder}")
        sys.exit(1)
    
    try:
        if args.overlaid:
            plot_population_overlaid_rates(
                run_folder,
                window_ms=args.window,
                max_neurons_per_panel=args.max_neurons
            )
        else:
            plot_smooth_firing_rates_grid(
                run_folder, 
                n_neurons_per_pop=args.neurons,
                seed=args.seed,
                window_ms=args.window
            )
        print("\nPlot generation complete!")
    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

