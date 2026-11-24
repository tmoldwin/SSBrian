"""
Analyze random search results and create metrics figure.
Selects 10 random runs and displays raster plots with metrics below.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import glob

def find_latest_random_search_results():
    """Find the most recent random search results folder"""
    base_folder = os.path.join(os.path.dirname(__file__), "Runs", "random_search")
    pattern = os.path.join(base_folder, "random_search_EtoI_ItoE_E100_I100", "*", "random_search_summary.pkl")
    summary_files = glob.glob(pattern)
    
    if not summary_files:
        raise FileNotFoundError("No random search results found")
    
    # Sort by modification time, get most recent
    latest_file = max(summary_files, key=os.path.getmtime)
    return latest_file, os.path.dirname(latest_file)

def calculate_per_neuron_metrics(result):
    """
    Calculate per-neuron metrics for a single run.
    
    Fano Factor: Measures variability in spike counts relative to mean.
    Fano = variance(spike_counts) / mean(spike_counts)
    For a Poisson process, Fano = 1. Values > 1 indicate over-dispersion (more variable),
    values < 1 indicate under-dispersion (more regular).
    """
    spike_times_all = result.get('spike_times_all')
    spike_neurons_all = result.get('spike_neurons_all')
    
    # Get simulation duration (assume 1000 ms if not available)
    sim_duration_ms = 1000.0
    sim_duration_s = sim_duration_ms / 1000.0
    total_neurons = 200  # 100E + 100I
    
    # Initialize lists for per-neuron metrics
    firing_rates = []
    cv_isis = []  # Coefficient of variation of ISIs (variance within neuron)
    fano_factors = []
    
    # Bin size for Fano factor calculation
    bin_size_ms = 50.0
    bins = np.arange(0, sim_duration_ms + bin_size_ms, bin_size_ms)
    
    for neuron_id in range(total_neurons):
        mask = spike_neurons_all == neuron_id
        spike_times_neuron = spike_times_all[mask] if spike_times_all is not None else np.array([])
        
        # 1. Firing rate per neuron
        n_spikes = len(spike_times_neuron)
        rate = n_spikes / sim_duration_s
        firing_rates.append(rate)
        
        # 2. CV of ISIs (variance within neuron)
        if len(spike_times_neuron) > 2:
            isis = np.diff(spike_times_neuron)
            if len(isis) > 0 and np.mean(isis) > 0:
                cv = np.std(isis) / np.mean(isis)
                cv_isis.append(cv)
            else:
                cv_isis.append(0.0)
        else:
            cv_isis.append(0.0)
        
        # 3. Fano factor per neuron (variance/mean of spike counts in time bins)
        # Fano Factor measures variability: Fano = 1 for Poisson, >1 for over-dispersed, <1 for regular
        if len(spike_times_neuron) > 0:
            spike_counts = np.histogram(spike_times_neuron, bins=bins)[0]
            if len(spike_counts) > 0 and np.mean(spike_counts) > 0:
                fano = np.var(spike_counts) / np.mean(spike_counts)
                fano_factors.append(fano)
            else:
                fano_factors.append(0.0)
        else:
            fano_factors.append(0.0)
    
    # 4. Per-neuron correlations using smoothed z-scored firing rates
    mean_correlations = calculate_per_neuron_correlations(spike_times_all, spike_neurons_all, sim_duration_ms)
    
    # 4. Aggregate firing rates (smoothed and summed)
    aggregate_rates = calculate_aggregate_firing_rates(spike_times_all, spike_neurons_all, sim_duration_ms)
    
    return {
        'firing_rates': np.array(firing_rates),
        'cv_isis': np.array(cv_isis),
        'fano_factors': np.array(fano_factors),
        'mean_correlations': mean_correlations,
        'aggregate_rates': aggregate_rates
    }

def calculate_aggregate_firing_rates(spike_times_all, spike_neurons_all, sim_duration_ms):
    """
    Calculate smoothed and summed aggregate firing rates for E, I, and both.
    Returns time series for E, I, and combined.
    """
    total_neurons = 200
    n_e = 100
    n_i = 100
    
    # Fine bins for smoothing
    bin_size_ms = 10.0
    bins = np.arange(0, sim_duration_ms + bin_size_ms, bin_size_ms)
    n_bins = len(bins) - 1
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    # Calculate binned firing rates
    e_rates = np.zeros(n_bins)
    i_rates = np.zeros(n_bins)
    
    if spike_times_all is not None and len(spike_times_all) > 0:
        # E neurons (neuron_id < 100)
        e_mask = spike_neurons_all < n_e
        e_spike_times = spike_times_all[e_mask]
        if len(e_spike_times) > 0:
            e_counts, _ = np.histogram(e_spike_times, bins=bins)
            e_rates = e_counts / (n_e * bin_size_ms / 1000.0)  # Convert to Hz
        
        # I neurons (neuron_id >= 100)
        i_mask = spike_neurons_all >= n_e
        i_spike_times = spike_times_all[i_mask]
        if len(i_spike_times) > 0:
            i_counts, _ = np.histogram(i_spike_times, bins=bins)
            i_rates = i_counts / (n_i * bin_size_ms / 1000.0)  # Convert to Hz
    
    # Smooth with Gaussian kernel (sigma = 2 bins = 20ms)
    sigma_bins = 2.0
    e_rates_smoothed = gaussian_filter1d(e_rates, sigma=sigma_bins)
    i_rates_smoothed = gaussian_filter1d(i_rates, sigma=sigma_bins)
    combined_rates_smoothed = e_rates_smoothed + i_rates_smoothed
    
    return {
        'time': bin_centers,
        'e_rate': e_rates_smoothed,
        'i_rate': i_rates_smoothed,
        'combined_rate': combined_rates_smoothed
    }

def calculate_per_neuron_correlations(spike_times_all, spike_neurons_all, sim_duration_ms):
    """
    Calculate mean correlation for each neuron with all other neurons.
    Uses smoothed z-scored firing rates.
    """
    total_neurons = 200
    bin_size_ms = 10.0  # Fine bins for smoothing
    bins = np.arange(0, sim_duration_ms + bin_size_ms, bin_size_ms)
    n_bins = len(bins) - 1
    
    # Calculate binned firing rates for each neuron
    firing_rate_matrix = np.zeros((total_neurons, n_bins))
    
    for neuron_id in range(total_neurons):
        mask = spike_neurons_all == neuron_id
        spike_times_neuron = spike_times_all[mask] if spike_times_all is not None else np.array([])
        if len(spike_times_neuron) > 0:
            counts, _ = np.histogram(spike_times_neuron, bins=bins)
            firing_rate_matrix[neuron_id, :] = counts
    
    # Smooth with Gaussian kernel (sigma = 2 bins = 20ms)
    sigma_bins = 2.0
    smoothed_rates = np.zeros_like(firing_rate_matrix)
    for neuron_id in range(total_neurons):
        smoothed_rates[neuron_id, :] = gaussian_filter1d(firing_rate_matrix[neuron_id, :], sigma=sigma_bins)
    
    # Z-score each neuron's firing rate time series
    z_scored_rates = np.zeros_like(smoothed_rates)
    for neuron_id in range(total_neurons):
        rates = smoothed_rates[neuron_id, :]
        if np.std(rates) > 0:
            z_scored_rates[neuron_id, :] = (rates - np.mean(rates)) / np.std(rates)
        else:
            z_scored_rates[neuron_id, :] = 0.0
    
    # Calculate pairwise correlations and mean correlation per neuron
    mean_correlations = np.zeros(total_neurons)
    
    for neuron_id in range(total_neurons):
        correlations = []
        for other_id in range(total_neurons):
            if neuron_id != other_id:
                rates_i = z_scored_rates[neuron_id, :]
                rates_j = z_scored_rates[other_id, :]
                if np.std(rates_i) > 0 and np.std(rates_j) > 0:
                    corr, _ = pearsonr(rates_i, rates_j)
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        if len(correlations) > 0:
            mean_correlations[neuron_id] = np.mean(correlations)
        else:
            mean_correlations[neuron_id] = 0.0
    
    return mean_correlations

def create_metrics_figure(all_results, save_folder, n_runs=10):
    """Create a figure with raster plots (top row) and per-neuron metric histograms (bottom rows) for n_runs"""
    print(f"\nGenerating metrics figure with {n_runs} random runs...")
    
    # Filter successful results
    successful_results = [r for r in all_results if r.get('success', False) and r.get('spike_times_all') is not None]
    
    if len(successful_results) == 0:
        print("  No successful runs with spike data to plot")
        return
    
    # Randomly select n_runs
    if len(successful_results) < n_runs:
        n_runs = len(successful_results)
    
    np.random.seed(42)  # For reproducibility
    selected_indices = np.random.choice(len(successful_results), size=n_runs, replace=False)
    selected_results = [successful_results[i] for i in selected_indices]
    
    # Calculate per-neuron metrics for each selected run
    per_neuron_metrics_list = []
    for result in selected_results:
        metrics = calculate_per_neuron_metrics(result)
        per_neuron_metrics_list.append(metrics)
    
    # Create figure: top row = rasters, bottom rows = metric histograms + aggregate rates
    # 4 histogram metrics + 1 aggregate rate plot, so 6 rows total (1 raster + 4 histograms + 1 aggregate)
    fig = plt.figure(figsize=(2*n_runs, 14))
    gs = fig.add_gridspec(6, n_runs, hspace=0.4, wspace=0.3)
    
    # Plot raster plots in top row
    for col, result in enumerate(selected_results):
        ax_raster = fig.add_subplot(gs[0, col])
        
        spike_times_all = result['spike_times_all']
        spike_neurons_all = result['spike_neurons_all']
        
        if spike_times_all is not None and len(spike_times_all) > 0:
            # Color by neuron type: Red for E (excitatory), Blue for I (inhibitory)
            # E neurons: neuron_id < 100 (0-99), I neurons: neuron_id >= 100 (100-199)
            colors = np.where(spike_neurons_all < 100, 'red', 'blue')
            ax_raster.scatter(spike_times_all, spike_neurons_all, c=colors, s=0.3, alpha=0.6, rasterized=True)
            ax_raster.axhline(y=99.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax_raster.set_xlim(0, 1000)
        ax_raster.set_ylim(-0.5, 199.5)
        ax_raster.set_title(f"Run {result.get('run_idx', col)}", fontsize=8, pad=2)
        ax_raster.set_xticks([])
        ax_raster.set_yticks([])
        ax_raster.spines['top'].set_visible(False)
        ax_raster.spines['right'].set_visible(False)
        ax_raster.spines['bottom'].set_visible(False)
        ax_raster.spines['left'].set_visible(False)
    
    # Plot histograms for each metric
    metric_configs = [
        ('firing_rates', 'Firing Rate (Hz)', 'steelblue', 'Neurons', 'Count'),
        ('cv_isis', 'CV of ISIs', 'coral', 'CV', 'Count'),
        ('fano_factors', 'Fano Factor\n(variance/mean)', 'mediumseagreen', 'Fano Factor', 'Count'),
        ('mean_correlations', 'Mean Correlation\n(smoothed z-scored)', 'purple', 'Correlation', 'Count')
    ]
    
    for row_idx, (metric_key, metric_label, color, xlabel, ylabel) in enumerate(metric_configs):
        # Find global range for consistent x-axis across runs
        all_values = []
        for metrics in per_neuron_metrics_list:
            values = metrics[metric_key]
            # For correlations, include all values (can be negative)
            if metric_key == 'mean_correlations':
                all_values.extend(values)
            else:
                all_values.extend(values[values > 0])  # Only non-zero values
        
        if len(all_values) > 0:
            if metric_key == 'mean_correlations':
                x_min = np.percentile(all_values, 5)  # Use 5th percentile for correlations
                x_max = np.percentile(all_values, 95)
            else:
                x_min = 0
                x_max = np.percentile(all_values, 95)  # Use 95th percentile to avoid outliers
            if x_max == 0:
                x_max = max(all_values) if len(all_values) > 0 else 1.0
        else:
            x_min, x_max = 0, 1.0
        
        for col, metrics in enumerate(per_neuron_metrics_list):
            ax = fig.add_subplot(gs[row_idx + 1, col])
            
            values = metrics[metric_key]
            # For correlations, include all values; for others, filter zeros
            if metric_key == 'mean_correlations':
                values_to_plot = values
            else:
                values_to_plot = values[values > 0]
            
            if len(values_to_plot) > 0:
                n, bins_edges, patches = ax.hist(values_to_plot, bins=20, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax.set_xlim(x_min, x_max)
                
                # Add axis labels
                if col == n_runs // 2:  # X-axis label on middle column
                    ax.set_xlabel(xlabel, fontsize=7)
                if col == 0:  # Y-axis label on first column
                    ax.set_ylabel(f'{metric_label}\n{ylabel}', fontsize=7, rotation=0, ha='right', va='center')
                    ax.yaxis.set_label_coords(-0.15, 0.5)
                
                # Add tick marks
                ax.tick_params(labelsize=6)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=7)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # Plot aggregate firing rates (last row)
    for col, metrics in enumerate(per_neuron_metrics_list):
        ax = fig.add_subplot(gs[5, col])
        
        agg_rates = metrics['aggregate_rates']
        time = agg_rates['time']
        e_rate = agg_rates['e_rate']
        i_rate = agg_rates['i_rate']
        combined_rate = agg_rates['combined_rate']
        
        # Plot E, I, and combined
        ax.plot(time, e_rate, color='red', linewidth=1.5, alpha=0.7, label='E')
        ax.plot(time, i_rate, color='blue', linewidth=1.5, alpha=0.7, label='I')
        ax.plot(time, combined_rate, color='black', linewidth=1.0, alpha=0.5, linestyle='--', label='Combined')
        
        ax.set_xlim(0, 1000)
        ax.set_ylim(bottom=0)
        
        # Add axis labels
        if col == n_runs // 2:  # X-axis label on middle column
            ax.set_xlabel('Time (ms)', fontsize=7)
        if col == 0:  # Y-axis label on first column
            ax.set_ylabel('Aggregate\nFiring Rate (Hz)', fontsize=7, rotation=0, ha='right', va='center')
            ax.yaxis.set_label_coords(-0.15, 0.5)
        
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend only on first column
        if col == 0:
            ax.legend(loc='upper right', fontsize=6, frameon=False)
    
    fig.suptitle('Random Search: Selected Runs with Per-Neuron Metrics', fontsize=12, fontweight='bold', y=0.98)
    
    save_path = os.path.join(save_folder, "metrics_figure_10runs.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Find latest results
    summary_file, results_folder = find_latest_random_search_results()
    figures_folder = os.path.join(results_folder, "Figures")
    
    print(f"Loading results from: {summary_file}")
    
    # Load summary
    with open(summary_file, 'rb') as f:
        all_results_summary = pickle.load(f)
    
    # Load full spike data for selected runs
    print("Loading full spike data...")
    all_results_with_spikes = []
    for summary in all_results_summary:
        if summary.get('success', False):
            pickle_file = summary.get('pickle_file')
            if pickle_file and os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as f:
                    full_result = pickle.load(f)
                    all_results_with_spikes.append(full_result)
    
    print(f"Loaded {len(all_results_with_spikes)} runs with spike data")
    
    # Create metrics figure
    create_metrics_figure(all_results_with_spikes, figures_folder, n_runs=10)

