"""
General parameter grid visualization tool.
Allows plotting individual run results in a grid with configurable which parameters vary vs are fixed.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from constants import (
    GRID_SEARCH_BASE_DIR,
    GRID_SEARCH_SUMMARY_FILENAME,
    FIGURES_SUBDIR_NAME,
)

# Constants matching recurrent/parameter_search/grid_search.py
N_REC_E = 200
N_REC_I = 200
BASE_WEIGHT_SCALE = 15.0
WEIGHT_SCALE_METHOD = 'sqrt_k'


def find_latest_grid_search_results():
    """Find the most recent grid_search_summary.pkl file"""
    if not GRID_SEARCH_BASE_DIR.exists():
        raise FileNotFoundError(f"Grid search base directory not found: {GRID_SEARCH_BASE_DIR}")
    
    timestamp_dirs = []
    for child in GRID_SEARCH_BASE_DIR.iterdir():
        if child.is_dir() and child.name != "old":
            summary_path = child / GRID_SEARCH_SUMMARY_FILENAME
            if summary_path.exists():
                timestamp_dirs.append((child.name, summary_path, child))
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"No {GRID_SEARCH_SUMMARY_FILENAME} files found in {GRID_SEARCH_BASE_DIR}")
    
    timestamp_dirs.sort(key=lambda x: x[0], reverse=True)
    _, summary_path, folder_path = timestamp_dirs[0]
    return summary_path, folder_path


def load_individual_runs(folder_path, param_filters=None):
    """
    Load individual run pickle files matching the specified parameter filters.
    
    Args:
        folder_path: Path to the run folder
        param_filters: Dict of {param_name: value} to filter runs. 
                      Use None to match any value.
    
    Returns:
        List of (run_data, param_dict) tuples
    """
    pickles_folder = folder_path / "Pickles"
    if not pickles_folder.exists():
        raise FileNotFoundError(f"Pickles folder not found: {pickles_folder}")
    
    runs = []
    for filename in sorted(pickles_folder.glob("run_*.pkl")):
        with open(filename, 'rb') as f:
            run_data = pickle.load(f)
        
        # Extract parameters from run data
        params = {
            'e_to_e_scale': run_data.get('e_to_e_scale'),
            'e_to_i_scale': run_data.get('e_to_i_scale'),
            'i_to_e_scale': run_data.get('i_to_e_scale'),
            'i_to_i_scale': run_data.get('i_to_i_scale'),
        }
        
        # Filter by parameters if specified
        if param_filters:
            match = True
            for param_name, filter_value in param_filters.items():
                if filter_value is not None:
                    param_value = params.get(param_name)
                    if param_value is None or not np.isclose(param_value, filter_value, rtol=1e-6):
                        match = False
                        break
            if not match:
                continue
        
        runs.append((run_data, params))
    
    return runs


def plot_raster_grid(runs, param_to_vary, fixed_params, save_path, 
                     plot_type='raster', figsize_per_subplot=(6, 4)):
    """
    Create a grid of plots showing individual run results.
    
    Args:
        runs: List of (run_data, param_dict) tuples
        param_to_vary: Dict of {param_name: list_of_values} - parameters to vary across grid
        fixed_params: Dict of {param_name: value} - parameters to keep fixed
        save_path: Where to save the figure
        plot_type: 'raster', 'rate', or 'both'
        figsize_per_subplot: Size of each subplot
    """
    # Filter runs to match fixed parameters
    filtered_runs = []
    for run_data, params in runs:
        match = True
        for param_name, fixed_value in fixed_params.items():
            if param_name not in params:
                match = False
                break
            if not np.isclose(params[param_name], fixed_value, rtol=1e-6):
                match = False
                break
        if match:
            filtered_runs.append((run_data, params))
    
    print(f"  Filtered to {len(filtered_runs)} runs matching fixed parameters: {fixed_params}")
    
    if not filtered_runs:
        print(f"  ERROR: No runs found matching fixed parameters: {fixed_params}")
        return
    
    # Determine grid dimensions from param_to_vary
    param_names = list(param_to_vary.keys())
    
    # Use the specified values (they should match what's in the data)
    unique_values = {name: param_to_vary[name] for name in param_names}
    
    print(f"  Varying parameters: {param_names}")
    for name, values in unique_values.items():
        print(f"    {name}: {len(values)} values {values}")
    
    # Determine grid layout
    if len(param_names) == 1:
        n_rows, n_cols = 1, len(unique_values[param_names[0]])
        row_param, col_param = None, param_names[0]
    elif len(param_names) == 2:
        n_rows = len(unique_values[param_names[0]])
        n_cols = len(unique_values[param_names[1]])
        row_param, col_param = param_names[0], param_names[1]
    else:
        raise ValueError("Can only vary 1 or 2 parameters at a time")
    
    # Create figure with 3 rows per subplot (weight matrix, raster, firing rates)
    fig = plt.figure(figsize=(figsize_per_subplot[0] * n_cols, 
                              (figsize_per_subplot[1] * 2 + 1.5) * n_rows))
    
    # Create grid with 3 rows per subplot
    gs = GridSpec(n_rows * 3, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # Create axes arrays
    axes_weight = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    axes_plot = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    axes_rates = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    
    for row in range(n_rows):
        for col in range(n_cols):
            # Weight matrix axis (top row)
            axes_weight[row][col] = fig.add_subplot(gs[row * 3, col])
            # Plot axis (middle row)
            axes_plot[row][col] = fig.add_subplot(gs[row * 3 + 1, col])
            # Firing rates axis (bottom row)
            axes_rates[row][col] = fig.add_subplot(gs[row * 3 + 2, col])
    
    # Create mapping from parameter values to run data
    run_map = {}
    for run_data, params in filtered_runs:
        if len(param_names) == 1:
            key = (params[param_names[0]],)
        else:
            key = (params[param_names[0]], params[param_names[1]])
        run_map[key] = (run_data, params)
    
    print(f"  Created grid: {n_rows}x{n_cols} = {n_rows*n_cols} subplots")
    print(f"  Found {len(run_map)}/{n_rows*n_cols} matching runs")
    
    # First pass: regenerate all weight matrices and calculate global min/max
    all_weight_matrices = {}
    all_weights_flat = []
    for key, (run_data, params) in run_map.items():
        weight_matrix = regenerate_weight_matrix(params)
        all_weight_matrices[key] = weight_matrix
        all_weights_flat.extend(weight_matrix.flatten())
    
    all_weights_flat = np.array(all_weights_flat)
    global_vmin = np.percentile(all_weights_flat, 1)
    global_vmax = np.percentile(all_weights_flat, 99)
    
    # Determine if log scaling is needed
    weight_range = global_vmax - global_vmin
    use_log = weight_range > 0 and (abs(global_vmax) / max(abs(global_vmin), 1e-10) > 10 or 
                                    abs(global_vmin) / max(abs(global_vmax), 1e-10) > 10)
    
    if use_log:
        abs_max = max(abs(global_vmin), abs(global_vmax))
        global_vmin_log = -np.log10(abs_max + 1)
        global_vmax_log = np.log10(abs_max + 1)
        print(f"  Using log color scale: vmin={global_vmin:.4f}, vmax={global_vmax:.4f} (raw)")
    else:
        print(f"  Using linear color scale: vmin={global_vmin:.4f}, vmax={global_vmax:.4f}")
    
    # Plot each subplot
    plotted_count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            ax_weight = axes_weight[row][col]
            ax_plot = axes_plot[row][col]
            
            if row_param:
                row_val = unique_values[row_param][row]
            else:
                row_val = None
            
            col_val = unique_values[col_param][col]
            
            # Find matching run
            if row_param:
                key = (row_val, col_val)
            else:
                key = (col_val,)
            
            if key in run_map:
                run_data, params = run_map[key]
                plotted_count += 1
                
                # Plot weight matrix on top with consistent global color scale
                weight_matrix = all_weight_matrices[key]
                if use_log:
                    plot_weight_matrix_small(ax_weight, weight_matrix, params, 
                                            vmin=global_vmin_log, vmax=global_vmax_log, use_log=True)
                else:
                    plot_weight_matrix_small(ax_weight, weight_matrix, params, 
                                            vmin=global_vmin, vmax=global_vmax, use_log=False)
                
                # Plot raster in middle
                plot_raster(ax_plot, run_data, params)
                
                # Plot firing rates at bottom
                ax_rates = axes_rates[row][col]
                plot_firing_rates(ax_rates, run_data, params)
                
                # Add title with parameter values on the plot axis
                def format_sci(val):
                    """Format value in scientific notation when appropriate"""
                    if abs(val) < 0.01 or abs(val) >= 100:
                        return f"{val:.2e}"
                    else:
                        return f"{val:.3f}"
                
                title_parts = []
                if row_param:
                    title_parts.append(f"{row_param}={format_sci(row_val)}")
                title_parts.append(f"{col_param}={format_sci(col_val)}")
                # Don't include fixed params in subplot titles to reduce clutter
                ax_plot.set_title(" | ".join(title_parts), fontsize=7, pad=3)
            else:
                ax_weight.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax_weight.transAxes, fontsize=8)
                ax_weight.axis('off')
                ax_plot.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax_plot.transAxes, fontsize=8)
                ax_plot.axis('off')
                ax_rates = axes_rates[row][col]
                ax_rates.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax_rates.transAxes, fontsize=8)
                ax_rates.axis('off')
    
    print(f"  Plotted {plotted_count} runs")
    
    # Add overall title
    def format_sci(val):
        """Format value in scientific notation when appropriate"""
        if abs(val) < 0.01 or abs(val) >= 100:
            return f"{val:.2e}"
        else:
            return f"{val:.3f}"
    
    title = f"Varying: {', '.join(param_names)}"
    if fixed_params:
        fixed_str = ', '.join(f'{k}={format_sci(v)}' for k, v in fixed_params.items())
        title += f" | Fixed: {fixed_str}"
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.98)
    
    # Adjust spacing to prevent title overlap
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3, bottom=0.05)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved grid plot to: {save_path}")


def regenerate_weight_matrix(params):
    """Regenerate weight matrix from connection scale parameters"""
    N_rec = N_REC_E + N_REC_I
    k_recurrent = N_rec
    
    if WEIGHT_SCALE_METHOD == 'k':
        r_w_scale_eff = BASE_WEIGHT_SCALE / k_recurrent
    elif WEIGHT_SCALE_METHOD == 'sqrt_k':
        r_w_scale_eff = BASE_WEIGHT_SCALE / np.sqrt(k_recurrent)
    else:
        raise ValueError(f"Unknown weight_scale_method: {WEIGHT_SCALE_METHOD}")
    
    # Generate lognormal weights (using same seed for reproducibility)
    np.random.seed(42)
    lognormal_weights = np.random.lognormal(mean=0.0, sigma=0.5, size=(N_rec, N_rec))
    weight_rec = r_w_scale_eff * lognormal_weights
    
    # Apply Dale's law
    weight_rec[:, :N_REC_E] = np.abs(weight_rec[:, :N_REC_E])  # E sources: positive
    weight_rec[:, N_REC_E:] = -np.abs(weight_rec[:, N_REC_E:])  # I sources: negative
    
    # Apply weight scaling: weight_rec is [target, source]
    weight_rec[:N_REC_E, :N_REC_E] *= params.get('e_to_e_scale', 1.0)  # E→E
    weight_rec[N_REC_E:, :N_REC_E] *= params.get('e_to_i_scale', 1.0)  # E→I
    weight_rec[:N_REC_E, N_REC_E:] *= params.get('i_to_e_scale', 1.0)  # I→E
    weight_rec[N_REC_E:, N_REC_E:] *= params.get('i_to_i_scale', 1.0)  # I→I
    
    return weight_rec


def plot_weight_matrix_small(ax, weight_matrix, params, vmin=None, vmax=None, use_log=False):
    """Plot weight matrix with consistent global color scale and quadrant labels"""
    from matplotlib.ticker import ScalarFormatter
    
    # Downsample for visualization if too large
    max_size = 100
    if weight_matrix.shape[0] > max_size:
        step = weight_matrix.shape[0] // max_size
        weight_vis = weight_matrix[::step, ::step]
        n_e_vis = N_REC_E // step
    else:
        weight_vis = weight_matrix
        n_e_vis = N_REC_E
    
    # Apply log scaling if requested
    if use_log and vmin is not None and vmax is not None:
        # Use signed log: sign(x) * log10(abs(x) + 1)
        weight_vis = np.sign(weight_vis) * np.log10(np.abs(weight_vis) + 1)
        # vmin/vmax are already in log space if provided
    
    # Plot with consistent colormap
    im = ax.imshow(weight_vis, cmap='RdBu_r', aspect='auto', 
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Add grid lines to separate E and I regions
    ax.axhline(y=n_e_vis - 0.5, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(x=n_e_vis - 0.5, color='black', linewidth=0.5, linestyle='--')
    
    # Add quadrant labels with parameter values
    def format_sci(val):
        """Format value in scientific notation when appropriate"""
        if abs(val) < 0.01 or abs(val) >= 100:
            return f"{val:.2e}"
        else:
            return f"{val:.3f}"
    
    # Labels positioned in each quadrant center (weight_matrix is [target, source])
    # Top-left: E targets from E sources = E→E
    # Top-right: E targets from I sources = I→E
    # Bottom-left: I targets from E sources = E→I
    # Bottom-right: I targets from I sources = I→I
    label_fontsize = 5
    total_h, total_w = weight_vis.shape
    
    # Top-left: E→E
    ax.text(n_e_vis/2, n_e_vis/2, f'E→E\n{format_sci(params.get("e_to_e_scale", 0))}', 
            ha='center', va='center', fontsize=label_fontsize,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    # Top-right: I→E
    ax.text(n_e_vis + (total_w-n_e_vis)/2, n_e_vis/2, f'I→E\n{format_sci(params.get("i_to_e_scale", 0))}', 
            ha='center', va='center', fontsize=label_fontsize,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    # Bottom-left: E→I
    ax.text(n_e_vis/2, n_e_vis + (total_h-n_e_vis)/2, f'E→I\n{format_sci(params.get("e_to_i_scale", 0))}', 
            ha='center', va='center', fontsize=label_fontsize,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    # Bottom-right: I→I
    ax.text(n_e_vis + (total_w-n_e_vis)/2, n_e_vis + (total_h-n_e_vis)/2, 
            f'I→I\n{format_sci(params.get("i_to_i_scale", 0))}', 
            ha='center', va='center', fontsize=label_fontsize,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Weight Matrix', fontsize=7, pad=2)
    
    return im


def plot_raster(ax, run_data, params):
    """Plot raster plot for a single run with enhanced visualization"""
    spike_times = run_data.get('spike_times_all')
    spike_neurons = run_data.get('spike_neurons_all')
    
    if spike_times is None or spike_neurons is None or len(spike_times) == 0:
        ax.text(0.5, 0.5, "No spikes", ha="center", va="center", 
               transform=ax.transAxes, fontsize=8, color='gray')
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 400)
        ax.axis('off')
        return
    
    # Determine neuron types (assuming first 200 are E, rest are I)
    N_E = 200
    N_I = 200
    N_total = N_E + N_I
    
    # Calculate firing rates for E and I populations
    T_max = 1000.0  # ms
    e_spikes = spike_times[spike_neurons < N_E]
    i_spikes = spike_times[spike_neurons >= N_E]
    e_rate = len(e_spikes) / (N_E * T_max / 1000.0) if len(e_spikes) > 0 else 0.0
    i_rate = len(i_spikes) / (N_I * T_max / 1000.0) if len(i_spikes) > 0 else 0.0
    
    # Color by neuron type: Red for E, Blue for I
    colors = np.where(spike_neurons < N_E, 'red', 'blue')
    
    # Plot with slightly larger markers for better visibility
    ax.scatter(spike_times, spike_neurons, c=colors, s=0.8, alpha=0.7, edgecolors='none')
    
    # Add firing rate info in top-right corner
    info_text = f"E: {e_rate:.1f} Hz\nI: {i_rate:.1f} Hz"
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, 
           ha='right', va='top', fontsize=6,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.5, N_total - 0.5)
    ax.axhline(y=N_E - 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Remove axis labels for cleaner grid (they're in the outer labels)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_firing_rates(ax, run_data, params):
    """Plot firing rates over time with individual neurons and bold average"""
    spike_times = run_data.get('spike_times_all')
    spike_neurons = run_data.get('spike_neurons_all')
    
    if spike_times is None or spike_neurons is None or len(spike_times) == 0:
        ax.text(0.5, 0.5, "No spikes", ha="center", va="center", 
               transform=ax.transAxes)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 50)
        return
    
    # Bin spikes over time to compute firing rates
    bin_size_ms = 50  # 50 ms bins
    T_max = 1000  # 1 second simulation
    time_bins = np.arange(0, T_max + bin_size_ms, bin_size_ms)
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # Compute firing rates for each neuron over time
    e_rates_over_time = np.zeros((N_REC_E, len(time_centers)))
    i_rates_over_time = np.zeros((N_REC_I, len(time_centers)))
    
    # E neurons
    for neuron_id in range(N_REC_E):
        neuron_spikes = spike_times[spike_neurons == neuron_id]
        if len(neuron_spikes) > 0:
            counts, _ = np.histogram(neuron_spikes, bins=time_bins)
            e_rates_over_time[neuron_id, :] = counts / (bin_size_ms / 1000)  # Convert to Hz
    
    # I neurons
    for neuron_id in range(N_REC_I):
        actual_id = neuron_id + N_REC_E
        neuron_spikes = spike_times[spike_neurons == actual_id]
        if len(neuron_spikes) > 0:
            counts, _ = np.histogram(neuron_spikes, bins=time_bins)
            i_rates_over_time[neuron_id, :] = counts / (bin_size_ms / 1000)  # Convert to Hz
    
    # Plot individual neuron rates (light lines)
    for neuron_id in range(N_REC_E):
        ax.plot(time_centers, e_rates_over_time[neuron_id, :], 
               color='red', alpha=0.2, linewidth=0.5)
    
    for neuron_id in range(N_REC_I):
        ax.plot(time_centers, i_rates_over_time[neuron_id, :], 
               color='blue', alpha=0.2, linewidth=0.5)
    
    # Plot average (bold lines)
    e_avg = np.mean(e_rates_over_time, axis=0)
    i_avg = np.mean(i_rates_over_time, axis=0)
    
    ax.plot(time_centers, e_avg, color='red', linewidth=2.5, label='E avg', alpha=0.9)
    ax.plot(time_centers, i_avg, color='blue', linewidth=2.5, label='I avg', alpha=0.9)
    
    ax.set_xlabel('Time (ms)', fontsize=8)
    ax.set_ylabel('Firing rate (Hz)', fontsize=8)
    ax.set_xlim(0, T_max)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
    
    # Use scientific notation for y-axis ticks (firing rates)
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linestyle='--')


def generate_all_raster_grids(folder_path, all_runs, figures_dir):
    """
    Generate raster grids for all combinations of parameter pairs.
    Creates organized folder structure with 5x5 grids where each cell is a raster.
    Structure: raster_grids/row_param/col_param/figures
    """
    # Create main folder for raster grids
    raster_grids_base = figures_dir / "raster_grids"
    raster_grids_base.mkdir(parents=True, exist_ok=True)
    print(f"Saving raster grids to: {raster_grids_base}")
    
    param_names = ['e_to_e_scale', 'e_to_i_scale', 'i_to_e_scale', 'i_to_i_scale']
    GRID_SCALES = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    # Normalize values to grid scales (handle floating point issues)
    
    def normalize_to_grid(val):
        """Find closest grid value"""
        if val is None:
            return None
        closest = min(GRID_SCALES, key=lambda x: abs(x - val))
        if abs(closest - val) < 1e-6:
            return closest
        return val  # Return original if not close to any grid value
    
    # Get unique values for each parameter from actual data, normalized to grid
    unique_values = {}
    for param in param_names:
        values = sorted(set(normalize_to_grid(params.get(param)) 
                           for _, params in all_runs 
                           if params.get(param) is not None))
        # Filter out None values and ensure we have grid values
        values = [v for v in values if v is not None and v in GRID_SCALES]
        unique_values[param] = values if values else GRID_SCALES  # fallback to full grid
        print(f"  {param}: {len(unique_values[param])} unique values {unique_values[param]}")
    
    # Create a map for quick lookup: (row_val, col_val, other_params) -> run
    run_map = {}
    for run_data, params in all_runs:
        key = (
            normalize_to_grid(params.get('e_to_e_scale')),
            normalize_to_grid(params.get('e_to_i_scale')),
            normalize_to_grid(params.get('i_to_e_scale')),
            normalize_to_grid(params.get('i_to_i_scale'))
        )
        if all(v is not None for v in key):
            run_map[key] = (run_data, params)
    
    # Short parameter names for display
    param_short = {
        'e_to_e_scale': 'E→E',
        'e_to_i_scale': 'E→I',
        'i_to_e_scale': 'I→E',
        'i_to_i_scale': 'I→I'
    }
    
    # Generate all 16 combinations (4x4)
    total_figures = 0
    for row_param in param_names:
        for col_param in param_names:
            total_figures += 1
            
            # Create folder structure: raster_grids/row_param/col_param/
            row_folder = raster_grids_base / row_param
            col_folder = row_folder / col_param
            col_folder.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"Figure {total_figures}/16: Rows={row_param}, Cols={col_param}")
            print(f"  Saving to: {col_folder}")
            print(f"{'='*60}")
            
            # Get unique values for row and column parameters
            row_values = sorted(unique_values[row_param])
            col_values = sorted(unique_values[col_param])
            
            # Create simple 5x5 grid
            n_rows = len(row_values)
            n_cols = len(col_values)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            if n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            def calculate_total_firing_rate(run_data):
                """Calculate total firing rate across all neurons"""
                spike_times = run_data.get('spike_times_all')
                if spike_times is None or len(spike_times) == 0:
                    return 0.0
                T_max = 1000.0  # ms
                N_total = 400  # E + I
                return len(spike_times) / (N_total * T_max / 1000.0)
            
            plotted_count = 0
            for i, row_val in enumerate(row_values):
                for j, col_val in enumerate(col_values):
                    ax = axes[i, j]
                    
                    # Find ALL runs that match row_param=row_val and col_param=col_val
                    # (sweeping over the other two parameters)
                    matching_runs = []
                    for (ee, ei, ie, ii), (rd, p) in run_map.items():
                        # Check if this run matches the row and column parameters
                        row_match = False
                        col_match = False
                        
                        if row_param == 'e_to_e_scale' and np.isclose(ee, row_val, rtol=1e-6):
                            row_match = True
                        elif row_param == 'e_to_i_scale' and np.isclose(ei, row_val, rtol=1e-6):
                            row_match = True
                        elif row_param == 'i_to_e_scale' and np.isclose(ie, row_val, rtol=1e-6):
                            row_match = True
                        elif row_param == 'i_to_i_scale' and np.isclose(ii, row_val, rtol=1e-6):
                            row_match = True
                        
                        if col_param == 'e_to_e_scale' and np.isclose(ee, col_val, rtol=1e-6):
                            col_match = True
                        elif col_param == 'e_to_i_scale' and np.isclose(ei, col_val, rtol=1e-6):
                            col_match = True
                        elif col_param == 'i_to_e_scale' and np.isclose(ie, col_val, rtol=1e-6):
                            col_match = True
                        elif col_param == 'i_to_i_scale' and np.isclose(ii, col_val, rtol=1e-6):
                            col_match = True
                        
                        if row_match and col_match:
                            matching_runs.append((rd, p, calculate_total_firing_rate(rd)))
                    
                    # Pick the most interesting run (highest firing rate = most active)
                    if matching_runs:
                        matching_runs.sort(key=lambda x: x[2], reverse=True)
                        run_data, params, _ = matching_runs[0]
                        plot_raster(ax, run_data, params)
                        plotted_count += 1
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                               transform=ax.transAxes, fontsize=8)
                        ax.axis('off')
                    
                    # Add labels
                    def format_sci(val):
                        if abs(val) < 0.01 or abs(val) >= 100:
                            return f"{val:.2e}"
                        return f"{val:.3f}"
                    
                    if i == 0:  # Top row: column labels
                        ax.set_title(f"{param_short.get(col_param, col_param)}\n{format_sci(col_val)}", 
                                    fontsize=7, pad=3)
                    if j == 0:  # Left column: row labels
                        ax.set_ylabel(f"{param_short.get(row_param, row_param)}\n{format_sci(row_val)}", 
                                     fontsize=7, rotation=0, ha='right', va='center')
            
            # Add overall title
            fig.suptitle(f"Raster Grid: Rows={param_short.get(row_param, row_param)}, "
                        f"Cols={param_short.get(col_param, col_param)}\n"
                        f"(Showing best match from all combinations of other 2 parameters)",
                        fontsize=10, y=0.995)
            
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            
            # Save figure
            filename = f"raster_grid_{row_param}_vs_{col_param}.png"
            save_path = col_folder / filename
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filename} ({plotted_count}/{n_rows*n_cols} cells filled)")
            plt.close(fig)
    
    print(f"\n{'='*60}")
    print(f"Generated {total_figures} raster grid figures!")
    print(f"{'='*60}")


def plot_low_firing_rate_rasters(all_runs, figures_dir, max_fr=500.0):
    """
    Create a single figure with all rasters where mean firing rate < max_fr Hz.
    """
    print(f"\n{'='*60}")
    print(f"Creating figure with rasters where mean FR < {max_fr} Hz")
    print(f"{'='*60}")
    
    def calculate_mean_firing_rate(run_data):
        """Calculate mean firing rate across all neurons"""
        spike_times = run_data.get('spike_times_all')
        if spike_times is None or len(spike_times) == 0:
            return 0.0
        T_max = 1000.0  # ms
        N_total = 400  # E + I
        return len(spike_times) / (N_total * T_max / 1000.0)
    
    # Filter runs by firing rate
    filtered_runs = []
    for run_data, params in all_runs:
        mean_fr = calculate_mean_firing_rate(run_data)
        if mean_fr < max_fr:
            filtered_runs.append((run_data, params, mean_fr))
    
    print(f"  Found {len(filtered_runs)} runs with mean FR < {max_fr} Hz")
    
    if len(filtered_runs) == 0:
        print("  No runs found matching criteria!")
        return
    
    # Sort by firing rate for better organization
    filtered_runs.sort(key=lambda x: x[2])
    
    # Calculate grid size (aim for roughly square)
    n_runs = len(filtered_runs)
    n_cols = int(np.ceil(np.sqrt(n_runs)))
    n_rows = int(np.ceil(n_runs / n_cols))
    
    print(f"  Creating {n_rows}x{n_cols} grid ({n_runs} rasters)")
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.5, n_rows*1.2))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for idx, (run_data, params, mean_fr) in enumerate(filtered_runs):
        ax = axes_flat[idx]
        plot_raster(ax, run_data, params)
        
        # Add small label with parameters and FR
        def format_sci(val):
            if abs(val) < 0.01 or abs(val) >= 100:
                return f"{val:.2e}"
            return f"{val:.3f}"
        
        param_str = f"E→E:{format_sci(params.get('e_to_e_scale', 0))} E→I:{format_sci(params.get('e_to_i_scale', 0))}\n"
        param_str += f"I→E:{format_sci(params.get('i_to_e_scale', 0))} I→I:{format_sci(params.get('i_to_i_scale', 0))}\n"
        param_str += f"FR:{mean_fr:.1f}Hz"
        
        ax.text(0.02, 0.02, param_str, transform=ax.transAxes, 
               fontsize=4, ha='left', va='bottom',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7, edgecolor='black', linewidth=0.5))
    
    # Hide unused subplots
    for idx in range(len(filtered_runs), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    fig.suptitle(f"All Rasters with Mean Firing Rate < {max_fr} Hz\n"
                 f"Total: {len(filtered_runs)} runs",
                 fontsize=12, y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    raster_grids_dir = figures_dir / "raster_grids"
    raster_grids_dir.mkdir(parents=True, exist_ok=True)
    save_path = raster_grids_dir / f"all_rasters_fr_lt_{int(max_fr)}hz.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path.name}")
    plt.close(fig)


def main():
    """Example usage: plot grid varying I→I with other parameters fixed"""
    summary_path, folder_path = find_latest_grid_search_results()
    print(f"Loading results from: {folder_path}")
    
    # Load all individual runs
    all_runs = load_individual_runs(folder_path)
    print(f"Loaded {len(all_runs)} individual runs")
    
    figures_dir = folder_path / FIGURES_SUBDIR_NAME
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all raster grids
    generate_all_raster_grids(folder_path, all_runs, figures_dir)
    
    # Generate figure with low firing rate rasters
    plot_low_firing_rate_rasters(all_runs, figures_dir, max_fr=500.0)
    
    print(f"\n{'='*60}")
    print("All plots generated successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

