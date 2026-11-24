"""
Plot all rasters from grid search where mean firing rate is between min and max threshold.
All rasters on a single figure.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

def find_latest_grid_search_results():
    """Find the most recent grid search run folder (checks pickles, not summary)"""
    base_folder = Path(__file__).parent / "Runs" / "no_feedforward"
    
    # Find all grid search folders
    grid_folders = [f for f in base_folder.glob("grid_search_EtoI_ItoE_E*_I*") if f.is_dir()]
    if not grid_folders:
        raise FileNotFoundError(f"No grid search folders found in {base_folder}")
    
    # Collect all timestamped run folders from all grid searches
    all_run_folders = []
    for grid_folder in grid_folders:
        for item in grid_folder.iterdir():
            if item.is_dir() and item.name != "old":
                # Check if it has a Pickles folder with some data
                pickles_folder = item / "Pickles"
                if pickles_folder.exists() and any(pickles_folder.glob("run_*.pkl")):
                    all_run_folders.append((item.name, item))
            elif item.is_dir() and item.name == "old":
                # Also check old folder
                for old_item in item.iterdir():
                    if old_item.is_dir():
                        pickles_folder = old_item / "Pickles"
                        if pickles_folder.exists() and any(pickles_folder.glob("run_*.pkl")):
                            all_run_folders.append((old_item.name, old_item))
    
    if not all_run_folders:
        raise FileNotFoundError(f"No run folders with pickle files found")
    
    # Sort by timestamp (folder name) and get the latest
    all_run_folders.sort(key=lambda x: x[0], reverse=True)
    latest_folder_name, latest_folder = all_run_folders[0]
    
    print(f"Using latest run folder: {latest_folder}")
    return latest_folder

def load_individual_runs(folder_path):
    """Load all individual run pickle files"""
    pickles_folder = folder_path / "Pickles"
    all_runs = []
    
    if not pickles_folder.exists():
        print(f"Pickles folder not found: {pickles_folder}")
        return all_runs
    
    # Load all pickle files
    pickle_files = sorted(pickles_folder.glob("run_*.pkl"))
    print(f"Loading {len(pickle_files)} run files...")
    
    for pkl_file in pickle_files:
        with open(pkl_file, 'rb') as f:
            run_data = pickle.load(f)
            
            # Extract parameters from run_data
            params = {
                'e_to_e_scale': run_data.get('e_to_e_scale'),
                'e_to_i_scale': run_data.get('e_to_i_scale'),
                'i_to_e_scale': run_data.get('i_to_e_scale'),
                'i_to_i_scale': run_data.get('i_to_i_scale')
            }
            all_runs.append((run_data, params))
    
    return all_runs

def check_firing_rate_bounds(run_data, min_fr=0.5, max_fr=70.0):
    """Check if BOTH E and I firing rates are within bounds"""
    e_rate = run_data.get('e_rate', 0)
    i_rate = run_data.get('i_rate', 0)
    return (min_fr <= e_rate <= max_fr) and (min_fr <= i_rate <= max_fr)

def plot_filtered_rasters_single_figure(all_runs, figures_dir, min_fr=0.5, max_fr=70.0):
    """
    Plot all rasters where BOTH E and I firing rates are between min_fr and max_fr Hz.
    All on a single figure.
    """
    print(f"\n{'='*60}")
    print(f"Creating figure with rasters where BOTH E and I: {min_fr} Hz <= FR <= {max_fr} Hz")
    print(f"{'='*60}")
    
    # Filter runs by firing rate
    filtered_runs = []
    for run_data, params in all_runs:
        if not run_data.get('success', False):
            continue
        
        if check_firing_rate_bounds(run_data, min_fr, max_fr):
            # Check if spike data exists
            spike_times = run_data.get('spike_times_all')
            spike_neurons = run_data.get('spike_neurons_all')
            if spike_times is not None and spike_neurons is not None and len(spike_times) > 0:
                e_rate = run_data.get('e_rate', 0)
                i_rate = run_data.get('i_rate', 0)
                mean_fr = (e_rate + i_rate) / 2.0
                filtered_runs.append((run_data, params, mean_fr))
    
    n_runs = len(filtered_runs)
    print(f"  Found {n_runs} runs where BOTH E and I FR in [{min_fr}, {max_fr}] Hz")
    
    if n_runs == 0:
        print("  No runs match criteria. Skipping plot.")
        return
    
    # Sort by mean firing rate
    filtered_runs.sort(key=lambda x: x[2])
    
    # Determine grid size (try to make it roughly square)
    n_cols = int(np.ceil(np.sqrt(n_runs)))
    n_rows = int(np.ceil(n_runs / n_cols))
    
    print(f"  Creating {n_rows}x{n_cols} grid ({n_runs} rasters)")
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Grid Search Rasters: BOTH E&I {min_fr} Hz ≤ FR ≤ {max_fr} Hz (n={n_runs})', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Assume network size from first run
    first_run = filtered_runs[0][0]
    # Try to infer network size from spike data or use defaults
    spike_neurons = first_run.get('spike_neurons_all')
    if spike_neurons is not None and len(spike_neurons) > 0:
        N_total = int(np.max(spike_neurons)) + 1
        N_E = N_total // 2  # Assume 50/50 split
    else:
        N_E = 200
        N_total = 400
    
    # Plot each raster
    for idx, (run_data, params, mean_fr) in enumerate(filtered_runs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        spike_times = run_data['spike_times_all']
        spike_neurons = run_data['spike_neurons_all']
        e_rate = run_data.get('e_rate', 0)
        i_rate = run_data.get('i_rate', 0)
        
        # Plot raster - flip Y axis so E is on bottom (red), I on top (blue)
        e_mask = spike_neurons < N_E
        i_mask = spike_neurons >= N_E
        
        # Flip neuron indices so E (0-99) appears at bottom
        flipped_neurons = N_total - 1 - spike_neurons
        
        ax.scatter(spike_times[e_mask], flipped_neurons[e_mask], 
                  c='red', s=0.3, alpha=0.7, rasterized=True)
        ax.scatter(spike_times[i_mask], flipped_neurons[i_mask], 
                  c='blue', s=0.3, alpha=0.7, rasterized=True)
        
        # Add horizontal line separating E and I (after flipping)
        ax.axhline(y=N_total - N_E - 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Format axes
        ax.set_xlim(0, 1000)  # Assuming 1000ms simulation
        ax.set_ylim(0, N_total)
        
        # Title with parameters and firing rates
        ee = params['e_to_e_scale']
        ei = params['e_to_i_scale']
        ie = params['i_to_e_scale']
        ii = params['i_to_i_scale']
        
        title = f'EE:{ee:.3f} EI:{ei:.3f}\nIE:{ie:.3f} II:{ii:.3f}\nE:{e_rate:.1f} I:{i_rate:.1f} Hz'
        ax.set_title(title, fontsize=6, pad=2)
        
        # Only show axis labels on outer edges
        if row == n_rows - 1:
            ax.set_xlabel('Time (ms)', fontsize=6)
        else:
            ax.set_xticks([])
        
        if col == 0:
            ax.set_ylabel('Neuron', fontsize=6)
        else:
            ax.set_yticks([])
        
        ax.tick_params(labelsize=5)
    
    # Turn off extra subplots
    for idx in range(n_runs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    save_path = figures_dir / f"all_rasters_both_EI_fr_{min_fr}hz_to_{max_fr}hz.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path.name}")
    
    # Show the figure
    plt.show()
    plt.close(fig)

def main():
    """Main function"""
    folder_path = find_latest_grid_search_results()
    print(f"Loading results from: {folder_path}")
    
    # Load all individual runs
    all_runs = load_individual_runs(folder_path)
    print(f"Loaded {len(all_runs)} individual runs")
    
    figures_dir = folder_path / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figure with rasters in specified firing rate range
    plot_filtered_rasters_single_figure(all_runs, figures_dir, min_fr=0.5, max_fr=70.0)
    
    print(f"\n{'='*60}")
    print("Plot generated successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

