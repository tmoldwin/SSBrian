"""
Plot grid search results from saved pickle files.
This script loads saved simulation data and generates plots without re-running simulations.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def find_latest_grid_search_results():
    """Find the most recent grid_search_summary.pkl file"""
    base_folder = os.path.join(os.path.dirname(__file__), "Runs", "no_feedforward")
    grid_folder_name = "grid_search_EtoI_ItoE_E200_I200"
    grid_base_folder = os.path.join(base_folder, grid_folder_name)
    
    if not os.path.exists(grid_base_folder):
        raise FileNotFoundError(f"Grid search folder not found: {grid_base_folder}")
    
    # Find all timestamped folders
    timestamp_folders = []
    for item in os.listdir(grid_base_folder):
        item_path = os.path.join(grid_base_folder, item)
        if os.path.isdir(item_path) and item != "old":
            results_file = os.path.join(item_path, "grid_search_summary.pkl")
            if os.path.exists(results_file):
                timestamp_folders.append((item, results_file))
    
    if not timestamp_folders:
        raise FileNotFoundError(f"No grid_search_summary.pkl files found in {grid_base_folder}")
    
    # Sort by timestamp (folder name) and get the latest
    timestamp_folders.sort(key=lambda x: x[0], reverse=True)
    latest_folder, latest_file = timestamp_folders[0]
    
    return latest_file, os.path.join(grid_base_folder, latest_folder)

def create_comprehensive_heatmap_grid(all_results, save_folder):
    """Create a 5x5 subplot figure where each subplot is a 5x5 heatmap showing different metrics.
    Each subplot shows one metric, with E→I and I→E as axes, and E→E and I→I fixed at specific values."""
    print("\nGenerating comprehensive 5x5 heatmap grid...")
    
    # Filter successful results
    successful_results = [r for r in all_results if r['success']]
    
    if not successful_results:
        print("  No successful runs to plot")
        return
    
    # Extract all 4 dimensions
    e_to_e_scales = [r['e_to_e_scale'] for r in successful_results]
    e_to_i_scales = [r['e_to_i_scale'] for r in successful_results]
    i_to_e_scales = [r['i_to_e_scale'] for r in successful_results]
    i_to_i_scales = [r['i_to_i_scale'] for r in successful_results]
    
    # Get unique values for each dimension
    unique_e_to_e = sorted(set(e_to_e_scales))
    unique_e_to_i = sorted(set(e_to_i_scales))
    unique_i_to_e = sorted(set(i_to_e_scales))
    unique_i_to_i = sorted(set(i_to_i_scales))
    
    print(f"  Unique E->E scales: {unique_e_to_e}")
    print(f"  Unique E->I scales: {unique_e_to_i}")
    print(f"  Unique I->E scales: {unique_i_to_e}")
    print(f"  Unique I->I scales: {unique_i_to_i}")
    
    def create_2d_slice(metric_data, fixed_e_to_e, fixed_i_to_i):
        """Create a 2D slice (E→I vs I→E) with E→E and I→I fixed"""
        grid = np.full((len(unique_i_to_e), len(unique_e_to_i)), np.nan)
        for data, eee, eei, iee, iii in zip(metric_data, e_to_e_scales, e_to_i_scales, i_to_e_scales, i_to_i_scales):
            # Only include data where E→E and I→I match the fixed values
            if abs(eee - fixed_e_to_e) < 1e-6 and abs(iii - fixed_i_to_i) < 1e-6:
                i_idx = unique_i_to_e.index(iee)
                e_idx = unique_e_to_i.index(eei)
                grid[i_idx, e_idx] = data
        return grid
    
    # Prepare all metrics
    metrics = {
        'E Rate (Hz)': ([r['e_rate'] for r in successful_results], 'Reds', None),
        'I Rate (Hz)': ([r['i_rate'] for r in successful_results], 'Blues', None),
        'E-I Rate Corr': ([r['rate_correlation'] for r in successful_results], 'RdBu_r', (-1, 1)),
        'Pairwise Corr E': ([r['mean_pairwise_corr_e'] for r in successful_results], 'Reds', None),
        'Pairwise Corr I': ([r['mean_pairwise_corr_i'] for r in successful_results], 'Blues', None),
        'CV_E': ([r['cv_e'] for r in successful_results], 'viridis', None),
        'CV_I': ([r['cv_i'] for r in successful_results], 'viridis', None),
        'E/I Rate Ratio': ([r['e_rate']/r['i_rate'] if r['i_rate'] > 0 else np.nan for r in successful_results], 'viridis', None),
        'Balance Ratio': ([r['balance_ratio'] for r in successful_results], 'coolwarm', None),
        'Mean E Weight': ([r['mean_e_weight'] for r in successful_results], 'Reds', None),
        'Mean I Weight': ([abs(r['mean_i_weight']) for r in successful_results], 'Blues', None),
        'Rate Diff (E-I)': ([r['e_rate'] - r['i_rate'] for r in successful_results], 'RdBu_r', None),
        'Rate Sum (E+I)': ([r['e_rate'] + r['i_rate'] for r in successful_results], 'YlOrRd', None),
        'Corr Sum': ([r['mean_pairwise_corr_e'] + r['mean_pairwise_corr_i'] for r in successful_results], 'viridis', None),
        'CV Ratio': ([r['cv_e']/r['cv_i'] if r['cv_i'] > 0 else np.nan for r in successful_results], 'coolwarm', None),
        'Weight Ratio': ([abs(r['mean_e_weight']/r['mean_i_weight']) if r['mean_i_weight'] != 0 else np.nan for r in successful_results], 'viridis', None),
        'E Rate (log)': ([np.log10(r['e_rate'] + 0.1) for r in successful_results], 'Reds', None),
        'I Rate (log)': ([np.log10(r['i_rate'] + 0.1) for r in successful_results], 'Blues', None),
        'E-I Corr Abs': ([abs(r['rate_correlation']) for r in successful_results], 'Reds', None),
        'Pairwise Diff': ([r['mean_pairwise_corr_e'] - r['mean_pairwise_corr_i'] for r in successful_results], 'RdBu_r', None),
        'CV Sum': ([r['cv_e'] + r['cv_i'] for r in successful_results], 'viridis', None),
        'Total Spikes E': ([r['e_rate'] * 200 * 1.0 for r in successful_results], 'Reds', None),
        'Total Spikes I': ([r['i_rate'] * 200 * 1.0 for r in successful_results], 'Blues', None),
        'E Rate Std': ([r['e_rate'] for r in successful_results], 'Reds', None),
        'I Rate Std': ([r['i_rate'] for r in successful_results], 'Blues', None),
    }
    
    # Create 5x5 figure - each subplot shows one metric with different fixed E→E and I→I values
    fig, axes = plt.subplots(5, 5, figsize=(30, 30))
    fig.suptitle('4D Grid Search: Each subplot shows one metric (E→I vs I→E axes, E→E and I→I fixed)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    metric_items = list(metrics.items())
    for idx in range(25):  # 5x5 = 25 subplots
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        # Each subplot fixes E→E and I→I at different values
        fixed_e_to_e = unique_e_to_e[row]  # Row determines E→E
        fixed_i_to_i = unique_i_to_i[col]  # Column determines I→I
        
        # Choose metric (cycle through if we have fewer than 25)
        metric_idx = idx % len(metric_items)
        metric_name, (data_list, cmap, vlim) = metric_items[metric_idx]
        
        # Create 2D slice
        grid = create_2d_slice(data_list, fixed_e_to_e, fixed_i_to_i)
        
        if vlim is not None:
            im = ax.imshow(grid, aspect='auto', cmap=cmap, origin='lower', vmin=vlim[0], vmax=vlim[1])
        else:
            im = ax.imshow(grid, aspect='auto', cmap=cmap, origin='lower')
        
        # Title shows metric and fixed values
        ax.set_title(f'{metric_name}\nE→E={fixed_e_to_e:.3f}, I→I={fixed_i_to_i:.3f}', 
                    fontsize=9, fontweight='bold', pad=3)
        ax.set_xlabel('E→I Scale', fontsize=8)
        ax.set_ylabel('I→E Scale', fontsize=8)
        
        # Set ticks and labels
        ax.set_xticks(range(len(unique_e_to_i)))
        ax.set_xticklabels([f'{x:.3f}' for x in unique_e_to_i], rotation=45, fontsize=6)
        ax.set_yticks(range(len(unique_i_to_e)))
        ax.set_yticklabels([f'{x:.3f}' for x in unique_i_to_e], fontsize=6)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    figures_folder = os.path.join(save_folder, "Figures")
    os.makedirs(figures_folder, exist_ok=True)
    save_path = os.path.join(figures_folder, "comprehensive_5x5_heatmap_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comprehensive 5x5 heatmap grid saved to: {save_path}")
    plt.close()

def create_summary_plots(all_results, save_folder):
    """Create exploratory plots of grid search results with proper row/column headers"""
    print("\nGenerating summary plots...")
    
    # Filter successful results
    successful_results = [r for r in all_results if r['success']]
    
    if not successful_results:
        print("  No successful runs to plot")
        return
    
    # Get unique values for all 4 dimensions
    unique_e_to_e = sorted(set([r['e_to_e_scale'] for r in successful_results]))
    unique_e_to_i = sorted(set([r['e_to_i_scale'] for r in successful_results]))
    unique_i_to_e = sorted(set([r['i_to_e_scale'] for r in successful_results]))
    unique_i_to_i = sorted(set([r['i_to_i_scale'] for r in successful_results]))
    
    n_e_to_e = len(unique_e_to_e)
    n_i_to_i = len(unique_i_to_i)
    n_e_to_i = len(unique_e_to_i)
    n_i_to_e = len(unique_i_to_e)
    
    def create_2d_grid_for_fixed_ee_ii(data_list, e_to_e_list, i_to_i_list, e_to_i_list, i_to_e_list, 
                                       target_e_to_e, target_i_to_i):
        """Create a 2D grid (E-to-I vs I-to-E) for a fixed E-to-E and I-to-I combination"""
        grid = np.full((n_i_to_e, n_e_to_i), np.nan)
        for data, ee, ii, ei, ie in zip(data_list, e_to_e_list, i_to_i_list, e_to_i_list, i_to_e_list):
            if abs(ee - target_e_to_e) < 1e-6 and abs(ii - target_i_to_i) < 1e-6:
                ei_idx = unique_e_to_i.index(ei)
                ie_idx = unique_i_to_e.index(ie)
                grid[ie_idx, ei_idx] = data
        return grid
    
    # Extract all data
    e_to_e_scales = [r['e_to_e_scale'] for r in successful_results]
    e_to_i_scales = [r['e_to_i_scale'] for r in successful_results]
    i_to_e_scales = [r['i_to_e_scale'] for r in successful_results]
    i_to_i_scales = [r['i_to_i_scale'] for r in successful_results]
    e_rates = [r['e_rate'] for r in successful_results]
    i_rates = [r['i_rate'] for r in successful_results]
    rate_corrs = [r['rate_correlation'] for r in successful_results]
    pairwise_corrs_e = [r['mean_pairwise_corr_e'] for r in successful_results]
    pairwise_corrs_i = [r['mean_pairwise_corr_i'] for r in successful_results]
    
    # Create figures for each metric
    metrics = [
        ('E Firing Rate (Hz)', e_rates, 'Reds', None),
        ('I Firing Rate (Hz)', i_rates, 'Blues', None),
        ('E-I Rate Correlation', rate_corrs, 'RdBu_r', (-1, 1)),
        ('Pairwise Corr (E)', pairwise_corrs_e, 'Reds', None),
        ('Pairwise Corr (I)', pairwise_corrs_i, 'Blues', None),
    ]
    
    for metric_name, data_list, cmap, vlim in metrics:
        # Create figure with extra space for row/column headers and colorbar
        # Rows = E→E values, Columns = I→I values
        fig = plt.figure(figsize=(3*n_i_to_i + 2.0, 3*n_e_to_e + 1.0))
        gs = fig.add_gridspec(n_e_to_e + 1, n_i_to_i + 2, 
                              width_ratios=[0.3] + [1]*n_i_to_i + [0.15],
                              height_ratios=[0.3] + [1]*n_e_to_e,
                              hspace=0.3, wspace=0.3)
        
        # Find global min/max for consistent color scaling
        all_grids = []
        for ee in unique_e_to_e:
            for ii in unique_i_to_i:
                grid = create_2d_grid_for_fixed_ee_ii(data_list, e_to_e_scales, i_to_i_scales, 
                                                      e_to_i_scales, i_to_e_scales, ee, ii)
                all_grids.append(grid)
        
        if vlim is None:
            vmin = np.nanmin([np.nanmin(g) for g in all_grids])
            vmax = np.nanmax([np.nanmax(g) for g in all_grids])
        else:
            vmin, vmax = vlim
        
        # Create grid of heatmaps
        # Row = E→E, Column = I→I
        im = None
        for row, ee in enumerate(unique_e_to_e):
            for col, ii in enumerate(unique_i_to_i):
                ax = fig.add_subplot(gs[row + 1, col + 1])
                grid = create_2d_grid_for_fixed_ee_ii(data_list, e_to_e_scales, i_to_i_scales,
                                                      e_to_i_scales, i_to_e_scales, ee, ii)
                
                im = ax.imshow(grid, aspect='auto', cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
                
                # Set labels only on outer edges
                if row == n_e_to_e - 1:
                    ax.set_xlabel('E→I Scale', fontsize=9)
                    ax.set_xticks(range(n_e_to_i))
                    ax.set_xticklabels([f'{x:.3f}' for x in unique_e_to_i], rotation=45, fontsize=7)
                else:
                    ax.set_xticks([])
                
                if col == 0:
                    ax.set_ylabel('I→E Scale', fontsize=9)
                    ax.set_yticks(range(n_i_to_e))
                    ax.set_yticklabels([f'{x:.3f}' for x in unique_i_to_e], fontsize=7)
                else:
                    ax.set_yticks([])
        
        # Add column headers (I→I labels) at the top
        for col, ii in enumerate(unique_i_to_i):
            ax_header = fig.add_subplot(gs[0, col + 1])
            ax_header.axis('off')
            ax_header.text(0.5, 0.5, f'I→I={ii:.3f}', 
                          ha='center', va='center', fontsize=11, fontweight='bold',
                          transform=ax_header.transAxes)
        
        # Add row headers (E→E labels) on the left
        for row, ee in enumerate(unique_e_to_e):
            ax_header = fig.add_subplot(gs[row + 1, 0])
            ax_header.axis('off')
            ax_header.text(0.5, 0.5, f'E→E={ee:.3f}', 
                          ha='center', va='center', fontsize=11, fontweight='bold',
                          rotation=90, transform=ax_header.transAxes)
        
        # Add overall title
        fig.suptitle(metric_name, fontsize=16, fontweight='bold', y=0.98)
        
        # Add colorbar on the right side of the entire figure
        if im is not None:
            cbar_ax = fig.add_subplot(gs[1:, n_i_to_i + 1])
            cbar = plt.colorbar(im, cax=cbar_ax, label=metric_name)
            cbar.ax.tick_params(labelsize=9)
            cbar.ax.yaxis.label.set_fontsize(10)
        
        # Save figure
        safe_name = metric_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        save_path = os.path.join(save_folder, f"grid_search_{safe_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    # Also create E/I rate ratio plot
    rate_ratios = [e / i if i > 0 else np.nan for e, i in zip(e_rates, i_rates)]
    # Rows = E→E values, Columns = I→I values
    fig = plt.figure(figsize=(3*n_i_to_i + 2.0, 3*n_e_to_e + 1.0))
    gs = fig.add_gridspec(n_e_to_e + 1, n_i_to_i + 2, 
                          width_ratios=[0.3] + [1]*n_i_to_i + [0.15],
                          height_ratios=[0.3] + [1]*n_e_to_e,
                          hspace=0.3, wspace=0.3)
    
    all_ratio_grids = []
    for ee in unique_e_to_e:
        for ii in unique_i_to_i:
            grid = create_2d_grid_for_fixed_ee_ii(rate_ratios, e_to_e_scales, i_to_i_scales,
                                                  e_to_i_scales, i_to_e_scales, ee, ii)
            all_ratio_grids.append(grid)
    
    vmin = np.nanmin([np.nanmin(g) for g in all_ratio_grids])
    vmax = np.nanmax([np.nanmax(g) for g in all_ratio_grids])
    
    im = None
    # Row = E→E, Column = I→I
    for row, ee in enumerate(unique_e_to_e):
        for col, ii in enumerate(unique_i_to_i):
            ax = fig.add_subplot(gs[row + 1, col + 1])
            grid = create_2d_grid_for_fixed_ee_ii(rate_ratios, e_to_e_scales, i_to_i_scales,
                                                  e_to_i_scales, i_to_e_scales, ee, ii)
            
            im = ax.imshow(grid, aspect='auto', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            
            if row == n_e_to_e - 1:
                ax.set_xlabel('E→I Scale', fontsize=9)
                ax.set_xticks(range(n_e_to_i))
                ax.set_xticklabels([f'{x:.3f}' for x in unique_e_to_i], rotation=45, fontsize=7)
            else:
                ax.set_xticks([])
            
            if col == 0:
                ax.set_ylabel('I→E Scale', fontsize=9)
                ax.set_yticks(range(n_i_to_e))
                ax.set_yticklabels([f'{x:.3f}' for x in unique_i_to_e], fontsize=7)
            else:
                ax.set_yticks([])
    
    # Add column headers (I→I labels) at the top
    for col, ii in enumerate(unique_i_to_i):
        ax_header = fig.add_subplot(gs[0, col + 1])
        ax_header.axis('off')
        ax_header.text(0.5, 0.5, f'I→I={ii:.3f}', 
                      ha='center', va='center', fontsize=11, fontweight='bold',
                      transform=ax_header.transAxes)
    
    # Add row headers (E→E labels) on the left
    for row, ee in enumerate(unique_e_to_e):
        ax_header = fig.add_subplot(gs[row + 1, 0])
        ax_header.axis('off')
        ax_header.text(0.5, 0.5, f'E→E={ee:.3f}', 
                      ha='center', va='center', fontsize=11, fontweight='bold',
                      rotation=90, transform=ax_header.transAxes)
    
    fig.suptitle('E/I Rate Ratio', fontsize=16, fontweight='bold', y=0.98)
    
    # Add colorbar on the right side
    if im is not None:
        cbar_ax = fig.add_subplot(gs[1:, n_i_to_i + 1])
        cbar = plt.colorbar(im, cax=cbar_ax, label='E/I Rate Ratio')
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.yaxis.label.set_fontsize(10)
    
    save_path = os.path.join(save_folder, "grid_search_ei_rate_ratio.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    print(f"All plots saved to: {save_folder}")

if __name__ == "__main__":
    print("="*60)
    print("PLOTTING GRID SEARCH RESULTS")
    print("="*60)
    
    # Find and load the latest results
    try:
        results_file, results_folder = find_latest_grid_search_results()
        print(f"\nLoading results from: {results_file}")
        
        with open(results_file, 'rb') as f:
            all_results = pickle.load(f)
        
        print(f"Loaded {len(all_results)} results")
        successful_results = [r for r in all_results if r['success']]
        print(f"Successful runs: {len(successful_results)}/{len(all_results)}")
        
        # Create summary plots into the standard Figures subfolder
        figures_folder = os.path.join(results_folder, "Figures")
        os.makedirs(figures_folder, exist_ok=True)
        create_summary_plots(all_results, figures_folder)
        
        # Optional: comprehensive 4D diagnostic plot (disabled for now to avoid shape issues)
        # create_comprehensive_heatmap_grid(all_results, results_folder)
        
        print(f"\n{'='*60}")
        print("PLOTTING COMPLETE")
        print(f"{'='*60}")
        print(f"Results folder: {results_folder}")
        print(f"{'='*60}\n")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease run recurrent/parameter_search/grid_search.py first to generate simulation data.")

