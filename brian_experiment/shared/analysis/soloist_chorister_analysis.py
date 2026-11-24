"""
Soloist/Chorister Analysis Module

Analyzes neurons based on their correlation patterns with E and I populations.
High correlation = chorister, low correlation = soloist.

Creates 4x3 panel figure showing:
- Row 1: E→E correlations (E neurons with E neurons)
- Row 2: E→I correlations (E neurons with I neurons)
- Row 3: I→E correlations (I neurons with E neurons)
- Row 4: I→I correlations (I neurons with I neurons)
- Columns: Heatmap, histogram of mean correlations, selectivity vs correlation
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import stats


def generate_soloist_chorister_analysis(run_folder, data, config):
    """
    Generate comprehensive soloist/chorister analysis.
    
    Parameters:
    -----------
    run_folder : str
        Path to the run directory
    data : dict
        Dictionary containing simulation data
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    dict : Analysis statistics
    """
    print("\n" + "="*60)
    print("SOLOIST/CHORISTER ANALYSIS")
    print("="*60)
    
    # Create subfolder
    correlation_folder = os.path.join(run_folder, "Figures", "correlation_analysis")
    os.makedirs(correlation_folder, exist_ok=True)
    
    # Extract data from final epoch
    final_rate_matrix = data['rate_matrices_arr'][-1]  # Shape: (neurons, stimuli)
    final_selectivity = data['selectivity_vec_arr'][-1, :]  # Shape: (neurons,)
    
    N_rec_E = config.get('N_rec_E', 100)
    N_rec_I = config.get('N_rec_I', 100)
    N_rec = N_rec_E + N_rec_I
    
    # Calculate mean firing rates
    mean_firing_rates = np.mean(final_rate_matrix, axis=1)
    
    # Only analyze active neurons
    active_mask = mean_firing_rates > 0
    active_rate_matrix = final_rate_matrix[active_mask, :]
    active_selectivity = final_selectivity[active_mask]
    active_firing_rates = mean_firing_rates[active_mask]
    
    # Get original indices for active neurons
    active_indices = np.where(active_mask)[0]
    active_is_E = active_indices < N_rec_E
    active_is_I = active_indices >= N_rec_E
    
    # Indices in the active neuron array for E and I neurons
    E_indices = np.where(active_is_E)[0]
    I_indices = np.where(active_is_I)[0]
    
    print(f"   Active neurons: {np.sum(active_mask)} out of {len(active_mask)}")
    print(f"   Active E neurons: {np.sum(active_is_E)} (indices 0:{N_rec_E})")
    print(f"   Active I neurons: {np.sum(active_is_I)} (indices {N_rec_E}:{N_rec})")
    
    # Calculate full correlation matrix (all active neurons x all active neurons)
    neural_correlations = np.corrcoef(active_rate_matrix)
    np.fill_diagonal(neural_correlations, np.nan)
    
    print(f"   Correlation matrix shape: {neural_correlations.shape}")
    
    # Extract correlations for each E/I combination
    # E→E: E neurons correlating with E neurons
    corr_E_to_E = neural_correlations[np.ix_(E_indices, E_indices)]
    # E→I: E neurons correlating with I neurons
    corr_E_to_I = neural_correlations[np.ix_(E_indices, I_indices)]
    # I→E: I neurons correlating with E neurons
    corr_I_to_E = neural_correlations[np.ix_(I_indices, E_indices)]
    # I→I: I neurons correlating with I neurons
    corr_I_to_I = neural_correlations[np.ix_(I_indices, I_indices)]
    
    # Calculate mean correlation per neuron (for choristerness metric)
    mean_corr_E_to_E = np.nanmean(corr_E_to_E, axis=1)
    mean_corr_E_to_I = np.nanmean(corr_E_to_I, axis=1)
    mean_corr_I_to_E = np.nanmean(corr_I_to_E, axis=1)
    mean_corr_I_to_I = np.nanmean(corr_I_to_I, axis=1)
    
    # Get selectivity for each group
    selectivity_E = active_selectivity[E_indices]
    selectivity_I = active_selectivity[I_indices]
    
    # Create 4x3 figure
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle('Soloist/Chorister Analysis: E/I Correlation Combinations', 
                 fontsize=16, fontweight='bold')
    
    # ===== ROW 1: E→E CORRELATIONS =====
    row = 0
    
    # Panel 1,1: Heatmap of E→E correlations
    ax = axes[row, 0]
    im = ax.imshow(corr_E_to_E, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    ax.set_xlabel('E Neuron Index')
    ax.set_ylabel('E Neuron Index')
    ax.set_title('E->E Correlations')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Panel 1,2: Histogram of mean E→E correlation
    ax = axes[row, 1]
    ax.hist(mean_corr_E_to_E, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(np.nanmean(mean_corr_E_to_E), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.nanmean(mean_corr_E_to_E):.3f}')
    ax.axvline(np.nanmedian(mean_corr_E_to_E), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.nanmedian(mean_corr_E_to_E):.3f}')
    ax.set_xlabel('Mean Correlation with E Neurons')
    ax.set_ylabel('Count')
    ax.set_title('E->E: Choristerness Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 1,3: Selectivity vs E→E correlation
    ax = axes[row, 2]
    ax.scatter(mean_corr_E_to_E, selectivity_E, c='red', alpha=0.6, 
               s=50, edgecolors='black', linewidth=0.5)
    
    valid_mask = ~np.isnan(mean_corr_E_to_E)
    if np.sum(valid_mask) > 2:
        z = np.polyfit(mean_corr_E_to_E[valid_mask], selectivity_E[valid_mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(np.nanmin(mean_corr_E_to_E), np.nanmax(mean_corr_E_to_E), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2)
        
        r, p_val = stats.pearsonr(mean_corr_E_to_E[valid_mask], selectivity_E[valid_mask])
        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3e}', 
                transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    ax.set_xlabel('Mean Correlation with E Neurons')
    ax.set_ylabel('Selectivity Index')
    ax.set_title('E->E: Selectivity vs Choristerness')
    ax.grid(True, alpha=0.3)
    
    # ===== ROW 2: E→I CORRELATIONS =====
    row = 1
    
    # Panel 2,1: Heatmap of E→I correlations
    ax = axes[row, 0]
    im = ax.imshow(corr_E_to_I, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    ax.set_xlabel('I Neuron Index')
    ax.set_ylabel('E Neuron Index')
    ax.set_title('E->I Correlations')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Panel 2,2: Histogram of mean E→I correlation
    ax = axes[row, 1]
    ax.hist(mean_corr_E_to_I, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(np.nanmean(mean_corr_E_to_I), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.nanmean(mean_corr_E_to_I):.3f}')
    ax.axvline(np.nanmedian(mean_corr_E_to_I), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.nanmedian(mean_corr_E_to_I):.3f}')
    ax.set_xlabel('Mean Correlation with I Neurons')
    ax.set_ylabel('Count')
    ax.set_title('E->I: Choristerness Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2,3: Selectivity vs E→I correlation
    ax = axes[row, 2]
    ax.scatter(mean_corr_E_to_I, selectivity_E, c='red', alpha=0.6,
               s=50, edgecolors='black', linewidth=0.5)
    
    valid_mask = ~np.isnan(mean_corr_E_to_I)
    if np.sum(valid_mask) > 2:
        z = np.polyfit(mean_corr_E_to_I[valid_mask], selectivity_E[valid_mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(np.nanmin(mean_corr_E_to_I), np.nanmax(mean_corr_E_to_I), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2)
        
        r, p_val = stats.pearsonr(mean_corr_E_to_I[valid_mask], selectivity_E[valid_mask])
        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3e}', 
                transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    ax.set_xlabel('Mean Correlation with I Neurons')
    ax.set_ylabel('Selectivity Index')
    ax.set_title('E->I: Selectivity vs Choristerness')
    ax.grid(True, alpha=0.3)
    
    # ===== ROW 3: I→E CORRELATIONS =====
    row = 2
    
    # Panel 3,1: Heatmap of I→E correlations
    ax = axes[row, 0]
    im = ax.imshow(corr_I_to_E, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    ax.set_xlabel('E Neuron Index')
    ax.set_ylabel('I Neuron Index')
    ax.set_title('I->E Correlations')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Panel 3,2: Histogram of mean I→E correlation
    ax = axes[row, 1]
    ax.hist(mean_corr_I_to_E, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(np.nanmean(mean_corr_I_to_E), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.nanmean(mean_corr_I_to_E):.3f}')
    ax.axvline(np.nanmedian(mean_corr_I_to_E), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.nanmedian(mean_corr_I_to_E):.3f}')
    ax.set_xlabel('Mean Correlation with E Neurons')
    ax.set_ylabel('Count')
    ax.set_title('I->E: Choristerness Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3,3: Selectivity vs I→E correlation
    ax = axes[row, 2]
    ax.scatter(mean_corr_I_to_E, selectivity_I, c='blue', alpha=0.6,
               s=50, edgecolors='black', linewidth=0.5)
    
    valid_mask = ~np.isnan(mean_corr_I_to_E)
    if np.sum(valid_mask) > 2:
        z = np.polyfit(mean_corr_I_to_E[valid_mask], selectivity_I[valid_mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(np.nanmin(mean_corr_I_to_E), np.nanmax(mean_corr_I_to_E), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2)
        
        r, p_val = stats.pearsonr(mean_corr_I_to_E[valid_mask], selectivity_I[valid_mask])
        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3e}', 
                transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    ax.set_xlabel('Mean Correlation with E Neurons')
    ax.set_ylabel('Selectivity Index')
    ax.set_title('I->E: Selectivity vs Choristerness')
    ax.grid(True, alpha=0.3)
    
    # ===== ROW 4: I→I CORRELATIONS =====
    row = 3
    
    # Panel 4,1: Heatmap of I→I correlations
    ax = axes[row, 0]
    im = ax.imshow(corr_I_to_I, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    ax.set_xlabel('I Neuron Index')
    ax.set_ylabel('I Neuron Index')
    ax.set_title('I->I Correlations')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Panel 4,2: Histogram of mean I→I correlation
    ax = axes[row, 1]
    ax.hist(mean_corr_I_to_I, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(np.nanmean(mean_corr_I_to_I), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.nanmean(mean_corr_I_to_I):.3f}')
    ax.axvline(np.nanmedian(mean_corr_I_to_I), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.nanmedian(mean_corr_I_to_I):.3f}')
    ax.set_xlabel('Mean Correlation with I Neurons')
    ax.set_ylabel('Count')
    ax.set_title('I->I: Choristerness Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4,3: Selectivity vs I→I correlation
    ax = axes[row, 2]
    ax.scatter(mean_corr_I_to_I, selectivity_I, c='blue', alpha=0.6,
               s=50, edgecolors='black', linewidth=0.5)
    
    valid_mask = ~np.isnan(mean_corr_I_to_I)
    if np.sum(valid_mask) > 2:
        z = np.polyfit(mean_corr_I_to_I[valid_mask], selectivity_I[valid_mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(np.nanmin(mean_corr_I_to_I), np.nanmax(mean_corr_I_to_I), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2)
        
        r, p_val = stats.pearsonr(mean_corr_I_to_I[valid_mask], selectivity_I[valid_mask])
        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3e}', 
                transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    ax.set_xlabel('Mean Correlation with I Neurons')
    ax.set_ylabel('Selectivity Index')
    ax.set_title('I->I: Selectivity vs Choristerness')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(correlation_folder, 'soloist_chorister_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {save_path}")
    
    # Compile statistics for all four combinations
    stats_dict = {
        'E_to_E': {
            'mean': np.nanmean(mean_corr_E_to_E),
            'median': np.nanmedian(mean_corr_E_to_E),
            'std': np.nanstd(mean_corr_E_to_E),
        },
        'E_to_I': {
            'mean': np.nanmean(mean_corr_E_to_I),
            'median': np.nanmedian(mean_corr_E_to_I),
            'std': np.nanstd(mean_corr_E_to_I),
        },
        'I_to_E': {
            'mean': np.nanmean(mean_corr_I_to_E),
            'median': np.nanmedian(mean_corr_I_to_E),
            'std': np.nanstd(mean_corr_I_to_E),
        },
        'I_to_I': {
            'mean': np.nanmean(mean_corr_I_to_I),
            'median': np.nanmedian(mean_corr_I_to_I),
            'std': np.nanstd(mean_corr_I_to_I),
        }
    }
    
    # Add correlation with selectivity for each combination
    valid_mask = ~np.isnan(mean_corr_E_to_E)
    if np.sum(valid_mask) > 2:
        r, p_val = stats.pearsonr(mean_corr_E_to_E[valid_mask], selectivity_E[valid_mask])
        stats_dict['E_to_E']['selectivity_correlation'] = r
        stats_dict['E_to_E']['selectivity_p_value'] = p_val
    
    valid_mask = ~np.isnan(mean_corr_E_to_I)
    if np.sum(valid_mask) > 2:
        r, p_val = stats.pearsonr(mean_corr_E_to_I[valid_mask], selectivity_E[valid_mask])
        stats_dict['E_to_I']['selectivity_correlation'] = r
        stats_dict['E_to_I']['selectivity_p_value'] = p_val
    
    valid_mask = ~np.isnan(mean_corr_I_to_E)
    if np.sum(valid_mask) > 2:
        r, p_val = stats.pearsonr(mean_corr_I_to_E[valid_mask], selectivity_I[valid_mask])
        stats_dict['I_to_E']['selectivity_correlation'] = r
        stats_dict['I_to_E']['selectivity_p_value'] = p_val
    
    valid_mask = ~np.isnan(mean_corr_I_to_I)
    if np.sum(valid_mask) > 2:
        r, p_val = stats.pearsonr(mean_corr_I_to_I[valid_mask], selectivity_I[valid_mask])
        stats_dict['I_to_I']['selectivity_correlation'] = r
        stats_dict['I_to_I']['selectivity_p_value'] = p_val
    
    # Print summary
    print("\n   Soloist/Chorister Analysis Summary:")
    print(f"   E->E: {stats_dict['E_to_E']['mean']:.3f} +/- {stats_dict['E_to_E']['std']:.3f}")
    print(f"   E->I: {stats_dict['E_to_I']['mean']:.3f} +/- {stats_dict['E_to_I']['std']:.3f}")
    print(f"   I->E: {stats_dict['I_to_E']['mean']:.3f} +/- {stats_dict['I_to_E']['std']:.3f}")
    print(f"   I->I: {stats_dict['I_to_I']['mean']:.3f} +/- {stats_dict['I_to_I']['std']:.3f}")
    
    for combo in ['E_to_E', 'E_to_I', 'I_to_E', 'I_to_I']:
        if 'selectivity_correlation' in stats_dict[combo]:
            r = stats_dict[combo]['selectivity_correlation']
            p = stats_dict[combo]['selectivity_p_value']
            print(f"   {combo} selectivity correlation: r = {r:.3f}, p = {p:.3e}")
    
    return stats_dict


if __name__ == "__main__":
    """Command-line interface for standalone soloist/chorister analysis"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python soloist_chorister_analysis.py <run_folder>")
        sys.exit(1)
    
    run_folder = sys.argv[1]
    data_file = os.path.join(run_folder, "Data", "simulation_data.pkl")
    
    if not os.path.exists(data_file):
        print(f"Error: No data file found at {data_file}")
        sys.exit(1)
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    config = data.get('config', {})
    stats = generate_soloist_chorister_analysis(run_folder, data, config)
    print("\nSoloist/Chorister analysis complete!")

