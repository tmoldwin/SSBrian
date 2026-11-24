"""
Neural Correlation Analysis: Firing rate correlations between neurons
and their relationship to selectivity/sparsity properties.

Creates correlation matrices and analyzes how neural correlations relate
to individual neuron properties like selectivity and sparsity.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

def load_simulation_data(data_folder_path):
    """Load simulation data from pickle file."""
    data_file = os.path.join(data_folder_path, "simulation_data.pkl")
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        print(f"[OK] Data loaded from: {data_file}")
        return data
    else:
        raise FileNotFoundError(f"No simulation data found at: {data_file}")

def analyze_neural_correlations(data_folder_path):
    """
    Analyze correlations between neural firing rates and their relationship
    to selectivity/sparsity properties.
    """
    # Load data
    data = load_simulation_data(data_folder_path)
    
    # Get final epoch data
    final_rate_matrix = data['rate_matrices_arr'][-1]  # Shape: (neurons, stimuli)
    final_selectivity = data['selectivity_vec_arr'][-1, :]  # Shape: (neurons,)
    final_sparsity = data['sparsity_vec_arr'][-1, :]  # Shape: (stimuli,) - sparsity is per stimulus
    
    # Calculate mean firing rates for each neuron across all stimuli
    mean_firing_rates = np.mean(final_rate_matrix, axis=1)
    
    # Only analyze neurons with some activity
    active_mask = mean_firing_rates > 0
    active_rate_matrix = final_rate_matrix[active_mask, :]
    active_selectivity = final_selectivity[active_mask]
    active_firing_rates = mean_firing_rates[active_mask]
    
    # For sparsity, we'll use the mean sparsity across all stimuli as a network property
    mean_network_sparsity = np.mean(final_sparsity)
    
    print(f"📊 Data shapes: rate_matrix={final_rate_matrix.shape}, selectivity={final_selectivity.shape}, sparsity={final_sparsity.shape}")
    print(f"📊 Active neurons: {np.sum(active_mask)} out of {len(active_mask)}")
    print(f"📊 Mean network sparsity: {mean_network_sparsity:.3f}")
    
    print(f"📊 Analyzing {len(active_firing_rates)} active neurons")
    
    # Calculate firing rate correlations between neurons
    # Each row is a neuron's response across all stimuli
    neural_correlations = np.corrcoef(active_rate_matrix)
    
    # Remove diagonal (self-correlation = 1)
    np.fill_diagonal(neural_correlations, np.nan)
    
    # Create the main figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Neural correlation matrix heatmap
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(neural_correlations, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax1.set_title('Neural Firing Rate Correlations', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Neuron Index')
    ax1.set_ylabel('Neuron Index')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Distribution of correlations
    ax2 = plt.subplot(2, 3, 2)
    correlations_flat = neural_correlations[~np.isnan(neural_correlations)]
    ax2.hist(correlations_flat, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(correlations_flat), color='red', linestyle='--', 
               label=f'Mean = {np.mean(correlations_flat):.3f}')
    ax2.set_xlabel('Correlation Coefficient')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Neural Correlations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Selectivity vs Mean Correlation
    ax3 = plt.subplot(2, 3, 3)
    # For each neuron, calculate mean correlation with all other neurons
    mean_correlations = np.nanmean(neural_correlations, axis=1)
    
    # Scatter plot: selectivity vs mean correlation
    scatter = ax3.scatter(active_selectivity, mean_correlations, 
                         c=active_firing_rates, cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Neuron Selectivity')
    ax3.set_ylabel('Mean Correlation with Other Neurons')
    ax3.set_title('Selectivity vs Mean Correlation\n(Color = Firing Rate)')
    plt.colorbar(scatter, ax=ax3, label='Firing Rate (Hz)')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    sel_corr_coeff, sel_p_value = stats.pearsonr(active_selectivity, mean_correlations)
    ax3.text(0.05, 0.95, f'r = {sel_corr_coeff:.3f}\np = {sel_p_value:.3e}', 
             transform=ax3.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 4. Firing Rate vs Mean Correlation
    ax4 = plt.subplot(2, 3, 4)
    scatter2 = ax4.scatter(active_firing_rates, mean_correlations, 
                          c=active_selectivity, cmap='plasma', 
                          alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Mean Firing Rate (Hz)')
    ax4.set_ylabel('Mean Correlation with Other Neurons')
    ax4.set_title('Firing Rate vs Mean Correlation\n(Color = Selectivity)')
    plt.colorbar(scatter2, ax=ax4, label='Selectivity')
    ax4.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    firing_corr_coeff, firing_p_value = stats.pearsonr(active_firing_rates, mean_correlations)
    ax4.text(0.05, 0.95, f'r = {firing_corr_coeff:.3f}\np = {firing_p_value:.3e}', 
             transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 5. Hierarchical clustering of neurons based on firing patterns
    ax5 = plt.subplot(2, 3, 5)
    if len(active_firing_rates) > 2:
        # Use correlation distance for clustering
        distances = pdist(active_rate_matrix, metric='correlation')
        linkage_matrix = linkage(distances, method='ward')
        
        # Create dendrogram
        dendrogram(linkage_matrix, ax=ax5, truncate_mode='lastp', p=min(20, len(active_firing_rates)))
        ax5.set_title('Neural Response Clustering')
        ax5.set_xlabel('Neuron Index')
        ax5.set_ylabel('Distance')
    else:
        ax5.text(0.5, 0.5, 'Too few neurons\nfor clustering', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Neural Response Clustering')
    
    # 6. High vs Low Selectivity Correlation Analysis
    ax6 = plt.subplot(2, 3, 6)
    
    # Split neurons into high and low selectivity groups
    median_selectivity = np.median(active_selectivity)
    high_sel_mask = active_selectivity >= median_selectivity
    low_sel_mask = active_selectivity < median_selectivity
    
    # Calculate correlations within each group
    high_sel_correlations = neural_correlations[high_sel_mask][:, high_sel_mask]
    low_sel_correlations = neural_correlations[low_sel_mask][:, low_sel_mask]
    
    # Remove diagonals
    np.fill_diagonal(high_sel_correlations, np.nan)
    np.fill_diagonal(low_sel_correlations, np.nan)
    
    # Plot distributions
    high_sel_flat = high_sel_correlations[~np.isnan(high_sel_correlations)]
    low_sel_flat = low_sel_correlations[~np.isnan(low_sel_correlations)]
    
    ax6.hist(high_sel_flat, bins=20, alpha=0.6, label=f'High Selectivity (n={len(high_sel_flat)})', 
             color='red', density=True)
    ax6.hist(low_sel_flat, bins=20, alpha=0.6, label=f'Low Selectivity (n={len(low_sel_flat)})', 
             color='blue', density=True)
    
    ax6.set_xlabel('Correlation Coefficient')
    ax6.set_ylabel('Density')
    ax6.set_title('Correlations: High vs Low Selectivity Neurons')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(data_folder_path, 'neural_correlation_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Figure saved to: {plot_path}")
    
    # Print summary statistics
    print(f"\n📊 NEURAL CORRELATION ANALYSIS SUMMARY:")
    print(f"   - Mean correlation between neurons: {np.mean(correlations_flat):.3f} ± {np.std(correlations_flat):.3f}")
    print(f"   - Selectivity vs Mean Correlation: r = {sel_corr_coeff:.3f}, p = {sel_p_value:.3e}")
    print(f"   - Firing Rate vs Mean Correlation: r = {firing_corr_coeff:.3f}, p = {firing_p_value:.3e}")
    print(f"   - Network sparsity: {mean_network_sparsity:.3f}")
    print(f"   - High selectivity neurons: {np.sum(high_sel_mask)}")
    print(f"   - Low selectivity neurons: {np.sum(low_sel_mask)}")
    
    if len(high_sel_flat) > 0 and len(low_sel_flat) > 0:
        print(f"   - High sel. mean correlation: {np.mean(high_sel_flat):.3f}")
        print(f"   - Low sel. mean correlation: {np.mean(low_sel_flat):.3f}")
    
    return {
        'neural_correlations': neural_correlations,
        'mean_correlations': mean_correlations,
        'selectivity_correlation': (sel_corr_coeff, sel_p_value),
        'firing_rate_correlation': (firing_corr_coeff, firing_p_value),
        'network_sparsity': mean_network_sparsity,
        'high_sel_correlations': high_sel_flat,
        'low_sel_correlations': low_sel_flat
    }

def main():
    """Run the neural correlation analysis."""
    # Find first available dataset
    folders_to_check = [
        "brian_experiment/Figures/DI_Sims/direct_input_REC_stim50_ep5_n200",
        "brian_experiment/Figures/DI_Sims/direct_input_FF_stim50_ep5_n200",
        "Figures/DI_Sims/direct_input_REC_stim50_ep5_n200",
        "Figures/DI_Sims/direct_input_FF_stim50_ep5_n200"
    ]
    
    data_folder = None
    for folder in folders_to_check:
        if os.path.exists(os.path.join(folder, "simulation_data.pkl")):
            data_folder = folder
            break
    
    if data_folder is None:
        print("❌ No simulation data found!")
        return
    
    print(f"🎯 Analyzing neural correlations: {data_folder}")
    results = analyze_neural_correlations(data_folder)

if __name__ == "__main__":
    main()
