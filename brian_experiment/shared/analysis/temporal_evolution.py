"""
Temporal Evolution Analysis Module

This module analyzes how network properties evolve over epochs.
Creates plots in the temporal_evolution/ subfolder.

Functions:
- generate_temporal_evolution_plots: Main entry point for temporal analysis
- plot_sparsity_selectivity_evolution: Evolution over epochs
- plot_selectivity_comparison: Before/after comparison
- plot_rate_matrices: Firing rate matrix visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import figures as figs
import sparsity_and_selectivity as ss


def generate_temporal_evolution_plots(run_folder, data, config, network=None):
    """
    Generate temporal evolution visualizations.
    
    Parameters:
    -----------
    run_folder : str
        Path to the run directory
    data : dict
        Dictionary containing simulation data
    config : dict
        Configuration dictionary
    network : object, optional
        Network object (may be None if loading from saved data)
    
    Returns:
    --------
    dict : Statistics about temporal evolution
    """
    print("\n" + "="*60)
    print("TEMPORAL EVOLUTION ANALYSIS")
    print("="*60)
    
    # Create subfolder
    temporal_folder = os.path.join(run_folder, "Figures", "temporal_evolution")
    os.makedirs(temporal_folder, exist_ok=True)
    
    # Extract data
    individual_rate_matrices = data.get('individual_rate_matrices', [])
    individual_sparsity = data.get('individual_sparsity', np.array([]))
    individual_selectivity = data.get('individual_selectivity', np.array([]))
    rate_matrices_arr = data.get('rate_matrices_arr', [])
    sparsity_vec_arr = data.get('sparsity_vec_arr', np.array([]))
    selectivity_vec_arr = data.get('selectivity_vec_arr', np.array([]))
    neuron_order = data.get('neuron_order')
    
    num_epochs = config.get('num_epochs', 2)
    N_rec_E = config.get('N_rec_E', 100)
    N_rec_I = config.get('N_rec_I', 100)
    
    stats = {}
    
    try:
        # ===== 1. SPARSITY AND SELECTIVITY EVOLUTION =====
        print("   Creating sparsity and selectivity evolution plot...")
        
        # Create minimal network object if needed
        class MinimalNetwork:
            def __init__(self, neuron_order):
                self.neuron_order = neuron_order
        
        net = network if network is not None else MinimalNetwork(neuron_order)
        
        figs.plotting_dashboard_figure_3(
            net, 
            individual_sparsity.T, 
            individual_selectivity.T, 
            sparsity_vec_arr.T, 
            selectivity_vec_arr.T, 
            save_folder=temporal_folder
        )
        print("      Saved: sparsity_selectivity_evolution_over_epochs.png")
        
        # Calculate statistics
        if len(selectivity_vec_arr) > 0:
            stats['selectivity_evolution'] = {
                'first_epoch_mean': np.mean(selectivity_vec_arr[0, :]),
                'last_epoch_mean': np.mean(selectivity_vec_arr[-1, :]),
                'first_epoch_std': np.std(selectivity_vec_arr[0, :]),
                'last_epoch_std': np.std(selectivity_vec_arr[-1, :]),
            }
            
        if len(sparsity_vec_arr) > 0:
            stats['sparsity_evolution'] = {
                'first_epoch_mean': np.mean(sparsity_vec_arr[0, :]),
                'last_epoch_mean': np.mean(sparsity_vec_arr[-1, :]),
                'first_epoch_std': np.std(sparsity_vec_arr[0, :]),
                'last_epoch_std': np.std(sparsity_vec_arr[-1, :]),
            }
        
    except Exception as e:
        print(f"   Error creating sparsity/selectivity evolution: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # ===== 2. SELECTIVITY COMPARISON ACROSS EPOCHS =====
        print("   Creating selectivity comparison plot...")
        
        if num_epochs >= 2 and len(individual_selectivity) >= 2:
            fig4, axs = plt.subplots(1, 2, figsize=(14, 6))
            fig4.suptitle('Selectivity Comparison: First vs Last Epoch', fontsize=16, fontweight='bold')
            
            # Get first and last epoch selectivity (per neuron)
            first_epoch_sel = individual_selectivity[0, :]
            last_epoch_sel = individual_selectivity[-1, :]
            
            # Remove NaNs
            valid_mask = ~(np.isnan(first_epoch_sel) | np.isnan(last_epoch_sel))
            first_clean = first_epoch_sel[valid_mask]
            last_clean = last_epoch_sel[valid_mask]
            
            # Left panel: Scatter plot
            axs[0].scatter(first_clean, last_clean, alpha=0.6, s=30)
            axs[0].plot([0, 1], [0, 1], 'r--', label='Identity', linewidth=2)
            axs[0].set_xlabel('Selectivity (First Epoch)', fontsize=12)
            axs[0].set_ylabel('Selectivity (Last Epoch)', fontsize=12)
            axs[0].set_title('Neuron Selectivity Change')
            axs[0].grid(True, alpha=0.3)
            axs[0].legend()
            axs[0].set_xlim(0, 1)
            axs[0].set_ylim(0, 1)
            
            # Calculate correlation only if defined (enough points, non-constant)
            from scipy.stats import pearsonr
            if len(first_clean) > 1 and np.std(first_clean) > 0 and np.std(last_clean) > 0:
                corr, p_value = pearsonr(first_clean, last_clean)
                corr_text = f'r = {corr:.3f}\np = {p_value:.3e}'
            else:
                corr = p_value = np.nan
                corr_text = 'r = N/A\np = N/A'
            
            axs[0].text(
                0.05, 0.95, corr_text,
                transform=axs[0].transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            
            # Right panel: Histograms
            axs[1].hist(first_clean, bins=30, alpha=0.6, label='First Epoch', 
                       color='blue', density=True)
            axs[1].hist(last_clean, bins=30, alpha=0.6, label='Last Epoch', 
                       color='red', density=True)
            if len(first_clean) > 0:
                axs[1].axvline(np.mean(first_clean), color='blue', linestyle='--', 
                              linewidth=2, label=f'First Mean = {np.mean(first_clean):.3f}')
            if len(last_clean) > 0:
                axs[1].axvline(np.mean(last_clean), color='red', linestyle='--', 
                              linewidth=2, label=f'Last Mean = {np.mean(last_clean):.3f}')
            axs[1].set_xlabel('Selectivity', fontsize=12)
            axs[1].set_ylabel('Density', fontsize=12)
            axs[1].set_title('Distribution Change')
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(temporal_folder, 'selectivity_comparison_first_vs_last_epoch.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close(fig4)
            print("      Saved: selectivity_comparison_first_vs_last_epoch.png")
            
            # Store stats only if correlation is defined
            if not np.isnan(corr):
                stats['selectivity_comparison'] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'mean_change': float(np.mean(last_clean) - np.mean(first_clean))
                }
        else:
            print("      Skipping selectivity comparison - requires at least 2 epochs")
            
    except Exception as e:
        print(f"   Error creating selectivity comparison: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # ===== 3. EXEMPLAR NEURON STEM PLOTS =====
        print("   Creating exemplar neuron stem plots...")
        
        if len(rate_matrices_arr) >= 2:
            import rate_analysis as ra
            
            # Use individual epochs for first and last, cumulative average for first row
            # Note: The rate matrices are already in Hz from analyze_firing_rates_over_time_long
            first_rate_matrix = individual_rate_matrices[0] if len(individual_rate_matrices) > 0 else rate_matrices_arr[0]
            last_rate_matrix = individual_rate_matrices[-1] if len(individual_rate_matrices) > 0 else rate_matrices_arr[-1]
            avg_rate_matrix = rate_matrices_arr[-1]  # Cumulative average of all epochs
            
            # Calculate standard deviation across epochs for error bars
            if len(individual_rate_matrices) > 0:
                rate_matrices_3d = np.array(individual_rate_matrices)  # Shape: (n_epochs, n_neurons, n_stimuli)
                # Rate matrices are already in Hz, no conversion needed
                std_across_epochs = np.std(rate_matrices_3d, axis=0)  # Shape: (n_neurons, n_stimuli)
            else:
                std_across_epochs = None
            
            # Comparison plot (average first, then all epochs)
            all_epoch_matrices = individual_rate_matrices if len(individual_rate_matrices) > 0 else None
            fig_comp = ra.plot_selectivity_comparison(avg_rate_matrix, first_rate_matrix, last_rate_matrix, 
                                                     std_matrix=std_across_epochs,
                                                     all_epoch_matrices=all_epoch_matrices,
                                                     title="Neuron Selectivity Comparison (Average vs All Epochs)")
            if fig_comp:
                save_path = os.path.join(temporal_folder, 'exemplar_neuron_comparison.png')
                fig_comp.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close(fig_comp)
                print("      Saved: exemplar_neuron_comparison.png")
            
            # Stimulus response comparison
            fig_stim = ra.plot_stimulus_response_comparison(first_rate_matrix, last_rate_matrix, N_rec_E,
                                                           num_to_plot=4,
                                                           title="Stimulus Response Comparison (First vs Last Epoch)")
            save_path = os.path.join(temporal_folder, 'stimulus_response_comparison.png')
            fig_stim.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig_stim)
            print("      Saved: stimulus_response_comparison.png")
            
    except Exception as e:
        print(f"   Error creating exemplar neuron stem plots: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # ===== 4. RATE MATRIX VISUALIZATION =====
        print("   Creating firing rate matrix visualization...")
        
        if len(rate_matrices_arr) >= 2:
            first_rate_matrix = rate_matrices_arr[0]
            last_rate_matrix = rate_matrices_arr[-1]
            
            # Create interleaved visualization
            fig = plt.figure(figsize=(16, 8))
            fig.suptitle('Firing Rate Matrices: First vs Last Epoch', fontsize=16, fontweight='bold')
            
            # Determine vmax for consistent color scale
            vmax = max(np.max(first_rate_matrix), np.max(last_rate_matrix))
            
            # First epoch
            ax1 = plt.subplot(1, 2, 1)
            im1 = ax1.imshow(first_rate_matrix, aspect='auto', cmap='viridis', 
                            vmin=0, vmax=vmax, interpolation='nearest')
            ax1.axhline(y=N_rec_E-0.5, color='red', linewidth=2, alpha=0.7)
            ax1.set_title('First Epoch', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Stimulus', fontsize=12)
            ax1.set_ylabel('Neuron', fontsize=12)
            ax1.text(first_rate_matrix.shape[1]//2, N_rec_E//2, 'E', 
                    ha='center', va='center', color='white', fontsize=14, fontweight='bold')
            ax1.text(first_rate_matrix.shape[1]//2, N_rec_E + N_rec_I//2, 'I', 
                    ha='center', va='center', color='white', fontsize=14, fontweight='bold')
            plt.colorbar(im1, ax=ax1, label='Firing Rate (Hz)')
            
            # Last epoch
            ax2 = plt.subplot(1, 2, 2)
            im2 = ax2.imshow(last_rate_matrix, aspect='auto', cmap='viridis', 
                            vmin=0, vmax=vmax, interpolation='nearest')
            ax2.axhline(y=N_rec_E-0.5, color='red', linewidth=2, alpha=0.7)
            ax2.set_title('Last Epoch', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Stimulus', fontsize=12)
            ax2.set_ylabel('Neuron', fontsize=12)
            ax2.text(last_rate_matrix.shape[1]//2, N_rec_E//2, 'E', 
                    ha='center', va='center', color='white', fontsize=14, fontweight='bold')
            ax2.text(last_rate_matrix.shape[1]//2, N_rec_E + N_rec_I//2, 'I', 
                    ha='center', va='center', color='white', fontsize=14, fontweight='bold')
            plt.colorbar(im2, ax=ax2, label='Firing Rate (Hz)')
            
            plt.tight_layout()
            save_path = os.path.join(temporal_folder, 'firing_rate_matrices_first_vs_last_epoch.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print("      Saved: firing_rate_matrices_first_vs_last_epoch.png")
            
            # Calculate change statistics
            stats['rate_matrix_change'] = {
                'first_mean': np.mean(first_rate_matrix),
                'last_mean': np.mean(last_rate_matrix),
                'mean_change': np.mean(last_rate_matrix) - np.mean(first_rate_matrix),
                'first_max': np.max(first_rate_matrix),
                'last_max': np.max(last_rate_matrix),
            }
            
    except Exception as e:
        print(f"   Error creating rate matrix visualization: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # ===== 4. FANO FACTOR ANALYSIS =====
        print("   Creating Fano factor analysis...")
        fano_stats = plot_fano_factor_analysis(run_folder, data, config)
        stats['fano_factor'] = fano_stats
        
    except Exception as e:
        print(f"   Error creating Fano factor analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary statistics
    if stats:
        print("\n   Temporal Evolution Statistics:")
        if 'selectivity_evolution' in stats:
            print(f"   Selectivity: {stats['selectivity_evolution']['first_epoch_mean']:.3f} -> {stats['selectivity_evolution']['last_epoch_mean']:.3f}")
        if 'sparsity_evolution' in stats:
            print(f"   Sparsity: {stats['sparsity_evolution']['first_epoch_mean']:.3f} -> {stats['sparsity_evolution']['last_epoch_mean']:.3f}")
        if 'rate_matrix_change' in stats:
            print(f"   Mean firing rate: {stats['rate_matrix_change']['first_mean']:.2f} -> {stats['rate_matrix_change']['last_mean']:.2f} Hz")
        if 'fano_factor' in stats:
            print(f"   Fano Factor: mean = {stats['fano_factor'].get('mean_ff', 'N/A'):.3f}")
    
    return stats


def plot_fano_factor_analysis(run_folder, data, config):
    """
    Calculate and plot Fano factor analysis across epochs.
    
    Fano Factor = Variance / Mean
    Measures stability of neural responses across epochs.
    
    Creates a comprehensive figure with:
    1. Heatmap: FF per neuron × stimulus
    2. Distribution: histogram of all FF values
    3. Per neuron: average FF per neuron (sorted)
    4. Per stimulus: average FF per stimulus (sorted)
    
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
    dict : Statistics about Fano factor
    """
    temporal_folder = os.path.join(run_folder, "Figures", "temporal_evolution")
    os.makedirs(temporal_folder, exist_ok=True)
    
    # Extract individual rate matrices (one per epoch)
    individual_rate_matrices = data.get('individual_rate_matrices', [])
    
    if len(individual_rate_matrices) < 2:
        print("   Warning: Need at least 2 epochs to calculate Fano factor")
        return {}
    
    # Stack all rate matrices into a 3D array: (epochs, neurons, stimuli)
    rate_matrices_3d = np.array(individual_rate_matrices)  # Shape: (n_epochs, n_neurons, n_stimuli)
    n_epochs, n_neurons, n_stimuli = rate_matrices_3d.shape
    
    print(f"   Calculating Fano factor for {n_neurons} neurons, {n_stimuli} stimuli, across {n_epochs} epochs...")
    
    # Calculate Fano factor per neuron per stimulus
    # FF(neuron_i, stimulus_j) = Var(rate across epochs) / Mean(rate across epochs)
    fano_matrix = np.zeros((n_neurons, n_stimuli))
    
    for i in range(n_neurons):
        for j in range(n_stimuli):
            rates_across_epochs = rate_matrices_3d[:, i, j]
            mean_rate = np.mean(rates_across_epochs)
            var_rate = np.var(rates_across_epochs)
            
            if mean_rate > 0:
                fano_matrix[i, j] = var_rate / mean_rate
            else:
                fano_matrix[i, j] = np.nan  # No activity = undefined FF
    
    # Calculate average FF per neuron (across stimuli)
    fano_per_neuron = np.nanmean(fano_matrix, axis=1)  # Average across stimuli
    
    # Calculate average FF per stimulus (across neurons)
    fano_per_stimulus = np.nanmean(fano_matrix, axis=0)  # Average across neurons
    
    # Get selectivity and sparsity indices for sorting
    # Use the last epoch's rate matrix or average across epochs
    final_rate_matrix = np.mean(rate_matrices_3d, axis=0)  # Average across epochs
    import sparsity_and_selectivity as ss
    selectivity_vec = np.array(ss.selectivity_indices_vectorized(final_rate_matrix))  # Per neuron
    sparsity_vec = np.array(ss.sparsity_indices_vectorized(final_rate_matrix))  # Per stimulus
    
    # Sort neurons by selectivity index
    neuron_order_by_selectivity = np.argsort(selectivity_vec)
    
    # Sort stimuli by sparsity index
    stimulus_order_by_sparsity = np.argsort(sparsity_vec)
    
    # Get all valid FF values for distribution
    valid_ff = fano_matrix[~np.isnan(fano_matrix)]
    
    # Calculate mean firing rate per neuron (across all stimuli and epochs)
    mean_firing_rate_per_neuron = np.mean(final_rate_matrix, axis=1)
    
    # Create 2x3 figure (6 panels)
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Fano Factor Analysis: Response Stability Across Epochs', 
                 fontsize=16, fontweight='bold')
    
    # ===== 1. Heatmap: FF per neuron × stimulus =====
    ax = axes[0, 0]
    # Order neurons by selectivity, stimuli by sparsity
    fano_matrix_ordered = fano_matrix[np.ix_(neuron_order_by_selectivity, stimulus_order_by_sparsity)]
    
    # Use percentile clamping for heatmap (to exclude outliers)
    vmin = np.nanpercentile(valid_ff, 1)
    vmax = np.nanpercentile(valid_ff, 99)
    im = ax.imshow(fano_matrix_ordered, cmap='viridis', aspect='auto', 
                  vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xlabel('Stimulus Index (sorted by sparsity)')
    ax.set_ylabel('Neuron Index (sorted by selectivity)')
    ax.set_title('Fano Factor Heatmap\n(Neuron × Stimulus)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Fano Factor')
    
    # Add E/I boundary if available
    N_rec_E = config.get('N_rec_E', 100)
    n_E_in_ordered = np.sum(neuron_order_by_selectivity < N_rec_E)
    if n_E_in_ordered > 0 and n_E_in_ordered < n_neurons:
        ax.axhline(y=n_E_in_ordered - 0.5, color='cyan', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label='E/I boundary')
        ax.legend(loc='upper right', fontsize=8)
    
    # ===== 2. Distribution: Histogram of all FF values =====
    ax = axes[0, 1]
    ax.hist(valid_ff, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    # Only show Poisson line if data range includes it
    if np.nanmax(valid_ff) >= 1.0:
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, 
                  label='Poisson (FF=1)', alpha=0.8)
    ax.set_xlabel('Fano Factor')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Fano Factors\n(All Neuron-Stimulus Pairs)')
    if np.nanmax(valid_ff) >= 1.0:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_ff = np.nanmean(valid_ff)
    median_ff = np.nanmedian(valid_ff)
    std_ff = np.nanstd(valid_ff)
    stats_text = f'Mean: {mean_ff:.3f}\nMedian: {median_ff:.3f}\nStd: {std_ff:.3f}\nn={len(valid_ff)}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           verticalalignment='top', horizontalalignment='right', fontsize=10)
    
    # ===== 3. Per Neuron: Average FF per neuron (sorted by selectivity) =====
    ax = axes[1, 0]
    sorted_neurons = np.arange(n_neurons)
    sorted_ff_neuron = fano_per_neuron[neuron_order_by_selectivity]
    
    # Color by E/I
    colors_neuron = ['red' if neuron_order_by_selectivity[i] < N_rec_E else 'blue' 
                     for i in range(n_neurons)]
    
    ax.scatter(sorted_neurons, sorted_ff_neuron, c=colors_neuron, 
              alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    # Only show Poisson line if data range includes it
    if np.nanmax(sorted_ff_neuron) >= 1.0:
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                  label='Poisson (FF=1)', alpha=0.8)
    ax.set_xlabel('Neuron Index (sorted by selectivity)')
    ax.set_ylabel('Average Fano Factor')
    ax.set_title('Average Fano Factor per Neuron\n(sorted by selectivity, red=E, blue=I)')
    # Don't set y-axis limits - let it auto-scale to data range
    if np.nanmax(sorted_ff_neuron) >= 1.0:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== 4. Per Stimulus: Average FF per stimulus (sorted by sparsity) =====
    ax = axes[1, 1]
    sorted_stimuli = np.arange(n_stimuli)
    sorted_ff_stimulus = fano_per_stimulus[stimulus_order_by_sparsity]
    
    ax.scatter(sorted_stimuli, sorted_ff_stimulus, 
              alpha=0.6, s=30, color='purple', edgecolors='black', linewidth=0.5)
    # Only show Poisson line if data range includes it
    if np.nanmax(sorted_ff_stimulus) >= 1.0:
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                  label='Poisson (FF=1)', alpha=0.8)
    ax.set_xlabel('Stimulus Index (sorted by sparsity)')
    ax.set_ylabel('Average Fano Factor')
    ax.set_title('Average Fano Factor per Stimulus\n(sorted by sparsity)')
    # Don't set y-axis limits - let it auto-scale to data range
    if np.nanmax(sorted_ff_stimulus) >= 1.0:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== 5. Firing Rate vs Fano Factor Correlation =====
    ax = axes[0, 2]
    
    # Color by E/I
    colors_neuron = ['red' if i < N_rec_E else 'blue' for i in range(n_neurons)]
    
    ax.scatter(mean_firing_rate_per_neuron, fano_per_neuron, 
              c=colors_neuron, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    # Calculate correlation
    from scipy.stats import pearsonr
    valid_mask = ~(np.isnan(mean_firing_rate_per_neuron) | np.isnan(fano_per_neuron))
    if np.sum(valid_mask) > 2:
        firing_rate_valid = mean_firing_rate_per_neuron[valid_mask]
        fano_valid = fano_per_neuron[valid_mask]
        corr, p_value = pearsonr(firing_rate_valid, fano_valid)
        
        # Fit linear regression line
        z = np.polyfit(firing_rate_valid, fano_valid, 1)
        p = np.poly1d(z)
        x_line = np.linspace(np.min(firing_rate_valid), np.max(firing_rate_valid), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
               label=f'Fit: r={corr:.3f}')
        
        # Add correlation text
        stats_text = f'r = {corr:.3f}\np = {p_value:.3e}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
               verticalalignment='top', horizontalalignment='left', fontsize=10)
    
    ax.set_xlabel('Mean Firing Rate (Hz)')
    ax.set_ylabel('Average Fano Factor')
    ax.set_title('Firing Rate vs Fano Factor\n(per neuron, red=E, blue=I)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Remove the 6th panel (axes[1, 2]) - leave it empty or hide it
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(temporal_folder, 'fano_factor_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      Saved: {save_path}")
    
    # Calculate and print statistics
    stats = {
        'mean_ff': float(np.nanmean(valid_ff)),
        'median_ff': float(np.nanmedian(valid_ff)),
        'std_ff': float(np.nanstd(valid_ff)),
        'min_ff': float(np.nanmin(valid_ff)),
        'max_ff': float(np.nanmax(valid_ff)),
        'mean_ff_per_neuron': float(np.nanmean(fano_per_neuron)),
        'mean_ff_per_stimulus': float(np.nanmean(fano_per_stimulus)),
        'n_valid_pairs': len(valid_ff),
        'n_epochs': n_epochs,
        'n_neurons': n_neurons,
        'n_stimuli': n_stimuli
    }
    
    print(f"   Fano Factor Statistics:")
    print(f"      Mean: {stats['mean_ff']:.3f} ± {stats['std_ff']:.3f}")
    print(f"      Median: {stats['median_ff']:.3f}")
    print(f"      Range: [{stats['min_ff']:.3f}, {stats['max_ff']:.3f}]")
    print(f"      Mean per neuron: {stats['mean_ff_per_neuron']:.3f}")
    print(f"      Mean per stimulus: {stats['mean_ff_per_stimulus']:.3f}")
    
    return stats


if __name__ == "__main__":
    """Command-line interface for standalone temporal evolution analysis"""
    import pickle
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python temporal_evolution.py <run_folder>")
        sys.exit(1)
    
    run_folder = sys.argv[1]
    data_file = os.path.join(run_folder, "Data", "simulation_data.pkl")
    
    if not os.path.exists(data_file):
        print(f"Error: No data file found at {data_file}")
        sys.exit(1)
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    config = data.get('config', {})
    stats = generate_temporal_evolution_plots(run_folder, data, config)
    print("\nTemporal evolution analysis complete!")

