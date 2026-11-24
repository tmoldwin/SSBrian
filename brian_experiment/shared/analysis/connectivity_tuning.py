"""
Connectivity Tuning Analysis Module

This module analyzes functional connectivity patterns, specifically the
Co-tuning Connectivity Index (CCI) which measures whether neurons with
similar response profiles are preferentially connected.

Creates plots in the connectivity_tuning/ subfolder.

Functions:
- generate_connectivity_tuning_plots: Main entry point for CCI analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import sparsity_and_selectivity as ss


def _normalize_tuning_rows(R):
    """
    Row-wise mean-center and L2-normalize tuning vectors (neurons × stimuli).
    Returns tilde R with safe handling for zero norms.
    """
    R = np.asarray(R)
    means = np.mean(R, axis=1, keepdims=True)
    centered = R - means
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    R_tilde = centered / safe_norms
    return R_tilde


def compute_cci(weight_matrix, response_matrix, p=1.0, excitatory_count=None):
    """
    Compute Co-tuning Connectivity Index (CCI) per postsynaptic neuron.
    
    CCI measures how much neurons with similar response profiles are connected.
    
    Parameters:
    -----------
    weight_matrix : np.ndarray
        W (neurons × neurons), where W[i, j] is i → j synapse
    response_matrix : np.ndarray
        R (neurons × stimuli), rows are neurons, cols are stimuli
    p : float
        Exponent for |W|^p normalization (p=1 default)
    excitatory_count : int, optional
        Number of excitatory neurons (first N_exc are E)
    
    Returns:
    --------
    dict with cci, cci_E, cci_I, sigma (cosine similarities), alpha (normalized weights)
    """
    print("\n   Computing Co-tuning Connectivity Index (CCI)...")
    W = np.asarray(weight_matrix)
    R = np.asarray(response_matrix)
    assert W.shape[0] == W.shape[1] == R.shape[0], "W must be (N×N) and R must be (N×S) with same N"
    
    # Normalize tuning vectors (rows)
    R_tilde = _normalize_tuning_rows(R)
    # Cosine similarity between all neuron tuning vectors (N × N)
    sigma = R_tilde @ R_tilde.T
    
    # Compute alpha: normalized |W|^p per postsynaptic column j
    absWp = np.abs(W) ** float(p)
    col_sums = absWp.sum(axis=0, keepdims=True)
    safe_col_sums = np.where(col_sums > 0, col_sums, 1.0)
    alpha = absWp / safe_col_sums  # columns sum to 1
    
    # Sign of W (Dale's law respected implicitly)
    signW = np.sign(W)
    
    # CCI_j (combined, signed): Σ_i alpha_ij * sign(W_ij) * sigma_ij
    cci = (alpha * signW * sigma).sum(axis=0)
    
    # Optional E/I split
    N = W.shape[0]
    if excitatory_count is None:
        excitatory_count = N // 2  # Default to half
    exc_mask = np.zeros(N, dtype=bool)
    exc_mask[:excitatory_count] = True
    inh_mask = ~exc_mask
    
    # For separated components, use absolute value of sigma: pure weighted correlation magnitude
    cci_E = (alpha[exc_mask, :] * np.abs(sigma[exc_mask, :])).sum(axis=0)
    cci_I = (alpha[inh_mask, :] * np.abs(sigma[inh_mask, :])).sum(axis=0)
    
    print(f"      CCI computed for {N} neurons (p={p})")
    print(f"      CCI range: [{np.min(cci):.3f}, {np.max(cci):.3f}], mean={np.mean(cci):.3f}")
    
    return {
        'cci': cci,
        'cci_E': cci_E,
        'cci_I': cci_I,
        'sigma': sigma,
        'alpha': alpha
    }


def _plot_cci_vs_selectivity(ax, cci, selectivity, title_prefix='CCI'):
    """Scatter CCI vs selectivity with Pearson/Spearman and regression line, removing outliers."""
    cci = np.asarray(cci)
    selectivity = np.asarray(selectivity)
    
    # Drop any non-finite
    mask = np.isfinite(cci) & np.isfinite(selectivity)
    cci_c = cci[mask]
    sel_c = selectivity[mask]
    
    # Remove outliers using IQR method
    if len(cci_c) > 0:
        # Remove outliers in CCI
        q1_cci, q3_cci = np.percentile(cci_c, [25, 75])
        iqr_cci = q3_cci - q1_cci
        lower_cci = q1_cci - 1.5 * iqr_cci
        upper_cci = q3_cci + 1.5 * iqr_cci
        
        # Remove outliers in selectivity
        q1_sel, q3_sel = np.percentile(sel_c, [25, 75])
        iqr_sel = q3_sel - q1_sel
        lower_sel = q1_sel - 1.5 * iqr_sel
        upper_sel = q3_sel + 1.5 * iqr_sel
        
        # Keep only non-outliers
        outlier_mask = (cci_c >= lower_cci) & (cci_c <= upper_cci) & (sel_c >= lower_sel) & (sel_c <= upper_sel)
        cci_c = cci_c[outlier_mask]
        sel_c = sel_c[outlier_mask]
    
    ax.scatter(cci_c, sel_c, s=16, alpha=0.6, color='tab:blue')
    
    # Regression
    if len(sel_c) > 1:
        z = np.polyfit(cci_c, sel_c, 1)
        p = np.poly1d(z)
        xs = np.linspace(cci_c.min(), cci_c.max(), 100)
        ax.plot(xs, p(xs), linestyle='--', color='red', linewidth=2)
        
        # Correlations
        pearson_r, pearson_p = stats.pearsonr(cci_c, sel_c)
        spearman_r, spearman_p = stats.spearmanr(cci_c, sel_c)
        ax.set_title(f'Selectivity vs {title_prefix}\nPearson r={pearson_r:.3f} (p={pearson_p:.1e}), Spearman rho={spearman_r:.3f} (p={spearman_p:.1e})', 
                    fontsize=10)
    ax.set_xlabel(title_prefix)
    ax.set_ylabel('Selectivity Index')
    ax.grid(True, alpha=0.3)


def generate_cci_analysis(run_folder, data, config):
    """
    Generate Co-tuning Connectivity Index (CCI) analysis.
    
    CCI measures whether neurons with similar response profiles are preferentially connected.
    
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
    dict : CCI statistics
    """
    print("\n   Creating Co-tuning Connectivity Index analysis...")
    
    # Create subfolder
    connectivity_folder = os.path.join(run_folder, "Figures", "connectivity_tuning")
    os.makedirs(connectivity_folder, exist_ok=True)
    
    # Get data
    weight_rec = data.get('weight_rec')
    rate_matrices_arr = data.get('rate_matrices_arr', [])
    selectivity_vec_arr = data.get('selectivity_vec_arr')
    
    N_rec_E = config.get('N_rec_E', 100)
    
    if weight_rec is None or len(rate_matrices_arr) == 0:
        print("      WARNING: Missing weight or rate data for CCI analysis")
        return {}
    
    try:
        # Average rate matrix across epochs
        averaged_rate_matrix = np.mean(rate_matrices_arr, axis=0)  # (neurons, stimuli)
        
        # Get final selectivity
        if selectivity_vec_arr is not None and len(selectivity_vec_arr) > 0:
            selectivity_vec = selectivity_vec_arr[-1, :]
        else:
            # Compute from rate matrix
            _, selectivity_vec = ss.create_vecs(averaged_rate_matrix)
            selectivity_vec = np.array(selectivity_vec)
        
        # Compute CCI
        cci_data = compute_cci(weight_rec, averaged_rate_matrix, p=1.0, excitatory_count=N_rec_E)
        
        # Create 4-panel figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Co-tuning Connectivity Index (CCI) Analysis', fontsize=16, fontweight='bold')
        
        # Panel 1: Combined CCI_E and CCI_I distributions
        try:
            from scipy.stats import gaussian_kde
            # Plot CCI_E
            kde_E = gaussian_kde(cci_data['cci_E'])
            x_E = np.linspace(np.min(cci_data['cci_E']), np.max(cci_data['cci_E']), 200)
            y_E = kde_E(x_E)
            ax1.plot(x_E, y_E, color='red', linewidth=2, label='CCI_E (Excitatory)')
            ax1.fill_between(x_E, y_E, alpha=0.3, color='red')
            
            # Plot CCI_I
            kde_I = gaussian_kde(cci_data['cci_I'])
            x_I = np.linspace(np.min(cci_data['cci_I']), np.max(cci_data['cci_I']), 200)
            y_I = kde_I(x_I)
            ax1.plot(x_I, y_I, color='blue', linewidth=2, label='CCI_I (Inhibitory)')
            ax1.fill_between(x_I, y_I, alpha=0.3, color='blue')
            
            ax1.set_xlabel('CCI', fontsize=11)
            ax1.set_ylabel('Density', fontsize=11)
            ax1.legend()
        except:
            # Fallback to histograms
            ax1.hist(cci_data['cci_E'], bins=30, alpha=0.5, color='red', label='CCI_E', density=True)
            ax1.hist(cci_data['cci_I'], bins=30, alpha=0.5, color='blue', label='CCI_I', density=True)
            ax1.set_xlabel('CCI', fontsize=11)
            ax1.set_ylabel('Density', fontsize=11)
            ax1.legend()
        ax1.set_title('CCI Distributions: Excitatory vs Inhibitory', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: CCI_E vs selectivity
        _plot_cci_vs_selectivity(ax2, cci_data['cci_E'], selectivity_vec, 'CCI_E (Excitatory)')
        
        # Panel 3: CCI_I vs selectivity
        _plot_cci_vs_selectivity(ax3, cci_data['cci_I'], selectivity_vec, 'CCI_I (Inhibitory)')
        
        # Panel 4: Net CCI (E - I) vs selectivity
        cci_net = cci_data['cci_E'] - cci_data['cci_I']
        _plot_cci_vs_selectivity(ax4, cci_net, selectivity_vec, 'CCI_E - CCI_I (Net)')
        
        plt.tight_layout()
        save_path = os.path.join(connectivity_folder, 'cotuning_connectivity_index.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      Saved: cotuning_connectivity_index.png")
        
        return {
            'cci_mean': np.mean(cci_data['cci']),
            'cci_E_mean': np.mean(cci_data['cci_E']),
            'cci_I_mean': np.mean(cci_data['cci_I']),
        }
        
    except Exception as e:
        print(f"      Error in CCI analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}


def generate_connectivity_tuning_plots(run_folder, data, config):
    """
    Generate functional connectivity tuning analysis (CCI).
    
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
    dict : CCI statistics
    """
    print("\n" + "="*60)
    print("CONNECTIVITY TUNING ANALYSIS (CCI)")
    print("="*60)
    
    # Run CCI analysis
    cci_stats = generate_cci_analysis(run_folder, data, config)
    
    return cci_stats


# DEPRECATED: Use network_structure.py and generate_connectivity_tuning_plots instead
def generate_connectivity_plots(run_folder, data, config):
    """
    DEPRECATED: This function combined weight matrix visualization with CCI analysis.
    Use network_structure.py for weight matrices and generate_connectivity_tuning_plots for CCI.
    
    For backward compatibility, this calls both new functions.
    """
    import network_structure
    print("\nWARNING: generate_connectivity_plots is deprecated. Use separate modules.")
    
    # Call both new modules for backward compatibility
    weight_stats = network_structure.generate_network_structure_plots(run_folder, data, config)
    cci_stats = generate_connectivity_tuning_plots(run_folder, data, config)
    
    # Combine stats
    if cci_stats:
        weight_stats['cci'] = cci_stats
    return weight_stats


# Original implementation kept for reference (will be removed in future)
def _OLD_generate_connectivity_plots(run_folder, data, config):
    """OLD VERSION - DO NOT USE"""
    print("\n" + "="*60)
    print("CONNECTIVITY TUNING ANALYSIS (OLD)")
    print("="*60)
    
    # Create subfolder
    connectivity_folder = os.path.join(run_folder, "Figures", "connectivity_tuning")
    os.makedirs(connectivity_folder, exist_ok=True)
    
    # Extract data
    input_weights = data.get('weight_poisson')  # Input weights
    weight_rec = data.get('weight_rec')  # Recurrent weights
    N_rec_E = config.get('N_rec_E', 100)
    N_rec_I = config.get('N_rec_I', 100)
    
    if input_weights is None or weight_rec is None:
        print("   WARNING: Weight data not available, skipping connectivity plots")
        return {}
    
    try:
        # Create figure with 2x2 subplots
        fig1, axs1 = plt.subplots(2, 2, figsize=(14, 12))
        fig1.suptitle('Weight Matrices and Distributions', fontsize=16, fontweight='bold')
        
        # ===== INPUT WEIGHTS =====
        # Top left: Input weight matrix
        im1 = axs1[0, 0].imshow(input_weights.T, aspect='auto', cmap='RdBu_r',
                               vmin=-np.max(np.abs(input_weights)), 
                               vmax=np.max(np.abs(input_weights)))
        
        # Add E/I boundary line for target neurons
        axs1[0, 0].axhline(y=N_rec_E-0.5, color='yellow', linewidth=2, alpha=0.8)
        axs1[0, 0].text(input_weights.shape[1]//2, N_rec_E//2, 'E neurons', 
                       ha='center', va='center', color='red', fontsize=10, fontweight='bold')
        axs1[0, 0].text(input_weights.shape[1]//2, N_rec_E + N_rec_I//2, 'I neurons', 
                       ha='center', va='center', color='blue', fontsize=10, fontweight='bold')
        
        axs1[0, 0].set_title('Input Weight Matrix (Input→Rec)')
        axs1[0, 0].set_xlabel('Input Dimension')
        axs1[0, 0].set_ylabel('Target Neuron')
        plt.colorbar(im1, ax=axs1[0, 0])
        
        # Top right: Input weight distribution
        E_to_E_weights = input_weights[:N_rec_E, :].flatten()
        I_to_E_weights = input_weights[N_rec_E:, :].flatten()
        
        axs1[0, 1].hist(E_to_E_weights, bins=30, alpha=0.7, color='red', 
                       label='E←input', density=True, histtype='step', linewidth=2)
        axs1[0, 1].hist(I_to_E_weights, bins=30, alpha=0.7, color='blue', 
                       label='I←input', density=True, histtype='step', linewidth=2)
        axs1[0, 1].axvline(0, color='black', linestyle='--', alpha=0.5)
        axs1[0, 1].set_title('Input Weight Distribution')
        axs1[0, 1].set_xlabel('Weight Value')
        axs1[0, 1].set_ylabel('Density')
        axs1[0, 1].legend()
        axs1[0, 1].set_xlim(-np.max(np.abs(input_weights)), np.max(np.abs(input_weights)))
        axs1[0, 1].grid(True, alpha=0.3)
        
        # ===== RECURRENT WEIGHTS =====
        # Bottom left: Recurrent weight matrix with E/I structure
        weight_matrix = weight_rec.T
        im2 = axs1[1, 0].imshow(weight_matrix, aspect='auto', cmap='RdBu_r',
                               vmin=-np.max(np.abs(weight_matrix)), 
                               vmax=np.max(np.abs(weight_matrix)))
        
        # Add E/I boundary lines
        axs1[1, 0].axhline(y=N_rec_E-0.5, color='red', linewidth=2, alpha=0.8)
        axs1[1, 0].axvline(x=N_rec_E-0.5, color='red', linewidth=2, alpha=0.8)
        
        # Add text labels for E/I regions
        axs1[1, 0].text(N_rec_E//2, N_rec_E//2, 'E→E', ha='center', va='center', 
                       color='red', fontsize=12, fontweight='bold', alpha=0.8)
        axs1[1, 0].text(N_rec_E + N_rec_I//2, N_rec_E//2, 'I→E', ha='center', va='center', 
                       color='blue', fontsize=12, fontweight='bold', alpha=0.8)
        axs1[1, 0].text(N_rec_E//2, N_rec_E + N_rec_I//2, 'E→I', ha='center', va='center', 
                       color='red', fontsize=12, fontweight='bold', alpha=0.8)
        axs1[1, 0].text(N_rec_E + N_rec_I//2, N_rec_E + N_rec_I//2, 'I→I', ha='center', va='center', 
                       color='blue', fontsize=12, fontweight='bold', alpha=0.8)
        
        axs1[1, 0].set_title('Recurrent Weight Matrix (E=Red, I=Blue)')
        axs1[1, 0].set_xlabel('Source Neuron')
        axs1[1, 0].set_ylabel('Target Neuron')
        plt.colorbar(im2, ax=axs1[1, 0])
        
        # Bottom right: Recurrent weight distribution with E/I separation
        E_to_E_rec = weight_rec[:N_rec_E, :N_rec_E].flatten()
        E_to_I_rec = weight_rec[:N_rec_E, N_rec_E:].flatten()
        I_to_E_rec = weight_rec[N_rec_E:, :N_rec_E].flatten()
        I_to_I_rec = weight_rec[N_rec_E:, N_rec_E:].flatten()
        
        axs1[1, 1].hist(E_to_E_rec, bins=30, alpha=0.7, color='red', 
                       label='E→E', density=True, histtype='step', linewidth=2)
        axs1[1, 1].hist(E_to_I_rec, bins=30, alpha=0.7, color='orange', 
                       label='E→I', density=True, histtype='step', linewidth=2)
        axs1[1, 1].hist(I_to_E_rec, bins=30, alpha=0.7, color='blue', 
                       label='I→E', density=True, histtype='step', linewidth=2)
        axs1[1, 1].hist(I_to_I_rec, bins=30, alpha=0.7, color='purple', 
                       label='I→I', density=True, histtype='step', linewidth=2)
        axs1[1, 1].axvline(0, color='black', linestyle='--', alpha=0.5)
        axs1[1, 1].set_title('Recurrent Weight Distribution')
        axs1[1, 1].set_xlabel('Weight Value')
        axs1[1, 1].set_ylabel('Density')
        axs1[1, 1].legend()
        axs1[1, 1].set_xlim(-np.max(np.abs(weight_rec)), np.max(np.abs(weight_rec)))
        axs1[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(connectivity_folder, 'weight_matrices_and_distributions.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"   Weight matrices saved to: {save_path}")
        
        # Calculate and return statistics
        stats = {
            'input_weights': {
                'E_mean': np.mean(E_to_E_weights),
                'E_std': np.std(E_to_E_weights),
                'I_mean': np.mean(I_to_E_weights),
                'I_std': np.std(I_to_E_weights),
            },
            'recurrent_weights': {
                'E_to_E_mean': np.mean(E_to_E_rec),
                'E_to_E_std': np.std(E_to_E_rec),
                'E_to_I_mean': np.mean(E_to_I_rec),
                'E_to_I_std': np.std(E_to_I_rec),
                'I_to_E_mean': np.mean(I_to_E_rec),
                'I_to_E_std': np.std(I_to_E_rec),
                'I_to_I_mean': np.mean(I_to_I_rec),
                'I_to_I_std': np.std(I_to_I_rec),
            }
        }
        
        print("\n   Weight Statistics:")
        print(f"   Input E mean: {stats['input_weights']['E_mean']:.4f} ± {stats['input_weights']['E_std']:.4f}")
        print(f"   Input I mean: {stats['input_weights']['I_mean']:.4f} ± {stats['input_weights']['I_std']:.4f}")
        print(f"   Recurrent E->E: {stats['recurrent_weights']['E_to_E_mean']:.4f} ± {stats['recurrent_weights']['E_to_E_std']:.4f}")
        print(f"   Recurrent I->E: {stats['recurrent_weights']['I_to_E_mean']:.4f} ± {stats['recurrent_weights']['I_to_E_std']:.4f}")
        
        # Add CCI analysis
        cci_stats = generate_cci_analysis(run_folder, data, config)
        if cci_stats:
            stats['cci'] = cci_stats
        
        return stats
        
    except Exception as e:
        print(f"   Error in connectivity analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    """Command-line interface for standalone connectivity analysis"""
    import pickle
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python connectivity_tuning.py <run_folder>")
        sys.exit(1)
    
    run_folder = sys.argv[1]
    data_file = os.path.join(run_folder, "Data", "simulation_data.pkl")
    
    if not os.path.exists(data_file):
        print(f"Error: No data file found at {data_file}")
        sys.exit(1)
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    config = data.get('config', {})
    stats = generate_connectivity_plots(run_folder, data, config)
    print("\nConnectivity analysis complete!")

