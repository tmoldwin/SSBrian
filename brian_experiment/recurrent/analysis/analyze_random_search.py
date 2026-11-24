"""
Analyze random search results:
- Correlations between all pairs of connection types (EE, EI, IE, II)
- Ratios vs firing rates
"""

import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

def find_latest_random_search_results():
    """Find the most recent random_search_summary.pkl file across random search runs"""
    base_folder = os.path.join(os.path.dirname(__file__), "Runs", "random_search")
    search_pattern = os.path.join(base_folder, "random_search_EtoI_ItoE_E*_I*", "*", "random_search_summary.pkl")
    summary_files = glob.glob(search_pattern)
    
    if not summary_files:
        raise FileNotFoundError(f"No random_search_summary.pkl files found under {base_folder}")
    
    latest_file = max(summary_files, key=os.path.getmtime)
    latest_path = os.path.dirname(latest_file)
    return latest_file, latest_path

def analyze_connection_correlations(all_results, save_folder):
    """Analyze correlations between all pairs of connection types"""
    print("\nAnalyzing correlations between connection types...")
    
    # Filter successful results
    successful_results = [r for r in all_results if r.get('success', False)]
    
    if len(successful_results) < 3:
        print("  Not enough successful runs for correlation analysis")
        return
    
    # Extract connection scales
    ee_scales = np.array([r['e_to_e_scale'] for r in successful_results])
    ei_scales = np.array([r['e_to_i_scale'] for r in successful_results])
    ie_scales = np.array([r['i_to_e_scale'] for r in successful_results])
    ii_scales = np.array([r['i_to_i_scale'] for r in successful_results])
    
    # Compute correlations (use log scale for better visualization)
    log_ee = np.log10(ee_scales + 1e-6)
    log_ei = np.log10(ei_scales + 1e-6)
    log_ie = np.log10(ie_scales + 1e-6)
    log_ii = np.log10(ii_scales + 1e-6)
    
    # Compute all pairwise correlations
    connection_names = ['E->E', 'E->I', 'I->E', 'I->I']
    connection_data = [log_ee, log_ei, log_ie, log_ii]
    connection_data_linear = [ee_scales, ei_scales, ie_scales, ii_scales]
    
    # Correlation matrix
    corr_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr, pval = pearsonr(connection_data[i], connection_data[j])
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
    
    # Print correlation matrix
    print("\nCorrelation Matrix (log scale):")
    print("        E->E      E->I      I->E      I->I")
    for i, name in enumerate(connection_names):
        print(f"{name:4}  ", end="")
        for j in range(4):
            print(f"{corr_matrix[i, j]:7.3f}  ", end="")
        print()
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(connection_names)
    ax.set_yticklabels(connection_names)
    ax.set_title('Correlation Matrix: Connection Type Scales (log scale)', fontsize=14, fontweight='bold')
    
    # Add correlation values to cells
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Pearson Correlation')
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, "connection_correlation_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    # Create scatter plot matrix
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Pairwise Scatter Plots: Connection Type Scales (log scale)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(connection_data[i], bins=15, alpha=0.7, edgecolor='black')
                ax.set_ylabel('Frequency', fontsize=9)
            else:
                # Off-diagonal: scatter plot
                ax.scatter(connection_data[j], connection_data[i], alpha=0.6, s=30)
                
                # Add correlation text
                corr = corr_matrix[i, j]
                ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Labels
            if i == 3:
                ax.set_xlabel(connection_names[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(connection_names[i], fontsize=10)
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(save_folder, "connection_scatter_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

def analyze_ratios_vs_rates(all_results, save_folder):
    """Comprehensive analysis of ALL ratios vs firing rates"""
    print("\nAnalyzing comprehensive ratios vs firing rates...")
    
    # Filter successful results
    successful_results = [r for r in all_results if r.get('success', False)]
    
    if len(successful_results) == 0:
        print("  No successful runs to analyze")
        return
    
    # Extract data
    e_rates = np.array([r['e_rate'] for r in successful_results])
    i_rates = np.array([r['i_rate'] for r in successful_results])
    ee_scales = np.array([r['e_to_e_scale'] for r in successful_results])
    ei_scales = np.array([r['e_to_i_scale'] for r in successful_results])
    ie_scales = np.array([r['i_to_e_scale'] for r in successful_results])
    ii_scales = np.array([r['i_to_i_scale'] for r in successful_results])
    mean_e_weights = np.array([r['mean_e_weight'] for r in successful_results])
    mean_i_weights = np.array([np.abs(r['mean_i_weight']) for r in successful_results])
    
    # Compute ALL pairwise individual connection type ratios (12 total: 4 choose 2, each direction)
    individual_ratios = {
        'EE/EI': ee_scales / (ei_scales + 1e-6),
        'EI/EE': ei_scales / (ee_scales + 1e-6),
        'EE/IE': ee_scales / (ie_scales + 1e-6),
        'IE/EE': ie_scales / (ee_scales + 1e-6),
        'EE/II': ee_scales / (ii_scales + 1e-6),
        'II/EE': ii_scales / (ee_scales + 1e-6),
        'EI/IE': ei_scales / (ie_scales + 1e-6),
        'IE/EI': ie_scales / (ei_scales + 1e-6),
        'EI/II': ei_scales / (ii_scales + 1e-6),
        'II/EI': ii_scales / (ei_scales + 1e-6),
        'IE/II': ie_scales / (ii_scales + 1e-6),
        'II/IE': ii_scales / (ie_scales + 1e-6),
    }
    
    # Compute signed sums for ratios (E connections +, I connections -)
    signed_weights_to_e = ee_scales - ie_scales  # EE - IE (net to E)
    signed_weights_to_i = ei_scales - ii_scales  # EI - II (net to I)
    signed_weights_from_e = ee_scales + ei_scales  # EE + EI (all from E, positive)
    signed_weights_from_i = -(ie_scales + ii_scales)  # -(IE + II) (all from I, negative)
    
    sum_ratios = {
        '(EE-IE)/(EI-II)': signed_weights_to_e / (signed_weights_to_i + 1e-6),  # Net TO E / Net TO I
        '(EI-II)/(EE-IE)': signed_weights_to_i / (signed_weights_to_e + 1e-6),  # Net TO I / Net TO E
        '(EE+EI)/-(IE+II)': signed_weights_from_e / (signed_weights_from_i + 1e-6),  # FROM E / FROM I (signed)
        '-(IE+II)/(EE+EI)': signed_weights_from_i / (signed_weights_from_e + 1e-6),  # FROM I / FROM E (signed)
        '(EE-IE)/(EE+EI)': signed_weights_to_e / (signed_weights_from_e + 1e-6),  # Net TO E / FROM E
        '(EE+EI)/(EE-IE)': signed_weights_from_e / (signed_weights_to_e + 1e-6),  # FROM E / Net TO E
        '(EI-II)/-(IE+II)': signed_weights_to_i / (signed_weights_from_i + 1e-6),  # Net TO I / FROM I (signed)
        '-(IE+II)/(EI-II)': signed_weights_from_i / (signed_weights_to_i + 1e-6),  # FROM I / Net TO I (signed)
        '(EE-IE)/-(IE+II)': signed_weights_to_e / (signed_weights_from_i + 1e-6),  # Net TO E / FROM I (signed)
        '-(IE+II)/(EE-IE)': signed_weights_from_i / (signed_weights_to_e + 1e-6),  # FROM I / Net TO E (signed)
        '(EE+EI)/(EI-II)': signed_weights_from_e / (signed_weights_to_i + 1e-6),  # FROM E / Net TO I
        '(EI-II)/(EE+EI)': signed_weights_to_i / (signed_weights_from_e + 1e-6),  # Net TO I / FROM E
    }
    
    # E/I rate ratio
    ei_rate_ratio = e_rates / (i_rates + 1e-6)
    mean_weight_ratio = mean_e_weights / (mean_i_weights + 1e-6)
    
    # Create comprehensive figure
    # Part 1: Individual connection type ratios (12 ratios, 2 rates each = 24 plots)
    # Organize as 6 rows x 4 columns (each row has 2 ratio pairs, 2 columns for E and I rates)
    fig1 = plt.figure(figsize=(20, 18))
    gs1 = fig1.add_gridspec(6, 4, hspace=0.4, wspace=0.4)
    fig1.suptitle('Individual Connection Type Ratios vs Firing Rates', fontsize=16, fontweight='bold', y=0.98)
    
    ratio_pairs = [
        ('EE/EI', 'EI/EE'),
        ('EE/IE', 'IE/EE'),
        ('EE/II', 'II/EE'),
        ('EI/IE', 'IE/EI'),
        ('EI/II', 'II/EI'),
        ('IE/II', 'II/IE'),
    ]
    
    for row_idx, (ratio1_name, ratio2_name) in enumerate(ratio_pairs):
        # E Rate plots
        ax1 = fig1.add_subplot(gs1[row_idx, 0])
        ratio1_data = individual_ratios[ratio1_name]
        ax1.scatter(ratio1_data, e_rates, alpha=0.6, s=30, color='blue')
        corr, pval = pearsonr(ratio1_data, e_rates)
        ax1.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
               transform=ax1.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax1.set_xlabel(f'{ratio1_name}', fontsize=9)
        ax1.set_ylabel('E Rate (Hz)', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=7)
        
        ax2 = fig1.add_subplot(gs1[row_idx, 1])
        ratio2_data = individual_ratios[ratio2_name]
        ax2.scatter(ratio2_data, e_rates, alpha=0.6, s=30, color='red')
        corr, pval = pearsonr(ratio2_data, e_rates)
        ax2.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
               transform=ax2.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.set_xlabel(f'{ratio2_name}', fontsize=9)
        ax2.set_ylabel('E Rate (Hz)', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=7)
        
        # I Rate plots
        ax3 = fig1.add_subplot(gs1[row_idx, 2])
        ax3.scatter(ratio1_data, i_rates, alpha=0.6, s=30, color='blue')
        corr, pval = pearsonr(ratio1_data, i_rates)
        ax3.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
               transform=ax3.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax3.set_xlabel(f'{ratio1_name}', fontsize=9)
        ax3.set_ylabel('I Rate (Hz)', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=7)
        
        ax4 = fig1.add_subplot(gs1[row_idx, 3])
        ax4.scatter(ratio2_data, i_rates, alpha=0.6, s=30, color='red')
        corr, pval = pearsonr(ratio2_data, i_rates)
        ax4.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
               transform=ax4.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax4.set_xlabel(f'{ratio2_name}', fontsize=9)
        ax4.set_ylabel('I Rate (Hz)', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=7)
    
    save_path = os.path.join(save_folder, "individual_ratios_vs_rates.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    # Part 2: Sum ratios (12 ratios, 2 rates each = 24 plots)
    fig2 = plt.figure(figsize=(20, 18))
    gs2 = fig2.add_gridspec(6, 4, hspace=0.4, wspace=0.4)
    fig2.suptitle('Signed Sum Ratios (E connections +, I connections -) vs Firing Rates', fontsize=16, fontweight='bold', y=0.98)
    
    sum_ratio_list = list(sum_ratios.items())
    for row_idx in range(6):
        # E Rate plots (columns 0 and 1)
        if row_idx * 2 < len(sum_ratio_list):
            ratio_name, ratio_data = sum_ratio_list[row_idx * 2]
            ax = fig2.add_subplot(gs2[row_idx, 0])
            ax.scatter(ratio_data, e_rates, alpha=0.6, s=30, color='green')
            corr, pval = pearsonr(ratio_data, e_rates)
            ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_xlabel(ratio_name, fontsize=9)
            ax.set_ylabel('E Rate (Hz)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        if row_idx * 2 + 1 < len(sum_ratio_list):
            ratio_name, ratio_data = sum_ratio_list[row_idx * 2 + 1]
            ax = fig2.add_subplot(gs2[row_idx, 1])
            ax.scatter(ratio_data, e_rates, alpha=0.6, s=30, color='orange')
            corr, pval = pearsonr(ratio_data, e_rates)
            ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_xlabel(ratio_name, fontsize=9)
            ax.set_ylabel('E Rate (Hz)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        # I Rate plots (columns 2 and 3)
        if row_idx * 2 < len(sum_ratio_list):
            ratio_name, ratio_data = sum_ratio_list[row_idx * 2]
            ax = fig2.add_subplot(gs2[row_idx, 2])
            ax.scatter(ratio_data, i_rates, alpha=0.6, s=30, color='green')
            corr, pval = pearsonr(ratio_data, i_rates)
            ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_xlabel(ratio_name, fontsize=9)
            ax.set_ylabel('I Rate (Hz)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        if row_idx * 2 + 1 < len(sum_ratio_list):
            ratio_name, ratio_data = sum_ratio_list[row_idx * 2 + 1]
            ax = fig2.add_subplot(gs2[row_idx, 3])
            ax.scatter(ratio_data, i_rates, alpha=0.6, s=30, color='orange')
            corr, pval = pearsonr(ratio_data, i_rates)
            ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_xlabel(ratio_name, fontsize=9)
            ax.set_ylabel('I Rate (Hz)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
    
    save_path = os.path.join(save_folder, "sum_ratios_vs_rates.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    print(f"  NOTE: Sum ratios are in a separate figure: sum_ratios_vs_rates.png")
    plt.close()
    
    # Part 3: Raw sums (not ratios) of different combinations - SIGNED
    # E connections (EE, EI) are positive, I connections (IE, II) are negative
    raw_sums = {
        'EE+EI': ee_scales + ei_scales,  # All FROM E (both positive)
        'IE+II': -(ie_scales + ii_scales),  # All FROM I (both negative, so negate)
        'EE-IE': ee_scales - ie_scales,    # TO E: positive from E, negative from I
        'EI-II': ei_scales - ii_scales,    # TO I: positive from E, negative from I
        'EE-II': ee_scales - ii_scales,  # E->E (positive) - I->I (negative)
        'EI-IE': ei_scales - ie_scales,  # E->I (positive) - I->E (negative)
        'EE+EI-IE': ee_scales + ei_scales - ie_scales,  # All FROM E minus I->E
        'EE+EI-II': ee_scales + ei_scales - ii_scales,  # All FROM E minus I->I
        'EE-IE-II': ee_scales - ie_scales - ii_scales,  # E->E minus all FROM I
        'EI-IE-II': ei_scales - ie_scales - ii_scales,  # E->I minus all FROM I
        'EE+EI-IE-II': ee_scales + ei_scales - ie_scales - ii_scales,  # Signed total sum
    }
    
    fig3 = plt.figure(figsize=(20, 16))
    gs3 = fig3.add_gridspec(6, 4, hspace=0.4, wspace=0.4)
    fig3.suptitle('Signed Raw Sums (E connections +, I connections -) vs Firing Rates', fontsize=16, fontweight='bold', y=0.98)
    
    sum_list = list(raw_sums.items())
    for row_idx in range(6):
        # E Rate plots (columns 0 and 1)
        if row_idx * 2 < len(sum_list):
            sum_name, sum_data = sum_list[row_idx * 2]
            ax = fig3.add_subplot(gs3[row_idx, 0])
            ax.scatter(sum_data, e_rates, alpha=0.6, s=30, color='blue')
            corr, pval = pearsonr(sum_data, e_rates)
            ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_xlabel(sum_name, fontsize=9)
            ax.set_ylabel('E Rate (Hz)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        if row_idx * 2 + 1 < len(sum_list):
            sum_name, sum_data = sum_list[row_idx * 2 + 1]
            ax = fig3.add_subplot(gs3[row_idx, 1])
            ax.scatter(sum_data, e_rates, alpha=0.6, s=30, color='cyan')
            corr, pval = pearsonr(sum_data, e_rates)
            ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_xlabel(sum_name, fontsize=9)
            ax.set_ylabel('E Rate (Hz)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        # I Rate plots (columns 2 and 3)
        if row_idx * 2 < len(sum_list):
            sum_name, sum_data = sum_list[row_idx * 2]
            ax = fig3.add_subplot(gs3[row_idx, 2])
            ax.scatter(sum_data, i_rates, alpha=0.6, s=30, color='blue')
            corr, pval = pearsonr(sum_data, i_rates)
            ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_xlabel(sum_name, fontsize=9)
            ax.set_ylabel('I Rate (Hz)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        if row_idx * 2 + 1 < len(sum_list):
            sum_name, sum_data = sum_list[row_idx * 2 + 1]
            ax = fig3.add_subplot(gs3[row_idx, 3])
            ax.scatter(sum_data, i_rates, alpha=0.6, s=30, color='cyan')
            corr, pval = pearsonr(sum_data, i_rates)
            ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_xlabel(sum_name, fontsize=9)
            ax.set_ylabel('I Rate (Hz)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
    
    save_path = os.path.join(save_folder, "raw_sums_vs_rates.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    # Print comprehensive summary statistics
    print("\n" + "="*70)
    print("COMPREHENSIVE CORRELATION SUMMARY (Ratios vs Firing Rates)")
    print("="*70)
    
    print("\n--- Individual Connection Type Ratios ---")
    for ratio_name, ratio_data in sorted(individual_ratios.items()):
        print(f"\n{ratio_name}:")
        corr_e, pval_e = pearsonr(ratio_data, e_rates)
        corr_i, pval_i = pearsonr(ratio_data, i_rates)
        print(f"  vs E Rate: r={corr_e:7.3f}, p={pval_e:.3e}")
        print(f"    X (ratio): min={np.min(ratio_data):.3f}, max={np.max(ratio_data):.3f}, mean={np.mean(ratio_data):.3f}, median={np.median(ratio_data):.3f}")
        print(f"    Y (E rate): min={np.min(e_rates):.2f}, max={np.max(e_rates):.2f}, mean={np.mean(e_rates):.2f}, median={np.median(e_rates):.2f}")
        # Identify quiescent E neurons (rate < 1 Hz)
        quiescent_mask = e_rates < 1.0
        if np.any(quiescent_mask):
            print(f"    WARNING: {np.sum(quiescent_mask)} quiescent E neurons (rate < 1 Hz)")
            print(f"      X values for quiescent: {ratio_data[quiescent_mask]}")
            print(f"      Y values for quiescent: {e_rates[quiescent_mask]}")
        print(f"  vs I Rate: r={corr_i:7.3f}, p={pval_i:.3e}")
        print(f"    X (ratio): min={np.min(ratio_data):.3f}, max={np.max(ratio_data):.3f}, mean={np.mean(ratio_data):.3f}, median={np.median(ratio_data):.3f}")
        print(f"    Y (I rate): min={np.min(i_rates):.2f}, max={np.max(i_rates):.2f}, mean={np.mean(i_rates):.2f}, median={np.median(i_rates):.2f}")
    
    print("\n--- Signed Sum Ratios (E connections +, I connections -) ---")
    for ratio_name, ratio_data in sorted(sum_ratios.items()):
        print(f"\n{ratio_name}:")
        corr_e, pval_e = pearsonr(ratio_data, e_rates)
        corr_i, pval_i = pearsonr(ratio_data, i_rates)
        print(f"  vs E Rate: r={corr_e:7.3f}, p={pval_e:.3e}")
        print(f"  vs I Rate: r={corr_i:7.3f}, p={pval_i:.3e}")
    
    print("\n--- Other Ratios ---")
    print(f"\nE/I Rate Ratio:")
    corr_e, pval_e = pearsonr(ei_rate_ratio, e_rates)
    corr_i, pval_i = pearsonr(ei_rate_ratio, i_rates)
    print(f"  vs E Rate: r={corr_e:7.3f}, p={pval_e:.3e}")
    print(f"  vs I Rate: r={corr_i:7.3f}, p={pval_i:.3e}")
    
    print(f"\nMean E Weight / Mean I Weight:")
    corr_e, pval_e = pearsonr(mean_weight_ratio, e_rates)
    corr_i, pval_i = pearsonr(mean_weight_ratio, i_rates)
    print(f"  vs E Rate: r={corr_e:7.3f}, p={pval_e:.3e}")
    print(f"  vs I Rate: r={corr_i:7.3f}, p={pval_i:.3e}")
    
    print("\n--- Signed Raw Sums (E connections +, I connections -) ---")
    for sum_name, sum_data in sorted(raw_sums.items()):
        print(f"\n{sum_name}:")
        corr_e, pval_e = pearsonr(sum_data, e_rates)
        corr_i, pval_i = pearsonr(sum_data, i_rates)
        print(f"  vs E Rate: r={corr_e:7.3f}, p={pval_e:.3e}")
        print(f"    X (sum): min={np.min(sum_data):.3f}, max={np.max(sum_data):.3f}, mean={np.mean(sum_data):.3f}, median={np.median(sum_data):.3f}")
        print(f"    Y (E rate): min={np.min(e_rates):.2f}, max={np.max(e_rates):.2f}, mean={np.mean(e_rates):.2f}, median={np.median(e_rates):.2f}")
        quiescent_mask = e_rates < 1.0
        if np.any(quiescent_mask):
            print(f"    WARNING: {np.sum(quiescent_mask)} quiescent E neurons (rate < 1 Hz)")
            print(f"      X values for quiescent: {sum_data[quiescent_mask]}")
            print(f"      Y values for quiescent: {e_rates[quiescent_mask]}")
        print(f"  vs I Rate: r={corr_i:7.3f}, p={pval_i:.3e}")
        print(f"    X (sum): min={np.min(sum_data):.3f}, max={np.max(sum_data):.3f}, mean={np.mean(sum_data):.3f}, median={np.median(sum_data):.3f}")
        print(f"    Y (I rate): min={np.min(i_rates):.2f}, max={np.max(i_rates):.2f}, mean={np.mean(i_rates):.2f}, median={np.median(i_rates):.2f}")
    
    # Summary: Identify parameters associated with quiescent E neurons
    print("\n" + "="*70)
    print("SUMMARY: Parameters Associated with Quiescent E Neurons (rate < 1 Hz)")
    print("="*70)
    quiescent_mask = e_rates < 1.0
    active_mask = e_rates >= 1.0
    
    if np.any(quiescent_mask):
        print(f"\nFound {np.sum(quiescent_mask)} quiescent runs out of {len(e_rates)} total")
        print(f"\nConnection scales for quiescent vs active:")
        print(f"  E->E: quiescent mean={np.mean(ee_scales[quiescent_mask]):.3f}, active mean={np.mean(ee_scales[active_mask]):.3f}")
        print(f"  E->I: quiescent mean={np.mean(ei_scales[quiescent_mask]):.3f}, active mean={np.mean(ei_scales[active_mask]):.3f}")
        print(f"  I->E: quiescent mean={np.mean(ie_scales[quiescent_mask]):.3f}, active mean={np.mean(ie_scales[active_mask]):.3f}")
        print(f"  I->I: quiescent mean={np.mean(ii_scales[quiescent_mask]):.3f}, active mean={np.mean(ii_scales[active_mask]):.3f}")
        print(f"\nNet E drive (EE-IE): quiescent mean={np.mean(ee_scales[quiescent_mask] - ie_scales[quiescent_mask]):.3f}, active mean={np.mean(ee_scales[active_mask] - ie_scales[active_mask]):.3f}")
        print(f"EE/IE ratio: quiescent mean={np.mean(ee_scales[quiescent_mask] / (ie_scales[quiescent_mask] + 1e-6)):.3f}, active mean={np.mean(ee_scales[active_mask] / (ie_scales[active_mask] + 1e-6)):.3f}")
    else:
        print("\nNo quiescent E neurons found (all rates >= 1 Hz)")

def analyze_connection_pairs_vs_rates(all_results, save_folder):
    """Analyze individual connection type pairs vs firing rates"""
    print("\nAnalyzing connection type pairs vs firing rates...")
    
    # Filter successful results
    successful_results = [r for r in all_results if r.get('success', False)]
    
    if len(successful_results) == 0:
        print("  No successful runs to analyze")
        return
    
    # Extract data
    e_rates = np.array([r['e_rate'] for r in successful_results])
    i_rates = np.array([r['i_rate'] for r in successful_results])
    total_rate = e_rates + i_rates
    mean_rate = (e_rates + i_rates) / 2.0
    
    ee_scales = np.array([r['e_to_e_scale'] for r in successful_results])
    ei_scales = np.array([r['e_to_i_scale'] for r in successful_results])
    ie_scales = np.array([r['i_to_e_scale'] for r in successful_results])
    ii_scales = np.array([r['i_to_i_scale'] for r in successful_results])
    
    # Use log scale for connection scales
    log_ee = np.log10(ee_scales + 1e-6)
    log_ei = np.log10(ei_scales + 1e-6)
    log_ie = np.log10(ie_scales + 1e-6)
    log_ii = np.log10(ii_scales + 1e-6)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('Connection Type Scales vs Firing Rates', fontsize=16, fontweight='bold', y=0.995)
    
    connections = [
        ('E->E', log_ee, 'red'),
        ('E->I', log_ei, 'orange'),
        ('I->E', log_ie, 'blue'),
        ('I->I', log_ii, 'purple'),
    ]
    
    rate_metrics = [
        ('E Rate (Hz)', e_rates),
        ('I Rate (Hz)', i_rates),
        ('Total Rate (Hz)', total_rate),
    ]
    
    for col, (conn_name, conn_data, color) in enumerate(connections):
        for row, (rate_name, rate_data) in enumerate(rate_metrics):
            ax = fig.add_subplot(gs[row, col])
            ax.scatter(conn_data, rate_data, alpha=0.6, s=50, color=color)
            
            corr, pval = pearsonr(conn_data, rate_data)
            ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(f'{conn_name} Scale (log10)', fontsize=10)
            ax.set_ylabel(rate_name, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if row == 0:
                ax.set_title(conn_name, fontsize=12, fontweight='bold')
    
    save_path = os.path.join(save_folder, "connection_scales_vs_rates.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    # Print correlations
    print("\nConnection Scales vs Firing Rates Correlations:")
    print("="*70)
    for conn_name, conn_data, _ in connections:
        print(f"\n{conn_name} (log scale):")
        for rate_name, rate_data in rate_metrics:
            corr, pval = pearsonr(conn_data, rate_data)
            print(f"  vs {rate_name:15}: r={corr:7.3f}, p={pval:.3e}")

def analyze_cv_isis(all_results, save_folder):
    """Analyze CV of ISIs vs firing rates, E/I ratios, and connection scales"""
    print("\nAnalyzing CV of ISIs vs firing rates, ratios, and connection scales...")
    
    # Filter successful results
    successful_results = [r for r in all_results if r.get('success', False)]
    
    if len(successful_results) == 0:
        print("  No successful runs to analyze")
        return
    
    # Extract data
    e_rates = np.array([r['e_rate'] for r in successful_results])
    i_rates = np.array([r['i_rate'] for r in successful_results])
    cv_e = np.array([r.get('cv_e', 0) for r in successful_results])
    cv_i = np.array([r.get('cv_i', 0) for r in successful_results])
    
    ee_scales = np.array([r['e_to_e_scale'] for r in successful_results])
    ei_scales = np.array([r['e_to_i_scale'] for r in successful_results])
    ie_scales = np.array([r['i_to_e_scale'] for r in successful_results])
    ii_scales = np.array([r['i_to_i_scale'] for r in successful_results])
    mean_e_weights = np.array([r['mean_e_weight'] for r in successful_results])
    mean_i_weights = np.array([np.abs(r['mean_i_weight']) for r in successful_results])
    
    # Compute ratios
    ei_rate_ratio = e_rates / (i_rates + 1e-6)
    weights_from_e = ee_scales + ei_scales
    weights_from_i = ie_scales + ii_scales
    weights_from_e_over_i = weights_from_e / (weights_from_i + 1e-6)
    mean_weight_ratio = mean_e_weights / (mean_i_weights + 1e-6)
    
    # Use log scale for connection scales
    log_ee = np.log10(ee_scales + 1e-6)
    log_ei = np.log10(ei_scales + 1e-6)
    log_ie = np.log10(ie_scales + 1e-6)
    log_ii = np.log10(ii_scales + 1e-6)
    
    # Create figure: CV vs Firing Rates, Ratios, and Connection Scales
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)
    fig.suptitle('CV of ISIs vs Firing Rates, Ratios, and Connection Scales', fontsize=16, fontweight='bold', y=0.98)
    
    # Row 1: CV vs Firing Rates
    for col_idx, (rate_name, rate_data, cv_name, cv_data) in enumerate([
        ('E Rate (Hz)', e_rates, 'CV_E', cv_e),
        ('I Rate (Hz)', i_rates, 'CV_I', cv_i),
        ('E Rate (Hz)', e_rates, 'CV_I', cv_i),
        ('I Rate (Hz)', i_rates, 'CV_E', cv_e)
    ]):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.scatter(rate_data, cv_data, alpha=0.6, s=50, color='blue' if 'CV_E' in cv_name else 'red')
        corr, pval = pearsonr(rate_data, cv_data)
        ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(rate_name, fontsize=11)
        ax.set_ylabel(f'{cv_name} (CV of ISIs)', fontsize=11)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_title('CV vs Firing Rates', fontsize=12, fontweight='bold')
    
    # Row 2: CV vs Ratios
    ratios_data = [
        ('E/I Rate Ratio', ei_rate_ratio),
        ('Weights FROM E/I', weights_from_e_over_i),
        ('Mean Weight Ratio', mean_weight_ratio),
        ('E/I Rate Ratio', ei_rate_ratio)
    ]
    cv_data_row2 = [cv_e, cv_e, cv_e, cv_i]
    
    for col_idx, ((ratio_name, ratio_data), cv_data) in enumerate(zip(ratios_data, cv_data_row2)):
        ax = fig.add_subplot(gs[1, col_idx])
        ax.scatter(ratio_data, cv_data, alpha=0.6, s=50, color='green')
        corr, pval = pearsonr(ratio_data, cv_data)
        ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(ratio_name, fontsize=11)
        ax.set_ylabel(f'CV of ISIs', fontsize=11)
        if 'Ratio' in ratio_name and 'Rate' not in ratio_name:
            ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_title('CV vs Ratios', fontsize=12, fontweight='bold')
    
    # Row 3: CV vs Connection Scales
    connections = [
        ('E->E', log_ee, cv_e),
        ('E->I', log_ei, cv_e),
        ('I->E', log_ie, cv_i),
        ('I->I', log_ii, cv_i)
    ]
    
    for col_idx, (conn_name, conn_data, cv_data) in enumerate(connections):
        ax = fig.add_subplot(gs[2, col_idx])
        ax.scatter(conn_data, cv_data, alpha=0.6, s=50, color='purple')
        corr, pval = pearsonr(conn_data, cv_data)
        ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(f'{conn_name} Scale (log10)', fontsize=11)
        ax.set_ylabel('CV of ISIs', fontsize=11)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_title('CV vs Connection Scales', fontsize=12, fontweight='bold')
    
    save_path = os.path.join(save_folder, "cv_isis_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    # Print correlation summary
    print("\nCV of ISIs Correlation Summary:")
    print("="*70)
    
    print("\nCV_E vs Firing Rates:")
    corr, pval = pearsonr(e_rates, cv_e)
    print(f"  vs E Rate: r={corr:7.3f}, p={pval:.3e}")
    corr, pval = pearsonr(i_rates, cv_e)
    print(f"  vs I Rate: r={corr:7.3f}, p={pval:.3e}")
    
    print("\nCV_I vs Firing Rates:")
    corr, pval = pearsonr(e_rates, cv_i)
    print(f"  vs E Rate: r={corr:7.3f}, p={pval:.3e}")
    corr, pval = pearsonr(i_rates, cv_i)
    print(f"  vs I Rate: r={corr:7.3f}, p={pval:.3e}")
    
    print("\nCV_E vs Ratios:")
    corr, pval = pearsonr(ei_rate_ratio, cv_e)
    print(f"  vs E/I Rate Ratio: r={corr:7.3f}, p={pval:.3e}")
    corr, pval = pearsonr(weights_from_e_over_i, cv_e)
    print(f"  vs Weights FROM E/I: r={corr:7.3f}, p={pval:.3e}")
    corr, pval = pearsonr(mean_weight_ratio, cv_e)
    print(f"  vs Mean Weight Ratio: r={corr:7.3f}, p={pval:.3e}")
    
    print("\nCV_I vs Ratios:")
    corr, pval = pearsonr(ei_rate_ratio, cv_i)
    print(f"  vs E/I Rate Ratio: r={corr:7.3f}, p={pval:.3e}")
    corr, pval = pearsonr(weights_from_e_over_i, cv_i)
    print(f"  vs Weights FROM E/I: r={corr:7.3f}, p={pval:.3e}")
    corr, pval = pearsonr(mean_weight_ratio, cv_i)
    print(f"  vs Mean Weight Ratio: r={corr:7.3f}, p={pval:.3e}")
    
    print("\nCV_E vs Connection Scales (log):")
    for conn_name, conn_data in [('E->E', log_ee), ('E->I', log_ei), ('I->E', log_ie), ('I->I', log_ii)]:
        corr, pval = pearsonr(conn_data, cv_e)
        print(f"  vs {conn_name}: r={corr:7.3f}, p={pval:.3e}")
    
        print("\nCV_I vs Connection Scales (log):")
        for conn_name, conn_data in [('E->E', log_ee), ('E->I', log_ei), ('I->E', log_ie), ('I->I', log_ii)]:
            corr, pval = pearsonr(conn_data, cv_i)
            print(f"  vs {conn_name}: r={corr:7.3f}, p={pval:.3e}")

def analyze_alpha_values(all_results, save_folder):
    """Analyze how Dirichlet alpha values relate to firing rates and CV of ISIs"""
    print("\nAnalyzing Dirichlet alpha values vs firing rates and CV of ISIs...")
    
    # Filter successful results
    successful_results = [r for r in all_results if r.get('success', False)]
    
    if len(successful_results) < 3:
        print("  Not enough successful runs for alpha analysis")
        return
    
    # Extract alpha values and metrics
    alpha_ee = []
    alpha_ei = []
    alpha_ie = []
    alpha_ii = []
    e_rates = []
    i_rates = []
    cv_e = []
    cv_i = []
    
    for r in successful_results:
        if 'dirichlet_alpha' in r and r['dirichlet_alpha'] is not None:
            alpha_vec = np.array(r['dirichlet_alpha'])
            if len(alpha_vec) == 4:
                alpha_ee.append(alpha_vec[0])
                alpha_ei.append(alpha_vec[1])
                alpha_ie.append(alpha_vec[2])
                alpha_ii.append(alpha_vec[3])
                e_rates.append(r.get('e_rate', 0))
                i_rates.append(r.get('i_rate', 0))
                cv_e.append(r.get('cv_e', 0))
                cv_i.append(r.get('cv_i', 0))
    
    if len(alpha_ee) < 3:
        print("  Not enough runs with alpha data")
        return
    
    alpha_ee = np.array(alpha_ee)
    alpha_ei = np.array(alpha_ei)
    alpha_ie = np.array(alpha_ie)
    alpha_ii = np.array(alpha_ii)
    e_rates = np.array(e_rates)
    i_rates = np.array(i_rates)
    cv_e = np.array(cv_e)
    cv_i = np.array(cv_i)
    
    # Create figure: Alpha values vs Firing Rates and CV
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)
    fig.suptitle('Dirichlet Alpha Values vs Firing Rates and CV of ISIs', fontsize=16, fontweight='bold', y=0.98)
    
    # Row 1: Alpha vs E Rate
    alpha_data = [alpha_ee, alpha_ei, alpha_ie, alpha_ii]
    alpha_names = ['Alpha_EE', 'Alpha_EI', 'Alpha_IE', 'Alpha_II']
    
    for col_idx, (alpha_name, alpha_vals) in enumerate(zip(alpha_names, alpha_data)):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.scatter(alpha_vals, e_rates, alpha=0.6, s=50, color='blue')
        corr, pval = pearsonr(alpha_vals, e_rates)
        ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(alpha_name, fontsize=11)
        ax.set_ylabel('E Rate (Hz)', fontsize=11)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_title('Alpha vs E Firing Rate', fontsize=12, fontweight='bold')
    
    # Row 2: Alpha vs I Rate
    for col_idx, (alpha_name, alpha_vals) in enumerate(zip(alpha_names, alpha_data)):
        ax = fig.add_subplot(gs[1, col_idx])
        ax.scatter(alpha_vals, i_rates, alpha=0.6, s=50, color='red')
        corr, pval = pearsonr(alpha_vals, i_rates)
        ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(alpha_name, fontsize=11)
        ax.set_ylabel('I Rate (Hz)', fontsize=11)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_title('Alpha vs I Firing Rate', fontsize=12, fontweight='bold')
    
    # Row 3: Alpha vs CV
    cv_data_row3 = [cv_e, cv_e, cv_i, cv_i]
    cv_names_row3 = ['CV_E', 'CV_E', 'CV_I', 'CV_I']
    
    for col_idx, (alpha_name, alpha_vals, cv_name, cv_vals) in enumerate(
        zip(alpha_names, alpha_data, cv_names_row3, cv_data_row3)
    ):
        ax = fig.add_subplot(gs[2, col_idx])
        ax.scatter(alpha_vals, cv_vals, alpha=0.6, s=50, color='green')
        corr, pval = pearsonr(alpha_vals, cv_vals)
        ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3e}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(alpha_name, fontsize=11)
        ax.set_ylabel(f'{cv_name} (CV of ISIs)', fontsize=11)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_title('Alpha vs CV of ISIs', fontsize=12, fontweight='bold')
    
    save_path = os.path.join(save_folder, "alpha_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    # Print correlation summary
    print("\nDirichlet Alpha Correlation Summary:")
    print("="*70)
    
    print("\nAlpha vs E Rate:")
    for alpha_name, alpha_vals in zip(alpha_names, alpha_data):
        corr, pval = pearsonr(alpha_vals, e_rates)
        print(f"  {alpha_name}: r={corr:7.3f}, p={pval:.3e}")
    
    print("\nAlpha vs I Rate:")
    for alpha_name, alpha_vals in zip(alpha_names, alpha_data):
        corr, pval = pearsonr(alpha_vals, i_rates)
        print(f"  {alpha_name}: r={corr:7.3f}, p={pval:.3e}")
    
    print("\nAlpha vs CV_E:")
    for alpha_name, alpha_vals in zip(alpha_names, alpha_data):
        corr, pval = pearsonr(alpha_vals, cv_e)
        print(f"  {alpha_name}: r={corr:7.3f}, p={pval:.3e}")
    
    print("\nAlpha vs CV_I:")
    for alpha_name, alpha_vals in zip(alpha_names, alpha_data):
        corr, pval = pearsonr(alpha_vals, cv_i)
        print(f"  {alpha_name}: r={corr:7.3f}, p={pval:.3e}")

if __name__ == "__main__":
    print("="*70)
    print("RANDOM SEARCH ANALYSIS")
    print("="*70)
    
    # Find and load the latest results
    try:
        results_file, results_folder = find_latest_random_search_results()
        print(f"\nLoading results from: {results_file}")
        
        with open(results_file, 'rb') as f:
            all_results = pickle.load(f)
        
        print(f"Loaded {len(all_results)} results")
        successful_results = [r for r in all_results if r.get('success', False)]
        print(f"Successful runs: {len(successful_results)}/{len(all_results)}")
        
        # Create analysis folder
        analysis_folder = os.path.join(results_folder, "Analysis")
        os.makedirs(analysis_folder, exist_ok=True)
        
        # Run analyses
        analyze_connection_correlations(all_results, analysis_folder)
        analyze_ratios_vs_rates(all_results, analysis_folder)
        analyze_connection_pairs_vs_rates(all_results, analysis_folder)
        analyze_cv_isis(all_results, analysis_folder)
        analyze_alpha_values(all_results, analysis_folder)
        
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved to: {analysis_folder}")
        print(f"{'='*70}\n")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease run recurrent/parameter_search/random_search.py first to generate simulation data.")

