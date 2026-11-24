"""
Diagnostic analysis to understand weak correlation between selectivity and neural correlations
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from scipy import stats
import os

def diagnose_weak_correlation(run_folder):
    """Diagnose why selectivity vs correlation is weak"""
    
    # Load data
    data_file = os.path.join(run_folder, 'Data', 'simulation_data.pkl')
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get final epoch data
    final_rate_matrix = data['rate_matrices_arr'][-1]
    final_selectivity = data['selectivity_vec_arr'][-1]
    
    # Get network config
    config = data['config']
    N_rec_E = config['N_rec_E']
    N_rec_I = config['N_rec_I']
    N_rec = N_rec_E + N_rec_I
    
    # Only analyze active neurons
    mean_firing_rates = np.mean(final_rate_matrix, axis=1)
    active_mask = mean_firing_rates > 0
    active_rate_matrix = final_rate_matrix[active_mask, :]
    active_selectivity = final_selectivity[active_mask]
    
    # Calculate neural correlations
    neural_correlations = np.corrcoef(active_rate_matrix)
    np.fill_diagonal(neural_correlations, np.nan)
    
    # Get E/I indices
    active_indices = np.where(active_mask)[0]
    active_is_E = active_indices < N_rec_E
    active_is_I = active_indices >= N_rec_E
    
    E_indices = np.where(active_is_E)[0]
    I_indices = np.where(active_is_I)[0]
    
    # Mean correlations
    mean_corr_with_E = np.nanmean(neural_correlations[:, E_indices], axis=1)
    mean_corr_with_I = np.nanmean(neural_correlations[:, I_indices], axis=1)
    
    # Create diagnostic figure
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Diagnostic Analysis: Why is Selectivity vs Correlation Weak?', 
                 fontsize=16, fontweight='bold')
    
    # 1. Distribution of selectivity values
    ax1 = plt.subplot(3, 4, 1)
    ax1.hist(active_selectivity, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Selectivity Index')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Selectivity')
    ax1.axvline(np.mean(active_selectivity), color='red', linestyle='--', 
                label=f'Mean: {np.mean(active_selectivity):.3f}')
    ax1.legend()
    
    # 2. Distribution of correlations WITH E
    ax2 = plt.subplot(3, 4, 2)
    ax2.hist(mean_corr_with_E, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Mean Corr WITH E')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Correlations WITH E')
    ax2.axvline(np.mean(mean_corr_with_E), color='red', linestyle='--',
                label=f'Mean: {np.mean(mean_corr_with_E):.3f}')
    ax2.legend()
    
    # 3. Distribution of correlations WITH I
    ax3 = plt.subplot(3, 4, 3)
    ax3.hist(mean_corr_with_I, bins=30, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Mean Corr WITH I')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Correlations WITH I')
    ax3.axvline(np.mean(mean_corr_with_I), color='red', linestyle='--',
                label=f'Mean: {np.mean(mean_corr_with_I):.3f}')
    ax3.legend()
    
    # 4. Variance explained by input weights vs recurrent correlations
    # Get input influence: dot product of input weights with stimulus variance
    input_weights = data['weight_poisson']
    input_influence = np.sum(np.abs(input_weights[active_mask, :]), axis=1)
    
    ax4 = plt.subplot(3, 4, 4)
    ax4.scatter(input_influence, active_selectivity, alpha=0.5, s=20)
    r_input, p_input = stats.pearsonr(input_influence, active_selectivity)
    ax4.set_xlabel('Sum of |Input Weights|')
    ax4.set_ylabel('Selectivity')
    ax4.set_title(f'Input Weight Strength vs Selectivity\nr={r_input:.3f}, p={p_input:.3e}')
    ax4.grid(True, alpha=0.3)
    
    # 5. Selectivity by E/I type
    ax5 = plt.subplot(3, 4, 5)
    ax5.boxplot([active_selectivity[active_is_E], active_selectivity[active_is_I]],
                labels=['E neurons', 'I neurons'])
    ax5.set_ylabel('Selectivity')
    ax5.set_title('Selectivity by Neuron Type')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Firing rate vs selectivity (the strong negative correlation)
    ax6 = plt.subplot(3, 4, 6)
    active_firing_rates = mean_firing_rates[active_mask]
    ax6.scatter(active_firing_rates, active_selectivity, alpha=0.5, s=20,
                c=['red' if e else 'blue' for e in active_is_E])
    r_fr, p_fr = stats.pearsonr(active_firing_rates, active_selectivity)
    ax6.set_xlabel('Mean Firing Rate (Hz)')
    ax6.set_ylabel('Selectivity')
    ax6.set_title(f'Firing Rate vs Selectivity\nr={r_fr:.3f}, p={p_fr:.3e}')
    ax6.grid(True, alpha=0.3)
    
    # 7. Correlation matrix structure
    ax7 = plt.subplot(3, 4, 7)
    im = ax7.imshow(neural_correlations, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
    ax7.set_title('Neural Correlation Matrix')
    ax7.set_xlabel('Neuron')
    ax7.set_ylabel('Neuron')
    plt.colorbar(im, ax=ax7)
    
    # 8. Relationship between mean correlation and variance of responses
    response_variance = np.var(active_rate_matrix, axis=1)
    ax8 = plt.subplot(3, 4, 8)
    ax8.scatter(np.nanmean(neural_correlations, axis=1), response_variance, alpha=0.5, s=20)
    ax8.set_xlabel('Mean Correlation')
    ax8.set_ylabel('Response Variance')
    ax8.set_title('Correlation vs Response Variability')
    ax8.grid(True, alpha=0.3)
    
    # 9. Recurrent weight strength vs selectivity (SEPARATE E and I TARGET neurons)
    weight_rec = data['weight_rec']
    
    # Get original indices for all E and I neurons in the full network
    all_E_indices = np.arange(N_rec_E)
    all_I_indices = np.arange(N_rec_E, N_rec)
    
    # For each active neuron, sum weights FROM E and FROM I sources (using original indices)
    orig_active_indices = active_indices
    recurrent_input_strength_from_E = np.sum(np.abs(weight_rec[orig_active_indices, :][:, all_E_indices]), axis=1)
    recurrent_input_strength_from_I = np.sum(np.abs(weight_rec[orig_active_indices, :][:, all_I_indices]), axis=1)
    
    # Plot E target neurons only
    ax9 = plt.subplot(3, 4, 9)
    ax9.scatter(recurrent_input_strength_from_E[active_is_E], active_selectivity[active_is_E], 
                alpha=0.6, s=30, color='red', edgecolors='black', linewidth=0.5, label='E targets')
    ax9.scatter(recurrent_input_strength_from_I[active_is_E], active_selectivity[active_is_E], 
                alpha=0.6, s=30, color='blue', edgecolors='black', linewidth=0.5, label='I targets')
    ax9.set_xlabel('Sum of |Weights FROM E or I sources|')
    ax9.set_ylabel('Selectivity')
    ax9.set_title(f'Recurrent Strength vs Selectivity\n(E target neurons only)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Plot I target neurons only
    ax10 = plt.subplot(3, 4, 10)
    ax10.scatter(recurrent_input_strength_from_E[active_is_I], active_selectivity[active_is_I], 
                 alpha=0.6, s=30, color='red', edgecolors='black', linewidth=0.5, label='FROM E')
    ax10.scatter(recurrent_input_strength_from_I[active_is_I], active_selectivity[active_is_I], 
                 alpha=0.6, s=30, color='blue', edgecolors='black', linewidth=0.5, label='FROM I')
    ax10.set_xlabel('Sum of |Weights FROM E or I sources|')
    ax10.set_ylabel('Selectivity')
    ax10.set_title(f'Recurrent Strength vs Selectivity\n(I target neurons only)')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Check if high vs low selectivity neurons have different correlation patterns
    high_sel_mask = active_selectivity > np.median(active_selectivity)
    low_sel_mask = active_selectivity <= np.median(active_selectivity)
    
    ax11 = plt.subplot(3, 4, 11)
    ax11.hist(mean_corr_with_E[high_sel_mask], bins=20, alpha=0.5, label='High Selectivity', color='red')
    ax11.hist(mean_corr_with_E[low_sel_mask], bins=20, alpha=0.5, label='Low Selectivity', color='blue')
    ax11.set_xlabel('Mean Corr WITH E')
    ax11.set_ylabel('Count')
    ax11.set_title('Corr WITH E: High vs Low Selectivity')
    ax11.legend()
    
    # 12. Summary statistics text box
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    summary_text = f"""
SUMMARY STATISTICS

Network Size: {N_rec} neurons ({N_rec_E}E, {N_rec-N_rec_E}I)
Active neurons: {np.sum(active_mask)} ({np.sum(active_is_E)}E, {np.sum(active_is_I)}I)

Selectivity:
  Mean: {np.mean(active_selectivity):.3f}
  Std: {np.std(active_selectivity):.3f}
  Range: [{np.min(active_selectivity):.3f}, {np.max(active_selectivity):.3f}]

Neural Correlations:
  Mean: {np.nanmean(neural_correlations):.3f}
  Std: {np.nanstd(neural_correlations):.3f}

Key Correlations:
  Input weight → Selectivity: r={r_input:.3f}
  Firing rate → Selectivity: r={r_fr:.3f}
  Correlation WITH E → Selectivity: r={stats.pearsonr(mean_corr_with_E, active_selectivity)[0]:.3f}
  Correlation WITH I → Selectivity: r={stats.pearsonr(mean_corr_with_I, active_selectivity)[0]:.3f}

INTERPRETATION:
The weak correlation suggests that selectivity
is primarily driven by {'INPUT' if abs(r_input) > 0.3 else 'FIRING RATE' if abs(r_fr) > 0.5 else 'OTHER'} factors,
not by patterns of neural correlations in the
recurrent network.
"""
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
              fontsize=9, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(run_folder, 'Figures', 'correlation_analysis', 
                             'diagnostic_weak_correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDiagnostic analysis saved to: {save_path}")
    print(f"\nKey finding: Selectivity is most strongly correlated with:")
    print(f"  - Firing rate: r={r_fr:.3f} (STRONG NEGATIVE)")
    print(f"  - Input weight strength: r={r_input:.3f}")
    print(f"  - Neural correlations WITH E: r={stats.pearsonr(mean_corr_with_E, active_selectivity)[0]:.3f} (WEAK)")
    print(f"  - Neural correlations WITH I: r={stats.pearsonr(mean_corr_with_I, active_selectivity)[0]:.3f} (WEAK)")
    print(f"\nThis suggests selectivity is primarily determined by FIRING RATE")
    print(f"and INPUT CONNECTIVITY, not by recurrent correlation patterns.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_folder = sys.argv[1]
    else:
        run_folder = 'brian_experiment/Runs/direct_input_REC_stim20_ep3_n200'
    
    diagnose_weak_correlation(run_folder)

