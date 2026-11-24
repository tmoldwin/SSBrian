"""
Simple analysis: Correlation between neural selectivity and firing rate.

Loads saved simulation data and creates ONE figure showing the relationship
between how selective neurons are and their average firing rates.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import stats

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

def analyze_selectivity_firing_rate_correlation(data_folder_path):
    """
    Analyze correlation between neural selectivity and firing rate.
    Creates ONE figure with the correlation analysis.
    """
    # Load data
    data = load_simulation_data(data_folder_path)
    
    # Get final epoch data
    final_rate_matrix = data['rate_matrices_arr'][-1]  # Last epoch
    final_selectivity = data['selectivity_vec_arr'][-1, :]  # Last epoch selectivity
    
    # Calculate mean firing rate for each neuron (across all stimuli)
    mean_firing_rates = np.mean(final_rate_matrix, axis=1)
    
    # Remove neurons with zero firing rate
    active_mask = mean_firing_rates > 0
    firing_rates_clean = mean_firing_rates[active_mask]
    selectivity_clean = final_selectivity[active_mask]
    
    # Calculate correlation
    corr_coeff, p_value = stats.pearsonr(firing_rates_clean, selectivity_clean)
    
    # Create ONE figure
    plt.figure(figsize=(8, 6))
    plt.scatter(firing_rates_clean, selectivity_clean, alpha=0.6, color='blue', s=30)
    
    # Add regression line
    z = np.polyfit(firing_rates_clean, selectivity_clean, 1)
    p = np.poly1d(z)
    x_line = np.linspace(firing_rates_clean.min(), firing_rates_clean.max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    plt.xlabel('Mean Firing Rate (Hz)', fontsize=12)
    plt.ylabel('Selectivity Index', fontsize=12)
    plt.title(f'Neural Selectivity vs Firing Rate\nPearson r = {corr_coeff:.3f}, p = {p_value:.3e}', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the ONE figure
    plot_path = os.path.join(data_folder_path, 'selectivity_firing_rate_correlation.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Figure saved to: {plot_path}")
    print(f"📊 Correlation: r = {corr_coeff:.3f}, p = {p_value:.3e}")
    print(f"📊 Analyzed {len(firing_rates_clean)} active neurons")
    
    return corr_coeff, p_value

def main():
    """Run the analysis on available data."""
    # Find first available dataset
    folders_to_check = [
        "Figures/DI_Sims/direct_input_REC_stim50_ep5_n200",
        "brian_experiment/Figures/DI_Sims/direct_input_REC_stim50_ep5_n200",
        "Figures/DI_Sims/direct_input_FF_stim50_ep5_n200", 
        "brian_experiment/Figures/DI_Sims/direct_input_FF_stim50_ep5_n200"
    ]
    
    data_folder = None
    for folder in folders_to_check:
        if os.path.exists(os.path.join(folder, "simulation_data.pkl")):
            data_folder = folder
            break
    
    if data_folder is None:
        print("❌ No simulation data found!")
        return
    
    print(f"🎯 Analyzing: {data_folder}")
    analyze_selectivity_firing_rate_correlation(data_folder)

if __name__ == "__main__":
    main()




