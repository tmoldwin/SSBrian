"""
Network Structure Analysis Module

This module analyzes the basic architecture and weight matrices of the network.
Creates plots in the network_structure/ subfolder.

Functions:
- generate_network_structure_plots: Main entry point for weight matrix visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def generate_network_structure_plots(run_folder, data, config):
    """
    Generate network structure and weight matrix visualizations.
    
    Parameters:
    -----------
    run_folder : str
        Path to the run directory
    data : dict
        Dictionary containing simulation data (weight_poisson, weight_rec, neuron_order, etc.)
    config : dict
        Configuration dictionary (N_rec_E, N_rec_I, etc.)
    
    Returns:
    --------
    dict : Statistics about the weight matrices
    """
    print("\n" + "="*60)
    print("NETWORK STRUCTURE ANALYSIS")
    print("="*60)
    
    # Create subfolder
    structure_folder = os.path.join(run_folder, "Figures", "network_structure")
    os.makedirs(structure_folder, exist_ok=True)
    
    # Extract data
    input_weights = data.get('weight_poisson')  # Input weights
    weight_rec = data.get('weight_rec')  # Recurrent weights
    N_rec_E = config.get('N_rec_E', 100)
    N_rec_I = config.get('N_rec_I', 100)
    
    if input_weights is None or weight_rec is None:
        print("   WARNING: Weight data not available, skipping network structure plots")
        return {}
    
    try:
        # Create figure with 2x2 subplots
        fig1, axs1 = plt.subplots(2, 2, figsize=(14, 12))
        fig1.suptitle('Network Structure: Weight Matrices and Distributions', fontsize=16, fontweight='bold')
        
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
        
        axs1[0, 0].set_title('Input Weight Matrix (Input->Rec)')
        axs1[0, 0].set_xlabel('Input Dimension')
        axs1[0, 0].set_ylabel('Target Neuron')
        plt.colorbar(im1, ax=axs1[0, 0])
        
        # Top right: Input weight distribution
        E_to_E_weights = input_weights[:N_rec_E, :].flatten()
        I_to_E_weights = input_weights[N_rec_E:, :].flatten()
        
        axs1[0, 1].hist(E_to_E_weights, bins=30, alpha=0.7, color='red', 
                       label='E<-input', density=True, histtype='step', linewidth=2)
        axs1[0, 1].hist(I_to_E_weights, bins=30, alpha=0.7, color='blue', 
                       label='I<-input', density=True, histtype='step', linewidth=2)
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
        axs1[1, 0].text(N_rec_E//2, N_rec_E//2, 'E->E', ha='center', va='center', 
                       color='red', fontsize=12, fontweight='bold', alpha=0.8)
        axs1[1, 0].text(N_rec_E + N_rec_I//2, N_rec_E//2, 'I->E', ha='center', va='center', 
                       color='blue', fontsize=12, fontweight='bold', alpha=0.8)
        axs1[1, 0].text(N_rec_E//2, N_rec_E + N_rec_I//2, 'E->I', ha='center', va='center', 
                       color='red', fontsize=12, fontweight='bold', alpha=0.8)
        axs1[1, 0].text(N_rec_E + N_rec_I//2, N_rec_E + N_rec_I//2, 'I->I', ha='center', va='center', 
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
                       label='E->E', density=True, histtype='step', linewidth=2)
        axs1[1, 1].hist(E_to_I_rec, bins=30, alpha=0.7, color='orange', 
                       label='E->I', density=True, histtype='step', linewidth=2)
        axs1[1, 1].hist(I_to_E_rec, bins=30, alpha=0.7, color='blue', 
                       label='I->E', density=True, histtype='step', linewidth=2)
        axs1[1, 1].hist(I_to_I_rec, bins=30, alpha=0.7, color='purple', 
                       label='I->I', density=True, histtype='step', linewidth=2)
        axs1[1, 1].axvline(0, color='black', linestyle='--', alpha=0.5)
        axs1[1, 1].set_title('Recurrent Weight Distribution')
        axs1[1, 1].set_xlabel('Weight Value')
        axs1[1, 1].set_ylabel('Density')
        axs1[1, 1].legend()
        axs1[1, 1].set_xlim(-np.max(np.abs(weight_rec)), np.max(np.abs(weight_rec)))
        axs1[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(structure_folder, 'weight_matrices_and_distributions.png')
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
        
        return stats
        
    except Exception as e:
        print(f"   Error in network structure analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    """Command-line interface for standalone network structure analysis"""
    import pickle
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python network_structure.py <run_folder>")
        sys.exit(1)
    
    run_folder = sys.argv[1]
    data_file = os.path.join(run_folder, "Data", "simulation_data.pkl")
    
    if not os.path.exists(data_file):
        print(f"Error: No data file found at {data_file}")
        sys.exit(1)
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    config = data.get('config', {})
    stats = generate_network_structure_plots(run_folder, data, config)
    print("\nNetwork structure analysis complete!")

