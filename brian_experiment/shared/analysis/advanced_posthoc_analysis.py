"""
Advanced Post-hoc Analysis of Neural Network Simulations

This module provides sophisticated analysis tools for examining saved simulation data,
including correlations between neural properties, population dynamics, and comparative studies.

Author: Neural Network Analysis Pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# Import project modules
import sparsity_and_selectivity as ss
import spike_analysis_utils as sau

class AdvancedNetworkAnalyzer:
    """
    Advanced analysis class for post-hoc examination of neural network simulation data.
    """
    
    def __init__(self, data_folder_path):
        """
        Initialize analyzer with saved simulation data.
        
        Parameters:
        -----------
        data_folder_path : str
            Path to folder containing simulation_data.pkl
        """
        self.data_folder_path = data_folder_path
        self.data = self.load_simulation_data()
        self.setup_analysis_data()
        
    def get_figures_folder(self):
        """Get the Figures subfolder path, creating it if needed."""
        figures_folder = os.path.join(self.data_folder_path, "Figures")
        os.makedirs(figures_folder, exist_ok=True)
        return figures_folder
        
    def load_simulation_data(self):
        """Load simulation data from pickle file."""
        # Try new structure first (Data subfolder)
        data_file = os.path.join(self.data_folder_path, "Data", "simulation_data.pkl")
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            print(f"[OK] Simulation data loaded from: {data_file}")
            return data
        
        # Fallback to old structure (direct in folder)
        data_file = os.path.join(self.data_folder_path, "simulation_data.pkl")
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            print(f"[OK] Simulation data loaded from: {data_file}")
            return data
        else:
            raise FileNotFoundError(f"No simulation data found at: {data_file}")
    
    def setup_analysis_data(self):
        """Extract and organize key data for analysis."""
        # Get the final epoch data for main analyses
        self.final_rate_matrix = self.data['rate_matrices_arr'][-1]  # Last epoch
        self.initial_rate_matrix = self.data['rate_matrices_arr'][0]  # First epoch
        
        # Sparsity and selectivity evolution
        self.sparsity_evolution = self.data['sparsity_vec_arr']  # Shape: (epochs, neurons)
        self.selectivity_evolution = self.data['selectivity_vec_arr']  # Shape: (epochs, neurons)
        
        # Network configuration
        self.config = self.data.get('config', {})
        self.n_neurons = self.final_rate_matrix.shape[0]
        self.n_stimuli = self.final_rate_matrix.shape[1]
        self.n_epochs = len(self.data['rate_matrices_arr'])
        
        # Separate E and I populations (assuming first 100 are E, rest are I)
        self.n_exc = self.config.get('N_rec_E', 100)
        self.n_inh = self.config.get('N_rec_I', 100)
        
        print(f"📊 Analysis setup complete:")
        print(f"   - {self.n_neurons} neurons ({self.n_exc}E + {self.n_inh}I)")
        print(f"   - {self.n_stimuli} stimuli")
        print(f"   - {self.n_epochs} epochs")
    
    def analyze_selectivity_firing_rate_correlation(self, epoch_idx=-1, population='all'):
        """
        Analyze correlation between neural selectivity and firing rate.
        
        Parameters:
        -----------
        epoch_idx : int
            Which epoch to analyze (-1 for final epoch)
        population : str
            'all', 'excitatory', 'inhibitory'
        
        Returns:
        --------
        dict : Analysis results including correlation coefficient, p-value, and plot
        """
        print(f"\n🔬 Analyzing Selectivity-Firing Rate Correlation (Epoch {epoch_idx})")
        
        # Get rate matrix for specified epoch
        rate_matrix = self.data['rate_matrices_arr'][epoch_idx]
        
        # Calculate firing rates (mean across stimuli) and selectivity for each neuron
        firing_rates = np.mean(rate_matrix, axis=1)  # Average firing rate per neuron
        selectivity_values = self.data['selectivity_vec_arr'][epoch_idx, :]  # Selectivity per neuron
        
        # Select population
        if population == 'excitatory':
            mask = slice(0, self.n_exc)
            pop_label = 'Excitatory'
            color = 'red'
        elif population == 'inhibitory':
            mask = slice(self.n_exc, self.n_neurons)
            pop_label = 'Inhibitory' 
            color = 'blue'
        else:
            mask = slice(None)
            pop_label = 'All'
            color = 'purple'
            
        firing_rates_pop = firing_rates[mask]
        selectivity_pop = selectivity_values[mask]
        
        # Remove neurons with zero firing rate (avoid issues with selectivity calculation)
        valid_mask = firing_rates_pop > 0
        firing_rates_clean = firing_rates_pop[valid_mask]
        selectivity_clean = selectivity_pop[valid_mask]
        
        # Calculate correlation
        if len(firing_rates_clean) > 3:  # Need at least a few points
            corr_coeff, p_value = stats.pearsonr(firing_rates_clean, selectivity_clean)
            spearman_coeff, spearman_p = stats.spearmanr(firing_rates_clean, selectivity_clean)
        else:
            corr_coeff = p_value = spearman_coeff = spearman_p = np.nan
            
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot with regression line
        ax1.scatter(firing_rates_clean, selectivity_clean, alpha=0.6, color=color, s=30)
        
        if not np.isnan(corr_coeff):
            # Add regression line
            z = np.polyfit(firing_rates_clean, selectivity_clean, 1)
            p = np.poly1d(z)
            x_line = np.linspace(firing_rates_clean.min(), firing_rates_clean.max(), 100)
            ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
        ax1.set_xlabel('Mean Firing Rate (Hz)', fontsize=12)
        ax1.set_ylabel('Selectivity Index', fontsize=12)
        ax1.set_title(f'{pop_label} Neurons: Selectivity vs Firing Rate\n' + 
                     f'Pearson r = {corr_coeff:.3f}, p = {p_value:.3e}', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Histogram of selectivity values
        ax2.hist(selectivity_clean, bins=20, alpha=0.7, color=color, edgecolor='black')
        ax2.axvline(np.mean(selectivity_clean), color='red', linestyle='--', 
                   label=f'Mean = {np.mean(selectivity_clean):.3f}')
        ax2.axvline(np.median(selectivity_clean), color='orange', linestyle='--',
                   label=f'Median = {np.median(selectivity_clean):.3f}')
        ax2.set_xlabel('Selectivity Index', fontsize=12)
        ax2.set_ylabel('Number of Neurons', fontsize=12)
        ax2.set_title(f'{pop_label} Selectivity Distribution', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        # Save to selectivity_activity_relationship subfolder
        selectivity_activity_folder = os.path.join(self.get_figures_folder(), 'selectivity_activity_relationship')
        os.makedirs(selectivity_activity_folder, exist_ok=True)
        plot_path = os.path.join(selectivity_activity_folder, 
                                f'selectivity_vs_firing_rate_{population}_population.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved to: {plot_path}")
        
        # Return results
        results = {
            'population': population,
            'epoch': epoch_idx,
            'n_neurons_analyzed': len(firing_rates_clean),
            'pearson_correlation': corr_coeff,
            'pearson_p_value': p_value,
            'spearman_correlation': spearman_coeff,
            'spearman_p_value': spearman_p,
            'firing_rate_stats': {
                'mean': np.mean(firing_rates_clean),
                'std': np.std(firing_rates_clean),
                'median': np.median(firing_rates_clean)
            },
            'selectivity_stats': {
                'mean': np.mean(selectivity_clean),
                'std': np.std(selectivity_clean),
                'median': np.median(selectivity_clean)
            },
            'figure': fig
        }
        
        # Print summary
        print(f"📊 Results Summary ({pop_label} neurons):")
        print(f"   - Neurons analyzed: {len(firing_rates_clean)}")
        print(f"   - Pearson correlation: r = {corr_coeff:.3f}, p = {p_value:.3e}")
        print(f"   - Spearman correlation: ρ = {spearman_coeff:.3f}, p = {spearman_p:.3e}")
        print(f"   - Mean firing rate: {np.mean(firing_rates_clean):.2f} ± {np.std(firing_rates_clean):.2f} Hz")
        print(f"   - Mean selectivity: {np.mean(selectivity_clean):.3f} ± {np.std(selectivity_clean):.3f}")
        
        return results
    
    def analyze_selectivity_firing_rate_correlation_combined(self, epoch_idx=-1):
        """
        Analyze correlation between neural selectivity and firing rate for all populations in one plot.
        
        Parameters:
        -----------
        epoch_idx : int
            Which epoch to analyze (-1 for final epoch)
        
        Returns:
        --------
        dict : Analysis results including correlation coefficient, p-value, and plot
        """
        print(f"\n🔬 Analyzing Selectivity-Firing Rate Correlation (Epoch {epoch_idx})")
        
        # Get rate matrix for specified epoch
        rate_matrix = self.data['rate_matrices_arr'][epoch_idx]
        
        # Calculate firing rates (mean across stimuli) and selectivity for each neuron
        firing_rates = np.mean(rate_matrix, axis=1)  # Average firing rate per neuron
        selectivity_values = self.data['selectivity_vec_arr'][epoch_idx, :]  # Selectivity per neuron
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        results = {}
        colors = {'all': 'purple', 'excitatory': 'red', 'inhibitory': 'blue'}
        labels = {'all': 'All', 'excitatory': 'Excitatory', 'inhibitory': 'Inhibitory'}
        
        for population in ['all', 'excitatory', 'inhibitory']:
            # Select population
            if population == 'excitatory':
                mask = slice(0, self.n_exc)
            elif population == 'inhibitory':
                mask = slice(self.n_exc, self.n_neurons)
            else:
                mask = slice(None)
                
            firing_rates_pop = firing_rates[mask]
            selectivity_pop = selectivity_values[mask]
            
            # Remove neurons with zero firing rate (avoid issues with selectivity calculation)
            valid_mask = firing_rates_pop > 0
            firing_rates_clean = firing_rates_pop[valid_mask]
            selectivity_clean = selectivity_pop[valid_mask]
            
            # Calculate correlation
            if len(firing_rates_clean) > 3:  # Need at least a few points
                corr_coeff, p_value = stats.pearsonr(firing_rates_clean, selectivity_clean)
                spearman_coeff, spearman_p = stats.spearmanr(firing_rates_clean, selectivity_clean)
            else:
                corr_coeff = p_value = spearman_coeff = spearman_p = np.nan
                
            # Scatter plot
            ax1.scatter(firing_rates_clean, selectivity_clean, alpha=0.6, 
                      color=colors[population], s=30, label=f'{labels[population]} (n={len(firing_rates_clean)})')
            
            # Add regression line
            if not np.isnan(corr_coeff):
                z = np.polyfit(firing_rates_clean, selectivity_clean, 1)
                p = np.poly1d(z)
                x_line = np.linspace(firing_rates_clean.min(), firing_rates_clean.max(), 100)
                ax1.plot(x_line, p(x_line), color=colors[population], linestyle='--', alpha=0.8, linewidth=2)
            
            # KDE plot of selectivity values (normalized)
            from scipy.stats import gaussian_kde
            if len(selectivity_clean) > 1:
                kde = gaussian_kde(selectivity_clean)
                x_kde = np.linspace(selectivity_clean.min(), selectivity_clean.max(), 100)
                y_kde = kde(x_kde)
                # Normalize the KDE
                y_kde = y_kde / np.trapz(y_kde, x_kde)
                ax2.plot(x_kde, y_kde, color=colors[population], linewidth=2, 
                        label=f'{labels[population]} (n={len(firing_rates_clean)})')
            
            # Store results
            results[population] = {
                'population': population,
                'epoch': epoch_idx,
                'n_neurons_analyzed': len(firing_rates_clean),
                'pearson_correlation': corr_coeff,
                'pearson_p_value': p_value,
                'spearman_correlation': spearman_coeff,
                'spearman_p_value': spearman_p,
                'firing_rate_stats': {
                    'mean': np.mean(firing_rates_clean),
                    'std': np.std(firing_rates_clean),
                    'min': np.min(firing_rates_clean),
                    'max': np.max(firing_rates_clean)
                },
                'selectivity_stats': {
                    'mean': np.mean(selectivity_clean),
                    'std': np.std(selectivity_clean),
                    'min': np.min(selectivity_clean),
                    'max': np.max(selectivity_clean)
                }
            }
            
            print(f"📊 Results Summary ({labels[population]} neurons):")
            print(f"   - Neurons analyzed: {len(firing_rates_clean)}")
            print(f"   - Pearson correlation: r = {corr_coeff:.3f}, p = {p_value:.3e}")
            print(f"   - Spearman correlation: ρ = {spearman_coeff:.3f}, p = {spearman_p:.3e}")
            print(f"   - Mean firing rate: {np.mean(firing_rates_clean):.2f} ± {np.std(firing_rates_clean):.2f} Hz")
            print(f"   - Mean selectivity: {np.mean(selectivity_clean):.3f} ± {np.std(selectivity_clean):.3f}")
        
        ax1.set_xlabel('Mean Firing Rate (Hz)', fontsize=12)
        ax1.set_ylabel('Selectivity Index', fontsize=12)
        ax1.set_title('Selectivity vs Firing Rate: All Populations', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Selectivity Index', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Selectivity Distribution: All Populations (KDE)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to selectivity_activity_relationship subfolder
        selectivity_activity_folder = os.path.join(self.get_figures_folder(), 'selectivity_activity_relationship')
        os.makedirs(selectivity_activity_folder, exist_ok=True)
        plot_path = os.path.join(selectivity_activity_folder, 'selectivity_vs_firing_rate_all_populations_combined.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved to: {plot_path}")
        
        plt.close()
        return results
    
    def create_averaged_rate_matrix(self):
        """
        Create a firing rate matrix averaged across epochs.
        Rows = neurons, Columns = stimuli
        """
        print(f"\n🔬 Creating Averaged Rate Matrix")
        
        # Get all rate matrices across epochs
        rate_matrices = self.data['rate_matrices_arr']  # Shape: (epochs, neurons, stimuli)
        
        # Average across epochs: (neurons, stimuli) - no transpose needed
        averaged_matrix = np.mean(rate_matrices, axis=0)  # Shape: (neurons, stimuli)
        
        print(f"   - Averaged matrix shape: {averaged_matrix.shape} (neurons × stimuli)")
        print(f"   - Mean firing rate: {np.mean(averaged_matrix):.3f} Hz")
        print(f"   - Std firing rate: {np.std(averaged_matrix):.3f} Hz")
        
        return averaged_matrix
    
    def create_zscored_matrix(self, rate_matrix):
        """
        Create z-scored matrix where each neuron's response is z-scored across all stimuli.
        """
        print(f"\n🔬 Creating Z-Scored Matrix")
        
        # Z-score each neuron (row) across stimuli (columns)
        zscored_matrix = np.zeros_like(rate_matrix)
        
        for neuron_idx in range(rate_matrix.shape[0]):
            neuron_responses = rate_matrix[neuron_idx, :]
            if np.std(neuron_responses) > 0:
                zscored_matrix[neuron_idx, :] = (neuron_responses - np.mean(neuron_responses)) / np.std(neuron_responses)
            else:
                zscored_matrix[neuron_idx, :] = 0
        
        print(f"   - Z-scored matrix shape: {zscored_matrix.shape}")
        print(f"   - Mean z-score: {np.mean(zscored_matrix):.3f}")
        print(f"   - Std z-score: {np.std(zscored_matrix):.3f}")
        
        return zscored_matrix
    
    def compute_neuron_similarity_matrix(self, zscored_matrix):
        """
        Compute pairwise similarity between neurons using dot product of z-scored responses.
        """
        print(f"\n🔬 Computing Neuron Similarity Matrix")
        
        # zscored_matrix is already (neurons, stimuli) - no transpose needed
        neuron_responses = zscored_matrix  # Shape: (neurons, stimuli)
        
        # Compute pairwise dot products (similarity matrix)
        similarity_matrix = np.dot(neuron_responses, neuron_responses.T)
        
        print(f"   - Similarity matrix shape: {similarity_matrix.shape}")
        print(f"   - Mean similarity: {np.mean(similarity_matrix):.3f}")
        print(f"   - Max similarity: {np.max(similarity_matrix):.3f}")
        print(f"   - Min similarity: {np.min(similarity_matrix):.3f}")
        
        return similarity_matrix
    
    def apply_hierarchical_clustering(self, rate_matrix):
        """
        Apply hierarchical clustering to reorder neurons based on response similarity.
        Uses z-scored firing-rate responses and Euclidean distance for robustness.
        """
        print(f"\n🔬 Applying Hierarchical Clustering")
        
        # Import here to avoid global dependency issues
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
        
        try:
            # Sanitize: replace non-finite values
            matrix = np.nan_to_num(rate_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Z-score each neuron row across stimuli (avoid div by zero)
            row_means = np.mean(matrix, axis=1)
            row_stds = np.std(matrix, axis=1)
            safe_stds = np.where(row_stds > 0, row_stds, 1.0)
            z_matrix = (matrix - row_means[:, np.newaxis]) / safe_stds[:, np.newaxis]  # shape: (neurons, stimuli)
            
            # Add tiny noise to perfectly constant rows (std == 0 originally) to break ties
            constant_rows = (row_stds <= 0)
            if np.any(constant_rows):
                rng = np.random.default_rng(42)
                z_matrix[constant_rows, :] += rng.normal(0.0, 1e-9, size=(np.sum(constant_rows), z_matrix.shape[1]))
            
            # Compute distances between neurons using Euclidean on z-scored responses
            distances = pdist(z_matrix, metric='euclidean')
            
            # Perform hierarchical clustering (Ward requires Euclidean distances)
            linkage_matrix = linkage(distances, method='ward')
            
            # Get the order of neurons from the dendrogram
            dendro = dendrogram(linkage_matrix, no_plot=True)
            neuron_order = dendro['leaves']
            
            print(f"   - Clustering successful, reordered {len(neuron_order)} neurons")
            return neuron_order
            
        except Exception as e:
            print(f"   - Clustering failed ({e}), using original order")
            return np.arange(rate_matrix.shape[1])
    
    def plot_comprehensive_analysis(self, averaged_matrix, zscored_matrix, similarity_matrix, neuron_order, save_path=None):
        """
        Plot comprehensive analysis with raw data, z-scored data, and clustered similarity matrix.
        All matrices are reordered by the same hierarchical clustering based on neural responses to stimuli.
        """
        print(f"\n🔬 Plotting Comprehensive Analysis")
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        # 1. Raw Rate Matrix (neurons x stimuli) - reordered by clustering
        averaged_reordered = averaged_matrix[neuron_order, :]
        im1 = ax1.imshow(averaged_reordered, cmap='viridis', aspect='auto')
        ax1.set_xlabel('Stimulus Index', fontsize=12)
        ax1.set_ylabel('Neuron Index (Clustered)', fontsize=12)
        ax1.set_title('Raw Firing Rate Matrix\n(Hierarchically Clustered)', fontsize=14)
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Firing Rate (Hz)', fontsize=10)
        
        # 2. Z-scored Matrix (neurons x stimuli) - reordered by clustering
        zscored_reordered = zscored_matrix[neuron_order, :]
        im2 = ax2.imshow(zscored_reordered, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
        ax2.set_xlabel('Stimulus Index', fontsize=12)
        ax2.set_ylabel('Neuron Index (Clustered)', fontsize=12)
        ax2.set_title('Z-scored Response Matrix\n(Hierarchically Clustered)', fontsize=14)
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Z-score', fontsize=10)
        
        # 3. Reordered Similarity Matrix (neurons x neurons) - reordered by clustering
        similarity_reordered = similarity_matrix[np.ix_(neuron_order, neuron_order)]
        im3 = ax3.imshow(similarity_reordered, cmap='viridis', aspect='auto')
        ax3.set_xlabel('Neuron Index (Clustered)', fontsize=12)
        ax3.set_ylabel('Neuron Index (Clustered)', fontsize=12)
        ax3.set_title('Neuron Similarity Matrix\n(Hierarchically Clustered)', fontsize=14)
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Similarity (Dot Product)', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            response_clustering_folder = os.path.join(self.get_figures_folder(), 'response_clustering')
            os.makedirs(response_clustering_folder, exist_ok=True)
            save_path = os.path.join(response_clustering_folder, 'neuron_response_similarity_hierarchical_clustering.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Comprehensive analysis saved to: {save_path}")
        
        plt.close()
        return save_path
    
    def analyze_neuron_response_similarity(self):
        """
        Complete analysis of neuron response similarity.
        """
        print(f"\n🔬 NEURON RESPONSE SIMILARITY ANALYSIS")
        print("="*50)
        
        # 1. Create averaged rate matrix
        averaged_matrix = self.create_averaged_rate_matrix()
        
        # 2. Create z-scored matrix
        zscored_matrix = self.create_zscored_matrix(averaged_matrix)
        
        # 3. Apply hierarchical clustering
        neuron_order = self.apply_hierarchical_clustering(averaged_matrix)
        
        # 4. Compute similarity matrix
        similarity_matrix = self.compute_neuron_similarity_matrix(zscored_matrix)
        
        # 5. Plot comprehensive analysis
        comprehensive_path = self.plot_comprehensive_analysis(
            averaged_matrix, zscored_matrix, similarity_matrix, neuron_order
        )
        
        # 6. Analyze similarity patterns
        # Remove diagonal (self-similarity)
        mask = np.eye(similarity_matrix.shape[0], dtype=bool)
        off_diagonal = similarity_matrix[~mask]
        
        print(f"\n📊 Similarity Analysis:")
        print(f"   - Mean off-diagonal similarity: {np.mean(off_diagonal):.3f}")
        print(f"   - Std off-diagonal similarity: {np.std(off_diagonal):.3f}")
        print(f"   - Max off-diagonal similarity: {np.max(off_diagonal):.3f}")
        print(f"   - Min off-diagonal similarity: {np.min(off_diagonal):.3f}")
        
        # Find most similar neuron pairs
        similarity_no_diag = similarity_matrix.copy()
        np.fill_diagonal(similarity_no_diag, -np.inf)  # Remove diagonal
        max_sim_idx = np.unravel_index(np.argmax(similarity_no_diag), similarity_no_diag.shape)
        print(f"   - Most similar pair: neurons {max_sim_idx[0]} & {max_sim_idx[1]} (similarity = {similarity_matrix[max_sim_idx]:.3f})")
        
        return {
            'averaged_matrix': averaged_matrix,
            'zscored_matrix': zscored_matrix,
            'similarity_matrix': similarity_matrix,
            'neuron_order': neuron_order,
            'comprehensive_path': comprehensive_path,
            'stats': {
                'mean_similarity': np.mean(off_diagonal),
                'std_similarity': np.std(off_diagonal),
                'max_similarity': np.max(off_diagonal),
                'min_similarity': np.min(off_diagonal),
                'most_similar_pair': max_sim_idx
            }
        }

    # ================= Functional Connectivity Index (per postsynaptic neuron) =================
    def _normalize_tuning_rows(self, R: np.ndarray) -> np.ndarray:
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

    def compute_cci(self, weight_matrix: np.ndarray, response_matrix: np.ndarray, p: float = 1.0,
                    excitatory_count: int | None = None):
        """
        Compute Co-tuning Connectivity Index (CCI) per postsynaptic neuron.
        - weight_matrix: W (neurons × neurons), where W[i, j] is i → j synapse.
        - response_matrix: R (neurons × stimuli), rows are neurons, cols are stimuli.
        - p: exponent for |W|^p normalization (p=1 default).
        - excitatory_count: number of excitatory neurons (first N_exc are E), infer from self.n_exc if None.
        Returns dict with cci, cci_E, cci_I, sigma (cosine similarities), alpha (normalized weights).
        """
        print("\n🔬 Computing Co-tuning Connectivity Index (CCI)")
        W = np.asarray(weight_matrix)
        R = np.asarray(response_matrix)
        assert W.shape[0] == W.shape[1] == R.shape[0], "W must be (N×N) and R must be (N×S) with same N"

        # Normalize tuning vectors (rows)
        R_tilde = self._normalize_tuning_rows(R)
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
        if excitatory_count is None:
            excitatory_count = int(self.n_exc)
        N = W.shape[0]
        exc_mask = np.zeros(N, dtype=bool)
        exc_mask[:excitatory_count] = True
        inh_mask = ~exc_mask

        # For separated components, use absolute value of sigma: pure weighted correlation magnitude
        cci_E = (alpha[exc_mask, :] * np.abs(sigma[exc_mask, :])).sum(axis=0)
        cci_I = (alpha[inh_mask, :] * np.abs(sigma[inh_mask, :])).sum(axis=0)

        print(f"   - CCI computed for {N} neurons (p={p})")
        print(f"   - CCI range: [{np.min(cci):.3f}, {np.max(cci):.3f}]  mean={np.mean(cci):.3f}")

        return {
            'cci': cci,
            'cci_E': cci_E,
            'cci_I': cci_I,
            'sigma': sigma,
            'alpha': alpha
        }

    def _plot_cci_kde(self, ax, cci: np.ndarray):
        """Plot normalized KDE of CCI on provided axis (fallback to histogram)."""
        try:
            from scipy.stats import gaussian_kde
            x = np.linspace(np.min(cci), np.max(cci), 200)
            kde = gaussian_kde(cci)
            y = kde(x)
            y = y / np.trapz(y, x)
            ax.plot(x, y, color='black', label='CCI density')
            ax.set_title('CCI (KDE)')
            ax.set_xlabel('CCI')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            ax.legend()
        except Exception:
            ax.hist(cci, bins=30, color='gray', edgecolor='black')
            ax.set_title('CCI (Histogram)')
            ax.set_xlabel('CCI')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)

    def _plot_cci_vs_selectivity(self, ax, cci: np.ndarray, selectivity: np.ndarray, title_prefix: str = 'CCI'):
        """Scatter CCI vs selectivity with Pearson/Spearman and regression line, removing outliers."""
        from scipy import stats
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
            ax.set_title(f'Selectivity vs {title_prefix}\nPearson r={pearson_r:.3f} (p={pearson_p:.1e}), Spearman rho={spearman_r:.3f} (p={spearman_p:.1e})')
        ax.set_xlabel(title_prefix)
        ax.set_ylabel('Selectivity Index')
        ax.grid(True, alpha=0.3)

    def analyze_connectivity_index(self, p: float = 1.0):
        """
        High-level wrapper: pull W and R, compute CCI, and plot correlation with selectivity.
        """
        print("\n🔬 CO-TUNING CONNECTIVITY INDEX ANALYSIS")
        print("=" * 50)

        # Responses: averaged across epochs (neurons × stimuli)
        R = self.create_averaged_rate_matrix()

        # Weights: prefer recurrent
        W_rec = self.data.get('weight_rec', None)
        if W_rec is None:
            raise ValueError("No recurrent weight matrix 'weight_rec' found in saved data")

        # Compute CCI
        cci_data = self.compute_cci(W_rec, R, p=p, excitatory_count=self.n_exc)
        cci = cci_data['cci']

        # Selectivity (final epoch per neuron)
        selectivity_vec = None
        if 'selectivity_vec_arr' in self.data and isinstance(self.data['selectivity_vec_arr'], np.ndarray):
            selectivity_vec = self.data['selectivity_vec_arr'][-1, :]
            print(f"   - Loaded selectivity vector (final epoch) shape: {selectivity_vec.shape}")
            # Check if shape matches number of neurons
            if selectivity_vec.shape[0] != R.shape[0]:
                print(f"   - WARNING: selectivity shape {selectivity_vec.shape} doesn't match neurons {R.shape[0]}, computing from rate matrix")
                selectivity_vec = None
        
        if selectivity_vec is None:
            # Compute selectivity from rate matrix
            import sparsity_and_selectivity as ss
            _, selectivity_vec = ss.create_vecs(R)
            selectivity_vec = np.array(selectivity_vec)
            print("   - Computed selectivity from rate matrix")
            print(f"   - Selectivity vector shape: {selectivity_vec.shape}")

        # Plot: 4 subplots in correct order
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Combined CCI_E and CCI_I histograms/KDEs on same panel
        try:
            from scipy.stats import gaussian_kde
            # Plot CCI_E
            kde_E = gaussian_kde(cci_data['cci_E'])
            x_E = np.linspace(np.min(cci_data['cci_E']), np.max(cci_data['cci_E']), 200)
            y_E = kde_E(x_E)
            ax1.plot(x_E, y_E, color='red', linewidth=2, label='CCI_E (Excitatory inputs)')
            ax1.fill_between(x_E, y_E, alpha=0.3, color='red')
            
            # Plot CCI_I
            kde_I = gaussian_kde(cci_data['cci_I'])
            x_I = np.linspace(np.min(cci_data['cci_I']), np.max(cci_data['cci_I']), 200)
            y_I = kde_I(x_I)
            ax1.plot(x_I, y_I, color='blue', linewidth=2, label='CCI_I (Inhibitory inputs)')
            ax1.fill_between(x_I, y_I, alpha=0.3, color='blue')
            
            ax1.set_xlabel('CCI', fontsize=11)
            ax1.set_ylabel('Density', fontsize=11)
            ax1.legend()
        except:
            # Fallback to histograms
            ax1.hist(cci_data['cci_E'], bins=30, alpha=0.5, color='red', label='CCI_E (Excitatory)', density=True)
            ax1.hist(cci_data['cci_I'], bins=30, alpha=0.5, color='blue', label='CCI_I (Inhibitory)', density=True)
            ax1.set_xlabel('CCI', fontsize=11)
            ax1.set_ylabel('Density', fontsize=11)
            ax1.legend()
        ax1.set_title('CCI Distributions: Excitatory vs Inhibitory Inputs', fontsize=12)
        
        # 2. CCI_E vs selectivity (excitatory co-tuning)
        self._plot_cci_vs_selectivity(ax2, cci_data['cci_E'], selectivity_vec, 'CCI_E (Excitatory)')
        
        # 3. CCI_I vs selectivity (inhibitory co-tuning) 
        self._plot_cci_vs_selectivity(ax3, cci_data['cci_I'], selectivity_vec, 'CCI_I (Inhibitory)')
        
        # 4. Net CCI (Excitatory - Inhibitory) vs selectivity
        cci_net = cci_data['cci_E'] - cci_data['cci_I']
        self._plot_cci_vs_selectivity(ax4, cci_net, selectivity_vec, 'CCI_E - CCI_I (Net)')

        # Save to connectivity_tuning subfolder with meaningful name
        connectivity_tuning_folder = os.path.join(self.get_figures_folder(), 'connectivity_tuning')
        os.makedirs(connectivity_tuning_folder, exist_ok=True)
        cci_fig_path = os.path.join(connectivity_tuning_folder, 'cotuning_connectivity_index_excitatory_inhibitory.png')
        plt.tight_layout()
        plt.savefig(cci_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"📈 CCI plots saved to: {cci_fig_path}")

        return {
            'cci': cci,
            'cci_E': cci_data['cci_E'],
            'cci_I': cci_data['cci_I'],
            'selectivity': selectivity_vec,
            'figure_path': cci_fig_path
        }
    
    def analyze_population_dynamics_evolution(self):
        """
        Analyze how population-level dynamics evolve over epochs.
        """
        print(f"\n🔬 Analyzing Population Dynamics Evolution")
        
        # Calculate population statistics for each epoch
        epochs = range(self.n_epochs)
        
        # Mean firing rates over epochs
        mean_firing_rates = []
        mean_selectivity_exc = []
        mean_selectivity_inh = []
        mean_sparsity = []
        
        for epoch_idx in range(self.n_epochs):
            rate_matrix = self.data['rate_matrices_arr'][epoch_idx]
            
            # Population firing rates
            mean_firing_rates.append(np.mean(rate_matrix))
            
            # Selectivity by population
            selectivity = self.data['selectivity_vec_arr'][epoch_idx, :]
            mean_selectivity_exc.append(np.mean(selectivity[:self.n_exc]))
            mean_selectivity_inh.append(np.mean(selectivity[self.n_exc:]))
            
            # Sparsity
            sparsity = self.data['sparsity_vec_arr'][epoch_idx, :]
            mean_sparsity.append(np.mean(sparsity))
        
        # Create evolution plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Firing rate evolution
        ax1.plot(epochs, mean_firing_rates, 'o-', color='green', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean Firing Rate (Hz)')
        ax1.set_title('Population Firing Rate Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Selectivity evolution by population
        ax2.plot(epochs, mean_selectivity_exc, 'o-', color='red', linewidth=2, 
                label='Excitatory', markersize=6)
        ax2.plot(epochs, mean_selectivity_inh, 'o-', color='blue', linewidth=2,
                label='Inhibitory', markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Selectivity Index')
        ax2.set_title('Selectivity Evolution by Population')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sparsity evolution
        ax3.plot(epochs, mean_sparsity, 'o-', color='purple', linewidth=2, markersize=6)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Mean Sparsity Index')
        ax3.set_title('Population Sparsity Evolution')
        ax3.grid(True, alpha=0.3)
        
        # Selectivity vs Sparsity relationship
        ax4.scatter(mean_sparsity, np.array(mean_selectivity_exc), color='red', 
                   label='Excitatory', s=60, alpha=0.7)
        ax4.scatter(mean_sparsity, np.array(mean_selectivity_inh), color='blue',
                   label='Inhibitory', s=60, alpha=0.7)
        ax4.set_xlabel('Mean Sparsity Index')
        ax4.set_ylabel('Mean Selectivity Index')
        ax4.set_title('Sparsity-Selectivity Relationship')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to temporal_evolution subfolder
        temporal_evolution_folder = os.path.join(self.get_figures_folder(), 'temporal_evolution')
        os.makedirs(temporal_evolution_folder, exist_ok=True)
        plot_path = os.path.join(temporal_evolution_folder, 'population_mean_sparsity_selectivity_over_epochs.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Population evolution plot saved to: {plot_path}")
        
        return {
            'epochs': epochs,
            'mean_firing_rates': mean_firing_rates,
            'mean_selectivity_exc': mean_selectivity_exc,
            'mean_selectivity_inh': mean_selectivity_inh,
            'mean_sparsity': mean_sparsity,
            'figure': fig
        }
    
    def analyze_stimulus_response_clustering(self, epoch_idx=-1):
        """
        Cluster neurons based on their stimulus response patterns.
        """
        print(f"\n🔬 Analyzing Stimulus Response Clustering (Epoch {epoch_idx})")
        
        # Get rate matrix for specified epoch
        rate_matrix = self.data['rate_matrices_arr'][epoch_idx]
        
        # Only analyze neurons with some activity
        active_neurons = np.where(np.sum(rate_matrix, axis=1) > 0)[0]
        active_rate_matrix = rate_matrix[active_neurons, :]
        
        if len(active_neurons) < 3:
            print("WARNING: Too few active neurons for clustering analysis")
            return None
        
        # Normalize responses for clustering
        normalized_responses = active_rate_matrix / (np.max(active_rate_matrix, axis=1, keepdims=True) + 1e-10)
        
        # Perform hierarchical clustering
        distances = pdist(normalized_responses, metric='cosine')
        linkage_matrix = linkage(distances, method='ward')
        
        # Create clustering visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Dendrogram
        dendrogram(linkage_matrix, ax=ax1, truncate_mode='lastp', p=20)
        ax1.set_title('Neuron Response Clustering Dendrogram')
        ax1.set_xlabel('Neuron Index')
        ax1.set_ylabel('Distance')
        
        # Heatmap of normalized responses
        im = ax2.imshow(normalized_responses, aspect='auto', cmap='viridis')
        ax2.set_title('Normalized Neural Responses')
        ax2.set_xlabel('Stimulus Index')
        ax2.set_ylabel('Neuron Index')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # Response diversity analysis
        response_diversity = np.std(normalized_responses, axis=1)  # Diversity across stimuli
        ax3.hist(response_diversity, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(np.mean(response_diversity), color='red', linestyle='--',
                   label=f'Mean = {np.mean(response_diversity):.3f}')
        ax3.set_xlabel('Response Diversity (std across stimuli)')
        ax3.set_ylabel('Number of Neurons')
        ax3.set_title('Neural Response Diversity Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Population vs individual neuron selectivity
        population_selectivity = []
        individual_selectivity = self.data['selectivity_vec_arr'][epoch_idx, active_neurons]
        
        for stim_idx in range(self.n_stimuli):
            stim_response = active_rate_matrix[:, stim_idx]
            pop_sparsity = ss.sparsity_index(stim_idx, active_rate_matrix)
            population_selectivity.append(pop_sparsity)
        
        ax4.scatter(individual_selectivity, np.mean(active_rate_matrix, axis=1), 
                   alpha=0.6, color='purple', s=30)
        ax4.set_xlabel('Individual Neuron Selectivity')
        ax4.set_ylabel('Mean Firing Rate (Hz)')
        ax4.set_title('Individual Selectivity vs Activity')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to response_clustering subfolder
        response_clustering_folder = os.path.join(self.get_figures_folder(), 'response_clustering')
        os.makedirs(response_clustering_folder, exist_ok=True)
        plot_path = os.path.join(response_clustering_folder, 'stimulus_response_patterns_hierarchical_clustering.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Clustering analysis plot saved to: {plot_path}")
        
        return {
            'active_neurons': active_neurons,
            'linkage_matrix': linkage_matrix,
            'normalized_responses': normalized_responses,
            'response_diversity': response_diversity,
            'population_selectivity': population_selectivity,
            'figure': fig
        }
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive analysis report with selectivity-firing rate analysis only.
        """
        print(f"\n📋 Generating Comprehensive Advanced Analysis Report")
        print("="*60)
        
        results = {}
        
        # 1. Selectivity-Firing Rate Correlations (Combined)
        print("\n1. SELECTIVITY-FIRING RATE ANALYSIS")
        results['selectivity_firing_rate'] = self.analyze_selectivity_firing_rate_correlation_combined()
        
        # 2. Neuron Response Similarity Analysis
        print("\n2. NEURON RESPONSE SIMILARITY ANALYSIS")
        results['neuron_similarity'] = self.analyze_neuron_response_similarity()
        
        # 3. Save summary report
        report_path = os.path.join(self.data_folder_path, 'advanced_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write("ADVANCED NEURAL NETWORK ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Dataset: {self.data_folder_path}\n")
            f.write(f"Network: {self.n_neurons} neurons ({self.n_exc}E + {self.n_inh}I)\n")
            f.write(f"Stimuli: {self.n_stimuli}\n")
            f.write(f"Epochs: {self.n_epochs}\n\n")
            
            # Selectivity-Firing Rate Results
            f.write("SELECTIVITY-FIRING RATE CORRELATIONS\n")
            f.write("-" * 40 + "\n")
            for pop in ['all', 'excitatory', 'inhibitory']:
                res = results['selectivity_firing_rate'][pop]
                f.write(f"{pop.upper()} POPULATION:\n")
                f.write(f"  Pearson r = {res['pearson_correlation']:.3f}, p = {res['pearson_p_value']:.3e}\n")
                f.write(f"  Spearman rho = {res['spearman_correlation']:.3f}, p = {res['spearman_p_value']:.3e}\n")
                f.write(f"  Mean firing rate: {res['firing_rate_stats']['mean']:.2f} Hz\n")
                f.write(f"  Mean selectivity: {res['selectivity_stats']['mean']:.3f}\n\n")
            
            # Neuron Similarity Results
            f.write("NEURON RESPONSE SIMILARITY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            sim_stats = results['neuron_similarity']['stats']
            f.write(f"Mean off-diagonal similarity: {sim_stats['mean_similarity']:.3f}\n")
            f.write(f"Std off-diagonal similarity: {sim_stats['std_similarity']:.3f}\n")
            f.write(f"Max off-diagonal similarity: {sim_stats['max_similarity']:.3f}\n")
            f.write(f"Min off-diagonal similarity: {sim_stats['min_similarity']:.3f}\n")
            f.write(f"Most similar pair: neurons {sim_stats['most_similar_pair'][0]} & {sim_stats['most_similar_pair'][1]}\n\n")
        
        print(f"\n📄 Comprehensive report saved to: {report_path}")
        print("="*60)
        
        return results


def main():
    """
    Main function to run advanced analysis on saved simulation data.
    """
    print("🚀 Advanced Post-hoc Neural Network Analysis")
    print("="*50)
    
    # Available data folders (update these paths as needed)
    available_folders = [
        "Runs/direct_input_REC_stim50_ep5_n200",
        "Runs/direct_input_FF_stim50_ep5_n200", 
        "Runs/direct_input_REC_stim5",
        "Runs/direct_input_REC_stim50",
        # Fallback to old structure
        "brian_experiment/Figures/DI_Sims/direct_input_REC_stim50_ep5_n200",
        "brian_experiment/Figures/DI_Sims/direct_input_FF_stim50_ep5_n200", 
        "brian_experiment/Figures/DI_Sims/direct_input_REC_stim5",
        "brian_experiment/Figures/DI_Sims/direct_input_REC_stim50"
    ]
    
    # Check which folders exist and have data
    valid_folders = []
    for folder in available_folders:
        # Check both new structure (Data subfolder) and old structure (direct)
        data_file_new = os.path.join(folder, "Data", "simulation_data.pkl")
        data_file_old = os.path.join(folder, "simulation_data.pkl")
        if os.path.exists(data_file_new) or os.path.exists(data_file_old):
            valid_folders.append(folder)
    
    if not valid_folders:
        print("❌ No valid simulation data folders found!")
        print("Available folders should contain 'simulation_data.pkl'")
        return
    
    print(f"📁 Found {len(valid_folders)} valid data folders:")
    for i, folder in enumerate(valid_folders):
        print(f"   {i+1}. {folder}")
    
    # Analyze the first available folder (or modify to analyze all)
    folder_to_analyze = valid_folders[0]
    print(f"\n🎯 Analyzing: {folder_to_analyze}")
    
    try:
        # Initialize analyzer
        analyzer = AdvancedNetworkAnalyzer(folder_to_analyze)
        
        # Run comprehensive analysis
        results = analyzer.generate_comprehensive_report()
        
        print("\n✅ Advanced analysis completed successfully!")
        print(f"📊 Results saved in: {folder_to_analyze}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



