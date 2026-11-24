"""
Aggregate Analysis Module

Analyzes data across multiple simulation runs from a parameter sweep.
Generates summary plots showing how metrics vary with parameters.

Key Features:
- Loads data from all runs in a sweep folder
- Generates aggregate plots (e.g., selectivity vs. parameter value)
- Supports 1D and 2D parameter sweeps
- Customizable plot types

Usage:
    python aggregate_analysis.py <sweep_folder>
    
Or programmatically:
    analyzer = AggregateAnalyzer(sweep_folder)
    analyzer.generate_all_plots()
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from scipy import stats
import seaborn as sns


class AggregateAnalyzer:
    """
    Analyzes aggregate statistics across a parameter sweep.
    """
    
    def __init__(self, sweep_folder):
        """
        Initialize analyzer with path to sweep folder.
        
        Parameters:
        -----------
        sweep_folder : str
            Path to the parameter sweep folder
        """
        self.sweep_folder = sweep_folder
        
        # Load metadata
        metadata_file = os.path.join(sweep_folder, "sweep_metadata.pkl")
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load aggregate stats
        aggregate_file = os.path.join(sweep_folder, "aggregate_stats.pkl")
        with open(aggregate_file, 'rb') as f:
            self.all_stats = pickle.load(f)
        
        # Check if we have any successful runs
        if len(self.all_stats) == 0:
            print(f"\n{'='*60}")
            print(f"ERROR: NO SUCCESSFUL RUNS")
            print(f"{'='*60}")
            print(f"Sweep: {self.metadata['sweep_name']}")
            print(f"Expected runs: {self.metadata['n_runs']}")
            print(f"Successful runs: 0")
            print(f"\nNo data to analyze. Check error logs in run folders.")
            print(f"{'='*60}\n")
            raise ValueError("No successful runs to analyze")
        
        # Create analysis folder
        self.analysis_folder = os.path.join(sweep_folder, "AggregateAnalysis")
        os.makedirs(self.analysis_folder, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"AGGREGATE ANALYSIS")
        print(f"{'='*60}")
        print(f"Sweep: {self.metadata['sweep_name']}")
        print(f"Total runs: {len(self.all_stats)}")
        print(f"Varied parameters: {list(self.metadata['varied_params'].keys())}")
        print(f"{'='*60}\n")
    
    def get_varied_param_values(self, param_name):
        """Extract all values of a varied parameter across runs"""
        return np.array([run['params'][param_name] for run in self.all_stats])
    
    def get_metric_values(self, metric_name):
        """Extract all values of a metric across runs"""
        return np.array([run[metric_name] for run in self.all_stats])
    
    def plot_metric_vs_parameter(self, param_name, metric_name, 
                                  ylabel=None, title=None, save_name=None):
        """
        Plot a metric vs. a parameter (1D plot).
        
        Parameters:
        -----------
        param_name : str
            Name of the parameter to plot on x-axis
        metric_name : str
            Name of the metric to plot on y-axis
        ylabel : str, optional
            Label for y-axis
        title : str, optional
            Plot title
        save_name : str, optional
            Filename for saving (without path)
        """
        param_values = self.get_varied_param_values(param_name)
        metric_values = self.get_metric_values(metric_name)
        
        # Group by parameter value (in case of duplicates)
        unique_params = np.unique(param_values)
        mean_metrics = []
        std_metrics = []
        
        for param_val in unique_params:
            mask = param_values == param_val
            mean_metrics.append(np.mean(metric_values[mask]))
            std_metrics.append(np.std(metric_values[mask]))
        
        mean_metrics = np.array(mean_metrics)
        std_metrics = np.array(std_metrics)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot with individual points
        ax.scatter(param_values, metric_values, alpha=0.5, s=50, 
                  color='steelblue', edgecolors='black', linewidth=0.5,
                  label='Individual runs')
        
        # Line plot with error bars for means
        if len(unique_params) > 1:
            ax.errorbar(unique_params, mean_metrics, yerr=std_metrics,
                       fmt='o-', color='red', linewidth=2, markersize=8,
                       capsize=5, capthick=2, label='Mean ± Std')
        
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel(ylabel or metric_name, fontsize=12)
        ax.set_title(title or f'{metric_name} vs {param_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save
        save_name = save_name or f"{metric_name}_vs_{param_name}.png"
        save_path = os.path.join(self.analysis_folder, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {save_name}")
        
        return fig
    
    def plot_metric_vs_two_parameters(self, param1_name, param2_name, metric_name,
                                       cmap='viridis', title=None, save_name=None):
        """
        Plot a metric vs. two parameters (2D heatmap).
        
        Parameters:
        -----------
        param1_name : str
            Name of first parameter (x-axis)
        param2_name : str
            Name of second parameter (y-axis)
        metric_name : str
            Name of metric (color)
        cmap : str
            Colormap name
        title : str, optional
            Plot title
        save_name : str, optional
            Filename for saving
        """
        param1_values = self.get_varied_param_values(param1_name)
        param2_values = self.get_varied_param_values(param2_name)
        metric_values = self.get_metric_values(metric_name)
        
        # Get unique parameter values
        unique_param1 = np.unique(param1_values)
        unique_param2 = np.unique(param2_values)
        
        # Create grid
        grid = np.full((len(unique_param2), len(unique_param1)), np.nan)
        
        for i, p1 in enumerate(unique_param1):
            for j, p2 in enumerate(unique_param2):
                mask = (param1_values == p1) & (param2_values == p2)
                if np.any(mask):
                    grid[j, i] = np.mean(metric_values[mask])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(grid, cmap=cmap, aspect='auto', origin='lower')
        
        # Set ticks
        ax.set_xticks(np.arange(len(unique_param1)))
        ax.set_yticks(np.arange(len(unique_param2)))
        ax.set_xticklabels(unique_param1)
        ax.set_yticklabels(unique_param2)
        
        ax.set_xlabel(param1_name, fontsize=12)
        ax.set_ylabel(param2_name, fontsize=12)
        ax.set_title(title or f'{metric_name} vs {param1_name} and {param2_name}',
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name, fontsize=12)
        
        # Add text annotations
        for i in range(len(unique_param1)):
            for j in range(len(unique_param2)):
                if not np.isnan(grid[j, i]):
                    text = ax.text(i, j, f'{grid[j, i]:.2f}',
                                 ha="center", va="center", color="white",
                                 fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        
        # Save
        save_name = save_name or f"{metric_name}_heatmap_{param1_name}_{param2_name}.png"
        save_path = os.path.join(self.analysis_folder, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {save_name}")
        
        return fig
    
    def plot_distribution_evolution(self, param_name, distribution_name,
                                    n_bins=30, title=None, save_name=None):
        """
        Plot how a distribution (e.g., selectivity distribution) changes with parameter.
        
        Parameters:
        -----------
        param_name : str
            Parameter to group by
        distribution_name : str
            Name of distribution ('selectivity_distribution', 'firing_rate_distribution', etc.)
        n_bins : int
            Number of histogram bins
        title : str, optional
            Plot title
        save_name : str, optional
            Filename for saving
        """
        param_values = self.get_varied_param_values(param_name)
        unique_params = np.unique(param_values)
        
        # Create subplots
        n_params = len(unique_params)
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5), sharey=True)
        if n_params == 1:
            axes = [axes]
        
        for idx, param_val in enumerate(unique_params):
            ax = axes[idx]
            
            # Get all distributions for this parameter value
            mask = param_values == param_val
            distributions = [self.all_stats[i][distribution_name] 
                           for i, m in enumerate(mask) if m]
            
            # Combine all distributions
            combined = np.concatenate(distributions)
            
            # Plot histogram
            ax.hist(combined, bins=n_bins, alpha=0.7, edgecolor='black', color='steelblue')
            ax.axvline(np.mean(combined), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(combined):.3f}')
            
            ax.set_xlabel(distribution_name.replace('_distribution', '').replace('_', ' ').title(),
                         fontsize=11)
            if idx == 0:
                ax.set_ylabel('Count', fontsize=11)
            ax.set_title(f'{param_name} = {param_val}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title or f'{distribution_name} across {param_name}',
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save
        save_name = save_name or f"{distribution_name}_evolution_{param_name}.png"
        save_path = os.path.join(self.analysis_folder, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {save_name}")
        
        return fig
    
    def plot_summary_dashboard(self, param_name=None):
        """
        Generate a comprehensive summary dashboard showing multiple metrics.
        
        Parameters:
        -----------
        param_name : str, optional
            Primary parameter to plot against. If None, uses first varied parameter.
        """
        if param_name is None:
            param_name = list(self.metadata['varied_params'].keys())[0]
        
        param_values = self.get_varied_param_values(param_name)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        
        metrics = [
            ('mean_selectivity', 'Mean Selectivity Index'),
            ('mean_sparsity', 'Mean Sparsity Index'),
            ('mean_firing_rate', 'Mean Firing Rate (Hz)'),
            ('n_active_neurons', 'Number of Active Neurons'),
            ('std_selectivity', 'Std. Selectivity'),
            ('std_firing_rate', 'Std. Firing Rate')
        ]
        
        for idx, (metric_name, ylabel) in enumerate(metrics, 1):
            ax = plt.subplot(2, 3, idx)
            
            metric_values = self.get_metric_values(metric_name)
            
            # Group by parameter
            unique_params = np.unique(param_values)
            means = []
            stds = []
            for pval in unique_params:
                mask = param_values == pval
                means.append(np.mean(metric_values[mask]))
                stds.append(np.std(metric_values[mask]))
            
            means = np.array(means)
            stds = np.array(stds)
            
            # Plot
            ax.scatter(param_values, metric_values, alpha=0.5, s=40, color='steelblue')
            if len(unique_params) > 1:
                ax.errorbar(unique_params, means, yerr=stds, fmt='o-', color='red',
                           linewidth=2, markersize=8, capsize=5)
            
            ax.set_xlabel(param_name, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(ylabel, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Summary Dashboard: {self.metadata["sweep_name"]}',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.analysis_folder, 'summary_dashboard.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: summary_dashboard.png")
        
        return fig
    
    def generate_all_plots(self):
        """Generate ONE consolidated figure"""
        print("\nGenerating aggregate plot...")
        
        varied_params = list(self.metadata['varied_params'].keys())
        
        # If E and I are varied together (paired), combine them
        if 'N_rec_E' in varied_params and 'N_rec_I' in varied_params:
            # Check if they're paired (same values)
            e_vals = self.get_varied_param_values('N_rec_E')
            i_vals = self.get_varied_param_values('N_rec_I')
            if np.array_equal(e_vals, i_vals):
                # They're paired - use total size
                print("Detected paired E/I parameters - using total network size")
                primary_param = 'N_rec_E'  # We'll compute total size from this
            else:
                primary_param = varied_params[0]
        else:
            primary_param = varied_params[0]
        
        # Generate single comprehensive figure
        try:
            self.plot_comprehensive_results(primary_param)
            print(f"\nPlot saved to: {self.analysis_folder}/results.png")
        except Exception as e:
            print(f"   Error: {str(e)[:80]}")
    
    def plot_comprehensive_results(self, param_name):
        """Create ONE comprehensive figure with all important results"""
        param_values = self.get_varied_param_values(param_name)
        
        # If this is N_rec_E and N_rec_I are paired, use total size
        if param_name == 'N_rec_E' and 'N_rec_I' in self.metadata['varied_params']:
            i_values = self.get_varied_param_values('N_rec_I')
            if np.array_equal(param_values, i_values):
                param_values = param_values + i_values  # Total network size
                xlabel = 'Total Network Size (E + I neurons)'
            else:
                xlabel = param_name
        else:
            xlabel = param_name
        
        unique_params = np.unique(param_values)
        n_params = len(unique_params)
        
        # Create figure: 2 rows x n_params columns
        fig, axes = plt.subplots(2, n_params, figsize=(5 * n_params, 10))
        
        # Top row: 3 key metrics (first 3 columns) - showing E, I, and All
        metrics = [
            ('mean_selectivity', 'Mean Selectivity Index'),
            ('mean_sparsity', 'Mean Sparsity Index'),
            ('mean_firing_rate', 'Mean Firing Rate (Hz)')
        ]
        
        for idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[0, idx]
            
            # Calculate for All, E, and I separately
            for pop_type, color, label in [('all', 'black', 'All'), 
                                            ('E', 'red', 'E neurons'), 
                                            ('I', 'blue', 'I neurons')]:
                means = []
                stds = []
                
                for pval in unique_params:
                    mask = param_values == pval
                    run_indices = np.where(mask)[0]
                    
                    # Get metric values for this population
                    if pop_type == 'all':
                        vals = [self.all_stats[i][metric] for i in run_indices]
                    else:
                        # Calculate from distributions or rate matrix - split by E/I
                        pop_vals = []
                        for i in run_indices:
                            n_e = self.all_stats[i]['params']['N_rec_E']
                            n_total = n_e + self.all_stats[i]['params']['N_rec_I']
                            
                            if 'sparsity' in metric:
                                # For sparsity: calculate per stimulus, then average
                                # Need to get rate matrix and compute sparsity per stimulus for E or I
                                if 'rate_matrices_arr' in self.all_stats[i]:
                                    rate_matrix = self.all_stats[i]['rate_matrices_arr'][-1]
                                    
                                    # Split by population
                                    if pop_type == 'E':
                                        pop_matrix = rate_matrix[:n_e, :]
                                    else:
                                        pop_matrix = rate_matrix[n_e:, :]
                                    
                                    # Calculate sparsity per stimulus (column), then average
                                    import sparsity_and_selectivity as ss
                                    n_stimuli = pop_matrix.shape[1]
                                    sparsity_per_stim = [ss.sparsity_index(stim_idx, pop_matrix) 
                                                        for stim_idx in range(n_stimuli)]
                                    pop_vals.append(np.mean(sparsity_per_stim))
                                else:
                                    # Fallback: use saved distribution
                                    full_dist = self.all_stats[i]['sparsity_distribution']
                                    if pop_type == 'E':
                                        pop_dist = full_dist[:n_e]
                                    else:
                                        pop_dist = full_dist[n_e:n_total]
                                    pop_vals.append(np.mean(pop_dist))
                            else:
                                # For selectivity and firing rate: use distributions
                                if 'selectivity' in metric:
                                    full_dist = self.all_stats[i]['selectivity_distribution']
                                elif 'firing_rate' in metric:
                                    full_dist = self.all_stats[i]['firing_rate_distribution']
                                
                                # Split by population
                                if pop_type == 'E':
                                    pop_dist = full_dist[:n_e]
                                else:
                                    pop_dist = full_dist[n_e:n_total]
                                
                                pop_vals.append(np.mean(pop_dist))
                        
                        vals = pop_vals
                    
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                
                means = np.array(means)
                stds = np.array(stds)
                
                # Plot
                ax.errorbar(unique_params, means, yerr=stds, fmt='o-', color=color,
                           linewidth=2, markersize=8, capsize=4, label=label, alpha=0.8)
            
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(ylabel, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide extra panels in top row
        for idx in range(len(metrics), n_params):
            axes[0, idx].axis('off')
        
        # Bottom row: Selectivity distributions at each size
        for idx, param_val in enumerate(unique_params):
            ax = axes[1, idx]
            
            mask = param_values == param_val
            distributions = [self.all_stats[i]['selectivity_distribution'] 
                           for i, m in enumerate(mask) if m]
            combined = np.concatenate(distributions)
            
            ax.hist(combined, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
            ax.axvline(np.mean(combined), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(combined):.3f}')
            ax.set_xlabel('Selectivity Index', fontsize=11)
            if idx == 0:
                ax.set_ylabel('Count', fontsize=11)
            ax.set_title(f'{xlabel} = {int(param_val)}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'Network Analysis: {xlabel}', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = os.path.join(self.analysis_folder, 'results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_main_results(self, param_name):
        """Create consolidated 2x2 figure with main results"""
        param_values = self.get_varied_param_values(param_name)
        
        # If this is N_rec_E and N_rec_I are paired, use total size
        if param_name == 'N_rec_E' and 'N_rec_I' in self.metadata['varied_params']:
            i_values = self.get_varied_param_values('N_rec_I')
            if np.array_equal(param_values, i_values):
                param_values = param_values + i_values  # Total network size
                xlabel = 'Total Network Size (E + I neurons)'
            else:
                xlabel = param_name
        else:
            xlabel = param_name
        
        metrics = [
            ('mean_selectivity', 'Mean Selectivity Index'),
            ('mean_sparsity', 'Mean Sparsity Index'),
            ('mean_firing_rate', 'Mean Firing Rate (Hz)'),
            ('n_active_neurons', 'Number of Active Neurons')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Network Metrics vs {xlabel}', fontsize=16, fontweight='bold')
        
        for idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            metric_values = self.get_metric_values(metric)
            
            # Group by parameter value
            unique_params = np.unique(param_values)
            means = []
            stds = []
            for pval in unique_params:
                mask = param_values == pval
                means.append(np.mean(metric_values[mask]))
                stds.append(np.std(metric_values[mask]))
            
            means = np.array(means)
            stds = np.array(stds)
            
            # Plot
            ax.scatter(param_values, metric_values, alpha=0.5, s=60, color='steelblue')
            ax.errorbar(unique_params, means, yerr=stds, fmt='o-', color='red',
                       linewidth=2, markersize=10, capsize=5, label='Mean ± Std')
            
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(ylabel, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        save_path = os.path.join(self.analysis_folder, 'main_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: main_results.png")
    
    def plot_distributions(self, param_name):
        """Create consolidated distribution evolution figure"""
        param_values = self.get_varied_param_values(param_name)
        
        # If this is N_rec_E and N_rec_I are paired, use total size
        if param_name == 'N_rec_E' and 'N_rec_I' in self.metadata['varied_params']:
            i_values = self.get_varied_param_values('N_rec_I')
            if np.array_equal(param_values, i_values):
                param_values = param_values + i_values
                xlabel = 'Total Network Size'
            else:
                xlabel = param_name
        else:
            xlabel = param_name
        
        unique_params = np.unique(param_values)
        n_params = len(unique_params)
        
        fig, axes = plt.subplots(2, n_params, figsize=(5 * n_params, 10), sharey='row')
        if n_params == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Distribution Evolution: {xlabel}', fontsize=16, fontweight='bold')
        
        # Row 1: Selectivity distributions
        for idx, param_val in enumerate(unique_params):
            ax = axes[0, idx]
            mask = param_values == param_val
            distributions = [self.all_stats[i]['selectivity_distribution'] 
                           for i, m in enumerate(mask) if m]
            combined = np.concatenate(distributions)
            
            ax.hist(combined, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
            ax.axvline(np.mean(combined), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(combined):.3f}')
            ax.set_xlabel('Selectivity Index', fontsize=11)
            if idx == 0:
                ax.set_ylabel('Count', fontsize=11)
            ax.set_title(f'{xlabel} = {int(param_val)}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # Row 2: Firing rate distributions  
        for idx, param_val in enumerate(unique_params):
            ax = axes[1, idx]
            mask = param_values == param_val
            distributions = [self.all_stats[i]['firing_rate_distribution'] 
                           for i, m in enumerate(mask) if m]
            combined = np.concatenate(distributions)
            
            ax.hist(combined, bins=30, alpha=0.7, edgecolor='black', color='coral')
            ax.axvline(np.mean(combined), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(combined):.3f}')
            ax.set_xlabel('Firing Rate (Hz)', fontsize=11)
            if idx == 0:
                ax.set_ylabel('Count', fontsize=11)
            ax.set_title(f'{xlabel} = {int(param_val)}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.analysis_folder, 'distributions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: distributions.png")


def main():
    """Command-line interface"""
    if len(sys.argv) < 2:
        print("Usage: python aggregate_analysis.py <sweep_folder>")
        print("\nExample:")
        print("  python aggregate_analysis.py Sweeps/network_size_sweep_20231015_143022")
        sys.exit(1)
    
    sweep_folder = sys.argv[1]
    
    if not os.path.exists(sweep_folder):
        print(f"Error: Sweep folder not found: {sweep_folder}")
        sys.exit(1)
    
    # Run analysis
    analyzer = AggregateAnalyzer(sweep_folder)
    analyzer.generate_all_plots()
    
    print("\nAggregate analysis complete!")


if __name__ == "__main__":
    main()

