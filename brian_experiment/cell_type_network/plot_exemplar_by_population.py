"""
Plot Exemplar Neurons by Population

Creates a figure showing randomly selected exemplar neurons from each population
(layer-cell type combination), with each row representing a different population
and each column showing a different neuron's response to all stimuli.

Usage:
    python plot_exemplar_by_population.py <run_folder>
    python plot_exemplar_by_population.py --config <config_name>
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def get_population_colors():
    """Return color mapping for populations."""
    return {
        'L1_E': '#E74C3C',  # Red
        'L1_I': '#3498DB',  # Blue  
        'L2_E': '#E67E22',  # Orange
        'L2_I': '#9B59B6',  # Purple
        'L3_E': '#27AE60',  # Green
        'L3_I': '#1ABC9C',  # Teal
        'L4_E': '#F39C12',  # Yellow-orange
        'L4_I': '#8E44AD',  # Dark purple
    }


def plot_exemplar_neurons_by_population(run_folder, n_neurons_per_pop=8, seed=42):
    """
    Create exemplar neuron comparison plot organized by population.
    
    Parameters
    ----------
    run_folder : str
        Path to the run directory containing Data/simulation_data.pkl
    n_neurons_per_pop : int
        Number of randomly selected neurons to show per population
    seed : int
        Random seed for reproducibility
    """
    data_path = os.path.join(run_folder, 'Data', 'simulation_data.pkl')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    population_data = data.get('population_data', {})
    config = data.get('config', {})
    
    # Get ordered population names (excluding input layer)
    ordered_pops = []
    for name in sorted(population_data.keys()):
        if name != 'Layer0' and name in population_data:
            ordered_pops.append(name)
    
    if not ordered_pops:
        raise ValueError("No population data found. Run the simulation first to generate population_data.")
    
    n_pops = len(ordered_pops)
    pop_colors = get_population_colors()
    
    # Get number of stimuli from first population's rate matrix
    if ordered_pops:
        first_pop_data = population_data[ordered_pops[0]]
        if 'rate_matrix' in first_pop_data:
            n_stimuli = first_pop_data['rate_matrix'].shape[1]
        elif 'individual_rate_matrices' in first_pop_data and len(first_pop_data['individual_rate_matrices']) > 0:
            n_stimuli = first_pop_data['individual_rate_matrices'][0].shape[1]
        else:
            n_stimuli = config.get('num_stimuli', 20)
    else:
        n_stimuli = config.get('num_stimuli', 20)
    
    # Set random seed
    np.random.seed(seed)
    
    # Create figure
    fig, axes = plt.subplots(n_pops, n_neurons_per_pop, 
                             figsize=(2 * n_neurons_per_pop, 2.5 * n_pops),
                             squeeze=False)
    
    # Create colormap for stimuli
    stim_colors = plt.cm.rainbow(np.linspace(0, 1, n_stimuli))
    
    for row_idx, pop_name in enumerate(ordered_pops):
        pop_data = population_data[pop_name]
        
        # Get average rate matrix and std across epochs
        if 'individual_rate_matrices' in pop_data and len(pop_data['individual_rate_matrices']) > 0:
            # Calculate mean and std across epochs
            rate_matrices_3d = np.array(pop_data['individual_rate_matrices'])  # (n_epochs, n_neurons, n_stimuli)
            avg_rate_matrix = np.mean(rate_matrices_3d, axis=0)  # (n_neurons, n_stimuli)
            std_matrix = np.std(rate_matrices_3d, axis=0)  # (n_neurons, n_stimuli)
        else:
            # Fallback to single rate matrix
            avg_rate_matrix = pop_data['rate_matrix']
            std_matrix = None
        
        selectivity = pop_data['selectivity_vec']
        
        n_neurons = avg_rate_matrix.shape[0]
        n_to_select = min(n_neurons_per_pop, n_neurons)
        
        # Randomly select neurons
        selected_indices = np.random.choice(n_neurons, size=n_to_select, replace=False)
        selected_indices = np.sort(selected_indices)
        
        # Get color for this population
        color = pop_colors.get(pop_name, '#95A5A6')
        
        for col_idx, neuron_idx in enumerate(selected_indices):
            ax = axes[row_idx, col_idx]
            
            # Get responses for this neuron
            avg_rates = avg_rate_matrix[neuron_idx, :]
            sli = selectivity[neuron_idx] if len(selectivity) > neuron_idx else 0.0
            
            # Sort stimuli by firing rate (descending)
            stim_order = np.argsort(avg_rates)[::-1]
            avg_rates_sorted = avg_rates[stim_order]
            colors_sorted = [stim_colors[stim_idx] for stim_idx in stim_order]
            
            x_positions = np.arange(len(stim_order))
            
            # Plot with error bars if available
            if std_matrix is not None:
                std_sorted = std_matrix[neuron_idx, :][stim_order]
                # Error bars in gray
                ax.errorbar(x_positions, avg_rates_sorted, yerr=std_sorted,
                          fmt='none', capsize=2, capthick=1,
                          ecolor='gray', alpha=0.5, elinewidth=1)
            
            # Scatter points colored by stimulus
            ax.scatter(x_positions, avg_rates_sorted, c=colors_sorted,
                      s=20, alpha=0.8, edgecolors='black', linewidth=0.3, zorder=3)
            
            ax.set_xlim(-0.5, len(stim_order) - 0.5)
            ax.set_ylim(bottom=0)
            
            # Title with neuron index and SLI
            if row_idx == 0:
                ax.set_title(f'Neuron {neuron_idx}\nSLI: {sli:.2f}', fontsize=9)
            else:
                ax.set_title(f'N{neuron_idx}, SLI: {sli:.2f}', fontsize=8)
            
            # Y-axis label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f'{pop_name}\n(Hz)', fontsize=10, 
                             color=color, fontweight='bold')
            
            # X-axis label on bottom row - only show 0 and max
            if row_idx == n_pops - 1:
                ax.set_xticks([0, len(stim_order) - 1])
                ax.set_xticklabels(['0', str(len(stim_order) - 1)], fontsize=8)
                ax.set_xlabel('Stimulus (sorted)', fontsize=8)
            else:
                ax.set_xticks([])
            
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for row_idx in range(n_pops):
        for col_idx in range(n_neurons_per_pop):
            if row_idx < len(ordered_pops):
                pop_name = ordered_pops[row_idx]
                pop_data = population_data[pop_name]
                n_neurons = pop_data['rate_matrix'].shape[0]
                n_to_select = min(n_neurons_per_pop, n_neurons)
                if col_idx >= n_to_select:
                    axes[row_idx, col_idx].axis('off')
    
    fig.suptitle('Exemplar Neuron Responses by Population', 
                fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    figures_folder = os.path.join(run_folder, 'Figures')
    os.makedirs(figures_folder, exist_ok=True)
    output_path = os.path.join(figures_folder, 'exemplar_neurons_by_population.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Plot exemplar neurons organized by population (layer-cell type)'
    )
    parser.add_argument(
        'run_folder',
        nargs='?',
        type=str,
        help='Path to run folder containing Data/simulation_data.pkl'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Config name (e.g., test_two_layers). Will look in Runs/single_runs/<config>/latest'
    )
    parser.add_argument(
        '--neurons', '-n',
        type=int,
        default=8,
        help='Number of neurons to show per population (default: 8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for neuron selection (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Resolve run folder path
    if args.config:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        run_folder = os.path.join(script_dir, 'Runs', 'single_runs', args.config, 'latest')
    elif args.run_folder:
        run_folder = args.run_folder
    else:
        parser.error("Must provide either run_folder or --config")
    
    if not os.path.exists(run_folder):
        print(f"Error: Run folder not found: {run_folder}")
        sys.exit(1)
    
    try:
        plot_exemplar_neurons_by_population(
            run_folder, 
            n_neurons_per_pop=args.neurons,
            seed=args.seed
        )
        print("Plot generation complete!")
    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

