"""
Correlation Analysis Module

This module performs comprehensive correlation analyses using a diagnostic plot
that shows all key relationships in a single comprehensive figure.

Creates plots in the correlation_analysis/ subfolder.

Functions:
- generate_correlation_plots: Main entry point for all correlation analyses
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import diagnose_correlation
import soloist_chorister_analysis
import ensemble_analysis
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import LinAlgError


def plot_weight_vs_correlation(run_folder, data, config):
    """
    Plot relationship between recurrent weights and activity correlations.
    
    Creates a scatter plot showing how synaptic weights relate to firing rate correlations
    between pairs of neurons.
    
    Parameters:
    -----------
    run_folder : str
        Path to the run directory
    data : dict
        Dictionary containing simulation data
    config : dict
        Configuration dictionary
    """
    correlation_folder = os.path.join(run_folder, "Figures", "correlation_analysis")
    os.makedirs(correlation_folder, exist_ok=True)
    
    # Extract data
    final_rate_matrix = data['rate_matrices_arr'][-1]  # Shape: (neurons, stimuli)
    weight_rec = data['weight_rec']  # Shape: (neurons, neurons) - recurrent weights
    weight_input = data.get('weight_poisson', None)  # Shape: (neurons, input_dims) - input weights
    
    N_rec_E = config.get('N_rec_E', 100)
    N_rec_I = config.get('N_rec_I', 100)
    N_total = N_rec_E + N_rec_I
    
    # Calculate mean firing rates and filter active neurons
    mean_firing_rates = np.mean(final_rate_matrix, axis=1)
    active_mask = mean_firing_rates > 0
    active_indices = np.where(active_mask)[0]
    active_rate_matrix = final_rate_matrix[active_mask, :]
    
    # Subsample for large networks to reduce computation
    MAX_PAIRS_PER_TYPE = 5000  # Maximum pairs to analyze per connection type
    subsample = len(active_indices) > 100
    
    # Calculate activity correlation matrix for active neurons
    neural_correlations = np.corrcoef(active_rate_matrix)
    np.fill_diagonal(neural_correlations, np.nan)
    
    # Get connection type for each pair (E->E, E->I, I->E, I->I)
    pairs_by_type = {'E->E': [], 'E->I': [], 'I->E': [], 'I->I': []}
    
    for i, orig_i in enumerate(active_indices):
        for j, orig_j in enumerate(active_indices):
            if i != j:  # Exclude self-connections
                correlation = neural_correlations[i, j]
                
                if not np.isnan(correlation):
                    # Determine connection type
                    if orig_i < N_rec_E and orig_j < N_rec_E:
                        conn_type = 'E->E'
                    elif orig_i < N_rec_E and orig_j >= N_rec_E:
                        conn_type = 'E->I'
                    elif orig_i >= N_rec_E and orig_j < N_rec_E:
                        conn_type = 'I->E'
                    else:
                        conn_type = 'I->I'
                    
                    # Get recurrent weight from neuron j to neuron i
                    rec_weight = weight_rec[orig_i, orig_j]
                    
                    # Get input weight cosine similarity
                    input_cosine_sim = np.nan
                    if weight_input is not None:
                        input_vec_i = weight_input[orig_i, :]
                        input_vec_j = weight_input[orig_j, :]
                        # Cosine similarity = dot product / (norm1 * norm2)
                        norm_i = np.linalg.norm(input_vec_i)
                        norm_j = np.linalg.norm(input_vec_j)
                        if norm_i > 0 and norm_j > 0:
                            input_cosine_sim = np.dot(input_vec_i, input_vec_j) / (norm_i * norm_j)
                    
                    # Get recurrent weight cosine similarity (using outgoing weight vectors)
                    # For each neuron, use the row (outgoing weights to all other neurons)
                    rec_vec_i = weight_rec[orig_i, :]  # Outgoing weights from neuron i
                    rec_vec_j = weight_rec[orig_j, :]  # Outgoing weights from neuron j
                    norm_rec_i = np.linalg.norm(rec_vec_i)
                    norm_rec_j = np.linalg.norm(rec_vec_j)
                    rec_cosine_sim = np.nan
                    if norm_rec_i > 0 and norm_rec_j > 0:
                        rec_cosine_sim = np.dot(rec_vec_i, rec_vec_j) / (norm_rec_i * norm_rec_j)
                    
                    pairs_by_type[conn_type].append({
                        'rec_weight': rec_weight,
                        'input_cosine_sim': input_cosine_sim,
                        'rec_cosine_sim': rec_cosine_sim,
                        'correlation': correlation
                    })
    
    # Subsample if needed
    if subsample:
        import random
        random.seed(42)
        for conn_type in pairs_by_type:
            if len(pairs_by_type[conn_type]) > MAX_PAIRS_PER_TYPE:
                original_count = len(pairs_by_type[conn_type])
                pairs_by_type[conn_type] = random.sample(
                    pairs_by_type[conn_type], MAX_PAIRS_PER_TYPE
                )
                print(f"   Subsampled {conn_type} pairs: {MAX_PAIRS_PER_TYPE} out of {original_count}")
    
    # Create figure: 3 rows (recurrent weight, input cosine sim, rec cosine sim) x 4 columns (connection types)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Weight vs Activity Correlation', fontsize=16, fontweight='bold')
    
    connection_types_order = ['E->E', 'E->I', 'I->E', 'I->I']
    colors = {'E->E': 'red', 'E->I': 'orange', 'I->E': 'purple', 'I->I': 'blue'}
    
    # Collect all correlations to set shared y-axis
    all_correlations_for_yaxis = []
    
    # Row 0: Recurrent Weight vs Correlation (one column per connection type)
    for col_idx, conn_type in enumerate(connection_types_order):
        ax = axes[0, col_idx]
        pairs = pairs_by_type[conn_type]
        
        if len(pairs) > 0:
            rec_weights = np.array([p['rec_weight'] for p in pairs])
            correlations = np.array([p['correlation'] for p in pairs])
            all_correlations_for_yaxis.extend(correlations.tolist())
            
            valid_mask = ~(np.isnan(rec_weights) | np.isnan(correlations))
            if np.sum(valid_mask) > 0:
                # Set x-axis limits based on percentiles to exclude outliers (use 99.5 instead of 99)
                x_data = rec_weights[valid_mask]
                x_min = np.percentile(x_data, 0.5)
                x_max = np.percentile(x_data, 99.5)
                ax.set_xlim(x_min, x_max)
                
                ax.scatter(rec_weights[valid_mask], correlations[valid_mask],
                         alpha=0.4, s=15, c=colors[conn_type], 
                         edgecolors='black', linewidth=0.2)
                
                if np.sum(valid_mask) > 2:
                    z = np.polyfit(rec_weights[valid_mask], correlations[valid_mask], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_min, x_max, 100)
                    ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2)
                    
                    r, p_val = stats.pearsonr(rec_weights[valid_mask], 
                                             correlations[valid_mask])
                    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3e}\nn = {np.sum(valid_mask)}', 
                           transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           verticalalignment='top')
            
            ax.set_xlabel('Recurrent Weight')
            ax.set_ylabel('Activity Correlation')
            ax.set_title(f'{conn_type}: Recurrent Weight')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No {conn_type} connections', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{conn_type}: Recurrent Weight')
    
    # Row 1: Input Weight Cosine Similarity vs Correlation (one column per connection type)
    for col_idx, conn_type in enumerate(connection_types_order):
        ax = axes[1, col_idx]
        pairs = pairs_by_type[conn_type]
        
        if len(pairs) > 0:
            input_cosine_sims = np.array([p['input_cosine_sim'] for p in pairs])
            correlations = np.array([p['correlation'] for p in pairs])
            all_correlations_for_yaxis.extend(correlations.tolist())
            
            if weight_input is not None:
                valid_mask = ~(np.isnan(input_cosine_sims) | np.isnan(correlations))
                if np.sum(valid_mask) > 0:
                    # Set x-axis limits based on percentiles to exclude outliers (use 99.5 instead of 99)
                    x_data = input_cosine_sims[valid_mask]
                    x_min = np.percentile(x_data, 0.5)
                    x_max = np.percentile(x_data, 99.5)
                    ax.set_xlim(x_min, x_max)
                    
                    ax.scatter(input_cosine_sims[valid_mask], correlations[valid_mask],
                             alpha=0.4, s=15, c=colors[conn_type], 
                             edgecolors='black', linewidth=0.2)
                    
                    if np.sum(valid_mask) > 2:
                        z = np.polyfit(input_cosine_sims[valid_mask], 
                                      correlations[valid_mask], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(x_min, x_max, 100)
                        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2)
                        
                        r, p_val = stats.pearsonr(input_cosine_sims[valid_mask], 
                                                 correlations[valid_mask])
                        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3e}\nn = {np.sum(valid_mask)}', 
                               transform=ax.transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               verticalalignment='top')
                
                ax.set_xlabel('Input Weight Cosine Similarity')
                ax.set_ylabel('Activity Correlation')
                ax.set_title(f'{conn_type}: Input Weight Cosine Similarity')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Input weights not available', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{conn_type}: Input Weight Cosine Similarity')
    
    # Row 2: Recurrent Weight Cosine Similarity vs Correlation (one column per connection type)
    for col_idx, conn_type in enumerate(connection_types_order):
        ax = axes[2, col_idx]
        pairs = pairs_by_type[conn_type]
        
        if len(pairs) > 0:
            rec_cosine_sims = np.array([p['rec_cosine_sim'] for p in pairs])
            correlations = np.array([p['correlation'] for p in pairs])
            all_correlations_for_yaxis.extend(correlations.tolist())
            
            valid_mask = ~(np.isnan(rec_cosine_sims) | np.isnan(correlations))
            if np.sum(valid_mask) > 0:
                # Set x-axis limits based on percentiles to exclude outliers (use 99.5 instead of 99)
                x_data = rec_cosine_sims[valid_mask]
                x_min = np.percentile(x_data, 0.5)
                x_max = np.percentile(x_data, 99.5)
                ax.set_xlim(x_min, x_max)
                
                ax.scatter(rec_cosine_sims[valid_mask], correlations[valid_mask],
                         alpha=0.4, s=15, c=colors[conn_type], 
                         edgecolors='black', linewidth=0.2)
                
                if np.sum(valid_mask) > 2:
                    z = np.polyfit(rec_cosine_sims[valid_mask], 
                                  correlations[valid_mask], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_min, x_max, 100)
                    ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2)
                    
                    r, p_val = stats.pearsonr(rec_cosine_sims[valid_mask], 
                                             correlations[valid_mask])
                    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3e}\nn = {np.sum(valid_mask)}', 
                           transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           verticalalignment='top')
            
            ax.set_xlabel('Recurrent Weight Cosine Similarity')
            ax.set_ylabel('Activity Correlation')
            ax.set_title(f'{conn_type}: Recurrent Weight Cosine Similarity')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No {conn_type} connections', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{conn_type}: Recurrent Weight Cosine Similarity')
    
    # Set shared y-axis for all subplots
    if len(all_correlations_for_yaxis) > 0:
        all_corrs_array = np.array(all_correlations_for_yaxis)
        valid_corrs = all_corrs_array[~np.isnan(all_corrs_array)]
        if len(valid_corrs) > 0:
            y_min = np.min(valid_corrs)
            y_max = np.max(valid_corrs)
            # Add small padding
            y_range = y_max - y_min
            y_min = y_min - 0.02 * y_range
            y_max = y_max + 0.02 * y_range
            
            for row in range(3):
                for col in range(4):
                    axes[row, col].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(correlation_folder, 'weight_vs_correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {save_path}")
    
    # Calculate overall statistics
    all_rec_weights = []
    all_input_cosine_sims = []
    all_rec_cosine_sims = []
    all_correlations = []
    
    for conn_type in connection_types_order:
        for pair in pairs_by_type[conn_type]:
            all_rec_weights.append(pair['rec_weight'])
            if weight_input is not None:
                all_input_cosine_sims.append(pair['input_cosine_sim'])
            all_rec_cosine_sims.append(pair['rec_cosine_sim'])
            all_correlations.append(pair['correlation'])
    
    all_rec_weights = np.array(all_rec_weights)
    all_rec_cosine_sims = np.array(all_rec_cosine_sims)
    all_correlations = np.array(all_correlations)
    
    valid_mask = ~(np.isnan(all_rec_weights) | np.isnan(all_correlations))
    rec_r, rec_p = np.nan, np.nan
    if np.sum(valid_mask) > 2:
        rec_r, rec_p = stats.pearsonr(all_rec_weights[valid_mask], 
                                     all_correlations[valid_mask])
        print(f"   Overall recurrent weight vs correlation: r = {rec_r:.3f}, p = {rec_p:.3e}")
    
    input_r, input_p = np.nan, np.nan
    if weight_input is not None:
        all_input_cosine_sims = np.array(all_input_cosine_sims)
        valid_mask = ~(np.isnan(all_input_cosine_sims) | np.isnan(all_correlations))
        if np.sum(valid_mask) > 2:
            input_r, input_p = stats.pearsonr(all_input_cosine_sims[valid_mask], 
                                             all_correlations[valid_mask])
            print(f"   Overall input cosine similarity vs correlation: r = {input_r:.3f}, p = {input_p:.3e}")
    
    rec_cosine_r, rec_cosine_p = np.nan, np.nan
    valid_mask = ~(np.isnan(all_rec_cosine_sims) | np.isnan(all_correlations))
    if np.sum(valid_mask) > 2:
        rec_cosine_r, rec_cosine_p = stats.pearsonr(all_rec_cosine_sims[valid_mask], 
                                                   all_correlations[valid_mask])
        print(f"   Overall recurrent weight cosine similarity vs correlation: r = {rec_cosine_r:.3f}, p = {rec_cosine_p:.3e}")
    
    return {
        'overall_rec_correlation': (rec_r, rec_p),
        'overall_input_cosine_correlation': (input_r, input_p),
        'overall_rec_cosine_correlation': (rec_cosine_r, rec_cosine_p),
        'n_pairs': len(all_correlations)
    }


def compute_weight_correlation_coefficients(data, config):
    """
    Compute correlation coefficients (r values) between weight metrics and activity correlations.
    
    Returns a dictionary with r values for each metric and connection type.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing simulation data
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    dict : Dictionary with structure:
        {
            'rec_weight': {'E->E': r, 'E->I': r, 'I->E': r, 'I->I': r},
            'input_cosine_sim': {'E->E': r, 'E->I': r, 'I->E': r, 'I->I': r},
            'rec_cosine_sim': {'E->E': r, 'E->I': r, 'I->E': r, 'I->I': r},
            'N_neurons': N_total
        }
    """
    # Extract data
    final_rate_matrix = data['rate_matrices_arr'][-1]  # Shape: (neurons, stimuli)
    weight_rec = data['weight_rec']  # Shape: (neurons, neurons) - recurrent weights
    weight_input = data.get('weight_poisson', None)  # Shape: (neurons, input_dims) - input weights
    
    N_rec_E = config.get('N_rec_E', 100)
    N_rec_I = config.get('N_rec_I', 100)
    N_total = N_rec_E + N_rec_I
    
    # Calculate mean firing rates and filter active neurons
    mean_firing_rates = np.mean(final_rate_matrix, axis=1)
    active_mask = mean_firing_rates > 0
    active_indices = np.where(active_mask)[0]
    active_rate_matrix = final_rate_matrix[active_mask, :]
    
    # Calculate activity correlation matrix for active neurons
    neural_correlations = np.corrcoef(active_rate_matrix)
    np.fill_diagonal(neural_correlations, np.nan)
    
    # Get connection type for each pair (E->E, E->I, I->E, I->I)
    pairs_by_type = {'E->E': [], 'E->I': [], 'I->E': [], 'I->I': []}
    
    for i, orig_i in enumerate(active_indices):
        for j, orig_j in enumerate(active_indices):
            if i != j:  # Exclude self-connections
                correlation = neural_correlations[i, j]
                
                if not np.isnan(correlation):
                    # Determine connection type
                    if orig_i < N_rec_E and orig_j < N_rec_E:
                        conn_type = 'E->E'
                    elif orig_i < N_rec_E and orig_j >= N_rec_E:
                        conn_type = 'E->I'
                    elif orig_i >= N_rec_E and orig_j < N_rec_E:
                        conn_type = 'I->E'
                    else:
                        conn_type = 'I->I'
                    
                    # Get recurrent weight from neuron j to neuron i
                    rec_weight = weight_rec[orig_i, orig_j]
                    
                    # Get input weight cosine similarity
                    input_cosine_sim = np.nan
                    if weight_input is not None:
                        input_vec_i = weight_input[orig_i, :]
                        input_vec_j = weight_input[orig_j, :]
                        # Cosine similarity = dot product / (norm1 * norm2)
                        norm_i = np.linalg.norm(input_vec_i)
                        norm_j = np.linalg.norm(input_vec_j)
                        if norm_i > 0 and norm_j > 0:
                            input_cosine_sim = np.dot(input_vec_i, input_vec_j) / (norm_i * norm_j)
                    
                    # Get recurrent weight cosine similarity (using outgoing weight vectors)
                    rec_vec_i = weight_rec[orig_i, :]  # Outgoing weights from neuron i
                    rec_vec_j = weight_rec[orig_j, :]  # Outgoing weights from neuron j
                    norm_rec_i = np.linalg.norm(rec_vec_i)
                    norm_rec_j = np.linalg.norm(rec_vec_j)
                    rec_cosine_sim = np.nan
                    if norm_rec_i > 0 and norm_rec_j > 0:
                        rec_cosine_sim = np.dot(rec_vec_i, rec_vec_j) / (norm_rec_i * norm_rec_j)
                    
                    pairs_by_type[conn_type].append({
                        'rec_weight': rec_weight,
                        'input_cosine_sim': input_cosine_sim,
                        'rec_cosine_sim': rec_cosine_sim,
                        'correlation': correlation
                    })
    
    # Compute correlation coefficients for each metric and connection type
    connection_types_order = ['E->E', 'E->I', 'I->E', 'I->I']
    results = {
        'rec_weight': {},
        'input_cosine_sim': {},
        'rec_cosine_sim': {},
        'N_neurons': N_total
    }
    
    for conn_type in connection_types_order:
        pairs = pairs_by_type[conn_type]
        
        if len(pairs) > 0:
            rec_weights = np.array([p['rec_weight'] for p in pairs])
            input_cosine_sims = np.array([p['input_cosine_sim'] for p in pairs])
            rec_cosine_sims = np.array([p['rec_cosine_sim'] for p in pairs])
            correlations = np.array([p['correlation'] for p in pairs])
            
            # Recurrent weight vs correlation
            valid_mask = ~(np.isnan(rec_weights) | np.isnan(correlations))
            if np.sum(valid_mask) > 2:
                r, _ = stats.pearsonr(rec_weights[valid_mask], correlations[valid_mask])
                results['rec_weight'][conn_type] = r
            else:
                results['rec_weight'][conn_type] = np.nan
            
            # Input cosine similarity vs correlation
            if weight_input is not None:
                valid_mask = ~(np.isnan(input_cosine_sims) | np.isnan(correlations))
                if np.sum(valid_mask) > 2:
                    r, _ = stats.pearsonr(input_cosine_sims[valid_mask], correlations[valid_mask])
                    results['input_cosine_sim'][conn_type] = r
                else:
                    results['input_cosine_sim'][conn_type] = np.nan
            else:
                results['input_cosine_sim'][conn_type] = np.nan
            
            # Recurrent weight cosine similarity vs correlation
            valid_mask = ~(np.isnan(rec_cosine_sims) | np.isnan(correlations))
            if np.sum(valid_mask) > 2:
                r, _ = stats.pearsonr(rec_cosine_sims[valid_mask], correlations[valid_mask])
                results['rec_cosine_sim'][conn_type] = r
            else:
                results['rec_cosine_sim'][conn_type] = np.nan
        else:
            results['rec_weight'][conn_type] = np.nan
            results['input_cosine_sim'][conn_type] = np.nan
            results['rec_cosine_sim'][conn_type] = np.nan
    
    return results


def plot_weight_correlation_vs_network_size(sweep_folder):
    """
    Plot regression lines for weight metrics vs activity correlations across different network sizes.
    
    Creates a 3x4 plot where:
    - Rows: three metrics (recurrent weight, input cosine sim, rec cosine sim)
    - Columns: four connection types (E->E, E->I, I->E, I->I)
    - X-axis: weight metric (e.g., cosine similarity, recurrent weight)
    - Y-axis: activity correlation
    - Multiple regression lines overlaid, one for each network size
    - Color scale: darker colors = larger network size
    
    Parameters:
    -----------
    sweep_folder : str
        Path to the network size sweep folder (e.g., 'Sweeps/network_size_100_to_500_...')
    """
    import pickle
    import glob
    
    # Find all run folders
    run_folders = sorted(glob.glob(os.path.join(sweep_folder, 'run_*')))
    
    if len(run_folders) == 0:
        print(f"Error: No run folders found in {sweep_folder}")
        return
    
    # Load sweep metadata to check if multiple runs per param were used
    metadata_file = os.path.join(sweep_folder, 'sweep_metadata.pkl')
    n_runs_per_param = 1
    if os.path.exists(metadata_file):
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
            n_runs_per_param = metadata.get('n_runs_per_param', 1)
    
    # Collect data from all runs, grouped by parameter combination
    data_by_param = {}  # key: (N_rec_E, N_rec_I), value: list of data/config tuples
    
    for run_folder in run_folders:
        data_file = os.path.join(run_folder, 'Data', 'simulation_data.pkl')
        if not os.path.exists(data_file):
            print(f"Warning: No data file found in {run_folder}")
            continue
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        config = data.get('config', {})
        N_rec_E = config.get('N_rec_E', 100)
        N_rec_I = config.get('N_rec_I', 100)
        N_total = N_rec_E + N_rec_I
        
        param_key = (N_rec_E, N_rec_I)
        if param_key not in data_by_param:
            data_by_param[param_key] = []
        data_by_param[param_key].append({
            'data': data,
            'config': config,
            'N_total': N_total
        })
    
    if len(data_by_param) == 0:
        print("Error: No valid data found")
        return
    
    # Sort by network size and create list of parameter groups
    sorted_param_keys = sorted(data_by_param.keys(), key=lambda x: x[0] + x[1])
    all_data_by_size = []
    for param_key in sorted_param_keys:
        all_data_by_size.append(data_by_param[param_key])
    
    # Set up color scale (darker = larger N)
    n_sizes = len(all_data_by_size)
    cmap = plt.cm.Reds
    colors = [cmap(0.3 + 0.7 * i / (n_sizes - 1)) if n_sizes > 1 else cmap(0.7) for i in range(n_sizes)]
    
    # Get network sizes for legend
    network_sizes_for_legend = []
    for param_group in all_data_by_size:
        if len(param_group) > 0:
            network_sizes_for_legend.append(param_group[0]['N_total'])
    
    # Create figure: 3 rows x 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Weight vs Activity Correlation: Regression Lines Across Network Sizes', 
                 fontsize=16, fontweight='bold')
    
    connection_types_order = ['E->E', 'E->I', 'I->E', 'I->I']
    metrics = ['rec_weight', 'input_cosine_sim', 'rec_cosine_sim']
    metric_titles = ['Recurrent Weight', 'Input Weight Cosine Similarity', 
                     'Recurrent Weight Cosine Similarity']
    
    # Subsample for plotting large datasets
    MAX_POINTS_PER_SIZE = 2000  # Maximum points to plot per network size per connection type
    
    # Collect ALL y values globally across all subplots for shared scale
    all_regression_y_values_global = []
    
    for row_idx, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
        for col_idx, conn_type in enumerate(connection_types_order):
            ax = axes[row_idx, col_idx]
            
            all_x = []
            all_y = []
            
            regression_y_values = []  # Collect y values from regression lines for tighter limits
            
            # Collect data and plot regression lines for each network size
            # If multiple runs per param, average the regression lines
            for size_idx, param_group in enumerate(all_data_by_size):
                if len(param_group) == 0:
                    continue
                
                N_total = param_group[0]['N_total']
                print(f"   Processing {conn_type} {metric_title} for N={N_total}...", flush=True)
                
                # Collect all regression lines for this parameter combination
                all_regression_coeffs = []  # Store (slope, intercept) for each run
                all_x_ranges = []  # Store (x_min, x_max) for each run
                
                for run_idx, size_data in enumerate(param_group):
                    if len(param_group) > 1:
                        print(f"      Run {run_idx+1}/{len(param_group)}...", flush=True)
                    data = size_data['data']
                    config = size_data['config']
                    
                    # Extract pairs for this metric and connection type
                    pairs = extract_pairs_for_metric(data, config, metric, conn_type, max_pairs=50000)
                    
                    if len(pairs) > 0:
                        x_vals = np.array([p[metric] for p in pairs])
                        y_vals = np.array([p['correlation'] for p in pairs])
                        
                        # Subsample if needed
                        if len(pairs) > MAX_POINTS_PER_SIZE:
                            import random
                            random.seed(42)
                            indices = random.sample(range(len(pairs)), MAX_POINTS_PER_SIZE)
                            x_vals = x_vals[indices]
                            y_vals = y_vals[indices]
                        
                        # Filter valid data
                        valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                        if np.sum(valid_mask) > 2:
                            x_valid = x_vals[valid_mask]
                            y_valid = y_vals[valid_mask]
                            
                            all_x.extend(x_valid.tolist())
                            all_y.extend(y_valid.tolist())
                            
                            # Fit regression line
                            z = np.polyfit(x_valid, y_valid, 1)
                            all_regression_coeffs.append((z[0], z[1]))  # (slope, intercept)
                            
                            # Store x range
                            x_min = np.percentile(x_valid, 0.5)
                            x_max = np.percentile(x_valid, 99.5)
                            all_x_ranges.append((x_min, x_max))
                
                # Average regression lines if multiple runs
                if len(all_regression_coeffs) > 0:
                    # Average slopes and intercepts
                    avg_slope = np.mean([coeff[0] for coeff in all_regression_coeffs])
                    avg_intercept = np.mean([coeff[1] for coeff in all_regression_coeffs])
                    p_avg = np.poly1d([avg_slope, avg_intercept])
                    
                    # Use average x range
                    if len(all_x_ranges) > 0:
                        x_min_avg = np.mean([r[0] for r in all_x_ranges])
                        x_max_avg = np.mean([r[1] for r in all_x_ranges])
                    else:
                        # Fallback to percentiles of all x data
                        if len(all_x) > 0:
                            all_x_array = np.array(all_x)
                            x_min_avg = np.percentile(all_x_array, 0.5)
                            x_max_avg = np.percentile(all_x_array, 99.5)
                        else:
                            continue
                    
                    x_line = np.linspace(x_min_avg, x_max_avg, 100)
                    y_line = p_avg(x_line)
                    
                    # Collect y values from regression line for tighter limits
                    regression_y_values.extend(y_line.tolist())
                    all_regression_y_values_global.extend(y_line.tolist())
                    
                    # Plot averaged regression line
                    label = f'N={N_total}'
                    if len(all_regression_coeffs) > 1:
                        label += f' (n={len(all_regression_coeffs)})'
                    ax.plot(x_line, y_line, '-', color=colors[size_idx], 
                           linewidth=2.5, alpha=0.8, 
                           label=label)
            
            # Set axis limits based on regression lines (tighter y-axis)
            if len(all_x) > 0:
                all_x_array = np.array(all_x)
                
                x_min = np.percentile(all_x_array, 1)
                x_max = np.percentile(all_x_array, 99)
                
                x_range = x_max - x_min
                ax.set_xlim(x_min - 0.02 * x_range, x_max + 0.02 * x_range)
                
                # Store y limits for this subplot (will be set globally later)
                if len(regression_y_values) > 0:
                    reg_y_array = np.array(regression_y_values)
                    y_min = np.min(reg_y_array)
                    y_max = np.max(reg_y_array)
                else:
                    # Fallback to data percentiles if no regression lines
                    all_y_array = np.array(all_y)
                    y_min = np.percentile(all_y_array, 2.5)
                    y_max = np.percentile(all_y_array, 97.5)
                
                # Set labels
                if metric == 'rec_weight':
                    ax.set_xlabel('Recurrent Weight')
                elif metric == 'input_cosine_sim':
                    ax.set_xlabel('Input Weight Cosine Similarity')
                elif metric == 'rec_cosine_sim':
                    ax.set_xlabel('Recurrent Weight Cosine Similarity')
                
                ax.set_ylabel('Activity Correlation')
                ax.set_title(f'{conn_type}: {metric_title}')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for {conn_type}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{conn_type}: {metric_title}')
    
    # Set shared y-axis limits across ALL subplots based on global regression values
    if len(all_regression_y_values_global) > 0:
        global_y_array = np.array(all_regression_y_values_global)
        global_y_min = np.min(global_y_array)
        global_y_max = np.max(global_y_array)
        global_y_range = global_y_max - global_y_min
        # Add small padding
        global_y_min = global_y_min - 0.05 * global_y_range
        global_y_max = global_y_max + 0.05 * global_y_range
        
        # Apply to all subplots
        for row in range(3):
            for col in range(4):
                # Check if subplot has any data (has lines or text)
                if len(axes[row, col].lines) > 0 or len(axes[row, col].texts) > 0:
                    axes[row, col].set_ylim(global_y_min, global_y_max)
    
    # Create a single shared legend outside the plot area
    handles = []
    labels = []
    for size_idx, N_total in enumerate(network_sizes_for_legend):
        handles.append(plt.Line2D([0], [0], color=colors[size_idx], linewidth=2.5, alpha=0.8))
        # Count how many runs per param
        param_group = all_data_by_size[size_idx]
        n_runs = len(param_group)
        if n_runs > 1:
            labels.append(f'N={N_total} (n={n_runs})')
        else:
            labels.append(f'N={N_total}')
    
    # Add legend to the figure (outside all subplots)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), 
               ncol=len(handles), fontsize=10, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for legend at top
    
    # Save figure
    correlation_folder = os.path.join(sweep_folder, "Figures", "correlation_analysis")
    os.makedirs(correlation_folder, exist_ok=True)
    save_path = os.path.join(correlation_folder, 'weight_correlation_vs_network_size.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {save_path}")
    print(f"   Analyzed {len(network_sizes_for_legend)} network sizes: {network_sizes_for_legend}")
    if n_runs_per_param > 1:
        print(f"   Averaged over {n_runs_per_param} runs per parameter combination")


def extract_pairs_for_metric(data, config, metric, conn_type, max_pairs=50000):
    """
    Extract pairs of (weight metric, activity correlation) for a specific metric and connection type.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing simulation data
    config : dict
        Configuration dictionary
    metric : str
        One of 'rec_weight', 'input_cosine_sim', 'rec_cosine_sim'
    conn_type : str
        One of 'E->E', 'E->I', 'I->E', 'I->I'
    max_pairs : int
        Maximum number of pairs to extract (for large networks, subsample early)
        
    Returns:
    --------
    list : List of dictionaries with metric and correlation values
    """
    # Extract data
    final_rate_matrix = data['rate_matrices_arr'][-1]  # Shape: (neurons, stimuli)
    weight_rec = data['weight_rec']  # Shape: (neurons, neurons) - recurrent weights
    weight_input = data.get('weight_poisson', None)  # Shape: (neurons, input_dims) - input weights
    
    N_rec_E = config.get('N_rec_E', 100)
    N_rec_I = config.get('N_rec_I', 100)
    N_total = N_rec_E + N_rec_I
    
    # Calculate mean firing rates and filter active neurons
    mean_firing_rates = np.mean(final_rate_matrix, axis=1)
    active_mask = mean_firing_rates > 0
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)
    
    # For very large networks, subsample neurons early
    if n_active > 500:
        import random
        random.seed(42)
        sample_size = min(500, n_active)
        sampled_indices = random.sample(range(n_active), sample_size)
        active_indices = active_indices[sampled_indices]
        n_active = len(active_indices)
        active_rate_matrix = final_rate_matrix[active_indices, :]
    else:
        active_rate_matrix = final_rate_matrix[active_mask, :]
    
    # Calculate activity correlation matrix for active neurons
    neural_correlations = np.corrcoef(active_rate_matrix)
    np.fill_diagonal(neural_correlations, np.nan)
    
    pairs = []
    
    # Pre-filter by connection type to avoid unnecessary computation
    connection_mask = {}
    for i, orig_i in enumerate(active_indices):
        for j, orig_j in enumerate(active_indices):
            if i >= j:  # Only process upper triangle
                continue
                
            # Determine connection type
            if orig_i < N_rec_E and orig_j < N_rec_E:
                pair_conn_type = 'E->E'
            elif orig_i < N_rec_E and orig_j >= N_rec_E:
                pair_conn_type = 'E->I'
            elif orig_i >= N_rec_E and orig_j < N_rec_E:
                pair_conn_type = 'I->E'
            else:
                pair_conn_type = 'I->I'
            
            if pair_conn_type == conn_type:
                correlation = neural_correlations[i, j]
                if not np.isnan(correlation):
                    pairs.append((i, j, orig_i, orig_j))
    
    # Subsample pairs if too many
    if len(pairs) > max_pairs:
        import random
        random.seed(42)
        pairs = random.sample(pairs, max_pairs)
    
    # Now compute metrics for selected pairs
    result_pairs = []
    for idx, (i, j, orig_i, orig_j) in enumerate(pairs):
        if idx % 10000 == 0 and len(pairs) > 10000:
            print(f"      Processing pair {idx+1}/{len(pairs)}...", flush=True)
        
        correlation = neural_correlations[i, j]
        
        # Get recurrent weight from neuron j to neuron i
        rec_weight = weight_rec[orig_i, orig_j]
        
        # Get input weight cosine similarity
        input_cosine_sim = np.nan
        if weight_input is not None:
            input_vec_i = weight_input[orig_i, :]
            input_vec_j = weight_input[orig_j, :]
            norm_i = np.linalg.norm(input_vec_i)
            norm_j = np.linalg.norm(input_vec_j)
            if norm_i > 0 and norm_j > 0:
                input_cosine_sim = np.dot(input_vec_i, input_vec_j) / (norm_i * norm_j)
        
        # Get recurrent weight cosine similarity
        rec_vec_i = weight_rec[orig_i, :]  # Outgoing weights from neuron i
        rec_vec_j = weight_rec[orig_j, :]  # Outgoing weights from neuron j
        norm_rec_i = np.linalg.norm(rec_vec_i)
        norm_rec_j = np.linalg.norm(rec_vec_j)
        rec_cosine_sim = np.nan
        if norm_rec_i > 0 and norm_rec_j > 0:
            rec_cosine_sim = np.dot(rec_vec_i, rec_vec_j) / (norm_rec_i * norm_rec_j)
        
        result_pairs.append({
            'rec_weight': rec_weight,
            'input_cosine_sim': input_cosine_sim,
            'rec_cosine_sim': rec_cosine_sim,
            'correlation': correlation
        })
    
    return result_pairs


def plot_clustered_correlation_matrix(run_folder, data, config):
    """
    Create a 2x2 figure showing firing rate matrices and correlation matrices,
    with hierarchical clustering based on firing rates.
    
    Parameters:
    -----------
    run_folder : str
        Path to the run directory
    data : dict
        Dictionary containing simulation data
    config : dict
        Configuration dictionary
    """
    correlation_folder = os.path.join(run_folder, "Figures", "correlation_analysis")
    os.makedirs(correlation_folder, exist_ok=True)
    
    # Import ensemble analysis for control matrix generation
    from ensemble_analysis import generate_control_rate_matrix
    
    # Extract data
    final_rate_matrix = data['rate_matrices_arr'][-1]  # Shape: (neurons, stimuli)
    
    N_rec_E = config.get('N_rec_E', 100)
    N_rec_I = config.get('N_rec_I', 100)
    
    # Calculate mean firing rates and filter active neurons
    mean_firing_rates = np.mean(final_rate_matrix, axis=1)
    active_mask = mean_firing_rates > 0
    active_indices = np.where(active_mask)[0]
    active_rate_matrix = final_rate_matrix[active_mask, :]
    
    # Identify E and I populations
    active_is_E = active_indices < N_rec_E
    active_is_I = active_indices >= N_rec_E
    
    print(f"   Computing correlation matrix for {len(active_indices)} active neurons...")
    
    # Generate control (shuffled) rate matrix
    control_rate_matrix = generate_control_rate_matrix(active_rate_matrix)
    
    # Z-score within each neuron across stimuli for clustering and correlation
    def zscore_rows(M):
        Z = np.zeros_like(M)
        for i in range(M.shape[0]):
            mu = np.mean(M[i, :])
            sd = np.std(M[i, :])
            Z[i, :] = (M[i, :] - mu) / sd if sd > 0 else 0
        return Z
    
    active_rate_matrix_z = zscore_rows(active_rate_matrix)
    control_rate_matrix_z = zscore_rows(control_rate_matrix)
    
    # Perform hierarchical clustering on firing rates (use z-scored for clustering)
    print(f"   Clustering based on firing rates (data)...")
    
    # Use ensemble_analysis function for clustering
    from ensemble_analysis import hierarchical_cluster_neurons
    
    neuron_order_data, _ = hierarchical_cluster_neurons(active_rate_matrix_z)
    
    print(f"   Clustering based on firing rates (control)...")
    neuron_order_control, _ = hierarchical_cluster_neurons(control_rate_matrix_z)
    
    # Calculate correlation matrices from Z-SCORED data
    print(f"   Computing correlation matrices from z-scored data...")
    neural_correlations = np.corrcoef(active_rate_matrix_z)
    np.fill_diagonal(neural_correlations, np.nan)
    
    control_correlations = np.corrcoef(control_rate_matrix_z)
    np.fill_diagonal(control_correlations, np.nan)
    
    # Compute correlation statistics (excluding diagonal)
    data_corr_flat = neural_correlations[~np.isnan(neural_correlations)]
    control_corr_flat = control_correlations[~np.isnan(control_correlations)]
    
    data_corr_mean = np.mean(data_corr_flat)
    data_corr_std = np.std(data_corr_flat)
    control_corr_mean = np.mean(control_corr_flat)
    control_corr_std = np.std(control_corr_flat)
    
    # Reorder matrices based on firing rate clustering
    rate_matrix_clustered = active_rate_matrix[neuron_order_data, :]
    control_rate_matrix_clustered = control_rate_matrix[neuron_order_control, :]
    rate_matrix_z_clustered = active_rate_matrix_z[neuron_order_data, :]
    control_rate_matrix_z_clustered = control_rate_matrix_z[neuron_order_control, :]
    corr_matrix_clustered = neural_correlations[np.ix_(neuron_order_data, neuron_order_data)]
    control_corr_matrix_clustered = control_correlations[np.ix_(neuron_order_control, neuron_order_control)]
    
    # Create 3x2 figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Hierarchically Clustered Firing Rates and Correlation Matrices', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Non-z-scored firing rate matrices
    # Column 1: Data firing rate matrix (non-z-scored)
    ax = axes[0, 0]
    vmin = np.percentile(rate_matrix_clustered, 1)
    vmax = np.percentile(rate_matrix_clustered, 99)
    im = ax.imshow(rate_matrix_clustered, cmap='hot', aspect='auto', 
                  vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xlabel('Stimulus Index')
    ax.set_ylabel('Neuron Index (Clustered)')
    ax.set_title('Firing Rate Matrix (Data) - Raw, Hierarchically Clustered')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Firing Rate')
    
    # Add E/I boundary
    n_E = np.sum(active_is_E[neuron_order_data])
    if n_E > 0 and n_E < len(neuron_order_data):
        ax.axhline(y=n_E - 0.5, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Column 2: Control firing rate matrix (non-z-scored)
    ax = axes[0, 1]
    vmin = np.percentile(control_rate_matrix_clustered, 1)
    vmax = np.percentile(control_rate_matrix_clustered, 99)
    im = ax.imshow(control_rate_matrix_clustered, cmap='hot', aspect='auto',
                  vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xlabel('Stimulus Index')
    ax.set_ylabel('Neuron Index (Clustered)')
    ax.set_title('Firing Rate Matrix (Control) - Raw, Hierarchically Clustered')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Firing Rate')
    
    # Add E/I boundary for control
    n_E_ctrl = np.sum(active_is_E[neuron_order_control])
    if n_E_ctrl > 0 and n_E_ctrl < len(neuron_order_control):
        ax.axhline(y=n_E_ctrl - 0.5, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Row 2: Z-scored firing rate matrices (blue-white-red colormap)
    # Column 1: Data z-scored firing rate matrix
    ax = axes[1, 0]
    vmin = np.percentile(rate_matrix_z_clustered, 1)
    vmax = np.percentile(rate_matrix_z_clustered, 99)
    # Use symmetric colormap for z-scored data (blue-white-red)
    max_abs = max(abs(vmin), abs(vmax))
    im = ax.imshow(rate_matrix_z_clustered, cmap='RdBu_r', aspect='auto',
                  vmin=-max_abs, vmax=max_abs, interpolation='nearest')
    ax.set_xlabel('Stimulus Index')
    ax.set_ylabel('Neuron Index (Clustered)')
    ax.set_title('Firing Rate Matrix (Data) - Z-scored, Hierarchically Clustered')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Z-scored Firing Rate')
    
    # Add E/I boundary
    if n_E > 0 and n_E < len(neuron_order_data):
        ax.axhline(y=n_E - 0.5, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Column 2: Control z-scored firing rate matrix
    ax = axes[1, 1]
    vmin = np.percentile(control_rate_matrix_z_clustered, 1)
    vmax = np.percentile(control_rate_matrix_z_clustered, 99)
    max_abs = max(abs(vmin), abs(vmax))
    im = ax.imshow(control_rate_matrix_z_clustered, cmap='RdBu_r', aspect='auto',
                  vmin=-max_abs, vmax=max_abs, interpolation='nearest')
    ax.set_xlabel('Stimulus Index')
    ax.set_ylabel('Neuron Index (Clustered)')
    ax.set_title('Firing Rate Matrix (Control) - Z-scored, Hierarchically Clustered')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Z-scored Firing Rate')
    
    # Add E/I boundary for control
    if n_E_ctrl > 0 and n_E_ctrl < len(neuron_order_control):
        ax.axhline(y=n_E_ctrl - 0.5, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Row 3: Correlation matrices (ordered by firing rate clustering, computed from z-scored data)
    # Column 1: Data correlation matrix
    ax = axes[2, 0]
    im = ax.imshow(corr_matrix_clustered, cmap='RdBu_r', aspect='auto',
                  vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xlabel('Neuron Index (Ordered by Firing Rate Clustering)')
    ax.set_ylabel('Neuron Index (Ordered by Firing Rate Clustering)')
    ax.set_title('Correlation Matrix (Data)\n(Ordered by Firing Rate Clustering, from Z-scored Data)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Correlation')
    
    # Add correlation statistics
    stats_text = f'Mean: {data_corr_mean:.3f}\nStd: {data_corr_std:.3f}'
    ax.text(0.02, 0.98, stats_text, 
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           verticalalignment='top', fontsize=10, fontweight='bold')
    
    # Add E/I boundary
    if n_E > 0 and n_E < len(neuron_order_data):
        ax.axhline(y=n_E - 0.5, color='white', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(x=n_E - 0.5, color='white', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Column 2: Control correlation matrix
    ax = axes[2, 1]
    im = ax.imshow(control_corr_matrix_clustered, cmap='RdBu_r', aspect='auto',
                  vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xlabel('Neuron Index (Ordered by Control Firing Rate Clustering)')
    ax.set_ylabel('Neuron Index (Ordered by Control Firing Rate Clustering)')
    ax.set_title('Correlation Matrix (Control)\n(Ordered by Control Firing Rate Clustering, from Z-scored Data)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Correlation')
    
    # Add correlation statistics
    stats_text = f'Mean: {control_corr_mean:.3f}\nStd: {control_corr_std:.3f}'
    ax.text(0.02, 0.98, stats_text, 
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           verticalalignment='top', fontsize=10, fontweight='bold')
    
    # Add E/I boundary for control
    if n_E_ctrl > 0 and n_E_ctrl < len(neuron_order_control):
        ax.axhline(y=n_E_ctrl - 0.5, color='white', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(x=n_E_ctrl - 0.5, color='white', linestyle='--', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(correlation_folder, 'clustered_correlation_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {save_path}")
    
    print(f"\n   Correlation Statistics:")
    print(f"   Data: mean = {data_corr_mean:.3f}, std = {data_corr_std:.3f}")
    print(f"   Control: mean = {control_corr_mean:.3f}, std = {control_corr_std:.3f}")
    
    return {
        'n_neurons': len(active_indices),
        'n_E': int(np.sum(active_is_E)),
        'n_I': int(np.sum(active_is_I)),
        'data_correlation': {'mean': data_corr_mean, 'std': data_corr_std},
        'control_correlation': {'mean': control_corr_mean, 'std': control_corr_std}
    }


def generate_correlation_plots(run_folder, data, config):
    """
    Generate comprehensive correlation analysis visualizations.
    
    Uses the comprehensive diagnostic plot which shows all correlation relationships
    in a single multi-panel figure, avoiding redundant separate plots.
    
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
    dict : Comprehensive correlation statistics
    """
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Create subfolder
    correlation_folder = os.path.join(run_folder, "Figures", "correlation_analysis")
    os.makedirs(correlation_folder, exist_ok=True)
    
    # Extract data from final epoch
    final_rate_matrix = data['rate_matrices_arr'][-1]  # Shape: (neurons, stimuli)
    final_selectivity = data['selectivity_vec_arr'][-1, :]  # Shape: (neurons,)
    
    N_rec_E = config.get('N_rec_E', 100)
    N_rec_I = config.get('N_rec_I', 100)
    
    # Calculate mean firing rates
    mean_firing_rates = np.mean(final_rate_matrix, axis=1)
    
    # Only analyze active neurons
    active_mask = mean_firing_rates > 0
    active_rate_matrix = final_rate_matrix[active_mask, :]
    active_selectivity = final_selectivity[active_mask]
    active_firing_rates = mean_firing_rates[active_mask]
    
    # Identify E and I populations in active neurons
    active_indices = np.where(active_mask)[0]
    active_is_E = active_indices < N_rec_E
    active_is_I = active_indices >= N_rec_E
    
    E_indices = np.where(active_is_E)[0]
    I_indices = np.where(active_is_I)[0]
    
    print(f"   Active neurons: {np.sum(active_mask)} out of {len(active_mask)}")
    print(f"   Active E neurons: {np.sum(active_is_E)}")
    print(f"   Active I neurons: {np.sum(active_is_I)}")
    
    all_stats = {}
    
    try:
        # ===== COMPREHENSIVE DIAGNOSTIC PLOT =====
        print("   Creating comprehensive diagnostic correlation plot...")
        
        # Call the comprehensive diagnostic function (generates diagnostic_weak_correlation.png)
        diagnose_correlation.diagnose_weak_correlation(run_folder)
        print(f"      Saved: diagnostic_weak_correlation.png")
        
        # ===== SOLOIST/CHORISTER ANALYSIS =====
        print("   Creating soloist/chorister analysis...")
        
        # Call the soloist/chorister analysis function
        soloist_stats = soloist_chorister_analysis.generate_soloist_chorister_analysis(run_folder, data, config)
        all_stats['soloist_chorister'] = soloist_stats
        
        # ===== ENSEMBLE ANALYSIS =====
        print("   Creating ensemble analysis...")
        ensemble_stats = ensemble_analysis.generate_ensemble_analysis(run_folder, data, config)
        all_stats['ensemble'] = ensemble_stats
        
        # ===== WEIGHT VS CORRELATION ANALYSIS =====
        print("   Creating weight vs correlation plot...")
        weight_corr_stats = plot_weight_vs_correlation(run_folder, data, config)
        all_stats['weight_vs_correlation'] = weight_corr_stats
        
        # ===== CLUSTERED CORRELATION MATRIX =====
        print("   Creating clustered correlation matrix...")
        clustered_corr_stats = plot_clustered_correlation_matrix(run_folder, data, config)
        all_stats['clustered_correlation_matrix'] = clustered_corr_stats
        
        # Calculate stats for return dict
        corr_coeff, p_value = stats.pearsonr(active_firing_rates, active_selectivity)
        all_stats['firing_rate_selectivity'] = {
            'correlation': corr_coeff,
            'p_value': p_value,
            'mean_firing_rate': np.mean(active_firing_rates),
            'mean_selectivity': np.mean(active_selectivity)
        }
        
        # Calculate neural correlations for stats
        neural_correlations = np.corrcoef(active_rate_matrix)
        np.fill_diagonal(neural_correlations, np.nan)
        
        # Calculate mean correlations WITH E and I neurons
        mean_corr_with_E = np.nanmean(neural_correlations[:, E_indices], axis=1)
        mean_corr_with_I = np.nanmean(neural_correlations[:, I_indices], axis=1)
        
        # Extract E-E, E-I, I-I correlations for stats
        EE_corr = neural_correlations[np.ix_(E_indices, E_indices)]
        np.fill_diagonal(EE_corr, np.nan)
        EE_flat = EE_corr[~np.isnan(EE_corr)]
        
        EI_corr = neural_correlations[np.ix_(E_indices, I_indices)]
        EI_flat = EI_corr.flatten()
        
        II_corr = neural_correlations[np.ix_(I_indices, I_indices)]
        np.fill_diagonal(II_corr, np.nan)
        II_flat = II_corr[~np.isnan(II_corr)]
        
        # Calculate weighted correlations
        weight_matrix = data['weight_rec']
        n_active = neural_correlations.shape[0]
        weighted_corr_with_E = np.zeros(n_active)
        weighted_corr_with_I = np.zeros(n_active)
        
        for i in range(n_active):
            orig_idx_i = active_indices[i]
            orig_E_indices = active_indices[E_indices]
            orig_I_indices = active_indices[I_indices]
            
            # Weights from E neurons to neuron i
            weights_from_E = np.abs(weight_matrix[orig_idx_i, orig_E_indices])
            corr_with_E = neural_correlations[i, E_indices]
            valid_E = ~np.isnan(corr_with_E)
            if np.sum(weights_from_E[valid_E]) > 0:
                weighted_corr_with_E[i] = np.sum(corr_with_E[valid_E] * weights_from_E[valid_E]) / np.sum(weights_from_E[valid_E])
            else:
                weighted_corr_with_E[i] = np.nan
            
            # Weights from I neurons to neuron i
            weights_from_I = np.abs(weight_matrix[orig_idx_i, orig_I_indices])
            corr_with_I = neural_correlations[i, I_indices]
            valid_I = ~np.isnan(corr_with_I)
            if np.sum(weights_from_I[valid_I]) > 0:
                weighted_corr_with_I[i] = np.sum(corr_with_I[valid_I] * weights_from_I[valid_I]) / np.sum(weights_from_I[valid_I])
            else:
                weighted_corr_with_I[i] = np.nan
        
        # Calculate correlation coefficients for stats
        sel_corrE_coeff, sel_corrE_p = stats.pearsonr(mean_corr_with_E, active_selectivity)
        sel_corrI_coeff, sel_corrI_p = stats.pearsonr(mean_corr_with_I, active_selectivity)
        
        valid_weighted_E = ~np.isnan(weighted_corr_with_E)
        if np.sum(valid_weighted_E) > 2:
            sel_wcorrE_coeff, sel_wcorrE_p = stats.pearsonr(weighted_corr_with_E[valid_weighted_E],
                                                             active_selectivity[valid_weighted_E])
        else:
            sel_wcorrE_coeff, sel_wcorrE_p = np.nan, np.nan
        
        valid_weighted_I = ~np.isnan(weighted_corr_with_I)
        if np.sum(valid_weighted_I) > 2:
            sel_wcorrI_coeff, sel_wcorrI_p = stats.pearsonr(weighted_corr_with_I[valid_weighted_I],
                                                             active_selectivity[valid_weighted_I])
        else:
            sel_wcorrI_coeff, sel_wcorrI_p = np.nan, np.nan
        
        all_stats['neural_correlations'] = {
            'mean_correlation': np.mean(neural_correlations[~np.isnan(neural_correlations)]),
            'std_correlation': np.std(neural_correlations[~np.isnan(neural_correlations)]),
            'selectivity_vs_corr_with_E': (sel_corrE_coeff, sel_corrE_p),
            'selectivity_vs_corr_with_I': (sel_corrI_coeff, sel_corrI_p),
            'selectivity_vs_weighted_corr_with_E': (sel_wcorrE_coeff, sel_wcorrE_p),
            'selectivity_vs_weighted_corr_with_I': (sel_wcorrI_coeff, sel_wcorrI_p),
            'EE_mean': np.mean(EE_flat) if len(EE_flat) > 0 else np.nan,
            'EI_mean': np.mean(EI_flat) if len(EI_flat) > 0 else np.nan,
            'II_mean': np.mean(II_flat) if len(II_flat) > 0 else np.nan,
        }
        
    except Exception as e:
        print(f"   Error in correlation analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    if all_stats:
        print("\n   Correlation Analysis Summary:")
        if 'firing_rate_selectivity' in all_stats:
            print(f"   Firing Rate vs Selectivity: r = {all_stats['firing_rate_selectivity']['correlation']:.3f}, "
                  f"p = {all_stats['firing_rate_selectivity']['p_value']:.3e}")
        if 'neural_correlations' in all_stats:
            print(f"   Mean Neural Correlation: {all_stats['neural_correlations']['mean_correlation']:.3f}")
            print(f"   Selectivity vs Correlation WITH E: r = {all_stats['neural_correlations']['selectivity_vs_corr_with_E'][0]:.3f}")
            print(f"   Selectivity vs Correlation WITH I: r = {all_stats['neural_correlations']['selectivity_vs_corr_with_I'][0]:.3f}")
            
            wcorr_E = all_stats['neural_correlations']['selectivity_vs_weighted_corr_with_E'][0]
            wcorr_I = all_stats['neural_correlations']['selectivity_vs_weighted_corr_with_I'][0]
            if not np.isnan(wcorr_E):
                print(f"   Selectivity vs WEIGHTED Correlation WITH E: r = {wcorr_E:.3f}")
            if not np.isnan(wcorr_I):
                print(f"   Selectivity vs WEIGHTED Correlation WITH I: r = {wcorr_I:.3f}")
    
    return all_stats


if __name__ == "__main__":
    """Command-line interface for standalone correlation analysis"""
    import pickle
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python correlation_analysis.py <run_folder>")
        sys.exit(1)
    
    run_folder = sys.argv[1]
    data_file = os.path.join(run_folder, "Data", "simulation_data.pkl")
    
    if not os.path.exists(data_file):
        print(f"Error: No data file found at {data_file}")
        sys.exit(1)
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    config = data.get('config', {})
    stats = generate_correlation_plots(run_folder, data, config)
    print("\nCorrelation analysis complete!")
