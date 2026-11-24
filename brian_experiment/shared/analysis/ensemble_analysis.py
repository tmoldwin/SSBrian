"""
Ensemble Analysis Module

Identifies and analyzes neural ensembles - groups of neurons with correlated
firing patterns. Uses hierarchical clustering to organize neurons and defines
ensembles based on correlation thresholds.

Key analyses:
- Hierarchically clustered firing rate matrices (first and last epoch)
- Ensemble identification at different correlation thresholds
- Ensemble size distributions
- Evolution of ensemble structure across epochs

Creates plots in the ensemble_analysis/ subfolder.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from numpy.linalg import LinAlgError
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def compute_neuron_correlation_matrix(rate_matrix):
    """
    Compute pairwise correlations between neurons based on their responses to stimuli.
    
    Parameters:
    -----------
    rate_matrix : ndarray
        Shape (neurons, stimuli) - firing rates
    
    Returns:
    --------
    corr_matrix : ndarray
        Shape (neurons, neurons) - correlation matrix
    """
    # Correlate across stimuli (columns)
    corr_matrix = np.corrcoef(rate_matrix)
    
    # Replace NaNs (from neurons with zero variance) with 0
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    return corr_matrix


def generate_control_rate_matrix(rate_matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Generate a control dataset that preserves per-neuron firing rate
    distributions and positivity while breaking cross-neuron coordination.

    Approach (bootstrap null):
    - For each neuron, resample (with replacement) its stimulus responses
      across stimuli. This preserves the marginal distribution (mean/variance,
      skew, positivity) and removes across-neuron structure.

    Returns a matrix of shape (neurons, stimuli). 
    This is a strong null preserving marginals and destroying coordination.
    """
    n_neurons, n_stimuli = rate_matrix.shape
    control = np.zeros_like(rate_matrix)
    rng = np.random.default_rng(seed=42)
    for i in range(n_neurons):
        control[i, :] = rng.choice(rate_matrix[i, :], size=n_stimuli, replace=True)
    return control


def hierarchical_cluster_neurons(rate_matrix):
    """
    Perform hierarchical clustering on neurons based on firing rate similarity.
    
    Parameters:
    -----------
    rate_matrix : ndarray
        Shape (neurons, stimuli)
    
    Returns:
    --------
    neuron_order : ndarray
        Indices reordering neurons by similarity
    linkage_matrix : ndarray
        Linkage matrix from hierarchical clustering
    """
    n_neurons = rate_matrix.shape[0]
    
    # Compute pairwise distances between neurons
    # Use correlation distance: 1 - correlation
    print(f"      Computing pairwise distances for {n_neurons} neurons...")
    distances = pdist(rate_matrix, metric='correlation')
    
    # Handle NaN and infinite values in distance matrix
    # These can occur when neurons have zero variance or identical responses
    if np.any(~np.isfinite(distances)):
        print(f"      Warning: Found {np.sum(~np.isfinite(distances))} non-finite distances, replacing with 1.0 (maximum distance)...")
        distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=1.0)
        # Ensure all values are finite
        distances = np.clip(distances, 0, 1)
    
    # For large datasets (>500 neurons), use 'average' linkage instead of 'ward'
    # Ward linkage is computationally expensive and can fail on large datasets
    # Average linkage is faster and still provides good clustering
    if n_neurons > 500:
        print(f"      Using 'average' linkage (faster for {n_neurons} neurons)...")
        linkage_method = 'average'
    else:
        linkage_method = 'ward'
    
    # Perform hierarchical clustering
    try:
        print(f"      Performing hierarchical clustering with {linkage_method} linkage...")
        linkage_matrix = linkage(distances, method=linkage_method)
    except (LinAlgError, MemoryError, ValueError) as e:
        # Fallback to average linkage if ward fails
        if linkage_method == 'ward':
            print(f"      Ward linkage failed ({e}), trying average linkage...")
            linkage_matrix = linkage(distances, method='average')
        else:
            # If average also fails, try complete linkage as last resort
            print(f"      {linkage_method} linkage failed ({e}), trying complete linkage...")
            linkage_matrix = linkage(distances, method='complete')
    
    # Get dendrogram leaf order (optimal ordering)
    print(f"      Extracting dendrogram ordering...")
    dendro = dendrogram(linkage_matrix, no_plot=True)
    neuron_order = dendro['leaves']
    
    return np.array(neuron_order), linkage_matrix


def identify_ensembles(corr_matrix, correlation_threshold=0.5):
    """
    Identify ensembles as groups of neurons with high mutual correlations.
    
    ENSEMBLE DEFINITION:
    An ensemble is a group of neurons that respond similarly to stimuli.
    Two neurons are in the same ensemble if their firing rate correlation 
    (across all stimuli) exceeds the correlation_threshold.
    
    Method:
    1. Compute pairwise correlations between all neurons (based on firing rates)
    2. Use hierarchical clustering (average linkage) with correlation distance
    3. Cut the dendrogram at height = (1 - correlation_threshold)
    4. Each resulting cluster is an ensemble
    5. Neurons not grouped with others form ensembles of size 1 (soloists)
    
    Parameters:
    -----------
    corr_matrix : ndarray
        Neuron-neuron correlation matrix (shape: N_neurons x N_neurons)
    correlation_threshold : float
        Minimum correlation to be in same ensemble (0 to 1)
        Higher threshold = more stringent, smaller ensembles
    
    Returns:
    --------
    ensembles : list of lists
        Each element is a list of neuron indices forming an ensemble
        Includes singletons (ensembles of size 1)
    """
    n_neurons = corr_matrix.shape[0]
    
    # Convert correlation to distance: d = 1 - correlation
    # High correlation -> low distance (neurons are similar)
    distance_matrix = 1 - corr_matrix
    
    # Use hierarchical clustering to group similar neurons
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # Cut tree at distance = 1 - correlation_threshold
    # Neurons within this distance are in the same ensemble
    distance_threshold = 1 - correlation_threshold
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    
    # Group neurons by cluster (including singletons)
    ensembles = []
    for cluster_id in np.unique(cluster_labels):
        ensemble_neurons = np.where(cluster_labels == cluster_id)[0]
        ensembles.append(ensemble_neurons.tolist())
    
    return ensembles


def analyze_ensemble_structure(corr_matrix, correlation_thresholds=None):
    """
    Analyze ensemble size distribution for different correlation thresholds.
    
    Parameters:
    -----------
    corr_matrix : ndarray
        Neuron-neuron correlation matrix
    correlation_thresholds : list
        Correlation thresholds to test
    
    Returns:
    --------
    results : dict
        Contains ensemble size distributions for each threshold
    """
    if correlation_thresholds is None:
        correlation_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    n_neurons = corr_matrix.shape[0]
    
    # For each threshold, get ensemble size distribution
    # Key: ensemble size, Value: count of ensembles of that size
    size_distributions = {}
    
    for corr_thresh in correlation_thresholds:
        ensembles = identify_ensembles(corr_matrix, corr_thresh)
        
        # Count ensemble sizes
        size_counts = {}
        for ens in ensembles:
            size = len(ens)
            size_counts[size] = size_counts.get(size, 0) + 1
        
        size_distributions[corr_thresh] = size_counts
    
    results = {
        'correlation_thresholds': correlation_thresholds,
        'size_distributions': size_distributions,
        'n_neurons': n_neurons
    }
    
    return results


def generate_dimensionality_reduction_analysis(rate_matrix, is_E, ensemble_folder):
    """
    Apply dimensionality reduction techniques to visualize neuron clustering.
    Shows both raw and z-scored data for comparison.
    
    Parameters:
    -----------
    rate_matrix : ndarray
        Firing rate matrix (neurons x stimuli)
    is_E : ndarray
        Boolean array indicating E (True) or I (False) neurons
    ensemble_folder : str
        Path to save figures
    """
    print("   Performing dimensionality reduction analysis...")
    
    # Z-score within each neuron (across stimuli)
    rate_matrix_z = np.zeros_like(rate_matrix)
    for i in range(rate_matrix.shape[0]):
        mean = np.mean(rate_matrix[i, :])
        std = np.std(rate_matrix[i, :])
        if std > 0:
            rate_matrix_z[i, :] = (rate_matrix[i, :] - mean) / std
        else:
            rate_matrix_z[i, :] = 0
    
    # Prepare both versions
    data_versions = [
        ('Raw', rate_matrix),
        ('Z-scored', rate_matrix_z)
    ]
    
    # Prepare control (matched statistics)
    control_rate_matrix = generate_control_rate_matrix(rate_matrix)
    # Z-score control within neuron
    control_rate_matrix_z = np.zeros_like(control_rate_matrix)
    for i in range(control_rate_matrix.shape[0]):
        mean = np.mean(control_rate_matrix[i, :])
        std = np.std(control_rate_matrix[i, :])
        control_rate_matrix_z[i, :] = (control_rate_matrix[i, :] - mean) / std if std > 0 else 0

    # Create figure: 3 rows (Raw, Z-scored, Control Z-scored) x 3 cols (PCA, t-SNE, UMAP)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    for row_idx, (data_name, data) in enumerate(data_versions):
        print(f"      Processing {data_name} data...")
        
        # PCA
        print(f"         Running PCA...")
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(data)
        var1, var2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]
        
        # t-SNE
        print(f"         Running t-SNE...")
        # For large datasets, use lower perplexity to speed up computation
        max_perplexity = 50 if len(data) > 500 else 30
        perplexity = min(max_perplexity, len(data) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                   max_iter=1000, verbose=0)  # Reduce iterations for speed
        tsne_coords = tsne.fit_transform(data)
        
        # UMAP
        print(f"         Running UMAP...")
        if HAS_UMAP:
            # For large datasets, use more neighbors but fewer iterations
            n_neighbors_val = min(30 if len(data) > 500 else 15, len(data) - 1)
            reducer = umap.UMAP(n_components=2, random_state=42, 
                               n_neighbors=n_neighbors_val,
                               n_epochs=500 if len(data) > 500 else None)  # Reduce epochs for speed
            umap_coords = reducer.fit_transform(data)
        else:
            umap_coords = None
        
        # Plot each method
        methods = [
            (pca_coords, f'PCA\n(Var: {var1:.1%}, {var2:.1%})'),
            (tsne_coords, 't-SNE'),
            (umap_coords, 'UMAP') if umap_coords is not None else (None, 'UMAP\n(not installed)')
        ]
        
        for col_idx, (coords, title) in enumerate(methods):
            ax = axes[row_idx, col_idx]
            
            if coords is not None:
                # Plot E neurons (red) and I neurons (blue)
                e_mask = is_E
                i_mask = ~is_E
                
                ax.scatter(coords[e_mask, 0], coords[e_mask, 1], 
                          c='red', s=40, alpha=0.6, edgecolors='black', linewidth=0.5,
                          label=f'E neurons')
                ax.scatter(coords[i_mask, 0], coords[i_mask, 1],
                          c='blue', s=40, alpha=0.6, edgecolors='black', linewidth=0.5,
                          label=f'I neurons')
                
                ax.set_xlabel('Component 1', fontsize=11)
                ax.set_ylabel('Component 2', fontsize=11)
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'UMAP not installed\npip install umap-learn', 
                       ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Title
            if row_idx == 0:
                ax.set_title(title, fontsize=13, fontweight='bold')
            
            # Row label on left
            if col_idx == 0:
                ax.set_ylabel(f'{data_name}\n\nComponent 2', fontsize=11, fontweight='bold')

    # Third row: Control Z-scored only for clear pattern-only comparison
    ctrl_row_idx = 2
    data = control_rate_matrix_z
    # PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(data)
    var1, var2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]
    # t-SNE
    max_perplexity = 50 if len(data) > 500 else 30
    perplexity = min(max_perplexity, len(data) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
               max_iter=1000, verbose=0)
    tsne_coords = tsne.fit_transform(data)
    # UMAP
    if HAS_UMAP:
        n_neighbors_val = min(30 if len(data) > 500 else 15, len(data) - 1)
        reducer = umap.UMAP(n_components=2, random_state=42, 
                           n_neighbors=n_neighbors_val,
                           n_epochs=500 if len(data) > 500 else None)
        umap_coords = reducer.fit_transform(data)
    else:
        umap_coords = None
    ctrl_methods = [
        (pca_coords, f'PCA\n(Control Z, Var: {var1:.1%}, {var2:.1%})'),
        (tsne_coords, 't-SNE\n(Control Z)'),
        (umap_coords, 'UMAP\n(Control Z)') if umap_coords is not None else (None, 'UMAP\n(not installed)')
    ]
    for col_idx, (coords, title) in enumerate(ctrl_methods):
        ax = axes[ctrl_row_idx, col_idx]
        if coords is not None:
            e_mask = is_E
            i_mask = ~is_E
            ax.scatter(coords[e_mask, 0], coords[e_mask, 1], 
                      c='red', s=40, alpha=0.6, edgecolors='black', linewidth=0.5,
                      label=f'E neurons')
            ax.scatter(coords[i_mask, 0], coords[i_mask, 1],
                      c='blue', s=40, alpha=0.6, edgecolors='black', linewidth=0.5,
                      label=f'I neurons')
            ax.set_xlabel('Component 1', fontsize=11)
            ax.set_ylabel('Component 2', fontsize=11)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'UMAP not installed\npip install umap-learn', 
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        if ctrl_row_idx == 0:
            ax.set_title(title, fontsize=13, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel('Control (Z-scored)\n\nComponent 2', fontsize=11, fontweight='bold')
    
    fig.suptitle('Dimensionality Reduction: Neural Firing Pattern Similarity',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(ensemble_folder, 'dimensionality_reduction.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def generate_ensemble_analysis(run_folder, data, config):
    """
    Generate comprehensive ensemble analysis figure.
    
    Creates a multi-panel figure showing:
    - Hierarchically clustered firing rate matrices (first and last epoch)
    - Correlation matrix with ensemble boundaries
    - Ensemble statistics across parameter space
    
    Parameters:
    -----------
    run_folder : str
        Path to the run directory
    data : dict
        Simulation data
    config : dict
        Configuration parameters
    
    Returns:
    --------
    stats : dict
        Ensemble analysis statistics
    """
    print("\n" + "="*60)
    print("ENSEMBLE ANALYSIS")
    print("="*60)
    
    # Create subfolder
    ensemble_folder = os.path.join(run_folder, "Figures", "ensemble_analysis")
    os.makedirs(ensemble_folder, exist_ok=True)
    
    # Extract data
    first_rate_matrix = data['rate_matrices_arr'][0]
    last_rate_matrix = data['rate_matrices_arr'][-1]
    
    N_rec_E = config.get('N_rec_E', 100)
    N_rec_I = config.get('N_rec_I', 100)
    n_neurons = N_rec_E + N_rec_I
    
    # Filter active neurons (based on last epoch)
    mean_rates_last = np.mean(last_rate_matrix, axis=1)
    active_mask = mean_rates_last > 0
    
    print(f"   Active neurons: {np.sum(active_mask)}/{n_neurons}")
    
    # Work with active neurons only
    first_matrix_active = first_rate_matrix[active_mask, :]
    last_matrix_active = last_rate_matrix[active_mask, :]
    
    # Determine E/I identity for active neurons
    active_indices = np.where(active_mask)[0]
    is_E = active_indices < N_rec_E
    
    # Prepare z-scored versions (within neuron) for clustering
    def zscore_rows(M: np.ndarray) -> np.ndarray:
        Z = np.zeros_like(M)
        for i in range(M.shape[0]):
            mu = np.mean(M[i, :])
            sd = np.std(M[i, :])
            Z[i, :] = (M[i, :] - mu) / sd if sd > 0 else 0
        return Z

    print("   Performing hierarchical clustering...")
    # Generate control matrices early (will be used for separate clustering)
    control_first_full = generate_control_rate_matrix(first_matrix_active)
    control_last_full = generate_control_rate_matrix(last_matrix_active)

    # Z-score within neuron for both data and control
    last_z = zscore_rows(last_matrix_active)
    first_z = zscore_rows(first_matrix_active)
    ctrl_last_z = zscore_rows(control_last_full)
    ctrl_first_z = zscore_rows(control_first_full)

    # Separate clustering orders for data and control (based on last epoch)
    neuron_order, linkage_matrix = hierarchical_cluster_neurons(last_z)
    neuron_order_ctrl, _ = hierarchical_cluster_neurons(ctrl_last_z)
    
    # Reorder matrices
    first_matrix_clustered = first_matrix_active[neuron_order, :]
    last_matrix_clustered = last_matrix_active[neuron_order, :]
    is_E_clustered = is_E[neuron_order]
    
    # Compute correlation matrix (last epoch)
    print("   Computing correlation matrix...")
    corr_matrix = compute_neuron_correlation_matrix(last_matrix_active)
    corr_matrix_clustered = corr_matrix[np.ix_(neuron_order, neuron_order)]
    
    # Analyze ensemble structure
    print("   Analyzing ensemble structure...")
    ensemble_stats = analyze_ensemble_structure(corr_matrix)
    
    # Dimensionality reduction analysis
    generate_dimensionality_reduction_analysis(last_matrix_active, is_E, ensemble_folder)
    
    # Create figures
    print("   Creating ensemble analysis figures...")
    
    # === Clustered heatmaps in ONE figure (2x2):
    # Compute ordered matrices using neuron_order (from last epoch clustering)
    # Use separate orders for data vs control
    first_data_ord = first_matrix_active[neuron_order, :]
    last_data_ord = last_matrix_active[neuron_order, :]
    first_ctrl_ord = control_first_full[neuron_order_ctrl, :]
    last_ctrl_ord = control_last_full[neuron_order_ctrl, :]

    # Color scaling: robust percentiles per panel to avoid a few outliers
    def robust_bounds(M):
        vmin = np.percentile(M, 1)
        vmax = np.percentile(M, 99)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        return vmin, vmax

    fig_hm, axes_hm = plt.subplots(2, 2, figsize=(16, 12))

    vmin, vmax = robust_bounds(first_data_ord)
    im = axes_hm[0, 0].imshow(first_data_ord, aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)
    axes_hm[0, 0].set_title('First Epoch (Data) - Clustered', fontsize=13, fontweight='bold')
    axes_hm[0, 0].set_xlabel('Stimulus Index'); axes_hm[0, 0].set_ylabel('Neuron (clustered)')
    fig_hm.colorbar(im, ax=axes_hm[0, 0], fraction=0.046, pad=0.04)

    vmin, vmax = robust_bounds(first_ctrl_ord)
    im = axes_hm[0, 1].imshow(first_ctrl_ord, aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)
    axes_hm[0, 1].set_title('First Epoch (Control) - Clustered', fontsize=13, fontweight='bold')
    axes_hm[0, 1].set_xlabel('Stimulus Index'); axes_hm[0, 1].set_ylabel('Neuron (clustered)')
    fig_hm.colorbar(im, ax=axes_hm[0, 1], fraction=0.046, pad=0.04)

    vmin, vmax = robust_bounds(last_data_ord)
    im = axes_hm[1, 0].imshow(last_data_ord, aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)
    axes_hm[1, 0].set_title('Last Epoch (Data) - Clustered', fontsize=13, fontweight='bold')
    axes_hm[1, 0].set_xlabel('Stimulus Index'); axes_hm[1, 0].set_ylabel('Neuron (clustered)')
    fig_hm.colorbar(im, ax=axes_hm[1, 0], fraction=0.046, pad=0.04)

    vmin, vmax = robust_bounds(last_ctrl_ord)
    im = axes_hm[1, 1].imshow(last_ctrl_ord, aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)
    axes_hm[1, 1].set_title('Last Epoch (Control) - Clustered', fontsize=13, fontweight='bold')
    axes_hm[1, 1].set_xlabel('Stimulus Index'); axes_hm[1, 1].set_ylabel('Neuron (clustered)')
    fig_hm.colorbar(im, ax=axes_hm[1, 1], fraction=0.046, pad=0.04)

    fig_hm.suptitle('Clustered Firing Rate Matrices (One Figure)', fontsize=16, fontweight='bold')
    combined_path = os.path.join(ensemble_folder, 'clustered_firing_rates_combined.png')
    fig_hm.tight_layout()
    fig_hm.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close(fig_hm)
    print(f"   Saved: {combined_path}")

    # === ENSEMBLE SIZE DISTRIBUTION ===
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot ensemble size distribution for each threshold
    colors = plt.cm.viridis(np.linspace(0, 1, len(ensemble_stats['correlation_thresholds'])))
    
    all_sizes = set()
    for i, corr_thresh in enumerate(ensemble_stats['correlation_thresholds']):
        size_counts = ensemble_stats['size_distributions'][corr_thresh]
        
        # Get all sizes and counts for data
        sizes = sorted(size_counts.keys())
        counts = [size_counts[s] for s in sizes]
        all_sizes.update(sizes)

        # Control overlay (from control_last_full)
        ctrl_corr = compute_neuron_correlation_matrix(control_last_full)
        ctrl_stats = analyze_ensemble_structure(ctrl_corr, [corr_thresh])
        ctrl_counts = ctrl_stats['size_distributions'][corr_thresh]
        ctrl_sizes = sorted(ctrl_counts.keys())
        ctrl_vals = [ctrl_counts[s] for s in ctrl_sizes]
        all_sizes.update(ctrl_sizes)
        
        # Get union of all sizes for this threshold
        all_sizes_thresh = sorted(set(sizes) | set(ctrl_sizes))
        
        # Offset for shifting data (left) and control (right)
        offset = 0.15
        
        # For each ensemble size, collect points and draw connecting lines
        data_x = []
        data_y = []
        ctrl_x = []
        ctrl_y = []
        
        for size in all_sizes_thresh:
            data_count = size_counts.get(size, 0)
            ctrl_count = ctrl_counts.get(size, 0)
            
            # Shift data points left, control points right
            if data_count > 0:
                data_x.append(size - offset)
                data_y.append(data_count)
            
            if ctrl_count > 0:
                ctrl_x.append(size + offset)
                ctrl_y.append(ctrl_count)
            
            # Draw line connecting data to control at the same ensemble size
            if data_count > 0 and ctrl_count > 0:
                ax.plot([size - offset, size + offset], [data_count, ctrl_count], '-', 
                       linewidth=1.5, color=colors[i], alpha=0.6, zorder=1)
        
        # Plot scatter points for data (circles) - shifted left
        if len(data_x) > 0:
            ax.scatter(data_x, data_y, s=100, marker='o', 
                      color=colors[i], edgecolors='black', linewidth=1.5,
                      label=f'Data r>{corr_thresh:.2f}', alpha=0.9, zorder=3)
        
        # Plot scatter points for control (squares) - shifted right
        if len(ctrl_x) > 0:
            ax.scatter(ctrl_x, ctrl_y, s=80, marker='s', 
                      color=colors[i], edgecolors='black', linewidth=1.5,
                      label=f'Control r>{corr_thresh:.2f}', alpha=0.9, zorder=3)
    
    ax.set_xlabel('Ensemble Size (Number of Neurons)', fontsize=14)
    ax.set_ylabel('Number of Ensembles', fontsize=14)
    ax.set_title('Ensemble Size Distribution by Correlation Threshold', 
                  fontsize=16, fontweight='bold')
    ax.legend(title='Correlation Threshold', fontsize=12, title_fontsize=13, loc='best')
    ax.grid(False)
    ax.set_yscale('log')
    
    # Set x-axis to only show integer values
    if all_sizes:
        min_size = min(all_sizes)
        max_size = max(all_sizes)
        # Show reasonable number of ticks
        if max_size - min_size <= 20:
            ax.set_xticks(sorted(all_sizes))
        else:
            # Show every 5th or 10th integer
            step = 5 if max_size <= 100 else 10
            ax.set_xticks(range(min_size, max_size + 1, step))
    
    save_path = os.path.join(ensemble_folder, 'ensemble_size_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")
    
    # Return statistics
    stats = {
        'n_active_neurons': int(np.sum(active_mask)),
        'ensemble_statistics': ensemble_stats
    }
    
    print("="*60)
    return stats

