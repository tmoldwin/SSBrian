"""
Analysis and visualization of synaptic weight evolution during plasticity.

This module provides functions to analyze and plot how synaptic weights
change over time due to plasticity rules.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def extract_E_to_E_weights(weights, N_rec_E, N_rec):
    """
    Extract only E->E connections from E->all weight matrix.
    
    Parameters:
    -----------
    weights : ndarray [n_synapses x n_timepoints]
        All E->all weights
    N_rec_E : int
        Number of excitatory neurons
    N_rec : int
        Total number of neurons
    
    Returns:
    --------
    weights_E_to_E : ndarray [n_E_to_E_synapses x n_timepoints]
        Only E->E weights
    """
    n_sources = N_rec_E
    n_targets_total = N_rec
    n_E_targets = N_rec_E
    
    # Extract only E->E connections
    E_to_E_indices = []
    for src in range(n_sources):
        start_idx = src * n_targets_total
        end_idx = start_idx + n_E_targets
        E_to_E_indices.extend(range(start_idx, end_idx))
    
    return weights[E_to_E_indices, :]


def plot_weight_evolution(network, connection_type='E->all', n_sample_synapses=20, only_plastic=True):
    """
    Plot weight evolution over time for a sample of synapses.
    
    Parameters:
    -----------
    network : PlasticDirectInputNetwork
        Network with weight monitors
    connection_type : str
        Type of connections to plot ('E->all' or 'I->all')
    n_sample_synapses : int
        Number of synapses to plot (randomly sampled)
    only_plastic : bool
        If True and connection_type='E->all', only plot E->E synapses (which have plasticity)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    times, weights = network.get_weight_evolution(connection_type)
    
    # Filter to only plastic synapses if requested
    if only_plastic and connection_type == 'E->all':
        weights = extract_E_to_E_weights(weights, network.N_rec_E, network.N_rec)
        connection_label = 'E->E'
    else:
        connection_label = connection_type
    
    # weights is [n_synapses x n_timepoints], need to transpose
    weights = weights.T  # Now [n_timepoints x n_synapses]
    
    n_synapses = weights.shape[1]
    
    # Pick synapses with the largest absolute change to make trends visible
    abs_change = np.abs(weights[-1, :] - weights[0, :])
    order = np.argsort(-abs_change)
    sample_indices = order[:min(n_sample_synapses, n_synapses)]
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Individual synapse traces (sample)
    ax1 = fig.add_subplot(gs[0, :])
    for idx in sample_indices:
        ax1.plot(times, weights[:, idx], alpha=0.6, linewidth=1.5)
    # Tight y-limits around the selected synapses for visibility
    subset = weights[:, sample_indices]
    y_min, y_max = subset.min(), subset.max()
    margin = 0.05 * (y_max - y_min + 1e-12)
    ax1.set_ylim(y_min - margin, y_max + margin)
    
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Weight (nA)', fontsize=12)
    ax1.set_title(f'Individual Synapse Weight Traces ({len(sample_indices)} synapses)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Mean ± std delta-weight evolution (relative to t=0)
    ax2 = fig.add_subplot(gs[1, 0])
    # Delta relative to initial weights per synapse at each time
    delta_weights = weights - weights[0, :]
    mean_delta = np.mean(delta_weights, axis=1)
    std_delta = np.std(delta_weights, axis=1)
    
    ax2.plot(times, mean_delta, 'k-', linewidth=2, label='Mean Δw')
    ax2.fill_between(times, mean_delta - std_delta, mean_delta + std_delta,
                     alpha=0.3, color='gray', label='±1 SD (Δw)')
    ax2.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('ΔWeight (nA)', fontsize=12)
    ax2.set_title('Mean ΔWeight Evolution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Median ± IQR delta-weight evolution
    ax3 = fig.add_subplot(gs[1, 1])
    median_delta = np.median(delta_weights, axis=1)
    q25_delta = np.percentile(delta_weights, 25, axis=1)
    q75_delta = np.percentile(delta_weights, 75, axis=1)
    
    ax3.plot(times, median_delta, 'b-', linewidth=2, label='Median Δw')
    ax3.fill_between(times, q25_delta, q75_delta, alpha=0.3, color='blue', label='IQR (Δw)')
    ax3.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    
    ax3.set_xlabel('Time (ms)', fontsize=12)
    ax3.set_ylabel('ΔWeight (nA)', fontsize=12)
    ax3.set_title('Median ΔWeight Evolution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Min/Max weight evolution
    ax4 = fig.add_subplot(gs[2, 0])
    min_weights = np.min(weights, axis=1)
    max_weights = np.max(weights, axis=1)
    mean_weights = np.mean(weights, axis=1)
    
    ax4.plot(times, max_weights, 'r-', linewidth=2, label='Max', alpha=0.7)
    ax4.plot(times, mean_weights, 'k-', linewidth=2, label='Mean', alpha=0.7)
    ax4.plot(times, min_weights, 'b-', linewidth=2, label='Min', alpha=0.7)
    ax4.fill_between(times, min_weights, max_weights, alpha=0.2, color='gray')
    
    ax4.set_xlabel('Time (ms)', fontsize=12)
    ax4.set_ylabel('Weight (nA)', fontsize=12)
    ax4.set_title('Min/Mean/Max Weight Evolution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Coefficient of variation over time
    ax5 = fig.add_subplot(gs[2, 1])
    std_weights_overall = np.std(weights, axis=1)
    cv = std_weights_overall / (mean_weights + 1e-10)  # Add small value to avoid division by zero
    
    ax5.plot(times, cv, 'g-', linewidth=2)
    ax5.set_xlabel('Time (ms)', fontsize=12)
    ax5.set_ylabel('Coefficient of Variation', fontsize=12)
    ax5.set_title('Weight Variability (CV) Over Time', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    fig.suptitle(f'Weight Evolution Analysis: {connection_label}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig


def plot_weight_distribution_evolution(network, connection_type='E->all', n_timepoints=6, only_plastic=True):
    """
    Plot how weight distribution changes over time using histograms.
    
    Parameters:
    -----------
    network : PlasticDirectInputNetwork
        Network with weight monitors
    connection_type : str
        Type of connections to plot
    n_timepoints : int
        Number of time points to show
    only_plastic : bool
        If True and connection_type='E->all', only plot E->E synapses
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    times, weights = network.get_weight_evolution(connection_type)
    
    # Filter to only plastic synapses if requested
    if only_plastic and connection_type == 'E->all':
        weights = extract_E_to_E_weights(weights, network.N_rec_E, network.N_rec)
        connection_label = 'E->E'
    else:
        connection_label = connection_type
    
    weights = weights.T  # Transpose to [n_timepoints x n_synapses]
    
    # Select time points evenly spaced
    n_recorded = len(times)
    if n_recorded < n_timepoints:
        time_indices = np.arange(n_recorded)
    else:
        time_indices = np.linspace(0, n_recorded-1, n_timepoints, dtype=int)
    
    # Create subplots
    n_cols = 3
    n_rows = int(np.ceil(n_timepoints / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    # Plot histogram for each time point
    for idx, time_idx in enumerate(time_indices):
        ax = axes[idx]
        weight_snapshot = weights[time_idx, :]
        
        ax.hist(weight_snapshot, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(weight_snapshot), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean={np.mean(weight_snapshot):.4f}')
        ax.axvline(np.median(weight_snapshot), color='green', linestyle='--', 
                  linewidth=2, label=f'Median={np.median(weight_snapshot):.4f}')
        
        ax.set_xlabel('Weight (nA)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f't = {times[time_idx]:.1f} ms', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(time_indices), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Weight Distribution Evolution: {connection_label}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_weight_matrices_timeline(network, connection_type='E->all', n_timepoints=4, only_plastic=True):
    """
    Plot weight matrices at different time points to visualize structural changes.
    
    Parameters:
    -----------
    network : PlasticDirectInputNetwork
        Network with weight monitors
    connection_type : str
        Type of connections to plot
    n_timepoints : int
        Number of time points to show
    only_plastic : bool
        If True and connection_type='E->all', only plot E->E synapses
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    times, weights = network.get_weight_evolution(connection_type)
    
    # Filter to only plastic synapses if requested
    if only_plastic and connection_type == 'E->all':
        weights = extract_E_to_E_weights(weights, network.N_rec_E, network.N_rec)
        connection_label = 'E->E'
        n_sources = network.N_rec_E
        n_targets = network.N_rec_E  # Only E targets
    else:
        connection_label = connection_type
        if connection_type == 'E->all':
            n_sources = network.N_rec_E
            n_targets = network.N_rec
        elif connection_type == 'I->all':
            n_sources = network.N_rec_I
            n_targets = network.N_rec
    
    weights = weights.T  # Transpose to [n_timepoints x n_synapses]
    
    # Select time points
    n_recorded = len(times)
    if n_recorded < n_timepoints:
        time_indices = np.arange(n_recorded)
    else:
        time_indices = np.linspace(0, n_recorded-1, n_timepoints, dtype=int)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_timepoints, figsize=(5*n_timepoints, 5))
    if n_timepoints == 1:
        axes = [axes]
    
    # Determine color scale limits (use same scale for all plots)
    vmin = weights.min()
    vmax = weights.max()
    
    # Plot matrix for each time point
    for idx, time_idx in enumerate(time_indices):
        ax = axes[idx]
        # Reshape the flattened weights for this time point into a matrix
        weight_vector = weights[time_idx, :]
        weight_matrix = weight_vector.reshape(n_sources, n_targets)
        
        im = ax.imshow(weight_matrix, aspect='auto', cmap='viridis', 
                      vmin=vmin, vmax=vmax, origin='lower')
        
        ax.set_xlabel('Target Neuron', fontsize=11)
        ax.set_ylabel('Source Neuron', fontsize=11)
        ax.set_title(f't = {times[time_idx]:.1f} ms\nMean={weight_matrix.mean():.4f} nA', 
                    fontsize=12, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Weight (nA)')
        
        # Add lines to separate E and I populations (only if showing E->all, not E->E)
        if connection_label == 'E->all':
            # Horizontal line at E/I boundary in targets
            ax.axhline(network.N_rec_E - 0.5, color='white', linestyle='--', 
                      linewidth=1.5, alpha=0.7)
    
    fig.suptitle(f'Weight Matrix Evolution: {connection_label}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_weight_change_statistics(network, connection_type='E->all'):
    """
    Plot statistics about weight changes: magnitude, direction, etc.
    
    Parameters:
    -----------
    network : PlasticDirectInputNetwork
        Network with weight monitors
    connection_type : str
        Type of connections to plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    times, weights = network.get_weight_evolution(connection_type)
    weights = weights.T  # Transpose to [n_timepoints x n_synapses]
    
    # Calculate weight changes
    initial_weights = weights[0, :]
    final_weights = weights[-1, :]
    weight_changes = final_weights - initial_weights
    percent_changes = (weight_changes / (initial_weights + 1e-10)) * 100
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Weight change distribution
    ax1 = axes[0, 0]
    ax1.hist(weight_changes, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax1.axvline(np.mean(weight_changes), color='green', linestyle='--', 
               linewidth=2, label=f'Mean={np.mean(weight_changes):.4f}')
    ax1.set_xlabel('Weight Change (nA)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Weight Changes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Percent change distribution
    ax2 = axes[0, 1]
    ax2.hist(percent_changes, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(percent_changes), color='green', linestyle='--', 
               linewidth=2, label=f'Mean={np.mean(percent_changes):.1f}%')
    ax2.set_xlabel('Percent Change (%)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Percent Changes', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Initial vs Final weights scatter
    ax3 = axes[1, 0]
    ax3.scatter(initial_weights, final_weights, s=10, alpha=0.5, c='steelblue')
    
    # Add diagonal line (no change)
    lims = [min(initial_weights.min(), final_weights.min()),
            max(initial_weights.max(), final_weights.max())]
    ax3.plot(lims, lims, 'r--', linewidth=2, label='No change', alpha=0.7)
    
    ax3.set_xlabel('Initial Weight (nA)', fontsize=12)
    ax3.set_ylabel('Final Weight (nA)', fontsize=12)
    ax3.set_title('Initial vs Final Weights', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # Panel 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    WEIGHT CHANGE STATISTICS
    ========================
    
    Initial Weights:
      Mean: {np.mean(initial_weights):.4f} nA
      Std:  {np.std(initial_weights):.4f} nA
      Min:  {np.min(initial_weights):.4f} nA
      Max:  {np.max(initial_weights):.4f} nA
    
    Final Weights:
      Mean: {np.mean(final_weights):.4f} nA
      Std:  {np.std(final_weights):.4f} nA
      Min:  {np.min(final_weights):.4f} nA
      Max:  {np.max(final_weights):.4f} nA
    
    Changes:
      Mean change: {np.mean(weight_changes):.4f} nA
      Std change:  {np.std(weight_changes):.4f} nA
      Mean % change: {np.mean(percent_changes):.2f}%
      
    Direction:
      Strengthened: {np.sum(weight_changes > 0)} synapses
      Weakened:     {np.sum(weight_changes < 0)} synapses
      Unchanged:    {np.sum(weight_changes == 0)} synapses
    
    Time:
      Duration: {times[-1] - times[0]:.1f} ms
      Snapshots: {len(times)} timepoints
    """
    
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle(f'Weight Change Analysis: {connection_type}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_synapse_trajectories(network, connection_type='E->all', n_synapses=10):
    """
    Plot weight trajectories for individual synapses in 2D phase space.
    
    Shows how different synapses evolve differently based on their initial
    conditions and activity patterns.
    
    Parameters:
    -----------
    network : PlasticDirectInputNetwork
        Network with weight monitors
    connection_type : str
        Type of connections to plot
    n_synapses : int
        Number of synapses to plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    times, weights = network.get_weight_evolution(connection_type)
    weights = weights.T  # Transpose to [n_timepoints x n_synapses]
    
    # Select synapses with diverse initial weights
    initial_weights = weights[0, :]
    sorted_indices = np.argsort(initial_weights)
    
    # Sample evenly from sorted list to get diverse initial conditions
    step = len(sorted_indices) // n_synapses
    sample_indices = sorted_indices[::step][:n_synapses]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Weight vs Time for selected synapses
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_indices)))
    
    for idx, syn_idx in enumerate(sample_indices):
        weight_traj = weights[:, syn_idx]
        ax1.plot(times, weight_traj, color=colors[idx], linewidth=2, 
                label=f'Syn {syn_idx} (init={weight_traj[0]:.3f})', alpha=0.7)
    
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Weight (nA)', fontsize=12)
    ax1.set_title('Individual Synapse Trajectories', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Rate of change vs Current weight
    ax2 = axes[1]
    
    # Calculate rate of change for each synapse
    dt = np.diff(times)
    for idx, syn_idx in enumerate(sample_indices):
        weight_traj = weights[:, syn_idx]
        dw_dt = np.diff(weight_traj) / dt
        
        # Plot rate vs weight
        ax2.scatter(weight_traj[:-1], dw_dt, s=20, color=colors[idx], 
                   alpha=0.6, label=f'Syn {syn_idx}')
    
    ax2.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Current Weight (nA)', fontsize=12)
    ax2.set_ylabel('Rate of Change (nA/ms)', fontsize=12)
    ax2.set_title('Weight Change Rate vs Current Weight', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Synapse Trajectory Analysis: {connection_type}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

