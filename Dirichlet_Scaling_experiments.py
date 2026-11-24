import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from brian_experiment.sparsity_and_selectivity import selectivity_index, sparsity_index
import os

Rest = 0
Threshold = 1

# Number of dimensions/neurons
N_DIMENSIONS = 100
ALPHA = [1.0] * N_DIMENSIONS
MAX_RATE = 50  # Hz per neuron (modest maximum firing rate)

def generate_stimuli(n_stimuli, n_dimensions, alpha):
    """Generate multiple stimulus vectors from Dirichlet distribution
    Args:
        n_stimuli (int): Number of stimuli to generate
        n_dimensions (int): Number of dimensions for each stimulus
        alpha (float): Concentration parameter for Dirichlet distribution
    Returns:
        np.ndarray: Array of shape (n_stimuli, n_dimensions)
    """
    return np.random.dirichlet([alpha] * n_dimensions, size=n_stimuli)

def generate_poisson_spikes(rate, n_bins=100):
    """Generate Poisson spike train
    Args:
        rate (float): Firing rate (Hz)
        n_bins (int): Number of time bins
    Returns:
        np.ndarray: Binary spike train (1s and 0s)
    """
    return np.random.poisson(rate/n_bins, n_bins) > 0

def plot_stimuli_comparison(n_stimuli=3, n_dimensions=100, alphas=[0.01, 1.0, 50.0]):
    """Plot multiple stimuli for different alpha values as stem plots
    Args:
        n_stimuli (int): Number of stimuli per alpha value
        n_dimensions (int): Number of dimensions for each stimulus
        alphas (list): List of alpha values to compare
    """
    fig, axes = plt.subplots(len(alphas), n_stimuli, figsize=(12, 8))
    for i, alpha in enumerate(alphas):
        stimuli = generate_stimuli(n_stimuli, n_dimensions, alpha)
        for j in range(n_stimuli):
            axes[i, j].stem(stimuli[j])
            axes[i, j].set_title(f'α={alpha:.1f}, stim {j+1}')
    
    plt.tight_layout()
    os.makedirs('Figures/DirichletWeightScaling', exist_ok=True)
    plt.savefig('Figures/DirichletWeightScaling/dirichlet_stimuli_comparison.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved: Figures/DirichletWeightScaling/dirichlet_stimuli_comparison.png')

def plot_raster_comparison(n_stimuli=3, n_dimensions=100, alphas=[0.1, 1.0, 5.0], n_bins=100):
    """Plot raster plots for different stimuli and alpha values
    Args:
        n_stimuli (int): Number of stimuli per alpha value
        n_dimensions (int): Number of dimensions for each stimulus
        alphas (list): List of alpha values to compare
        n_bins (int): Number of time bins for spike trains
    """
    fig, axes = plt.subplots(len(alphas), n_stimuli, figsize=(15, 10))
    for i, alpha in enumerate(alphas):
        stimuli = generate_stimuli(n_stimuli, n_dimensions, alpha)
        for j in range(n_stimuli):
            spikes = np.array([generate_poisson_spikes(rate * MAX_RATE, n_bins) for rate in stimuli[j]])
            axes[i, j].imshow(spikes, aspect='auto', cmap='binary')
            axes[i, j].set_title(f'α={alpha:.1f}, stim {j+1}')

def plot_weight_distribution_comparison(n_dimensions=100, stim_alphas=[0.01, 1.0, 50.0], weight_alphas=[0.01, 1.0, 50.0], n_perceptrons=50, n_examples=100):
    """Plot comparison of different weight distributions (no scaling, pure Dirichlet)"""
    
    # Create figure with 7 columns (stimulus, 3 weight heatmaps, responses, selectivity, sparsity)
    fig, axes = plt.subplots(len(stim_alphas), 7, figsize=(35, 3*len(stim_alphas)))
    
    # Pre-compute all responses and weights to find separate normalizations
    all_responses = []
    all_weights = []
    for stim_alpha in stim_alphas:
        multi_stim = generate_stimuli(n_examples, n_dimensions, stim_alpha)
        for weight_alpha in weight_alphas:
            weights = generate_stimuli(n_perceptrons, n_dimensions, weight_alpha)  # Pure Dirichlet weights
            all_weights.append(weights)
            responses = np.dot(weights, multi_stim.T)
            all_responses.append(responses)
    
    # Better normalizations for visibility
    from matplotlib.colors import LogNorm, Normalize
    # Use log normalization for weights to show sparsity patterns clearly
    weight_vmin = max(1e-6, min(np.min(w[w > 0]) for w in all_weights if np.any(w > 0)))
    weight_vmax = max(np.max(w) for w in all_weights)
    weight_norm = LogNorm(vmin=weight_vmin, vmax=weight_vmax)
    
    # Extended range for response distributions  
    response_vmin = min(np.min(resp) for resp in all_responses)
    response_vmax = max(np.max(resp) for resp in all_responses)
    response_norm = Normalize(vmin=response_vmin, vmax=response_vmax)
    
    for i, stim_alpha in enumerate(stim_alphas):
        # Generate single stimulus for visualization
        stim = generate_stimuli(1, n_dimensions, stim_alpha)[0]
        axes[i, 0].stem(stim, basefmt=' ')
        axes[i, 0].set_title(f'Stimulus α={stim_alpha:.2f}')
        axes[i, 0].set_ylabel('Probability')
        
        # Generate weight matrices and responses for each weight alpha
        weight_matrices = []
        response_sets = []
        multi_stim = generate_stimuli(n_examples, n_dimensions, stim_alpha)
        
        for j, weight_alpha in enumerate(weight_alphas):
            weights = generate_stimuli(n_perceptrons, n_dimensions, weight_alpha)  # Pure Dirichlet weights
            weight_matrices.append(weights)
            responses = np.dot(weights, multi_stim.T)
            response_sets.append(responses)
            
            # Plot weight matrix heatmap (columns 1-3) with better colormap for sparsity
            im = axes[i, j+1].imshow(weights.T, aspect='auto', cmap='hot', norm=weight_norm)
            # Add sparsity summary to title
            sparsity_pct = np.mean(weights < 0.001) * 100
            axes[i, j+1].set_title(f'Weights α={weight_alpha:.2f}\n({sparsity_pct:.0f}% sparse)')
            axes[i, j+1].set_xlabel('Perceptron')
            axes[i, j+1].set_ylabel('Weight Dimension')
            fig.colorbar(im, ax=axes[i, j+1], shrink=0.6, aspect=20)
        
        # Plot response distributions (column 4) with better range and statistics
        colors = ['blue', 'green', 'red']
        linestyles = ['-', '--', ':']
        
        # Find better range for this stimulus condition
        all_flat = np.concatenate([resp.flatten() for resp in response_sets])
        x_min, x_max = np.percentile(all_flat, [1, 99])  # Use 1-99 percentile range
        
        for j, (weight_alpha, responses) in enumerate(zip(weight_alphas, response_sets)):
            flat_responses = responses.flatten()
            mean_resp = np.mean(flat_responses)
            
            # Better binning with extended range
            axes[i, 4].hist(flat_responses, bins=50, density=True, alpha=0.7, 
                           color=colors[j], label=f'α={weight_alpha:.2f} (μ={mean_resp:.3f})',
                           histtype='step', linewidth=3, linestyle=linestyles[j],
                           range=(x_min, x_max))
            
            # Add mean line
            axes[i, 4].axvline(mean_resp, color=colors[j], linestyle=':', alpha=0.8)
        
        axes[i, 4].set_title(f'Response Distributions\nStim α={stim_alpha:.2f}')
        axes[i, 4].set_xlabel('Response')
        axes[i, 4].set_ylabel('Density')
        axes[i, 4].set_xlim(x_min, x_max)
        if i == 0:  # Only show legend on first row
            axes[i, 4].legend(fontsize=8)
        axes[i, 4].grid(True, alpha=0.3)
        
        # Calculate selectivity and sparsity for this row
        row_selectivity_sets = []
        row_sparsity_sets = []
        
        for j, (weight_alpha, responses) in enumerate(zip(weight_alphas, response_sets)):
            # Calculate selectivity with outlier protection
            selectivity = []
            for p in range(n_perceptrons):
                try:
                    sel = selectivity_index(p, responses)
                    if np.isfinite(sel):  # Only keep finite values
                        selectivity.append(sel)
                except:
                    pass
            
            # Calculate sparsity with outlier protection  
            sparsity = []
            for s in range(n_examples):
                try:
                    spar = sparsity_index(s, responses)
                    if np.isfinite(spar):  # Only keep finite values
                        sparsity.append(spar)
                except:
                    pass
            
            row_selectivity_sets.append(selectivity)
            row_sparsity_sets.append(sparsity)
        
        # Plot selectivity distributions (column 5) with conservative range
        if any(len(sel) > 0 for sel in row_selectivity_sets):
            all_sel_data = [x for sel_set in row_selectivity_sets for x in sel_set]
            if len(all_sel_data) > 0:
                sel_min, sel_max = np.percentile(all_sel_data, [5, 95])
                sel_range = sel_max - sel_min
                if sel_range > 0:
                    sel_min -= sel_range * 0.1
                    sel_max += sel_range * 0.1
                else:
                    sel_min, sel_max = sel_min - 0.1, sel_max + 0.1
                
                sel_bins = np.linspace(sel_min, sel_max, 25)
                for j, (weight_alpha, selectivity) in enumerate(zip(weight_alphas, row_selectivity_sets)):
                    if len(selectivity) > 0:
                        axes[i, 5].hist(selectivity, bins=sel_bins, density=True, color=colors[j], 
                                       label=f'α={weight_alpha:.2f}', histtype='step', linewidth=3, linestyle=linestyles[j])
                
                axes[i, 5].set_xlim(sel_min, sel_max)
                axes[i, 5].set_xticks([sel_min, sel_max])
                axes[i, 5].set_xticklabels([f'{sel_min:.2f}', f'{sel_max:.2f}'])
        
        axes[i, 5].set_title(f'Selectivity Distribution\nStim α={stim_alpha:.2f}')
        axes[i, 5].set_xlabel('Selectivity Index')
        axes[i, 5].set_ylabel('Probability Density')
        if i == 0:  # Only show legend on first selectivity plot
            axes[i, 5].legend(fontsize=8)
        axes[i, 5].grid(True, alpha=0.3)
        
        # Plot sparsity distributions (column 6) with conservative range
        if any(len(spar) > 0 for spar in row_sparsity_sets):
            all_spar_data = [x for spar_set in row_sparsity_sets for x in spar_set]
            if len(all_spar_data) > 0:
                spar_min, spar_max = np.percentile(all_spar_data, [5, 95])
                spar_range = spar_max - spar_min
                if spar_range > 0:
                    spar_min -= spar_range * 0.1
                    spar_max += spar_range * 0.1
                else:
                    spar_min, spar_max = spar_min - 0.1, spar_max + 0.1
                
                spar_bins = np.linspace(spar_min, spar_max, 25)
                for j, (weight_alpha, sparsity) in enumerate(zip(weight_alphas, row_sparsity_sets)):
                    if len(sparsity) > 0:
                        axes[i, 6].hist(sparsity, bins=spar_bins, density=True, color=colors[j], 
                                       label=f'α={weight_alpha:.2f}', histtype='step', linewidth=3, linestyle=linestyles[j])
                
                axes[i, 6].set_xlim(spar_min, spar_max)
                axes[i, 6].set_xticks([spar_min, spar_max])
                axes[i, 6].set_xticklabels([f'{spar_min:.2f}', f'{spar_max:.2f}'])
        
        axes[i, 6].set_title(f'Sparsity Distribution\nStim α={stim_alpha:.2f}')
        axes[i, 6].set_xlabel('Sparsity Index')
        axes[i, 6].set_ylabel('Probability Density')
        if i == 0:  # Only show legend on first sparsity plot
            axes[i, 6].legend(fontsize=8)
        axes[i, 6].grid(True, alpha=0.3)
    
    # Add proper spacing and save
    os.makedirs('Figures/DirichletWeightScaling', exist_ok=True)
    fig.suptitle('Stimulus-Weight Sparsity Interaction: Pure Dirichlet Distributions', fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig('Figures/DirichletWeightScaling/dirichlet_weight_distributions.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('Saved: Figures/DirichletWeightScaling/dirichlet_weight_distributions.png')

def plot_combined_comparison(n_dimensions=100, alphas=[0.01, 1.0, 50.0], n_bins=1000, rate_scale=50, n_perceptrons=100, n_examples=100):
    """Plot stem and raster plots side by side for each stimulus"""
    # Generate base weights for perceptrons
    base_weights = np.random.uniform(0, 1, (n_perceptrons, n_dimensions))
    weights_k = base_weights / n_dimensions
    weights_sqrtk = base_weights / np.sqrt(n_dimensions)
    
    # Create two separate figures without axis sharing
    fig_dist, axes_dist = plt.subplots(len(alphas), 4, figsize=(16, 3*len(alphas)))
    fig_weights, axes_weights = plt.subplots(len(alphas), 7, figsize=(28, 3*len(alphas)))
    
    # Pre-compute all responses to find global min/max (using unscaled stimuli)
    all_responses_k = []
    all_responses_sqrtk = []
    for alpha in alphas:
        multi_stim = generate_stimuli(n_examples, n_dimensions, alpha)
        all_responses_k.append(np.dot(weights_k, multi_stim.T))
        all_responses_sqrtk.append(np.dot(weights_sqrtk, multi_stim.T))
    
    # Use linear normalization since responses are now much smaller
    vmin = min(np.min(all_responses_k), np.min(all_responses_sqrtk))
    vmax = max(np.max(all_responses_k), np.max(all_responses_sqrtk))
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Each plot will use its own natural x-axis range
    
    for i, alpha in enumerate(alphas):
        # Single stimulus for first column (unscaled for visualization)
        stim = generate_stimuli(1, n_dimensions, alpha)[0]
        axes_dist[i, 0].stem(stim, basefmt=' ')
        axes_dist[i, 0].set_title(f'Single Stimulus, α={alpha:.2f}')
        axes_dist[i, 0].set_ylabel('Probability')
        
        # Exemplar weights scaled by sqrt(k) - column 1
        exemplar_perceptron = i  # Use different perceptron for each row
        exemplar_weights_sqrtk = weights_sqrtk[exemplar_perceptron]
        axes_dist[i, 1].stem(exemplar_weights_sqrtk, basefmt=' ', linefmt='hotpink', markerfmt='mo')
        axes_dist[i, 1].set_title(f'Exemplar Weights (√K-scaling), α={alpha:.2f}')
        axes_dist[i, 1].set_xlabel('Weight Dimension')
        axes_dist[i, 1].set_ylabel('Weight Value')
        
        # Exemplar weights scaled by k - column 2
        exemplar_weights_k = weights_k[exemplar_perceptron]
        axes_dist[i, 2].stem(exemplar_weights_k, basefmt=' ', linefmt='green', markerfmt='go')
        axes_dist[i, 2].set_title(f'Exemplar Weights (K-scaling), α={alpha:.2f}')
        axes_dist[i, 2].set_xlabel('Weight Dimension')
        axes_dist[i, 2].set_ylabel('Weight Value')
        
        # Add inset histogram showing weight distributions
        axins = axes_dist[i, 2].inset_axes([0.6, 0.65, 0.35, 0.3])
        axins.hist(weights_k.flatten(), bins=30, density=True, alpha=0.7, color='green', label='K-scaling')
        axins.hist(weights_sqrtk.flatten(), bins=30, density=True, alpha=0.7, color='hotpink', label='√K-scaling')
        axins.set_xlabel('Weight Value', fontsize=7)
        axins.set_ylabel('Density', fontsize=7)
        axins.legend(fontsize=6)
        axins.tick_params(labelsize=6)
        
        # Distribution of sums for last column (unscaled for visualization)
        multi_stim = generate_stimuli(n_examples, n_dimensions, alpha)
        sums_k = np.dot(weights_k, multi_stim.T).flatten()
        sums_sqrtk = np.dot(weights_sqrtk, multi_stim.T).flatten()
        
        axes_dist[i, 3].hist(sums_k, bins=50, density=True, alpha=0.5, color='green', label='K-scaling')
        axes_dist[i, 3].hist(sums_sqrtk, bins=50, density=True, alpha=0.5, color='hotpink', label='√K-scaling')
        axes_dist[i, 3].set_title(f'Response Distributions, α={alpha:.2f}')
        axes_dist[i, 3].set_xlabel('Weighted Sum')
        axes_dist[i, 3].set_ylabel('Density')
        axes_dist[i, 3].set_xlim(0, 0.11)  # Hardcode x-limits for comparison
        axes_dist[i, 3].legend()
    
    # Make weight plots share y-limits for comparison
    for i in range(len(alphas)):
        # Get the y-limits from both weight plots for this row
        ylim1 = axes_dist[i, 1].get_ylim()
        ylim2 = axes_dist[i, 2].get_ylim()
        # Use the combined range
        ymin = min(ylim1[0], ylim2[0])
        ymax = max(ylim1[1], ylim2[1])
        # Set both plots to use the same y-limits
        axes_dist[i, 1].set_ylim(ymin, ymax)
        axes_dist[i, 2].set_ylim(ymin, ymax)
        
        # Multiple stimuli for weight plots (unscaled for visualization)
        for j in range(3):
            stim = generate_stimuli(1, n_dimensions, alpha)[0]
            axes_weights[i, j].stem(stim, basefmt=' ')
            axes_weights[i, j].set_title(f'α={alpha:.2f}, stim {j+1}')
            axes_weights[i, j].set_ylabel('Probability')
        
        # K-scaling responses
        responses_k = np.dot(weights_k, multi_stim.T)
        im1 = axes_weights[i, 3].imshow(responses_k, aspect='auto', cmap='viridis', norm=norm)
        axes_weights[i, 3].set_title(f'K-scaling responses, α={alpha:.2f}')
        axes_weights[i, 3].set_xlabel('Stimulus')
        axes_weights[i, 3].set_ylabel('Perceptron')
        # Add colorbar for this specific subplot
        fig_weights.colorbar(im1, ax=axes_weights[i, 3], shrink=0.6, aspect=20)
        
        # √K-scaling responses  
        responses_sqrtk = np.dot(weights_sqrtk, multi_stim.T)
        im2 = axes_weights[i, 4].imshow(responses_sqrtk, aspect='auto', cmap='viridis', norm=norm)
        axes_weights[i, 4].set_title(f'√K-scaling responses, α={alpha:.2f}')
        axes_weights[i, 4].set_xlabel('Stimulus')
        axes_weights[i, 4].set_ylabel('Perceptron')
        # Add colorbar for this specific subplot  
        fig_weights.colorbar(im2, ax=axes_weights[i, 4], shrink=0.6, aspect=20)
        
        # Calculate selectivity and sparsity 
        # Selectivity: how selective each perceptron is across stimuli
        selectivity_k = [selectivity_index(p, responses_k) for p in range(n_perceptrons)]
        selectivity_sqrtk = [selectivity_index(p, responses_sqrtk) for p in range(n_perceptrons)]
        
        # Sparsity: how sparse the population response is to each stimulus  
        sparsity_k = [sparsity_index(s, responses_k) for s in range(n_examples)]
        sparsity_sqrtk = [sparsity_index(s, responses_sqrtk) for s in range(n_examples)]
        
        # Plot selectivity distributions as TRACES with normalized probability density
        axes_weights[i, 5].hist(selectivity_k, bins=20, density=True, color='blue', label='K-scaling', 
                               histtype='step', linewidth=3, linestyle='-')
        axes_weights[i, 5].hist(selectivity_sqrtk, bins=20, density=True, color='red', label='√K-scaling', 
                               histtype='step', linewidth=3, linestyle='--')
        axes_weights[i, 5].set_title(f'Selectivity Distribution, α={alpha:.2f}')
        axes_weights[i, 5].set_xlabel('Selectivity Index')
        axes_weights[i, 5].set_ylabel('Probability Density')
        if i == 0:  # Only show legend on first selectivity plot
            axes_weights[i, 5].legend()
        axes_weights[i, 5].grid(True, alpha=0.3)
        
        # Limit x-axis labels to first and last only
        xticks = axes_weights[i, 5].get_xticks()
        if len(xticks) > 2:
            axes_weights[i, 5].set_xticks([xticks[0], xticks[-1]])
        
        # Plot sparsity distributions as TRACES with normalized probability density
        axes_weights[i, 6].hist(sparsity_k, bins=20, density=True, color='blue', label='K-scaling', 
                               histtype='step', linewidth=3, linestyle='-')
        axes_weights[i, 6].hist(sparsity_sqrtk, bins=20, density=True, color='red', label='√K-scaling', 
                               histtype='step', linewidth=3, linestyle='--')
        axes_weights[i, 6].set_title(f'Sparsity Distribution, α={alpha:.2f}')
        axes_weights[i, 6].set_xlabel('Sparsity Index')
        axes_weights[i, 6].set_ylabel('Probability Density')
        if i == 0:  # Only show legend on first sparsity plot
            axes_weights[i, 6].legend()
        axes_weights[i, 6].grid(True, alpha=0.3)
        
        # Limit x-axis labels to first and last only
        xticks = axes_weights[i, 6].get_xticks()
        if len(xticks) > 2:
            axes_weights[i, 6].set_xticks([xticks[0], xticks[-1]])
    
    # Colorbars are now added individually for each subplot to avoid sharing
    
    # Add proper spacing and save both figures
    os.makedirs('Figures/DirichletWeightScaling', exist_ok=True)
    
    fig_dist.suptitle('Dirichlet Stimuli and Response Distributions', fontsize=14)
    fig_dist.tight_layout()
    fig_dist.savefig('Figures/DirichletWeightScaling/dirichlet_distributions.png', bbox_inches='tight', dpi=150)
    plt.close(fig_dist)
    print('Saved: Figures/DirichletWeightScaling/dirichlet_distributions.png')
    
    fig_weights.suptitle('Dirichlet Stimuli and Network Responses', fontsize=14)
    fig_weights.tight_layout()
    fig_weights.savefig('Figures/DirichletWeightScaling/dirichlet_network_responses.png', bbox_inches='tight', dpi=150)
    plt.close(fig_weights)
    print('Saved: Figures/DirichletWeightScaling/dirichlet_network_responses.png')

def plot_stimulus_response_analysis(alphas=[0.01, 1.0, 50.0], n_dimensions=100, n_bins=1000, 
                                  rate_scale=1, n_perceptrons=10, tau=10):
    """Create separate figures for each alpha showing stimulus, spikes, and voltage responses"""
    # Generate base weights for perceptrons (same weights used across all alphas)
    base_weights = np.random.uniform(0, 1, (n_perceptrons, n_dimensions))
    weights_k = base_weights / n_dimensions
    weights_sqrtk = base_weights / np.sqrt(n_dimensions)
    
    # Create color map for perceptrons
    colors = plt.cm.rainbow(np.linspace(0, 1, n_perceptrons))
    
    for alpha in alphas:
        # Create figure for this alpha with constrained layout
        fig, axes = plt.subplots(3, 5, figsize=(25, 12), constrained_layout=True)
        fig.suptitle(f'α = {alpha:.2f}', fontsize=16, y=1.02)
        
        # Generate 3 stimuli for this alpha (unscaled for visualization)
        stimuli = generate_stimuli(3, n_dimensions, alpha)
        
        for row in range(3):
            stim = stimuli[row]
            
            # Column 0: Stimulus pattern
            axes[row, 0].stem(stim, basefmt=' ')
            axes[row, 0].set_title(f'Stimulus {row+1}')
            axes[row, 0].set_ylabel('Probability')
            
            # Column 1: Spike trains (scale only for spike generation)
            spike_times = []
            stim_scaled = stim * n_dimensions * rate_scale  # Scale for realistic firing rates
            for i in range(n_dimensions):
                rate = stim_scaled[i]
                if rate > 0:
                    n_spikes = np.random.poisson(rate * n_bins / 1000)
                    times = np.random.uniform(0, n_bins, n_spikes)
                    spike_times.extend([(t, i) for t in times])
            
            if spike_times:
                times, neurons = zip(*spike_times)
                axes[row, 1].scatter(times, neurons, s=1, alpha=0.7)
            axes[row, 1].set_title(f'Spike Raster')
            axes[row, 1].set_ylabel('Neuron')
            
            # Column 2: Responses
            response_k = np.dot(weights_k, stim)
            response_sqrtk = np.dot(weights_sqrtk, stim)
            
            x = np.arange(n_perceptrons)
            axes[row, 2].bar(x - 0.2, response_k, 0.4, label='K-scaling', alpha=0.7)
            axes[row, 2].bar(x + 0.2, response_sqrtk, 0.4, label='√K-scaling', alpha=0.7)
            axes[row, 2].set_title('Perceptron Responses')
            axes[row, 2].set_ylabel('Response')
            axes[row, 2].legend()
            
            # Column 3: Voltage traces (K-scaling) - use scaled stim for spike generation
            dt = 1.0
            time = np.arange(n_bins) * dt
            v_k = []
            for p in range(n_perceptrons):
                v = np.zeros(n_bins)
                for i in range(n_dimensions):
                    rate = stim_scaled[i]  # Use scaled stimulus for spike generation
                    if rate > 0:
                        n_spikes = np.random.poisson(rate * n_bins / 1000)
                        spike_times = np.random.uniform(0, n_bins, n_spikes).astype(int)
                        for t in spike_times:
                            if t < n_bins:
                                v[t:] += weights_k[p, i] * np.exp(-(np.arange(n_bins - t)) / tau)
                v_k.append(v)
                axes[row, 3].plot(v[p], color=colors[p], alpha=0.7,
                                label=f'P{p}' if row == 0 else '')
            
            axes[row, 3].set_title('Voltage Traces (K-scaling)')
            axes[row, 3].set_ylabel('Voltage')
            if row == 0:
                axes[row, 3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Column 4: Voltage traces (√K-scaling) - use scaled stim for spike generation
            v_sqrtk = []
            for p in range(n_perceptrons):
                v = np.zeros(n_bins)
                for i in range(n_dimensions):
                    rate = stim_scaled[i]  # Use scaled stimulus for spike generation
                    if rate > 0:
                        n_spikes = np.random.poisson(rate * n_bins / 1000)
                        spike_times = np.random.uniform(0, n_bins, n_spikes).astype(int)
                        for t in spike_times:
                            if t < n_bins:
                                v[t:] += weights_sqrtk[p, i] * np.exp(-(np.arange(n_bins - t)) / tau)
                v_sqrtk.append(v)
                axes[row, 4].plot(v_sqrtk[p], color=colors[p], alpha=0.7)
            
            axes[row, 4].set_title('Voltage Traces (√K-scaling)')
            axes[row, 4].set_ylabel('Voltage')
        
        # Save figure for this alpha
        os.makedirs('Figures/DirichletWeightScaling', exist_ok=True)
        filename = f'Figures/DirichletWeightScaling/dirichlet_stimulus_response_alpha_{alpha:.2f}.png'
        fig.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f'Saved: {filename}')

def window_responses(responses, window_size=100):
    """Convert response timeseries into windowed responses
    
    Args:
        responses: array of shape (n_neurons, n_stimuli, n_timepoints)
        window_size: number of timepoints per window
    
    Returns:
        windowed_means: array of shape (n_neurons, n_stimuli, n_windows)
    """
    n_neurons, n_stimuli, n_timepoints = responses.shape
    n_windows = n_timepoints // window_size
    
    # Reshape into windows
    windowed = responses[:, :, :n_windows*window_size].reshape(n_neurons, n_stimuli, n_windows, window_size)
    
    # Calculate mean rate for each window
    return windowed.mean(axis=3)  # shape: (n_neurons, n_stimuli, n_windows)

def consistency_ratio(response_means):
    """Calculate consistency ratio for neural responses
    
    Args:
        response_means: array of shape (n_neurons, n_stimuli, n_trials/windows)
            Mean responses for each neuron to each stimulus across trials/windows
    
    Returns:
        consistency_ratios: array of shape (n_neurons)
        
    For each neuron:
    - Between-stimulus variance: how different are responses to different stimuli
    - Within-stimulus variance: how variable are responses to same stimulus across trials
    - Ratio = between / within (higher = more consistent coding)
    """
    n_neurons = response_means.shape[0]
    ratios = np.zeros(n_neurons)
    
    for n in range(n_neurons):
        # Between-stimulus variance: variance of means across stimuli
        between_var = np.var(response_means[n].mean(axis=1))
        
        # Within-stimulus variance: mean of variances across trials for each stimulus
        within_var = np.mean([np.var(response_means[n, s]) for s in range(response_means.shape[1])])
        
        # Compute ratio (add small constant to avoid division by zero)
        ratios[n] = between_var / (within_var + 1e-6)
    
    return ratios

# Run the analysis

# Only run the function that creates the dirichlet_distributions.png figure
plot_combined_comparison()



