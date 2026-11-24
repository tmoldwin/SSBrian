"""
Input-Response Analysis Module

This module analyzes the relationship between input drive (dot product of 
input weights with stimulus) and neural response (firing rate).

Creates plots in the input_response_analysis/ subfolder.

Functions:
- generate_input_response_plots: Main entry point for input-response analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.optimize import curve_fit
import sparsity_and_selectivity as ss
import seaborn as sns


def compute_input_drive(input_weights, input_vectors, stim_type_vec):
    """
    Compute the input drive (dot product) for each neuron and each stimulus.
    
    Parameters:
    -----------
    input_weights : np.ndarray
        W_in (n_neurons × n_input_dims), input weight matrix
    input_vectors : np.ndarray
        Input vectors (n_input_dims × time_steps)
    stim_type_vec : np.ndarray
        Stimulus labels over time
    
    Returns:
    --------
    drive_matrix : np.ndarray
        (n_neurons × n_stimuli) matrix of input drives
    """
    n_neurons = input_weights.shape[0]
    
    # Get unique stimuli
    unique_stims = np.unique(stim_type_vec)
    unique_stims = unique_stims[unique_stims != -1]  # Exclude breaks
    n_stims = len(unique_stims)
    
    drive_matrix = np.zeros((n_neurons, n_stims))
    
    # For each stimulus, calculate mean input drive
    for i, stim in enumerate(unique_stims):
        # Get time points for this stimulus
        stim_mask = (stim_type_vec == stim)
        
        # Ensure mask matches input_vectors time dimension
        if len(stim_mask) > input_vectors.shape[1]:
            stim_mask = stim_mask[:input_vectors.shape[1]]
        
        # Get mean input vector for this stimulus
        if np.any(stim_mask):
            mean_input = np.mean(input_vectors[:, stim_mask], axis=1)
            
            # Compute dot product for each neuron
            drive_matrix[:, i] = input_weights @ mean_input
    
    return drive_matrix


def generate_input_response_plots(network, run_folder):
    """
    Generate plots analyzing relationship between input drive and firing rate.
    
    Parameters:
    -----------
    network : DirectInputNetwork
        The network object with simulation data
    run_folder : str
        Path to the run folder where plots will be saved
    
    Returns:
    --------
    dict : Statistics and results from the analysis
    """
    print("\n" + "="*60)
    print("INPUT-RESPONSE ANALYSIS")
    print("="*60)
    
    # Create output folder
    output_folder = os.path.join(run_folder, 'Figures', 'input_response_analysis')
    os.makedirs(output_folder, exist_ok=True)
    
    stats_dict = {}
    
    try:
        # Get necessary data - works with both live network and saved data
        if network is None:
            # Load from data dict (when regenerating)
            import pickle
            data_path = os.path.join(run_folder, 'Data', 'simulation_data.pkl')
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            input_weights = np.array(data['weight_poisson'])
            input_vectors = np.array(data['input_vectors'])
            stim_type_vec = np.array(data['stim_type_vec'])
            rate_matrix = np.array(data['rate_matrices_arr'][-1])  # Use last epoch
            sli_arr_from_data = np.array(data['selectivity_vec_arr'][-1])
            
            # For spike-based firing rates, we need rec_spikes which isn't saved
            # So we'll use the rate_matrix directly (but need to convert properly)
            # This is already in the right format from the saved data
            use_saved_rates = True
        else:
            # Get from live network
            input_weights = network.get_input_weight_matrix()
            input_vectors = network.input_vectors
            stim_type_vec = network.stim_type_vec
            rate_matrix = network.rate_matrix_rec  # (n_neurons × n_stimuli) - for selectivity only
            use_saved_rates = False
        
        print("   Computing input drive for each neuron-stimulus pair...")
        
        # Compute input drive
        drive_matrix = compute_input_drive(input_weights, input_vectors, stim_type_vec)
        
        # Compute firing rates
        if use_saved_rates:
            # When using saved data, we don't have spike-level information
            # So we need to reconstruct Hz from the rate_matrix
            # rate_matrix is the PROPORTION of time bins with spikes
            # To convert to Hz: rate = proportion / dt
            
            unique_stims = np.unique(stim_type_vec)
            unique_stims = unique_stims[unique_stims != -1]
            n_stims = len(unique_stims)
            n_neurons = rate_matrix.shape[0]
            
            # Get dt from config
            config = data.get('config', {})
            dt_ms = config.get('dt', 0.1)  # Default 0.1 ms
            dt_s = dt_ms / 1000.0
            
            # Convert proportion to Hz
            # If a neuron spikes in X proportion of time bins, and each bin is dt_s seconds,
            # then the firing rate is: X / dt_s Hz
            rate_matrix_hz = rate_matrix / dt_s
            
            # Use selectivity from saved data
            sli_arr = sli_arr_from_data
        else:
            # Compute from live network spike data
            from brian2 import ms, second
            spike_times = network.rec_spikes.t / ms  # in ms
            spike_neurons = network.rec_spikes.i
            
            # Get unique stimuli
            unique_stims = np.unique(stim_type_vec)
            unique_stims = unique_stims[unique_stims != -1]
            n_stims = len(unique_stims)
            n_neurons = network.N_rec
            
            # Compute firing rate for each neuron-stimulus pair by counting spikes
            rate_matrix_hz = np.zeros((n_neurons, n_stims))
            
            for i, stim in enumerate(unique_stims):
                # Get time indices for this stimulus
                stim_mask = (stim_type_vec == stim)
                
                # Find time range for this stimulus (in ms)
                stim_time_indices = np.where(stim_mask)[0]
                if len(stim_time_indices) > 0:
                    from brian2 import defaultclock
                    dt = float(defaultclock.dt / ms)
                    stim_start_ms = stim_time_indices[0] * dt
                    stim_end_ms = (stim_time_indices[-1] + 1) * dt
                    stim_duration_s = (stim_end_ms - stim_start_ms) / 1000.0  # in seconds
                    
                    # Count spikes for each neuron during this stimulus
                    for neuron_id in range(n_neurons):
                        # Find spikes from this neuron during this stimulus
                        neuron_spike_times = spike_times[spike_neurons == neuron_id]
                        spikes_in_stim = np.sum((neuron_spike_times >= stim_start_ms) & 
                                               (neuron_spike_times < stim_end_ms))
                        
                        # Convert to Hz
                        rate_matrix_hz[neuron_id, i] = spikes_in_stim / stim_duration_s
            
            # Compute selectivity for coloring
            _, sli_arr = ss.create_vecs(rate_matrix)
            sli_arr = np.array(sli_arr)
        
        # Flatten matrices for scatter plot
        n_neurons, n_stims = rate_matrix_hz.shape
        
        drive_flat = drive_matrix.flatten()
        rate_flat = rate_matrix_hz.flatten()
        
        # Create selectivity array for each point (repeat for each stimulus)
        sli_repeated = np.repeat(sli_arr, n_stims)
        
        # Compute correlation
        corr_coeff, p_value = stats.pearsonr(drive_flat, rate_flat)
        
        print(f"   Correlation between input drive and firing rate: r = {corr_coeff:.3f}, p = {p_value:.3e}")
        
        stats_dict['correlation'] = corr_coeff
        stats_dict['p_value'] = p_value
        stats_dict['n_points'] = len(drive_flat)
        
        # Create scatter plot
        print("   Creating input-response scatter plot...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot colored by selectivity
        scatter = ax.scatter(drive_flat, rate_flat, 
                           c=sli_repeated, 
                           cmap='Reds', 
                           alpha=0.5, 
                           s=10,
                           edgecolors='none')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Selectivity Index', fontsize=12)
        
        # Add regression line
        from scipy.stats import linregress
        slope, intercept, r_value, p_value_reg, std_err = linregress(drive_flat, rate_flat)
        x_line = np.array([np.min(drive_flat), np.max(drive_flat)])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.7, 
               label=f'Linear fit: r={corr_coeff:.3f}')
        
        # Labels and title
        ax.set_xlabel('Input Drive (W · stimulus)', fontsize=12)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
        ax.set_title('Input Drive vs. Firing Rate\n(colored by neuron selectivity)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'n = {n_neurons} neurons × {n_stims} stimuli = {len(drive_flat)} points\n'
        stats_text += f'r = {corr_coeff:.3f}, p = {p_value:.3e}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(output_folder, 'input_drive_vs_firing_rate.png')
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print(f"      Saved: input_drive_vs_firing_rate.png")

        # Add joint distribution plot with marginals (scatter + hist/kde)
        print("   Creating input-response joint distribution with marginals...")
        jp = sns.jointplot(x=drive_flat, y=rate_flat,
                           kind='kde', fill=True, cmap='mako',
                           height=8, space=0, thresh=0.05)
        jp.set_axis_labels('Input Drive (W · stimulus)', 'Firing Rate (Hz)')
        jp.fig.suptitle('Input Drive vs Firing Rate (Joint KDE with Marginals)', fontsize=14, fontweight='bold')
        jp.fig.tight_layout()
        joint_path = os.path.join(output_folder, 'input_drive_vs_firing_rate_joint.png')
        jp.fig.savefig(joint_path, bbox_inches='tight', dpi=300)
        plt.close(jp.fig)
        print("      Saved: input_drive_vs_firing_rate_joint.png")
        
        # Print summary
        print(f"\n   Input-Response Analysis Summary:")
        print(f"   Correlation: r = {corr_coeff:.3f}, p = {p_value:.3e}")
        print(f"   Total data points: {len(drive_flat)} (neurons × stimuli)")
        
        # Create 5x5 grid for highly selective neurons
        print("   Creating 5x5 grid for highly selective neurons...")
        beta_params = plot_selective_neurons_grid(drive_matrix, rate_matrix_hz, sli_arr, output_folder)
        
        # Create sigmoid parameter analysis plot
        print("   Creating sigmoid parameter analysis...")
        plot_sigmoid_parameters(drive_matrix, rate_matrix_hz, sli_arr, output_folder)
        
    except Exception as e:
        print(f"   Error in input-response analysis: {e}")
        import traceback
        traceback.print_exc()
    
    return stats_dict


def plot_selective_neurons_grid(drive_matrix, rate_matrix_hz, sli_arr, output_folder):
    """
    Create a 5x5 grid showing input drive vs firing rate for 25 randomly selected active neurons.
    Each subplot shows one neuron's responses across all stimuli.
    
    Parameters:
    -----------
    drive_matrix : np.ndarray
        Input drive matrix (n_neurons × n_stimuli)
    rate_matrix_hz : np.ndarray
        Firing rate matrix in Hz (n_neurons × n_stimuli)
    sli_arr : np.ndarray
        Selectivity indices for all neurons
    output_folder : str
        Path to save the plot
        
    Returns:
    --------
    dict : Dictionary containing fitted sigmoid parameters for selected neurons
    """
    # Filter for neurons with mean firing rate > 0
    mean_firing_rates = np.mean(rate_matrix_hz, axis=1)
    active_neurons = np.where(mean_firing_rates > 0)[0]
    
    # Randomly select 25 from active neurons
    np.random.seed(42)  # For reproducibility
    if len(active_neurons) >= 25:
        selected_25_indices = np.random.choice(active_neurons, size=25, replace=False)
    else:
        print(f"      Warning: Only {len(active_neurons)} active neurons, using all of them")
        selected_25_indices = active_neurons
    
    # Sort by neuron index for consistent ordering
    selected_25_indices = np.sort(selected_25_indices)
    
    # Create 5x5 subplot grid
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.flatten()  # Flatten for easier indexing
    
    n_stims = drive_matrix.shape[1]
    
    # Define sigmoid function
    def sigmoid(x, beta, x0, ymax):
        """Sigmoid function: ymax / (1 + exp(-beta * (x - x0)))"""
        return ymax / (1 + np.exp(-beta * (x - x0)))
    
    # Store fitted parameters
    fitted_params = {
        'neuron_ids': [],
        'beta': [],
        'x0': [],
        'ymax': [],
        'sli': []
    }
    
    for i, neuron_idx in enumerate(selected_25_indices):
        ax = axes[i]
        
        # Get data for this neuron
        neuron_drive = drive_matrix[neuron_idx, :]
        neuron_rate = rate_matrix_hz[neuron_idx, :]
        neuron_sli = sli_arr[neuron_idx]
        
        # Plot points and connect with lines
        colors = plt.cm.tab10(np.linspace(0, 1, n_stims))
        
        # Plot points for each stimulus
        for stim_idx in range(n_stims):
            ax.scatter(neuron_drive[stim_idx], neuron_rate[stim_idx], 
                      c=[colors[stim_idx]], s=30, edgecolors='black', linewidth=0.5)
        
        # Fit sigmoid to the data
        try:
            # Initial parameter guesses
            beta0 = 10  # steepness
            x0_guess = np.median(neuron_drive)  # inflection point
            ymax_guess = np.max(neuron_rate) if np.max(neuron_rate) > 0 else 1.0
            
            # Fit sigmoid
            popt, _ = curve_fit(sigmoid, neuron_drive, neuron_rate, 
                              p0=[beta0, x0_guess, ymax_guess],
                              maxfev=5000,
                              bounds=([-100, -np.inf, 0], [100, np.inf, np.inf]))
            
            beta_fit, x0_fit, ymax_fit = popt
            
            # Store parameters
            fitted_params['neuron_ids'].append(neuron_idx)
            fitted_params['beta'].append(beta_fit)
            fitted_params['x0'].append(x0_fit)
            fitted_params['ymax'].append(ymax_fit)
            fitted_params['sli'].append(neuron_sli)
            
            # Plot fitted sigmoid
            x_smooth = np.linspace(np.min(neuron_drive), np.max(neuron_drive), 100)
            y_smooth = sigmoid(x_smooth, beta_fit, x0_fit, ymax_fit)
            ax.plot(x_smooth, y_smooth, 'r-', alpha=0.7, linewidth=2)
            
            # Update title to include beta
            ax.set_title(f'Neuron {neuron_idx}\nSLI={neuron_sli:.3f}, β={beta_fit:.1f}', 
                        fontsize=10, fontweight='bold')
        except Exception as e:
            # If fitting fails, just show without sigmoid
            ax.set_title(f'Neuron {neuron_idx}\nSLI={neuron_sli:.3f}, β=N/A', 
                        fontsize=10, fontweight='bold')
        ax.set_xlabel('Input Drive', fontsize=8)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Set consistent x-axis limits for better comparison
        ax.set_xlim(np.min(drive_matrix), np.max(drive_matrix))
        # Let y-axis scale independently for each neuron
    
    # Add overall title
    fig.suptitle('Input Drive vs Firing Rate: 25 Randomly Selected Active Neurons\n' +
                '(Each subplot shows one neuron across all stimuli)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(output_folder, 'selective_neurons_grid.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"      Saved: selective_neurons_grid.png")
    
    return fitted_params


def plot_sigmoid_parameters(drive_matrix, rate_matrix_hz, sli_arr, output_folder):
    """
    Fit sigmoids to all active neurons and plot beta vs SLI and x0 vs beta.
    
    Parameters:
    -----------
    drive_matrix : np.ndarray
        Input drive matrix (n_neurons × n_stimuli)
    rate_matrix_hz : np.ndarray
        Firing rate matrix in Hz (n_neurons × n_stimuli)
    sli_arr : np.ndarray
        Selectivity indices for all neurons
    output_folder : str
        Path to save the plot
    """
    # Filter for neurons with mean firing rate > 0
    mean_firing_rates = np.mean(rate_matrix_hz, axis=1)
    active_neurons = np.where(mean_firing_rates > 0)[0]
    
    # Define sigmoid function
    def sigmoid(x, beta, x0, ymax):
        """Sigmoid function: ymax / (1 + exp(-beta * (x - x0)))"""
        return ymax / (1 + np.exp(-beta * (x - x0)))
    
    # Fit sigmoid to all active neurons
    beta_list = []
    x0_list = []
    ymax_list = []
    sli_list = []
    
    for neuron_idx in active_neurons:
        neuron_drive = drive_matrix[neuron_idx, :]
        neuron_rate = rate_matrix_hz[neuron_idx, :]
        neuron_sli = sli_arr[neuron_idx]
        
        try:
            # Initial parameter guesses
            beta0 = 10
            x0_guess = np.median(neuron_drive)
            ymax_guess = np.max(neuron_rate) if np.max(neuron_rate) > 0 else 1.0
            
            # Fit sigmoid
            popt, _ = curve_fit(sigmoid, neuron_drive, neuron_rate,
                              p0=[beta0, x0_guess, ymax_guess],
                              maxfev=5000,
                              bounds=([-100, -np.inf, 0], [100, np.inf, np.inf]))
            
            beta_fit, x0_fit, ymax_fit = popt
            
            beta_list.append(beta_fit)
            x0_list.append(x0_fit)
            ymax_list.append(ymax_fit)
            sli_list.append(neuron_sli)
        except:
            # Skip neurons where fitting fails
            continue
    
    beta_arr = np.array(beta_list)
    x0_arr = np.array(x0_list)
    ymax_arr = np.array(ymax_list)
    sli_arr_filtered = np.array(sli_list)
    
    # Remove outliers using IQR method for each parameter
    def remove_outliers(data, sli, multiplier=3.0):
        """Remove outliers using IQR method"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
        return data[mask], sli[mask]
    
    # Remove outliers from beta
    beta_clean, sli_beta = remove_outliers(beta_arr, sli_arr_filtered)
    
    # Remove outliers from x0
    x0_clean, sli_x0 = remove_outliers(x0_arr, sli_arr_filtered)
    
    # Remove outliers from ymax
    ymax_clean, sli_ymax = remove_outliers(ymax_arr, sli_arr_filtered)
    
    print(f"      Removed {len(beta_arr) - len(beta_clean)} outliers from beta")
    print(f"      Removed {len(x0_arr) - len(x0_clean)} outliers from x0")
    print(f"      Removed {len(ymax_arr) - len(ymax_clean)} outliers from ymax")
    
    # Create 1x3 subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Left plot: SLI vs Beta (outliers removed)
    ax1 = axes[0]
    ax1.scatter(beta_clean, sli_beta, c='black', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Beta (β) - Sigmoid Steepness', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Selectivity Index (SLI)', fontsize=12, fontweight='bold')
    ax1.set_title('Selectivity vs Sigmoid Steepness', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation info
    if len(beta_clean) > 2:
        corr_beta_sli, p_beta_sli = stats.pearsonr(beta_clean, sli_beta)
        ax1.text(0.05, 0.95, f'r = {corr_beta_sli:.3f}\np = {p_beta_sli:.3e}\nn = {len(beta_clean)}',
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Middle plot: SLI vs x0 (shift, outliers removed)
    ax2 = axes[1]
    ax2.scatter(x0_clean, sli_x0, c='black', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('x₀ - Sigmoid Shift (Inflection Point)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Selectivity Index (SLI)', fontsize=12, fontweight='bold')
    ax2.set_title('Selectivity vs Sigmoid Shift', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation info
    if len(x0_clean) > 2:
        corr_x0_sli, p_x0_sli = stats.pearsonr(x0_clean, sli_x0)
        ax2.text(0.05, 0.95, f'r = {corr_x0_sli:.3f}\np = {p_x0_sli:.3e}\nn = {len(x0_clean)}',
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Right plot: SLI vs ymax (scaling/max firing rate, outliers removed)
    ax3 = axes[2]
    ax3.scatter(ymax_clean, sli_ymax, c='black', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('yₘₐₓ - Maximum Firing Rate (Hz)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Selectivity Index (SLI)', fontsize=12, fontweight='bold')
    ax3.set_title('Selectivity vs Sigmoid Scaling', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation info
    if len(ymax_clean) > 2:
        corr_ymax_sli, p_ymax_sli = stats.pearsonr(ymax_clean, sli_ymax)
        ax3.text(0.05, 0.95, f'r = {corr_ymax_sli:.3f}\np = {p_ymax_sli:.3e}\nn = {len(ymax_clean)}',
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(output_folder, 'sigmoid_parameters_analysis.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"      Saved: sigmoid_parameters_analysis.png")
    print(f"      Fitted sigmoids for {len(beta_arr)} active neurons")

