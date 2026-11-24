import numpy as np
from brian2 import *
import matplotlib
from matplotlib import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import weight_matrices as wm
import itertools as it
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from constants import SAVE_FOLDER
import os


plt.rcParams['legend.fontsize'] = 8  # You can adjust this value as needed
plt.rcParams['legend.title_fontsize'] = 9  # For legend titles, if you have any

prefs.codegen.target = "numpy"



def plot_weight_matrix(ax, weight_matrix, weight_matrix_poisson):
    cmap = mcolors.TwoSlopeNorm(vcenter=0)
    im = ax.imshow(weight_matrix,  cmap='bwr', norm=cmap, aspect='auto')
    im = ax.imshow(weight_matrix,  cmap='bwr', norm=cmap, aspect='auto')
    ax.set_xlabel('Presynaptic Neuron')
    ax.set_ylabel('Postsynaptic Neuron')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Weight (mV)')
    if(str(weight_matrix) == str(weight_matrix_poisson)):
        ax.set_title('Weight Matrix Poisson Group')
    else:
        ax.set_title('Weight Matrix Recurrent Group')

def plot_external_input(ax, matrix, T, N):
    im = ax.imshow(matrix, cmap='coolwarm', aspect='auto', extent=[0, T / ms, 0, N])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Pitch')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Amplitude dB')
    ax.set_title('External Input')

def plot_external_stimuli(ax, matrix):
    im = ax.imshow(matrix, cmap='coolwarm', origin = 'lower', aspect = 'auto',
                   extent=[0, matrix.shape[1], 0, matrix.shape[0]])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='External Input (mV)')
    ax.set_title('External Input')

def plot_membrane_potential(ax, voltage, T):
    ax.plot(voltage.t / ms, voltage.v.T / mV)
    ax.set_xlim(0, T / ms)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title('Membrane Potential of rec neurons')

def plot_raster(ax, spikemon_rec, num_E, num_I, T, neuron_order=None, stim_info=None, title="Spike Raster Plot"):
    t = spikemon_rec.t / ms
    i = spikemon_rec.i
    N = num_E + num_I

    # Determine colors based on original neuron ordering
    colors = np.where(i < num_E, 'red', 'blue')

    if neuron_order is not None:
        # Create a mapping from old to new indices
        index_map = {old: new for new, old in enumerate(neuron_order)}
        new_i = np.array([index_map[idx] for idx in i])

        # Sort t, i, and colors based on the new ordering
        sort_indices = np.argsort(new_i)
        t = t[sort_indices]
        i = new_i[sort_indices]
        colors = colors[sort_indices]

    # Use scatter instead of individual plot calls
    ax.scatter(t, i, c=colors, marker='.', s=2)
    # Use scatter instead of individual plot calls
    ax.scatter(t, i, c=colors, marker='.', s=2)

    ax.set_xlim(0, T/ms)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_title(title)

    if neuron_order is None:
        ax.axhline(y=num_E - 0.5, color='k', linestyle=':', linewidth=0.7)

        ax.axhline(y=num_E - 0.5, color='k', linestyle=':', linewidth=0.7)

    # Add a legend
    ax.plot([], [], '.r', label='Excitatory')
    ax.plot([], [], '.b', label='Inhibitory')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    # Calculate and plot running average firing rates
    bin_size = 50  # ms
    bins = np.arange(0, T/ms, bin_size)

    ax2 = ax.twinx()

    for label, color, start, end in [('E', 'red', 0, num_E),
                                     ('I', 'blue', num_E, N),
                                     ('All', 'black', 0, N)]:
        mask = (i >= start) & (i < end)
        spike_counts, _ = np.histogram(t[mask], bins=bins)
        firing_rates = spike_counts / (bin_size * (end - start) / 1000)  # Convert to Hz
        ax2.plot(bins[:-1], firing_rates, color=color, linewidth = 1.2, linestyle = '--', alpha = 0.5, label=label)

    ax2.set_ylabel('Average Firing Rate (Hz)')
    ax2.tick_params(axis='y')
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 0.7))

    if stim_info is not None:
        show_stim_v_lines(stim_info, T, ax)
    return ax, ax2

def show_stim_v_lines(stim_info, T, ax):
    for stim, start, end in stim_info:
        start = start / ms
        end = end / ms
        if start < T/ms:
            ax.axvline(x=start, color='red', linestyle='--', alpha=0.8, linewidth=1.2)
            ax.text(start, ax.get_ylim()[1], f'{stim}', rotation=90, 
                    verticalalignment='top', horizontalalignment='right', 
                    fontsize=12, color='red', fontweight='bold')
    
    
def imshow_membrane_potential(ax, state_monitor, T, N, neuron_order=None):
    v = state_monitor.v / mV
    if neuron_order is not None:
        v = v[neuron_order]
    im = ax.imshow(v, cmap='coolwarm', aspect='auto', extent=[0, T/ms, 0, N])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Membrane Potential (mV)')
    #change to the threshold.
    ax.set_title('Recurrent Group membrane potential')


def plot_smoothed_rates(smoothed_rates, time_points, ax=None, title="Smoothed Firing Rates", plot_type='heatmap',
                        neuron_order=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    # Ensure smoothed_rates is a numpy array
    smoothed_rates = np.asarray(smoothed_rates)

    # Apply neuron ordering if provided
    if neuron_order is not None:
        smoothed_rates = smoothed_rates[neuron_order]

    # Handle 1D array case
    if smoothed_rates.ndim == 1:
        smoothed_rates = smoothed_rates.reshape(1, -1)

    num_neurons, num_timepoints = smoothed_rates.shape

    # Convert time_points to milliseconds if they have units
    if hasattr(time_points, 'unit'):
        time_points = time_points / ms

    if plot_type == 'line':
        for i in range(num_neurons):
            ax.plot(time_points, smoothed_rates[i], label=f'Neuron {i + 1}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    elif plot_type == 'heatmap':
        ax.imshow(smoothed_rates, aspect='auto', extent=[time_points[0], time_points[-1], num_neurons - 0.5, -0.5],
                  cmap='viridis', origin='lower')
    else:
        raise ValueError("plot_type must be either 'line' or 'heatmap'")

        # Format x-axis ticks
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron' if num_neurons > 1 else 'Firing Rate (spikes/s)')
        ax.set_title(title)

        plt.tight_layout()
    return ax


def plot_average_firing_rates(smoothed_rates, time_points, neuron_groups, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'E': 'red', 'I': 'blue', 'all': 'black'}
    
    # Convert time_points from seconds to milliseconds for consistency with other panels
    time_points_ms = time_points * 1000
    
    # Calculate and plot average for all neurons
    all_avg = np.mean(smoothed_rates, axis=0)
    ax.plot(time_points_ms, all_avg, color=colors['all'], label='All neurons', linewidth=2)
    
    # Calculate and plot average for each neuron group
    for group_name, indices in neuron_groups.items():
        if group_name != 'all':
            group_avg = np.mean(smoothed_rates[indices], axis=0)
            ax.plot(time_points_ms, group_avg, color=colors.get(group_name, 'gray'), 
                    label=f'{group_name} neurons', linewidth=2)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Average Firing Rate (Hz)')
    ax.set_title('Average Smoothed Firing Rates by Neuron Group')
    ax.legend()
    
    return ax


def label_panels_mosaic(fig, axes, xloc = 0, yloc = 1.0, size = 20):
    for key in axes.keys():
    # label physical distance to the left and up:
        ax = axes[key]
        trans = transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
        ax.text(xloc, yloc, key, transform=ax.transAxes + trans,
                fontsize=size, va='bottom')

def plot_histogram(ax, weight_matrix):
    ax.hist(np.ravel(weight_matrix), bins = 20)
    ax.hist(np.ravel(weight_matrix), bins = 20)


def visualise_connectivity(S, wm):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    norm = mcolors.TwoSlopeNorm(vcenter=0)
    # Colormap
    num_lines = len(S.i)
    cmap = plt.get_cmap('bwr')

    plt.subplot(121)
    plt.plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
    plt.plot(np.ones(Nt), np.arange(Nt), 'ok', ms=10)
    for idx, (i, j) in enumerate(zip(S.i, S.j)):
        plt.plot([0, 1], [i, j], c = cmap(norm(wm[i, j])))
        plt.plot([0, 1], [i, j], c = cmap(norm(wm[i, j])))
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))

    plt.subplot(122)
    plt.scatter(S.i, S.j, c=np.arange(num_lines), cmap=cmap)
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')
    plt.colorbar(label='Connection index')

def plot_firing_rates_matrix(spike_matrix, stims_by_time, neuron_order=None, title="Firing Rates Matrix", ax = None):
    # Generate firing rates matrix
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    unique_stims = np.unique(stims_by_time)
    num_neurons, num_timesteps = spike_matrix.shape
    mat = np.zeros((num_neurons, len(unique_stims)))
    
    for i, stim in enumerate(unique_stims):
        stim_mask = stims_by_time == stim
        mat[:, i] = np.mean(spike_matrix[:, stim_mask], axis=1)
    
    # Apply neuron ordering if provided
    if neuron_order is not None:
        mat = mat[neuron_order]
    
    # Create a colormap for non-negative values
    cmap = plt.cm.Reds
    
    # Plot the matrix
    im = ax.imshow(mat[::-1], cmap=cmap, vmin=0, vmax=np.max(mat), aspect='auto', interpolation='None')
    # Add vertical gridlines separating the stimuli
    for x in range(1, len(unique_stims)):
        ax.axvline(x - 0.5, color='white', linestyle='--', linewidth=0.5)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Firing Rate')
    
    # Set labels and title
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Neuron')
    ax.set_title(title)
    
    # Set x-ticks to show stimulus numbers
    ax.set_xticks(range(len(unique_stims)))
    ax.set_xticklabels(unique_stims)
    
    return ax

def plot_rate_correlation(ax, poisson_rates, N_poiss_E, N_poiss_I, recurrent_rates):
    # Calculate average smoothed rates for poisson and recurrent groups
    avg_poisson_E_rate = np.mean(poisson_rates[:N_poiss_E], axis=0) if N_poiss_E > 0 else np.zeros_like(
        recurrent_rates[0])
    avg_poisson_I_rate = np.mean(poisson_rates[N_poiss_E:], axis=0) if N_poiss_I > 0 else np.zeros_like(
        recurrent_rates[0])
    avg_recurrent_rate = np.mean(recurrent_rates, axis=0)

    # Create separate scatter plots for E and I poisson rates vs recurrent rates
    ax.scatter(avg_poisson_E_rate, avg_recurrent_rate, alpha=0.6, color='red', label='E neurons', s=20)
    ax.scatter(avg_poisson_I_rate, avg_recurrent_rate, alpha=0.6, color='blue', label='I neurons', s=20)
    
    ax.set_xlabel('Average Poisson Rate (Hz)')
    ax.set_ylabel('Average Recurrent Rate (Hz)')
    ax.set_title('Correlation: Poisson vs Recurrent Rates')
    ax.legend()

    # Calculate separate correlation coefficients
    corr_coef_E = np.corrcoef(avg_poisson_E_rate, avg_recurrent_rate)[0, 1]
    corr_coef_I = np.corrcoef(avg_poisson_I_rate, avg_recurrent_rate)[0, 1]
    
    # Display both correlation coefficients
    ax.text(0.05, 0.95, f'E Correlation: {corr_coef_E:.2f}',
            transform=ax.transAxes, verticalalignment='top', color='red')
    ax.text(0.05, 0.88, f'I Correlation: {corr_coef_I:.2f}',
            transform=ax.transAxes, verticalalignment='top', color='blue')


def plot_weight_histogram(ax, weight_matrix, bins=30, type = 'Recurrent'):
    """
    Plots a histogram of excitatory weights, inhibitory weights, and all weights from a weight matrix on the given axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which the histogram will be plotted.
    weight_matrix (numpy.ndarray): The weight matrix containing excitatory and inhibitory weights.
    bins (int): Number of bins for the histogram. Default is 30.
    """
    # Flatten the weight matrix to get all weights
    all_weights = weight_matrix.flatten()

    # Separate excitatory and inhibitory weights
    excitatory_weights = all_weights[all_weights > 0]
    inhibitory_weights = all_weights[all_weights < 0]

    # Plot the histogram
    ax.hist(all_weights, bins=bins, alpha=0.7, label='Combined', color='purple', density=True)
    ax.hist(excitatory_weights, bins=bins, alpha=0.8, label='Excitatory', color='blue', density=True)
    ax.hist(inhibitory_weights, bins=bins, alpha=0.8, label='Inhibitory', color='red', density=True)

    # Add labels, title, and legend
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Histogram of {type} Weights')
    ax.legend('upper right')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)



# Load and display an image
def display_image(ax, image_path):
    """
    Displays an image using imshow.
    Parameters: image path
    image_path (str): The file path to the image to be displayed.
    """
    img = mpimg.imread(image_path)  # Load the image
    ax.imshow(img)  # Display the image
    ax.axis('off')  # Hide axes for a clean display


def plot_two_vector_arrays_in_panels_with_colors(sparsity_vec_arr, selectivity_vec_arr, x_label=None, title=None):
    """
    Plot sparsity and selectivity vectors over time with color coding.

    Parameters:
    - sparsity_vec_arr: Array of sparsity vectors
    - selectivity_vec_arr: Array of selectivity vectors
    - x_label: Label for x-axis (optional)
    - title: Title for the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot sparsity vectors
    for i in range(len(sparsity_vec_arr)):
        ax1.plot(sparsity_vec_arr[i], label=f'Stim {i}')
    ax1.set_ylabel('Sparsity')
    if x_label:
        ax1.set_xlabel(x_label)
    ax1.legend()

    # Plot selectivity vectors
    for i in range(len(selectivity_vec_arr)):
        ax2.plot(selectivity_vec_arr[i], label=f'Neuron {i}')
    ax2.set_ylabel('Selectivity')
    if x_label:
        ax2.set_xlabel(x_label)
    ax2.legend()

    if title:
        fig.suptitle(title)

    plt.tight_layout()

def plot_vector_array(ax, vec_arr, ylabel=None, x_label=None, title=None):
    """
    Plot a single vector array over time with color coding.

    Parameters:
    - ax: Matplotlib axis on which to plot the vector array.
    - vec_arr: Array of vectors to plot.
    - ylabel: Label for y-axis (optional).
    - x_label: Label for x-axis (optional).
    - title: Title for the plot (optional).
    """
    # Plot each vector in the array
    for i in range(len(vec_arr)):
        ax.plot(vec_arr[i], label=f'Item {i}')

    # Set axis labels and title
    if ylabel:
        ax.set_ylabel(ylabel)
    if x_label:
        ax.set_xlabel(x_label)
    if title:
        ax.set_title(title)

    # Adjust layout (handled outside for a dashboard)
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_selectivity_comparison(ax, rate_matrix_before, sli_arr_before, rate_matrix_after, sli_arr_after):
    """
    Plots selectivity comparison for the same neurons before and after simulation.
    The stimulus order is determined by the 'before' rates and preserved for the 'after' plot.
    """
    # Randomly choose 9 neurons
    n_neurons = rate_matrix_before.shape[0]
    selected_indices = np.random.choice(n_neurons, 9, replace=False)
    
    # Get data for selected neurons
    selected_rate_before = rate_matrix_before[selected_indices, :]
    selected_sli_before = np.array(sli_arr_before[selected_indices])
    selected_rate_after = rate_matrix_after[selected_indices, :]
    selected_sli_after = np.array(sli_arr_after[selected_indices])
    
    # Create colors for each stimulus
    n_stims = rate_matrix_before.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_stims))
    
    # Create 2x9 grid layout
    grid = ax.figure.add_gridspec(2, 9, left=ax.get_position().x0, right=ax.get_position().x1,
                                  top=ax.get_position().y1, bottom=ax.get_position().y0, 
                                  hspace=0.6, wspace=0.4)
    
    for i, neuron_idx in enumerate(selected_indices):
        # --- Before simulation (top row) ---
        ax_before = ax.figure.add_subplot(grid[0, i])
        row_before = selected_rate_before[i]
        sli_before = selected_sli_before[i]
        
        # Sort stimuli by response strength (descending)
        stim_order = np.argsort(row_before)[::-1]
        sorted_rates = row_before[stim_order]
        stim_colors = colors[stim_order]
        
        # Plot stems
        for j, (rate, color) in enumerate(zip(sorted_rates, stim_colors)):
            markerline, stemlines, baseline = ax_before.stem([j], [rate], linefmt='-', markerfmt='o', basefmt=' ')
            plt.setp(stemlines, 'color', color)
            plt.setp(markerline, 'color', color)
            
        ax_before.set_title(f'Neuron {neuron_idx}\nSLI = {sli_before:.2f}', fontsize=8)
        if i == 0:
            ax_before.set_ylabel('Before\nRate (Hz)', fontsize=8)

        # --- After simulation (bottom row) ---
        ax_after = ax.figure.add_subplot(grid[1, i])
        row_after = selected_rate_after[i]
        sli_after = selected_sli_after[i]
        
        # Apply the same stimulus order from the 'before' plot
        sorted_rates_after = row_after[stim_order]
        
        # Plot stems with the same colors
        for j, (rate, color) in enumerate(zip(sorted_rates_after, stim_colors)):
            markerline, stemlines, baseline = ax_after.stem([j], [rate], linefmt='-', markerfmt='o', basefmt=' ')
            plt.setp(stemlines, 'color', color)
            plt.setp(markerline, 'color', color)
            
        ax_after.set_title(f'SLI = {sli_after:.2f}', fontsize=8)
        if i == 0:
            ax_after.set_ylabel('After\nRate (Hz)', fontsize=8)
            
        # Common formatting for both subplots
        for sub_ax in [ax_before, ax_after]:
            sub_ax.set_ylim(bottom=0)
            sub_ax.spines['top'].set_visible(False)
            sub_ax.spines['right'].set_visible(False)
            sub_ax.tick_params(axis='both', which='major', labelsize=6)
            
    ax.axis('off')

def plot_stimuli_imshow(stim_matrix, stim_by_time=None, ax=None, title="Stimulus Patterns"):
    """
    Create a heatmap visualization of stimulus patterns in a square figure.
    
    Parameters:
    -----------
    stim_matrix : ndarray
        Matrix of stimulus patterns with shape (n_neurons, n_stimuli)
    stim_by_time : ndarray, optional
        Vector indicating which stimulus is active at each time point
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    title : str, optional
        Title for the plot
    """
    # Create a new figure if ax is not provided
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
        fig.set_size_inches(8, 8)
    
    # Create a mesh grid for pcolormesh
    X, Y = np.meshgrid(np.arange(stim_matrix.shape[1]+1), np.arange(stim_matrix.shape[0]+1))
    
    # Plot the heatmap with pcolormesh
    im = ax.pcolormesh(X, Y, stim_matrix, cmap='Reds', shading='flat')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    plt.colorbar(im, cax=cax, label="Stimulus Strength")
    
    # Set labels and title
    ax.set_xlabel('Stimulus ID')
    ax.set_ylabel('Neuron Index')
    ax.set_title(title)
    
    # Adjust x-ticks to be in the center of the cells
    ax.set_xticks(np.arange(0.5, stim_matrix.shape[1]))
    
    # Set x-tick labels if we don't have too many
    if stim_matrix.shape[1] <= 20:
        if stim_by_time is not None:
            unique_stims = np.unique(stim_by_time)
            unique_stims = unique_stims[unique_stims != -1]  # Remove -1 (no stimulus)
            ax.set_xticklabels(unique_stims)
        else:
            ax.set_xticklabels(range(stim_matrix.shape[1]))
    
    # Add vertical lines to separate different stimuli if needed
    if stim_by_time is not None:
        unique_stims = np.unique(stim_by_time)
        unique_stims = unique_stims[unique_stims != -1]  # Remove -1 (no stimulus)
        for i in range(1, len(unique_stims)):
            ax.axvline(x=i, color='black', linestyle='-', linewidth=1)
    
    # Set appropriate limits
    ax.set_xlim(0, stim_matrix.shape[1])
    ax.set_ylim(0, stim_matrix.shape[0])
    
    # Use a fixed ratio for the display box (not the cells)
    ax.set_box_aspect(1)
    
    # Make sure the figure is square
    plt.tight_layout()
    
    return im

def plot_stimuli_lollipop(stim_patterns, fig=None, figsize=(12, 8), title="Individual Stimulus Patterns"):
    """
    Plot each stimulus as a separate lollipop plot.
    
    Parameters:
    - stim_patterns: dict or numpy.ndarray, stimulus patterns
                     If dict: {stim_id: pattern_array}
                     If array: shape (num_stims, num_inputs)
    - fig: matplotlib.figure.Figure, figure to plot on (optional)
    - figsize: tuple, figure size (default: (12, 8))
    - title: str, figure title (default: "Individual Stimulus Patterns")
    
    Returns:
    - matplotlib.figure.Figure: The figure with the plots
    - list: List of axes with the plots
    """
    # Convert input to a dictionary if it's an array
    if isinstance(stim_patterns, np.ndarray):
        stim_patterns = {i: stim_patterns[i] for i in range(len(stim_patterns))}
    
    num_stims = len(stim_patterns)
    
    # Calculate grid dimensions
    n_cols = min(4, num_stims)
    n_rows = (num_stims + n_cols - 1) // n_cols
    
    if fig is None:
        fig = plt.figure(figsize=figsize)
    
    fig.suptitle(title, fontsize=16)
    
    axes = []
    for i, (stim_id, pattern) in enumerate(stim_patterns.items()):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Create lollipop plot
        x = np.arange(len(pattern))
        markerline, stemlines, baseline = ax.stem(x, pattern, linefmt='grey', markerfmt='o', basefmt=' ')
        plt.setp(markerline, markersize=4)
        plt.setp(stemlines, linewidth=1)
        
        # Set labels and title
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Stimulus Strength')
        ax.set_title(f'Stimulus {stim_id}')
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        axes.append(ax)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    return fig, axes

def plot_stimuli_combined(stim_pattern_matrix, stim_patterns, unique_stims, title="Stimulus Patterns"):
    """
    Create a combined figure with lollipop plots and imshow visualization.
    
    Parameters:
    -----------
    stim_pattern_matrix : ndarray
        Matrix of stimulus patterns with shape (n_neurons, n_stimuli)
    stim_patterns : dict
        Dictionary mapping stimulus IDs to their patterns
    unique_stims : array-like
        Array of unique stimulus IDs
    title : str, optional
        Title for the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    n_stims = len(unique_stims)
    fig = plt.figure(figsize=(n_stims*3 + 4, 8))
    
    # Create a grid with n_stims+1 columns (n_stims for lollipops + 1 for imshow)
    gs = fig.add_gridspec(1, n_stims+1, width_ratios=[1]*n_stims + [1.5])
    
    # Add lollipop plots for each stimulus
    for i, stim in enumerate(unique_stims):
        ax = fig.add_subplot(gs[0, i])
        pattern = stim_patterns[stim]
        x = np.arange(len(pattern))
        markerline, stemlines, baseline = ax.stem(x, pattern, linefmt='grey', markerfmt='o', basefmt=' ')
        plt.setp(markerline, markersize=4)
        plt.setp(stemlines, linewidth=1)
        ax.set_title(f'Stimulus {stim}')
        ax.set_xlabel('Neuron Index')
        if i == 0:  # Only add y-label to first subplot
            ax.set_ylabel('Stimulus Strength')
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add imshow plot as the final column
    ax_im = fig.add_subplot(gs[0, -1])
    im = ax_im.imshow(stim_pattern_matrix, aspect='auto', cmap='viridis')
    ax_im.set_title('All Patterns')
    ax_im.set_xlabel('Stimulus ID')
    ax_im.set_ylabel('Neuron Index')
    ax_im.set_xticks(np.arange(len(unique_stims)))
    ax_im.set_xticklabels(unique_stims)
    fig.colorbar(im, ax=ax_im, label='Stimulus Strength')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    return fig

def plot_rate_matrix(rate_matrix, ax, title="Firing Rate Matrix"):
    """
    Plots a firing rate matrix directly.
    """
    im = ax.imshow(rate_matrix, aspect='auto', interpolation='none')
    ax.set_title(title)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Neuron')
    plt.colorbar(im, ax=ax)

def plot_weight_distribution(ax, W, color='blue', label=''):
    flattened_weights = W.flatten()
    ax.hist(flattened_weights, bins=50, color=color, alpha=0.7, label=label)


def plot_detailed_currents(num_neurons_to_plot, rec_current_mon, input_current_mon, stim_info, T, n_rec_e):
    """
    Plots a detailed breakdown of currents for a selection of random neurons.

    Parameters:
    - num_neurons_to_plot (int): How many random neurons to display.
    - rec_current_mon (StateMonitor): Monitor for recurrent synaptic current (I_syn_rec).
    - input_current_mon (StateMonitor): Monitor for external input current (I_input).
    - stim_info (list): List of tuples containing (stim_id, start_time, end_time).
    - T (Quantity): Total simulation time.
    - n_rec_e (int): Number of excitatory neurons.
    """
    if rec_current_mon is None or input_current_mon is None:
        print("Cannot plot detailed currents, required monitors are missing.")
        return None

    # Select random neurons from the monitored population
    monitored_indices = rec_current_mon.record
    if num_neurons_to_plot > len(monitored_indices):
        print(f"Warning: Requested to plot {num_neurons_to_plot} neurons, but only {len(monitored_indices)} were monitored. Plotting all monitored neurons.")
        num_neurons_to_plot = len(monitored_indices)
    
    selected_indices_in_monitor = np.random.choice(len(monitored_indices), num_neurons_to_plot, replace=False)
    
    fig, axs = plt.subplots(3, num_neurons_to_plot, 
                            figsize=(4 * num_neurons_to_plot, 10), 
                            sharex=True, sharey='row')

    fig.suptitle('Detailed Current Breakdown for Representative Neurons', fontsize=16)

    time_ms = rec_current_mon.t / ms

    for i, monitor_idx in enumerate(selected_indices_in_monitor):
        actual_neuron_idx = monitored_indices[monitor_idx]
        neuron_type = 'Excitatory' if actual_neuron_idx < n_rec_e else 'Inhibitory'

        i_input = input_current_mon.I_input[monitor_idx, :] / nA
        i_syn_rec = rec_current_mon.I_syn_rec[monitor_idx, :] / nA
        i_total = i_input + i_syn_rec

        # Plot I_input
        axs[0, i].plot(time_ms, i_input, color='black')
        axs[0, i].set_title(f'Neuron {actual_neuron_idx} ({neuron_type})')
        if i == 0:
            axs[0, i].set_ylabel('I_input (nA)')

        # Plot I_syn_rec
        axs[1, i].plot(time_ms, i_syn_rec, color='blue')
        if i == 0:
            axs[1, i].set_ylabel('I_syn_rec (nA)')

        # Plot Total Current
        axs[2, i].plot(time_ms, i_total, color='purple')
        axs[2, i].set_xlabel('Time (ms)')
        if i == 0:
            axs[2, i].set_ylabel('Total Current (nA)')

    # Add stimulus background shading and annotations to all subplots
    for i in range(num_neurons_to_plot):
        for row in range(3):
            ax = axs[row, i]
            ax.set_xlim(0, T/ms)
            ax.grid(True, alpha=0.3)
            # Use axvspan for shaded regions
            for stim_id, start, end in stim_info:
                ax.axvspan(start/ms, end/ms, facecolor=f'C{int(stim_id)}', alpha=0.2)
                # Add text annotation for stimulus ID
                ax.text((start/ms + end/ms) / 2, ax.get_ylim()[1], f'{int(stim_id)}',
                        ha='center', va='top', fontsize=8, color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_interleaved_rate_matrix(rate_matrix_before, rate_matrix_after, neuron_order=None, title="Interleaved Firing Rate Matrix"):
    """
    Plots two firing rate matrices interleaved by column for comparison.
    """
    if rate_matrix_before.shape != rate_matrix_after.shape:
        raise ValueError("Both rate matrices must have the same shape.")

    num_neurons, num_stims = rate_matrix_before.shape

    # Apply neuron sorting if provided
    if neuron_order is not None:
        rate_matrix_before = rate_matrix_before[neuron_order]
        rate_matrix_after = rate_matrix_after[neuron_order]

    # Interleave the columns of the two matrices
    interleaved_matrix = np.empty((num_neurons, num_stims * 2), dtype=rate_matrix_before.dtype)
    interleaved_matrix[:, 0::2] = rate_matrix_before
    interleaved_matrix[:, 1::2] = rate_matrix_after

    fig, ax = plt.subplots(figsize=(max(12, num_stims * 0.4), 10))
    
    # Plot the heatmap
    vmax = np.max(interleaved_matrix)
    im = ax.imshow(interleaved_matrix, aspect='auto', cmap='viridis', interpolation='none', vmin=0, vmax=vmax)
    
    # Add vertical lines
    for i in range(num_stims):
        # White line between epochs of the same stimulus
        if i < num_stims:
            ax.axvline(x=i * 2 + 0.5, color='white', linestyle='--', linewidth=0.7)
        # Black line between different stimuli
        if i < num_stims - 1:
            ax.axvline(x=i * 2 + 1.5, color='black', linestyle='-', linewidth=1)

    # Configure ticks and labels
    # For y-axis (neurons), show a reasonable number of ticks (max ~20)
    if num_neurons > 20:
        tick_step = max(1, num_neurons // 20)
        ytick_positions = np.arange(0, num_neurons, tick_step)
        if neuron_order is not None:
            neuron_order_arr = np.array(neuron_order)
            ytick_labels = neuron_order_arr[ytick_positions]
        else:
            ytick_labels = ytick_positions
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels)
    else:
        ax.set_yticks(np.arange(num_neurons))
        ax.set_yticklabels(neuron_order if neuron_order is not None else np.arange(num_neurons))
    ax.set_ylabel('Neuron Index')
    
    ax.set_xticks(np.arange(0, num_stims * 2, 2) + 0.5)
    ax.set_xticklabels(np.arange(num_stims))
    ax.set_xlabel('Stimulus ID')
    
    ax.set_title(title)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Firing Rate (Hz)')

    plt.tight_layout()
    return fig


def plot_connection_metric_regression_grid(successful_results,
                                           metrics_config,
                                           save_path,
                                           connection_fields=None,
                                           figure_title="Connection Strength vs Metric Regressions",
                                           log_x_axis=False):
    """
    Create a scatter/regression grid showing each connection scale against multiple metrics.

    Args:
        successful_results (list[dict]): Simulation outputs that include connection scales and metrics.
        metrics_config (list[dict]): Each dict must include
            - name (str): label used for subplot titles/colorbars
            - value_fn (callable): function that maps a result dict -> metric value
            - color (str, optional): scatter color
        save_path (str or Path): Where to save the resulting figure.
        connection_fields (list[tuple], optional): (label, key) pairs for connection scales.
        figure_title (str): Overall title for the figure.
    """
    if connection_fields is None:
        connection_fields = [
            ("E→E", "e_to_e_scale"),
            ("E→I", "e_to_i_scale"),
            ("I→E", "i_to_e_scale"),
            ("I→I", "i_to_i_scale"),
        ]

    resolved_fields = []
    for field in connection_fields:
        if isinstance(field, dict):
            label = field["label"]
            value_fn = field["value_fn"]
        else:
            label, key = field
            value_fn = lambda result, k=key: result.get(k, np.nan)
        resolved_fields.append((label, value_fn))

    n_rows = len(resolved_fields)
    n_cols = len(metrics_config)

    if n_rows == 0 or n_cols == 0:
        raise ValueError("connection_fields and metrics_config must both be non-empty.")

    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=(4.0 * n_cols, 3.0 * n_rows),
                             squeeze=False,
                             sharex=False,
                             sharey=False)

    for row, (conn_label, conn_fn) in enumerate(resolved_fields):
        x_values = np.asarray([conn_fn(result) for result in successful_results], dtype=float)

        for col, metric in enumerate(metrics_config):
            ax = axes[row, col]
            metric_values = np.asarray([metric["value_fn"](result) for result in successful_results], dtype=float)
            metric_log_scale = bool(metric.get("log_y", False))

            valid_mask = np.isfinite(x_values) & np.isfinite(metric_values)
            if log_x_axis:
                valid_mask &= x_values > 0
            if metric_log_scale:
                valid_mask &= metric_values > 0

            if not np.any(valid_mask):
                ax.set_title(f"No data for {conn_label} vs {metric['name']}")
                ax.axis("off")
                continue

            x_valid = x_values[valid_mask]
            y_valid = metric_values[valid_mask]

            scatter_color = metric.get("color", None)
            ax.scatter(x_valid, y_valid, color=scatter_color, alpha=0.6, edgecolor="none", s=18)

            if x_valid.size >= 2 and np.max(x_valid) - np.min(x_valid) > 1e-12:
                x_for_fit = np.log10(x_valid) if log_x_axis else x_valid
                y_for_fit = np.log10(y_valid) if metric_log_scale else y_valid

                slope, intercept = np.polyfit(x_for_fit, y_for_fit, 1)

                if log_x_axis:
                    x_line_vals = np.logspace(np.log10(x_valid.min()), np.log10(x_valid.max()), 200)
                    x_line_for_fit = np.log10(x_line_vals)
                else:
                    x_line_vals = np.linspace(x_valid.min(), x_valid.max(), 200)
                    x_line_for_fit = x_line_vals

                y_line_for_fit = slope * x_line_for_fit + intercept
                if metric_log_scale:
                    y_line_vals = 10 ** y_line_for_fit
                else:
                    y_line_vals = y_line_for_fit

                ax.plot(x_line_vals, y_line_vals, color="black", linewidth=1.2)

                if y_for_fit.size >= 2:
                    r_matrix = np.corrcoef(x_for_fit, y_for_fit)
                    r_val = r_matrix[0, 1] if r_matrix.shape == (2, 2) else np.nan
                    if np.isfinite(r_val):
                        ax.text(0.05,
                                0.9,
                                f"R²={r_val**2:.2f}",
                                transform=ax.transAxes,
                                fontsize=9,
                                fontweight="bold")

            ax.set_xlabel(conn_label)
            ax.set_ylabel(metric["name"])

            ax.grid(True, linestyle="--", alpha=0.3)
            if log_x_axis:
                ax.set_xscale("log")
            if metric_log_scale:
                ax.set_yscale("log")

    fig.suptitle(figure_title, fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_multiple_regression_coefficients(successful_results,
                                          metrics_config,
                                          save_path,
                                          figure_title="Multiple Regression: Connection Factors vs Metrics"):
    """
    Perform multiple regression for each metric using all connection scales and ratios,
    then plot R² and coefficients as bar plots.
    
    Args:
        successful_results (list[dict]): Simulation outputs
        metrics_config (list[dict]): Metric configurations
        save_path (str or Path): Where to save the figure
        figure_title (str): Overall title
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # Build feature matrix: 4 raw scales + 6 ratios
    feature_names = [
        "E→E", "E→I", "I→E", "I→I",
        "E→E/E→I", "E→E/I→E", "E→E/I→I",
        "E→I/I→E", "E→I/I→I", "I→E/I→I"
    ]
    
    n_features = len(feature_names)
    n_metrics = len(metrics_config)
    
    # Extract features
    X = np.zeros((len(successful_results), n_features))
    for i, result in enumerate(successful_results):
        # Raw scales (all should be positive, but take abs to be safe)
        X[i, 0] = np.abs(result.get("e_to_e_scale", np.nan))
        X[i, 1] = np.abs(result.get("e_to_i_scale", np.nan))
        X[i, 2] = np.abs(result.get("i_to_e_scale", np.nan))
        X[i, 3] = np.abs(result.get("i_to_i_scale", np.nan))
        # Ratios (use absolute values for scales, but preserve sign if meaningful)
        # For inhibitory connections, the actual weights are negative, but scales are positive
        # So ratios should be positive when using scales
        X[i, 4] = X[i, 0] / X[i, 1] if X[i, 1] != 0 else np.nan  # E→E/E→I
        X[i, 5] = X[i, 0] / X[i, 2] if X[i, 2] != 0 else np.nan  # E→E/I→E
        X[i, 6] = X[i, 0] / X[i, 3] if X[i, 3] != 0 else np.nan  # E→E/I→I
        X[i, 7] = X[i, 1] / X[i, 2] if X[i, 2] != 0 else np.nan  # E→I/I→E
        X[i, 8] = X[i, 1] / X[i, 3] if X[i, 3] != 0 else np.nan  # E→I/I→I
        X[i, 9] = X[i, 2] / X[i, 3] if X[i, 3] != 0 else np.nan  # I→E/I→I
    
    # Use log transform (all values should be positive after taking abs of scales)
    # Handle any edge cases with signed log transform just to be safe
    X_log = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            val = X[i, j]
            if np.isfinite(val) and val > 0:
                X_log[i, j] = np.log10(val)
            elif np.isfinite(val) and val < 0:
                # Handle negative values with signed log (shouldn't happen with scales, but be safe)
                X_log[i, j] = np.sign(val) * np.log10(np.abs(val))
            else:
                X_log[i, j] = np.nan
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), squeeze=False)
    fig.suptitle(figure_title, fontsize=16, fontweight="bold", y=0.995)
    
    for metric_idx, metric in enumerate(metrics_config):
        # Extract metric values
        y = np.asarray([metric["value_fn"](result) for result in successful_results], dtype=float)
        
        # Apply log transform if needed
        metric_log_scale = bool(metric.get("log_y", False))
        if metric_log_scale:
            valid_mask = np.isfinite(y) & (y > 0)
            y_for_fit = np.log10(y[valid_mask])
        else:
            valid_mask = np.isfinite(y)
            y_for_fit = y[valid_mask]
        
        # Filter features to valid samples
        X_valid = X_log[valid_mask, :]
        X_valid = X_valid[:, np.all(np.isfinite(X_valid), axis=0)]  # Remove columns with any NaN
        feature_names_valid = [feature_names[i] for i in range(n_features) 
                              if np.all(np.isfinite(X_log[valid_mask, i]))]
        
        if X_valid.shape[0] < X_valid.shape[1] + 1:
            # Not enough samples for regression
            axes[metric_idx, 0].text(0.5, 0.5, "Insufficient data", 
                                     ha="center", va="center", transform=axes[metric_idx, 0].transAxes)
            continue
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # Fit multiple regression
        reg = LinearRegression()
        reg.fit(X_scaled, y_for_fit)
        
        # Calculate R²
        y_pred = reg.predict(X_scaled)
        ss_res = np.sum((y_for_fit - y_pred) ** 2)
        ss_tot = np.sum((y_for_fit - np.mean(y_for_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Plot coefficients
        ax_coef = axes[metric_idx, 0]
        coef_values = reg.coef_
        colors_coef = ["red" if c < 0 else "blue" for c in coef_values]
        bars = ax_coef.barh(range(len(feature_names_valid)), coef_values, color=colors_coef, alpha=0.7)
        ax_coef.set_yticks(range(len(feature_names_valid)))
        ax_coef.set_yticklabels(feature_names_valid, fontsize=9)
        ax_coef.set_xlabel("Coefficient", fontsize=11, fontweight="bold")
        ax_coef.set_title(f"{metric['name']} (R² = {r_squared:.3f})", fontsize=11, fontweight="bold")
        ax_coef.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax_coef.grid(True, axis="x", linestyle="--", alpha=0.3)
        
        # Add coefficient values as text
        for i, (bar, val) in enumerate(zip(bars, coef_values)):
            ax_coef.text(val, i, f" {val:.3f}", va="center", 
                        ha="left" if val >= 0 else "right", fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    # Simple test for just the plot_stimuli_imshow function
    print("Testing plot_stimuli_imshow function...")
    
    # Create sample data - a simple 50x20 matrix
    n_neurons = 50
    n_stimuli = 20
    stim_pattern_matrix = np.random.rand(n_neurons, n_stimuli)
    
    # Test the function
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plot_stimuli_imshow(stim_pattern_matrix, None, ax, title="Test Stimulus Patterns")
