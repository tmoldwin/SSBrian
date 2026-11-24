import numpy as np
import math
import matplotlib.pyplot as plt
import sparsity_and_selectivity as ss
import matplotlib.lines as mlines

import numpy as np
import math
import matplotlib.pyplot as plt



def selectivity_stem_plot(rate_matrix, sli_arr, title="Selectivity Stem Plot"):
    """
    Plots a random subset of 9 neurons from the rate matrix as sorted stem plots
    in a 3x3 layout.

    Each of the 9 neurons has an associated SLI value, which is displayed inside
    the subplot (top-left corner). The color of the stems is determined by the SLI:
    - High SLI (≥0.5): red
    - Low SLI (<0.5): blue
    """
    # Randomly choose 9 neurons
    n_neurons = rate_matrix.shape[0]

    # Ensure there are at least 9 neurons to choose from
    if n_neurons < 9:
        print("Not enough neurons to plot (requires at least 9).")
        return

    selected_indices = np.random.choice(n_neurons, 9, replace=False)
    print("selected_indices:", selected_indices)

    selected_rate_matrix = rate_matrix[selected_indices, :]
    selected_sli_arr = np.array(sli_arr)[selected_indices]

    # Create the figure and subplots
    fig, axes = plt.subplots(3, 3, figsize=(6, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, idx in enumerate(selected_indices):
        row = selected_rate_matrix[i]
        sli_val = selected_sli_arr[i]

        # Sort the stimuli
        sorted_stims = np.sort(row)[::-1]

        # Determine color based on SLI
        color = 'red' if sli_val >= 0.5 else 'blue'

        # Plot stems
        markerline, stemlines, baseline = axes[i].stem(sorted_stims, linefmt=color, markerfmt='.', basefmt=' ')

        # Customize appearance
        plt.setp(markerline, 'markerfacecolor', color, 'markersize', 4)
        plt.setp(stemlines, 'color', color)
        plt.setp(baseline, 'color', 'black', 'linewidth', 1)

        # Remove extra spines if desired (for a cleaner look)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

        # Set SLI text at top-left inside the subplot
        # Using axes fraction coordinates: x=0.05 means slightly from left, y=0.95 near the top
        axes[i].text(0.95, 0.95, f'SLI = {sli_val:.2f}', transform=axes[i].transAxes,
                     ha='right', va='top', fontsize=10, color='black')

        # Adjust y-limit to start at 0
        axes[i].set_ylim(bottom=0)

    # Set shared labels for the figure
    # We can use fig.text to set one label for all subplots:
    fig.text(0.5, 0.04, 'Stimulus', ha='center', va='center')
    fig.text(0.04, 0.5, 'Rate (Hz)', ha='center', va='center', rotation='vertical')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0.1, 0.1, 0.95, 0.95])
    fig.suptitle(title)
    return fig


def plot_selectivity_comparison(avg_rate_matrix, first_rate_matrix=None, last_rate_matrix=None, std_matrix=None, 
                                all_epoch_matrices=None, title="Selectivity Comparison (Average vs All Epochs)"):
    """
    Plots a comparison of selectivity for a random subset of neurons, showing the state
    in the average and all individual epochs.

    - A figure with (1 + n_epochs) rows and N columns is created for N randomly selected neurons.
    - Top row: Circles with error bars for averaged rates (std across epochs).
    - Subsequent rows: One row per epoch showing firing rates.
    - Stimuli are ordered on the x-axis based on the average firing rates (descending).
      This ordering is maintained for all rows.
    - Points are colored based on the objective stimulus ID.
    
    Parameters
    ----------
    avg_rate_matrix : ndarray
        Average rate matrix (neurons x stimuli)
    first_rate_matrix : ndarray, optional
        First epoch rate matrix (used only if all_epoch_matrices not provided)
    last_rate_matrix : ndarray, optional
        Last epoch rate matrix (used only if all_epoch_matrices not provided)
    std_matrix : ndarray, optional
        Standard deviation matrix across epochs (neurons x stimuli)
    all_epoch_matrices : list of ndarray, optional
        List of all epoch rate matrices. If provided, shows all epochs.
    title : str
        Figure title
    """
    n_neurons = avg_rate_matrix.shape[0]
    num_to_plot = min(9, n_neurons)

    if num_to_plot == 0:
        print("Not enough neurons to plot.")
        return None

    selected_indices = np.random.choice(n_neurons, num_to_plot, replace=False)

    # Determine number of rows: 1 for average + all epochs
    if all_epoch_matrices is not None and len(all_epoch_matrices) > 0:
        n_epochs = len(all_epoch_matrices)
        epoch_matrices = all_epoch_matrices
    else:
        # Fallback to just first and last if all_epoch_matrices not provided
        if first_rate_matrix is None or last_rate_matrix is None:
            raise ValueError("Either all_epoch_matrices must be provided, or both first_rate_matrix and last_rate_matrix")
        n_epochs = 2
        epoch_matrices = [first_rate_matrix, last_rate_matrix]
    
    n_rows = 1 + n_epochs  # Average row + all epoch rows

    # Get data for selected neurons
    selected_rate_avg = avg_rate_matrix[selected_indices, :]
    selected_epoch_rates = [mat[selected_indices, :] for mat in epoch_matrices]
    
    # Get standard deviation if provided
    if std_matrix is not None:
        selected_std = std_matrix[selected_indices, :]
    else:
        selected_std = None
    
    # Calculate SLI for the selected neurons
    _, sli_avg_all = ss.create_vecs(avg_rate_matrix)
    selected_sli_avg = np.array(sli_avg_all)[selected_indices]
    
    # Calculate SLI for each epoch
    selected_sli_epochs = []
    for epoch_mat in epoch_matrices:
        _, sli_epoch_all = ss.create_vecs(epoch_mat)
        selected_sli_epochs.append(np.array(sli_epoch_all)[selected_indices])

    # Create colors for each stimulus using a cyclical colormap for distinctness
    n_stims = avg_rate_matrix.shape[1]
    colors = plt.cm.hsv(np.linspace(0, 1, n_stims))

    # Create grid layout: (1 + n_epochs) rows x num_to_plot columns
    fig, axs = plt.subplots(n_rows, num_to_plot, figsize=(2 * num_to_plot, 2.5 * n_rows))
    if num_to_plot == 1:
        axs = axs.reshape(n_rows, 1)
    if n_rows == 1:
        axs = axs.reshape(1, -1)
    
    fig.suptitle(title, fontsize=16)

    for i, neuron_idx in enumerate(selected_indices):
        # Sort by average response strength to determine the order
        avg_rates = selected_rate_avg[i, :]
        stim_order = np.argsort(avg_rates)[::-1]
        x_positions = np.arange(len(stim_order))
        colors_sorted = [colors[stim_idx] for stim_idx in stim_order]
        
        # Row 0: Average with error bars
        ax_avg = axs[0, i]
        avg_rates_sorted = avg_rates[stim_order]
        
        if selected_std is not None:
            std_sorted = selected_std[i, :][stim_order]
            ax_avg.errorbar(x_positions, avg_rates_sorted, yerr=std_sorted, 
                          fmt='none', capsize=2, capthick=1,
                          ecolor='gray', alpha=0.5, elinewidth=1)
        ax_avg.scatter(x_positions, avg_rates_sorted, c=colors_sorted,
                      s=15, alpha=0.8, edgecolors='black', linewidth=0.3, zorder=3)
        ax_avg.set_title(f'Neuron {neuron_idx}\nSLI: {selected_sli_avg[i]:.2f}')
        if i == 0:
            ax_avg.set_ylabel('Average (Hz)')
        ax_avg.set_xticks([])
        ax_avg.set_ylim(bottom=0)
        ax_avg.grid(True, alpha=0.3, axis='y')
        
        # Rows 1 to n_epochs: Individual epochs
        for epoch_idx, epoch_rates in enumerate(selected_epoch_rates):
            ax_epoch = axs[epoch_idx + 1, i]
            epoch_rates_sorted = epoch_rates[i, :][stim_order]
            ax_epoch.scatter(x_positions, epoch_rates_sorted, c=colors_sorted,
                           s=15, alpha=0.8, edgecolors='black', linewidth=0.3)
            ax_epoch.set_title(f'Epoch {epoch_idx + 1}, SLI: {selected_sli_epochs[epoch_idx][i]:.2f}')
            if i == 0:
                ax_epoch.set_ylabel(f'Epoch {epoch_idx + 1} (Hz)')
            ax_epoch.set_xticks([])
            ax_epoch.set_ylim(bottom=0)
            ax_epoch.grid(True, alpha=0.3, axis='y')

    # Configure x-axis to show only 0 and max on the bottom row
    n_stims = len(stim_order)
    for i in range(num_to_plot):
        bottom_ax = axs[-1, i]
        bottom_ax.set_xticks([0, n_stims - 1])
        bottom_ax.set_xticklabels(['0', str(n_stims - 1)], fontsize=10)
        bottom_ax.set_xlabel('Stimulus (sorted)', fontsize=9)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.92])
    return fig

def plot_stimulus_response_comparison(before_rate_matrix, after_rate_matrix, n_rec_e, num_to_plot=4, title="Stimulus Response Comparison"):
    """
    Plots the population response to a selection of random stimuli using colored stem plots,
    comparing the first and last epochs. Stems are colored by each neuron's ID.
    """
    if before_rate_matrix.shape != after_rate_matrix.shape:
        raise ValueError("Both rate matrices must have the same shape.")

    num_neurons, num_stims = before_rate_matrix.shape
    
    if num_stims < num_to_plot:
        num_to_plot = num_stims
        
    selected_stim_indices = np.random.choice(num_stims, num_to_plot, replace=False)
    
    fig, axs = plt.subplots(2, num_to_plot, figsize=(4 * num_to_plot, 8), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=16)

    # --- Determine neuron properties for coloring/marking ---
    # 1. Neuron types (E/I)
    neuron_markers = np.array(['o'] * n_rec_e + ['^'] * (num_neurons - n_rec_e))
    
    # 2. Create a colormap for neurons
    neuron_colors = plt.cm.hsv(np.linspace(0, 1, num_neurons))

    for i, stim_idx in enumerate(selected_stim_indices):
        ax_top = axs[0, i]
        ax_bottom = axs[1, i]

        # Get response of all neurons to this specific stimulus
        before_response = before_rate_matrix[:, stim_idx]
        after_response = after_rate_matrix[:, stim_idx]
        
        # Calculate Sparsity for this stimulus response vector
        spi_before, _ = ss.create_vecs(before_response.reshape(-1, 1))
        spi_after, _ = ss.create_vecs(after_response.reshape(-1, 1))

        # Sort neurons by response strength for plotting order
        neuron_order = np.argsort(before_response)[::-1]
        
        # --- Plot stems for 'before' state ---
        for j, neuron_idx in enumerate(neuron_order):
            rate = before_response[neuron_idx]
            color = neuron_colors[neuron_idx] # Color by neuron ID
            marker = neuron_markers[neuron_idx]
            
            markerline, stemlines, baseline = ax_top.stem([j], [rate], linefmt='grey', markerfmt=marker, basefmt=' ')
            plt.setp(stemlines, 'color', color, 'linewidth', 0.5)
            plt.setp(markerline, 'markerfacecolor', 'none', 'markeredgecolor', color, 'markersize', 4)

        ax_top.set_title(f'Stimulus {stim_idx}\nSPI: {spi_before[0]:.2f}')
        if i == 0:
            ax_top.set_ylabel('First Epoch (Hz)')

        # --- Plot stems for 'after' state using the same neuron order ---
        for j, neuron_idx in enumerate(neuron_order):
            rate = after_response[neuron_idx]
            color = neuron_colors[neuron_idx] # Color is always based on neuron ID
            marker = neuron_markers[neuron_idx]
            
            markerline, stemlines, baseline = ax_bottom.stem([j], [rate], linefmt='grey', markerfmt=marker, basefmt=' ')
            plt.setp(stemlines, 'color', color, 'linewidth', 0.5)
            plt.setp(markerline, 'markerfacecolor', 'none', 'markeredgecolor', color, 'markersize', 4)
            
        ax_bottom.set_title(f'SPI: {spi_after[0]:.2f}')
        ax_bottom.set_xlabel('Sorted Neuron Index')
        if i == 0:
            ax_bottom.set_ylabel('Last Epoch (Hz)')
        
        ax_top.set_ylim(bottom=0)
        ax_bottom.set_ylim(bottom=0)
        ax_top.grid(True, alpha=0.3)
        ax_bottom.grid(True, alpha=0.3)

    # Add legends for markers
    e_patch = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', label='Excitatory')
    i_patch = mlines.Line2D([], [], color='gray', marker='^', linestyle='None', label='Inhibitory')
    fig.legend(handles=[e_patch, i_patch], loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


if __name__ == "__main__":
    # Example usage
    # Create a mock rate_matrix (e.g. 30 neurons, 50 stimuli)
    rate_matrix = np.random.rand(30, 50) * 50
    after_rate_matrix = rate_matrix * (1 + np.random.rand(30, 50) * 0.5)

    # Create a mock sli_array for 30 neurons
    sli_array = np.random.rand(30)

    fig = selectivity_stem_plot(rate_matrix, sli_array)
    plt.show()

    fig_comp = plot_selectivity_comparison(rate_matrix, after_rate_matrix)
    if fig_comp:
        plt.show()
