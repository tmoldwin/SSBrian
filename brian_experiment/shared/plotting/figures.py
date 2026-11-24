import numpy as np
import matplotlib.pyplot as plt
from brian_experiment.shared.plotting import plots as pf
import os


#1.	The Network
#A. Network schematic – missing
#B. Poisson WM – we have
#C. Recurrent WM – we have
#D. Histogram poisson – we have
#E. Histogram recurrent – we have

def plotting_dashboard_figure_1(weight_poisson, weight_recurrent, save_folder=None):
    fig, axs = plt.subplot_mosaic([['B', 'C'], ['E', 'F']], figsize = (12, 8))
    #poiss matrix
    pf.plot_weight_matrix(axs['B'], weight_poisson, weight_poisson)
    #rec matrix
    pf.plot_weight_matrix(axs['C'], weight_recurrent, weight_poisson)
    #histogram poisson
    pf.plot_weight_histogram(axs['E'], weight_poisson, bins=30, type='Poisson')
    #histogram recurrent
    pf.plot_weight_histogram(axs['F'], weight_recurrent, bins=30, type='Recurrent')
    plt.tight_layout()
    if save_folder:
        plt.savefig(os.path.join(save_folder, 'network_architecture_and_weights.png'))

def plotting_dashboard_figure_2(h_net, save_folder=None):
    fig, axs = plt.subplot_mosaic([['A', 'B', 'C'], ['D', 'E', 'F']], figsize=(18, 10))
    pf.plot_external_stimuli(axs['A'], h_net.external_input_poisson)
    pf.plot_raster(axs['B'], h_net.poisson_spikes, h_net.N_poisson_E, h_net.N_poisson_I, h_net.T, stim_info=h_net.stim_info, title='Poisson Neurons Spike Raster')
    pf.plot_rate_correlation(axs['C'], h_net.smoothed_rates_poiss, h_net.N_poisson_E, h_net.N_poisson_I,
                               h_net.smoothed_rates_rec)
    pf.plot_membrane_potential(axs['D'], h_net.rec_voltage, h_net.T)
    pf.plot_raster(axs['E'], h_net.rec_spikes, h_net.N_rec_E, h_net.N_rec_I, h_net.T, neuron_order=h_net.neuron_order,
                   stim_info=h_net.stim_info, title='Recurrent Neurons Spike Raster')
    pf.plot_smoothed_rates(h_net.smoothed_rates_rec, h_net.smoothed_time_points, neuron_order=h_net.neuron_order,
                           ax=axs['F'], title='Smoothed Firing Rates')
    plt.tight_layout()
    if save_folder:
        plt.savefig(os.path.join(save_folder, 'network_dynamics_and_spiking.png'))


def plotting_dashboard_figure_3(network, individual_sparsity, individual_selectivity, cumulative_sparsity, cumulative_selectivity, save_folder=None):
    """
    Creates the third figure of the dashboard, showing the evolution of sparsity
    and selectivity, along with their distributions before and after.
    
    Layout: 2 rows (Sparsity, Selectivity) x 4 columns (Individual Evolution, Cumulative Evolution, First Epoch Dist, Last Epoch Dist)
    
    Data structure expected:
    - individual_sparsity: (stimuli, epochs) - sparsity per stimulus per epoch
    - individual_selectivity: (neurons, epochs) - selectivity per neuron per epoch  
    - cumulative_sparsity: (stimuli, epochs) - cumulative sparsity per stimulus
    - cumulative_selectivity: (neurons, epochs) - cumulative selectivity per neuron
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 1.5, 1, 1], height_ratios=[1, 1])

    epochs = np.arange(individual_sparsity.shape[1])  # Number of epochs
    
    # --- Row 1: Sparsity ---
    ax_sparsity_ind = fig.add_subplot(gs[0, 0])  # Individual evolution
    ax_sparsity_cum = fig.add_subplot(gs[0, 1])  # Cumulative evolution
    ax_sparsity_before = fig.add_subplot(gs[0, 2])  # First epoch distribution
    ax_sparsity_after = fig.add_subplot(gs[0, 3])  # Last epoch distribution

    # Individual Sparsity Evolution
    for i in range(min(20, individual_sparsity.shape[0])):  # Show up to 20 stimuli
        ax_sparsity_ind.plot(epochs, individual_sparsity[i, :], 'o-', alpha=0.7, linewidth=1, markersize=3)
    mean_sparsity_ind = np.mean(individual_sparsity, axis=0)
    ax_sparsity_ind.plot(epochs, mean_sparsity_ind, 'o-', label='Mean Sparsity', markersize=6, color='black', linewidth=3)
    ax_sparsity_ind.set_xlabel('Epoch')
    ax_sparsity_ind.set_ylabel('Sparsity')
    ax_sparsity_ind.set_title('Individual Epoch Sparsity Evolution (per Stimulus)')
    ax_sparsity_ind.legend()
    ax_sparsity_ind.grid(True, alpha=0.3)
    # Set x-axis ticks to only show whole epochs
    ax_sparsity_ind.set_xticks(epochs)

    # Cumulative Sparsity Evolution
    for i in range(min(20, cumulative_sparsity.shape[0])):  # Show up to 20 stimuli
        ax_sparsity_cum.plot(epochs, cumulative_sparsity[i, :], 'o-', alpha=0.7, linewidth=1, markersize=3)
    mean_sparsity_cum = np.mean(cumulative_sparsity, axis=0)
    ax_sparsity_cum.plot(epochs, mean_sparsity_cum, 'o-', label='Mean Sparsity', markersize=6, color='black', linewidth=3)
    ax_sparsity_cum.set_xlabel('Cumulative Epochs')
    ax_sparsity_cum.set_ylabel('Sparsity')
    ax_sparsity_cum.set_title('Cumulative Sparsity Evolution (per Stimulus)')
    ax_sparsity_cum.legend()
    ax_sparsity_cum.grid(True, alpha=0.3)
    # Set x-axis ticks to only show whole epochs
    ax_sparsity_cum.set_xticks(epochs)

    # Sparsity Distributions
    bins_sparsity = np.linspace(0, 1, 30)
    ax_sparsity_before.hist(individual_sparsity[:, 0], bins=bins_sparsity, color='blue', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax_sparsity_before.set_title('Sparsity Distribution (First Epoch)')
    ax_sparsity_before.set_xlabel('Sparsity')
    ax_sparsity_before.set_ylabel('Count')
    
    ax_sparsity_after.hist(individual_sparsity[:, -1], bins=bins_sparsity, color='green', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax_sparsity_after.set_title('Sparsity Distribution (Last Epoch)')
    ax_sparsity_after.set_xlabel('Sparsity')
    ax_sparsity_after.set_ylabel('Count')

    # --- Row 2: Selectivity ---
    ax_selectivity_ind = fig.add_subplot(gs[1, 0])  # Individual evolution
    ax_selectivity_cum = fig.add_subplot(gs[1, 1])  # Cumulative evolution
    ax_selectivity_before = fig.add_subplot(gs[1, 2])  # First epoch distribution
    ax_selectivity_after = fig.add_subplot(gs[1, 3])  # Last epoch distribution

    # Individual Selectivity Evolution
    for i in range(min(20, individual_selectivity.shape[0])):  # Show up to 20 neurons
        ax_selectivity_ind.plot(epochs, individual_selectivity[i, :], 'o-', alpha=0.7, linewidth=1, markersize=3)
    mean_selectivity_ind = np.mean(individual_selectivity, axis=0)
    ax_selectivity_ind.plot(epochs, mean_selectivity_ind, 'o-', label='Mean Selectivity', markersize=6, color='black', linewidth=3)
    ax_selectivity_ind.set_xlabel('Epoch')
    ax_selectivity_ind.set_ylabel('Selectivity')
    ax_selectivity_ind.set_title('Individual Epoch Selectivity Evolution (per Neuron)')
    ax_selectivity_ind.legend()
    ax_selectivity_ind.grid(True, alpha=0.3)
    # Set x-axis ticks to only show whole epochs
    ax_selectivity_ind.set_xticks(epochs)

    # Cumulative Selectivity Evolution
    for i in range(min(20, cumulative_selectivity.shape[0])):  # Show up to 20 neurons
        ax_selectivity_cum.plot(epochs, cumulative_selectivity[i, :], 'o-', alpha=0.7, linewidth=1, markersize=3)
    mean_selectivity_cum = np.mean(cumulative_selectivity, axis=0)
    ax_selectivity_cum.plot(epochs, mean_selectivity_cum, 'o-', label='Mean Selectivity', markersize=6, color='black', linewidth=3)
    ax_selectivity_cum.set_xlabel('Cumulative Epochs')
    ax_selectivity_cum.set_ylabel('Selectivity')
    ax_selectivity_cum.set_title('Cumulative Selectivity Evolution (per Neuron)')
    ax_selectivity_cum.legend()
    ax_selectivity_cum.grid(True, alpha=0.3)
    # Set x-axis ticks to only show whole epochs
    ax_selectivity_cum.set_xticks(epochs)

    # Selectivity Distributions - Use full range of selectivity values
    # Get the actual min/max range from all selectivity data
    selectivity_min = min(np.min(individual_selectivity[0]), np.min(individual_selectivity[-1]))
    selectivity_max = max(np.max(individual_selectivity[0]), np.max(individual_selectivity[-1]))
    # Add some padding to the range
    selectivity_range = selectivity_max - selectivity_min
    padding = selectivity_range * 0.05  # 5% padding
    bins_selectivity = np.linspace(max(0, selectivity_min - padding), min(1, selectivity_max + padding), 30)
    
    ax_selectivity_before.hist(individual_selectivity[:, 0], bins=bins_selectivity, color='blue', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax_selectivity_before.set_title('Selectivity Distribution (First Epoch)')
    ax_selectivity_before.set_xlabel('Selectivity')
    ax_selectivity_before.set_ylabel('Count')
    
    ax_selectivity_after.hist(individual_selectivity[:, -1], bins=bins_selectivity, color='purple', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax_selectivity_after.set_title('Selectivity Distribution (Last Epoch)')
    ax_selectivity_after.set_xlabel('Selectivity')
    ax_selectivity_after.set_ylabel('Count')

    plt.tight_layout()
    if save_folder:
        plt.savefig(os.path.join(save_folder, 'sparsity_selectivity_evolution_over_epochs.png'), dpi=300)
    plt.close()


def plotting_dashboard_figure_4(rate_matrix_before, sli_arr_before, rate_matrix_after, sli_arr_after, save_folder=None):
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    pf.plot_selectivity_comparison(ax, rate_matrix_before, sli_arr_before, rate_matrix_after, sli_arr_after)
    plt.tight_layout()
    if save_folder:
        plt.savefig(os.path.join(save_folder, 'selectivity_comparison_before_after.png'))


if __name__ == "__main__":
    weight_poisson = np.random.rand(100, 100)
    weight_recurrent = np.random.rand(100, 100)
    plotting_dashboard_figure_1(weight_poisson, weight_recurrent)