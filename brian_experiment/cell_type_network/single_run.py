import argparse
import os
import shutil
import random as py_random
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

import numpy as np
from brian2 import *
from datetime import datetime

from brian_experiment.cell_type_network import network as ctn
from brian_experiment.cell_type_network.configs import config_manager as cfg_manager
from brian_experiment.cell_type_network.visualize_network import draw_network_diagram
from brian_experiment.shared.core import stimuli as stim
from brian_experiment.shared.core import spike_analysis_utils as sau
from brian_experiment.shared.core import spike_analysis as sa
from brian_experiment.shared.analysis import sparsity_and_selectivity as sp
from brian_experiment.shared.plotting.generate_plots import generate_all_plots

# Ensure we keep access to Python's random module (Brian2 overrides the name)
random = py_random

prefs.codegen.target = "numpy"


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run a cell type network simulation using JSON configs."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Name or path to the run config JSON (falls back to default).",
    )
    return parser.parse_args()


def _compute_weight_scales(connection_configs, population_configs, input_layer_config,
                           base_weight_scale, weight_scale_method, 
                           inhibitory_weight_scale, input_amplitude_scale):
    """
    Compute and store weight scales for all connections based on config parameters.
    This matches the feedforward network weight scaling logic.
    
    Modifies connection_configs in place by adding '_computed_scale' or updating weight_dist values.
    """
    print("\nComputing weight scaling for all connections...")
    
    for conn in connection_configs:
        source_name = conn['source']
        target_name = conn['target']
        
        # Find source population size
        source_pop = next((p for p in population_configs if p['name'] == source_name), None)
        if not source_pop and source_name == input_layer_config['name']:
            source_pop = input_layer_config
            
        if source_pop:
            n_source = source_pop['n']
            p = conn.get('p', 1.0)
            k_expected = p * n_source
            
            # Calculate base scale factor
            if weight_scale_method == 'sqrt_k':
                scale_factor = base_weight_scale / np.sqrt(k_expected)
            elif weight_scale_method == 'k':
                scale_factor = base_weight_scale / k_expected
            else:
                scale_factor = 1.0
                
            # Apply additional scaling
            # Input scaling
            if source_name == input_layer_config['name']:
                scale_factor *= input_amplitude_scale
            
            # Inhibitory scaling (if source is I) - matches feedforward network
            source_type = source_pop.get('type', 'E')  # Default E for input
            
            if source_type == 'I':
                scale_factor *= inhibitory_weight_scale
            
            # Store computed scale for lognormal distributions (applied multiplicatively after generation)
            # For fixed distributions, modify the value directly
            weight_dist = conn.get('weight_dist', {'dist': 'fixed', 'val': 1.0})
            
            if weight_dist['dist'] == 'fixed':
                original_val = weight_dist.get('val', 1.0)
                weight_dist['val'] = original_val * scale_factor
                print(f"   {source_name}->{target_name}: Fixed weight {original_val} -> {weight_dist['val']:.6f} (K={k_expected:.1f})")
            elif weight_dist['dist'] == 'lognormal':
                # Store scale to apply multiplicatively after generation (matches feedforward)
                conn['_computed_scale'] = scale_factor
                print(f"   {source_name}->{target_name}: Lognormal scale={scale_factor:.6f} (K={k_expected:.1f})")
                
            conn['weight_dist'] = weight_dist
        else:
            print(f"   WARNING: Could not find source population '{source_name}' for connection to '{target_name}'")
    
    return connection_configs

def save_run_parameters(config_dict, run_type, fixed_params, folder_path):
    """Save run parameters to a text file in the timestamped folder"""
    param_file = os.path.join(folder_path, "run_parameters.txt")
    
    with open(param_file, 'w') as f:
        f.write(f"Cell Type Network Simulation Run Parameters\n")
        f.write(f"{'='*45}\n")
        f.write(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration Type: {run_type}\n\n")
        
        f.write("Configuration Parameters:\n")
        f.write(f"{'='*25}\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nFixed Parameters:\n")
        f.write(f"{'='*17}\n")
        for key, value in fixed_params.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    args = parse_cli_args()
    loaded_config = cfg_manager.load_run_config(args.config)
    
    # Define a single random seed for reproducibility
    simulation_seed = 42
    np.random.seed(simulation_seed)
    random.seed(simulation_seed)
    
    # Extract parameters from config (NO DEFAULTS - all must come from config)
    config_params = loaded_config.params
    num_stimuli = config_params['num_stimuli']
    num_epochs = config_params['num_epochs']
    stim_duration = config_params['stim_duration']
    break_time = config_params['break_time']
    start_time = config_params['start_time']
    stimulus_distribution = config_params['stimulus_distribution']
    stimulus_alpha = config_params['stimulus_alpha']
    scale_factor = config_params['scale_factor']
    
    # Scaling parameters (from config only)
    base_weight_scale = config_params['base_weight_scale']
    weight_scale_method = config_params['weight_scale_method']
    inhibitory_weight_scale = config_params['inhibitory_weight_scale']
    input_amplitude_scale = config_params['input_amplitude_scale']
    
    # Simulation parameters (from config only)
    simulation_params = config_params['simulation_params']
    
    # Network structure from config
    input_layer_config = config_params['input_layer']
    population_configs = config_params['populations']
    connection_configs = config_params['connections']
    motif_constraints = config_params.get('motif_constraints', None)  # Optional motif constraints
    
    # Create folder for this run
    run_folder_path, _, _ = cfg_manager.prepare_single_run_folders(loaded_config.name)
    run_folder = str(run_folder_path)
    
    print(f"Creating directories for config '{loaded_config.name}': {run_folder}")
    
    # Save run parameters
    fixed_params = {
        'stim_duration': stim_duration,
        'break_time': break_time,
        'start_time': start_time,
        'stimulus_distribution': stimulus_distribution,
        'stimulus_alpha': stimulus_alpha,
        'num_stimuli': num_stimuli,
        'num_epochs': num_epochs,
        'base_weight_scale': base_weight_scale,
        'weight_scale_method': weight_scale_method,
        'inhibitory_weight_scale': inhibitory_weight_scale,
        'input_amplitude_scale': input_amplitude_scale
    }
    
    config_dict = {
        'input_layer': input_layer_config,
        'populations': population_configs,
        'connections': connection_configs,
        'motif_constraints': motif_constraints
    }
    
    save_run_parameters(config_dict, loaded_config.name, fixed_params, run_folder)
    
    # Copy the config file to the run folder for record-keeping
    if loaded_config.path and loaded_config.path.exists():
        config_copy_path = os.path.join(run_folder, "config.json")
        shutil.copy2(loaded_config.path, config_copy_path)
        print(f"Config file copied to: {config_copy_path}")
    
    print(f"Run parameters saved to: {run_folder}")
    
    # Compute weight scaling for all connections based on config parameters
    # This matches feedforward network weight scaling logic
    connection_configs = _compute_weight_scales(
        connection_configs, 
        population_configs, 
        input_layer_config,
        base_weight_scale,
        weight_scale_method,
        inhibitory_weight_scale,
        input_amplitude_scale
    )
    
    # Calculate total simulation time
    stim_list = np.arange(num_stimuli)
    epoch_time = (stim_duration + break_time) * len(stim_list) * ms
    T = start_time * ms + num_epochs * epoch_time
    
    # Generate stimulus schedule
    stims_by_time, stim_info = stim.create_stim_list(
        T/ms, 
        stim_duration, 
        stim_list, 
        break_time, 
        start_time=start_time
    )
    
    # Generate input patterns (firing rates in Hz)
    # We generate at 1ms resolution to avoid huge arrays
    n_input = input_layer_config['n']
    pattern_dt = 1.0  # ms
    n_time_steps = int(T/ms / pattern_dt)
    
    # Create time array at pattern_dt resolution
    time_array = np.arange(0, T/ms, pattern_dt)[:n_time_steps]
    
    # Map time to stimulus ID
    stims_by_time_coarse = np.zeros(n_time_steps, dtype=int) - 1
    for i, t in enumerate(time_array):
        # Find which stimulus is active at this time
        idx = int(t / (defaultclock.dt/ms))
        if idx < len(stims_by_time):
            stims_by_time_coarse[i] = int(stims_by_time[idx])
    
    # Generate patterns at coarse resolution
    # To match feedforward "probabilistic input", we generate rates that are high
    input_patterns_matrix = stim.generate_stimuli_from_distribution(
        stims_by_time_coarse, 
        n_input, 
        distribution=stimulus_distribution,
        alpha=stimulus_alpha,
        scale_factor=scale_factor
    )
    
    # Initialize network
    print("\nInitializing Cell Type Network...")
    if motif_constraints:
        print(f"   Using motif constraints: {len(motif_constraints)} constraint(s)")
        for mc in motif_constraints:
            print(f"      - {mc.get('name', 'unnamed')}: {mc.get('type')} (enhancement={mc.get('enhancement', 1.0)})")
    
    net = ctn.CellTypeNetwork(
        population_configs=population_configs,
        connection_configs=connection_configs,
        input_layer_config=input_layer_config,
        simulation_params=simulation_params,
        motif_constraints=motif_constraints,
        verbose=True
    )
    
    print(f"\n>> Starting simulation with {num_stimuli} stimuli across {num_epochs} epochs")
    print(f"   Total simulation time: {T/ms:.1f}ms")
    print(f"   Stimulus duration: {stim_duration}ms, Break time: {break_time}ms")
    print(f"   Input layer: {n_input} neurons")
    print(f"   Populations: {len(population_configs)}")
    print("="*60)
    
    # Run simulation
    monitoring_level = 'basic'
    # Use pattern_dt for TimedArray (1ms resolution)
    net.run_simulation(T, input_patterns=input_patterns_matrix, input_dt=pattern_dt*ms, monitoring_level=monitoring_level)
    
    # Save simulation data
    data_folder = os.path.join(run_folder, "Data")
    os.makedirs(data_folder, exist_ok=True)
    data_file = os.path.join(data_folder, "simulation_data.pkl")
    
    simulation_data = {
        'stims_by_time': stims_by_time,
        'stim_info': stim_info,
        'T': T,
        'input_patterns': input_patterns_matrix,
        'population_configs': population_configs,
        'connection_configs': connection_configs,
        'input_layer_config': input_layer_config,
        'config': config_dict
    }
    
    # Extract spike data
    spike_data = {}
    for name in net.populations.keys():
        if f'spikes_{name}' in net.monitors:
            spike_mon = net.monitors[f'spikes_{name}']
            spike_data[name] = {
                'times': np.array(spike_mon.t),
                'indices': np.array(spike_mon.i),
                'count': np.array(spike_mon.count)
            }
    
    simulation_data['spike_data'] = spike_data
    
    with open(data_file, "wb") as f:
        pickle.dump(simulation_data, f)
    print(f"   Simulation data saved to: {data_file}")
    
    # Calculate and print average firing rates
    total_time_s = T / second
    total_spikes = {}
    total_neurons = {}
    
    # Use population order from network for consistent display
    ordered_pop_names = list(net.populations.keys())
    
    for name in ordered_pop_names:
        if f'spikes_{name}' in net.monitors:
            spike_mon = net.monitors[f'spikes_{name}']
            total_spikes[name] = np.sum(spike_mon.count)
            total_neurons[name] = len(net.populations[name])
    
    print(f"\n>> Average Firing Rates:")
    for name in ordered_pop_names:
        if name in total_spikes:
            rate = total_spikes[name] / (total_neurons[name] * total_time_s)
            print(f"   {name}: {rate:.2f} Hz")
    
    # Calculate Synchronization (Average Pairwise Correlation of binned spike counts)
    print(f"\n>> Network Synchronization (Validation):")
    bin_size = 10*ms
    n_bins = int(T/bin_size)
    
    for name in ordered_pop_names:
        if f'spikes_{name}' in net.monitors:
            spike_mon = net.monitors[f'spikes_{name}']
            n_neurons = len(net.populations[name])
            
            # Skip empty populations or input layer if not relevant (keep Layer0 to check input sync)
            if n_neurons < 2: 
                continue
                
            # Create binned spike matrix: (n_neurons, n_bins)
            # Use histogram2d for efficiency
            # i are neuron indices, t are spike times
            # bins for time are defined by edges
            time_edges = np.linspace(0, T/second, n_bins+1)
            
            # We need to bin per neuron. 
            # Efficient way: iterate neurons (if N is small) or use dense histogram
            # For N=100 it's fast enough to do dense matrix
            binned_spikes = np.zeros((n_neurons, n_bins))
            
            # Vectorized binning is tricky with variable spike counts per neuron
            # But we can just use histogram2d on (i, t)
            # y-edges are neuron indices: -0.5 to N-0.5
            neuron_edges = np.arange(n_neurons + 1) - 0.5
            
            H, _, _ = np.histogram2d(spike_mon.i, spike_mon.t/second, bins=[neuron_edges, time_edges])
            binned_spikes = H # shape (n_neurons, n_bins)
            
            # Calculate correlation matrix
            # Remove silent neurons (std=0) to avoid NaNs
            active_mask = np.std(binned_spikes, axis=1) > 0
            if np.sum(active_mask) < 2:
                print(f"   {name}: Not enough active neurons to calculate sync.")
                continue
                
            binned_active = binned_spikes[active_mask]
            corr_matrix = np.corrcoef(binned_active)
            
            # Average off-diagonal elements
            # Get indices of upper triangle, k=1 excludes diagonal
            triu_indices = np.triu_indices(corr_matrix.shape[0], k=1)
            avg_corr = np.mean(corr_matrix[triu_indices])
            
            print(f"   {name}: Sync Index (Avg Corr) = {avg_corr:.4f}")
            
            # Validation Warning
            if avg_corr > 0.2:
                print(f"     WARNING: High synchronization detected in {name}!")
    
    # Calculate Neural Selectivity and Sparsity and build data structure for modular plotting
    print(f"\n>> Neural Selectivity and Sparsity (by Population):")
    
    # Store rate matrices and selectivity/sparsity data per population for modular plotting
    population_rate_matrices = {}
    population_data = {}
    
    # Create spike matrices and rate matrices for each population
    for name in ordered_pop_names:
        if f'spikes_{name}' in net.monitors and name != 'Layer0':
            spike_mon = net.monitors[f'spikes_{name}']
            n_neurons = len(net.populations[name])
            
            # Use per-epoch analysis (same as feedforward) for proper temporal evolution
            (individual_rate_matrices, individual_sparsity, individual_selectivity,
             rate_matrices_arr, sparsity_vec_arr, selectivity_vec_arr) = sa.analyze_both_individual_and_cumulative_epochs(
                spike_mon, T, stims_by_time, num_epochs, n_neurons, 
                dt=defaultclock.dt, start_time=start_time * ms
            )
            
            # Use the last epoch's rate matrix as the "final" rate matrix
            rate_matrix = rate_matrices_arr[-1] if rate_matrices_arr else np.zeros((n_neurons, num_stimuli))
            
            # Store rate matrix for this population
            population_rate_matrices[name] = rate_matrix
            
            # Use the last epoch's sparsity and selectivity as the summary
            sparsity_vec = sparsity_vec_arr[-1] if len(sparsity_vec_arr) > 0 else np.array([])
            selectivity_vec = selectivity_vec_arr[-1] if len(selectivity_vec_arr) > 0 else np.array([])
            sparsity_vec = np.array(sparsity_vec)
            selectivity_vec = np.array(selectivity_vec)
            
            # Store data for this population
            population_data[name] = {
                'rate_matrices_arr': rate_matrices_arr,
                'sparsity_vec_arr': sparsity_vec_arr,
                'selectivity_vec_arr': selectivity_vec_arr,
                'individual_rate_matrices': individual_rate_matrices,
                'individual_sparsity': individual_sparsity,
                'individual_selectivity': individual_selectivity,
                'rate_matrix': rate_matrix,
                'sparsity_vec': sparsity_vec,
                'selectivity_vec': selectivity_vec
            }
            
            # Filter out silent neurons for selectivity (selectivity is 0 or NaN for neurons that never fire)
            active_sel_mask = ~(np.isnan(selectivity_vec) | (selectivity_vec == 0))
            if np.sum(active_sel_mask) == 0:
                print(f"   {name}: No active neurons")
                continue
            
            active_selectivity = selectivity_vec[active_sel_mask]
            
            # Basic selectivity statistics
            mean_sel = np.mean(active_selectivity)
            q25_sel = np.percentile(active_selectivity, 25)
            q50_sel = np.median(active_selectivity)
            q75_sel = np.percentile(active_selectivity, 75)
            max_sel = np.max(active_selectivity)
            n_high_sel = np.sum(active_selectivity > 0.5)
            
            # Sparsity statistics across stimuli
            if sparsity_vec.size > 0:
                mean_spi = np.mean(sparsity_vec)
                q25_spi = np.percentile(sparsity_vec, 25)
                q50_spi = np.median(sparsity_vec)
                q75_spi = np.percentile(sparsity_vec, 75)
                max_spi = np.max(sparsity_vec)
            else:
                mean_spi = q25_spi = q50_spi = q75_spi = max_spi = 0.0
            
            print(f"   {name}:")
            print(f"      Selectivity (active neurons):")
            print(f"         Mean: {mean_sel:.4f}")
            print(f"         Q25: {q25_sel:.4f}, Median: {q50_sel:.4f}, Q75: {q75_sel:.4f}")
            print(f"         Max: {max_sel:.4f} ({n_high_sel} neurons > 0.5)")
            print(f"      Sparsity (across stimuli):")
            print(f"         Mean: {mean_spi:.4f}")
            print(f"         Q25: {q25_spi:.4f}, Median: {q50_spi:.4f}, Q75: {q75_spi:.4f}")
            print(f"         Max: {max_spi:.4f}")

    # Save population_data to the pickle file for later use
    try:
        with open(data_file, "rb") as f:
            simulation_data = pickle.load(f)
        simulation_data['population_data'] = population_data
        with open(data_file, "wb") as f:
            pickle.dump(simulation_data, f)
    except Exception as e:
        print(f"   Warning: Could not update simulation_data with population_data: {e}")
    
    # Generate figures
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    figures_folder = os.path.join(run_folder, "Figures")
    os.makedirs(figures_folder, exist_ok=True)
    
    # Use modular plotting system for selectivity/sparsity analysis (per population)
    # This generates exemplar neuron plots with error bars and coloring
    for pop_name, pop_data in population_data.items():
        print(f"\n   Generating selectivity/sparsity plots for {pop_name}...")
        try:
            # Create population-specific run folder to avoid overwriting
            pop_run_folder = os.path.join(run_folder, f"analysis_{pop_name}")
            os.makedirs(pop_run_folder, exist_ok=True)
            
            # Build config dict for this population
            pop_config = config_dict.copy()
            pop_config['N_rec_E'] = pop_data['rate_matrix'].shape[0]  # All neurons in this pop
            pop_config['N_rec_I'] = 0  # Not applicable for per-population analysis
            
            # Build plot_data dict matching feedforward format
            plot_data = {
                'individual_rate_matrices': pop_data['individual_rate_matrices'],
                'individual_sparsity': pop_data['individual_sparsity'],
                'individual_selectivity': pop_data['individual_selectivity'],
                'rate_matrices_arr': pop_data['rate_matrices_arr'],
                'sparsity_vec_arr': pop_data['sparsity_vec_arr'],
                'selectivity_vec_arr': pop_data['selectivity_vec_arr'],
                'neuron_order': None,
                'weight_poisson': None,
                'weight_rec': None,
                'stim_info': stim_info,
                'T': T,
                'config': pop_config
            }
            
            # Call temporal evolution module directly to generate selectivity exemplar plots
            # It will create temporal_evolution/ subfolder in Figures/
            from brian_experiment.shared.analysis import temporal_evolution
            temporal_stats = temporal_evolution.generate_temporal_evolution_plots(
                run_folder=pop_run_folder,  # Population-specific folder
                data=plot_data,
                config=pop_config,
                network=None
            )
            
            # Move plots from population-specific folder to main Figures folder with population prefix
            pop_figures_src = os.path.join(pop_run_folder, "Figures", "temporal_evolution")
            if os.path.exists(pop_figures_src):
                pop_figures_dst = os.path.join(figures_folder, f"temporal_evolution_{pop_name}")
                if os.path.exists(pop_figures_dst):
                    import shutil
                    shutil.rmtree(pop_figures_dst)
                shutil.move(pop_figures_src, pop_figures_dst)
                # Clean up empty folders
                try:
                    os.rmdir(os.path.join(pop_run_folder, "Figures"))
                    os.rmdir(pop_run_folder)
                except:
                    pass
            
            print(f"      [OK] Selectivity/sparsity plots saved for {pop_name}")
        except Exception as e:
            print(f"      [ERROR] Failed to generate selectivity plots for {pop_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 1. Weight matrix
    print("   Generating weight matrix plot...")
    try:
        # plot_weight_matrix creates its own figure and returns it
        fig_weights = net.plot_weight_matrix()
        if fig_weights is not None:
            print(f"      Got figure object: {type(fig_weights)}")
            fig_weights.set_size_inches(20, 12)
            save_path = os.path.join(figures_folder, 'weight_matrix.png')
            fig_weights.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig_weights)
            if os.path.exists(save_path):
                print(f"      [OK] Saved: weight_matrix.png ({(os.path.getsize(save_path)/1024):.1f} KB)")
            else:
                print(f"      [ERROR] File not created at {save_path}")
        else:
            print(f"      [ERROR] plot_weight_matrix returned None")
    except Exception as e:
        print(f"      [ERROR] generating weight matrix plot: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Raster plot
    print("   Generating raster plot...")
    fig_raster = plt.figure(figsize=(14, 8))
    ax_raster = fig_raster.add_subplot(111)
    net.plot_raster(ax=ax_raster)
    
    # Add vertical lines between epochs
    epoch_time = (stim_duration + break_time) * num_stimuli
    for epoch_idx in range(1, num_epochs):
        epoch_boundary = start_time + epoch_idx * epoch_time
        ax_raster.axvline(epoch_boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=10)
    
    fig_raster.savefig(os.path.join(figures_folder, 'raster_plot.png'), 
                      dpi=150, bbox_inches='tight')
    plt.close(fig_raster)
    print(f"      Saved: raster_plot.png")
    
    # 3. Population statistics summary (Firing Rates + Selectivity KDE histograms)
    print("   Generating population statistics summary...")
    from scipy import stats
    
    fig_stats = plt.figure(figsize=(14, 10))
    gs = fig_stats.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Define distinct colors for each population
    pop_colors = {
        'L1_E': '#E74C3C',  # Red
        'L1_I': '#3498DB',  # Blue  
        'L2_E': '#E67E22',  # Orange
        'L2_I': '#9B59B6',  # Purple
        'L3_E': '#27AE60',  # Green
        'L3_I': '#1ABC9C',  # Teal
    }
    
    # Collect data per population
    pop_rates = {}
    pop_selectivity = {}
    
    for name in ordered_pop_names:
        if name == 'Layer0':
            continue
        if name in population_data:
            pop_data = population_data[name]
            rate_matrix = pop_data['rate_matrix']
            neuron_rates = np.mean(rate_matrix, axis=1)
            selectivity = pop_data['selectivity_vec']
            
            # Filter valid values
            neuron_rates = neuron_rates[np.isfinite(neuron_rates)]
            selectivity = selectivity[np.isfinite(selectivity)]
            
            if len(neuron_rates) > 1:
                pop_rates[name] = neuron_rates
            if len(selectivity) > 1:
                pop_selectivity[name] = selectivity
    
    # Panel 1: Firing Rate KDE by population
    ax1 = fig_stats.add_subplot(gs[0, 0])
    if pop_rates:
        max_rate = max(r.max() for r in pop_rates.values())
        x_range = np.linspace(0, max_rate * 1.1, 200)
        for name, rates in pop_rates.items():
            color = pop_colors.get(name, '#95A5A6')
            kde = stats.gaussian_kde(rates)
            ax1.fill_between(x_range, kde(x_range), alpha=0.4, color=color, label=f'{name} (n={len(rates)})')
            ax1.plot(x_range, kde(x_range), color=color, linewidth=2)
    ax1.set_xlabel('Firing Rate (Hz)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Firing Rate Distribution by Population', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(left=0)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Selectivity KDE by population
    ax2 = fig_stats.add_subplot(gs[0, 1])
    x_sel = np.linspace(0, 1.1, 200)
    for name, sel in pop_selectivity.items():
        color = pop_colors.get(name, '#95A5A6')
        kde = stats.gaussian_kde(sel)
        ax2.fill_between(x_sel, kde(x_sel), alpha=0.4, color=color, label=f'{name} (n={len(sel)})')
        ax2.plot(x_sel, kde(x_sel), color=color, linewidth=2)
    ax2.set_xlabel('Selectivity Index (SLI)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Selectivity Distribution by Population', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Mean firing rate by population (bar chart)
    ax3 = fig_stats.add_subplot(gs[1, 0])
    pop_names_plot = [name for name in ordered_pop_names if name in total_spikes and name != 'Layer0']
    rates = [total_spikes[name] / (total_neurons[name] * total_time_s) for name in pop_names_plot]
    colors = [pop_colors.get(name, '#95A5A6') for name in pop_names_plot]
    bars = ax3.bar(range(len(pop_names_plot)), rates, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xticks(range(len(pop_names_plot)))
    ax3.set_xticklabels(pop_names_plot, rotation=45, ha='right', fontsize=10)
    ax3.set_ylabel('Mean Firing Rate (Hz)', fontsize=11)
    ax3.set_title('Mean Firing Rate by Population', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, rate in zip(bars, rates):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{rate:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Panel 4: Mean selectivity by population (bar chart)
    ax4 = fig_stats.add_subplot(gs[1, 1])
    sel_means = []
    for name in pop_names_plot:
        if name in population_data:
            sel = population_data[name]['selectivity_vec']
            sel_means.append(np.nanmean(sel))
        else:
            sel_means.append(0)
    bars = ax4.bar(range(len(pop_names_plot)), sel_means, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_xticks(range(len(pop_names_plot)))
    ax4.set_xticklabels(pop_names_plot, rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel('Mean Selectivity (SLI)', fontsize=11)
    ax4.set_title('Mean Selectivity by Population', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1.0)
    for bar, sel in zip(bars, sel_means):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{sel:.2f}', ha='center', va='bottom', fontsize=9)
    
    fig_stats.suptitle('Population Statistics Summary', fontsize=14, fontweight='bold', y=0.98)
    fig_stats.tight_layout(rect=[0, 0, 1, 0.96])
    fig_stats.savefig(os.path.join(figures_folder, 'population_statistics.png'), 
                     dpi=150, bbox_inches='tight')
    plt.close(fig_stats)
    print(f"      Saved: population_statistics.png")
    
    # 4. Voltage traces with exemplars (if available)
    if monitoring_level != 'minimal':
        print("   Generating voltage traces with exemplars...")
        fig_voltage = plt.figure(figsize=(16, 12))
        
        # Plot exemplar voltage traces for each population
        plots_to_make = []
        for name in ordered_pop_names:
            if f'state_{name}' in net.monitors:
                state_mon = net.monitors[f'state_{name}']
                if hasattr(state_mon, 'v'):
                    plots_to_make.append((name, state_mon))
        
        n_plots = len(plots_to_make)
        n_exemplars = 10  # Number of exemplar neurons to show
        
        if n_plots > 0:
            gs = fig_voltage.add_gridspec(n_plots, 1, hspace=0.4)
            
            for idx, (name, state_mon) in enumerate(plots_to_make):
                ax = fig_voltage.add_subplot(gs[idx, 0])
                
                n_neurons = state_mon.v.shape[0]
                # Select exemplar neurons (spread across population)
                exemplar_indices = np.linspace(0, n_neurons-1, n_exemplars, dtype=int)
                
                # Colormap for exemplars – each neuron gets a distinct color
                cmap = plt.get_cmap('tab10', len(exemplar_indices))
                
                # Plot exemplar traces (lighter, thinner), each in its own color
                for i, neuron_idx in enumerate(exemplar_indices):
                    color_i = cmap(i)
                    alpha = 0.6
                    ax.plot(
                        state_mon.t/ms,
                        state_mon.v[neuron_idx]/mV,
                        linewidth=0.8,
                        color=color_i,
                        alpha=alpha,
                        label=f'Neuron {neuron_idx}' if i < 3 else ''
                    )
                
                # Plot mean on top (thicker, darker) in population color
                base_color = 'red' if 'E' in name else 'blue' if 'I' in name else 'black'
                mean_v = np.mean(state_mon.v, axis=0)
                ax.plot(
                    state_mon.t/ms,
                    mean_v/mV,
                    linewidth=2.0,
                    color=base_color,
                    linestyle='--',
                    label='Mean',
                    alpha=0.9
                )
                
                ax.set_ylabel(f'{name}\nVoltage (mV)', fontsize=10)
                ax.set_title(f'Voltage Traces: {name} ({n_exemplars} exemplars + mean)', fontsize=10, loc='left')
                ax.grid(True, alpha=0.3)
                if idx == n_plots - 1:
                    ax.set_xlabel('Time (ms)')
                else:
                    ax.set_xticklabels([])
                if idx == 0:
                    ax.legend(loc='upper right', fontsize=7, ncol=4)
            
            fig_voltage.suptitle('Exemplar Voltage Traces by Population', fontsize=14, fontweight='bold')
            fig_voltage.savefig(os.path.join(figures_folder, 'voltage_traces.png'), 
                              dpi=150, bbox_inches='tight')
            plt.close(fig_voltage)
            print(f"      Saved: voltage_traces.png")
        
        # 5. Current traces for sample neurons
        print("   Generating current traces for sample neurons...")
        fig_currents = plt.figure(figsize=(16, 10))
        
        n_sample_neurons = 5  # Show 5 sample neurons
        plots_to_make_currents = []
        for name in ordered_pop_names:
            if f'state_{name}' in net.monitors:
                state_mon = net.monitors[f'state_{name}']
                if hasattr(state_mon, 'I_syn_E') and hasattr(state_mon, 'I_syn_I'):
                    plots_to_make_currents.append((name, state_mon))
        
        n_current_plots = len(plots_to_make_currents)
        
        if n_current_plots > 0:
            gs_currents = fig_currents.add_gridspec(n_current_plots, 1, hspace=0.4)
            
            for idx, (name, state_mon) in enumerate(plots_to_make_currents):
                ax = fig_currents.add_subplot(gs_currents[idx, 0])
                
                n_neurons = state_mon.I_syn_E.shape[0]
                # Select sample neurons
                sample_indices = np.linspace(0, n_neurons-1, n_sample_neurons, dtype=int)
                
                # Plot I_syn_E (excitatory) in red
                for neuron_idx in sample_indices:
                    ax.plot(state_mon.t/ms, state_mon.I_syn_E[neuron_idx]/nA, 
                           linewidth=0.8, color='red', alpha=0.3, label='I_syn_E' if neuron_idx == sample_indices[0] else '')
                
                # Plot I_syn_I (inhibitory) in blue
                for neuron_idx in sample_indices:
                    ax.plot(state_mon.t/ms, -state_mon.I_syn_I[neuron_idx]/nA, 
                           linewidth=0.8, color='blue', alpha=0.3, label='-I_syn_I' if neuron_idx == sample_indices[0] else '')
                
                # Plot mean currents
                mean_I_E = np.mean(state_mon.I_syn_E, axis=0)
                mean_I_I = np.mean(state_mon.I_syn_I, axis=0)
                ax.plot(state_mon.t/ms, mean_I_E/nA, linewidth=2.0, color='red', 
                       linestyle='--', label='Mean I_syn_E', alpha=0.9)
                ax.plot(state_mon.t/ms, -mean_I_I/nA, linewidth=2.0, color='blue', 
                       linestyle='--', label='Mean -I_syn_I', alpha=0.9)
                
                ax.axhline(0, color='black', linestyle=':', linewidth=0.5, alpha=0.5)
                ax.set_ylabel(f'{name}\nCurrent (nA)', fontsize=10)
                ax.set_title(f'Synaptic Currents: {name} ({n_sample_neurons} samples + mean)', fontsize=10, loc='left')
                ax.grid(True, alpha=0.3)
                if idx == n_current_plots - 1:
                    ax.set_xlabel('Time (ms)')
                else:
                    ax.set_xticklabels([])
                if idx == 0:
                    ax.legend(loc='upper right', fontsize=8, ncol=2)
            
            fig_currents.suptitle('Synaptic Current Traces (Sample Neurons)', fontsize=14, fontweight='bold')
            fig_currents.savefig(os.path.join(figures_folder, 'current_traces.png'), 
                                dpi=150, bbox_inches='tight')
            plt.close(fig_currents)
            print(f"      Saved: current_traces.png")
    
    # Network architecture diagram
    print("   Generating network architecture diagram...")
    try:
        network_diagram_path = os.path.join(figures_folder, 'network_architecture.png')
        draw_network_diagram(config_dict, network_diagram_path)
    except Exception as e:
        print(f"      [ERROR] generating network diagram: {e}")
    
    # Exemplar neurons by population
    print("   Generating exemplar neurons by population...")
    try:
        from brian_experiment.cell_type_network.plot_exemplar_by_population import plot_exemplar_neurons_by_population
        plot_exemplar_neurons_by_population(run_folder, n_neurons_per_pop=8, seed=42)
    except Exception as e:
        print(f"      [ERROR] generating exemplar neurons plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Smooth firing rate plots by population (exemplar neurons)
    print("   Generating smooth firing rate plots (exemplar neurons)...")
    try:
        from brian_experiment.cell_type_network.plot_smooth_firing_rates import plot_smooth_firing_rates_grid, plot_population_overlaid_rates
        plot_smooth_firing_rates_grid(run_folder, n_neurons_per_pop=5, seed=42, window_ms=20.0)
    except Exception as e:
        print(f"      [ERROR] generating smooth firing rate plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Smooth firing rates overlaid (all neurons)
    print("   Generating overlaid population smooth firing rate plots...")
    try:
        plot_population_overlaid_rates(run_folder, window_ms=20.0, max_neurons_per_panel=50)
    except Exception as e:
        print(f"      [ERROR] generating overlaid smooth firing rate plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Motif analysis
    print("   Generating motif connectivity analysis...")
    try:
        from brian_experiment.shared.analysis import motif_analysis
        
        motif_figures_folder = os.path.join(figures_folder, 'motif_analysis')
        os.makedirs(motif_figures_folder, exist_ok=True)
        
        motif_save_path = os.path.join(motif_figures_folder, 'motif_analysis.png')
        observed, expected, fig_motif = motif_analysis.analyze_network_motifs(
            network=net,
            connection_configs=connection_configs,
            population_configs=population_configs,
            group_by_type=True,
            save_path=motif_save_path
        )
        plt.close(fig_motif)
        
    except Exception as e:
        print(f"      [ERROR] generating motif analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n>> Plot generation complete!")
    print(f"   Plots saved to: {figures_folder}")
    
    print(f"\n>> Simulation completed successfully!")
    print(f"   Results saved to: {run_folder}")
