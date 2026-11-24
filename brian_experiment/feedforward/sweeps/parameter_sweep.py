"""
Parameter Sweep Module

A general-purpose system for running multiple simulations with different parameter
configurations and organizing the results for aggregate analysis.

Key Features:
- Flexible parameter specification (vary any parameters from direct_input_main)
- Organized data storage (one folder per sweep with subfolders for each run)
- Optional individual plots per run (disabled by default for speed)
- Metadata tracking for easy aggregation
- Support for parameter grids, lists, and random sampling

Usage Example:
    sweep = ParameterSweep(
        sweep_name="n_rec_sweep",
        base_params={...},  # Default parameters
        varied_params={
            'N_rec_E': [50, 100, 200],
            'N_rec_I': [50, 100, 200]
        }
    )
    sweep.run()
"""

import os
import numpy as np
import pickle
import datetime as dt
from itertools import product
from brian2 import *
import re
import random as py_random

# Import simulation modules
from brian_experiment.feedforward import network as din
from brian_experiment.feedforward.configs import config_manager as cfg_manager
from brian_experiment.shared.analysis import sparsity_and_selectivity as ss
from brian_experiment.shared.core import stimuli as stim
from brian_experiment.shared.core import weight_matrices as wm
from brian_experiment.shared.core import spike_analysis as sa
from brian_experiment.shared.core import spike_analysis_utils as sau


class ParameterSweep:
    """
    Main class for running parameter sweeps.
    
    Handles parameter combination generation, simulation execution,
    and data storage organization.
    """
    
    def __init__(self, sweep_name, base_params=None, varied_params=None, 
                 sweep_type='grid', random_samples=None, random_seed=42,
                 generate_individual_plots=False, plots_config=None,
                 n_runs_per_param=1):
        """
        Initialize parameter sweep.
        
        Parameters:
        -----------
        sweep_name : str
            Name for this parameter sweep (used for folder naming)
        base_params : dict
            Base parameter configuration (parameters that don't vary)
        varied_params : dict
            Parameters to vary. Format: {'param_name': [val1, val2, ...]}
        sweep_type : str
            'grid' (all combinations) or 'list' (paired values) or 'random'
        random_samples : int
            Number of random samples (only used if sweep_type='random')
        random_seed : int
            Random seed for reproducibility
        generate_individual_plots : bool
            Whether to generate plots for each individual run
        plots_config : dict
            Configuration for which plots to generate (if any)
        n_runs_per_param : int
            Number of runs to perform for each parameter combination (for averaging)
        """
        self.sweep_name = sweep_name
        self.random_seed = random_seed
        self.generate_individual_plots = generate_individual_plots
        self.plots_config = plots_config or {}
        self.n_runs_per_param = n_runs_per_param
        
# Set default base parameters (from feedforward/single_run.py defaults)
        self.base_params = self._get_default_params()
        if base_params:
            self.base_params.update(base_params)
        
        self.varied_params = varied_params or {}
        self.sweep_type = sweep_type
        self.random_samples = random_samples
        
        # Create main sweep folder under the shared Runs hierarchy
        sweep_root, timestamp = cfg_manager.prepare_sweep_folder(sweep_name)
        self.sweep_folder = str(sweep_root)
        
        # Generate parameter combinations
        self.param_combinations_base = self._generate_combinations()
        
        # Expand combinations to include multiple runs per parameter
        self.param_combinations = []
        self.param_combination_groups = {}  # Map from run_idx to base param index
        run_idx = 0
        for base_idx, base_params_combo in enumerate(self.param_combinations_base):
            for run_repeat in range(n_runs_per_param):
                self.param_combinations.append(base_params_combo.copy())
                self.param_combination_groups[run_idx] = base_idx
                run_idx += 1
        
        # Store sweep metadata
        self.metadata = {
            'sweep_name': sweep_name,
            'timestamp': timestamp,
            'base_params': self.base_params,
            'varied_params': self.varied_params,
            'sweep_type': sweep_type,
            'n_param_combinations': len(self.param_combinations_base),
            'n_runs_per_param': n_runs_per_param,
            'n_runs': len(self.param_combinations),
            'random_seed': random_seed,
            'generate_individual_plots': generate_individual_plots
        }
        self._save_metadata()
        
        print(f"\n{'='*60}")
        print(f"PARAMETER SWEEP: {sweep_name}")
        print(f"{'='*60}")
        print(f"Sweep folder: {self.sweep_folder}")
        print(f"Parameter combinations: {len(self.param_combinations_base)}")
        print(f"Runs per parameter: {n_runs_per_param}")
        print(f"Total runs: {len(self.param_combinations)}")
        print(f"Varied parameters: {list(varied_params.keys())}")
        print(f"Individual plots: {generate_individual_plots}")
        print(f"{'='*60}\n")
    
    def _get_default_params(self):
        """Get default parameters matching feedforward/single_run.py"""
        return {
            # Network Architecture
            'NUM_STIMULI': 100,
            'NUM_EPOCHS': 2,
            'N_rec_E': 100,
            'N_rec_I': 100,
            'N_external_input_dims': 100,
            
            # Simulation Timing (in ms)
            'stim_duration': 100,
            'break_time': 20,
            'start_time': 100,
            
            # Weight & Input Scaling
            'base_weight_scale': 1,
            'weight_scale_method': 'sqrt_k',
            'inhibitory_weight_scale': 5,
            'input_amplitude_scale': 1.0,
            
            # Stimulus Generation
            'stimulus_distribution': 'dirichlet',
            'stimulus_alpha': 0.5,
            
            # Input Mode
            'probabilistic_input': True,
            'unitary_current_nA': 1.0,
            
            # Recurrent Connections
            'disable_recurrent': False,
            
            # Plot Configuration
            'max_stimuli_to_plot': 10,
            'max_neurons_voltage_traces': 20,
            'max_neurons_raster': 100,
        }
    
    def _generate_combinations(self):
        """Generate parameter combinations based on sweep type"""
        if self.sweep_type == 'grid':
            # Grid search: all combinations
            param_names = list(self.varied_params.keys())
            param_values = list(self.varied_params.values())
            combinations = []
            for values in product(*param_values):
                combo = self.base_params.copy()
                combo.update(dict(zip(param_names, values)))
                combinations.append(combo)
            return combinations
        
        elif self.sweep_type == 'list':
            # List: paired values (all lists must be same length)
            param_names = list(self.varied_params.keys())
            param_values = list(self.varied_params.values())
            n_combos = len(param_values[0])
            
            # Verify all lists are same length
            if not all(len(v) == n_combos for v in param_values):
                raise ValueError("For 'list' sweep type, all parameter lists must be same length")
            
            combinations = []
            for i in range(n_combos):
                combo = self.base_params.copy()
                for param_name, param_list in zip(param_names, param_values):
                    combo[param_name] = param_list[i]
                combinations.append(combo)
            return combinations
        
        elif self.sweep_type == 'random':
            # Random sampling from parameter ranges
            if self.random_samples is None:
                raise ValueError("Must specify random_samples for 'random' sweep type")
            
            np.random.seed(self.random_seed)
            combinations = []
            for _ in range(self.random_samples):
                combo = self.base_params.copy()
                for param_name, param_range in self.varied_params.items():
                    # Assume param_range is [min, max]
                    if isinstance(param_range[0], int):
                        value = np.random.randint(param_range[0], param_range[1] + 1)
                    else:
                        value = np.random.uniform(param_range[0], param_range[1])
                    combo[param_name] = value
                combinations.append(combo)
            return combinations
        
        else:
            raise ValueError(f"Unknown sweep_type: {self.sweep_type}")
    
    def _save_metadata(self):
        """Save sweep metadata"""
        metadata_file = os.path.join(self.sweep_folder, "sweep_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Also save human-readable version
        metadata_txt = os.path.join(self.sweep_folder, "sweep_info.txt")
        with open(metadata_txt, 'w') as f:
            f.write(f"Parameter Sweep: {self.sweep_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Timestamp: {self.metadata['timestamp']}\n")
            f.write(f"Total runs: {self.metadata['n_runs']}\n")
            f.write(f"Sweep type: {self.sweep_type}\n")
            f.write(f"Random seed: {self.random_seed}\n\n")
            
            f.write(f"Varied Parameters:\n")
            f.write(f"{'-'*60}\n")
            for param, values in self.varied_params.items():
                f.write(f"  {param}: {values}\n")
            
            f.write(f"\nBase Parameters:\n")
            f.write(f"{'-'*60}\n")
            for param, value in self.base_params.items():
                if param not in self.varied_params:
                    f.write(f"  {param}: {value}\n")
    
    def _run_single_simulation(self, params, run_idx):
        """
        Run a single simulation with given parameters.
        Returns simulation data and summary statistics.
        """
        # Setup device
        prefs.codegen.target = 'numpy'
        
        # Seed for reproducibility
        # Use base param index to ensure different seeds for different parameter combinations
        # but same seed for repeated runs of same parameter (for averaging)
        base_param_idx = self.param_combination_groups.get(run_idx, run_idx)
        sim_seed = self.random_seed + base_param_idx * 1000 + run_idx
        np.random.seed(sim_seed)
        py_random.seed(sim_seed)
        
        # Create run folder
        run_name = f"run_{run_idx:04d}"
        run_folder = os.path.join(self.sweep_folder, run_name)
        data_folder = os.path.join(run_folder, "Data")
        os.makedirs(data_folder, exist_ok=True)
        
        # Extract parameters
        N_rec_E = params['N_rec_E']
        N_rec_I = params['N_rec_I']
        N_external_input_dims = params['N_external_input_dims']
        num_epochs = params['NUM_EPOCHS']
        NUM_STIMULI = params['NUM_STIMULI']
        stim_list = np.arange(NUM_STIMULI)
        
        stim_duration = params['stim_duration']
        break_time = params['break_time']
        start_time = params['start_time']
        
        is_feedforward = params['disable_recurrent']
        
        # Get which parameter combination this is
        base_param_idx = self.param_combination_groups.get(run_idx, run_idx)
        run_within_param = run_idx % self.n_runs_per_param
        param_run_num = run_within_param + 1
        
        print(f"\n{'='*60}", flush=True)
        print(f"Run {run_idx + 1}/{len(self.param_combinations)} (Parameter Set {base_param_idx + 1}/{len(self.param_combinations_base)}, Run {param_run_num}/{self.n_runs_per_param})", flush=True)
        print(f"Parameters: N_E={N_rec_E}, N_I={N_rec_I}, N_total={N_rec_E + N_rec_I}, stimuli={NUM_STIMULI}, epochs={num_epochs}", flush=True)
        print(f"{'='*60}", flush=True)
        
        # Calculate simulation time
        epoch_time = (stim_duration + break_time) * len(stim_list) * ms
        T = start_time * ms + num_epochs * epoch_time
        
        # Calculate effective recurrent weight scale
        k_recurrent = N_rec_E + N_rec_I
        if params['weight_scale_method'] == 'sqrt_k':
            r_w_scale_eff = params['base_weight_scale'] / np.sqrt(k_recurrent)
        else:
            r_w_scale_eff = params['base_weight_scale'] / k_recurrent
        
        # Generate weight matrix
        if is_feedforward:
            weight_rec = np.zeros((N_rec_E + N_rec_I, N_rec_E + N_rec_I))
        else:
            lognormal_params_rec = [0.0, 0.5, (N_rec_E + N_rec_I, N_rec_E + N_rec_I)]
            weight_rec = r_w_scale_eff * wm.weight_matrix_factory(
                {'recurrent': [N_rec_E, N_rec_I]}, 
                {'lognormal': lognormal_params_rec}
            )
            
            # Apply inhibitory weight scaling
            inhibitory_scale = params['inhibitory_weight_scale']
            weight_rec[N_rec_E:, :] *= inhibitory_scale  # I->E and I->I connections
        
        # Calculate effective input weight scale
        if params['weight_scale_method'] == 'sqrt_k':
            k_factor = np.sqrt(N_external_input_dims)
        else:
            k_factor = N_external_input_dims
        input_weight_scale_eff = params['input_amplitude_scale'] / k_factor
        
        # Generate stimulus schedule  
        stims_by_time, stim_info = stim.create_stim_list(
            T, stim_duration, stim_list, break_time
        )
        
        # Fix first stimulus if needed
        if stims_by_time[0] == -1:
            stims_by_time[0] = stim_list[0]
        
        # Generate input vectors
        from brian_experiment.feedforward.single_run import generate_direct_input_vectors
        input_vectors, stimulus_patterns = generate_direct_input_vectors(
            stims_by_time,
            N_external_input_dims,
            distribution=params['stimulus_distribution'],
            alpha=params['stimulus_alpha'],
            probabilistic_input=params['probabilistic_input']
        )
        
        # Initialize network
        lognormal_params_in = [0.0, 0.5, (N_rec_E + N_rec_I, N_external_input_dims)]
        direct_network = din.DirectInputNetwork(
            N_rec_E, N_rec_I, N_external_input_dims, 
            weight_rec, input_weight_scale_eff, 
            lognormal_params_in, is_feedforward
        )
        
        # Run simulation
        unitary_current = params['unitary_current_nA'] * nA
        direct_network.run_simulation(
            T, input_vectors, stims_by_time, stim_info, 0*nA,
            monitoring_level='minimal',  # Use minimal for speed
            unitary_current=unitary_current,
            num_epochs=num_epochs
        )
        
        # Analyze results  
        rate_matrices_arr, sparsity_vec_arr, selectivity_vec_arr = \
            sa.analyze_firing_rates_over_time_long(
                direct_network.rec_spikes, T, stims_by_time,
                num_epochs, N_rec_E + N_rec_I, defaultclock.dt, start_time * ms
            )
        
        # Calculate summary statistics
        final_rate_matrix = rate_matrices_arr[-1]
        final_selectivity = selectivity_vec_arr[-1, :]
        final_sparsity = sparsity_vec_arr[-1, :]
        mean_firing_rates = np.mean(final_rate_matrix, axis=1)
        
        summary_stats = {
            'run_idx': run_idx,
            'params': params.copy(),
            'mean_selectivity': np.mean(final_selectivity),
            'std_selectivity': np.std(final_selectivity),
            'mean_sparsity': np.mean(final_sparsity),
            'std_sparsity': np.std(final_sparsity),
            'mean_firing_rate': np.mean(mean_firing_rates),
            'std_firing_rate': np.std(mean_firing_rates),
            'n_active_neurons': np.sum(mean_firing_rates > 0),
            'selectivity_distribution': final_selectivity,
            'sparsity_distribution': final_sparsity,
            'firing_rate_distribution': mean_firing_rates,
        }
        
        # Save full simulation data (match format from direct_input_main)
        simulation_data = {
            'config': params,
            'rate_matrices_arr': rate_matrices_arr,
            'selectivity_vec_arr': selectivity_vec_arr,
            'sparsity_vec_arr': sparsity_vec_arr,
            'weight_rec': weight_rec,
            'weight_poisson': direct_network.input_weights,
            'stims_by_time': stims_by_time,
            'stim_info': stim_info,
            'T': T,
            'input_vectors': input_vectors,
            'stimulus_patterns': stimulus_patterns,
            'neuron_order': getattr(direct_network, 'neuron_order', None),
            'summary_stats': summary_stats
        }
        
        data_file = os.path.join(data_folder, "simulation_data.pkl")
        with open(data_file, 'wb') as f:
            pickle.dump(simulation_data, f)
        
        # Save summary stats separately for quick access
        stats_file = os.path.join(run_folder, "summary_stats.pkl")
        with open(stats_file, 'wb') as f:
            pickle.dump(summary_stats, f)
        
        # Generate individual plots if requested
        if self.generate_individual_plots:
            from brian_experiment.shared.plotting.generate_plots import generate_all_plots
            figures_folder = os.path.join(run_folder, "Figures")
            os.makedirs(figures_folder, exist_ok=True)
            
            try:
                generate_all_plots(
                    run_folder=run_folder,
                    network=None,  # No live network
                    data=simulation_data,
                    config=params,
                    plots_config=self.plots_config
                )
            except Exception as e:
                print(f"Warning: Error generating plots for run {run_idx}: {e}")
        
        print(f"   Mean selectivity: {summary_stats['mean_selectivity']:.3f}")
        print(f"   Mean sparsity: {summary_stats['mean_sparsity']:.3f}")
        print(f"   Mean firing rate: {summary_stats['mean_firing_rate']:.3f} Hz")
        print(f"   Active neurons: {summary_stats['n_active_neurons']}/{N_rec_E + N_rec_I}")
        
        return summary_stats
    
    def run(self):
        """Run all simulations in the parameter sweep"""
        print(f"\n{'='*60}")
        print(f"Starting parameter sweep with {len(self.param_combinations)} total runs")
        print(f"  - {len(self.param_combinations_base)} unique parameter combinations")
        print(f"  - {self.n_runs_per_param} runs per parameter combination")
        print(f"{'='*60}\n")
        
        all_stats = []
        for idx, params in enumerate(self.param_combinations):
            try:
                stats = self._run_single_simulation(params, idx)
                all_stats.append(stats)
            except Exception as e:
                print(f"\nERROR in run {idx}: {e}")
                import traceback
                traceback.print_exc()
                # Save error info
                error_file = os.path.join(self.sweep_folder, f"run_{idx:04d}", "error.txt")
                os.makedirs(os.path.dirname(error_file), exist_ok=True)
                with open(error_file, 'w') as f:
                    f.write(f"Error in run {idx}:\n")
                    f.write(f"{e}\n\n")
                    f.write(traceback.format_exc())
                continue
        
        # Save aggregate statistics
        aggregate_file = os.path.join(self.sweep_folder, "aggregate_stats.pkl")
        with open(aggregate_file, 'wb') as f:
            pickle.dump(all_stats, f)
        
        print(f"\n{'='*60}")
        print(f"Parameter sweep complete!")
        print(f"Results saved to: {self.sweep_folder}")
        print(f"Successful runs: {len(all_stats)}/{len(self.param_combinations)}")
        print(f"{'='*60}\n")
        
        return all_stats


def example_sweeps():
    """Example parameter sweep configurations"""
    
    # Example 1: Sweep over network size
    sweep1 = ParameterSweep(
        sweep_name="network_size_sweep",
        varied_params={
            'N_rec_E': [50, 100, 200],
            'N_rec_I': [50, 100, 200]
        },
        sweep_type='grid',
        generate_individual_plots=False
    )
    
    # Example 2: Sweep over stimulus parameters
    sweep2 = ParameterSweep(
        sweep_name="stimulus_alpha_sweep",
        varied_params={
            'stimulus_alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        },
        sweep_type='list',
        generate_individual_plots=False
    )
    
    # Example 3: Sweep over multiple parameters with reduced stimuli
    sweep3 = ParameterSweep(
        sweep_name="multi_param_sweep",
        base_params={
            'NUM_STIMULI': 50,  # Reduce for speed
            'NUM_EPOCHS': 2
        },
        varied_params={
            'N_rec_E': [50, 100, 150],
            'inhibitory_weight_scale': [3, 5, 7]
        },
        sweep_type='grid',
        generate_individual_plots=False
    )
    
    return sweep1, sweep2, sweep3


if __name__ == "__main__":
    """
    Example usage of parameter sweep.
    Uncomment the sweep you want to run.
    """
    
    # Example: Simple sweep over network size
    sweep = ParameterSweep(
        sweep_name="test_sweep",
        base_params={
            'NUM_STIMULI': 20,  # Small for testing
            'NUM_EPOCHS': 1
        },
        varied_params={
            'N_rec_E': [50, 100],
            'N_rec_I': [50, 100]
        },
        sweep_type='grid',
        generate_individual_plots=False
    )
    
    # Run the sweep
    results = sweep.run()
    
    print("\nTo analyze results, run:")
    print(f"  python aggregate_analysis.py {sweep.sweep_folder}")

