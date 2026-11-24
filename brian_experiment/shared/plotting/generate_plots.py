"""
Generate Plots - Main Orchestration Module

This module orchestrates all plot generation by calling specialized analysis modules.
Each analysis module creates plots in its corresponding subfolder.

Analysis Modules:
- network_structure.py → network_structure/
- connectivity_tuning.py → connectivity_tuning/
- temporal_evolution.py → temporal_evolution/
- voltage_current_monitors.py → voltage_current_monitors/
- correlation_analysis.py → correlation_analysis/
- input_response_analysis.py → input_response_analysis/

Usage:
    python generate_plots.py <run_folder>
Or import and call:
    generate_all_plots(run_folder, network, data, config, plots_config, plot_params)
"""

import os
import pickle
import matplotlib.pyplot as plt

# Import modular analysis modules
from brian_experiment.feedforward.analysis import input_response_analysis
from brian_experiment.shared.analysis import connectivity_tuning
from brian_experiment.shared.analysis import correlation_analysis
from brian_experiment.shared.analysis import temporal_evolution
from brian_experiment.shared.analysis import voltage_current_monitors
from brian_experiment.shared.core import network_structure


def load_simulation_data(run_folder):
    """Load simulation data from pickle file"""
    data_file = os.path.join(run_folder, "Data", "simulation_data.pkl")
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"   Simulation data loaded from: {data_file}")
        return data_dict
    else:
        raise FileNotFoundError(f"No simulation data found at: {data_file}")


def generate_all_plots(run_folder, network=None, data=None, config=None, 
                      plots_config=None, plot_params=None):
    """
    Generate all plots by calling modular analysis files.
    
    Parameters:
    -----------
    run_folder : str
        Path to the run directory
    network : object, optional
        Live network object with monitors (needed for voltage/current plots)
    data : dict, optional
        Simulation data dictionary (if None, will load from file)
    config : dict, optional
        Configuration dictionary (if None, will load from data)
    plots_config : dict, optional
        Configuration for which plots to generate
    plot_params : dict, optional
        Parameters for plotting (subsampling, etc.)
    
    Returns:
    --------
    dict : Combined statistics from all analyses
    """
    print("\n" + "="*60)
    print("GENERATING PLOTS - MODULAR ANALYSIS")
    print("="*60)
    
    # Create main Figures folder
    figures_folder = os.path.join(run_folder, "Figures")
    os.makedirs(figures_folder, exist_ok=True)
    
    # Load data if not provided
    if data is None:
        try:
            data = load_simulation_data(run_folder)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return {}
    
    # Extract config from data if not provided
    if config is None:
        if 'config' in data:
            config = data['config']
        else:
            print("WARNING: No config found in saved data, using defaults")
            config = {
                'num_epochs': 2,
                'N_rec_E': 100,
                'N_rec_I': 100,
                'N_external_input_dims': 100,
            }
    
    # Default plots configuration
    if plots_config is None:
        plots_config = {
            'network_structure': True,
            'connectivity_tuning': True,
            'temporal_evolution': True,
            'voltage_current_monitors': True,
            'correlation_analysis': True,
            'input_response_analysis': True,
        }
    
    # Default plot parameters
    if plot_params is None:
        plot_params = {
            'MAX_STIMULI_TO_PLOT': 10,
            'MAX_NEURONS_VOLTAGE_TRACES': 20,
            'MAX_NEURONS_RASTER': 100,
        }
    
    # Store all statistics
    all_stats = {}
    
    # ===== 1. NETWORK STRUCTURE =====
    if plots_config.get('network_structure', True):
        try:
            stats = network_structure.generate_network_structure_plots(run_folder, data, config)
            all_stats['network_structure'] = stats
        except Exception as e:
            print(f"Error in network structure module: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== 2. CONNECTIVITY TUNING (CCI) =====
    if plots_config.get('connectivity_tuning', True):
        try:
            stats = connectivity_tuning.generate_connectivity_tuning_plots(run_folder, data, config)
            all_stats['connectivity_tuning'] = stats
        except Exception as e:
            print(f"Error in connectivity tuning module: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== 3. TEMPORAL EVOLUTION =====
    if plots_config.get('temporal_evolution', True):
        try:
            stats = temporal_evolution.generate_temporal_evolution_plots(
                run_folder, data, config, network
            )
            all_stats['temporal'] = stats
        except Exception as e:
            print(f"Error in temporal evolution module: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== 4. VOLTAGE & CURRENT MONITORS =====
    if plots_config.get('voltage_current_monitors', True):
        try:
            stats = voltage_current_monitors.generate_voltage_current_plots(
                run_folder, network, config, plot_params
            )
            all_stats['voltage_current'] = stats
            
            # Also try detailed current breakdown if available
            if network is not None:
                stats2 = voltage_current_monitors.generate_detailed_current_breakdown(
                    run_folder, network, data, config
                )
                if stats2:
                    all_stats['voltage_current'].update(stats2)
                    
        except Exception as e:
            print(f"Error in voltage/current monitors module: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== 5. CORRELATION ANALYSIS =====
    if plots_config.get('correlation_analysis', True):
        try:
            stats = correlation_analysis.generate_correlation_plots(
                run_folder, data, config
            )
            all_stats['correlation'] = stats
        except Exception as e:
            print(f"Error in correlation analysis module: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== 6. INPUT-RESPONSE ANALYSIS =====
    if plots_config.get('input_response_analysis', True):
        try:
            stats = input_response_analysis.generate_input_response_plots(
                network, run_folder
            )
            all_stats['input_response'] = stats
        except Exception as e:
            print(f"Error in input-response analysis module: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("   Plot generation complete")
    print(f"   Plots saved to: {figures_folder}")
    print("="*60 + "\n")
    
    return all_stats


def main():
    """Command-line interface for regenerating plots from saved data"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_plots.py <run_folder>")
        print("\nExample:")
        print("  python generate_plots.py brian_experiment/Runs/direct_input_REC_stim20_ep3_n200")
        sys.exit(1)
    
    run_folder = sys.argv[1]
    
    if not os.path.exists(run_folder):
        print(f"Error: Run folder not found: {run_folder}")
        sys.exit(1)
    
    print(f"Regenerating plots for: {run_folder}")
    
    # Load data
    data = load_simulation_data(run_folder)
    
    # Extract config from data
    config = data.get('config', {})
    
    # Generate all plots (without network object, so voltage/current will be skipped)
    stats = generate_all_plots(
        run_folder=run_folder,
        network=None,  # No live network when regenerating
        data=data,
        config=config,
        plots_config={
            'network_structure': True,
            'connectivity_tuning': True,
            'temporal_evolution': True,
            'voltage_current_monitors': False,  # Skip - requires live network
            'correlation_analysis': True,
            'input_response_analysis': True,  # Can regenerate from data
        }
    )
    
    print("\nPlot regeneration complete!")
    print(f"Results saved to: {os.path.join(run_folder, 'Figures')}")


if __name__ == "__main__":
    main()
