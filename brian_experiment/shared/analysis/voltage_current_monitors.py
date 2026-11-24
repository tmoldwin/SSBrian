"""
Voltage and Current Monitors Analysis Module

This module creates detailed visualizations of voltage traces, current dynamics,
and spike rasters. Requires live network object with monitors.
Creates plots in the voltage_current_monitors/ subfolder.

Functions:
- generate_voltage_current_plots: Main entry point for voltage/current visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def generate_voltage_current_plots(run_folder, network, config, plot_params):
    """
    Generate voltage and current monitoring visualizations.
    
    This requires a LIVE network object with monitors - cannot be regenerated from saved data.
    
    Parameters:
    -----------
    run_folder : str
        Path to the run directory
    network : object
        Network object with monitors (rec_voltage, rec_current, etc.)
    config : dict
        Configuration dictionary
    plot_params : dict
        Plotting parameters (MAX_STIMULI_TO_PLOT, MAX_NEURONS_VOLTAGE_TRACES, etc.)
    
    Returns:
    --------
    dict : Statistics about voltage/current dynamics
    """
    print("\n" + "="*60)
    print("VOLTAGE & CURRENT MONITORS ANALYSIS")
    print("="*60)
    
    # Create subfolder
    monitors_folder = os.path.join(run_folder, "Figures", "voltage_current_monitors")
    os.makedirs(monitors_folder, exist_ok=True)
    
    if network is None:
        print("   WARNING: Cannot generate voltage/current plots - requires live network object")
        return {}
    
    if not hasattr(network, 'plotting_dashboard_with_subsampling'):
        print("   WARNING: Network does not have plotting_dashboard_with_subsampling method")
        return {}
    
    try:
        print("   Creating voltage/current dynamics dashboard...")
        
        # Generate dashboard from the network with subsampling
        dashboard_fig = network.plotting_dashboard_with_subsampling(
            max_stimuli=plot_params.get('MAX_STIMULI_TO_PLOT', 10),
            max_neurons_voltage=plot_params.get('MAX_NEURONS_VOLTAGE_TRACES', 20),
            max_neurons_raster=plot_params.get('MAX_NEURONS_RASTER', 100),
            num_epochs=config.get('num_epochs', 2)
        )
        
        # Save the dashboard
        save_path = os.path.join(monitors_folder, 'voltage_current_dynamics.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"      Saved: voltage_current_dynamics.png")
        
        # Generate individual neuron current analysis
        if hasattr(network, 'plot_individual_neuron_currents'):
            print("   Creating individual neuron current analysis...")
            try:
                # Plot currents for a few neurons
                current_fig = network.plot_individual_neuron_currents(
                    neuron_ids=None,  # Will default to first 3 E and 3 I
                    time_window=None,  # Plot entire simulation
                    save_path=None
                )
                if current_fig is not None:
                    save_path = os.path.join(monitors_folder, 'individual_neuron_currents.png')
                    current_fig.savefig(save_path, bbox_inches='tight', dpi=300)
                    plt.close(current_fig)
                    print(f"      Saved: individual_neuron_currents.png")
            except Exception as e:
                print(f"      Error creating individual neuron currents: {e}")
        
        # Extract some statistics if possible
        stats = {}
        if hasattr(network, 'rec_voltage') and network.rec_voltage is not None:
            voltage_data = network.rec_voltage.v
            stats['voltage'] = {
                'mean': float(np.mean(voltage_data)),
                'std': float(np.std(voltage_data)),
                'min': float(np.min(voltage_data)),
                'max': float(np.max(voltage_data)),
            }
            print(f"\n   Voltage Statistics:")
            print(f"   Mean: {stats['voltage']['mean']:.4f} V")
            print(f"   Range: [{stats['voltage']['min']:.4f}, {stats['voltage']['max']:.4f}] V")
        
        return stats
        
    except Exception as e:
        print(f"   Error creating voltage/current dynamics: {e}")
        import traceback
        traceback.print_exc()
        return {}


def generate_detailed_current_breakdown(run_folder, network, data, config):
    """
    Generate detailed current breakdown analysis (if monitors are available).
    
    Parameters:
    -----------
    run_folder : str
        Path to the run directory
    network : object
        Network object with current monitors
    data : dict
        Simulation data dictionary
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    dict : Statistics about current breakdown
    """
    if network is None:
        print("   Skipping detailed current breakdown - requires live network object")
        return {}
    
    if not hasattr(network, 'rec_current') or network.rec_current is None:
        print("   Skipping detailed current breakdown - no current monitors")
        return {}
    
    try:
        from brian_experiment.shared.plotting import plots as pf
        from brian2 import ms
        
        monitors_folder = os.path.join(run_folder, "Figures", "voltage_current_monitors")
        
        stim_info = data.get('stim_info')
        T = data.get('T')
        N_rec_E = config.get('N_rec_E', 100)
        
        if stim_info is not None and T is not None:
            print("   Creating detailed current breakdown...")
            
            current_fig = pf.plot_detailed_currents(
                num_neurons_to_plot=4,
                rec_current_mon=network.rec_current,
                input_current_mon=network.input_current,
                stim_info=stim_info,
                T=T,
                n_rec_e=N_rec_E
            )
            
            if current_fig:
                save_path = os.path.join(monitors_folder, 'detailed_current_breakdown.png')
                current_fig.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close(current_fig)
                print(f"      Saved: detailed_current_breakdown.png")
                return {'detailed_breakdown': True}
        else:
            print("   Skipping detailed current breakdown - stim_info or T not available")
            
    except Exception as e:
        print(f"   Error creating detailed current breakdown: {e}")
        import traceback
        traceback.print_exc()
    
    return {}


if __name__ == "__main__":
    """Command-line interface for standalone voltage/current analysis"""
    print("NOTE: Voltage/current analysis requires a live network object with monitors.")
    print("This analysis cannot be run from saved data alone.")
    print("Run the main simulation script to generate these plots.")

