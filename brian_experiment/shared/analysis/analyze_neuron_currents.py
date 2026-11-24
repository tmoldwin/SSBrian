"""
Script to analyze individual neuron currents (feed-forward excitatory, 
recurrent excitatory, recurrent inhibitory, and net excitatory).

This script demonstrates how to:
1. Load existing simulation data
2. Plot current analysis for specific neurons
3. Customize time windows for detailed inspection
"""

import matplotlib.pyplot as plt
import dill as pickle
import os
from constants import SAVE_FOLDER

def analyze_currents_from_saved_data(run_name, neuron_ids=None, time_window=None):
    """
    Load a saved simulation and plot current analysis for specific neurons.
    
    Parameters:
    -----------
    run_name : str
        Name of the run folder (e.g., "direct_input_REC_stim20_ep3_n200")
    neuron_ids : list or None
        List of neuron IDs to plot. If None, uses default selection.
    time_window : tuple or None
        (start_ms, end_ms) to limit the time window.
    """
    
    # For existing data, you would need to re-run the simulation with monitoring
    print("NOTE: To analyze currents, you need to run the simulation with monitoring_level='basic' or 'full'")
    print("The current saved data does not include individual current traces.")
    print("\nExample of how to run a new simulation with current monitoring:")
    print("""
    # In your simulation script:
    network.run_simulation(
        T=T, 
        input_vectors=input_vectors,
        stims_by_time=stims_by_time,
        stim_info=stim_info,
        current_injection=0*nA,
        monitoring_level='basic',  # Enable current monitoring
        unitary_current=1.0*nA,
        num_epochs=num_epochs
    )
    
    # After simulation, plot individual neuron currents
    fig = network.plot_individual_neuron_currents(
        neuron_ids=[0, 1, 2, 150, 151, 152],  # 3 E neurons and 3 I neurons
        time_window=(100, 1000),  # First 900ms of stimulation
        save_path='Figures/individual_neuron_currents.png'
    )
    plt.show()
    """)


def example_usage():
    """
    Example showing the typical workflow for current analysis.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Individual Neuron Current Analysis")
    print("="*60)
    
    print("\nThe network now tracks three separate current sources:")
    print("  1. Feed-forward Excitatory (I_input) - from external input")
    print("  2. Recurrent Excitatory (I_syn_rec_E) - from E neurons")
    print("  3. Recurrent Inhibitory (I_syn_rec_I) - from I neurons")
    print("  4. Net Excitatory - sum of feed-forward and recurrent excitatory")
    
    print("\nKey features:")
    print("  - Separate synaptic groups for E and I sources")
    print("  - Individual current tracking per neuron")
    print("  - Plotting function: network.plot_individual_neuron_currents()")
    
    print("\n" + "="*60)
    print("To use this feature:")
    print("="*60)
    print("1. Run your simulation with monitoring_level='basic' or 'full'")
    print("2. Call network.plot_individual_neuron_currents() after simulation")
    print("3. Optionally specify neuron_ids and time_window for focus")
    
    print("\nExample parameters:")
    print("  neuron_ids=[0, 1, 2, 150, 151, 152]  # First 3 E and 3 I neurons")
    print("  time_window=(100, 600)  # First 500ms after initial delay")


if __name__ == "__main__":
    example_usage()
    
    # If you have a specific run you want to analyze, uncomment below:
    # analyze_currents_from_saved_data(
    #     run_name="direct_input_REC_stim20_ep3_n200",
    #     neuron_ids=[0, 1, 2, 150, 151, 152],
    #     time_window=(100, 1000)
    # )

