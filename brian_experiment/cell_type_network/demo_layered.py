from brian2 import *
from brian_experiment.cell_type_network.network import CellTypeNetwork
import matplotlib.pyplot as plt
import numpy as np
from brian_experiment.shared.core import stimuli as stim

def run_layered_demo():
    # Define populations
    populations = []
    
    # Layer 1
    populations.append({'name': 'L1_E', 'n': 400, 'type': 'E'})
    populations.append({'name': 'L1_I', 'n': 100, 'type': 'I'}) 
    
    # Layer 2
    populations.append({'name': 'L2_E', 'n': 400, 'type': 'E'})
    populations.append({'name': 'L2_I', 'n': 100, 'type': 'I'})
    
    # Layer 3
    populations.append({'name': 'L3_E', 'n': 400, 'type': 'E'})
    populations.append({'name': 'L3_I', 'n': 100, 'type': 'I'})
    
    # Define connections
    connections = []
    
    # Input Layer 0 -> Layer 1 E
    connections.append({
        'source': 'Layer0', 'target': 'L1_E',
        'p': 0.1,
        'weight_dist': {'dist': 'fixed', 'val': 1.0}, # Strong drive
        'weight_scale': 1.0
    })

    layers = ['L1', 'L2', 'L3']
    
    for i, layer in enumerate(layers):
        e_name = f"{layer}_E"
        i_name = f"{layer}_I"
        
        # Recurrent connections within layer
        # E -> E, E -> I, I -> E, I -> I
        
        # E->E
        connections.append({
            'source': e_name, 'target': e_name, 
            'p': 0.1, 
            'weight_dist': {'dist': 'fixed', 'val': 0.1}, # 0.1 nA
            'weight_scale': 1.0
        })
        
        # E->I
        connections.append({
            'source': e_name, 'target': i_name, 
            'p': 0.1, 
            'weight_dist': {'dist': 'fixed', 'val': 0.1},
            'weight_scale': 1.0
        })
        
        # I->E
        connections.append({
            'source': i_name, 'target': e_name, 
            'p': 0.1, 
            'weight_dist': {'dist': 'fixed', 'val': 0.5}, # Stronger inhibition
            'weight_scale': 1.0
        })
        
        # I->I
        connections.append({
            'source': i_name, 'target': i_name, 
            'p': 0.1, 
            'weight_dist': {'dist': 'fixed', 'val': 0.5},
            'weight_scale': 1.0
        })
        
        # Feedforward to next layer (E -> Next E)
        if i < len(layers) - 1:
            next_layer_e = f"{layers[i+1]}_E"
            connections.append({
                'source': e_name, 'target': next_layer_e,
                'p': 0.05,
                'weight_dist': {'dist': 'fixed', 'val': 0.2},
                'weight_scale': 1.0
            })
            print(f"Adding FF connection: {e_name} -> {next_layer_e}")

    # Initialize network
    n_input = 100
    net = CellTypeNetwork(
        population_configs=populations, 
        connection_configs=connections,
        input_layer_config={'n': n_input, 'name': 'Layer0'},
        simulation_params={'tau_m': 20*ms},
        verbose=True
    )
    
    # ---------------------------------------------------------
    # Generate Stimuli using Shared Core
    # ---------------------------------------------------------
    num_stimuli = 5
    stim_duration = 200 # ms
    break_time = 50 # ms
    start_time = 100 # ms
    
    # Total duration calculation (rough)
    # T = start_time + num_stimuli * (stim_duration + break_time)
    # Using stim.create_stim_list logic
    
    stim_list = np.arange(num_stimuli)
    
    # We define T long enough to hold all stimuli
    total_epoch_time = (stim_duration + break_time) * num_stimuli
    T = (start_time + total_epoch_time) * ms
    
    # Create schedule
    stims_by_time, stim_info = stim.create_stim_list(
        T/ms, 
        stim_duration, 
        stim_list, 
        break_time, 
        start_time=start_time
    )
    
    # Generate patterns (firing rates in Hz)
    # Using Dirichlet distribution for patterns
    input_patterns_matrix = stim.generate_stimuli_from_distribution(
        stims_by_time, 
        n_input, 
        distribution='dirichlet',
        alpha=0.5,
        scale_factor=50.0 # Scale to 50 Hz max roughly
    )
    
    # Visualize connectivity
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3)
    
    # Weight Matrix (Top Left)
    ax_weights = fig.add_subplot(gs[0, 0])
    net.plot_weight_matrix(ax=ax_weights)
    
    # Run simulation with patterns
    # Note: input_patterns_matrix is [n_input, time_steps]
    net.run_simulation(T, input_patterns=input_patterns_matrix, monitoring_level='basic')
    
    # Raster Plot (Top Middle/Right)
    ax_raster = fig.add_subplot(gs[0, 1:])
    net.plot_raster(ax=ax_raster)
    
    # Mean Voltages (Bottom Row)
    for i, layer in enumerate(layers):
        ax_volt = fig.add_subplot(gs[1, i])
        e_name = f"{layer}_E"
        mon = net.monitors[f'state_{e_name}']
        mean_v = np.mean(mon.v, axis=0)
        ax_volt.plot(mon.t/ms, mean_v/mV, label=f'{e_name} Mean V')
        ax_volt.set_ylabel('Voltage (mV)')
        ax_volt.legend()
        ax_volt.set_title(f"Mean Voltage {layer}")
        ax_volt.set_xlabel('Time (ms)')
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_layered_demo()

