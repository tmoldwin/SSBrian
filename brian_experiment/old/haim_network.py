from brian2 import *
import matplotlib.pyplot as plt
prefs.codegen.target = "numpy"
import plots as pf
import stimuli as stim
import weight_matrices as wm
import numpy as np
import spike_analysis_utils as sau
import dill as pickle
from constants import SAVE_FOLDER  # Import the constant
import os




class HaimNetwork:
    def __init__(self, N_rec_E, N_rec_I, N_poisson_E, N_poisson_I, weight_rec, weight_poisson):
        print("Initializing HaimNetwork...")
        self.net = None
        start_scope()
        # N_rec for number of recurrent and so on.
        self.N_rec_E = N_rec_E
        self.N_rec_I = N_rec_I
        self.N_poisson_I = N_poisson_I
        self.N_poisson_E = N_poisson_E
        self.weight_rec = weight_rec
        self.weight_poisson = weight_poisson
        self.N_rec = self.N_rec_I + self.N_rec_E
        self.N_poisson = self.N_poisson_I + self.N_poisson_E

        # Biological parameters
        self.tau_m = 20*ms       # Membrane time constant
        self.tau_syn = 1*ms      # Synaptic time constant
        self.R_m = 100*Mohm      # Membrane resistance
        self.v_rest = -0*mV     # Resting potential
        self.v_thresh = 1*mV   # Spike threshold
        
        # Derived parameter
        self.C_m = self.tau_m/self.R_m  # Membrane capacitance

        # Fixed equations without comments for CUDA compatibility
        self.eqs = '''
        dv/dt = (v_rest - v + R_m * (I_syn_rec + I_syn_poiss + I_ext))/tau_m : volt
        dI_syn_rec/dt = -I_syn_rec/tau_syn : amp
        dI_syn_poiss/dt = -I_syn_poiss/tau_syn : amp
        I_ext : amp
        '''

        # recurrent neurons
        self.rec_neurons = NeuronGroup(
            self.N_rec, 
            self.eqs, 
            threshold='v > v_thresh',
            reset='v = v_rest', 
            refractory=2*ms,
            method='euler',  # Changed from 'exact' to 'euler' for CUDA compatibility
            namespace={
                'v_rest': self.v_rest,
                'v_thresh': self.v_thresh,
                'tau_m': self.tau_m,
                'R_m': self.R_m,
                'tau_syn': self.tau_syn
            }
        )
        self.rec_neurons.v = self.v_rest
        
        # Current-based recurrent synapses
        self.rec_synapses = Synapses(
            self.rec_neurons, 
            self.rec_neurons,
            'w_syn : amp',
            on_pre='I_syn_rec_post += w_syn'
        )
        self.rec_synapses.connect()
        self.rec_synapses.w_syn = self.weight_rec.T.flatten() * nA

        #poisson neurons
        self.poisson_neurons = PoissonGroup(self.N_poisson, rates=10*Hz)
        self.poiss_rec_synapses = Synapses(
            self.poisson_neurons, 
            self.rec_neurons,
            'w_poisson : amp',
            on_pre='I_syn_poiss_post += w_poisson'
        )
        self.poiss_rec_synapses.connect()
        self.poiss_rec_synapses.w_poisson = self.weight_poisson.T.flatten() * nA

        print(f"Network sizes - Rec: {self.N_rec} ({N_rec_E}E, {N_rec_I}I), Poisson: {self.N_poisson} ({N_poisson_E}E, {N_poisson_I}I)")

    def run_simulation(self, T, external_input_poisson, external_input_recurrent, stims_by_time, stim_info, current_injection, monitoring_level='minimal'):
        """
        Run the network simulation with configurable monitoring levels.
        
        Parameters:
        - monitoring_level: 'minimal', 'basic', or 'full'
            * 'minimal': Only spike monitors (fastest, recommended for long runs)
            * 'basic': Spikes + voltage from subset of neurons  
            * 'full': All data (slowest, only for detailed analysis)
        """
        print("\nStarting simulation...")
        print(f"Simulation time: {T/ms}ms")
        print(f"Monitoring level: {monitoring_level}")
        
        self.external_input_poisson = external_input_poisson
        self.external_input_recurrent = external_input_recurrent
        
        self.T = T
        self.current_injection = current_injection
        self.rec_input_rate = external_input_recurrent
        self.poisson_input_rate = external_input_poisson
        self.rec_neurons.I_ext = self.current_injection
        self.stim_type_vec = stims_by_time
        self.time_vec = self.T
        self.stim_info = stim_info

        # Essential monitors (always needed)
        self.rec_spikes = SpikeMonitor(self.rec_neurons)
        self.poisson_spikes = SpikeMonitor(self.poisson_neurons)
        
        # Configurable monitoring based on level
        self.monitoring_level = monitoring_level
        
        if monitoring_level == 'minimal':
            # Only spike data - fastest option
            print("Using minimal monitoring (spikes only) for maximum speed")
            self.rec_voltage = None
            self.I_syn_rec_mon = None
            self.I_syn_poiss_mon = None
            
        elif monitoring_level == 'basic':
            # Subset of neurons for voltage, no current monitoring
            subset_size = min(20, self.N_rec // 4)  # Monitor 25% of neurons, max 20
            subset_indices = np.random.choice(self.N_rec, subset_size, replace=False)
            print(f"Using basic monitoring (spikes + voltage from {subset_size} neurons)")
            
            self.rec_voltage = StateMonitor(self.rec_neurons, 'v', record=subset_indices)
            self.I_syn_rec_mon = None
            self.I_syn_poiss_mon = None
            
        elif monitoring_level == 'full':
            # All data - slowest but complete
            print("Using full monitoring (all data) - this will be slow for long simulations")
            self.rec_voltage = StateMonitor(self.rec_neurons, 'v', record=True)
            self.I_syn_rec_mon = StateMonitor(self.rec_neurons, 'I_syn_rec', record=True)
            self.I_syn_poiss_mon = StateMonitor(self.rec_neurons, 'I_syn_poiss', record=True)
        else:
            raise ValueError(f"Unknown monitoring level: {monitoring_level}")

        # Calculate epoch duration for progress tracking
        unique_stims = np.unique(stims_by_time)
        unique_stims = unique_stims[unique_stims != -1]
        epoch_duration = len(unique_stims) * (stim_info[0][2] - stim_info[0][1]) / ms
        current_epoch = 0
        last_epoch_time = 0

        # AGGRESSIVE OPTIMIZATION: Pre-compute all rate changes to eliminate runtime lookups
        print("Pre-computing stimulus changes for maximum simulation speed...")
        
        # Create lookup table for all rate changes 
        max_time_ms = int(T/ms)
        rate_lookup_table = {}
        last_rate_hash = None
        
        # Pre-compute unique rate vectors and their timestamps
        for t_ms in range(max_time_ms):
            if t_ms < len(stims_by_time):
                current_rate = external_input_poisson[:, t_ms]
                rate_hash = hash(current_rate.tobytes())  # Fast hash for comparison
                
                if rate_hash != last_rate_hash:
                    rate_lookup_table[t_ms] = current_rate * Hz
                    last_rate_hash = rate_hash
        
        # Convert to sorted list for efficient iteration
        rate_change_times = sorted(rate_lookup_table.keys())
        print(f"Pre-computed {len(rate_change_times)} rate changes (vs {max_time_ms} total timesteps)")
        
        # Pre-compute epoch boundaries
        epoch_boundaries = []
        for epoch in range(1, 50):  # Max reasonable epochs
            boundary = epoch * epoch_duration
            if boundary < max_time_ms:
                epoch_boundaries.append(int(boundary))
            else:
                break
        
        # Optimization: Use much less frequent updates when possible
        # For periods with no stimulus changes, we can update every 10ms instead of 1ms
        update_interval = 1*ms  # Start with 1ms, will be adaptive
        
        # ULTRA OPTIMIZATION: Create multiple network operations with different frequencies
        # This dramatically reduces function calls during stable periods
        next_change_time = rate_change_times[0] if rate_change_times else max_time_ms
        next_epoch_time = epoch_boundaries[0] if epoch_boundaries else max_time_ms
        
        # State variables for optimized loop
        rate_change_idx = 0
        epoch_boundary_idx = 0
        last_update_time = -1
        
        # Fast path for most timesteps (no changes)
        @network_operation(dt=1*ms)  # Still called every 1ms but ultra-fast
        def change_rates():
            nonlocal current_epoch, rate_change_idx, epoch_boundary_idx, last_update_time
            
            current_time_ms = int(defaultclock.t / ms)  # Single conversion, cached
            
            # Ultra-fast bailout: Check if we need to do anything at this timestep
            has_rate_change = (rate_change_idx < len(rate_change_times) and 
                              current_time_ms >= rate_change_times[rate_change_idx])
            has_epoch_change = (epoch_boundary_idx < len(epoch_boundaries) and 
                               current_time_ms >= epoch_boundaries[epoch_boundary_idx])
            
            if not (has_rate_change or has_epoch_change):
                return  # Exit immediately - no changes needed (ultra-fast path)
            
            # Only execute when we actually have changes
            if has_epoch_change:
                current_epoch += 1
                epoch_boundary_idx += 1
                print(f"Epoch {current_epoch} completed at {current_time_ms}ms")
            
            if has_rate_change:
                # Apply the pre-computed rate directly (no array operations)
                new_rates = rate_lookup_table[rate_change_times[rate_change_idx]]
                self.poisson_neurons.rates = new_rates
                rate_change_idx += 1

        # Build network with appropriate monitors
        self.net = Network()
        network_objects = [self.rec_neurons, self.poisson_neurons, self.rec_synapses, self.poiss_rec_synapses,
                          self.rec_spikes, self.poisson_spikes, change_rates]
        
        # Add optional monitors
        if self.rec_voltage is not None:
            network_objects.append(self.rec_voltage)
        if self.I_syn_rec_mon is not None:
            network_objects.append(self.I_syn_rec_mon)
        if self.I_syn_poiss_mon is not None:
            network_objects.append(self.I_syn_poiss_mon)
            
        self.net.add(*network_objects)
        
        print("Running network...")
        self.net.run(self.T)
        print("Simulation completed")
              
    def analyze_spikes(self, smoothing_window_size=50 * ms):
        # Convert spike monitor to a matrix
        spike_matrix_poiss = sau.spike_matrix_from_monitor(self.poisson_spikes, self.N_poisson, self.T, dt=defaultclock.dt)
        self.spike_matrix_poiss = spike_matrix_poiss
        
        spike_matrix_rec = sau.spike_matrix_from_monitor(self.rec_spikes, self.N_rec, self.T, dt=defaultclock.dt)
        self.spike_matrix_rec = spike_matrix_rec

        # Smooth the spike matrix
        window_size = smoothing_window_size  # 50 ms window for smoothing
        smoothed_rates_rec, smoothed_time_points = sau.smoothed_firing_rate(self.spike_matrix_rec, window_size, defaultclock.dt)
        smoothed_rates_poiss, smoothed_time_points = sau.smoothed_firing_rate(spike_matrix_poiss, window_size, defaultclock.dt)
        self.smoothed_rates_poiss = smoothed_rates_poiss
        self.smoothed_rates_rec = smoothed_rates_rec
        self.smoothed_time_points = smoothed_time_points

        # Get the sorting order based on similarity
        neuron_order = sau.sort_neurons_by_similarity(smoothed_rates_rec)
        self.neuron_order = neuron_order

        # Store neuron types
        self.neuron_types = np.array(['E'] * self.N_rec_E + ['I'] * self.N_rec_I)

        # Create neuron groups dictionary
        self.neuron_groups = {
            'E': np.arange(self.N_rec_E),
            'I': np.arange(self.N_rec_E, self.N_rec),
            'all': np.arange(self.N_rec)
        }
        
        # Calculate rate matrix per stimulus
        self.rate_matrix_rec = sau.gen_firing_rates_matrix(self.stim_type_vec, self.spike_matrix_rec)
        self.rate_matrix_poiss = sau.gen_firing_rates_matrix(self.stim_type_vec, self.spike_matrix_poiss)

        print("Spike analysis completed")

    def plot_aggregate_correlation(self, ax=None, window_size=50):
        """
        Plot the aggregate input correlation analysis.
        
        Parameters:
        - ax: matplotlib axis (if None, creates a new figure)
        - window_size: time window for spike counting (in timesteps)
        
        Returns:
        - matplotlib figure (if ax is None)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            return_fig = True
        else:
            return_fig = False
            
        pf.plot_aggregate_input_correlation(ax, self.spike_matrix_poiss, self.spike_matrix_rec, 
                                          self.stim_type_vec, self.N_poisson_E, self.N_poisson_I, 
                                          self.N_rec_E, self.N_rec_I, window_size=window_size)
        
        if return_fig:
            plt.tight_layout()
            return fig

    def save_spikes(self, filename):
        """
        Saves the spike data to a pickle file.
        :param filename:
        :return:
        """
        # Use os.path.join with SAVE_FOLDER if filename doesn't include a path
        if '/' not in filename and '\\' not in filename:
            filename = os.path.join(SAVE_FOLDER, filename)
            
        my_dict = {
            "stim_by_time": self.stim_type_vec,
            "spike_matrix_rec": self.spike_matrix_rec,
            "spike_matrix_poiss": self.spike_matrix_poiss,
            "N_poisson_E": self.N_poisson_E,
            "N_poisson_I": self.N_poisson_I,
            "T": self.T,
            "stim_info": self.stim_info,
            "N_rec_E": self.N_rec_I,
            "N_rec_I": self.N_rec_E,
            "neuron_order": self.neuron_order
        }
        with open(filename, "wb") as file:
            pickle.dump(my_dict, file)

    # why do we need external input for the recurrent, it's the poisson
    def plotting_dashboard(self):
        print("\nGenerating dashboard plots...")
        print("Creating subplot mosaic...")
        fig, axs = plt.subplot_mosaic([['A', 'B', 'C'], ['D', 'E', 'F'], ['G','H','I']], figsize = (18, 10))

        # poisson group
        print("Plotting poisson group panels (A, B, C)...")
        pf.plot_weight_matrix(axs['A'], self.weight_poisson, self.weight_poisson)
        pf.plot_external_stimuli(axs['B'], self.external_input_poisson)
        pf.plot_raster(axs['C'], self.poisson_spikes, self.N_poisson_E, self.N_poisson_I, self.T, stim_info = self.stim_info)

        # recurrent group
        print("Plotting recurrent group panels (D, E, F)...")
        print("Plotting weight matrix for recurrent group...")
        pf.plot_weight_matrix(axs['D'], self.weight_rec, self.weight_poisson)
        
        # Handle voltage plotting based on monitoring level
        if self.rec_voltage is not None:
            print("Plotting membrane potential for recurrent group...")
            pf.plot_membrane_potential(axs['E'], self.rec_voltage, self.T)
        else:
            print("Skipping membrane potential plot (not recorded with current monitoring level)")
            axs['E'].text(0.5, 0.5, 'Membrane potential\nnot recorded\n(minimal monitoring)', 
                         ha='center', va='center', transform=axs['E'].transAxes, fontsize=12)
            axs['E'].set_title('Membrane Potential (Not Recorded)')
        
        print("Plotting raster plot for recurrent group...")
        pf.plot_raster(axs['F'], self.rec_spikes, self.N_rec_E, self.N_rec_I, self.T, neuron_order = self.neuron_order, stim_info = self.stim_info)
        
        print("Plotting analysis panels (G, H, I)...")
        pf.plot_firing_rates_matrix(self.spike_matrix_rec, self.stim_type_vec, self.neuron_order, ax = axs['G'])
        # pf.plot_average_firing_rates(self.smoothed_rates_rec, self.stims_by_time, self.neuron_groups, ax=axs['H'])


        # Call the new aggregate input correlation function instead of the old rate correlation
        pf.plot_aggregate_input_correlation(axs['H'], self.spike_matrix_poiss, self.spike_matrix_rec, 
                                          self.stim_type_vec, self.N_poisson_E, self.N_poisson_I, 
                                          self.N_rec_E, self.N_rec_I, window_size=50)
        
        # Create a simplified current analysis or skip it
        if self.I_syn_rec_mon is not None and self.I_syn_poiss_mon is not None:
            # Original current plotting code
            print("Plotting current analysis...")
            ax1 = plt.subplot(3, 3, 9)
            total_E = np.mean(self.I_syn_rec_mon.I_syn_rec[:self.N_rec_E] + 
                             self.I_syn_poiss_mon.I_syn_poiss[:self.N_rec_E], axis=0)/nA
            total_I = np.mean(self.I_syn_rec_mon.I_syn_rec[self.N_rec_E:] + 
                             self.I_syn_poiss_mon.I_syn_poiss[self.N_rec_E:], axis=0)/nA
            
            ax1.plot(self.I_syn_rec_mon.t/ms, total_E, label='E population', color='blue')
            ax1.plot(self.I_syn_rec_mon.t/ms, total_I, label='I population', color='red')
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Total Current (nA)')
            ax1.set_title('Total Current by Population')
            ax1.legend()
        else:
            print("Skipping current analysis (not recorded with current monitoring level)")
            axs['I'].text(0.5, 0.5, 'Current data\nnot recorded\n(minimal monitoring)', 
                         ha='center', va='center', transform=axs['I'].transAxes, fontsize=12)
            axs['I'].set_title('Current Analysis (Not Recorded)')

        plt.tight_layout()
        return fig







