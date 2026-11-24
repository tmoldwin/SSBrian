from brian2 import *
import matplotlib.pyplot as plt
prefs.codegen.target = "numpy"
from brian_experiment.shared.core import spike_analysis_utils as sau
from brian_experiment.shared.core import stimuli as stim
from brian_experiment.shared.core import weight_matrices as wm
from brian_experiment.shared.plotting import plots as pf
import numpy as np
import dill as pickle
from brian_experiment.shared.core.constants import SAVE_FOLDER
import os


class DirectInputNetwork:
    def __init__(self, N_rec_E, N_rec_I, N_external_input_dims, weight_rec, input_weight_scale=1.0, lognormal_params_in=None, is_feedforward=False):
        """
        Initialize DirectInputNetwork with direct input vector presentation.
        
        Parameters:
        - N_rec_E, N_rec_I: Number of excitatory and inhibitory recurrent neurons
        - N_external_input_dims: Number of excitatory external input dimensions
        - weight_rec: Recurrent weight matrix
        - input_weight_scale: Scaling factor for input weights
        - is_feedforward: If True, adjust neuron parameters to compensate for lack of recurrent inhibition
        """
        print("Initializing DirectInputNetwork...")
        self.net = None
        start_scope()
        
        # Network dimensions
        self.N_rec_E = N_rec_E
        self.N_rec_I = N_rec_I
        self.N_external_input_dims = N_external_input_dims
        self.weight_rec = weight_rec
        self.input_weight_scale = input_weight_scale
        self.N_rec = self.N_rec_I + self.N_rec_E
        self.lognormal_params_in = lognormal_params_in
        self.is_feedforward = is_feedforward
        
        # Initialize monitors as None
        self.rec_current = None
        self.rec_voltage = None
        self.rec_current_E = None
        self.rec_current_I = None
        self.input_current = None

        # Biological parameters (same as haim_network)
        self.tau_m = 20*ms       # Membrane time constant
        self.tau_syn = 1*ms      # Synaptic time constant
        self.R_m = 100*Mohm      # Membrane resistance
        
        # Adjust parameters for feedforward mode to reduce firing rates
        if is_feedforward:
            self.v_rest = -115*mV    # Very low resting potential (more negative)
            self.v_thresh = -55*mV   # Keep normal threshold
            print(f"Feedforward mode: adjusted v_rest={self.v_rest/mV}mV (lowered from -65mV)")
        else:
            self.v_rest = -65*mV     # Normal resting potential
            self.v_thresh = -55*mV   # Normal threshold
        
        # Derived parameter
        self.C_m = self.tau_m/self.R_m  # Membrane capacitance

        # Generate input weight matrix: each neuron has weights for all input dimensions
        # All input dimensions are excitatory, so all weights are positive
        self.input_weights = self._generate_input_weights()
        
        # Simplified LIF voltage equation with separate excitatory and inhibitory currents
        self.eqs = '''
        dv/dt = (v_rest - v) / tau_m + (I_input + I_syn_rec_E + I_syn_rec_I) / C_m : volt (unless refractory)
        dI_syn_rec_E/dt = -I_syn_rec_E/tau_syn : amp
        dI_syn_rec_I/dt = -I_syn_rec_I/tau_syn : amp
        I_input : amp
        '''

        # Recurrent neurons
        self.rec_neurons = NeuronGroup(
            self.N_rec, 
            self.eqs, 
            threshold='v > v_thresh',
            reset='v = v_rest', 
            refractory=2*ms,
            method='euler',
            namespace={
                'v_rest': self.v_rest,
                'v_thresh': self.v_thresh,
                'tau_m': self.tau_m,
                'C_m': self.C_m,
                'tau_syn': self.tau_syn
            }
        )
        self.rec_neurons.v = self.v_rest
        self.rec_neurons.I_input = 0*nA  # Initialize input current
        
        # Apply Dale's law correctly to recurrent weights
        self.weight_rec = self._apply_dales_law(self.weight_rec)
        
        # Current-based recurrent synapses - separate for E and I sources
        # Excitatory recurrent synapses (from E neurons)
        self.rec_synapses_E = Synapses(
            self.rec_neurons[:self.N_rec_E],  # Only E neurons as sources
            self.rec_neurons,
            'w_syn : amp',
            on_pre='I_syn_rec_E_post += w_syn'
        )
        self.rec_synapses_E.connect()
        # Extract E->all weights from weight matrix (first N_rec_E rows)
        weight_E_to_all = self.weight_rec[:self.N_rec_E, :].flatten()
        self.rec_synapses_E.w_syn = weight_E_to_all * nA
        
        # Inhibitory recurrent synapses (from I neurons)
        self.rec_synapses_I = Synapses(
            self.rec_neurons[self.N_rec_E:],  # Only I neurons as sources
            self.rec_neurons,
            'w_syn : amp',
            on_pre='I_syn_rec_I_post += w_syn'
        )
        self.rec_synapses_I.connect()
        # Extract I->all weights from weight matrix (last N_rec_I rows)
        weight_I_to_all = self.weight_rec[self.N_rec_E:, :].flatten()
        self.rec_synapses_I.w_syn = weight_I_to_all * nA

        print(f"Network sizes - Rec: {self.N_rec} ({N_rec_E}E, {N_rec_I}I), Input dims: {self.N_external_input_dims}E")
        print(f"Input weight matrix shape: {self.input_weights.shape}")

    def _generate_input_weights(self):
        """
        Generate input weight matrix for input neurons.
        All input dimensions are excitatory, so all weights are positive.
        """
        n_inputs = self.N_external_input_dims
        weights = np.zeros((self.N_rec, n_inputs))
        # All neurons (E and I) get positive weights from E input channels
        weights[:, :] = np.abs(np.random.normal(0, 1, (self.N_rec, n_inputs)))
        weights *= self.input_weight_scale
        return weights

    def _apply_dales_law(self, weight_matrix):
        """
        Apply Dale's law to weight matrix: excitatory neurons (sources) have positive outgoing weights,
        inhibitory neurons (sources) have negative outgoing weights.
        
        Parameters:
        - weight_matrix: [N_rec x N_rec] matrix where rows are source neurons, columns are target neurons
        
        Returns:
        - weight_matrix with Dale's law applied
        """
        corrected_weights = weight_matrix.copy()
        
        # Excitatory source neurons (first N_rec_E rows) have positive outgoing weights
        corrected_weights[:self.N_rec_E, :] = np.abs(corrected_weights[:self.N_rec_E, :])
        
        # Inhibitory source neurons (last N_rec_I rows) have negative outgoing weights  
        corrected_weights[self.N_rec_E:, :] = -np.abs(corrected_weights[self.N_rec_E:, :])
        
        return corrected_weights

    def run_simulation(self, T, input_vectors, stims_by_time, stim_info, current_injection, monitoring_level='minimal', unitary_current=1.0*nA, num_epochs=1):
        """
        Run the network simulation with direct input vector presentation.
        
        Parameters:
        - T: Simulation time
        - input_vectors: Input vectors over time [N_external_input_dims x time_steps]
        - stims_by_time: Stimulus labels over time
        - stim_info: Stimulus information
        - current_injection: External current injection
        - monitoring_level: 'minimal', 'basic', or 'full'
        - unitary_current: The strength of a single input event in probabilistic mode.
        """
        print("\nStarting simulation...")
        print(f"Simulation time: {T/ms}ms")
        print(f"Monitoring level: {monitoring_level}")
        print(f"Input vector shape: {input_vectors.shape}")
        
        self.input_vectors = input_vectors
        self.T = T
        self.current_injection = current_injection
        self.stim_type_vec = stims_by_time
        self.time_vec = self.T
        self.stim_info = stim_info

        # Essential monitors (always needed)
        self.rec_spikes = SpikeMonitor(self.rec_neurons)
        # NOTE: This network uses direct input (currents), not Poisson input neurons
        # So there are no input spikes to monitor
        self.input_spikes = None
        
        # Configurable monitoring based on level
        self.monitoring_level = monitoring_level
        
        if monitoring_level == 'minimal':
            print("Using minimal monitoring (spikes only) for maximum speed")
            self.rec_voltage = None
            self.rec_current_E = None
            self.rec_current_I = None
            self.input_current = None
            
        elif monitoring_level == 'basic':
            # Monitor ALL neurons for voltage to get accurate population means
            # This is necessary for meaningful mean voltage traces
            subset_indices = True  # True = record all neurons
            subset_size = self.N_rec
            print(f"Using basic monitoring (spikes + voltage + currents from ALL {subset_size} neurons)")
            
            self.rec_voltage = StateMonitor(self.rec_neurons, 'v', record=subset_indices)
            self.rec_current_E = StateMonitor(self.rec_neurons, 'I_syn_rec_E', record=subset_indices, dt=10*ms)
            self.rec_current_I = StateMonitor(self.rec_neurons, 'I_syn_rec_I', record=subset_indices, dt=10*ms)
            self.input_current = StateMonitor(self.rec_neurons, 'I_input', record=subset_indices, dt=10*ms)
            
        elif monitoring_level == 'full':
            print("Using full monitoring (all data) - this will be slow for long simulations")
            self.rec_voltage = StateMonitor(self.rec_neurons, 'v', record=True)
            self.rec_current_E = StateMonitor(self.rec_neurons, 'I_syn_rec_E', record=True)
            self.rec_current_I = StateMonitor(self.rec_neurons, 'I_syn_rec_I', record=True)
            self.input_current = StateMonitor(self.rec_neurons, 'I_input', record=True)
        else:
            raise ValueError(f"Unknown monitoring level: {monitoring_level}")

        # Apply input vectors at each simulation time step
        print("Applying input vectors at each timestep...")

        # Track stimulus progress for printouts
        last_stim = -1
        last_print_time = 0
        last_epoch = -1
        
        @network_operation(dt=defaultclock.dt)
        def update_input_currents():
            """
            Update the input current at each time step based on the pre-generated
            input_vectors array. The index into the array corresponds to the
            current simulation time step.
            """
            nonlocal last_stim, last_print_time, last_epoch
            current_step = int(defaultclock.t / defaultclock.dt)
            current_time = defaultclock.t / ms

            if current_step < input_vectors.shape[1]:
                # Apply the pre-calculated input vector for the current time step
                current_input = input_vectors[:, current_step]
                raw_currents = np.dot(self.input_weights, current_input)
                self.rec_neurons.I_input = raw_currents * unitary_current
                
                # Check if stimulus changed and print progress
                current_stim = stims_by_time[current_step] if current_step < len(stims_by_time) else -1
                if current_stim != last_stim and current_stim != -1:
                    last_stim = current_stim
                    
                    # Calculate current epoch based on stimulus presentation
                    unique_stims = np.unique(stims_by_time[stims_by_time != -1])
                    total_stim_per_epoch = len(unique_stims)
                    current_epoch = (current_stim // total_stim_per_epoch) + 1 if total_stim_per_epoch > 0 else 1
                    
                    # Print progress with epoch information
                    if current_time - last_print_time > 5000:  # 5 seconds in ms
                        print(f">> Time: {current_time:.1f}ms - Epoch {current_epoch}/{num_epochs} - Stimulus {current_stim}/{total_stim_per_epoch-1}")
                        last_print_time = current_time
            else:
                # This case is unlikely if T is consistent, but good for safety
                self.rec_neurons.I_input = 0 * nA

        # Build network with appropriate monitors
        self.net = Network()
        network_objects = [self.rec_neurons, self.rec_synapses_E, self.rec_synapses_I, 
                          self.rec_spikes, update_input_currents]
        
        # Add optional monitors based on the monitoring level
        if self.rec_voltage is not None:
            network_objects.append(self.rec_voltage)
        if self.rec_current_E is not None:
            network_objects.append(self.rec_current_E)
        if self.rec_current_I is not None:
            network_objects.append(self.rec_current_I)
        if self.input_current is not None:
            network_objects.append(self.input_current)
            
        self.net.add(*network_objects)
        
        print("Running network...")
        self.net.run(self.T)
        
        # Count total stimuli presented
        unique_stims = np.unique(stims_by_time[stims_by_time != -1])
        total_stimuli_presented = len(unique_stims) * num_epochs
        print(f">> Simulation completed successfully!")
        print(f"   Processed {total_stimuli_presented} stimuli across {num_epochs} epochs")
              
    def analyze_spikes(self, smoothing_window_size=50 * ms):
        """Analyze spike data (similar to haim_network but without Poisson analysis)"""
        spike_matrix_rec = sau.spike_matrix_from_monitor(self.rec_spikes, self.N_rec, self.T, dt=defaultclock.dt)
        self.spike_matrix_rec = spike_matrix_rec

        # Smooth the spike matrix
        window_size = smoothing_window_size
        dt_seconds = float(defaultclock.dt / second)  # Convert Brian2 time unit to seconds
        smoothed_rates_rec, smoothed_time_points = sau.smoothed_firing_rate(self.spike_matrix_rec, window_size, dt_seconds)
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

        print("Spike analysis completed")

    def save_spikes(self, filename):
        """Save spike data to pickle file"""
        if '/' not in filename and '\\' not in filename:
            filename = os.path.join(SAVE_FOLDER, filename)
            
        my_dict = {
            "stim_by_time": self.stim_type_vec,
            "spike_matrix_rec": self.spike_matrix_rec,
            "input_vectors": self.input_vectors,
            "input_weights": self.input_weights,
            "N_external_input_dims": self.N_external_input_dims,
            "T": self.T,
            "stim_info": self.stim_info,
            "N_rec_E": self.N_rec_E,
            "N_rec_I": self.N_rec_I,
            "neuron_order": self.neuron_order
        }
        with open(filename, "wb") as file:
            pickle.dump(my_dict, file)

    def plotting_dashboard_with_subsampling(self, max_stimuli=10, max_neurons_voltage=20, max_neurons_raster=100, num_epochs=1):
        """
        Generate dashboard plots with subsampling for manageable visualization.
        Each row shows one epoch with the same subsampled stimuli and neurons.
        
        Parameters:
        -----------
        max_stimuli : int
            Maximum number of stimuli to plot
        max_neurons : int
            Maximum number of neurons to show in voltage traces
        num_epochs : int
            Number of epochs in the simulation
        """
        print("\n" + "="*60)
        print("GENERATING VOLTAGE/CURRENT DYNAMICS DASHBOARD WITH SUBSAMPLING")
        print("="*60)
        
        # Calculate epoch duration and stimulus timing
        unique_stims = np.unique(self.stim_type_vec)
        unique_stims = unique_stims[unique_stims != -1]
        n_stimuli_total = len(unique_stims)
        
        # Get stimulus timing info
        stim_duration_ms = 100  # ms per stimulus
        break_time_ms = 20      # ms break between stimuli
        epoch_duration_ms = n_stimuli_total * (stim_duration_ms + break_time_ms)
        
        print(f"\n>> Simulation structure:")
        print(f"   Total stimuli: {n_stimuli_total}")
        print(f"   Total epochs: {num_epochs}")
        print(f"   Epoch duration: {epoch_duration_ms} ms")
        print(f"   Total neurons: {self.N_rec}")
        print(f"   Monitored neurons: {len(self.rec_voltage.record) if self.rec_voltage is not None else 0}")
        
        # SUBSAMPLING STEP 1: Select stimuli to plot (FIRST N, not random)
        if n_stimuli_total <= max_stimuli:
            selected_stim_indices = np.arange(n_stimuli_total)
            print(f"\n   Plotting all {n_stimuli_total} stimuli (under max of {max_stimuli})")
        else:
            # Take FIRST max_stimuli (not random, not evenly spaced)
            selected_stim_indices = np.arange(max_stimuli)
            print(f"\n   SUBSAMPLING: Plotting FIRST {max_stimuli} out of {n_stimuli_total} stimuli")
        
        selected_stim_ids = unique_stims[selected_stim_indices]
        print(f"   Selected stimulus IDs: {selected_stim_ids}")
        
        # SUBSAMPLING STEP 2A: Select neurons for VOLTAGE TRACES (small subset from monitored)
        if self.rec_voltage is not None:
            monitored_neurons = np.array(self.rec_voltage.record)
            n_monitored = len(monitored_neurons)
            
            if n_monitored <= max_neurons_voltage:
                selected_voltage_indices = np.arange(n_monitored)
                selected_voltage_ids = monitored_neurons
                print(f"\n   Plotting all {n_monitored} monitored neurons for voltage traces")
            else:
                np.random.seed(42)
                selected_voltage_indices = np.sort(np.random.choice(n_monitored, max_neurons_voltage, replace=False))
                selected_voltage_ids = monitored_neurons[selected_voltage_indices]
                print(f"\n   VOLTAGE TRACES: Plotting {max_neurons_voltage} out of {n_monitored} monitored neurons")
            
            n_e_volt = np.sum(selected_voltage_ids < self.N_rec_E)
            n_i_volt = len(selected_voltage_ids) - n_e_volt
            print(f"   Voltage trace neuron IDs: {selected_voltage_ids}")
            print(f"   E neurons: {n_e_volt}, I neurons: {n_i_volt}")
        else:
            selected_voltage_indices = []
            selected_voltage_ids = []
            print("\nWARNING: No voltage data recorded")
        
        # SUBSAMPLING STEP 2B: Select neurons for RASTER (larger subset from all neurons)
        # For raster, we use spike data which includes ALL neurons, not just monitored
        n_total_neurons = self.N_rec
        if n_total_neurons <= max_neurons_raster:
            selected_raster_ids = np.arange(n_total_neurons)
            print(f"\n   RASTER: Plotting all {n_total_neurons} neurons")
        else:
            np.random.seed(43)  # Different seed for different selection
            selected_raster_ids = np.sort(np.random.choice(n_total_neurons, max_neurons_raster, replace=False))
            print(f"\n   RASTER: Plotting {max_neurons_raster} out of {n_total_neurons} neurons")
        
        n_e_raster = np.sum(selected_raster_ids < self.N_rec_E)
        n_i_raster = len(selected_raster_ids) - n_e_raster
        print(f"   Raster neuron count: {len(selected_raster_ids)} (E: {n_e_raster}, I: {n_i_raster})")
        
        # Identify top 5 selective neurons AMONG THE SELECTED RASTER NEURONS
        # Filter by minimum spike count first
        MIN_SPIKE_COUNT = 10  # Minimum total spike count across simulation
        if hasattr(self, 'rate_matrix_rec') and self.rate_matrix_rec is not None:
            import sparsity_and_selectivity as ss
            _, sli_arr = ss.create_vecs(self.rate_matrix_rec)
            sli_arr = np.array(sli_arr)
            
            # Calculate total spike counts for each neuron
            total_spike_counts = np.sum(self.spike_matrix_rec, axis=1)  # Total spikes per neuron
            
            # Get selectivity and spike counts for neurons in the raster plot
            raster_sli = sli_arr[selected_raster_ids]
            raster_spike_counts = total_spike_counts[selected_raster_ids]
            
            # Filter to neurons above minimum spike count
            active_mask = raster_spike_counts >= MIN_SPIKE_COUNT
            
            if np.sum(active_mask) >= 5:
                # Get indices of active neurons
                active_indices = np.where(active_mask)[0]
                # Get selectivity of active neurons
                active_sli = raster_sli[active_indices]
                # Get top 5 by selectivity among active neurons
                top_5_among_active = np.argsort(active_sli)[-5:][::-1]
                top_5_in_raster = active_indices[top_5_among_active]
            else:
                # If not enough active neurons, just take top 5 by selectivity
                print(f"      Warning: Only {np.sum(active_mask)} neurons with >={MIN_SPIKE_COUNT} spikes")
                top_5_in_raster = np.argsort(raster_sli)[-5:][::-1]
            
            # Map back to actual neuron IDs
            self.highly_selective_neurons_in_raster = selected_raster_ids[top_5_in_raster]
            selected_spike_counts = total_spike_counts[self.highly_selective_neurons_in_raster]
            print(f"   Top 5 selective neurons in raster: {self.highly_selective_neurons_in_raster}")
            print(f"      SLI: {sli_arr[self.highly_selective_neurons_in_raster]}")
            print(f"      Total spike counts: {selected_spike_counts}")
        else:
            self.highly_selective_neurons_in_raster = None
        
        # CREATE FIGURE: num_epochs rows × 4 columns
        # Columns: [Input Stimuli | Individual Voltage Traces | Mean Voltage (E/I/Net) | Raster Plot]
        fig, axs = plt.subplots(num_epochs, 4, figsize=(24, 4*num_epochs))
        if num_epochs == 1:
            axs = axs.reshape(1, -1)  # Ensure 2D array even for single epoch
        
        print(f"\n   Figure layout: {num_epochs} epochs x 4 columns")
        
        # Plot each epoch as a separate row
        dt_ms = float(defaultclock.dt / ms)
        
        for epoch_idx in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"PLOTTING EPOCH {epoch_idx + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Calculate FULL epoch time window
            epoch_start_ms = epoch_idx * epoch_duration_ms + 100  # +100 for start_time
            epoch_end_ms = epoch_start_ms + epoch_duration_ms
            
            # Calculate SELECTED STIMULI time window (only the stimuli we're plotting)
            first_stim_id = selected_stim_ids[0]
            last_stim_id = selected_stim_ids[-1]
            plot_start_ms = epoch_start_ms + first_stim_id * (stim_duration_ms + break_time_ms)
            plot_end_ms = epoch_start_ms + (last_stim_id + 1) * (stim_duration_ms + break_time_ms)
            
            print(f"   Epoch time window: [{epoch_start_ms:.1f}, {epoch_end_ms:.1f}] ms")
            print(f"   Plotting stimuli {int(first_stim_id)}-{int(last_stim_id)}: [{plot_start_ms:.1f}, {plot_end_ms:.1f}] ms")
            
            # COLUMN 1: Input stimuli raster
            ax_input = axs[epoch_idx, 0]
            self._plot_input_raster_subsampled(
                ax_input, epoch_idx, plot_start_ms, plot_end_ms, epoch_start_ms,
                selected_stim_ids, stim_duration_ms, break_time_ms
            )
            
            # COLUMN 2: Individual voltage traces for selected neurons
            ax_voltage = axs[epoch_idx, 1]
            self._plot_voltage_traces_subsampled(
                ax_voltage, epoch_idx, plot_start_ms, plot_end_ms, epoch_start_ms,
                selected_stim_ids, selected_voltage_indices, selected_voltage_ids,
                stim_duration_ms, break_time_ms
            )
            
            # COLUMN 3: Mean voltage traces (E/I/Net) - AVERAGE OVER ALL MONITORED NEURONS
            ax_mean_voltage = axs[epoch_idx, 2]
            self._plot_mean_voltage_traces(
                ax_mean_voltage, epoch_idx, plot_start_ms, plot_end_ms, epoch_start_ms,
                selected_stim_ids, monitored_neurons,  # ALL monitored neurons for mean
                stim_duration_ms, break_time_ms
            )
            
            # COLUMN 4: Raster plot for larger selection of neurons
            ax_raster = axs[epoch_idx, 3]
            self._plot_raster_subsampled(
                ax_raster, epoch_idx, plot_start_ms, plot_end_ms, epoch_start_ms,
                selected_stim_ids, selected_raster_ids,
                stim_duration_ms, break_time_ms
            )
            
            # Add epoch label on left
            axs[epoch_idx, 0].set_ylabel(f'Epoch {epoch_idx + 1}', fontsize=14, fontweight='bold')
        
        # Add column titles at the top
        axs[0, 0].set_title('Input Vectors\n(Direct Currents)', fontsize=12, fontweight='bold')
        axs[0, 1].set_title('Voltage Traces\n(Selected Neurons)', fontsize=12, fontweight='bold')
        axs[0, 2].set_title('Mean Voltage\n(E / I / Net)', fontsize=12, fontweight='bold')
        axs[0, 3].set_title('Raster Plot\n(Selected Neurons)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        print(f"\n>> Dashboard generation complete!")
        return fig
    
    def _plot_input_raster_subsampled(self, ax, epoch_idx, plot_start_ms, plot_end_ms, epoch_start_ms,
                                      selected_stim_ids, stim_duration_ms, break_time_ms):
        """Plot input vectors (heatmap) for selected stimuli in this epoch."""
        if not hasattr(self, 'input_vectors') or self.input_vectors is None:
            ax.text(0.5, 0.5, 'No input data', ha='center', va='center', transform=ax.transAxes)
            return
        
        dt_ms = float(defaultclock.dt / ms)
        
        # Convert time window to indices
        t_start_idx = int(plot_start_ms / dt_ms)
        t_end_idx = int(plot_end_ms / dt_ms)
        
        # Extract input vectors for this time window
        input_window = self.input_vectors[:, t_start_idx:t_end_idx]
        
        # Plot as heatmap
        n_input_dims = input_window.shape[0]
        time_axis = np.linspace(0, plot_end_ms - plot_start_ms, input_window.shape[1])
        im = ax.imshow(input_window, aspect='auto', cmap='Greens', origin='lower',
                      extent=[0, plot_end_ms - plot_start_ms, 0, n_input_dims],
                      interpolation='nearest', vmin=0, vmax=np.max(input_window) if np.max(input_window) > 0 else 1)
        
        # Add stimulus boundaries for SELECTED stimuli
        for stim_id in selected_stim_ids:
            stim_start_absolute = epoch_start_ms + stim_id * (stim_duration_ms + break_time_ms)
            stim_start_relative = stim_start_absolute - plot_start_ms
            ax.axvline(stim_start_relative, color='lightgray', linestyle='--', alpha=0.7, linewidth=1.2)
            ax.text(stim_start_relative, n_input_dims - 1, f'{int(stim_id)}',
                   rotation=90, verticalalignment='top', horizontalalignment='right',
                   fontsize=10, color='gray', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Input Dimension')
        ax.set_xlim(0, plot_end_ms - plot_start_ms)
        ax.set_ylim(0, n_input_dims)
        ax.grid(False)
    
    def _plot_mean_voltage_traces(self, ax, epoch_idx, plot_start_ms, plot_end_ms, epoch_start_ms,
                                   selected_stim_ids, monitored_neuron_ids,
                                   stim_duration_ms, break_time_ms):
        """Plot mean voltage traces for E population, I population, and Net (averaged over ALL monitored neurons)."""
        if self.rec_voltage is None or len(monitored_neuron_ids) == 0:
            ax.text(0.5, 0.5, 'No voltage data', ha='center', va='center', transform=ax.transAxes)
            return
        
        dt_ms = float(defaultclock.dt / ms)
        t_start_idx = int(plot_start_ms / dt_ms)
        t_end_idx = int(plot_end_ms / dt_ms)
        
        time_window = self.rec_voltage.t[t_start_idx:t_end_idx] / ms - plot_start_ms
        
        # Get ALL monitored neurons and separate into E and I
        e_mask = monitored_neuron_ids < self.N_rec_E
        i_mask = monitored_neuron_ids >= self.N_rec_E
        
        # Get indices in the monitor for E and I neurons
        e_neuron_indices = np.where(e_mask)[0]
        i_neuron_indices = np.where(i_mask)[0]
        
        # Compute mean voltage traces over ALL monitored neurons
        if len(e_neuron_indices) > 0:
            mean_voltage_e = np.mean(self.rec_voltage.v[e_neuron_indices, t_start_idx:t_end_idx], axis=0) / mV
            ax.plot(time_window, mean_voltage_e, color='red', linewidth=2, label=f'E (n={len(e_neuron_indices)})', alpha=0.9)
        
        if len(i_neuron_indices) > 0:
            mean_voltage_i = np.mean(self.rec_voltage.v[i_neuron_indices, t_start_idx:t_end_idx], axis=0) / mV
            ax.plot(time_window, mean_voltage_i, color='blue', linewidth=2, label=f'I (n={len(i_neuron_indices)})', alpha=0.9)
        
        if len(e_neuron_indices) > 0 and len(i_neuron_indices) > 0:
            # Net average over ALL monitored neurons
            all_indices = np.arange(len(monitored_neuron_ids))
            mean_voltage_net = np.mean(self.rec_voltage.v[all_indices, t_start_idx:t_end_idx], axis=0) / mV
            ax.plot(time_window, mean_voltage_net, color='black', linewidth=2, label=f'Net (n={len(monitored_neuron_ids)})', alpha=0.7, linestyle='--')
        
        # Add stimulus boundaries for SELECTED stimuli
        for stim_id in selected_stim_ids:
            stim_start_absolute = epoch_start_ms + stim_id * (stim_duration_ms + break_time_ms)
            stim_start_relative = stim_start_absolute - plot_start_ms
            ax.axvline(stim_start_relative, color='lightgray', linestyle='--', alpha=0.7, linewidth=1.2)
            ax.text(stim_start_relative, ax.get_ylim()[1], f'{int(stim_id)}',
                   rotation=90, verticalalignment='top', horizontalalignment='right',
                   fontsize=10, color='gray', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Mean Membrane Potential (mV)')
        ax.set_xlim(0, plot_end_ms - plot_start_ms)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(False)
    
    def _plot_voltage_traces_subsampled(self, ax, epoch_idx, plot_start_ms, plot_end_ms, epoch_start_ms,
                                        selected_stim_ids, selected_neuron_indices, selected_neuron_ids,
                                        stim_duration_ms, break_time_ms):
        """Plot voltage traces for selected neurons during selected stimuli in this epoch."""
        if self.rec_voltage is None or len(selected_neuron_ids) == 0:
            ax.text(0.5, 0.5, 'No voltage data', ha='center', va='center', transform=ax.transAxes)
            return
        
        dt_ms = float(defaultclock.dt / ms)
        n_e_selected = np.sum(selected_neuron_ids < self.N_rec_E)
        n_i_selected = len(selected_neuron_ids) - n_e_selected
        
        colors_e = plt.cm.Reds(np.linspace(0.4, 0.9, max(n_e_selected, 1)))
        colors_i = plt.cm.Blues(np.linspace(0.4, 0.9, max(n_i_selected, 1)))
        
        e_idx = 0
        i_idx = 0
        
        # Plot voltage traces for the SELECTED STIMULI time window only
        t_start_idx = int(plot_start_ms / dt_ms)
        t_end_idx = int(plot_end_ms / dt_ms)
        
        time_window = self.rec_voltage.t[t_start_idx:t_end_idx] / ms - plot_start_ms
        
        # Plot traces for selected neurons across the whole epoch
        for neuron_monitor_idx, neuron_id in zip(selected_neuron_indices, selected_neuron_ids):
            voltage = self.rec_voltage.v[neuron_monitor_idx, t_start_idx:t_end_idx] / mV
            
            if neuron_id < self.N_rec_E:
                color = colors_e[e_idx % len(colors_e)]
                label = f'E{neuron_id}'
                e_idx += 1
            else:
                color = colors_i[i_idx % len(colors_i)]
                label = f'I{neuron_id}'
                i_idx += 1
            
            ax.plot(time_window, voltage, color=color, alpha=0.7, linewidth=0.8, label=label)
        
        # Mark selected stimulus boundaries with vertical lines and labels
        for stim_id in selected_stim_ids:
            # Calculate when this stimulus starts (absolute then relative to plot window)
            stim_start_absolute = epoch_start_ms + stim_id * (stim_duration_ms + break_time_ms)
            stim_start_relative = stim_start_absolute - plot_start_ms
            
            # Draw vertical line at stimulus start
            ax.axvline(stim_start_relative, color='lightgray', linestyle='--', alpha=0.7, linewidth=1.2)
            
            # Add stimulus ID label at the top
            ax.text(stim_start_relative, ax.get_ylim()[1], f'{int(stim_id)}',
                   rotation=90, verticalalignment='top', horizontalalignment='right',
                   fontsize=10, color='gray', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Membrane Potential (mV)')
        ax.set_xlim(0, plot_end_ms - plot_start_ms)
        if epoch_idx == 0 and len(selected_neuron_ids) <= 15:
            ax.legend(fontsize=6, ncol=2, loc='upper right')
        ax.grid(False)
    
    def _plot_raster_subsampled(self, ax, epoch_idx, plot_start_ms, plot_end_ms, epoch_start_ms,
                                selected_stim_ids, selected_neuron_ids,
                                stim_duration_ms, break_time_ms):
        """Plot raster for selected neurons in this epoch."""
        spike_times = self.rec_spikes.t / ms
        spike_neurons = self.rec_spikes.i
        
        # Filter spikes in SELECTED STIMULI time window and from selected neurons
        window_mask = (spike_times >= plot_start_ms) & (spike_times < plot_end_ms)
        neuron_mask = np.isin(spike_neurons, selected_neuron_ids)
        mask = window_mask & neuron_mask
        
        filtered_times = spike_times[mask] - plot_start_ms
        filtered_neurons = spike_neurons[mask]
        
        # Remap neuron IDs to y-axis positions
        neuron_to_ypos = {nid: idx for idx, nid in enumerate(selected_neuron_ids)}
        y_positions = [neuron_to_ypos[nid] for nid in filtered_neurons]
        
        # Color by E/I
        colors = ['red' if nid < self.N_rec_E else 'blue' for nid in filtered_neurons]
        
        ax.scatter(filtered_times, y_positions, s=1, c=colors, alpha=0.6, rasterized=True)
        
        # Highlight top 5 selective neurons with different colors
        if hasattr(self, 'highly_selective_neurons_in_raster') and self.highly_selective_neurons_in_raster is not None:
            colors = ['magenta', 'cyan', 'yellow', 'lime', 'orange']
            for idx, neuron_id in enumerate(self.highly_selective_neurons_in_raster):
                if neuron_id in neuron_to_ypos:
                    y_pos = neuron_to_ypos[neuron_id]
                    # Draw a tighter box around this neuron's row
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((0, y_pos - 0.7), plot_end_ms - plot_start_ms, 1.4,
                                   linewidth=0.8, edgecolor=colors[idx], facecolor='none', alpha=0.8)
                    ax.add_patch(rect)
        
        # Add stimulus boundaries for SELECTED stimuli only
        for stim_id in selected_stim_ids:
            # Calculate when this stimulus starts (absolute then relative to plot window)
            stim_start_absolute = epoch_start_ms + stim_id * (stim_duration_ms + break_time_ms)
            stim_start_relative = stim_start_absolute - plot_start_ms
            
            # Draw vertical line at stimulus start
            ax.axvline(stim_start_relative, color='lightgray', linestyle='--', alpha=0.7, linewidth=1.2)
            
            # Add stimulus ID label at the top
            ax.text(stim_start_relative, len(selected_neuron_ids) - 0.5, f'{int(stim_id)}',
                   rotation=90, verticalalignment='bottom', horizontalalignment='right',
                   fontsize=10, color='gray', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron Index (in selection)')
        ax.set_xlim(0, plot_end_ms - plot_start_ms)
        ax.set_ylim(-0.5, len(selected_neuron_ids) - 0.5)
        ax.grid(False)
    
    def _plot_firing_rates_subsampled(self, ax, epoch_idx, epoch_start_ms, epoch_end_ms,
                                       selected_stim_ids, selected_neuron_ids,
                                       stim_duration_ms, break_time_ms):
        """Plot firing rates per stimulus for selected neurons."""
        spike_times = self.rec_spikes.t / ms
        spike_neurons = self.rec_spikes.i
        
        # Calculate firing rate for each neuron × stimulus combination
        n_neurons = len(selected_neuron_ids)
        n_stims = len(selected_stim_ids)
        firing_rates = np.zeros((n_neurons, n_stims))
        
        for stim_idx, stim_id in enumerate(selected_stim_ids):
            stim_start = epoch_start_ms + stim_id * (stim_duration_ms + break_time_ms)
            stim_end = stim_start + stim_duration_ms
            
            for neuron_idx, neuron_id in enumerate(selected_neuron_ids):
                # Count spikes in this window
                mask = (spike_times >= stim_start) & (spike_times < stim_end) & (spike_neurons == neuron_id)
                n_spikes = np.sum(mask)
                firing_rates[neuron_idx, stim_idx] = n_spikes / (stim_duration_ms / 1000.0)  # Hz
        
        # Plot as heatmap
        im = ax.imshow(firing_rates, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('Stimulus ID')
        ax.set_ylabel('Neuron Index (in selection)')
        ax.set_xticks(range(n_stims))
        ax.set_xticklabels(selected_stim_ids)
        
        # Color y-ticks by E/I
        ytick_colors = ['red' if nid < self.N_rec_E else 'blue' for nid in selected_neuron_ids]
        ax.set_yticks(range(n_neurons))
        ax.set_yticklabels([f'{nid}' for nid in selected_neuron_ids])
        for ytick, color in zip(ax.get_yticklabels(), ytick_colors):
            ytick.set_color(color)
        
        plt.colorbar(im, ax=ax, label='Firing Rate (Hz)')
    
    def plotting_dashboard(self):
        """Generate dashboard plots for the direct input network (OLD VERSION - kept for compatibility)"""
        print("\nGenerating dashboard plots...")
        print("Creating 2x2 subplot layout...")
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot input vectors over time with interpolation for better visualization
        dt = float(defaultclock.dt/ms)  # in ms
        n_timepoints = self.input_vectors.shape[1]
        time_axis = np.arange(n_timepoints) * dt

        # Check if input vectors are constant within each stimulus window
        stim_duration = 100  # ms, or fetch from your config if available
        steps_per_stim = int(stim_duration / dt)
        for i in range(0, self.input_vectors.shape[1], steps_per_stim):
            window = self.input_vectors[:, i:i+steps_per_stim]
            if window.shape[1] > 0 and not np.allclose(window, window[:, [0]]):
                print(f"WARNING: Input vector changes within window starting at t={i*dt} ms!")

        input_sample = self.input_vectors  # No downsampling
        time_sample = time_axis

        # Panel A: Input vectors over time (top left)
        print("Plotting input vectors (top left)...")
        im_A = axs[0, 0].imshow(
            input_sample,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest',
            extent=[time_sample[0], time_sample[-1], 0, input_sample.shape[0]]
        )
        axs[0, 0].set_title('Input Vectors Over Time')
        axs[0, 0].set_xlabel('Time (ms)')
        axs[0, 0].set_ylabel('Input Dimension')

        # Add vertical lines and annotations at stimulus change points
        for t in range(1, len(self.stim_type_vec)):
            if self.stim_type_vec[t] != self.stim_type_vec[t-1]:
                current_stim = int(self.stim_type_vec[t])
                if current_stim >= 0:  # Only show for valid stimuli (not -1)
                    axs[0, 0].axvline(x=t*dt, color='lightgray', linestyle='--', alpha=0.7, linewidth=1.2)
                    # Add text annotation
                    axs[0, 0].text(t*dt, input_sample.shape[0], f'{current_stim}', rotation=90, 
                                 verticalalignment='top', horizontalalignment='right', 
                                 fontsize=12, color='gray', fontweight='bold')

        # Panel B: Raster plot (top right)
        print("Plotting raster plot (top right)...")
        pf.plot_raster(axs[0, 1], self.rec_spikes, self.N_rec_E, self.N_rec_I, self.T, 
                      neuron_order=self.neuron_order, stim_info=self.stim_info)

        # Panel C: Membrane potential (bottom left)
        print("Plotting membrane potential (bottom left)...")
        if self.rec_voltage is not None:
            pf.plot_membrane_potential(axs[1, 0], self.rec_voltage, self.T)
        else:
            print("Skipping membrane potential plot (not recorded with current monitoring level)")
            axs[1, 0].text(0.5, 0.5, 'Membrane potential\nnot recorded\n(minimal monitoring)', 
                         ha='center', va='center', transform=axs[1, 0].transAxes, fontsize=12)
            axs[1, 0].set_title('Membrane Potential (Not Recorded)')

        # Panel D: Smoothed firing rates (bottom right)
        print("Plotting smoothed firing rates (bottom right)...")
        pf.plot_average_firing_rates(
            self.smoothed_rates_rec, 
            self.smoothed_time_points, 
            self.neuron_groups, 
            ax=axs[1, 1]
        )

        plt.tight_layout()
        return fig

    def plot_individual_neuron_currents(self, neuron_ids=None, time_window=None, save_path=None):
        """
        Plot feed-forward excitatory, recurrent excitatory, recurrent inhibitory, and net excitatory
        currents for individual neurons.
        
        Parameters:
        -----------
        neuron_ids : list or None
            List of neuron IDs to plot. If None, plots first 3 E and 3 I neurons.
        time_window : tuple or None
            (start_ms, end_ms) to limit the time window. If None, plots entire simulation.
        save_path : str or None
            Path to save the figure. If None, just shows it.
        """
        if self.rec_current_E is None or self.rec_current_I is None or self.input_current is None:
            print("Error: Current monitors not initialized. Run with monitoring_level='basic' or 'full'")
            return
        
        # Select neurons to plot
        if neuron_ids is None:
            # Default: 1 E neuron and 1 I neuron
            e_neurons = [0] if self.N_rec_E >= 1 else []
            i_neurons = [self.N_rec_E] if self.N_rec_I >= 1 else []
            neuron_ids = e_neurons + i_neurons
        
        n_neurons = len(neuron_ids)
        
        # Get time array and apply time window
        time_ms = self.rec_current_E.t / ms
        if time_window is not None:
            start_idx = np.searchsorted(time_ms, time_window[0])
            end_idx = np.searchsorted(time_ms, time_window[1])
            time_ms = time_ms[start_idx:end_idx]
        else:
            start_idx = 0
            end_idx = len(time_ms)
        
        # Create figure with landscape orientation (wider than tall)
        fig, axes = plt.subplots(n_neurons, 1, figsize=(16, 4*n_neurons), sharex=True)
        if n_neurons == 1:
            axes = [axes]
        
        # Plot each neuron
        for idx, neuron_id in enumerate(neuron_ids):
            ax = axes[idx]
            
            # Get the monitor index for this neuron
            recorded_indices = np.array(self.rec_current_E.record)
            monitor_idx = np.where(recorded_indices == neuron_id)[0]
            
            if len(monitor_idx) == 0:
                ax.text(0.5, 0.5, f'Neuron {neuron_id} not monitored', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            monitor_idx = monitor_idx[0]
            
            # Extract currents (in nA)
            I_ff = self.input_current.I_input[monitor_idx, start_idx:end_idx] / nA
            I_rec_E = self.rec_current_E.I_syn_rec_E[monitor_idx, start_idx:end_idx] / nA
            I_rec_I = self.rec_current_I.I_syn_rec_I[monitor_idx, start_idx:end_idx] / nA
            I_net_E = I_ff + I_rec_E  # Net excitatory
            I_net_total = I_ff + I_rec_E + I_rec_I  # Net total (all currents)
            
            # Plot currents
            ax.plot(time_ms, I_ff, label='Feed-forward E', color='green', linewidth=1.5, alpha=0.8)
            ax.plot(time_ms, I_rec_E, label='Recurrent E', color='red', linewidth=1.5, alpha=0.8)
            ax.plot(time_ms, I_rec_I, label='Recurrent I', color='blue', linewidth=1.5, alpha=0.8)
            ax.plot(time_ms, I_net_E, label='Net Excitatory', color='orange', linewidth=1.5, linestyle='--', alpha=0.7)
            ax.plot(time_ms, I_net_total, label='Net Total', color='black', linewidth=2, linestyle='-', alpha=0.8)
            
            # Add zero line
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            
            # Add stimulus time markers
            if hasattr(self, 'stim_info') and self.stim_info is not None:
                for stim_id, stim_start, stim_end in self.stim_info:
                    stim_start_ms = stim_start / ms
                    if stim_start_ms >= time_ms[0] and stim_start_ms <= time_ms[-1]:
                        ax.axvline(stim_start_ms, color='lightgray', linestyle='--', alpha=0.7, linewidth=1.0)
            
            # Labels and legend
            neuron_type = 'E' if neuron_id < self.N_rec_E else 'I'
            ax.set_ylabel(f'Current (nA)\nNeuron {neuron_id} ({neuron_type})', fontsize=10)
            ax.legend(loc='upper right', fontsize=7, ncol=5)
            ax.grid(False)
        
        axes[-1].set_xlabel('Time (ms)', fontsize=12)
        fig.suptitle('Individual Neuron Currents: FF Exc, Rec Exc, Rec Inh, Net Exc, Net Total', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        return fig
    
    def _plot_synaptic_current(self, ax):
        """
        Plot the total recurrent synaptic current TO E and I neurons, plus net current.
        This shows the synaptic current that each neuron receives from the recurrent network.
        """
        if self.rec_current_E is None or self.rec_current_I is None:
            print("Error: rec_current monitors are not initialized.")
            return

        # Combine excitatory and inhibitory recurrent currents
        synaptic_currents_monitored = self.rec_current_E.I_syn_rec_E + self.rec_current_I.I_syn_rec_I
        time_points = self.rec_current_E.t / ms  # Convert to ms
        
        # Ensure dimensions match first
        if len(time_points) != synaptic_currents_monitored.shape[0]:
            min_length = min(len(time_points), synaptic_currents_monitored.shape[0])
            time_points = time_points[:min_length]
            synaptic_currents_monitored = synaptic_currents_monitored[:min_length, :]
        
        # Downsample if too many time points
        max_points = 1000
        if len(time_points) > max_points:
            indices = np.linspace(0, len(time_points)-1, max_points, dtype=int)
            time_points_ds = time_points[indices]
            synaptic_currents_monitored_ds = synaptic_currents_monitored[indices, :]
        else:
            time_points_ds = time_points
            synaptic_currents_monitored_ds = synaptic_currents_monitored
        
        # Separate E and I neurons based on their indices
        # In Brian2, we need to access the recorded indices differently
        recorded_indices = self.rec_current_E.record
        E_indices = [i for i, idx in enumerate(recorded_indices) if idx < self.N_rec_E]
        I_indices = [i for i, idx in enumerate(recorded_indices) if idx >= self.N_rec_E]
        
        # Plot mean ± std for E neurons (current TO E neurons)
        if len(E_indices) > 0:
            E_curr = synaptic_currents_monitored_ds[:, E_indices] / nA
            mean_E = E_curr.mean(axis=1)
            std_E = E_curr.std(axis=1)
            ax.plot(time_points_ds, mean_E, color='red', linewidth=2)
            ax.fill_between(time_points_ds, mean_E-std_E, mean_E+std_E, color='red', alpha=0.2)
        
        # Plot mean ± std for I neurons (current TO I neurons)
        if len(I_indices) > 0:
            I_curr = synaptic_currents_monitored_ds[:, I_indices] / nA
            mean_I = I_curr.mean(axis=1)
            std_I = I_curr.std(axis=1)
            ax.plot(time_points_ds, mean_I, color='blue', linewidth=2)
            ax.fill_between(time_points_ds, mean_I-std_I, mean_I+std_I, color='blue', alpha=0.2)
        
        # Plot net current (total across all monitored neurons)
        all_curr = synaptic_currents_monitored_ds / nA
        mean_net = all_curr.mean(axis=1)
        std_net = all_curr.std(axis=1)
        ax.plot(time_points_ds, mean_net, color='black', linewidth=2, linestyle='--')
        ax.fill_between(time_points_ds, mean_net-std_net, mean_net+std_net, color='black', alpha=0.1)
        
        ax.set_title('Recurrent Synaptic Current (TO E/I Neurons)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Synaptic Current (nA)')
        ax.grid(False)

    def _plot_input_current(self, ax):
        """
        Plot the total external input current to E and I neurons.
        This shows the current that each neuron receives directly from the input vector.
        """
        if self.input_current is None:
            print("Error: input_current monitor is not initialized.")
            return

        input_currents_monitored = self.input_current.I_input
        time_points = self.input_current.t / ms  # Convert to ms
        
        # Ensure dimensions match first
        if len(time_points) != input_currents_monitored.shape[0]:
            min_length = min(len(time_points), input_currents_monitored.shape[0])
            time_points = time_points[:min_length]
            input_currents_monitored = input_currents_monitored[:min_length, :]
        
        # Downsample if too many time points
        max_points = 1000
        if len(time_points) > max_points:
            indices = np.linspace(0, len(time_points)-1, max_points, dtype=int)
            time_points_ds = time_points[indices]
            input_currents_monitored_ds = input_currents_monitored[indices, :]
        else:
            time_points_ds = time_points
            input_currents_monitored_ds = input_currents_monitored
        
        # Separate E and I neurons based on their indices
        # In Brian2, we need to access the recorded indices differently
        recorded_indices = self.input_current.record
        E_indices = [i for i, idx in enumerate(recorded_indices) if idx < self.N_rec_E]
        I_indices = [i for i, idx in enumerate(recorded_indices) if idx >= self.N_rec_E]
        
        # Plot mean ± std for E neurons (current TO E neurons)
        if len(E_indices) > 0:
            E_curr = input_currents_monitored_ds[:, E_indices] / nA
            mean_E = E_curr.mean(axis=1)
            std_E = E_curr.std(axis=1)
            ax.plot(time_points_ds, mean_E, color='red', linewidth=2)
            ax.fill_between(time_points_ds, mean_E-std_E, mean_E+std_E, color='red', alpha=0.2)
        
        # Plot mean ± std for I neurons (current TO I neurons)
        if len(I_indices) > 0:
            I_curr = input_currents_monitored_ds[:, I_indices] / nA
            mean_I = I_curr.mean(axis=1)
            std_I = I_curr.std(axis=1)
            ax.plot(time_points_ds, mean_I, color='blue', linewidth=2)
            ax.fill_between(time_points_ds, mean_I-std_I, mean_I+std_I, color='blue', alpha=0.2)
        
        # Plot net current (total across all monitored neurons)
        all_curr = input_currents_monitored_ds / nA
        mean_net = all_curr.mean(axis=1)
        std_net = all_curr.std(axis=1)
        ax.plot(time_points_ds, mean_net, color='black', linewidth=2, linestyle='--')
        ax.fill_between(time_points_ds, mean_net-std_net, mean_net+std_net, color='black', alpha=0.1)
        
        ax.set_title('External Input Current (TO E/I Neurons)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Input Current (nA)')
        ax.grid(False)

    def _plot_total_current(self, ax):
        """
        Plot the total current (external + recurrent) to E and I neurons.
        This shows the current that each neuron receives from both the input vector and the recurrent network.
        """
        if self.rec_current_E is None or self.rec_current_I is None or self.input_current is None:
            print("Error: rec_current or input_current monitor is not initialized.")
            return

        # Combine I_syn_rec_E, I_syn_rec_I and I_input for total current
        total_currents_monitored = self.rec_current_E.I_syn_rec_E + self.rec_current_I.I_syn_rec_I + self.input_current.I_input
        time_points = self.rec_current_E.t / ms  # Convert to ms
        
        # Ensure dimensions match first
        if len(time_points) != total_currents_monitored.shape[0]:
            min_length = min(len(time_points), total_currents_monitored.shape[0])
            time_points = time_points[:min_length]
            total_currents_monitored = total_currents_monitored[:min_length, :]
        
        # Downsample if too many time points
        max_points = 1000
        if len(time_points) > max_points:
            indices = np.linspace(0, len(time_points)-1, max_points, dtype=int)
            time_points_ds = time_points[indices]
            total_currents_monitored_ds = total_currents_monitored[indices, :]
        else:
            time_points_ds = time_points
            total_currents_monitored_ds = total_currents_monitored
        
        # Separate E and I neurons based on their indices
        # In Brian2, we need to access the recorded indices differently
        recorded_indices = self.rec_current_E.record # Assuming rec_current_E.record contains all recorded indices
        E_indices = [i for i, idx in enumerate(recorded_indices) if idx < self.N_rec_E]
        I_indices = [i for i, idx in enumerate(recorded_indices) if idx >= self.N_rec_E]
        
        # Plot mean ± std for E neurons (total current TO E neurons)
        if len(E_indices) > 0:
            E_curr = total_currents_monitored_ds[:, E_indices] / nA
            mean_E = E_curr.mean(axis=1)
            std_E = E_curr.std(axis=1)
            ax.plot(time_points_ds, mean_E, color='red', linewidth=2)
            ax.fill_between(time_points_ds, mean_E-std_E, mean_E+std_E, color='red', alpha=0.2)
        
        # Plot mean ± std for I neurons (total current TO I neurons)
        if len(I_indices) > 0:
            I_curr = total_currents_monitored_ds[:, I_indices] / nA
            mean_I = I_curr.mean(axis=1)
            std_I = I_curr.std(axis=1)
            ax.plot(time_points_ds, mean_I, color='blue', linewidth=2)
            ax.fill_between(time_points_ds, mean_I-std_I, mean_I+std_I, color='blue', alpha=0.2)
        
        # Plot net current (total across all monitored neurons)
        all_curr = total_currents_monitored_ds / nA
        mean_net = all_curr.mean(axis=1)
        std_net = all_curr.std(axis=1)
        ax.plot(time_points_ds, mean_net, color='black', linewidth=2, linestyle='--')
        ax.fill_between(time_points_ds, mean_net-std_net, mean_net+std_net, color='black', alpha=0.1)
        
        ax.set_title('Total Current (TO E/I Neurons)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Total Current (nA)')
        ax.grid(False)

    def get_input_weight_matrix(self):
        """Return the input weight matrix"""
        return self.input_weights
        
    def get_network_info(self):
        """Return dictionary with network information"""
        return {
            'N_rec_E': self.N_rec_E,
            'N_rec_I': self.N_rec_I,
            'N_external_input_dims': self.N_external_input_dims,
            'input_weight_scale': self.input_weight_scale,
            'input_weights_shape': self.input_weights.shape
        } 