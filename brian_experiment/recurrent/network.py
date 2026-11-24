from brian2 import *
import matplotlib.pyplot as plt
prefs.codegen.target = "numpy"
from brian_experiment.shared.core import spike_analysis_utils as sau
from brian_experiment.shared.core.constants import SAVE_FOLDER
from brian_experiment.shared.plotting import plots as pf
import numpy as np
import dill as pickle
import os


class BalancedRecurrentNetwork:
    def __init__(
        self,
        N_rec_E,
        N_rec_I,
        weight_rec,
        initial_state_config=None,
        background_current=0*nA,
        verbose=True,
    ):
        """
        Initialize BalancedRecurrentNetwork for spontaneous chaotic dynamics.
        
        Parameters:
        - N_rec_E, N_rec_I: Number of excitatory and inhibitory recurrent neurons
        - weight_rec: Recurrent weight matrix (should be properly balanced for chaos)
        - initial_state_config: Dict with 'type' and parameters for initial state
                               {'type': 'random_voltage', 'v_min': -70*mV, 'v_max': -50*mV}
                               {'type': 'random_current', 'I_mean': 1*nA, 'I_std': 0.5*nA, 'duration': 50*ms}
        - background_current: Constant external drive to all neurons (for sustained activity)
        """
        self.verbose = verbose
        self._log("Initializing BalancedRecurrentNetwork...")
        self.net = None
        start_scope()
        
        # Network dimensions
        self.N_rec_E = N_rec_E
        self.N_rec_I = N_rec_I
        self.weight_rec = weight_rec
        self.N_rec = self.N_rec_I + self.N_rec_E
        self.initial_state_config = initial_state_config or {'type': 'random_voltage', 'v_min': -70*mV, 'v_max': -50*mV}
        self.background_current = background_current
        
        # Initialize monitors as None
        self.rec_current = None
        self.rec_voltage = None
        self.rec_current_E = None
        self.rec_current_I = None

        # Biological parameters
        self.tau_m = 20*ms       # Membrane time constant
        self.tau_syn = 1*ms      # Synaptic time constant
        self.R_m = 100*Mohm      # Membrane resistance
        self.v_rest = -65*mV     # Resting potential
        self.v_thresh = -55*mV   # Spike threshold
        
        # Derived parameter
        self.C_m = self.tau_m/self.R_m  # Membrane capacitance

        # LIF voltage equation with separate excitatory and inhibitory currents
        self.eqs = '''
        dv/dt = (v_rest - v) / tau_m + (I_syn_rec_E + I_syn_rec_I + I_external) / C_m : volt (unless refractory)
        dI_syn_rec_E/dt = -I_syn_rec_E/tau_syn : amp
        dI_syn_rec_I/dt = -I_syn_rec_I/tau_syn : amp
        I_external : amp
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
        
        # Set initial state
        self._set_initial_state()
        
        # Add constant background current to all neurons
        self.rec_neurons.I_external += self.background_current
        
        # Weight matrix is (target, source) - Dale's law already applied
        # Sources: columns 0 to N_REC_E-1 are E, columns N_REC_E to end are I
        # Targets: rows 0 to N_REC_E-1 are E, rows N_REC_E to end are I
        
        # Current-based recurrent synapses - separate for E and I sources
        # Excitatory recurrent synapses (from E neurons)
        self.rec_synapses_E = Synapses(
            self.rec_neurons[:self.N_rec_E],  # Only E neurons as sources
            self.rec_neurons,
            'w_syn : amp',
            on_pre='I_syn_rec_E_post += w_syn'
        )
        self.rec_synapses_E.connect()
        # Extract E source weights: columns 0 to N_REC_E-1, all rows
        # Brian's connect() creates connections in source-major order (for each source, all targets)
        # So we need to transpose to match: weight_rec is [target, source], we need [source, target] order
        weight_E_to_all = self.weight_rec[:, :self.N_rec_E].T.flatten()
        self.rec_synapses_E.w_syn = weight_E_to_all * nA
        
        # Inhibitory recurrent synapses (from I neurons)
        self.rec_synapses_I = Synapses(
            self.rec_neurons[self.N_rec_E:],  # Only I neurons as sources
            self.rec_neurons,
            'w_syn : amp',
            on_pre='I_syn_rec_I_post += w_syn'
        )
        self.rec_synapses_I.connect()
        # Extract I source weights: columns N_REC_E to end, all rows
        # Transpose to match Brian's source-major connection order
        weight_I_to_all = self.weight_rec[:, self.N_rec_E:].T.flatten()
        self.rec_synapses_I.w_syn = weight_I_to_all * nA

        self._log(f"Network sizes - Rec: {self.N_rec} ({N_rec_E}E, {N_rec_I}I)")
        self._log(f"Initial state: {self.initial_state_config['type']}")

    def _log(self, message):
        if self.verbose:
            print(message)

    def _set_initial_state(self):
        """Set initial state of the network to kickstart dynamics"""
        config = self.initial_state_config
        
        if config['type'] == 'random_voltage':
            # Initialize with random voltages
            v_min = config.get('v_min', -70*mV)
            v_max = config.get('v_max', -50*mV)
            self.rec_neurons.v = np.random.uniform(v_min/mV, v_max/mV, self.N_rec) * mV
            self.rec_neurons.I_external = 0*nA
            self._log(f"   Initial voltages: uniform [{v_min/mV:.1f}, {v_max/mV:.1f}] mV")
            
        elif config['type'] == 'random_current':
            # Initialize at rest with external current for initialization
            self.rec_neurons.v = self.v_rest
            I_mean = config.get('I_mean', 1*nA)
            I_std = config.get('I_std', 0.5*nA)
            self.rec_neurons.I_external = np.random.normal(I_mean/nA, I_std/nA, self.N_rec) * nA
            self.initial_current_duration = config.get('duration', 50*ms)
            self._log(
                f"   Initial current injection: N({I_mean/nA:.2f}, {I_std/nA:.2f}) nA for {self.initial_current_duration/ms:.1f} ms"
            )
            
        elif config['type'] == 'subset_spikes':
            # Initialize subset of neurons above threshold
            self.rec_neurons.v = self.v_rest
            fraction = config.get('fraction', 0.1)
            n_init = int(self.N_rec * fraction)
            init_indices = np.random.choice(self.N_rec, n_init, replace=False)
            self.rec_neurons.v[init_indices] = self.v_thresh + 1*mV
            self.rec_neurons.I_external = 0*nA
            self._log(f"   Initial spikes: {n_init} neurons ({fraction*100:.1f}%) above threshold")
            
        else:
            # Default: random voltages near threshold
            self.rec_neurons.v = np.random.uniform(-70, -55, self.N_rec) * mV
            self.rec_neurons.I_external = 0*nA
            self._log("   Initial state: random voltages (default)")

    def _apply_dales_law(self, weight_matrix):
        """
        Apply Dale's law to weight matrix.
        """
        corrected_weights = weight_matrix.copy()
        
        # Excitatory source neurons have positive outgoing weights
        corrected_weights[:self.N_rec_E, :] = np.abs(corrected_weights[:self.N_rec_E, :])
        
        # Inhibitory source neurons have negative outgoing weights  
        corrected_weights[self.N_rec_E:, :] = -np.abs(corrected_weights[self.N_rec_E:, :])
        
        return corrected_weights

    def run_simulation(self, T, monitoring_level='basic'):
        """
        Run the network simulation with spontaneous dynamics.
        
        Parameters:
        - T: Simulation time
        - monitoring_level: 'minimal', 'basic', or 'full'
        """
        self._log("\nStarting balanced recurrent network simulation...")
        self._log(f"Simulation time: {T/ms}ms")
        self._log(f"Monitoring level: {monitoring_level}")
        
        self.T = T
        self.stim_type_vec = -np.ones(int(T/defaultclock.dt))  # No stimuli
        self.time_vec = self.T
        self.stim_info = []

        # Essential monitors
        self.rec_spikes = SpikeMonitor(self.rec_neurons)
        self.input_spikes = None
        
        # Configurable monitoring
        self.monitoring_level = monitoring_level
        
        if monitoring_level == 'minimal':
            self._log("Using minimal monitoring (spikes only)")
            self.rec_voltage = None
            self.rec_current_E = None
            self.rec_current_I = None
            
        elif monitoring_level == 'basic':
            subset_size = self.N_rec
            self._log(f"Using basic monitoring (spikes + voltage + currents from ALL {subset_size} neurons)")
            
            self.rec_voltage = StateMonitor(self.rec_neurons, 'v', record=True)
            self.rec_current_E = StateMonitor(self.rec_neurons, 'I_syn_rec_E', record=True, dt=1*ms)
            self.rec_current_I = StateMonitor(self.rec_neurons, 'I_syn_rec_I', record=True, dt=1*ms)
            
        elif monitoring_level == 'full':
            self._log("Using full monitoring (all data)")
            self.rec_voltage = StateMonitor(self.rec_neurons, 'v', record=True)
            self.rec_current_E = StateMonitor(self.rec_neurons, 'I_syn_rec_E', record=True)
            self.rec_current_I = StateMonitor(self.rec_neurons, 'I_syn_rec_I', record=True)
        else:
            raise ValueError(f"Unknown monitoring level: {monitoring_level}")

        # Handle initial current injection if configured
        if self.initial_state_config['type'] == 'random_current':
            @network_operation(dt=defaultclock.dt)
            def turn_off_initial_current():
                if defaultclock.t >= self.initial_current_duration:
                    self.rec_neurons.I_external = 0*nA
            
            network_objects = [self.rec_neurons, self.rec_synapses_E, self.rec_synapses_I, 
                              self.rec_spikes, turn_off_initial_current]
        else:
            network_objects = [self.rec_neurons, self.rec_synapses_E, self.rec_synapses_I, 
                              self.rec_spikes]
        
        # Add optional monitors
        if self.rec_voltage is not None:
            network_objects.append(self.rec_voltage)
        if self.rec_current_E is not None:
            network_objects.append(self.rec_current_E)
        if self.rec_current_I is not None:
            network_objects.append(self.rec_current_I)
            
        # Build and run network
        self.net = Network()
        self.net.add(*network_objects)
        
        self._log("Running network...")
        self.net.run(self.T)
        self._log(">> Simulation completed!")
              
    def analyze_spikes(self, smoothing_window_size=50 * ms):
        """Analyze spike data"""
        spike_matrix_rec = sau.spike_matrix_from_monitor(self.rec_spikes, self.N_rec, self.T, dt=defaultclock.dt)
        self.spike_matrix_rec = spike_matrix_rec

        # Smooth the spike matrix
        window_size = smoothing_window_size
        dt_seconds = float(defaultclock.dt / second)
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

        self._log("Spike analysis completed")

    def save_spikes(self, filename):
        """Save spike data to pickle file"""
        if '/' not in filename and '\\' not in filename:
            filename = os.path.join(SAVE_FOLDER, filename)
            
        my_dict = {
            "stim_by_time": self.stim_type_vec,
            "spike_matrix_rec": self.spike_matrix_rec,
            "T": self.T,
            "N_rec_E": self.N_rec_E,
            "N_rec_I": self.N_rec_I,
            "neuron_order": self.neuron_order,
            "initial_state_config": self.initial_state_config
        }
        with open(filename, "wb") as file:
            pickle.dump(my_dict, file)

    def plot_spontaneous_activity(self, time_window=None, save_folder=None):
        """
        Generate comprehensive plots for spontaneous activity analysis.
        
        Parameters:
        - time_window: tuple (start_ms, end_ms) or None for full simulation
        - save_folder: folder to save individual plots (if None, just returns figure)
        """
        self._log("\nGenerating spontaneous activity plots...")
        
        if time_window is None:
            t_start = 0*ms
            t_end = self.T
        else:
            t_start = time_window[0]*ms
            t_end = time_window[1]*ms
        
        # Create figure with 4 panels
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Raster plot (full width, top)
        ax_raster = fig.add_subplot(gs[0, :])
        self._log("  Plotting raster...")
        spike_times = self.rec_spikes.t
        spike_neurons = self.rec_spikes.i
        
        mask = (spike_times >= t_start) & (spike_times < t_end)
        filtered_times = spike_times[mask] / ms
        filtered_neurons = spike_neurons[mask]
        
        colors = ['red' if n < self.N_rec_E else 'blue' for n in filtered_neurons]
        ax_raster.scatter(filtered_times, filtered_neurons, s=1, c=colors, alpha=0.6, rasterized=True)
        ax_raster.set_ylabel('Neuron Index', fontsize=11)
        ax_raster.set_title('Spontaneous Activity - Raster Plot (Red=E, Blue=I)', fontsize=12, fontweight='bold')
        ax_raster.set_xlim(t_start/ms, t_end/ms)
        ax_raster.axhline(self.N_rec_E, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Panel 2: Population firing rates
        ax_rates = fig.add_subplot(gs[1, :])
        self._log("  Plotting population rates...")
        if hasattr(self, 'smoothed_rates_rec'):
            time_points = self.smoothed_time_points
            mask_t = (time_points >= t_start/second) & (time_points < t_end/second)
            time_window_pts = time_points[mask_t]
            
            # smoothed_rates_rec is [neurons, time], so average over neurons axis (0)
            rates_e = np.mean(self.smoothed_rates_rec[:self.N_rec_E, :], axis=0)[mask_t]
            rates_i = np.mean(self.smoothed_rates_rec[self.N_rec_E:, :], axis=0)[mask_t]
            
            ax_rates.plot(time_window_pts*1000, rates_e, 'r-', linewidth=2, label='E population', alpha=0.8)
            ax_rates.plot(time_window_pts*1000, rates_i, 'b-', linewidth=2, label='I population', alpha=0.8)
            ax_rates.set_ylabel('Firing Rate (Hz)', fontsize=11)
            ax_rates.set_title('Population Firing Rates', fontsize=12, fontweight='bold')
            ax_rates.legend(fontsize=10)
            ax_rates.set_xlim(t_start/ms, t_end/ms)
            ax_rates.grid(True, alpha=0.3)
        
        # Panel 3: Mean membrane potential
        ax_voltage = fig.add_subplot(gs[2, :])
        self._log("  Plotting mean voltage...")
        if self.rec_voltage is not None:
            time_ms = self.rec_voltage.t / ms
            mask_v = (time_ms >= t_start/ms) & (time_ms < t_end/ms)
            time_window_v = time_ms[mask_v]
            
            v_e = np.mean(self.rec_voltage.v[:self.N_rec_E, mask_v], axis=0) / mV
            v_i = np.mean(self.rec_voltage.v[self.N_rec_E:, mask_v], axis=0) / mV
            
            ax_voltage.plot(time_window_v, v_e, 'r-', linewidth=1.5, label='E population', alpha=0.8)
            ax_voltage.plot(time_window_v, v_i, 'b-', linewidth=1.5, label='I population', alpha=0.8)
            ax_voltage.axhline(self.v_thresh/mV, color='gray', linestyle='--', alpha=0.5, label='Threshold')
            ax_voltage.set_ylabel('Membrane Potential (mV)', fontsize=11)
            ax_voltage.set_title('Mean Membrane Potential', fontsize=12, fontweight='bold')
            ax_voltage.legend(fontsize=10)
            ax_voltage.set_xlim(t_start/ms, t_end/ms)
            ax_voltage.grid(True, alpha=0.3)
        
        # Panel 4: Mean synaptic currents (E and I)
        ax_currents = fig.add_subplot(gs[3, 0])
        self._log("  Plotting synaptic currents...")
        if self.rec_current_E is not None and self.rec_current_I is not None:
            time_ms_curr = self.rec_current_E.t / ms
            mask_c = (time_ms_curr >= t_start/ms) & (time_ms_curr < t_end/ms)
            time_window_c = time_ms_curr[mask_c]
            
            I_e_mean = np.mean(self.rec_current_E.I_syn_rec_E[:, mask_c], axis=0) / nA
            I_i_mean = np.mean(self.rec_current_I.I_syn_rec_I[:, mask_c], axis=0) / nA
            
            ax_currents.plot(time_window_c, I_e_mean, 'r-', linewidth=1.5, label='Excitatory', alpha=0.8)
            ax_currents.plot(time_window_c, I_i_mean, 'b-', linewidth=1.5, label='Inhibitory', alpha=0.8)
            ax_currents.plot(time_window_c, I_e_mean + I_i_mean, 'k--', linewidth=1.5, label='Net', alpha=0.7)
            ax_currents.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax_currents.set_ylabel('Current (nA)', fontsize=11)
            ax_currents.set_xlabel('Time (ms)', fontsize=11)
            ax_currents.set_title('Mean Synaptic Currents', fontsize=12, fontweight='bold')
            ax_currents.legend(fontsize=9)
            ax_currents.set_xlim(t_start/ms, t_end/ms)
            ax_currents.grid(True, alpha=0.3)
        
        # Panel 5: Individual neuron voltage traces (sample)
        ax_traces = fig.add_subplot(gs[3, 1])
        self._log("  Plotting individual voltage traces...")
        if self.rec_voltage is not None:
            # Sample a few neurons
            n_sample = min(10, self.N_rec_E)
            sample_e = np.random.choice(self.N_rec_E, n_sample//2, replace=False)
            sample_i = np.random.choice(range(self.N_rec_E, self.N_rec), n_sample//2, replace=False)
            
            time_ms_v = self.rec_voltage.t / ms
            mask_v = (time_ms_v >= t_start/ms) & (time_ms_v < t_end/ms)
            time_window_v = time_ms_v[mask_v]
            
            for idx in sample_e:
                ax_traces.plot(time_window_v, self.rec_voltage.v[idx, mask_v]/mV, 
                             'r-', alpha=0.4, linewidth=0.8)
            for idx in sample_i:
                ax_traces.plot(time_window_v, self.rec_voltage.v[idx, mask_v]/mV, 
                             'b-', alpha=0.4, linewidth=0.8)
            
            ax_traces.axhline(self.v_thresh/mV, color='gray', linestyle='--', alpha=0.5)
            ax_traces.set_ylabel('Voltage (mV)', fontsize=11)
            ax_traces.set_xlabel('Time (ms)', fontsize=11)
            ax_traces.set_title(f'Sample Neurons (n={n_sample})', fontsize=12, fontweight='bold')
            ax_traces.set_xlim(t_start/ms, t_end/ms)
            ax_traces.grid(True, alpha=0.3)
        
        return fig
    
    def get_network_info(self):
        """Return dictionary with network information"""
        return {
            'N_rec_E': self.N_rec_E,
            'N_rec_I': self.N_rec_I,
            'initial_state_config': self.initial_state_config
        }

