"""
DirectInputNetwork with plasticity support and weight monitoring.

This extends the DirectInputNetwork to support synaptic plasticity using
the plasticity kernel system.
"""

from brian2 import *
import numpy as np
from brian_experiment.feedforward.network import DirectInputNetwork
from brian_experiment.shared.core import spike_analysis_utils as sau

prefs.codegen.target = "numpy"


class PlasticDirectInputNetwork(DirectInputNetwork):
    """
    DirectInputNetwork with plasticity support.
    
    Extends DirectInputNetwork to allow plastic synapses that change
    their weights during simulation based on plasticity kernels.
    """
    
    def __init__(self, N_rec_E, N_rec_I, N_external_input_dims, weight_rec, 
                 input_weight_scale=1.0, lognormal_params_in=None, is_feedforward=False,
                 plasticity_config=None, weight_monitoring_dt=100*ms):
        """
        Initialize PlasticDirectInputNetwork.
        
        Parameters:
        -----------
        plasticity_config : dict or None
            Dictionary mapping connection types to PlasticityKernel objects.
            Keys: 'E->E', 'E->I', 'I->E', 'I->I', 'input->E', 'input->I'
            Values: PlasticityKernel instance or None (None = no plasticity)
        weight_monitoring_dt : Quantity (time)
            Time step for recording weight snapshots (default: 100ms)
        
        All other parameters as in DirectInputNetwork.__init__()
        """
        self.plasticity_config = plasticity_config or {}
        self.weight_monitoring_dt = weight_monitoring_dt
        self.weight_monitors = {}
        
        # Initialize parent class
        # We need to override synapse creation, so we'll call parent's __init__
        # but intercept the synapse creation
        self._init_with_plasticity(N_rec_E, N_rec_I, N_external_input_dims, 
                                   weight_rec, input_weight_scale, 
                                   lognormal_params_in, is_feedforward)
    
    def _init_with_plasticity(self, N_rec_E, N_rec_I, N_external_input_dims, 
                              weight_rec, input_weight_scale, lognormal_params_in, is_feedforward):
        """Initialize network with plasticity support."""
        print("Initializing PlasticDirectInputNetwork...")
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
        # Small input noise (dimensionless, added before scaling by unitary_current)
        # Increases spike timing variability so plastic weights can fluctuate up/down
        self.input_noise_std = 0.0  # disabled by default

        # Biological parameters
        self.tau_m = 20*ms
        self.tau_syn = 1*ms
        self.R_m = 100*Mohm
        
        if is_feedforward:
            self.v_rest = -115*mV
            self.v_thresh = -55*mV
            print(f"Feedforward mode: adjusted v_rest={self.v_rest/mV}mV")
        else:
            self.v_rest = -65*mV
            self.v_thresh = -55*mV
        
        self.C_m = self.tau_m/self.R_m

        # Generate input weights
        self.input_weights = self._generate_input_weights()
        
        # LIF equations
        self.eqs = '''
        dv/dt = (v_rest - v) / tau_m + (I_input + I_syn_rec_E + I_syn_rec_I) / C_m : volt (unless refractory)
        dI_syn_rec_E/dt = -I_syn_rec_E/tau_syn : amp
        dI_syn_rec_I/dt = -I_syn_rec_I/tau_syn : amp
        I_input : amp
        '''

        # Create recurrent neurons
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
        self.rec_neurons.I_input = 0*nA
        
        # Apply Dale's law
        self.weight_rec = self._apply_dales_law(self.weight_rec)
        
        # Create plastic synapses
        self._create_plastic_synapses()
        
        print(f"Network sizes - Rec: {self.N_rec} ({N_rec_E}E, {N_rec_I}I), Input dims: {self.N_external_input_dims}E")
        print(f"Plasticity enabled for: {list(self.plasticity_config.keys())}")
    
    def _create_plastic_synapses(self):
        """Create synapses with plasticity support."""
        # E->E and E->I synapses (from excitatory neurons)
        self.rec_synapses_E = self._create_synapse_group(
            self.rec_neurons[:self.N_rec_E],  # E sources
            self.rec_neurons,  # All targets
            'E->all',
            on_pre_current='I_syn_rec_E_post += w_syn'
        )
        
        # Set initial E weights
        weight_E_to_all = self.weight_rec[:self.N_rec_E, :].flatten()
        self.rec_synapses_E.w_syn = weight_E_to_all * nA
        
        # I->E and I->I synapses (from inhibitory neurons)
        self.rec_synapses_I = self._create_synapse_group(
            self.rec_neurons[self.N_rec_E:],  # I sources
            self.rec_neurons,  # All targets
            'I->all',
            on_pre_current='I_syn_rec_I_post += w_syn'
        )
        
        # Set initial I weights
        weight_I_to_all = self.weight_rec[self.N_rec_E:, :].flatten()
        self.rec_synapses_I.w_syn = weight_I_to_all * nA
    
    def _create_synapse_group(self, source, target, connection_type, on_pre_current):
        """
        Create a synapse group with optional plasticity.
        
        Parameters:
        -----------
        source : NeuronGroup
            Source neuron group
        target : NeuronGroup
            Target neuron group
        connection_type : str
            Type identifier for looking up plasticity rules
        on_pre_current : str
            Current injection code (e.g., 'I_syn_rec_E_post += w_syn')
        """
        # Determine which plasticity rules apply
        # Map 'E->all' to 'E->E' and 'E->I', etc.
        plasticity_kernel = None
        if connection_type == 'E->all':
            # For E->all, we could use E->E plasticity (simplified)
            plasticity_kernel = self.plasticity_config.get('E->E', None)
        elif connection_type == 'I->all':
            # For I->all, we could use I->E plasticity (simplified)
            plasticity_kernel = self.plasticity_config.get('I->E', None)
        
        # Base synapse equation
        equations = 'w_syn : amp'
        
        # Build on_pre and on_post code
        on_pre_code = on_pre_current
        on_post_code = ''
        namespace = {}
        
        if plasticity_kernel is not None:
            # Add plasticity equations
            plasticity_eqs = plasticity_kernel.get_brian2_equations()
            if plasticity_eqs:
                equations += '\n' + plasticity_eqs
            
            # Add plasticity update code
            plasticity_on_pre = plasticity_kernel.get_brian2_on_pre()
            if plasticity_on_pre:
                on_pre_code += '\n' + plasticity_on_pre
            
            plasticity_on_post = plasticity_kernel.get_brian2_on_post()
            if plasticity_on_post:
                on_post_code = plasticity_on_post
            
            # Get namespace parameters
            namespace = plasticity_kernel.get_brian2_namespace()
            
            print(f"  Plasticity enabled for {connection_type}")
        
        # Create synapses with namespace
        if on_post_code:
            synapses = Synapses(source, target, equations, 
                              on_pre=on_pre_code, on_post=on_post_code,
                              namespace=namespace if namespace else None)
        else:
            synapses = Synapses(source, target, equations, on_pre=on_pre_code,
                              namespace=namespace if namespace else None)
        
        synapses.connect()
        
        return synapses
    
    def run_simulation(self, T, input_vectors, stims_by_time, stim_info, current_injection, 
                      monitoring_level='minimal', unitary_current=1.0*nA, num_epochs=1):
        """
        Run simulation with weight monitoring.
        
        Same parameters as DirectInputNetwork.run_simulation(), but also
        records weight evolution over time.
        """
        print("\nStarting simulation with plasticity...")
        print(f"Simulation time: {T/ms}ms")
        print(f"Weight monitoring every {self.weight_monitoring_dt/ms}ms")
        
        self.input_vectors = input_vectors
        self.T = T
        self.current_injection = current_injection
        self.stim_type_vec = stims_by_time
        self.time_vec = self.T
        self.stim_info = stim_info

        # Essential monitors
        self.rec_spikes = SpikeMonitor(self.rec_neurons)
        self.input_spikes = None
        
        # Setup monitoring based on level
        self.monitoring_level = monitoring_level
        
        if monitoring_level == 'minimal':
            print("Using minimal monitoring (spikes only)")
            self.rec_voltage = None
            self.rec_current_E = None
            self.rec_current_I = None
            self.input_current = None
            
        elif monitoring_level == 'basic':
            subset_indices = True
            subset_size = self.N_rec
            print(f"Using basic monitoring (spikes + voltage + currents from ALL {subset_size} neurons)")
            
            self.rec_voltage = StateMonitor(self.rec_neurons, 'v', record=subset_indices)
            self.rec_current_E = StateMonitor(self.rec_neurons, 'I_syn_rec_E', record=subset_indices, dt=10*ms)
            self.rec_current_I = StateMonitor(self.rec_neurons, 'I_syn_rec_I', record=subset_indices, dt=10*ms)
            self.input_current = StateMonitor(self.rec_neurons, 'I_input', record=subset_indices, dt=10*ms)
            
        elif monitoring_level == 'full':
            print("Using full monitoring")
            self.rec_voltage = StateMonitor(self.rec_neurons, 'v', record=True)
            self.rec_current_E = StateMonitor(self.rec_neurons, 'I_syn_rec_E', record=True)
            self.rec_current_I = StateMonitor(self.rec_neurons, 'I_syn_rec_I', record=True)
            self.input_current = StateMonitor(self.rec_neurons, 'I_input', record=True)
        
        # Add weight monitors
        print("Setting up weight monitors...")
        self.weight_monitors['E_synapses'] = StateMonitor(
            self.rec_synapses_E, 'w_syn', record=True, dt=self.weight_monitoring_dt
        )
        self.weight_monitors['I_synapses'] = StateMonitor(
            self.rec_synapses_I, 'w_syn', record=True, dt=self.weight_monitoring_dt
        )
        
        # Add plasticity trace monitors for E synapses (sample a few)
        # Check if E synapses have plasticity traces
        if hasattr(self.rec_synapses_E, 'apost_narrow'):
            n_sample_synapses = min(20, len(self.rec_synapses_E))  # Monitor 20 synapses
            sample_indices = np.random.choice(len(self.rec_synapses_E), n_sample_synapses, replace=False)
            self.weight_monitors['E_trace_narrow'] = StateMonitor(
                self.rec_synapses_E, 'apost_narrow', record=sample_indices, dt=1*ms
            )
            self.weight_monitors['E_trace_wide'] = StateMonitor(
                self.rec_synapses_E, 'apost_wide', record=sample_indices, dt=1*ms
            )
            self.trace_sample_indices = sample_indices
            print(f"Monitoring plasticity traces for {n_sample_synapses} sample E synapses")
        else:
            self.weight_monitors['E_trace_narrow'] = None
            self.weight_monitors['E_trace_wide'] = None
            self.trace_sample_indices = None
        
        # Input current update function
        print("Setting up input current updates...")
        last_stim = -1
        last_print_time = 0
        
        @network_operation(dt=defaultclock.dt)
        def update_input_currents():
            nonlocal last_stim, last_print_time
            current_step = int(defaultclock.t / defaultclock.dt)
            current_time = defaultclock.t / ms

            if current_step < input_vectors.shape[1]:
                current_input = input_vectors[:, current_step]
                raw_currents = np.dot(self.input_weights, current_input)
                # Add small zero-mean Gaussian noise per neuron (pre-scaling)
                if self.input_noise_std and self.input_noise_std > 0:
                    noise = np.random.normal(0.0, self.input_noise_std, size=self.N_rec)
                    raw_currents = raw_currents + noise
                self.rec_neurons.I_input = raw_currents * unitary_current
                
                current_stim = stims_by_time[current_step] if current_step < len(stims_by_time) else -1
                if current_stim != last_stim and current_stim != -1:
                    last_stim = current_stim
                    
                    unique_stims = np.unique(stims_by_time[stims_by_time != -1])
                    total_stim_per_epoch = len(unique_stims)
                    current_epoch = (current_stim // total_stim_per_epoch) + 1 if total_stim_per_epoch > 0 else 1
                    
                    if current_time - last_print_time > 5000:
                        print(f">> Time: {current_time:.1f}ms - Epoch {current_epoch}/{num_epochs} - Stimulus {current_stim}")
                        last_print_time = current_time
            else:
                self.rec_neurons.I_input = 0 * nA

        # Build network
        self.net = Network()
        network_objects = [self.rec_neurons, self.rec_synapses_E, self.rec_synapses_I, 
                          self.rec_spikes, update_input_currents]
        
        # Add optional monitors
        if self.rec_voltage is not None:
            network_objects.append(self.rec_voltage)
        if self.rec_current_E is not None:
            network_objects.append(self.rec_current_E)
        if self.rec_current_I is not None:
            network_objects.append(self.rec_current_I)
        if self.input_current is not None:
            network_objects.append(self.input_current)
        
        # Add weight monitors
        for monitor in self.weight_monitors.values():
            network_objects.append(monitor)
            
        self.net.add(*network_objects)
        
        print("Running network...")
        self.net.run(self.T)
        
        print(f">> Simulation completed successfully!")
        print(f"   Weight snapshots recorded: {len(self.weight_monitors['E_synapses'].t)}")
    
    def get_weight_evolution(self, connection_type='E->all'):
        """
        Get weight evolution over time.
        
        Parameters:
        -----------
        connection_type : str
            'E->all' for excitatory synapses or 'I->all' for inhibitory
        
        Returns:
        --------
        times : array
            Time points (in ms) when weights were recorded
        weights : array
            Weight values at each time point [n_timepoints x n_synapses]
        """
        if connection_type == 'E->all':
            monitor = self.weight_monitors['E_synapses']
        elif connection_type == 'I->all':
            monitor = self.weight_monitors['I_synapses']
        else:
            raise ValueError(f"Unknown connection type: {connection_type}")
        
        times = monitor.t / ms
        weights = monitor.w_syn / nA  # Convert to nA
        
        return times, weights
    
    def get_weight_matrix_at_time(self, time_idx, connection_type='E->all'):
        """
        Get weight matrix at a specific time index.
        
        Parameters:
        -----------
        time_idx : int
            Index into the weight monitor times
        connection_type : str
            'E->all' for excitatory or 'I->all' for inhibitory
        
        Returns:
        --------
        weight_matrix : array
            2D weight matrix [sources x targets]
        """
        times, weights = self.get_weight_evolution(connection_type)
        
        if connection_type == 'E->all':
            n_sources = self.N_rec_E
            n_targets = self.N_rec
        elif connection_type == 'I->all':
            n_sources = self.N_rec_I
            n_targets = self.N_rec
        
        # Reshape from flat to matrix
        weight_vector = weights[:, time_idx]
        weight_matrix = weight_vector.reshape(n_sources, n_targets)
        
        return weight_matrix

