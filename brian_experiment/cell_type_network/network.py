from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
import os
from brian_experiment.shared.core import spike_analysis_utils as sau
from brian_experiment.shared.core.constants import SAVE_FOLDER

class CellTypeNetwork:
    def __init__(self, population_configs, connection_configs, input_layer_config=None, 
                 simulation_params=None, motif_constraints=None, verbose=True):
        """
        Initialize a cell type network with multiple populations and custom connectivity.
        
        Parameters:
        - population_configs: List of dicts, each defining a population.
        - connection_configs: List of dicts, each defining a projection.
        - input_layer_config: Optional dict defining Layer 0 input:
            { 'n': int, 'name': str (default 'Layer0') }
        - simulation_params: Dict of simulation parameters (tau_m, etc.)
        - motif_constraints: Optional list of motif constraint dicts for structured connectivity.
            Example: [{"name": "reciprocal_EI", "type": "reciprocal", 
                       "node_filters": {"A": {"type": "E"}, "B": {"type": "I"}},
                       "enhancement": 3.0}]
        """
        self.verbose = verbose
        self._log("Initializing CellTypeNetwork...")
        self.motif_constraints = motif_constraints
        
        start_scope()
        
        self.population_configs = population_configs
        self.connection_configs = connection_configs
        self.input_layer_config = input_layer_config
        
        # ALL parameters from config - NO DEFAULTS
        if simulation_params is None:
            raise ValueError("simulation_params is required - no defaults allowed")
        self.simulation_params = simulation_params
        
        # Biological parameters from config (no defaults)
        self.tau_m = simulation_params['tau_m_ms'] * ms
        self.tau_syn = simulation_params['tau_syn_ms'] * ms
        self.R_m = simulation_params['R_m_Mohm'] * Mohm
        self.v_rest = simulation_params['v_rest_mV'] * mV
        self.v_thresh = simulation_params['v_thresh_mV'] * mV
        self.v_reset = simulation_params['v_reset_mV'] * mV
        self.refractory = simulation_params['refractory_ms'] * ms
        self.C_m = self.tau_m / self.R_m
        
        # Equation for neurons
        # We separate E and I synaptic currents for monitoring
        self.eqs = '''
        dv/dt = (v_rest - v) / tau_m + (I_syn_E + I_syn_I + I_ext) / C_m : volt (unless refractory)
        dI_syn_E/dt = -I_syn_E/tau_syn : amp
        dI_syn_I/dt = -I_syn_I/tau_syn : amp
        I_ext : amp
        '''
        
        self.populations = {}
        self.synapses = []
        self.input_sources = []
        self.monitors = {}
        
        self._build_input_layer()
        self._build_populations()
        self._build_connections()
        
        self.net = Network()
        self.net.add([p for p in self.populations.values()])
        self.net.add(self.synapses)
        self.net.add(self.input_sources)
        
        self._log(f"Network built with {len(self.populations)} populations and {len(self.synapses)} projections.")

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _build_populations(self):
        for config in self.population_configs:
            name = config['name']
            n = config['n']
            
            # Create NeuronGroup
            group = NeuronGroup(
                n,
                self.eqs,
                threshold='v > v_thresh',
                reset='v = v_reset',
                refractory=self.refractory,
                method='euler',
                name=name,
                namespace={
                    'v_rest': self.v_rest,
                    'v_thresh': self.v_thresh,
                    'v_reset': self.v_reset,
                    'tau_m': self.tau_m,
                    'C_m': self.C_m,
                    'tau_syn': self.tau_syn
                }
            )
            
            # Initialization
            group.v = self.v_rest + (self.v_thresh - self.v_rest) * np.random.rand(n)
            group.I_ext = 0*nA
            
            self.populations[name] = group
            self._log(f"Created population '{name}': {n} neurons, type {config.get('type')}")
            
            # Handle External Input
            if config.get('external_input', False):
                rate = config.get('external_input_rate', 10*Hz)
                weight = config.get('external_input_weight', 1*nA) # This acts as peak current for an input spike
                
                # Create PoissonInput or PoissonGroup
                # Using PoissonInput for efficiency if we just want background noise
                # PoissonInput(target, target_var, N, rate, weight)
                # Adds 'weight' to 'target_var' for each spike from N inputs at 'rate'
                # But here we want to drive them.
                # Let's use PoissonGroup + Synapses for more control if needed, but PoissonInput is simpler for "background"
                
                # If the user wants specific "FF input" which usually implies structure, they might need PoissonGroup.
                # But "external poisson" often implies background noise or generic drive.
                # I'll use PoissonGroup to allow monitoring input spikes if needed.
                
                input_N = n # One-to-one or global? usually independent per neuron or shared.
                # Let's assume independent Poisson process per neuron for now (like background)
                # Or a source layer. 
                # "all the E neurons get FF input" -> likely external drive.
                
                # Simplest: PoissonGroup with N neurons, one-to-one connection
                poisson_source = PoissonGroup(n, rates=rate, name=f"input_{name}")
                self.input_sources.append(poisson_source)
                
                input_syn = Synapses(poisson_source, group, on_pre='I_syn_E += weight', namespace={'weight': weight})
                input_syn.connect(j='i') # One-to-one
                self.synapses.append(input_syn)
                
                self._log(f"  Added external Poisson input to '{name}' at {rate} with weight {weight}")

    def _build_input_layer(self):
        if self.input_layer_config:
            name = self.input_layer_config.get('name', 'Layer0')
            n = self.input_layer_config['n']
            
            # Create PoissonGroup for Layer 0
            # We reference 'input_rates(t, i)' which will be populated in run_simulation
            # Initialize with a dummy function in namespace to avoid errors during creation
            # Note: TimedArray expects (time, index), so we use (t, i)
            self.input_layer = PoissonGroup(n, rates='input_rates(t, i)', name=name, 
                                          namespace={'input_rates': lambda t, i: 0*Hz})
            
            self.populations[name] = self.input_layer
            
            # We add it to self.input_sources so it gets added to the network
            self.input_sources.append(self.input_layer)
            
            self._log(f"Created Input Layer '{name}': {n} neurons (PoissonGroup)")

    def _build_connections(self):
        # Check if we should use motif-constrained connectivity
        if self.motif_constraints:
            self._build_connections_with_motifs()
        else:
            self._build_connections_random()
    
    def _build_connections_with_motifs(self):
        """Build connections using motif-constrained connectivity."""
        from brian_experiment.shared.core.motif_connectivity import generate_motif_connectivity
        
        self._log("Using motif-constrained connectivity...")
        
        # Generate connectivity structure
        connectivity = generate_motif_connectivity(
            population_configs=self.population_configs,
            connection_configs=self.connection_configs,
            motif_constraints=self.motif_constraints
        )
        
        # Log motif stats
        for (src, tgt), conns in connectivity.items():
            self._log(f"  Motif connectivity: {src}->{tgt}: {len(conns)} connections")
        
        # Create synapses for each connection config
        for config in self.connection_configs:
            source_name = config['source']
            target_name = config['target']
            
            if source_name not in self.populations or target_name not in self.populations:
                self._log(f"WARNING: Skipping connection {source_name}->{target_name}: population not found.")
                continue
            
            source_pop = self.populations[source_name]
            target_pop = self.populations[target_name]
            
            # Determine synapse type
            source_type = self._get_source_type(source_name)
            target_var = 'I_syn_E' if source_type == 'E' else 'I_syn_I'
            
            # Create Synapses
            syn = Synapses(source_pop, target_pop, 'w : amp', on_pre=f'{target_var} += w', 
                          name=f"syn_{source_name}_{target_name}")
            
            # Check if source is input layer (not in population_configs, use random connectivity)
            source_in_pop_configs = any(p['name'] == source_name for p in self.population_configs)
            
            if source_in_pop_configs:
                # Use motif-constrained connectivity
                conn_list = connectivity.get((source_name, target_name), [])
                
                if conn_list:
                    source_indices = [c[0] for c in conn_list]
                    target_indices = [c[1] for c in conn_list]
                    syn.connect(i=source_indices, j=target_indices)
                else:
                    p = config.get('p', 0.1)
                    syn.connect(p=p)
                
                n_synapses = len(conn_list) if conn_list else len(syn)
            else:
                # Input layer: use random connectivity
                p = config.get('p', 0.1)
                syn.connect(p=p)
                n_synapses = len(syn)
            
            # Set weights
            self._set_synapse_weights(syn, config, source_type)
            
            self.synapses.append(syn)
            self._log(f"Connected {source_name} -> {target_name} ({n_synapses} synapses, type={source_type})")
    
    def _build_connections_random(self):
        """Build connections using standard random connectivity."""
        for config in self.connection_configs:
            source_name = config['source']
            target_name = config['target']
            
            if source_name not in self.populations or target_name not in self.populations:
                self._log(f"WARNING: Skipping connection {source_name}->{target_name}: population not found.")
                continue
                
            source_pop = self.populations[source_name]
            target_pop = self.populations[target_name]
            
            # Determine synapse type
            source_type = self._get_source_type(source_name)
            target_var = 'I_syn_E' if source_type == 'E' else 'I_syn_I'
            
            # Create Synapses
            syn = Synapses(source_pop, target_pop, 'w : amp', on_pre=f'{target_var} += w', 
                          name=f"syn_{source_name}_{target_name}")
            
            # Connect with probability
            p = config.get('p', 0.1)
            syn.connect(p=p)
            
            # Set weights
            self._set_synapse_weights(syn, config, source_type)
            
            self.synapses.append(syn)
            self._log(f"Connected {source_name} -> {target_name} (p={p}, type={source_type})")
    
    def _get_source_type(self, source_name):
        """Get the type (E or I) for a source population."""
        source_config = next((c for c in self.population_configs if c['name'] == source_name), None)
        
        if source_config is None and self.input_layer_config and source_name == self.input_layer_config.get('name', 'Layer0'):
            return 'E'  # Input layer is typically Excitatory
        
        return source_config.get('type', 'E') if source_config else 'E'
    
    def _set_synapse_weights(self, syn, config, source_type):
        """Set weights for a synapse object based on config."""
        weight_dist = config.get('weight_dist', {'dist': 'fixed', 'val': 1.0})
        config_scale = config.get('weight_scale', 1.0)
        computed_scale = config.get('_computed_scale', None)
        
        # Determine sign (Dale's law)
        sign = 1.0 if source_type == 'E' else -1.0
        
        if weight_dist['dist'] == 'fixed':
            val = weight_dist.get('val', 1.0)
            effective_scale = computed_scale if computed_scale is not None else config_scale
            syn.w = val * effective_scale * sign * nA
        elif weight_dist['dist'] == 'normal':
            mu = weight_dist.get('mu', 1.0)
            sigma = weight_dist.get('sigma', 0.1)
            N_syn = len(syn)
            if N_syn > 0:
                w_raw = np.abs(np.random.normal(mu, sigma, N_syn))
                effective_scale = computed_scale if computed_scale is not None else config_scale
                syn.w = w_raw * effective_scale * sign * nA
        elif weight_dist['dist'] == 'lognormal':
            mu = weight_dist.get('mu', 0.0)
            sigma = weight_dist.get('sigma', 0.5)
            if len(syn) > 0:
                w_raw = np.random.lognormal(mu, sigma, len(syn))
                effective_scale = computed_scale if computed_scale is not None else config_scale
                syn.w = w_raw * effective_scale * sign * nA

    def run_simulation(self, duration, input_patterns=None, input_dt=None, monitoring_level='basic'):
        self._log(f"Starting simulation for {duration}...")
        
        # Handle Input Patterns for Layer 0
        if input_patterns is not None and hasattr(self, 'input_layer'):
            # input_patterns: [N_inputs x Time] array of rates (in Hz)
            # TimedArray expects [Time x N_inputs]
            
            if input_dt is None:
                input_dt = defaultclock.dt
                
            patterns_t = input_patterns.T * Hz
            # Create TimedArray
            self.input_rates_array = TimedArray(patterns_t, dt=input_dt)
            
            # Update the namespace of the input layer to use the new TimedArray
            self.input_layer.namespace['input_rates'] = self.input_rates_array
            
            self._log(f"  Applied input patterns with shape {input_patterns.shape}")
        
        # Setup monitors
        self.monitors = {}
        
        # Spike monitors for all populations
        for name, pop in self.populations.items():
            self.monitors[f'spikes_{name}'] = SpikeMonitor(pop, name=f'spikes_{name}')
            self.net.add(self.monitors[f'spikes_{name}'])
            
        if monitoring_level != 'minimal':
            # State monitors
            for name, pop in self.populations.items():
                # Skip PoissonGroup for state monitoring (no voltage)
                if isinstance(pop, PoissonGroup):
                    continue
                    
                # Record a subset if large? For now record all or subset logic
                self.monitors[f'state_{name}'] = StateMonitor(pop, ['v', 'I_syn_E', 'I_syn_I'], record=True, name=f'state_{name}')
                self.net.add(self.monitors[f'state_{name}'])
        
        self.net.run(duration)
        self._log("Simulation finished.")

    def plot_weight_matrix(self, ax=None):
        """
        Plot the full weight matrix of the network with histograms of weights for each projection block.
        """
        # Calculate total neurons and population offsets
        total_neurons = int(np.sum([len(p) for p in self.populations.values()]))
        pop_offsets = {}
        current_offset = 0
        pop_names = list(self.populations.keys())
        
        for name in pop_names:
            pop_offsets[name] = current_offset
            current_offset += len(self.populations[name])
            
        # Initialize full weight matrix
        W = np.zeros((total_neurons, total_neurons))
        
        # Store weights by connection type for histograms
        weights_by_conn = {}
        
        # Fill weights from synapses
        for syn in self.synapses:
            source_name = syn.source.name
            target_name = syn.target.name
            
            if source_name not in pop_offsets or target_name not in pop_offsets:
                continue
                
            source_offset = pop_offsets[source_name]
            target_offset = pop_offsets[target_name]
            
            global_source_indices = syn.i[:] + source_offset
            global_target_indices = syn.j[:] + target_offset
            
            try:
                weights = syn.w[:] / nA
            except:
                weights = np.array(syn.w)
            
            W[global_target_indices, global_source_indices] = weights
            
            conn_key = f"{source_name}->{target_name}"
            weights_by_conn[conn_key] = weights
            
        # Plotting
        if ax is None:
            # Create a complex layout: Matrix on left, Histograms on right
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
            ax_matrix = fig.add_subplot(gs[0, 0])
            
            # Create sub-gridspec for histograms in the right column
            n_conns = len(weights_by_conn)
            n_cols_hist = 2
            n_rows_hist = int(np.ceil(n_conns / n_cols_hist))
            gs_hist = gs[0, 1].subgridspec(n_rows_hist, n_cols_hist, hspace=0.6, wspace=0.3)
            
            show_plot = True
        else:
            ax_matrix = ax
            show_plot = False
            # If ax provided, we skip histograms or need another ax
            # For now, just plot matrix on provided ax
            
        # Calculate vmin/vmax, handling edge case where all weights are zero
        w_max = np.max(np.abs(W))
        if w_max == 0:
            vmin, vmax = -1, 1  # Default range if all weights are zero
        else:
            vmin, vmax = -w_max, w_max
        im = ax_matrix.imshow(W, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax, interpolation='none')
        if show_plot:
            plt.colorbar(im, ax=ax_matrix, label='Weight (nA)')
        
        # Grid lines for matrix
        current_offset = 0
        ticks = []
        labels = []
        for name in pop_names:
            n = len(self.populations[name])
            center = current_offset + n/2
            ticks.append(center)
            labels.append(name)
            current_offset += n
            if current_offset < total_neurons:
                ax_matrix.axhline(current_offset - 0.5, color='black', linewidth=0.5)
                ax_matrix.axvline(current_offset - 0.5, color='black', linewidth=0.5)
                
        ax_matrix.set_xticks(ticks)
        ax_matrix.set_xticklabels(labels, rotation=90)
        ax_matrix.set_yticks(ticks)
        ax_matrix.set_yticklabels(labels)
        ax_matrix.set_xlabel("Source Neuron (Pre)")
        ax_matrix.set_ylabel("Target Neuron (Post)")
        ax_matrix.set_title("Network Weight Matrix")
        
        # Plot Histograms if we created the figure
        if show_plot:
            hist_axes = []
            for i in range(n_rows_hist):
                for j in range(n_cols_hist):
                    hist_axes.append(fig.add_subplot(gs_hist[i, j]))
            
            # First pass: compute histograms to find global max count for shared y-axis
            hist_data = []
            max_count = 0
            for conn_name, weights in weights_by_conn.items():
                counts, bin_edges = np.histogram(weights, bins=30)
                max_count = max(max_count, counts.max())
                
                # Determine source type for color (E=red, I=blue)
                source_name = conn_name.split('->')[0]
                source_config = next((c for c in self.population_configs if c['name'] == source_name), None)
                if source_config is None and self.input_layer_config and source_name == self.input_layer_config.get('name', 'Layer0'):
                    source_type = 'E'  # Input layer is excitatory
                else:
                    source_type = source_config.get('type', 'E') if source_config else 'E'
                
                color = 'red' if source_type == 'E' else 'blue'
                hist_data.append((conn_name, weights, color))
            
            # Second pass: plot with shared y-axis scale
            for idx, (conn_name, weights, color) in enumerate(hist_data):
                if idx < len(hist_axes):
                    ax_h = hist_axes[idx]
                    ax_h.hist(weights, bins=30, color=color, alpha=0.7)
                    ax_h.set_title(conn_name, fontsize=8)
                    ax_h.set_xlabel('Weight (nA)', fontsize=7)
                    ax_h.set_ylim(0, max_count * 1.1)  # Same y-axis scale
                    ax_h.tick_params(labelsize=7)
            
            # Hide unused axes
            for idx in range(len(weights_by_conn), len(hist_axes)):
                hist_axes[idx].axis('off')
            
            plt.tight_layout()
            # Don't show here - save first, then show if needed
            # Return figure for caller to handle saving/displaying
            return fig
        else:
            # If ax was provided, return None (caller handles their own figure)
            return None

    def plot_raster(self, ax=None):
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            show_plot = True
        else:
            show_plot = False
            
        # We need to offset neuron indices to plot them in layers/groups
        current_offset = 0
        yticks = []
        yticklabels = []
        
        for name in self.populations: # Ordered by insertion usually
            spike_mon = self.monitors[f'spikes_{name}']
            n = len(self.populations[name])
            
            if len(spike_mon.t) > 0:
                ax.scatter(spike_mon.t/ms, spike_mon.i + current_offset, s=1, label=name)
            
            # Label position
            yticks.append(current_offset + n/2)
            yticklabels.append(name)
            
            # Draw separator
            ax.axhline(current_offset + n, color='gray', linestyle=':', alpha=0.5)
            
            current_offset += n
            
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Population")
        ax.set_title("Network Raster Plot")
        # Flip y-axis so first population appears at top
        ax.invert_yaxis()
        # ax.legend(loc='upper right', markerscale=5) # Legend might be too crowded
        
        if show_plot:
            plt.tight_layout()
            # Return figure for caller to handle saving/displaying
            return plt.gcf()
        # If ax was provided, return None (caller handles their own figure)
        return None


