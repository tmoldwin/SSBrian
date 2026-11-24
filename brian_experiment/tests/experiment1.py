import numpy as np
from brian2 import *
import matplotlib
from matplotlib import transforms
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
prefs.codegen.target = "numpy"
import stimuli as stim
import plots as pf
import stimuli as stim



N = 50 # Number of neurons
tau = 20 * ms  # Membrane time constant
v_rest = -70 * mV  # Resting membrane potential
num_inputs = 100 # Number of inputs
input_rate = 70 * Hz  # Input firing rate
a, c = stim.generate_bump_matrix(num_inputs, sigma=0.1)
c = 100*c


# Neuron model
eqs = '''
dv/dt = (v_rest - v + I) / tau + I / (1*ms): volt
I : volt
'''

# Create the neuron group
neurons = NeuronGroup(N, eqs, threshold='v > -50 * mV', reset='v = v_rest', method='exact')
bump_location = 0.2
num_inputs_E = int(0.7 * num_inputs)
num_inputs_I = num_inputs - num_inputs_E
input_rate_E = stim.generate_bump_stimulus(num_inputs_E, sigma=0.1, center=bump_location)[1]
input_rate_I = stim.generate_bump_stimulus(num_inputs_I, sigma=0.1, center=bump_location)[1]
input_rates = np.concatenate((input_rate_E, input_rate_I)) * Hz
P = PoissonGroup(num_inputs, rates=input_rates)


S_input = Synapses(P, neurons, 'w_poisson : volt', on_pre='v_post += w_poisson')
S_input.connect()

weight_matrix_poisson = np.random.normal(1, 1, (N, num_inputs)) * mV
weight_matrix_poisson[:, :num_inputs_E] = abs(weight_matrix_poisson[:, :num_inputs_E])
weight_matrix_poisson[:, num_inputs_E:] = -abs(weight_matrix_poisson[:, num_inputs_E:])
S_input.w_poisson = 5 * weight_matrix_poisson.T.flatten()


# Set the initial conditions
neurons.v = v_rest

# Define the synaptic connections
S_recurrent = Synapses(neurons, neurons, 'w_syn : volt', on_pre='v_post += w_syn')
S_recurrent.connect()  # Connect all neurons except self-connections

# Define the weight matrix
weight_matrix_rec = np.random.normal(1, 1, (N, N)) * mV
num_rec_E = int(0.7 * N)
weight_matrix_rec[:, :num_rec_E] = abs(weight_matrix_rec[:, :num_rec_E])
weight_matrix_rec[:, num_rec_E:] = -abs(weight_matrix_rec[:, num_rec_E:])
S_recurrent.w_syn = weight_matrix_rec.T.flatten()


# Record the membrane potential of a subset of neurons
subset_neurons = neurons
voltage = StateMonitor(subset_neurons, 'v', record=True)

# Raster plot
spikemon_rec = SpikeMonitor(neurons)
spikemon_poisson = SpikeMonitor(P)
state_monitor = StateMonitor(neurons, 'v', record=True)


#it runs for each time step this it has internal loop.
@network_operation(dt=1*ms)
def change_rates():
    #here the center changes with time
    print(stim_matrix.shape)
    rate = stim_matrix[:, int(defaultclock.t / ms)]
    P.rates = rate * Hz


def plotting_dashboard():
    fig, axs = plt.subplot_mosaic([['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']])

    # poisson group
    pf.plot_weight_matrix(axs['A'], weight_matrix_poisson, weight_matrix_poisson)
    pf.plot_external_stimuli(axs['B'], stim_trials)
    pf.raster_plot_poisson(axs['C'], spikemon_poisson,num_inputs_E, num_inputs, T)

    # recurrent group
    pf.plot_weight_matrix(axs['D'], weight_matrix_rec, weight_matrix_poisson)
    pf.plot_external_input(axs['E'], I_ext_array, T, N)
    pf.plot_raster(axs['F'], spikemon_rec, num_rec_E, T)
    pf.imshow_membrane_potential(axs['H'], state_monitor, T, N)
    pf.plot_membrane_potential(axs['I'], voltage, T)

    axs['B'].sharex(axs['C'])

    # Adjust spacing between subplots
    plt.tight_layout()
    pf.label_panels_mosaic(fig, axs, xloc=0, yloc=1.0, size=20)

if __name__ == "__main__":
    T = 600*ms
    neuron_indices = np.arange(0, 10, 1)
    T_start = [] * N * ms
    T_end = T_start + 5 * ms
    stim_amp = [] * N
    I_ext_array = stim.step_stim(N, T, [], [], [], [])
    stim_matrix = stim.build_P_external_input(T/ms, num_inputs_E, num_inputs_I)
    stim_list = np.array([0.1])
    stim_duration = 30
    num_inputs_I = 100
    num_inputs_E = 100
    break_time = 50
    stim_trials = stim.present_bump_trials(num_inputs_E, num_inputs_I, T / ms, stim_duration, stim_list, break_time)
    run(T)
    plotting_dashboard()