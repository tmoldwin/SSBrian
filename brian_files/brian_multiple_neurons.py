from brian2 import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
prefs.codegen.target = "numpy"

start_scope()

N = 100
tau = 10*ms
eqs = '''
dv/dt = (2-v)/tau : 1
'''

# N is the number of neurons.
G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', method='exact')

#assigns random initial values for the initial voltage.
G.v = 'rand()'

spike_monitor = SpikeMonitor(G)
run(50*ms)

# spike_monitor.i: This attribute represents the indices of the neurons that have spiked.
# It provides a list or array of integers indicating the indices of the neurons that fired action potentials.
# spike_monitor.t: This attribute represents the times at which the spikes occurred.
# It provides a list or array of time values indicating the times at which the corresponding neurons
# fired action potentials.
plot(spike_monitor.t / ms, spike_monitor.i, '.c')
xlabel('Time (ms)')
ylabel('Neuron Index')

print(spike_monitor.i)
print(len(spike_monitor.i))
print(len(spike_monitor.t))
plt.show()

