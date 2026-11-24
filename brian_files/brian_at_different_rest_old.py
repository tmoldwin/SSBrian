from brian2 import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
prefs.codegen.target = "numpy"

start_scope()

N = 100
tau = 50*ms
v0_max = 3.
duration = 1000*ms

eqs = '''
dv/dt = (v0-v)/tau : 1 (unless refractory)
v0 : 1
'''

threshold = 1

G = NeuronGroup(N, eqs, threshold= 'v > threshold',
                reset='v=0', refractory=5*ms, method='exact')
spike_monitor = SpikeMonitor(G)
voltage_monitor = StateMonitor(G, 'v', record = True)

# Initial value of the membrane potential, noted as v for each neuron in the group.
G.v0 = 'i*v0_max/(N-1)'

run(duration)

figure(figsize=(12, 4))
subplot(131)
plot(spike_monitor.t/ms, spike_monitor.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
subplot(132)
plot(G.v0, spike_monitor.count/duration)
xlabel('v0')
ylabel('Firing rate (sp/s)')
subplot(133)
neurons = [30, 40, 50]
for i in neurons:
    plot(voltage_monitor.t/ms, voltage_monitor.v[i], label='v0 = '+str(G.v0[i]))
plt.legend()
# plot(voltage_monitor.t/ms, voltage_monitor.v[1])
# for t in voltage_monitor.t:
#     axvline(t/ms, ls='--', c='C1', lw=3)
axhline(threshold, ls=':', c='C2', lw=3)
xlabel('Time (ms)')
ylabel('v')

plt.show()
