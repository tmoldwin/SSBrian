from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

start_scope()

# prefs.codegen.target = "numpy"
start_scope()
tau = 5*ms
eqs = '''
dv/dt = (1-v)/tau : 1 (unless refractory)
'''

G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0',
               refractory= 1*ms, method='exact')
# M is the voltage monitor.
voltage_monitor = StateMonitor(G, 'v', record=True)
spike_monitor = SpikeMonitor(G)
run(30*ms)

# these are horizontal and vertical lines
# drawn on the plot respectively.
for t in spike_monitor.t:
    axvline(t/ms, ls='--', c='C1', lw=3)
axhline(0.8, ls=':', c='C2', lw=3)

print("Spike times: %s" % spike_monitor.t[:])

plot(voltage_monitor.t / ms, voltage_monitor.v[0])
xlabel('Time (ms)')
ylabel('V')
plt.show()