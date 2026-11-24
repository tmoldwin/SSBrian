from brian2 import *
import matplotlib.pyplot as plt
prefs.codegen.target = "numpy"
import brian_plotting_functions as pf


start_scope()

N = 100
tau = 10*ms
v0_max = 3.
duration = 1000*ms
sigma = 0.02

# eqs = '''
# dv/dt = (v0-v)/tau : 1 (unless refractory)
# v0 : 1
# '''
eqs = '''
dv/dt = (v0-v)/tau+sigma*xi*tau**-0.5 : 1 (unless refractory)
v0 : 1
'''

# G = NeuronGroup(N, eqs, threshold='v>1', reset = 'v=0', refractory= 5*ms, method = 'exact')
G = NeuronGroup(N, eqs, threshold='v>1', reset = 'v=0', refractory= 5*ms, method = 'Euler')
spike_monitor = SpikeMonitor(G)
voltage_monitor = StateMonitor(G, 'v', record = True)

G.v0 = 'i*v0_max/(N-1)'

run(duration)

def plot_dashboard():
    fig, axs = plt.subplot_mosaic([['A', 'B', 'C']])
    pf.spike_raster_plot(axs['A'], spike_monitor)
    pf.spike_voltage_plot(axs['B'], voltage_monitor, G, [10, 15, 20])
    pf.firing_rate_plot(axs['C'], spike_monitor, duration, G)
    plt.show()


plot_dashboard()
