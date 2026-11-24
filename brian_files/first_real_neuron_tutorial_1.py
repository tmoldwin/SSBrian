from brian2 import *
import matplotlib.pyplot as plt
prefs.codegen.target = "numpy"
import brian_plotting_functions as pf

# End of tutorial 1

start_scope()

N = 5
tau = 10*ms
vr = -70*mV  #resting membrane potential in milli volts.
vt0 = -50*mV  #the inital threshold voltage of a neuron that will generate an action potential.
delta_vt0 = 5*mV  #the change of the threshold voltage.
tau_t = 100*ms  #tau = time constant that determines the rate in which a variable changes over time.
sigma = 0.5*(vt0-vr) #standard deviation of a noise term.
v_drive = 2*(vt0-vr) #the driving volatage, input strength.
duration = 100*ms #time to run the simulation

eqs = '''
dv/dt = (v_drive+vr-v)/tau + sigma*xi*tau**-0.5 : volt
dvt/dt = (vt0-vt)/tau_t : volt
'''

reset = '''
v = vr
vt += delta_vt0
'''

G = NeuronGroup(N, eqs, threshold='v> vt', reset=reset, refractory=5*ms, method='euler')
spike_monitor = SpikeMonitor(G)
voltage_monitor = StateMonitor(G, 'v', record = True)
vt_monitor = StateMonitor(G, 'vt', record = True)

G.v = 'rand()*(vt0-vr)+vr'
G.vt = vt0
run(duration)



# _ = plt.hist(spike_monitor.t / ms, 100, histtype= 'stepfilled',
#              facecolor='k', weights=list(ones(len(spike_monitor)) / (N * defaultclock.dt)))
# xlabel('Time (ms)')
# ylabel('Instantaneous firing rate (sp/s)')
# plt.show()

def plot_dashboard():
    fig, axs = plt.subplot_mosaic([['A', 'B', 'C']])
    pf.spike_raster_plot(axs['A'], spike_monitor)
    pf.spike_voltage_plot(axs['B'], voltage_monitor, [0], spike_monitor= spike_monitor, vt_monitor = vt_monitor)
    pf.firing_rate_plot(axs['C'], spike_monitor, duration, G)
    plt.show()

plot_dashboard()