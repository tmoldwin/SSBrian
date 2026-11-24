import matplotlib.pyplot as plt
from brian2 import *
prefs.codegen.target = "numpy"
import homework_plotting_func as pf
import brian_plotting_functions as bpf

start_scope()

taupre = 10*ms
taupost = 20*ms
wmax = 1
Apre = 0.01
Apost = -Apre*taupre/taupost*1.05
delta_w = 0.0
tau = 5*ms
sigma = 0.01


eqs = '''
dv/dt = (1-v)/tau +sigma*xi*tau**-0.5 : 1 (unless refractory)
tau : second
'''

G = NeuronGroup(2, eqs, threshold='v>0.8', reset='v = 0',
               refractory= 1*ms, method='euler')
G.tau = [12*ms, 5*ms]

S = Synapses(G, G,
             '''
             w : 1 
             dapre/dt = -apre/taupre : 1 (clock-driven)
             dapost/dt = -apost/taupost : 1 (clock-driven)
             ''',
             on_pre='''
             v_post += w
             apre += Apre
             w = clip(w+apost, 0, wmax)
             ''',
             on_post='''
             apost += Apost
             w = clip(w+apre, 0, wmax)
             ''', method='linear')

S.connect(i=0, j=1)
S.w = 0.1

M = StateMonitor(S, ['w', 'apre', 'apost'], record=True)

spike_monitor = SpikeMonitor(G)
voltage_monitor = StateMonitor(G, 'v', record=True)
synapse_state_monitor = StateMonitor(S, 'w', record=True)
apre_monitor = StateMonitor(S, 'apre', record=True)
apost_monitor = StateMonitor(S, 'apost', record=True)



def weight_function(delta_t, Apre, taupre, Apost, taupost):
    if delta_t > 0:
        return Apre * (np.exp(-delta_t / taupre))
    elif delta_t < 0:
        return -Apost * (np.exp(delta_t / taupost))
    else:
        return 0.0


def final_weight_change(pre_spike_times, post_spike_times, delta_w):
    for i in range(len(pre_spike_times)):
        for j in range(len(post_spike_times)):
            delta_w += weight_function(post_spike_times[j] - pre_spike_times[i])
    return delta_w


run(100*ms)


def plot_dashboard():
    fig, ax = plt.subplot_mosaic([['A'], ['B'], ['C'], ['D'], ['E']], sharex = True,
                                 constrained_layout=True)
    # pf.spike_plotting(ax['A'], spike_monitor, 0)
    # pf.spike_plotting(ax['A'], spike_monitor, 1)
    bpf.spike_raster_plot(ax['A'], spike_monitor)
    pf.voltage_plotting(ax['B'], voltage_monitor, 0)
    pf.voltage_plotting(ax['B'], voltage_monitor, 1)
    pf.weight_plotting(ax['C'], synapse_state_monitor)
    pf.apre_plotting(ax['D'], apre_monitor)
    pf.apost_plotting(ax['E'], apost_monitor)
    plt.show()


plot_dashboard()

#TODO: figure out how to get the final weight change, and print it out.

#calling the functions here :
pre_spike_times = spike_monitor.t[spike_monitor.i == 0]
post_spike_times = spike_monitor.t[spike_monitor.i == 1]
delta_w = final_weight_change(pre_spike_times, post_spike_times, delta_w)
print("final value of weight: %s" % delta_w)
print(synapse_state_monitor.w[-1])