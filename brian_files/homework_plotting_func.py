import matplotlib.pyplot as plt
import numpy as np
from brian2 import *


def spike_plotting(ax, spike_monitor, neuron_index):
    spike_times = spike_monitor.t[spike_monitor.i == neuron_index]
    #first arg is x-axis, second arg is y-axis, and the size should be the same.
    ax.plot(spike_times/ms, np.ones_like(spike_times))
    ax.set_xlabel("time")
    ax.set_title("spike times of neuron " + str(neuron_index))


def voltage_plotting(ax, voltage_monitor, neuron_index):
    ax.plot(voltage_monitor.t/ms, voltage_monitor.v[neuron_index])
    ax.set_xlabel("time")
    ax.set_title("voltage of neuron " + str(neuron_index))


def weight_plotting(ax, synaptic_state_monitor):
    ax.plot(synaptic_state_monitor.t/ms, synaptic_state_monitor.w[0])
    ax.set_xlabel("time")
    ax.set_ylabel("weight")
    ax.set_title("synaptic weight of synapse (0, 1)")
    # print("weights are: %s" % synaptic_state_monitor.w[0])

def apre_plotting(ax, apre_monitor):
    ax.plot(apre_monitor.t/ms, apre_monitor.apre[0])
    ax.set_xlabel("time")
    ax.set_ylabel("apre")
    ax.set_title("apre of synapse (0, 1)")
    # print("apre are: %s" % apre_monitor.apre[0])

def apost_plotting(ax, apost_monitor):
    ax.plot(apost_monitor.t/ms, apost_monitor.apost[0])
    ax.set_xlabel("time")
    ax.set_ylabel("apost")
    ax.set_title("apsot of synapse (0, 1)")
    # print("apost are: %s" % apost_monitor.apost[0])



