import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from brian2 import *

FIG_WIDTH = 7.5
TINY_SIZE = 6
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

#show them all in mosaic plot.
#TODO all the plotting fucntions should be in one file
# raster plot - make it general. takes axes, and spikemonitor.
# voltage plot - make it general. takes axes, and voltagemonitor.
# firing rate plot - make it general. takes axes, and spikemonitor.

# is it not ok to put x, y in argument ?
def spike_raster_plot(ax, spike_monitor):
    spike_times = spike_monitor.t
    neuron_indices = spike_monitor.i
    ax.plot(spike_times/ms , neuron_indices, 'k.')
    ax.set_xlabel('TIME')
    ax.set_ylabel('NEURON INDEX')
    ax.set_title('RASTER PLOT')


# voltage_monitor.t = the time point at which the membrane
# potential was recorded during the simulation.
# voltage_monitor.v[i] = membrane potential of neuron i.
def spike_voltage_plot(ax, voltage_monitor, neurons, spike_monitor=None, vt_monitor = None):
    time_point = voltage_monitor.t / ms
    ax.set_xlabel('TIME')
    ax.set_ylabel('NEURON INDEX')

    for i in neurons:
        line_color = next(ax._get_lines.prop_cycler)['color']  # Get the color of the line plot
        ax.plot(time_point, voltage_monitor.v[i], label=str(i), color=line_color)

        if vt_monitor is not None:
            ax.plot(time_point, vt_monitor.vt[i], label=str(i), color=line_color, ls=':')


        if spike_monitor is not None:
            # Access the spike times of neuron i
            spike_times_i = spike_monitor.t[spike_monitor.i == i]
            ax.scatter(spike_times_i/ms, [-0.04]*len(spike_times_i), color=line_color, marker='o')

    # ax.axhline(threshold, ls=':', c='C2', lw=3)
    plt.legend(bbox_to_anchor=(0.5, -0.2))
    ax.set_title('SPIKE VOLTAGE PLOT')


#it will be a figure that describes the firing rate if neurons,
# associated with their initial potential.
def firing_rate_plot(ax, spike_monitor, duration, G):
    firing_rate = spike_monitor.count / duration
    ax.set_xlabel('INITIAL POTENTIAL - v0')
    ax.set_ylabel('FIRING RATE')
    ax.plot(firing_rate)
    ax.set_title('FIRING RATE PLOT')





