from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm



def euler_method(Tau, k, v_0, T):
    voltage = [v_0]
    dv_dt = []
    time = np.arange(0, T, 1)
    for t in time:
        dv_dt.append((k - voltage[t])/Tau)
        voltage.append(voltage[t] + dv_dt[t])
    return time, dv_dt, voltage[:-1]


def scatter_plot_v(ax, voltage, time, label=None, color='k'):
    ax.scatter(time, voltage, label=label, color=color)
    ax.set_xlabel("time")
    ax.set_ylabel("voltage")


def scatter_plot_eqs(ax, dv_dt, time):
    ax.scatter(time, dv_dt)
    ax.set_xlabel("time")
    ax.set_ylabel("dv_dt")
    ax.set_title("dv_dt values")

def same_tau_different_k(Tau, k_list, v_0, T, ax):
    cmap = cm.get_cmap('coolwarm')
    normalize = plt.Normalize(vmin=min(k_list), vmax=max(k_list))
    colors = [cmap(normalize(k)) for k in k_list]
    for i,k in enumerate(k_list):
        time, dv_dt, voltage = euler_method(Tau, k, v_0, T)
        scatter_plot_v(ax, voltage, time, label="v_rest = " + str(k), color=colors[i])
    # ax.legend(fontsize='small')
    ax.cbar = plt.colorbar(cm.ScalarMappable(norm=normalize, cmap=cmap), ax=ax)
    ax.cbar.set_label('v_rest')

def same_k_different_tau(Tau_list, k, v_0, T, ax):
    cmap = cm.get_cmap('coolwarm')
    normalize = plt.Normalize(vmin=min(Tau_list), vmax=max(Tau_list))
    colors = [cmap(normalize(Tau)) for Tau in Tau_list]
    for i,Tau in enumerate(Tau_list):
        time, dv_dt, voltage = euler_method(Tau, k, v_0, T)
        scatter_plot_v(ax, voltage, time, label="Tau = " + str(Tau), color=colors[i])
    ax.legend(fontsize='small')

def same_tau_and_k_different_v0(Tau, k, v_0_list, T, ax):
    cmap = cm.get_cmap('coolwarm')
    normalize = plt.Normalize(vmin=min(v_0_list), vmax=max(v_0_list))
    colors = [cmap(normalize(v_0)) for v_0 in v_0_list]
    for i,v_0 in enumerate(v_0_list):
        time, dv_dt, voltage = euler_method(Tau, k, v_0, T)
        scatter_plot_v(ax, voltage, time, label="v_0 = " + str(v_0), color = colors[i])
    ax.legend(fontsize='small')


def plot_dashboard():
    fig, ax = plt.subplot_mosaic([['A', 'B'], ['C', 'D'], ['E', 'F']], constrained_layout=True)
    scatter_plot_v(ax['A'], voltage, time, )
    scatter_plot_eqs(ax['B'], dv_dt, time)
    v_0 = 0
    same_tau_different_k(Tau, k_list, v_0, T, ax['C'])
    same_k_different_tau(Tau_list, k, v_0, T, ax['D'])
    same_tau_and_k_different_v0(Tau, k, v_0_list, T, ax['E'])
    plt.show()



Tau = 10
k = -70
T = 60
v_0 = -20
k_list = [-20, -10, 0, 10, 20]
Tau_list = [10, 20, 30, 40, 50]
v_0_list = [-20, -10, 0, 10, 20]


time, dv_dt, voltage = euler_method(Tau, k, v_0, T)
plot_dashboard()

