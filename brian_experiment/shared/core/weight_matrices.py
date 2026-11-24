import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
import plots as pf


#TODO - pass the rows and cols only, and make the another join function for the poisson and rec matrices.
def weight_matrix_factory(mat_dict, dist_dict):
    wm = None
    n_rec_e = n_poiss_e = rows = cols = 0
    if 'recurrent' in mat_dict:
        n_rec_e = mat_dict['recurrent'][0]
        n_rec_i = mat_dict['recurrent'][1]
        rows = cols = n_rec_i + n_rec_e
    elif 'poisson' in mat_dict:
        n_rec_e = mat_dict['poisson'][0]
        n_rec_i = mat_dict['poisson'][1]
        n_poiss_e = mat_dict['poisson'][2]
        n_poiss_i = mat_dict['poisson'][3]
        rows = n_rec_i + n_rec_e
        cols = n_poiss_i + n_poiss_e

    wm = gen_by_dist(dist_dict, rows, cols)

    return weight_mat_abs(wm, n_poiss_e) if 'poisson' in mat_dict else weight_mat_abs(wm, n_rec_e)

def gen_by_dist(dist_dict, rows, cols):
    if 'normal' in dist_dict:
        wm = np.random.normal(dist_dict['normal'][0], dist_dict['normal'][1], (rows, cols))
    elif 'uniform' in dist_dict:
        wm = np.random.uniform(dist_dict['uniform'][0], dist_dict['uniform'][1], (rows, cols))
    elif 'exponential' in dist_dict:
        wm = np.random.exponential(dist_dict['exponential'], (rows, cols))
    elif 'lognormal' in dist_dict:
        wm = np.random.lognormal(dist_dict['lognormal'][0],dist_dict['lognormal'][1], dist_dict['lognormal'][2])
    return wm

def weight_mat_abs(wm, n_e):
    wm[:, :n_e] = abs(wm[:, :n_e])
    wm[:, n_e:] = -abs(wm[:, n_e:])
    return wm

#TODO - see different distributions for the weight matrix in the plot.
#TODO - in figure 2 - weight matricx, the distribution of the matrix in hist,
# and the connectivity structure with time.
# with the networkx (or something else).
def plotting_dashboard_for_matrices(synapses, wm1, wm2, spike_matrix):
    fig, axs = plt.subplot_mosaic([['A', 'B', 'C'], ['D', 'E', 'F']], figsize=(15, 7))
    pf.plot_weight_matrix(axs['A'], wm1, wm2)
    pf.plot_histogram(axs['B'], wm1)
    pf.plot_weight_matrix(axs['C'], spike_matrix, wm2)
    # figure unrolling with time
    pf.visualise_connectivity(synapses, wm1)





