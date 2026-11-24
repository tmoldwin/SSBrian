import networkx as nx

import DynamicalSystemSim as dss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from scipy.linalg import expm

#Several functions to create binary matrices of different types. E.g. Scale free, ring lattice, hierachical, etc., modular, etc.


def random_binary(N, sparsity=0.1):
    matrix = np.zeros((N, N))
    num_ones = int(sparsity * N * N)
    indices = np.random.choice(N * N, num_ones, replace=False)
    np.put(matrix, indices, 1)
    return matrix


def feed_forward(layer_sizes, noise):
    N = np.sum(layer_sizes)
    matrix = np.zeros((N, N))

    start = 0
    for i, size in enumerate(layer_sizes[:-1]):
        end = start + size
        next_start = end
        next_end = next_start + layer_sizes[i + 1]
        matrix[next_start:next_end, start:end] = 1  # Swapped indices
        start = end

    num_noise = int(noise * N * N)
    noise_indices = np.random.choice(N * N, num_noise, replace=False)
    np.put(matrix, noise_indices, 1)

    return matrix






# response_times = range(0, T)
# respones_times_to_plot = [response_times[1], response_times[int(T / 2)], response_times[-1]]

# matrix_types = [scale_free, ring_lattice, hierarchical, modular, random]
# weight_matrices = [f(N, 2) for f in matrix_types]

#visualize each matrix, the connectivity, and then the dynamics
# Define matrix generating functions and their parameters

# Fixed code snippet with customized parameters for each matrix type
layer_sizes = [10,10,10]
T = 5
N = sum(layer_sizes)
x0 = np.abs(np.random.randn(N))
matrix_parameters = [{'func':feed_forward, 'params': {'layer_sizes': layer_sizes,
                                                       'noise': i * 0.1}}
                     for i in range(10)]

fig, axes = plt.subplots(4, len(matrix_parameters), figsize=(20, 5 * len(matrix_parameters)))

# Adjusted loop to generate matrices with correct parameters
for i, experiment in enumerate(matrix_parameters):
    func = experiment['func']
    params = experiment['params']
    A = func(**params)

    # Continue with visualization and simulation as before
# Setup the figure for plotting

    # Visualize the weight matrix
    dss.plot_weight_mat(A, title=f'experiment', axis=axes[0][i])

    # Simulate dynamics
    times, x = dss.dynamical_simulator(T, x0, A, dss.BFS, dt='step')

    # Visualize dynamics
    dss.plot_dynamical_sim(times, x, axis=axes[2][i])

    # Scatterplot eigenvalues
    dss.scatterplot_eigenvalues(A, ax=axes[1][i])

plt.tight_layout()
plt.show()
# Define the weight matrices

#
# # Consolidate all information into a single dictionary
# simulations = {
#
# }
#
# fig, axes = plt.subplots(len(simulations) + 1, len(weight_matrices), figsize=(20, 22))
# for ax in axes.flat:
#     ax.xaxis.set_major_formatter(formatter)
#     ax.yaxis.set_major_formatter(formatter)
#
# for j, A in enumerate(weight_matrices):
#     dss.plot_weight_mat(A, title=f'Weight Matrix {j + 1}', axis=axes[0][j])
#     # scatterplot_eigenvalues(A, ax=axes[0][2 * j + 1])
#     for i, (key, sim) in enumerate(simulations.items()):
#         colors = plt.cm.viridis(np.linspace(0, 1, len(response_times)))
#         times, x = dss.dynamical_simulator(T, x0, A, sim['func'], dt=sim['dt'], **sim['kwargs'])
#         dss.plot_dynamical_sim(times, x, title=key, axis=axes[i + 1][j])
#         # plot_analytical_solution(response_times, x0, rfuncs, axes=axes[i + 1][j])
#         # plot_eigenvalues(rfuncs, response_times, ax=axes[i + 1][2 * j + 1])
#
# # Assuming 'axes' is a list of Axes objects and 'simulations' is your dictionary of simulations
#
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.3, left=0.1, bottom=0.08)
#
# plt.savefig('test.png')
# plt.show()
