import numpy as np
from matplotlib import pyplot as plt
import math
sigmoid = lambda x: 1 / (1+math.exp(-x))
from networks_files import network_architectures as na
import stimuli as inp
import plotting_functions as pf
from numpy.linalg import eig



class Network:
    #i define the feilds in the init function
    def __init__(self, weight_matrix, activation_func = "linear", neuron_colors = None):
        self.weight_matrix = weight_matrix
        self.M = weight_matrix - np.identity(len(weight_matrix[0]))
        self.eigenvalues, self.eigenvectors = eig(self.M)
        #creates a diagonal matrix with eig values
        self.eig_diagonal_mat = np.diag(self.eigenvalues)
        self.N = len(self.weight_matrix[0])
        self.activation_func = activation_func
        self.activity_mat = None
        self.neuron_colors = neuron_colors


    def activation(self, x):
        if self.activation_func == "sigmoid":
            return 1 / (1 + np.exp(-x))
        if self.activation_func == "linear":
            return x
        if self.activation_func == "tanh":
            return np.tanh(x)

    def simulate(self, y_0, T, input = None):
        if input is None:
            input = np.zeros((self.N , T))
        activity_mat = np.zeros((self.N, T))
        activity_mat[:, 0] = y_0 + input[:, 0]
        for t in range(1, T):
            # activity_mat[:, t] = self.activation(np.dot(self.weight_matrix, activity_mat[:, t-1]))
            activity_mat[:, t] = self.activation(np.dot(self.M, activity_mat[:, t-1]) + activity_mat[:, t-1]) + input[:, t]
        self.activity_mat = activity_mat
        self.stim_mat = input
        return activity_mat

    def simulate_eigen_basis(self, y_0, T):
        activity_mat = np.zeros((self.N, T), dtype=complex)
        np.seterr(all='ignore')
        activity_mat[:, 0] = np.dot(self.eigenvectors, y_0)
        for t in range(1, T):
            # activity_mat[:, t] = self.activation(np.dot(self.weight_matrix, activity_mat[:, t-1]))
            activity_mat[:, t] = self.activation(np.dot(self.eig_diagonal_mat, activity_mat[:, t-1]) + activity_mat[:, t-1])
        self.eigen_activity_mat = activity_mat
        return activity_mat

    def plot_weight_matrix(self, ax, annot = False):
        pf.plot_real_heatmap(self.weight_matrix, ax, annot=annot)
        ax.set_title("weight matrix")

    def plot_activity_heat_map(self, ax, annot = False):
        pf.plot_real_heatmap(self.activity_mat, ax, annot=annot)
        ax.set_title("activity matrix")

    def plot_activity_mat(self, ax, annot = False, legend = False, colors = None, colormap='viridis'):
        ax.plot(self.activity_mat.T)
        num_lines = self.activity_mat.shape[0]
        if colors is None:
            cmap = plt.get_cmap(colormap)
            colors = [cmap(i / num_lines) for i in range(num_lines)]
        for i in range(self.activity_mat.shape[0]):
            ax.plot(self.activity_mat[i], label=f'Line {i + 1}', color = colors[i])
        if legend:
            ax.legend()

    def plot_eigenvalues(self, ax, annot = False):
        pf.plot_complex_heat_map(self.eig_diagonal_mat, ax, annot=annot)
        ax.set_title("eigenvalues")

    def plot_eigenvectors(self, ax, annot = True):
        pf.plot_complex_heat_map(self.eigenvectors, ax, annot=annot)
        ax.set_title("eigenvectors")

    def plot_eigenvalues_distribution(self, ax, annot = False):
        pf.plot_eigenvalues(self.eigenvalues, ax)
        ax.set_title("eigenvalues distribution")

    def plot_eigenbasis_activity(self, ax, annot = False):
        ax.plot(self.eigen_activity_mat.T)
        ax.set_title("eigenbasis activity")


    def eig_phase_plane_plot_real(self, ax):
        pf.phase_plane_plot(ax, pf.derivative_fn, np.real(self.eig_diagonal_mat))
        ax.set_title("eig real")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.set_title("eig real")
    def eig_phase_plane_plot_imag(self, ax):
        pf.phase_plane_plot(ax, pf.derivative_fn, np.imag(self.eig_diagonal_mat))
        ax.set_title("eig imag")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.set_title("eig imag")
    def phase_plain_plot_weight_matrix(self, ax):
        # pf.phase_plane_plot(ax, pf.derivative_fn, self.weight_matrix)
        pf.phase_plane_plot_3D()
        ax.set_title("weight matrix")

    def plot_stim(self, ax):
        pf.plot_real_heatmap(self.stim_mat, ax, annot = False)
        ax.set_title("stim_map")


    def plot_dashboard(self):
        fig, axs = plt.subplot_mosaic([['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']],
                                      layout='constrained')
        self.plot_weight_matrix(axs['A'])
        self.plot_activity_mat(axs['B'], colors=self.neuron_colors)
        self.plot_activity_heat_map(axs['C'])
        self.plot_stim(axs['F'])
        self.eig_phase_plane_plot_real(axs['H'])
        self.eig_phase_plane_plot_imag(axs['I'])


class Recurrent_Network(Network):

    def __init__(self, weight_matrix = None, activation_func = "linear", N =2, mean = 0, variance = 1):
        if weight_matrix is None:
            weight_matrix = np.random.normal(mean, variance, (N, N))
        super().__init__(weight_matrix, activation_func)
        self.N = N

    def plot_dashboard(self):
        fig, axs = plt.subplot_mosaic([['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']],
                                     layout='constrained')
        self.plot_weight_matrix(axs['A'], annot = True)
        self.plot_activity_mat(axs['B'])
        # self.plot_activity_heat_map(axs['C'])
        self.phase_plain_plot_weight_matrix(axs['C'])
        self.plot_eigenvectors(axs['D'])
        self.plot_eigenvalues(axs['E'])
        self.plot_eigenvalues_distribution(axs['F'])
        self.plot_eigenbasis_activity(axs['G'])
        self.eig_phase_plane_plot_real(axs['H'])
        self.eig_phase_plane_plot_imag(axs['I'])

class FF_Network(Network):

    def __init__(self, layer_sizes, activation_func = "tanh", lateral_inhibition = False, random_weights = True, colormap = 'tab20'):
        weight_matrix = na.gen_ff_net(layer_sizes, lateral_inhibition = False, random_weights = True)
        super().__init__(weight_matrix, activation_func = activation_func)
        num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i / num_layers) for i in range(num_layers)]
        self.color_list = []
        for i in range(num_layers):
            self.color_list += [colors[i]] * layer_sizes[i]

    def plot_dashboard(self):
        fig, axs = plt.subplot_mosaic([['A', 'B'], ['C', 'D']],
                                      layout='constrained')
        self.plot_weight_matrix(axs['A'])
        self.plot_activity_mat(axs['C'], colors=self.color_list)
        self.plot_activity_heat_map(axs['B'])
        self.plot_stim(axs['D'])


class FF_Recurrent_Network(Network):

    def __init__(self, input_size, recurrent_size, activation_func = "sigmoid" ,random_weights = False, mean = 0, variance = 1):
        weight_matrix = na.gen_rec_ff_net(input_size, recurrent_size, input_weight_init=random_weights, mean = mean, variance = variance)
        super().__init__(weight_matrix, activation_func = activation_func)
        self.input_size = input_size
        self.recurrent_size = recurrent_size

    def weight_matrix_histogram(self, ax):
        ax.hist(self.weight_matrix[self.input_size:].flatten(), bins=40)
        ax.set_title("weight matrix histogram")

    def plot_activity_mat(self, ax, annot = False, legend = False, colors = None, colormap='viridis'):
        print(self.input_size)
        recur_activity_mat = self.activity_mat[self.input_size:]
        num_lines = self.activity_mat.shape[0]
        if colors is None:
            cmap = plt.get_cmap(colormap)
            colors = [cmap(i / num_lines) for i in range(num_lines)]
        for i in range(recur_activity_mat.shape[0]):
            ax.plot(recur_activity_mat[i], label=f'Line {i + 1}', color = colors[i])
        if legend:
            ax.legend()

    def plot_dashboard(self):
        fig, axs = plt.subplot_mosaic([['A', 'B'], ['C', 'D'], ['E', 'F']],
                                      layout='constrained')
        self.plot_weight_matrix(axs['A'])
        self.plot_activity_mat(axs['C'])
        self.plot_activity_heat_map(axs['B'])
        self.plot_stim(axs['D'])
        self.weight_matrix_histogram(axs['E'])



if __name__ == '__main__':
    #recurrent_network, random init.
    N = 2
    T = 10
    # y_0 = np.random.randn(N)
    # recurrent_network = Recurrent_Network(N = N, mean = 0, variance = 0.5)
    # activity_mat = recurrent_network.simulate(y_0, T)
    # eig_activity_mat = recurrent_network.simulate_eigen_basis(y_0, 100)
    # recurrent_network.plot_dashboard()
    #
    # #recurrent_network, matrix init.
    # weight_matrix = np.array([[1, 1], [2, -0.5]])
    # recurrent_network = Recurrent_Network(weight_matrix)
    # activity_mat = recurrent_network.simulate(y_0, T)
    # eig_activity_mat = recurrent_network.simulate_eigen_basis(y_0, 100)
    # recurrent_network.plot_dashboard()




    #feedforward network, random init.
    #
    # layer_sizes = [2, 3, 2, 1]
    # N = np.sum(layer_sizes)
    # y_0 = np.zeros(np.sum(layer_sizes))
    # ff_network = FF_Network(layer_sizes, random_weights=True)
    # stim = inp.single_stim(2, T, N, 0.1 * np.random.randn(layer_sizes[0]))
    # activity_mat = ff_network.simulate(y_0, T, input=stim)
    # ff_network.plot_dashboard()

    #ff_recurrent_network, random init.
    input_size = 10
    recurrent_size = 50
    N = input_size + recurrent_size
    T = 100
    ff_rec_network = FF_Recurrent_Network(input_size, recurrent_size, activation_func= 'sigmoid', random_weights=True, mean = 0, variance = 0.1)
    y_0 = np.random.randn(input_size + recurrent_size)
    y_0[0:input_size] = 0
    stim = inp.single_stim(50, T, N, np.random.randn(input_size))
    activity_mat = ff_rec_network.simulate(y_0, T, stim)
    ff_rec_network.plot_dashboard()



    plt.show()