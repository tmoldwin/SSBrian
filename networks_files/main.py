import numpy as np
import math
import seaborn as sns
from numpy.linalg import eig
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import animation, colors
sigmoid = lambda x: 1 / (1+math.exp(-x))
import plotting_functions as pf
import network_architectures as na
import stimuli as inp
import networks as net

# This is a sample Python script.


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # weights = [1, 1, 8]
    # input = [0.3, 0.6, 0.2]
    # neuron = Neuron(weights, "sigmoid")
    # output = neuron.output(input, "sigmoid")
    # print("Neuron output =", output)
    # weight_matrix = [[0.5, -0.9, 1], [0.1, 0.3, 1], [0.4, 0.3, 0.2]]
    # weight_matrix = np.array([[0.5, -3, 0.4], [0.1, 0.3, -0.3], [-0.4, 0.3, 0.2]])
    T = 50
    for i in range(5):
        layer_lengths = [2, 3, 2, 1]
        # color_list = ['k', 'k', 'g', 'g', 'g', 'b', 'b', 'r']
        weight_matrix = na.gen_recurrent_ff_net_old(2, 3, random_weights = True)
        stim = inp.single_stim(1, T, 5, np.array([1, 2]))
        y_0 = np.zeros(len(weight_matrix[0]))
        # y_0[[0, 1]] = [0.1, 0.3]
        # weight_matrix = np.array([[2, 0], [0, 4]])
        network = net.Network(weight_matrix, activation_func = "sigmoid", neuron_colors = None)
        activity_mat = network.simulate(y_0, T, input = stim)
        network.plot_dashboard()




        #TODO plot the eigenvalues on the top of this plot. like in the paper with Lambda.
        # pf.plot_eigenvalues(network.eigenvalues, ax)


    plt.show()
    position_mat = np.array([[1, 0], [0, 1]])
    # activity_video(np.array(weight_matrix), np.array(activity_mat), np.array(position_mat))








# See PyCharm help at https://www.jetbrains.com/help/pycharm/
