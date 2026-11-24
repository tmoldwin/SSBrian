import numpy as np
import matplotlib.pyplot as plt
import plotting_functions as pf

#each of these functions will be a way to produce a weight matrix

#layer_sizes is a list of the number of neurons in each layer
#make a list for each layers, and then make a list of lists, ans then

def gen_rec_net(distribution = "normal", mean = 0, variance = 1,
                dale = True, inh_fraction = 0.2, total_neurons = 10):
    if distribution == "normal":
        weight_matrix = np.random.normal(mean, np.sqrt(variance), (total_neurons, total_neurons))
    if dale:
        weight_matrix =  np.abs(weight_matrix)
        weight_matrix[:, 0:int(inh_fraction*total_neurons)] = -1 * weight_matrix[:, 0:int(inh_fraction*total_neurons)]
    return weight_matrix

def gen_recurrent_ff_net(input_size, recur_weight_matrix, input_weight_init = "ones"):
    total_neurons = input_size + len(recur_weight_matrix)
    weight_matrix = np.zeros((total_neurons, total_neurons))
    weight_matrix[input_size:, input_size:] = recur_weight_matrix
    weight_matrix[input_size:, 0:input_size] = 1
    if input_weight_init == "random":
        weight_matrix[input_size:, 0:input_size] = np.random.normal(0, 1,  (len(recur_weight_matrix), input_size))
    print(weight_matrix)
    return weight_matrix


def gen_recurrent_ff_net_old(input_size, recurrent_size, random_weights = False, mean = 0, variance = 1):
    total_neurons = input_size + recurrent_size
    weight_matrix = np.zeros((total_neurons, total_neurons))
    weight_matrix[input_size:total_neurons, :] = 1
    if random_weights:
        new_mat = np.random.normal(mean, np.sqrt(variance), (total_neurons, total_neurons))
        weight_matrix = weight_matrix * new_mat
    return weight_matrix

def gen_ff_net(layer_sizes, lateral_inhibition = False, random_weights = False, mean = 0, variance = 1):
    total_neurons = np.sum(layer_sizes)
    weight_matrix = np.zeros((total_neurons, total_neurons))
    layers = []
    layers.append(np.arange(0, layer_sizes[0]))
    for i in range(1, len(layer_sizes)):
        start = layers[i - 1][-1] + 1
        end = start + layer_sizes[i]
        layers.append(np.arange(start, end))
    print(layers)
    for i in range(len(layers) - 1):
        for j in layers[i]:
            for k in layers[i + 1]:
                weight_matrix[k][j] = 1
            if lateral_inhibition:
                for k in layers[i]:
                    if k != j:
                        weight_matrix[k][j] = -1
    if random_weights:
        new_mat = np.random.normal(mean, np.sqrt(variance), (total_neurons, total_neurons))
        weight_matrix = weight_matrix * new_mat
    return weight_matrix

# def gen_recurrent_ff_net_old(input_size, recurrent_size,  input_weight_init = "random",
#                          recur_weight_init = "random", mean = 0, variance = 1,
#                          inh_fraction = 0.2, distribution = "normal", dale = True):
#     total_neurons = input_size + recurrent_size
#     if input_weight_init == "ones_mirrored":
#         total_neurons = input_size * 2 + recurrent_size
#     weight_matrix = np.zeros((total_neurons, total_neurons))
#     if input_weight_init == "normal":
#         new_mat = np.random.normal(mean, np.sqrt(variance), (total_neurons, total_neurons))
#         weight_matrix = weight_matrix * new_mat
#     return weight_matrix



if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2)
    weight_matrix_1 = gen_rec_net(dale = True)
    weight_matrix_2 = gen_rec_net(dale = False)
    weight_matrix_3 = gen_rec_ff_net(3, weight_matrix_1)
    weight_matrix_4 = gen_rec_ff_net(3, weight_matrix_2, input_weight_init = "random")
    pf.plot_real_heatmap(weight_matrix_1, axes[0][0])
    pf.plot_real_heatmap(weight_matrix_2, axes[0][1])
    pf.plot_real_heatmap(weight_matrix_3, axes[1][0])
    pf.plot_real_heatmap(weight_matrix_4, axes[1][1])
    plt.show()
