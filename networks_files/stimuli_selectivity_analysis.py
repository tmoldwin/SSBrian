import numpy as np
import matplotlib.pyplot as plt
from networks_files import plotting_functions as pf


def find_indeices(sequence, number):
    indices = []
    for i in range(len(sequence)):
        if sequence[i] == number:
            indices.append(i)
    return indices


def stimulus_response_average(activity_matrix, sequence, averaging_window = 5):
    N = activity_matrix.shape[0]
    T = len(activity_matrix[0])
    num_stimuli = len(set(sequence)) -1
    average_selectivity_mat = np.zeros((N, num_stimuli))
    baseline = np.mean(activity_matrix, axis = 1)
    # for neuron in range(N):
    for number in range(num_stimuli):
        sum = 0
        indices = np.where(sequence == number)[0]
        for index in indices:
            sum += np.sum(activity_matrix[:, index:index + averaging_window], axis = 1)
        if len(indices) == 0:
            average = np.zeros(N)
        else:
            average = sum/len(indices)
        average_selectivity_mat[:, number] = average - baseline
    return average_selectivity_mat


def stim_neural_trial_response(activity_matrix, sequence, time_before =2, time_after=5):
    N = activity_matrix.shape[0]
    T = len(activity_matrix[0])
    num_stimuli = len(set(sequence)) -1
    fig, axs = plt.subplots(N, num_stimuli, sharex =True)
    baseline = np.mean(activity_matrix, axis = 1)
    for neuron in range(N):
        for number in range(num_stimuli):
            indices = np.where(sequence == number)[0]
            trial_mat = np.zeros((len(indices), time_before + time_after +1))
            for i, index in enumerate(indices):
                trial_mat[i, :] = activity_matrix[neuron, index - time_before: index + time_after + 1] - baseline[neuron]
            pf.plot_real_heatmap(trial_mat, axs[neuron, number], annot = False)



def average_response_heatmap(average_selectivity_mat, ax,  annot = False):
    pf.plot_real_heatmap(average_selectivity_mat, ax, annot=annot)
    ax.set_title("average selectivity matrix")


def plot_selectivity_dashboard(average_selectivity_mat, sequence):
    fig, axs = plt.subplot_mosaic([['A', 'B'], ['C', 'D'], ['E', 'F']],
                                  layout='constrained')
    average_response_heatmap(average_selectivity_mat, axs["A"])


















