import matplotlib.pyplot as plt
import networks as net
import numpy as np
import stimuli as stimuli
import stimuli_selectivity_analysis as ssa


if __name__ == '__main__':
    input_size = 10
    recurrent_size = 10
    N = input_size + recurrent_size
    T = 100
    ff_rec_network = net.FF_Recurrent_Network(input_size, recurrent_size, activation_func='sigmoid', random_weights=True, mean =0, variance=0.1)
    y_0 = np.random.randn(input_size + recurrent_size)
    y_0[0:input_size] = 0
    num_stims = 3
    stim_list = stimuli.binary_stim(N, num_stims)
    sequence = stimuli.rand_seq_func(T, num_stims, spacing=10, start=10)
    stim = stimuli.stimgen(stim_list, sequence, variance=0.1)
    activity_mat = ff_rec_network.simulate(y_0, T, stim)
    ff_rec_network.plot_dashboard()
    average_selectivity_mat = ssa.stimulus_response_average(activity_mat, sequence, averaging_window=5)
    ssa.plot_selectivity_dashboard(average_selectivity_mat, sequence)
    ssa.stim_neural_trial_response(activity_mat[input_size:,:], sequence, time_before=2, time_after=5)
    plt.show()





