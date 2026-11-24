import numpy as np



def spike_mats_per_stim(stim_type_vec, spike_matrix):
    matrix_arr =[]
    spike_matrix = np.array(spike_matrix)
    print(spike_matrix)
    stim_types = set(stim_type_vec)
    stim_types = sorted(stim_types)
    for stim in stim_types:
        index_arr = np.where(stim_type_vec == stim)[0]
        new_matrix_by_stim = spike_matrix[:, index_arr]
        matrix_arr.append(new_matrix_by_stim)
    return matrix_arr


if __name__ == '__main__':
    stim_type_vec = np.array([1, 2, 3, 1, 1, 3, 3, 3])
    spike_matrix = np.random.randint(0,2 ,size = (10, 7))
    print(spike_mats_per_stim(stim_type_vec, spike_matrix))