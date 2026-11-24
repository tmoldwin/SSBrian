import numpy as np

def single_stim(stim_time, T, N, vec):
    stim = np.zeros((N,T))
    stim[0: len(vec), stim_time] = vec
    return stim

def stimgen(stim_list, sequence):
    stim = np.zeros((stim_list.shape[1], len(sequence)))
    for i in range(len(sequence)):
        if sequence[i] == -1:
            stim[:, i] = np.zeros(stim_list.shape[1])
        else:
            stim[:, i] = stim_list[sequence[i]]
    return stim

def binary_stim(N,num_stims):
    stim_list = np.zeros((num_stims, N))
    for i in range(num_stims):
        stim_list[i, i] = 1
    return stim_list

def rand_seq_func(T, num_stims, spacing = 2, start = 10):
    sequence = np.ones(T, dtype= int)*-1
    for i in range(start, T):
        if i % spacing == 0:
            sequence[i] = np.random.randint(0,num_stims)
    return sequence

if __name__ == "__main__":
    stim = single_stim(1, 10, 5, np.array([1, 2, 3]))
    N = 5
    T = 20
    num_stims = 3
    stim_list = binary_stim(N, num_stims)
    print(stim_list)
    sequence = rand_seq_func(T, num_stims, spacing=3, start = 5)
    print(sequence)
    stim = stimgen(stim_list, sequence)
    print(stim)
