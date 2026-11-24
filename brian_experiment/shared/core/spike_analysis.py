import dill as pickle
import sparsity_and_selectivity as sp
import numpy as np
import seaborn as sns
import spike_analysis_utils as sau
from brian2 import *
import sparsity_and_selectivity as sp
import numpy as np
import seaborn as sns
import spike_analysis_utils as sau
from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
def load_from_pickle(filename):
    """
    Loads an object from a pickle file.

    Parameters:
    filename (str): The name of the file from which to load the object.

    Returns:
    The object that was loaded from the file.
    """
    with open(filename, "rb") as file:
        obj = pickle.load(file)
    return obj

def generate_log_intervals(T, num_intervals):
    """
    This function creates a list of time duration from a given time T and number of intervals required.
    :param T: the duration of the whole simulation.
    :param num_intervals: the numbers of intervals T will be divided to.
    :return: a list od time intervals that summed up to T.
    """
    boundaries = np.logspace(np.log10(10), np.log10(T / ms), num_intervals)  # Generate logarithmic boundaries
    # Calculate durations as differences between consecutive boundaries
    time_intervals = np.diff(boundaries)
    # Normalize durations to sum to T
    time_intervals = time_intervals / np.sum(time_intervals) * (T / ms)
    return time_intervals * ms


def divide_spike_matrix_into_cumulative_intervals(total_time, num_desired_intervals, spike_matrix, dt=0.1):
    """
    Divides a spike matrix into cumulative intervals for analyzing how network behavior 
    evolves over time.
    
    Parameters:
    - total_time: Total simulation time (in ms)
    - num_desired_intervals: Number of analysis points wanted
    - spike_matrix: Matrix of spikes (neurons × timesteps)
    - dt: Time step of simulation (in ms)
    
    Returns:
    - List of spike matrices, where each matrix contains all spikes from start 
      until that interval point
    """
    # Convert time values to dimensionless quantities for calculations
    total_time_val = float(total_time/ms)
    dt_val = float(dt/ms)
    
    timesteps_per_interval = int(total_time_val / (dt_val * num_desired_intervals))
    cumulative_matrices = []
    
    for interval in range(1, int(num_desired_intervals) + 1):
        end_timestep = timesteps_per_interval * interval
        cumulative_matrix = spike_matrix[:, 0:end_timestep]
        cumulative_matrices.append(cumulative_matrix)
    
    return cumulative_matrices


def analyze_firing_rates_over_time(total_time, stims_by_time, spike_matrix, num_intervals, dt=defaultclock.dt, start_time=100*ms):
    """
    Analyzes how firing rates, sparsity, and selectivity evolve over the course of training.
    
    Parameters:
    - total_time: Total simulation time
    - stims_by_time: Vector indicating stimulus at each timestep
    - spike_matrix: Full spike recording matrix
    - num_intervals: Number of analysis points wanted
    - dt: Simulation timestep
    - start_time: Initial delay before stimuli begin
    """
    rate_matrices = []
    sparsity_vectors = []
    selectivity_vectors = []
    
    # Convert start_time to timesteps
    start_timesteps = int(start_time / dt)
    
    # Skip the initial start_time period
    spike_matrix = spike_matrix[:, start_timesteps:]
    stims_by_time = stims_by_time[start_timesteps:]
    total_time = total_time - start_time
    
    # Get cumulative spike matrices
    cumulative_matrices = divide_spike_matrix_into_cumulative_intervals(
        total_time, num_intervals, spike_matrix, dt
    )
    
    # For each cumulative interval
    for i, spike_mat in enumerate(cumulative_matrices):
        stim_length = spike_mat.shape[1]
        current_stims = stims_by_time[:stim_length]
        current_rate_matrix = sau.gen_firing_rates_matrix(current_stims, spike_mat)
        
        if current_rate_matrix.size > 0:
            sparsity_vec, selectivity_vec = sp.create_vecs(current_rate_matrix)
            rate_matrices.append(current_rate_matrix)
            sparsity_vectors.append(sparsity_vec)
            selectivity_vectors.append(selectivity_vec)
    
    # Ensure all vectors have the same length by padding with NaN
    if rate_matrices:
        max_sparsity_len = max(len(vec) for vec in sparsity_vectors)
        max_selectivity_len = max(len(vec) for vec in selectivity_vectors)
        
        padded_sparsity = np.zeros((len(sparsity_vectors), max_sparsity_len))
        padded_selectivity = np.zeros((len(selectivity_vectors), max_selectivity_len))
        
        for i, (spars_vec, sel_vec) in enumerate(zip(sparsity_vectors, selectivity_vectors)):
            padded_sparsity[i, :len(spars_vec)] = spars_vec
            padded_selectivity[i, :len(sel_vec)] = sel_vec
            
        return rate_matrices, padded_sparsity, padded_selectivity
    
    return [], [], []


def analyze_firing_rates_over_time_long(spike_monitor, T, stim_by_time, num_intervals, num_neurons, dt, start_time=0*ms):
    """
    Analyzes firing rates over time in a memory-efficient way for long simulations.
    Instead of creating a full spike_matrix, it processes spikes in chunks corresponding
    to each analysis interval.
    """
    interval_duration = (T - start_time) / num_intervals
    
    rate_matrices_arr = []
    sparsity_vec_arr = []
    selectivity_vec_arr = []

    for i in range(num_intervals):
        interval_start = start_time + i * interval_duration
        interval_end = interval_start + interval_duration
        
        # Filter spikes within the current interval
        interval_spikes = (spike_monitor.t >= interval_start) & (spike_monitor.t < interval_end)
        
        # Create a temporary spike matrix for this interval
        interval_spike_matrix = sau.spike_matrix_from_spikes(
            num_neurons,
            spike_monitor.i[interval_spikes],
            spike_monitor.t[interval_spikes] - interval_start, # Adjust times to be relative to interval start
            interval_duration,
            dt
        )
        
        # Generate the firing rate matrix for the stimuli presented in this interval
        interval_stim_by_time = stim_by_time[int(interval_start/dt):int(interval_end/dt)]
        rate_matrix = sau.gen_firing_rates_matrix(interval_stim_by_time, interval_spike_matrix)
        
        # Convert from proportion (spikes per bin) to Hz by dividing by dt (in seconds)
        # rate_matrix is currently in units of "spikes per bin" (proportion)
        # To get Hz: (spikes/bin) / (dt in seconds) = Hz
        from brian2 import second
        dt_seconds = float(dt / second)
        rate_matrix = rate_matrix / dt_seconds
        
        # Calculate sparsity and selectivity
        sparsity_vec, selectivity_vec = sp.create_vecs(rate_matrix)
        
        rate_matrices_arr.append(rate_matrix)
        sparsity_vec_arr.append(sparsity_vec)
        selectivity_vec_arr.append(selectivity_vec)

    sparsity_vec_arr = np.array(sparsity_vec_arr)
    selectivity_vec_arr = np.array(selectivity_vec_arr)

    return rate_matrices_arr, sparsity_vec_arr, selectivity_vec_arr


def analyze_both_individual_and_cumulative_epochs(spike_monitor, T, stim_by_time, num_intervals, num_neurons, dt, start_time=0*ms):
    """
    Analyzes firing rates in two ways:
    1. Individual epochs: each epoch analyzed separately
    2. Cumulative epochs: average firing rates across progressively more epochs, then compute metrics
    
    Returns both sets of analyses for comprehensive evolution tracking.
    """
    # Individual epoch analysis (each epoch separately)
    individual_rate_matrices, individual_sparsity, individual_selectivity = analyze_firing_rates_over_time_long(
        spike_monitor, T, stim_by_time, num_intervals, num_neurons, dt, start_time
    )
    
    # Cumulative epoch analysis: average rate matrices across epochs first, then compute metrics
    cumulative_rate_matrices = []
    cumulative_sparsity_list = []
    cumulative_selectivity_list = []
    
    for i in range(num_intervals):
        # Average rate matrices from epoch 0 to epoch i (inclusive)
        if i == 0:
            # First epoch only
            avg_rate_matrix = individual_rate_matrices[0]
        else:
            # Average from first epoch up to current epoch
            matrices_to_average = individual_rate_matrices[:i+1]
            avg_rate_matrix = np.mean(matrices_to_average, axis=0)
        
        # Compute sparsity and selectivity from the averaged rate matrix
        sparsity_vec, selectivity_vec = sp.create_vecs(avg_rate_matrix)
        
        cumulative_rate_matrices.append(avg_rate_matrix)
        cumulative_sparsity_list.append(sparsity_vec)
        cumulative_selectivity_list.append(selectivity_vec)
    
    cumulative_sparsity = np.array(cumulative_sparsity_list)
    cumulative_selectivity = np.array(cumulative_selectivity_list)
    
    return (individual_rate_matrices, individual_sparsity, individual_selectivity,
            cumulative_rate_matrices, cumulative_sparsity, cumulative_selectivity)


def spike_analysis_plots(rate_matrices_arr, sparsity_vec_arr, selectivity_vec_arr, interval_size, num_intervals):
    """
    Creates a mosaic plot showing the rate matrix, selectivity vector, and sparsity vector for each interval.

    :param rate_matrices_arr: List of firing rate matrices, one per interval.
    :param sparsity_vec_arr: List of sparsity vectors, one per interval.
    :param selectivity_vec_arr: List of selectivity vectors, one per interval.
    :param time_intervals: List of interval durations (in ms).
    """
    fig, axs = plt.subplots(3, num_intervals, figsize=(5 * num_intervals, 15), constrained_layout=True)

    for i in range(num_intervals):
        # Retrieve data for the current interval
        rate_matrix = rate_matrices_arr[i]
        sparsity_vec = sparsity_vec_arr[i]
        selectivity_vec = selectivity_vec_arr[i]

        duration = interval_size/ ms

        # Plot the rate matrix as a heatmap
        ax = axs[0, i]
        im = ax.imshow(rate_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(f"Interval {i + 1} (Rate Matrix)\nDuration: {int(duration)} ms")
        ax.set_xlabel("Stimulus")
        ax.set_ylabel("Neuron")
        fig.colorbar(im, ax=ax, orientation="vertical")

        # Plot the selectivity vector as a heatmap
        ax = axs[1, i]
        selectivity_matrix = np.expand_dims(selectivity_vec, axis=1)  # Make it 2D for imshow
        im = ax.imshow(selectivity_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(f"Interval {i + 1} (Selectivity Vector)")
        ax.set_xlabel("Duration")
        ax.set_ylabel("Neuron")
        fig.colorbar(im, ax=ax, orientation="vertical")

        # Plot the sparsity vector as a heatmap
        ax = axs[2, i]
        sparsity_matrix = np.expand_dims(sparsity_vec, axis=1)  # Make it 2D for imshow
        im = ax.imshow(sparsity_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(f"Interval {i + 1} (Sparsity Vector)")
        ax.set_xlabel("Duration")
        ax.set_ylabel("Stimulus")
        fig.colorbar(im, ax=ax, orientation="vertical")

    # Add overall titles for each row
    axs[0, 0].set_ylabel("Rate Matrix\nNeuron")
    axs[1, 0].set_ylabel("Selectivity Vector\nNeuron")
    axs[2, 0].set_ylabel("Sparsity Vector\nStimulus")

    # Add an overall title for the figure
    fig.suptitle("Spike Analysis Plots", fontsize=16)


if __name__ == "__main__":
    T = 2000* ms
    N = 10
    cols = int(N * T / defaultclock.dt)
    num_intervals = 5
    time_intervals = T/num_intervals
    dt = 0.1 * ms
    spike_matrix = np.random.randint(0, 2, (N, cols))
    print(spike_matrix.shape)
    print(spike_matrix.sum())
    sub_matrices = divide_spike_matrix_into_cumulative_intervals(T, num_intervals, spike_matrix, dt=0.1)
    for submat in sub_matrices:
        print(submat.shape)
        print(submat.sum())
    stims_by_time = np.random.randint(0, 2, cols)
    rate_matrices_arr, sparsity_vec_arr, selectivity_vec_arr = analyze_firing_rates_over_time(T, stims_by_time, spike_matrix, num_intervals, dt)
    for i, sli_vec in enumerate(sparsity_vec_arr):
        print(f"the {i} sparsity vec sum is {sum(sli_vec)}")
    spike_analysis_plots(rate_matrices_arr, sparsity_vec_arr, selectivity_vec_arr, time_intervals, num_intervals)

