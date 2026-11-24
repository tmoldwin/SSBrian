from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from plots import plot_smoothed_rates
from scipy import ndimage


def gen_firing_rates_matrix_vectorized(stim_type_vec, spike_matrix):
    """
    Vectorized version of firing rate matrix generation.
    Much faster than the original loop-based version.
    """
    stim_type_vec = np.asarray(stim_type_vec)
    spike_matrix = np.asarray(spike_matrix)
    
    # Get unique stimulus types (excluding -1)
    unique_stims = np.unique(stim_type_vec)
    unique_stims = unique_stims[unique_stims != -1]
    
    if len(unique_stims) == 0:
        return np.array([])
    
    n_neurons, n_timepoints = spike_matrix.shape
    n_stims = len(unique_stims)
    
    # Pre-allocate output matrix
    firing_rate_matrix = np.zeros((n_neurons, n_stims))
    
    # Vectorized computation for all stimuli at once
    for i, stim in enumerate(unique_stims):
        # Create boolean mask for this stimulus
        stim_mask = (stim_type_vec == stim)
        
        # Ensure the mask is the same size as the spike matrix's time dimension
        if len(stim_mask) > spike_matrix.shape[1]:
            stim_mask = stim_mask[:spike_matrix.shape[1]]
        elif len(stim_mask) < spike_matrix.shape[1]:
            spike_matrix = spike_matrix[:, :len(stim_mask)]
        
        # Calculate mean firing rate for each neuron during this stimulus
        # The spike_matrix is binary (0 or 1), so np.mean gives proportion of bins with spikes
        # To convert to Hz, we need to divide by dt (in seconds)
        # But we don't have dt here, so we'll return proportions and convert later
        if np.any(stim_mask):
            firing_rate_matrix[:, i] = np.mean(spike_matrix[:, stim_mask], axis=1)
    
    return firing_rate_matrix


def gen_firing_rates_matrix(stim_type_vec, spike_matrix):
    """
    Generate a firing rate matrix from spike data for each stimulus type.
    Now uses vectorized implementation for better performance.
    """
    # Use vectorized version
    return gen_firing_rates_matrix_vectorized(stim_type_vec, spike_matrix)


def gen_firing_rates_matrix_original(stim_type_vec, spike_matrix):
    """
    Original version with loops - kept for comparison/testing.
    """
    matrix_arr = spike_mats_per_stim(stim_type_vec, spike_matrix)
    
    if not matrix_arr:
        return np.array([])
        
    firing_rate_matrix = np.zeros((len(matrix_arr[0]), len(matrix_arr)))
    for i, matrix in enumerate(matrix_arr):
        rate_vector_for_matrix = np.mean(matrix, axis=1)
        firing_rate_matrix[:, i] = rate_vector_for_matrix
    
    return firing_rate_matrix


def spike_matrix_from_monitor(spikemon, N, T, dt):
    """
    Optimized version: More memory efficient for large simulations.
    """
    rows = N
    cols = int(T/dt)
    
    # Pre-allocate with appropriate dtype to save memory
    spike_matrix = np.zeros((rows, cols), dtype=np.uint8)  # uint8 saves memory vs float64
    
    if len(spikemon.t) > 0:  # Only process if there are spikes
        time_indices = (spikemon.t/dt).astype(np.int32)
        neuron_indices = np.array(spikemon.i, dtype=np.int32)  # Convert VariableView to array first
        
        # Bounds checking to avoid index errors
        valid_mask = (time_indices < cols) & (neuron_indices < rows)
        if np.any(valid_mask):
            spike_matrix[neuron_indices[valid_mask], time_indices[valid_mask]] = 1
    
    return spike_matrix


def spike_matrix_from_spikes(num_neurons, spike_indices, spike_times, T, dt):
    """
    Creates a spike matrix from spike indices and times for a specific interval.
    """
    num_timesteps = int(T / dt)
    spike_matrix = np.zeros((num_neurons, num_timesteps))
    
    time_indices = (spike_times / dt).astype(int)
    
    # Ensure indices are within bounds
    valid_indices = (time_indices < num_timesteps)
    
    spike_matrix[spike_indices[valid_indices], time_indices[valid_indices]] = 1
    
    return spike_matrix


def spike_mats_per_stim(stim_type_vec, spike_matrix):
    """
    Separate spike matrix into matrices for each stimulus type.
    """
    matrix_arr = []
    spike_matrix = np.array(spike_matrix)
    stim_types = sorted(set(stim_type_vec))
    
    for stim in stim_types:
        if stim == -1:  # Skip the "no stimulus" periods
            continue
            
        index_arr = np.where(stim_type_vec == stim)[0]
        valid_index_arr = index_arr[index_arr < spike_matrix.shape[1]]
        
        if valid_index_arr.size == 0:
            continue
            
        new_matrix_by_stim = spike_matrix[:, valid_index_arr]
        matrix_arr.append(new_matrix_by_stim)
    
    return matrix_arr


def calculate_cv_for_neurons(spike_monitor):
    """
    Calculates the Coefficient of Variation (CV) of Inter-Spike Intervals (ISIs) for each neuron.

    Parameters:
    - spike_monitor: A Brian2 SpikeMonitor object.

    Returns:
    - cv_values: A numpy array containing the CV for each neuron. 
                 Returns np.nan for neurons with fewer than 2 spikes.
    """
    num_neurons = spike_monitor.source.N
    spike_trains = spike_monitor.spike_trains()
    cv_values = np.full(num_neurons, np.nan)

    for i in range(num_neurons):
        spike_times = spike_trains[i]
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            mean_isi = np.mean(isis)
            std_isi = np.std(isis)
            if mean_isi > 0:
                cv_values[i] = std_isi / mean_isi
    
    return cv_values


#for testing
def gen_spike_matrix_by_prob(num_neurons, T,stim_type_vec, prob_dict):
    rows = num_neurons
    cols = T
    stim_types = set(stim_type_vec)
    prob_matrix = np.zeros((rows, cols))
    
    for t in range(cols):
        stim_type = stim_type_vec[t]
        for neuron in range(rows):
            prob_matrix[neuron, t] = prob_dict[neuron][stim_type]
    
    uniform_matrix = np.random.uniform(0, 1, (rows, cols))
    return prob_matrix > uniform_matrix


def smoothed_firing_rate_vectorized(spike_matrix, window_size, dt=1):
    """
    Vectorized version of smoothed firing rate calculation using convolution.
    Much faster than the loop-based version.
    
    Parameters:
    spike_matrix (np.array): A binary matrix of shape (neurons, time) where 1 indicates a spike.
    window_size (float): The size of the sliding window for smoothing, in seconds.
    dt (float): The time step of the simulation, in seconds.
    
    Returns:
    np.array: Smoothed firing rates of shape (neurons, time)
    np.array: Time points corresponding to the smoothed rates
    """
    spike_matrix = np.asarray(spike_matrix, dtype=float)
    num_neurons, num_timepoints = spike_matrix.shape
    window_steps = int(window_size / dt)
    
    # Use uniform filter for each neuron - this is much faster than Python loops
    smoothed_rates = np.zeros_like(spike_matrix)
    
    for neuron in range(num_neurons):
        # Apply uniform filter (moving average) to each neuron's spike train
        smoothed_rates[neuron, :] = ndimage.uniform_filter1d(
            spike_matrix[neuron, :].astype(float), 
            size=window_steps, 
            mode='constant'
        ) / dt  # Convert to rate by dividing by dt
    
    time_points = np.arange(num_timepoints) * dt
    return smoothed_rates, time_points


def smoothed_firing_rate(spike_matrix, window_size, dt=1):
    """
    Calculate the running smoothed average of the firing rate over time.
    Now uses vectorized implementation for much better performance.
    
    Parameters:
    spike_matrix (np.array): A binary matrix of shape (neurons, time) where 1 indicates a spike.
    window_size (float): The size of the sliding window for smoothing, in seconds.
    dt (float): The time step of the simulation, in seconds. Default is 1 (assuming each time step is 1 second).
    Returns:
    np.array: Smoothed firing rates of shape (neurons, time)
    np.array: Time points corresponding to the smoothed rates
    """
    # Use fast vectorized version
    return smoothed_firing_rate_vectorized(spike_matrix, window_size, dt)


def smoothed_firing_rate_original(spike_matrix, window_size, dt=1):
    """
    Original slow version with Python loop - kept for comparison/testing.
    """
    num_neurons, num_timepoints = spike_matrix.shape
    window_steps = int(window_size / dt)
    # Use a sliding window to compute the average
    smoothed_rates = np.zeros_like(spike_matrix, dtype=float)
    for i in range(num_timepoints):
        start = max(0, i - window_steps // 2)
        end = min(num_timepoints, i + window_steps // 2)
        window = spike_matrix[:, start:end]
        smoothed_rates[:, i] = np.sum(window, axis=1) / (window_size)
    time_points = np.arange(num_timepoints) * dt
    return smoothed_rates, time_points


def sort_neurons_by_similarity(spike_rates):
    """
    Sort neurons by similarity of their firing rate patterns using hierarchical clustering.
    Handles edge cases like silent neurons or identical patterns.
    """
    spike_rates = np.asarray(spike_rates)
    
    # Handle edge cases
    if spike_rates.shape[0] <= 1:
        return np.arange(spike_rates.shape[0])
    
    # Check for neurons with no activity or identical patterns
    # Add small random noise to break ties and avoid correlation issues
    spike_rates_clean = spike_rates.copy()
    
    # Add tiny random noise to avoid identical patterns
    np.random.seed(42)  # Reproducible noise
    spike_rates_clean += np.random.normal(0, 1e-10, spike_rates_clean.shape)
    
    # Handle neurons with zero variance (silent neurons)
    row_stds = np.std(spike_rates_clean, axis=1)
    silent_neurons = row_stds < 1e-10
    
    if np.all(silent_neurons):
        # All neurons are silent, return simple order
        return np.arange(spike_rates.shape[0])
    
    if np.any(silent_neurons):
        # Add small variance to silent neurons
        spike_rates_clean[silent_neurons] += np.random.normal(0, 1e-8, (np.sum(silent_neurons), spike_rates_clean.shape[1]))
    
    try:
        # Calculate pairwise distances between neurons
        distances = pdist(spike_rates_clean, metric='correlation')
        
        # Check for non-finite distances
        if not np.all(np.isfinite(distances)):
            print("Warning: Non-finite distances detected, using euclidean metric instead")
            distances = pdist(spike_rates_clean, metric='euclidean')
        
        # Perform hierarchical clustering
        linkage = hierarchy.linkage(distances, method='ward')
        
        # Get the order of neurons from the dendrogram
        dendro = hierarchy.dendrogram(linkage, no_plot=True)
        neuron_order = dendro['leaves']
        
        return neuron_order
        
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"Clustering failed ({e}), returning original order")
        return np.arange(spike_rates.shape[0])


if __name__ == "__main__":
    # Example usage
    num_neurons = 3
    T = 1000
    samples = np.random.choice([1, 3, 4], size=T)

    prob_dict = {
                0: {1: 0.9, 3: 0.1, 4: 0.1},
                1: {1: 0.1, 3: 0.9, 4: 0.1},
                2: {1: 0.1, 3: 0.1, 4: 0.9}
    }
    spike_matrix = gen_spike_matrix_by_prob(3, T, samples, prob_dict)
    matrix_arr = spike_mats_per_stim(samples, spike_matrix)
    print(matrix_arr)
    matrix_arr = spike_mats_per_stim(samples, spike_matrix)
    print(matrix_arr)

    # Test smoothed firing rate optimization
    window_size = 50  # 50 time steps for smoothing
    dt = 1  # Assuming each time step is 1 (adjust if necessary)

    # Test bigger matrix for timing
    big_spike_matrix = np.random.randint(0, 2, (50, 5000))
    
    import time
    
    print("\nTesting smoothed firing rate optimization...")
    print("Matrix size: 50 neurons x 5000 time points")
    
    # Time vectorized version
    start = time.time()
    result_fast = smoothed_firing_rate_vectorized(big_spike_matrix, window_size, dt)
    fast_time = time.time() - start
    
    # Time original version
    start = time.time()
    result_slow = smoothed_firing_rate_original(big_spike_matrix, window_size, dt)
    slow_time = time.time() - start
    
    print(f"Vectorized: {fast_time:.4f}s")
    print(f"Original:   {slow_time:.4f}s")
    print(f"Speedup:    {slow_time/fast_time:.1f}x faster")
    
    # Calculate smoothed firing rates
    smoothed_rates, time_points = smoothed_firing_rate(spike_matrix, window_size, dt)

    # Plot the smoothed rates
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the original spike matrix
    ax1.imshow(spike_matrix, cmap='viridis', aspect='auto')
    ax1.set_title('Original Spike Matrix')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Neuron')

    # Plot the smoothed firing rates
    plot_smoothed_rates(smoothed_rates, time_points, ax=ax2, title="Smoothed Firing Rates")

    plt.tight_layout()

    # Existing code
    plt.figure(figsize=(5, 5))  # Create a new figure for each matrix
    plt.imshow(spike_matrix, cmap='viridis', aspect='auto')  # Display the matrix
    plt.colorbar()  # Show the color scale

    mat = gen_firing_rates_matrix(samples, spike_matrix)
    print(mat)
    plt.figure(figsize=(5, 5))  # Create a new figure for each matrix
    plt.imshow(mat, cmap='viridis', aspect='auto')  # Display the matrix
    plt.colorbar()  # Show the color scale
    plt.show()

