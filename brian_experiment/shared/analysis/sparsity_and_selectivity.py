import matplotlib.pyplot as plt
import numpy as np


def sparsity_index(column_index, rate_matrix_rec):
    """
    Calculate the sparsity index for a given stim (column) in a rate matrix
    :param column_index:
    :param rate_matrix_rec:
    :return: a scalar, the sparsity index
    :return: a scalar, the sparsity index
    """
    N_rec = len(rate_matrix_rec)
    rows_sum_by_stim = np.sum(rate_matrix_rec[:, column_index])
    numerator = np.power((1/N_rec)*rows_sum_by_stim, 2)
    squared_vec = np.power(rate_matrix_rec[:, column_index], 2)
    summed_squared_vec = np.sum(squared_vec)
    
    # Avoid division by zero
    if summed_squared_vec == 0:
        return 0.0
        
    denominator = (1/N_rec)*summed_squared_vec
    frac = numerator/denominator
    return (1/(1-1/N_rec)) * (1-frac)


def selectivity_index(row_index, rate_matrix_rec):
    """
    Calculate the selectivity index for a given neuron (row) in a rate matrix
    :param row_index: index of the neuron (row) to analyze
    :param rate_matrix_rec: rate matrix with neurons as rows and stimuli as columns
    :return: a scalar, the selectivity index
    """
    N_stims = len(rate_matrix_rec[0])
    cols_sum_by_neuron = np.sum(rate_matrix_rec[row_index, :])
    numerator = np.power((1/N_stims) * cols_sum_by_neuron, 2)
    squared_vec = np.power(rate_matrix_rec[row_index, :], 2)
    summed_squared_vec = np.sum(squared_vec)
    
    # Avoid division by zero
    if summed_squared_vec == 0:
        return 0.0
        
    denominator = (1/N_stims)*summed_squared_vec
    frac = numerator/denominator
    return (1/(1-(1/N_stims))) * (1-frac)


def sparsity_indices_vectorized(rate_matrix_rec):
    """
    Vectorized calculation of sparsity indices for all stimuli.
    Orders of magnitude faster than the loop version.
    """
    rate_matrix = np.asarray(rate_matrix_rec)
    N_rec, N_stims = rate_matrix.shape
    
    # Calculate all column sums at once
    col_sums = np.sum(rate_matrix, axis=0)  # Shape: (N_stims,)
    
    # Calculate all column sum-of-squares at once  
    col_sum_squares = np.sum(rate_matrix**2, axis=0)  # Shape: (N_stims,)
    
    # Vectorized sparsity calculation
    inv_N_rec = 1.0 / N_rec
    numerators = (inv_N_rec * col_sums) ** 2
    denominators = inv_N_rec * col_sum_squares
    
    # Handle division by zero
    denominators = np.where(denominators == 0, 1, denominators)  # Avoid div by 0
    fracs = numerators / denominators
    fracs = np.where(col_sum_squares == 0, 0, fracs)  # Set to 0 where original denom was 0
    
    sparsity_factor = 1.0 / (1.0 - inv_N_rec)
    sparsity_indices = sparsity_factor * (1.0 - fracs)
    
    return sparsity_indices.tolist()


def selectivity_indices_vectorized(rate_matrix_rec):
    """
    Vectorized calculation of selectivity indices for all neurons.
    Orders of magnitude faster than the loop version.
    """
    rate_matrix = np.asarray(rate_matrix_rec)
    N_rec, N_stims = rate_matrix.shape
    
    # Selectivity is not well-defined for fewer than 2 stimuli.
    if N_stims < 2:
        return [0.0] * N_rec
        
    # Calculate all row sums at once
    row_sums = np.sum(rate_matrix, axis=1)  # Shape: (N_rec,)
    
    # Calculate all row sum-of-squares at once
    row_sum_squares = np.sum(rate_matrix**2, axis=1)  # Shape: (N_rec,)
    
    # Vectorized selectivity calculation
    inv_N_stims = 1.0 / N_stims
    numerators = (inv_N_stims * row_sums) ** 2
    denominators = inv_N_stims * row_sum_squares
    
    # Handle division by zero
    denominators = np.where(denominators == 0, 1, denominators)  # Avoid div by 0
    fracs = numerators / denominators
    fracs = np.where(row_sum_squares == 0, 0, fracs)  # Set to 0 where original denom was 0
    
    selectivity_factor = 1.0 / (1.0 - inv_N_stims)
    selectivity_indices = selectivity_factor * (1.0 - fracs)
    
    return selectivity_indices.tolist()


def create_vecs(rate_matrix_rec):
    """
    Create the sparsity and selectivity index vectors for a given rate matrix
    :param rate_matrix_rec:
    :return: sparsity vector, selectivity vector
    """
    # Use vectorized versions - much faster than the original loops
    spi_arr = sparsity_indices_vectorized(rate_matrix_rec)
    sli_arr = selectivity_indices_vectorized(rate_matrix_rec)
    return spi_arr, sli_arr


def create_vecs_slow(rate_matrix_rec):
    """
    Original slow version with Python loops - kept for comparison/testing.
    """
    spi_arr = []
    sli_arr = []
    N_rec = len(rate_matrix_rec)
    N_stims = len(rate_matrix_rec[0])
    for i in range(N_stims):
        spi_arr.append(sparsity_index(i, rate_matrix_rec))
    for i in range(N_rec):
        sli_arr.append(selectivity_index(i, rate_matrix_rec))
    return spi_arr, sli_arr


def plot_rate_selectivity_sparsity(rate_ax, selectivity_ax, sparsity_ax, rate_matrix_rec):
    spi_arr = create_vecs(rate_matrix_rec)[0]
    sli_arr = create_vecs(rate_matrix_rec)[1]
    rate_ax.imshow(rate_matrix_rec)
    selectivity_ax.stem(range(len(sli_arr)), sli_arr)
    sparsity_ax.stem(range(len(spi_arr)), spi_arr)


if __name__ == "__main__":
    mat = np.array([[10, 0, 1], [9, 1, 5], [10, 9, 8], [1, 0, 0]])
    
    # Test that vectorized version gives same results
    spi_fast, sli_fast = create_vecs(mat)
    spi_slow, sli_slow = create_vecs_slow(mat)
    
    print("Vectorized results match original:", 
          np.allclose(spi_fast, spi_slow) and np.allclose(sli_fast, sli_slow))
    
    # Simple timing comparison
    import time
    
    # Make larger matrix for timing
    big_mat = np.random.rand(200, 50) * 10
    
    start = time.time()
    create_vecs(big_mat)
    fast_time = time.time() - start
    
    start = time.time() 
    create_vecs_slow(big_mat)
    slow_time = time.time() - start
    
    print(f"Speedup: {slow_time/fast_time:.1f}x faster")
    
    fig, axis = plt.subplot_mosaic([['A', 'B'], ['C', 'D']], figsize = (18, 10))
    plot_rate_selectivity_sparsity(axis['A'], axis['B'], axis['C'], mat)
    print(selectivity_index(1, mat))

