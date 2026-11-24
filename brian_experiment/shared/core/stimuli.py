import numpy as np
from brian2 import *
prefs.codegen.target = "numpy"
import scipy.linalg
import scipy.linalg as alg
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable


#TODO - change and add the comments, make sure you understand the code.
# and what is the input and the output of each function.
def generate_bump_matrix(num_neurons, sigma = 0.1):
    # Generate the vector a
    mu = 0.5

    # Generate the vector a with num_neurons elements and numbers from 0 to 1
    a = np.linspace(0, 1, num=num_neurons)

    #b is the gaussian function
    b = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((a - mu) / sigma) ** 2)

    # Generate the circulant matrix c - a matrix with the first row equal to b,
    # and the rest are shifted versions of b.
    c = alg.circulant(b)
    c = np.roll(c, shift=int(num_neurons/2), axis=0)

    return a, c

#it is the inpt for the poisson neurons
def generate_bump_stimulus(num_neurons,  sigma = 0.1, center = 0.5):
    mu = 0.5
    #a is the x axis of the gaussian function
    a= np.linspace(0, 1, num=num_neurons)

    # b is the gaussian function
    b = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((a - mu) / sigma) ** 2)
    b = roll(b, shift=int(num_neurons/2) + int(center * num_neurons), axis=0)

    return a, b


# if neuron is not in the neuron_indices, then the external input is 0.
def step_stim(N, T, neuron_indices, T_start, T_end, stim_amp):
    stim = np.zeros((N, int(T / defaultclock.dt)))
    for i, neuron_index in enumerate(neuron_indices):
        # for t in range(int(T_start[neuron_index]/defaultclock.dt),
        #                int(T_end[neuron_index]/defaultclock.dt)):
        #     stim[neuron_index, t] = stim_amp[neuron_index]
        stim[neuron_index, int(T_start[i] / defaultclock.dt):
                           int(T_end[i] / defaultclock.dt)] = stim_amp[i]
    return stim


# we use this function to stop the simulation at a certain time.
# we can use it to present different stimuli at different time.
# external input for the poisson neurons.
def build_P_external_input(T, num_inputs_E, num_inputs_I):
    stim_matrix = []
    for t in range(int(T)):
        center = t / (T)
        E_rate = generate_bump_stimulus(num_inputs_E, sigma=0.1, center=center)[1]
        I_rate = generate_bump_stimulus(num_inputs_I, sigma=0.1, center=center)[1]
        rate = np.concatenate((E_rate, I_rate))
        stim_matrix.append(rate)
    stim_matrix = np.array(stim_matrix).T
    return stim_matrix


def build_P_external_input_switch(T, num_inputs_E, num_inputs_I):
    stim_matrix = []
    for t in range(int(T)):
        center = t / (T)
        E_rate = generate_bump_stimulus(num_inputs_E, sigma=0.1, center=center)[1]
        I_rate = generate_bump_stimulus(num_inputs_I, sigma=0.1, center=center)[1]
        rate = np.concatenate((E_rate, I_rate))
        stim_matrix.append(rate)
    stim_matrix = np.array(stim_matrix).T
    return stim_matrix

#returns which stimulus is presented in each time point.
def create_stim_list(T, stim_duration, stim_list, break_time, start_time=100):
    """
    Creates a stimulus sequence where each stimulus is presented once before repeating.
    
    Parameters:
    - T: Total simulation time
    - stim_duration: Duration of each stimulus presentation
    - stim_list: List of possible stimuli
    - break_time: Duration of break between stimuli
    - start_time: Time to start presenting stimuli
    
    Returns:
    - stim_trials: Array of stimulus values at each timestep
    - stim_info: List of (stimulus, start_time, end_time) tuples
    """
    sim_length = int(T/defaultclock.dt)
    stim_trials = -1 * np.ones(sim_length)
    stim_info = []
    iter = int(start_time / float(defaultclock.dt/ms))
    steps_per_stim = int(stim_duration / float(defaultclock.dt/ms))
    steps_per_break = int(break_time / float(defaultclock.dt/ms))
    
    # Continue presenting stimuli until we run out of time
    while iter < sim_length:
        # Present stimuli in fixed order for this epoch
        epoch_stims = stim_list
        
        # Present each stimulus in the fixed order
        for stim in epoch_stims:
            if iter + steps_per_stim >= sim_length:
                return stim_trials, stim_info
                
            stim_info.append((stim, int(iter * float(defaultclock.dt/ms)) * ms, int((iter+steps_per_stim) * float(defaultclock.dt/ms)) * ms))
            stim_trials[iter:iter + steps_per_stim] = stim
            iter = int(iter + steps_per_stim + steps_per_break)
    
    return stim_trials, stim_info

def generate_stimuli_from_distribution(stim_by_time, num_inputs, distribution='normal', **dist_params):
    """
    Generate stimuli based on a given probability distribution.
    
    Parameters:
    - stim_by_time: array-like, stimulus values over time
    - num_inputs: int, number of inputs (dimension N)
    - distribution: str, name of the distribution (default: 'normal')
    - **dist_params: additional parameters for the chosen distribution
    
    Returns:
    - numpy.ndarray: N x len(stim_by_time) matrix of stimuli
    """
    unique_stims = np.unique(stim_by_time)
    unique_stims = unique_stims[unique_stims != -1]  # Exclude -1 if present
    
    # Create a dictionary to store stimuli patterns
    stim_patterns = {}
    
    # Generate stimuli for each unique value
    for stim in unique_stims:
        if distribution == 'normal':
            mean = dist_params.get('mean', 1)
            std = dist_params.get('std', 0.1)
            pattern = np.abs(stats.norm.rvs(loc=mean, scale=std, size=num_inputs))
        elif distribution == 'uniform':
            low = dist_params.get('low', 0)
            high = dist_params.get('high', 1)
            pattern = stats.uniform.rvs(loc=low, scale=high-low, size=num_inputs)
        elif distribution == 'exponential':
            scale = dist_params.get('scale', 1)
            pattern = stats.expon.rvs(scale=scale, size=num_inputs)
        elif distribution == 'gamma':
            shape = dist_params.get('shape', 1)
            scale = dist_params.get('scale', 1)
            pattern = stats.gamma.rvs(a=shape, scale=scale, size=num_inputs)
        elif distribution == 'beta':
            a = dist_params.get('a', 1)
            b = dist_params.get('b', 1)
            pattern = stats.beta.rvs(a=a, b=b, size=num_inputs)
        elif distribution == 'poisson':
            mu = dist_params.get('mu', 1)
            pattern = stats.poisson.rvs(mu=mu, size=num_inputs)
        elif distribution == 'pareto':
            shape = dist_params.get('shape', 1.5)
            scale = dist_params.get('scale', 1)
            pattern = stats.pareto.rvs(shape, scale=scale, size=num_inputs)
        elif distribution == 'lognormal':
            mean = dist_params.get('mean', 0)
            sigma = dist_params.get('sigma', 1)
            pattern = stats.lognorm.rvs(s=sigma, scale=np.exp(mean), size=num_inputs)
        elif distribution == 'cauchy':
            loc = dist_params.get('loc', 0)
            scale = dist_params.get('scale', 1)
            pattern = np.abs(stats.cauchy.rvs(loc=loc, scale=scale, size=num_inputs))
        elif distribution == 'levy':
            loc = dist_params.get('loc', 0)
            scale = dist_params.get('scale', 1)
            pattern = stats.levy.rvs(loc=loc, scale=scale, size=num_inputs)
        elif distribution == 'weibull':
            shape = dist_params.get('shape', 0.5)  # Shape < 1 for heavy tail
            scale = dist_params.get('scale', 1)
            pattern = stats.weibull_min.rvs(c=shape, scale=scale, size=num_inputs)
        elif distribution == 'dirichlet':
            alpha = dist_params.get('alpha', np.ones(num_inputs))
            # If alpha is a scalar, create an array of that value
            if np.isscalar(alpha):
                alpha = np.ones(num_inputs) * alpha
            # Generate a single sample from Dirichlet distribution
            pattern = stats.dirichlet.rvs(alpha, size=1)[0]
            # Scale the pattern to have a reasonable range for firing rates
            scale_factor = dist_params.get('scale_factor', 5.0)
            pattern = pattern * scale_factor
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        stim_patterns[stim] = pattern
    
    # Create the output matrix
    output_matrix = np.zeros((num_inputs, len(stim_by_time)))
    
    # Fill the output matrix
    for t, stim in enumerate(stim_by_time):
        if stim != -1:
            output_matrix[:, t] = stim_patterns[stim]
    
    return output_matrix




#returns the matrix of the stim_list over time. this is the external input for the poisson neurons.
def present_bump_trials(num_syn_E, num_syn_I, stim_by_time):
    stim_trials = np.zeros((num_syn_E + num_syn_I, len(stim_by_time)))
    for t in range(len(stim_by_time)):
        stim = stim_by_time[t]
        if not stim == -1:
            stimulus_E = generate_bump_stimulus(num_syn_E, sigma=0.1, center=stim)[1]
            stimulus_I = generate_bump_stimulus(num_syn_I, sigma=0.1, center=stim)[1]
            stimulus = np.concatenate((stimulus_E, stimulus_I))
            stim_trials[:, t] = np.squeeze(stimulus[:, np.newaxis])
    return stim_trials


def build_P_external_input_new(T, num_inputs_E, num_inputs_I):
    stim_trials= []
    for t in range(int(T)):
        center = t / (T)
        E_rate = generate_bump_stimulus(num_inputs_E, sigma=0.1, center=center)[1]
        I_rate = generate_bump_stimulus(num_inputs_I, sigma=0.1, center=center)[1]
        rate = np.concatenate((E_rate, I_rate))
        stim_trials.append(rate)
    stim_trials = np.array(stim_trials).T
    return stim_trials

#this function is used to generate the spike matrix from the spike monitor.
#TODO - add the comments and the input and output of the function.

def plot_stimuli_imshow(stim_matrix, stim_by_time=None, ax=None, title="Stimulus Patterns"):
    """
    Plot all stimuli as an imshow visualization.
    
    Parameters:
    - stim_matrix: numpy.ndarray, stimulus matrix of shape (num_inputs, time_steps)
    - stim_by_time: numpy.ndarray, stimulus type at each time step (optional)
    - ax: matplotlib.axes.Axes, axis to plot on (optional)
    - title: str, plot title (default: "Stimulus Patterns")
    
    Returns:
    - matplotlib.axes.Axes: The axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the stimulus matrix
    im = ax.imshow(stim_matrix, aspect='auto', cmap='viridis', origin='lower')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Stimulus Strength')
    
    # Add stimulus type markers if provided
    if stim_by_time is not None:
        unique_stims = np.unique(stim_by_time)
        unique_stims = unique_stims[unique_stims != -1]  # Exclude -1
        
        # Add vertical lines to separate different stimuli
        prev_stim = None
        for t, stim in enumerate(stim_by_time):
            if stim != prev_stim and stim != -1:
                ax.axvline(x=t, color='white', linestyle='--', linewidth=0.5)
                ax.text(t, ax.get_ylim()[1] + 0.5, f'Stim {stim}', 
                        rotation=90, verticalalignment='bottom', 
                        horizontalalignment='right', fontsize=8)
            prev_stim = stim
    
    # Set labels and title
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Neuron Index')
    ax.set_title(title)
    
    return ax

def plot_stimuli_lollipop(stim_patterns, fig=None, figsize=(12, 8), title="Individual Stimulus Patterns"):
    """
    Plot each stimulus as a separate lollipop plot.
    
    Parameters:
    - stim_patterns: dict or numpy.ndarray, stimulus patterns
                     If dict: {stim_id: pattern_array}
                     If array: shape (num_stims, num_inputs)
    - fig: matplotlib.figure.Figure, figure to plot on (optional)
    - figsize: tuple, figure size (default: (12, 8))
    - title: str, figure title (default: "Individual Stimulus Patterns")
    
    Returns:
    - matplotlib.figure.Figure: The figure with the plots
    - list: List of axes with the plots
    """
    # Convert input to a dictionary if it's an array
    if isinstance(stim_patterns, np.ndarray):
        stim_patterns = {i: stim_patterns[i] for i in range(len(stim_patterns))}
    
    num_stims = len(stim_patterns)
    
    # Calculate grid dimensions
    n_cols = min(4, num_stims)
    n_rows = (num_stims + n_cols - 1) // n_cols
    
    if fig is None:
        fig = plt.figure(figsize=figsize)
    
    fig.suptitle(title, fontsize=16)
    
    axes = []
    for i, (stim_id, pattern) in enumerate(stim_patterns.items()):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Create lollipop plot
        x = np.arange(len(pattern))
        markerline, stemlines, baseline = ax.stem(x, pattern, linefmt='grey', markerfmt='o', basefmt=' ')
        plt.setp(markerline, markersize=4)
        plt.setp(stemlines, linewidth=1)
        
        # Set labels and title
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Stimulus Strength')
        ax.set_title(f'Stimulus {stim_id}')
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        axes.append(ax)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    return fig, axes

if __name__ == "__main__":

    # Original example code
    num_neurons = 100
    sigma = 0.1  # standard deviation (sqrt of variance)
    num_points = 100
    T = 1000
    a, c = generate_bump_matrix(num_points, sigma)
    a, b = generate_bump_stimulus(num_neurons, sigma = 0.1, center=0.2)
    plt.plot(a,b)

    # Example of using the new plotting functions
    # Generate some example stimuli
    num_inputs = 250  # 150 E + 100 I
    T = 500
    stim_list = np.arange(5)
    stim_duration = 100
    break_time = 20
    
    # Create stimulus sequence
    stims_by_time, stim_info = create_stim_list(T, stim_duration, stim_list, break_time)
    
    # Generate stimuli using Dirichlet distribution
    stim_matrix = generate_stimuli_from_distribution(
        stims_by_time, 
        num_inputs, 
        distribution='dirichlet',
        alpha=0.5,
        scale_factor=5.0
    )
    
    # Plot using imshow
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(211)
    plot_stimuli_imshow(stim_matrix, stims_by_time, ax=ax1, title="Dirichlet Stimulus Patterns (imshow)")
    
    # Extract individual stimulus patterns for lollipop plots
    unique_stims = np.unique(stims_by_time)
    unique_stims = unique_stims[unique_stims != -1]  # Exclude -1
    stim_patterns = {}
    
    for stim in unique_stims:
        # Find first occurrence of this stimulus
        t = np.where(stims_by_time == stim)[0][0]
        stim_patterns[stim] = stim_matrix[:, t]
    
    # Plot using lollipop plots
    fig, axes = plot_stimuli_lollipop(stim_patterns, title="Dirichlet Stimulus Patterns (lollipop)")
    

