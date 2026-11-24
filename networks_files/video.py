import numpy as np
import math
import seaborn as sns
from numpy.linalg import eig
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import animation, colors
sigmoid = lambda x: 1 / (1+math.exp(-x))



def draw_connections(W, position_mat, ax):
    """
    Draw the weighted connections between neurons.
    Parameters:
        W (ndarray): N*N weight matrix.
        position_mat (ndarray): N*2 matrix of neuron positions in 2D space.
        ax (Axes): Matplotlib axes object to draw on.
    Returns:
        List of Line2D objects representing the connections.
    """
    connections = []
    for i, pos_i in enumerate(position_mat):
        for j, pos_j in enumerate(position_mat):
            weight = W[i, j]
            if weight != 0:
                line_color = 'red' if weight > 0 else 'blue'
                line_alpha = abs(weight)/np.max(abs(W))
                line = ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], color=line_color,
                               linewidth=1, alpha=line_alpha)
                connections.append(line[0])
    return connections

def activity_video(weight_mat, activity_mat, position_mat):
    """
    Generate a video showing the neurons' activity over time.
    Parameters:
        weight_mat (ndarray): N*N weight matrix.
        activity_mat (ndarray): N*T matrix of neuron activity.
        position_mat (ndarray): N*2 matrix of neuron positions in 2D space.
    """
    N, T = activity_mat.shape

    # Define color map for activity
    cmap = plt.get_cmap('coolwarm')
    #vmin is minimum value, vmax is the maximum value
    norm = colors.Normalize(vmin=np.min(activity_mat), vmax=np.max(activity_mat))

    # Create figure and axes
    fig, ax = plt.subplots()
    ax.set_title('Neuron Activity - Frame 0')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(np.min(position_mat[:, 0]) - 1, np.max(position_mat[:, 0]) + 1)
    ax.set_ylim(np.min(position_mat[:, 1]) - 1, np.max(position_mat[:, 1]) + 1)

    # Initialize scatter plot and connections
    scatter = ax.scatter(position_mat[:, 0], position_mat[:, 1], c=activity_mat[:, 0], cmap=cmap, norm=norm,
                         edgecolors='black', s=200)
    connections = draw_connections(weight_mat, position_mat, ax)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Activity')

    # Function to update the scatter plot
    def update(frame):
        scatter.set_array(activity_mat[:, frame])
        ax.set_title(f'Neuron Activity - Frame {frame}')

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=T, interval=500, repeat=True)

    # Display the video
    plt.show()
