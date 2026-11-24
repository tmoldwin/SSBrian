import numpy as np
import math
import seaborn as sns
from numpy.linalg import eig
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import animation, colors
sigmoid = lambda x: 1 / (1+math.exp(-x))


def derivative_fn(X, Y, W):
    """
    Compute the derivatives based on the weight matrix and position vector.

    Parameters:
        X (ndarray): X values.
        Y (ndarray): Y values.
        W (ndarray): Weight matrix.

    Returns:
        Tuple of derivative arrays (dX, dY).
    """
    dX = W[0, 0] * X + W[0, 1] * Y - X
    dY = W[1, 0] * X + W[1, 1] * Y - Y

    return dX, dY

def plot_trajectory(ax, x, y):
    ax.plot(x, y)
    # plt.gca().set_aspect('equal')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Trajectory Plot')
    for i in range(len(x) - 1):
        ax.arrow(x[i], y[i], x[i + 1] - x[i], y[i + 1] - y[i], width=0.01, color='r', length_includes_head=True)


def phase_plane_plot(ax, derivative_fn,  *derivative_params, density = 20):
    """
    Generate a phase plane plot based on the derivative function.

    Parameters:
        derivative_fn (function): Derivative function.
        derivative_params (tuple): Parameters for the derivative function.
    """
    # Generate all possible combinations of x and y values
    x_values = np.linspace(-2, 2, density)
    y_values = np.linspace(-2, 2, density)
    X, Y = np.meshgrid(x_values, y_values)

    # Compute the derivatives for each combination of x and y values
    dX, dY = derivative_fn(X, Y, *derivative_params)
    # Normalize the derivatives
    max_derivative = max(np.max(np.abs(dX)), np.max(np.abs(dY)))


    # Plot the phase plane
    # fig, ax = plt.subplots(1,1)
    ax.quiver(X, Y, dX, dY, angles='xy')

#create a heat map out of a complex matrix, either color by
# magnitude(abs value) or color by the real part.
def plot_complex_heat_map(matrix, ax, color_by_magnitude = False, cmap='coolwarm', annot = True):
    """
    Plot a complex matrix as a heat map.

    Parameters:
        matrix (ndarray): Complex matrix.
        ax (matplotlib.axes.Axes): Axes to plot on.
        color_by_magnitude (bool): Whether to color by magnitude or real part.
        cmap (str): Colormap to use.
    """
    matrix = np.array(matrix)[::-1]
    # Create a heatmap
    if color_by_magnitude:
        sns.heatmap(np.abs(matrix), annot=False, center= 0, linewidths=0.5, linecolor='k', cmap=cmap, ax=ax)
    else:
        sns.heatmap(matrix.real, annot=False, center=0, linewidths=0.5, linecolor='black', cmap=cmap, ax=ax)
    if annot == True:
    # Add the values to the heatmap
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j + 0.5, i + 0.5, f'{value.real:.2f}+{value.imag:.2f}j', ha='center', va='center', color='black')

    # Set the x and y limits
    ax.set_xlim([0, matrix.shape[1]])
    ax.set_ylim([0, matrix.shape[0]])

    # Add labels
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # # Add a title
    # if color_by_magnitude:
    #     ax.set_title('Magnitude')
    # else:
    #     ax.set_title('Real Part')

def plot_vectors(matrix, ax, linestyle='solid', linewidth=2):

    x = np.array(matrix[0])
    y = np.array(matrix[1])

    print(x, y)
    # # Create a figure and axis
    # fig, ax = plt.subplots()
    # Create a figure and axis

    # Define a colormap
    cmap = plt.cm.get_cmap('tab10')

    # Plot the vectors using quiver with colormap
    for i in range(len(x)):
        ax.plot([0, x[i]],[0, y[i]],
                  linestyle = linestyle, linewidth = linewidth,  color=cmap(i / len(x)))
    # Set the x and y limits
    # ax.set_xlim([np.minimum(0, np.min(x) - 1), np.max(x) + 1])
    # ax.set_ylim([np.minimum(0, np.min(y) - 1), np.max(y) + 1])

    # Add labels to the vectors
    for i in range(len(x)):
        ax.text(x[i], y[i], f'({x[i]:.2f}, {y[i]:.2f})', ha='center', va='bottom', fontsize=4)

    # Set aspect ratio to 'equal'

    #x and y are on the same size
    ax.set_aspect('equal')

#add a function for plotting heatmap with real values.
#move parameter ax to the end in all plot fuctions.

def plot_heatmap_with_annot(magnitude_matrix, eigenvectors, eigenvalues, ax):

    sns.heatmap(magnitude_matrix, cmap='Reds', ax = ax)

    # Add text annotations
    for i in range(magnitude_matrix.shape[0]):
        for j in range(magnitude_matrix.shape[1]):
            value = eigenvectors[i, j]
            ax.text(j + 0.5, i + 0.5, f'{value.real:.2f}+{value.imag:.2f}j', ha='center', va='center', color='blue')

    plt.title("Heatmap with Complex Value Annotations")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")


def plot_eigenvalues(eigenvalues, ax):
    real_parts = [z.real for z in eigenvalues]
    imag_parts = [z.imag for z in eigenvalues]
    ax.scatter(real_parts, imag_parts)
    # plt.xlim(-1, 1)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_title("Scatter Plot of Complex Vector")

    #these are the titles for the real and imaginary parts.
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")

def plot_real_heatmap(matrix, ax, cmap='coolwarm', annot=True):
        """
        Plot a real matrix as a heat map.
        Parameters:
            matrix (ndarray): real matrix.
            ax (matplotlib.axes.Axes): Axes to plot on.
            cmap (str): Colormap to use.
        """
        matrix = np.array(matrix)[::-1]
        # Create a heatmap
        sns.heatmap(matrix, annot=annot, center=0, linewidths=0.5, linecolor='black', cmap=cmap, ax=ax)
        # Set the x and y limits
        ax.set_xlim([0, matrix.shape[1]])
        ax.set_ylim([0, matrix.shape[0]])

        # Add labels
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')


def phase_plane_plot_3D():
        # Define the system of differential equations
        def dx_dt(x, y, z):
            return -y - z

        def dy_dt(x, y, z):
            return x + 0.2 * y

        def dz_dt(x, y, z):
            return 0.2 + z * (x - 5.7)

        # Define the time points
        t = np.linspace(0, 10, 1000)

        # Solve the system of differential equations
        x, y, z = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
        u = dx_dt(x, y, z)
        v = dy_dt(x, y, z)
        w = dz_dt(x, y, z)

        # Create a 3D phase plane plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)


        # Set labels and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Phase Plane Plot')


