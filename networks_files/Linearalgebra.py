import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns
from numpy.linalg import eig
import plotting_functions as pf

# Plot the vectors


fig, axs = plt.subplot_mosaic([['A', 'B'], ['C', 'D'], ['E', 'F']],
                              layout='constrained')
print(axs)


vectors = np.array([[1, 1, -1, -1],
                   [1, -1, 1, -1]])

M = np.array([[1.4, 2.8], [3.2, 2.5]])

M = np.array([[1, 1], [2, -0.5]])

angle = np.pi/4
# M = np.array([[np.cos(angle), -np.sin(angle)],
#  [np.sin(angle), np.cos(angle)]])

eigenvalues, eigenvectors = eig(M)
new_vectors = np.dot(M, vectors)


pf.plot_complex_heat_map(M, axs['A'])
pf.plot_complex_heat_map(vectors, axs['B'])
pf.plot_vectors(vectors, axs['C'])
pf.plot_vectors(M,  axs['C'], linestyle='dotted')
pf.plot_vectors(M,  axs['D'], linestyle='dotted')
pf.plot_vectors(new_vectors, axs['D'])
pf.plot_complex_heat_map(np.diag(eigenvalues), axs['E'], color_by_magnitude=False)
pf.plot_complex_heat_map(eigenvectors, axs['F'], color_by_magnitude=False)


# Show the plot
plt.show()


