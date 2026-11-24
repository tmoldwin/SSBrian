import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Define the adjacency matrix
adj_matrix = np.array([
    [9, 2, 1],  # Edge from A to B with weight 2, and from A to C with weight 1
    [1, 6, 1.1],  # Edge from B to A with weight 1
    [0, 1, 2.5]   # Edge from C to B with weight 1
])

# Create a directed graph
G = nx.MultiDiGraph()

# Add nodes
nodes = ['A', 'B', 'C']
G.add_nodes_from(nodes)

# Add edges based on the adjacency matrix and manually adjust rad for parallel edges
for i, row in enumerate(adj_matrix):
    for j, val in enumerate(row):
        if val != 0:
            rad = 0.1 * (j - i) if i != j else 0.3  # Adjust radius for parallel edges
            G.add_edge(nodes[i], nodes[j], weight=val, rad=rad)

# Position nodes using a layout
pos = nx.circular_layout(G)

# Draw the graph
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=5000, font_size=15, arrowstyle='-|>', arrowsize=20)

# Manually draw edge labels for multiedges
for edge in G.edges(data=True):
    edge_data = edge[2]
    label = f"{edge_data['weight']}"
    # Find midpoint for the edge label, adjusted for parallel edges
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    if 'rad' in edge_data:
        rad = edge_data['rad']
    else:
        rad = 0
    # Adjust label position based on rad
    label_pos_x = (x0 + x1) / 2 + rad
    label_pos_y = (y0 + y1) / 2 + rad
    plt.text(label_pos_x, label_pos_y, label, size=12, color='red', ha='center', va='center')

# Show the plot
# plt.show()
