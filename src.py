import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import networkx as nx

LINE_WIDTH = 0.1  # Default linewidth for arrows
SIZE = 50  # Default size for scatter plot points
ALPHA = 0.2

# LINE_WIDTH = 0.5  # Default linewidth for arrows
# SIZE = 200  # Default size for scatter plot points
# ALPHA = 0.5

def laplacian(adjacancy):
    s = np.sum(adjacancy, axis=0)
    d = np.diag(s)
    return d - adjacancy


def laplacian_matrix(graph):
    adj = nx.to_numpy_array(graph)
    return laplacian(adj)


def generate_layout_coords(G, layout_type='spring'):
    if layout_type == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    
    # 1. Get all coordinates as a single (N, 2) NumPy array
    coords = np.array(list(pos.values()))

    # 2. Extract separate 1D NumPy arrays for X and Y
    xc = coords[:, 0]
    yc = coords[:, 1]

    return xc, yc

def scatter_grapth(ax, G, layout_type='spring'): 
    xc, yc = generate_layout_coords(G, layout_type=layout_type)
    n = len(G.nodes)
    s = ax.scatter(xc, yc, s=SIZE)  # Node scatter plot
    adjacency = nx.to_numpy_array(G)

    arrows = []  # List to store arrow references
    for i in range(n):
        for j in range(n):
            if adjacency[j, i] > 0:
                arrow = ax.arrow(
                    xc[i], 
                    yc[i], 
                    0.9 * (xc[j] - xc[i]), 
                    0.9 * (yc[j] - yc[i]), 
                    head_width=0.02, 
                    head_length=0.03,
                    linewidth=LINE_WIDTH,  # Initial linewidth
                )
                arrows.append(arrow)  # Store the arrow reference

    return s, arrows  # Return both the scatter plot and the arrows


class MakeAnimation:
    def __init__(self, G, sol, layout_type='spring'):
        self.n = len(G)
        self.adj = nx.to_numpy_array(G)
        self.sol = sol
        self.f = plt.figure(figsize=(12, 5))
        self.ax1 = self.f.add_subplot(121)
        self.ax2 = self.f.add_subplot(122, xlim=[0, max(sol.t)], ylim=[-0.5, 1])

        self.s, self.arrows = scatter_grapth(self.ax1, G, layout_type=layout_type)  # Get arrows
        self.s.set_clim(0, 1)

        self.lines = []
        for j in range(self.n):
            line, = self.ax2.plot([], [], label=str(j + 1), c='g', alpha=ALPHA)
            self.lines.append(line)

        self.ax2.legend().remove()
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])

        self.ax1.set_xticks([])
        self.ax1.set_yticks([])

        self.animation = animation.FuncAnimation(self.f, self.anim, frames=len(self.sol.t), interval=20, blit=False)

    def anim(self, i):
        self.s.set_array(
            self.sol.y[0:self.n, i]
            )

        for j in range(self.n):
            self.lines[j].set_data(self.sol.t[0:i], self.sol.y[j, 0:i].T)

        # Update arrow linewidths based on the origin node's solution value
        adjacency = self.adj  # Adjacency matrix of the graph
        arrow_idx = 0  # Index to track the current arrow
        for origin_node in range(self.n):
            for target_node in range(self.n):
                if adjacency[origin_node, target_node] > 0:  # If an edge exists
                    # Scale the linewidth based on the origin node's solution value
                    new_linewidth = LINE_WIDTH + LINE_WIDTH*4 * abs(self.sol.y[origin_node, i])  # Scale dynamically
                    self.arrows[arrow_idx].set_linewidth(new_linewidth)  # Update the linewidth
                    arrow_idx += 1  # Move to the next arrow

    def save(self, filename, writer='imagemagick', fps=5):
        """Save the animation and close the figure to suppress the static plot."""
        self.animation.save(filename, writer=writer, fps=fps)
        plt.close(self.f)  # Close the figure to suppress the static plot