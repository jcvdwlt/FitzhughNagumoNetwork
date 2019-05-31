import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_circuit(ax, adjacency):
    n, _ = adjacency.shape
    ax.axis('off')
    degs = np.linspace(0, 2 * np.pi - 2 * np.pi / n, n)
    xc = np.sin(degs)
    yc = np.cos(degs)

    s = ax.scatter(xc, yc, s=200)

    for i in range(n):
        ax.annotate(str(i + 1), [xc[i] * 0.88 - 0.02, yc[i] * 0.88 - 0.02])
        for j in range(n):
            if adjacency[j, i] > 0:
                ax.arrow(xc[i], yc[i], 0.9* (xc[j] - xc[i]), 0.9*(yc[j] - yc[i]), head_width=0.02, head_length=0.03)

    return s


class MakeAnimation:
    def __init__(self, adjacency, sol):
        self.n, _ = adjacency.shape
        self.sol = sol
        self.f = plt.figure(figsize=(12, 5))
        self.ax1 = self.f.add_subplot(121)
        self.ax2 = self.f.add_subplot(122, xlim=[0, max(sol.t)], ylim=[-0.5, 1])

        self.s = plot_circuit(self.ax1, adjacency)
        self.s.set_clim(0, 1)

        self.lines = []
        for j in range(self.n):
            line, = self.ax2.plot([], [], label=str(j + 1))
            self.lines.append(line)

        self.ax2.legend()
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])

        self.animation = animation.FuncAnimation(self.f, self.anim, frames=len(self.sol.t), interval=20, blit=False)

    def anim(self, i):
        self.s.set_array(self.sol.y[0:self.n, i])

        for j in range(self.n):
            self.lines[j].set_data(self.sol.t[0:i], self.sol.y[j, 0:i].T)
