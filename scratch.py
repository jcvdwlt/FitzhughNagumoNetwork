import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation
import src

# PROBLEM PARAMETERS
N = 8
A = 1
DELTA = 0.03  # 0.02
GAMMA = 0.5
ETA = 0.01
G = 2

D = np.diag(np.ones(N-1), -1)
D[0, N-1] = 1
ic = np.zeros(N * 2)
ic[0] = 0.3

# TIMESCALE
T = 3
T_N = 100
t_eval = np.linspace(0, T, T_N)


def laplacian(adjacancy):
    s = np.sum(adjacancy, axis=0)
    d = np.diag(s)
    return d - adjacancy


L = laplacian(D)


def fun_system(t, y):
    d = int(len(y) / 2)
    v = y[0:d]
    n = y[d:]

    dv = (A * v * (DELTA - v) * (v - 1) - n) / ETA - G * np.matmul(L, v)
    dn = - GAMMA * n + v

    return np.concatenate([dv, dn])


# Solve the initial value problem
sol = solve_ivp(fun_system, [0, T], ic, t_eval=t_eval)

# Show the outputs
animator = src.MakeAnimation(adjacency=D, sol=sol)
animator.animation.save('figs/neuron.gif', writer='imagemagick', fps=5)
plt.show()


