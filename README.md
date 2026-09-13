# FitzhughNagumoNetwork

## 1. Excitable Systems and the FitzHugh-Nagumo Model

Excitable systems, such as neurons, exhibit rapid deviation from equilibrium in response to stimuli followed by recovery. The FitzHugh-Nagumo equations describe this behavior with two variables: the membrane potential `v` and a recovery variable `n`:

```
dv/dt = (A * v * (Δ - v) * (v - 1) - n) / η 

dn/dt = -γ * n + v
```

Below is an example of a single FitzHugh-Nagumo system:

<img src="https://github.com/jcvdwlt/FitzhughNagumoNetwork/blob/master/figs/single_neuron.png">

---

## 2. Coupled Systems in a Circular Network

When multiple FitzHugh-Nagumo systems are connected via directed diffusive coupling, their dynamics are influenced by their neighbors. The coupling is represented by a Laplacian matrix `L`:

```
dv/dt = (A * v * (Δ - v) * (v - 1) - n) / η - G * Lv

dn/dt = -γ * n + v
```


Below is an example of 8 systems connected in a circular network:

<img src="https://github.com/jcvdwlt/FitzhughNagumoNetwork/blob/master/figs/neuron_circle.gif">


---

## 3. Larger Connected Networks

In larger, more complex networks, the collective dynamics become richer and more intricate. The same equations apply, but the network structure plays a critical role in shaping the behavior.

Below is an example of a larger connected network, leading to an emergent pulsing that mimics brain waves:

<img src="https://github.com/jcvdwlt/FitzhughNagumoNetwork/blob/master/figs/neurons_200.gif">
