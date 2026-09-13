# FitzhughNagumoNetwork

## 1. Excitable Systems and the FitzHugh-Nagumo Model

Excitable systems, such as neurons, exhibit rapid responses to stimuli followed by recovery. The FitzHugh-Nagumo equations describe this behavior with two variables: the membrane potential \(v\) and a recovery variable \(n\):

\[
\frac{dv}{dt} = v (v - \Delta) (1 - v) - n
\]
\[
\frac{dn}{dt} = \gamma (v - n)
\]

Below is an example of a single FitzHugh-Nagumo system:

<!-- ![Single System](figs/single_system.png) -->
<img src="https://github.com/jcvdwlt/FitzhughNagumoNetwork/blob/master/figs/single_neuron.png">

---

## 2. Coupled Systems in a Circular Network

When multiple FitzHugh-Nagumo systems are connected via directed diffusive coupling, their dynamics are influenced by their neighbors. The coupling is represented by a Laplacian matrix \(L\):

\[
\frac{dv_i}{dt} = v_i (v_i - \Delta) (1 - v_i) - n_i - g \sum_j L_{ij} v_j
\]
\[
\frac{dn_i}{dt} = \gamma (v_i - n_i)
\]

Below is an example of 8 systems connected in a circular network:

<!-- ![Circular Network](figs/circular_network.png) -->
<img src="https://github.com/jcvdwlt/FitzhughNagumoNetwork/blob/master/figs/neuron_circle.gif">


---

## 3. Larger Connected Networks

In larger, more complex networks, the collective dynamics become richer and more intricate. The same equations apply, but the network structure plays a critical role in shaping the behavior.

Below is an example of a larger connected network:

![Larger Network](figs/larger_network.png)
<img src="https://github.com/jcvdwlt/FitzhughNagumoNetwork/blob/master/figs/neurons_200.gif">
