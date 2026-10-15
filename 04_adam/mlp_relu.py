"""
Section 6.2's network: "two fully connected hidden layers with 1000 hidden
units each and ReLU activation... minibatch size of 128... the standard
deterministic cross-entropy objective function with L2 weight decay on the
parameters to prevent over-fitting."

784-1000-1000-10, ReLU hidden units, softmax+cross-entropy output, L2 decay
on the weight matrices (not the biases, standard convention). Hand-rolled
forward/backward (duplicated in spirit from every prior paper here, not
imported, so this folder stays self-contained). params()/grads() expose a
flat list so optimizer.py's classes can drive this network generically.
"""
import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class ReLUMLP:
    def __init__(self, layer_sizes, weight_std=0.01, l2=1e-4, seed=None):
        rng = np.random.default_rng(seed)
        self.sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.l2 = l2
        self.W = [rng.normal(0.0, weight_std, size=(n_out, n_in))
                   for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros(n_out) for n_out in layer_sizes[1:]]

    def params(self):
        return self.W + self.b

    def forward(self, x):
        activations = [x]
        pre_act = []
        for l in range(self.n_layers):
            z = activations[-1] @ self.W[l].T + self.b[l]
            pre_act.append(z)
            a = softmax(z) if l == self.n_layers - 1 else relu(z)
            activations.append(a)
        return activations, pre_act

    def backward(self, activations, pre_act, D):
        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers

        y_out = activations[-1]
        dE_dz = y_out - D  # softmax + cross-entropy, same clean form as every prior paper here

        for l in reversed(range(self.n_layers)):
            a_in = activations[l]
            grads_W[l] = dE_dz.T @ a_in + self.l2 * self.W[l]  # L2 weight decay, weights only
            grads_b[l] = dE_dz.sum(axis=0)
            if l > 0:
                dE_da_in = dE_dz @ self.W[l]
                z_prev = pre_act[l - 1]
                dE_dz = dE_da_in * (z_prev > 0)  # ReLU derivative

        return grads_W + grads_b  # same order as params()

    def loss(self, y_pred, D, eps=1e-12):
        data_loss = -np.sum(D * np.log(np.clip(y_pred, eps, 1.0)))
        l2_loss = 0.5 * self.l2 * sum(np.sum(W * W) for W in self.W)
        return data_loss + l2_loss
