"""
784-100-100-100-10 sigmoid MLP, matching Section 4.1 of the paper exactly:
"a very simple network, with a 28x28 binary image as input, and 3
fully-connected hidden layers with 100 activations each. Each hidden layer
computes y = g(Wu+b) with sigmoid nonlinearity... followed by a
fully-connected layer with 10 activations (one per class) and cross-entropy
loss."

One class, `use_bn` flag switches BatchNorm1d on/off before each hidden
sigmoid, everything else identical between the two conditions -- so any
difference in the experiment is attributable to BN alone, same controlled-
ablation approach as 02_dropout.

No autodiff library, hand-derived forward/backward (duplicated in spirit
from 01/02, not imported, so this folder stays self-contained).
"""
import numpy as np
from bn_layer import BatchNorm1d


def sigmoid(x):
    # clip to avoid np.exp overflow at large |x| -- asymptotic value (0 or 1)
    # is correct either way, this just silences the RuntimeWarning
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class BNMLP:
    def __init__(self, layer_sizes, use_bn, weight_std=0.01, seed=None):
        rng = np.random.default_rng(seed)
        self.sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.use_bn = use_bn

        self.W = []
        self.b = []       # only used on layers where use_bn is False (entry 2: bias subsumed by beta)
        self.bn = []
        for l, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            is_output = (l == self.n_layers - 1)
            self.W.append(rng.normal(0.0, weight_std, size=(n_out, n_in)))
            if use_bn and not is_output:
                self.b.append(None)
                self.bn.append(BatchNorm1d(n_out))
            else:
                self.b.append(np.zeros(n_out))
                self.bn.append(None)

    def forward_train(self, x):
        activations = [x]
        bn_caches = [None]
        for l in range(self.n_layers):
            is_output = (l == self.n_layers - 1)  # BN is never applied to the output layer
            z = activations[-1] @ self.W[l].T
            if self.bn[l] is not None:
                y_bn, cache = self.bn[l].forward_train(z)
                bn_caches.append(cache)
                a = sigmoid(y_bn)
            else:
                z = z + self.b[l]
                bn_caches.append(None)
                a = softmax(z) if is_output else sigmoid(z)
            activations.append(a)
        return activations, bn_caches

    def forward_eval(self, x):
        a = x
        for l in range(self.n_layers):
            is_output = (l == self.n_layers - 1)
            z = a @ self.W[l].T
            if self.bn[l] is not None:
                z = self.bn[l].forward_eval(z)
            else:
                z = z + self.b[l]
            a = softmax(z) if is_output else sigmoid(z)
        return a

    def backward(self, activations, bn_caches, D):
        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers
        grads_bn = [None] * self.n_layers  # (dgamma, dbeta) or None

        y_out = activations[-1]
        dE_dz = y_out - D  # softmax + cross-entropy, same clean form as papers 01/02

        for l in reversed(range(self.n_layers)):
            a_in = activations[l]
            is_output = (l == self.n_layers - 1)

            if self.bn[l] is not None:
                dE_dy_bn = dE_dz
                dz, dgamma, dbeta = self.bn[l].backward(dE_dy_bn, bn_caches[l + 1])
                grads_bn[l] = (dgamma, dbeta)
                grads_W[l] = dz.T @ a_in
                grads_b[l] = None
                dE_da_in = dz @ self.W[l]
            else:
                grads_W[l] = dE_dz.T @ a_in
                grads_b[l] = dE_dz.sum(axis=0)
                grads_bn[l] = None
                dE_da_in = dE_dz @ self.W[l]

            if l > 0:
                a_hidden = activations[l]
                dE_dz = dE_da_in * a_hidden * (1.0 - a_hidden)  # sigmoid derivative, next layer down

        return grads_W, grads_b, grads_bn

    def step(self, grads_W, grads_b, grads_bn, lr):
        for l in range(self.n_layers):
            self.W[l] -= lr * grads_W[l]
            if self.bn[l] is not None:
                dgamma, dbeta = grads_bn[l]
                self.bn[l].gamma -= lr * dgamma
                self.bn[l].beta -= lr * dbeta
            else:
                self.b[l] -= lr * grads_b[l]

    def loss(self, y_pred, D, eps=1e-12):
        return -np.sum(D * np.log(np.clip(y_pred, eps, 1.0)))

    def probe_last_hidden_sigmoid_input(self, x):
        """Fig. 1(b,c) reproduction: 'the evolution of input distributions to
        a typical sigmoid'. Runs x through every layer up to (not including)
        the output layer and returns the value actually fed into the LAST
        hidden sigmoid -- y_bn if this network uses BN, raw z=Wu+b
        otherwise. Uses bn.forward_probe so calling this during training
        does not disturb the running stats used by forward_eval."""
        a = x
        last_l = self.n_layers - 2  # index of the last hidden layer (n_layers-1 is the output layer)
        for l in range(last_l + 1):
            z = a @ self.W[l].T
            if self.bn[l] is not None:
                sigmoid_input = self.bn[l].forward_probe(z)
            else:
                sigmoid_input = z + self.b[l]
            if l < last_l:
                a = sigmoid(sigmoid_input)
        return sigmoid_input
