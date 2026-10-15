"""
784-1000-1000-10 ReLU MLP (same architecture as 04_adam's Section 6.2
network, and close to 03_batchnorm's own toy net) with a `norm_mode` flag:
"none" / "batchnorm" / "layernorm", inserted before each hidden ReLU (not
the output layer), matching both prior papers' convention. One class
covering all three conditions so the Section 6.6 comparison varies only
norm_mode, nothing else -- same controlled-ablation approach used in every
prior paper here.

Activation is ReLU though the paper doesn't state one for this specific
experiment (Section 6.6's text, quoted in JOURNAL.md entry 3, doesn't say)
-- filled-in gap, matching 04_adam's own network for this folder's
comparison, documented rather than silently assumed.
"""
import numpy as np
from bn_layer import BatchNorm1d
from ln_layer import LayerNorm1d


def relu(x):
    return np.maximum(x, 0.0)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class NormMLP:
    def __init__(self, layer_sizes, norm_mode="none", weight_std=0.01, seed=None):
        assert norm_mode in ("none", "batchnorm", "layernorm")
        rng = np.random.default_rng(seed)
        self.sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.norm_mode = norm_mode

        self.W = []
        self.b = []
        self.norm = []
        for l, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            is_output = (l == self.n_layers - 1)
            self.W.append(rng.normal(0.0, weight_std, size=(n_out, n_in)))
            use_norm = (norm_mode != "none") and not is_output
            if use_norm:
                self.b.append(None)  # subsumed by the norm layer's beta, same as 03_batchnorm
                self.norm.append(BatchNorm1d(n_out) if norm_mode == "batchnorm" else LayerNorm1d(n_out))
            else:
                self.b.append(np.zeros(n_out))
                self.norm.append(None)

    def forward_train(self, x):
        activations = [x]
        norm_caches = [None]
        for l in range(self.n_layers):
            is_output = (l == self.n_layers - 1)
            z = activations[-1] @ self.W[l].T
            if self.norm[l] is not None:
                if self.norm_mode == "batchnorm":
                    y_norm, cache = self.norm[l].forward_train(z)
                else:
                    y_norm, cache = self.norm[l].forward(z)
                norm_caches.append(cache)
                a = relu(y_norm)
            else:
                z = z + self.b[l]
                norm_caches.append(None)
                a = softmax(z) if is_output else relu(z)
            activations.append(a)
        return activations, norm_caches

    def forward_eval(self, x):
        a = x
        for l in range(self.n_layers):
            is_output = (l == self.n_layers - 1)
            z = a @ self.W[l].T
            if self.norm[l] is not None:
                z = self.norm[l].forward_eval(z) if self.norm_mode == "batchnorm" else self.norm[l].forward(z)[0]
            else:
                z = z + self.b[l]
            a = softmax(z) if is_output else relu(z)
        return a

    def backward(self, activations, norm_caches, D):
        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers
        grads_norm = [None] * self.n_layers  # (dgamma, dbeta) or None

        y_out = activations[-1]
        dE_dz = y_out - D

        for l in reversed(range(self.n_layers)):
            a_in = activations[l]

            if self.norm[l] is not None:
                z_pre_relu = activations[l + 1]  # post-ReLU; need the pre-relu norm output's sign, recovered below
                # ReLU derivative needs the sign of the norm layer's OUTPUT
                # (pre-ReLU), which post-ReLU activation alone doesn't
                # distinguish at exactly 0 -- but relu(x)>0 iff x>0, and
                # relu(x)==0 whenever x<=0, so the post-ReLU activation's
                # own positivity is an equivalent test here.
                dE_dy_norm = dE_dz * (activations[l + 1] > 0)
                dz, dgamma, dbeta = self.norm[l].backward(dE_dy_norm, norm_caches[l + 1])
                grads_norm[l] = (dgamma, dbeta)
                grads_W[l] = dz.T @ a_in
                grads_b[l] = None
                dE_dz = dz @ self.W[l]
            else:
                grads_W[l] = dE_dz.T @ a_in
                grads_b[l] = dE_dz.sum(axis=0)
                grads_norm[l] = None
                dE_da_in = dE_dz @ self.W[l]
                if l > 0:
                    dE_dz = dE_da_in * (activations[l] > 0)

        return grads_W, grads_b, grads_norm

    def step(self, grads_W, grads_b, grads_norm, lr):
        for l in range(self.n_layers):
            self.W[l] -= lr * grads_W[l]
            if self.norm[l] is not None:
                dgamma, dbeta = grads_norm[l]
                self.norm[l].gamma -= lr * dgamma
                self.norm[l].beta -= lr * dbeta
            else:
                self.b[l] -= lr * grads_b[l]

    def loss(self, y_pred, D, eps=1e-12):
        return -np.sum(D * np.log(np.clip(y_pred, eps, 1.0)))
