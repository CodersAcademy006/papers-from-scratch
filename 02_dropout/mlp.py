"""
Improving Neural Networks by Preventing Co-adaptation of Feature Detectors
Hinton, Srivastava, Krizhevsky, Sutskever & Salakhutdinov, arXiv:1207.0580 (2012).

A feedforward MLP with the paper's exact mechanism: Bernoulli dropout
masks at train time, the "mean network" weight-halving trick at test
time, a per-hidden-unit max-norm weight constraint, and the scheduled
momentum update rule from Appendix A.1. No autodiff library; forward
and backward are hand-derived, same spirit as 01_backprop_autograd
(duplicated here rather than imported, so each paper's folder stays
self-contained and independently readable).

Note on activation function: the paper's main text does not state a
nonlinearity for the plain-MLP MNIST/TIMIT/Reuters experiments (only
Appendix F, for the CIFAR-10/ImageNet *convolutional* nets, explicitly
says "max-with-zero", i.e. ReLU). Appendix A.1 does say "cross-entropy
objective function" for MNIST, which is used here. Hidden units are
implemented as logistic sigmoid, matching the era and 01_backprop's
architecture, since the paper leaves this specific choice unstated for
this experiment -- documented as a filled-in gap, not a transcribed fact.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class DropoutMLP:
    def __init__(self, layer_sizes, hidden_keep=0.5, input_keep=1.0,
                 maxnorm_sq=15.0, weight_std=0.01, seed=None):
        """layer_sizes: e.g. [784, 256, 128, 10] (last size is n_classes).
        hidden_keep: probability a hidden unit survives a dropout mask (paper: 0.5).
        input_keep: probability a raw input unit survives (paper: 0.8 when used, 1.0 = off).
        maxnorm_sq: upper bound on squared L2 norm of each hidden unit's incoming weight row.
        """
        rng = np.random.default_rng(seed)
        self.sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.hidden_keep = hidden_keep
        self.input_keep = input_keep
        self.maxnorm_sq = maxnorm_sq

        self.W = []
        self.b = []
        self.vW = []
        self.vb = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            W = rng.normal(0.0, weight_std, size=(n_out, n_in))
            b = np.zeros(n_out)
            self.W.append(W)
            self.b.append(b)
            self.vW.append(np.zeros_like(W))
            self.vb.append(np.zeros_like(b))

    def _masks(self, X, rng):
        """One Bernoulli mask per dropped layer (input layer + every hidden
        layer except the output layer), scaled by 1 so this is the paper's
        raw-masking convention (no train-time rescaling; scaling happens
        once, at test time, on the weights -- see predict())."""
        masks = [rng.binomial(1, self.input_keep, size=X.shape).astype(float)]
        for l in range(self.n_layers - 1):  # hidden layers only, not the output layer
            n_units = self.sizes[l + 1]
            masks.append(rng.binomial(1, self.hidden_keep, size=(X.shape[0], n_units)).astype(float))
        return masks

    def forward_train(self, X, rng):
        masks = self._masks(X, rng)
        return self.forward_with_masks(X, masks), masks

    def forward_with_masks(self, X, masks):
        """Forward pass with an explicit, externally-supplied set of masks.
        Split out from forward_train so a gradient check can hold the masks
        fixed across the +eps/-eps perturbations (otherwise a fresh random
        mask each call would make the finite-difference estimate meaningless,
        since E would then depend on the mask draw as well as the weight)."""
        activations = [X * masks[0]]
        for l in range(self.n_layers):
            x = activations[-1] @ self.W[l].T + self.b[l]
            if l < self.n_layers - 1:
                a = sigmoid(x) * masks[l + 1]
            else:
                a = softmax(x)  # output layer: no dropout, no mask
            activations.append(a)
        return activations

    def backward(self, activations, masks, D):
        """D: (n_cases, n_classes) one-hot targets. Softmax+cross-entropy
        gradient at the output is the same clean form as paper 01's eq (4)
        for sigmoid+squared-error: dL/dx_out = y - d. Hidden-layer gradients
        get multiplied by that layer's own dropout mask, since a masked
        unit contributed nothing to the forward pass and must get zero
        gradient, not just zero activation."""
        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers

        y_out = activations[-1]
        dE_dx = y_out - D  # softmax + cross-entropy, summed-not-averaged to match eq (3)'s sum_c convention

        for l in reversed(range(self.n_layers)):
            a_in = activations[l]
            grads_W[l] = dE_dx.T @ a_in
            grads_b[l] = dE_dx.sum(axis=0)
            if l > 0:
                dE_da = dE_dx @ self.W[l]
                a_hidden = activations[l]  # already masked on the way up
                dE_da = dE_da * masks[l]   # a dropped unit gets zero gradient, not just zero activation
                sig = a_hidden  # masked sigmoid output; derivative below uses the pre-mask sigmoid value
                # recover the pre-mask sigmoid activation for the y(1-y) derivative term:
                # masked units have a_hidden == 0 already, and get zeroed again by masks[l] above,
                # so using a_hidden directly here is safe (0*(1-0)=0 contributes nothing for dropped units).
                dE_dx = dE_da * sig * (1.0 - sig)

        return grads_W, grads_b

    def step(self, grads_W, grads_b, epsilon, p):
        """Appendix A.1's scheduled-momentum update:
        delta_w_t = p_t * delta_w_{t-1} - (1 - p_t) * epsilon_t * <grad>_t
        followed by the per-hidden-unit max-norm constraint.
        """
        for l in range(self.n_layers):
            self.vW[l] = p * self.vW[l] - (1 - p) * epsilon * grads_W[l]
            self.vb[l] = p * self.vb[l] - (1 - p) * epsilon * grads_b[l]
            self.W[l] += self.vW[l]
            self.b[l] += self.vb[l]
            if l < self.n_layers - 1:  # max-norm on hidden layers only, not the output layer
                row_sq = np.sum(self.W[l] ** 2, axis=1, keepdims=True)
                scale = np.sqrt(self.maxnorm_sq / np.maximum(row_sq, self.maxnorm_sq))
                self.W[l] *= scale

    def predict(self, X):
        """Paper's 'mean network': run inference through the FULL network
        (no masking), but scale each weight matrix by the keep-probability
        of its SENDING layer, since during training only that fraction of
        the sending units were active on average."""
        keep_probs = [self.input_keep] + [self.hidden_keep] * (self.n_layers - 1)
        a = X
        for l in range(self.n_layers):
            W_scaled = self.W[l] * keep_probs[l]
            x = a @ W_scaled.T + self.b[l]
            a = sigmoid(x) if l < self.n_layers - 1 else softmax(x)
        return a

    def loss(self, y_pred, D, eps=1e-12):
        """Summed cross-entropy, sum_c sum_j (matching eq (3)'s sum_c sum_j convention)."""
        return -np.sum(D * np.log(np.clip(y_pred, eps, 1.0)))
