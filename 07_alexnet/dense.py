"""Fully-connected layer with ReLU, AlexNet's own init (Section 5):
weights ~ N(0, 0.01^2), bias constant (1 for the first two hidden FC
layers to help ReLUs start active, 0 for the output layer)."""
import numpy as np


class Dense:
    def __init__(self, n_in, n_out, bias_init=0.0, activation=True, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(n_out, n_in))
        self.b = np.full(n_out, bias_init, dtype=np.float64)
        self.activation = activation

    def forward(self, x):
        self.x = x
        self.a = x @ self.W.T + self.b
        return np.maximum(0, self.a) if self.activation else self.a

    def backward(self, dout):
        da = dout * (self.a > 0) if self.activation else dout
        dW = da.T @ self.x
        db = da.sum(axis=0)
        dx = da @ self.W
        return dx, dW, db
