"""Fully-connected layer with ReLU, duplicated from 07_alexnet/dense.py
(self-contained-folder convention). He init (fan-in-scaled), not the
paper's flat std=0.01 -- see conv2d.py and JOURNAL.md entry 2/3."""
import numpy as np


class Dense:
    def __init__(self, n_in, n_out, bias_init=0.0, activation=True, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_out, n_in))
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
