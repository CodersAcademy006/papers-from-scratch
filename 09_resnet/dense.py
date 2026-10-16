"""Fully-connected layer, duplicated from 08_vggnet/dense.py (self-
contained-folder convention). Used once here, as the final FC-10
(activation=False -- softmax+CE applied outside, same as 08_vggnet's
fc8). He init per the paper's own init citation ([13], see conv2d.py)."""
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
