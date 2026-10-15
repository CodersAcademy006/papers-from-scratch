"""
LeNet-5 (Fig. 2 / Section II.B, JOURNAL.md entry 2): C1-S2-C3-S4-C5-F6,
each hand-rolled from Conv2D/SubsampleLayer. Output substituted with
softmax+cross-entropy on a linear 84->10 layer, see JOURNAL.md entry 3.
"""
import numpy as np
from conv2d import Conv2D
from subsample import SubsampleLayer
from squash import squash, dsquash

# Table I, JOURNAL.md entry 2 -- verified against the paper's own 1,516
# parameter count for C3.
C3_TABLE = [
    [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 0, 1],
    [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 0], [4, 5, 0, 1], [5, 0, 1, 2],
    [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5],
    [0, 1, 2, 3, 4, 5],
]


class Dense:
    def __init__(self, n_in, n_out, seed=None, activation=True):
        rng = np.random.default_rng(seed)
        bound = 2.4 / n_in
        self.W = rng.uniform(-bound, bound, size=(n_out, n_in))
        self.b = np.zeros(n_out)
        self.activation = activation

    def forward(self, x):
        self.x = x
        self.a = x @ self.W.T + self.b
        return squash(self.a) if self.activation else self.a

    def backward(self, dout):
        da = dout * dsquash(self.a) if self.activation else dout
        dW = da.T @ self.x
        db = da.sum(axis=0)
        dx = da @ self.W
        return dx, dW, db


class LeNet5:
    def __init__(self, seed=0):
        rng_seed = iter(range(seed * 100, seed * 100 + 10))
        self.C1 = Conv2D(1, 6, 5, seed=next(rng_seed))
        self.S2 = SubsampleLayer(6, seed=next(rng_seed))
        self.C3 = Conv2D(6, 16, 5, connection_table=C3_TABLE, seed=next(rng_seed))
        self.S4 = SubsampleLayer(16, seed=next(rng_seed))
        self.C5 = Conv2D(16, 120, 5, seed=next(rng_seed))
        self.F6 = Dense(120, 84, seed=next(rng_seed))
        self.out = Dense(84, 10, seed=next(rng_seed), activation=False)

    def forward(self, x):
        self._c1_pre = self.C1.forward(x)
        self.h1 = squash(self._c1_pre)
        self.h2 = self.S2.forward(self.h1)
        self._c3_pre = self.C3.forward(self.h2)
        self.h3 = squash(self._c3_pre)
        self.h4 = self.S4.forward(self.h3)
        self._c5_pre = self.C5.forward(self.h4)
        self.h5 = squash(self._c5_pre)  # (N, 120, 1, 1)
        flat = self.h5.reshape(self.h5.shape[0], -1)
        self.h6 = self.F6.forward(flat)
        logits = self.out.forward(self.h6)
        return logits

    @staticmethod
    def softmax_ce(logits, y_onehot):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        N = logits.shape[0]
        loss = -np.sum(y_onehot * np.log(probs + 1e-12)) / N
        dlogits = (probs - y_onehot) / N
        return loss, dlogits, probs

    def backward(self, dlogits):
        dh6, dW_out, db_out = self.out.backward(dlogits)
        dh5_flat, dW_F6, db_F6 = self.F6.backward(dh6)
        dh5 = dh5_flat.reshape(self.h5.shape)

        dc5_pre = dh5 * dsquash(self._c5_pre)
        dh4, dW_C5, db_C5 = self.C5.backward(dc5_pre)

        dh3, dcoeff_S4, dbias_S4 = self.S4.backward(dh4)

        dc3_pre = dh3 * dsquash(self._c3_pre)
        dh2, dW_C3, db_C3 = self.C3.backward(dc3_pre)

        dh1, dcoeff_S2, dbias_S2 = self.S2.backward(dh2)

        dc1_pre = dh1 * dsquash(self._c1_pre)
        dx, dW_C1, db_C1 = self.C1.backward(dc1_pre)

        grads = {
            'C1.W': dW_C1, 'C1.b': db_C1,
            'S2.coeff': dcoeff_S2, 'S2.bias': dbias_S2,
            'C3.W': dW_C3, 'C3.b': db_C3,
            'S4.coeff': dcoeff_S4, 'S4.bias': dbias_S4,
            'C5.W': dW_C5, 'C5.b': db_C5,
            'F6.W': dW_F6, 'F6.b': db_F6,
            'out.W': dW_out, 'out.b': db_out,
        }
        return dx, grads
