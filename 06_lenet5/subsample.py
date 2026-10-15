"""
S2/S4 sub-sampling (Section II.B, JOURNAL.md entry 2): NOT plain average
pooling. Each feature map has one trainable scalar coefficient and one
trainable scalar bias applied to the 2x2 average, before the squash.
"""
import numpy as np
from squash import squash, dsquash


class SubsampleLayer:
    def __init__(self, num_maps, seed=None):
        rng = np.random.default_rng(seed)
        bound = 2.4 / 4  # F_i = 4 inputs per unit (Appendix A)
        self.coeff = rng.uniform(-bound, bound, size=num_maps)
        self.bias = np.zeros(num_maps)

    def forward(self, x):
        N, C, H, W = x.shape
        self.avg = x.reshape(N, C, H // 2, 2, W // 2, 2).mean(axis=(3, 5))
        self.a = self.coeff[None, :, None, None] * self.avg + self.bias[None, :, None, None]
        self.out = squash(self.a)
        return self.out

    def backward(self, dout):
        da = dout * dsquash(self.a)
        dcoeff = (da * self.avg).sum(axis=(0, 2, 3))
        dbias = da.sum(axis=(0, 2, 3))
        davg = da * self.coeff[None, :, None, None]
        N, C, Ho, Wo = davg.shape
        dx = np.repeat(np.repeat(davg, 2, axis=2), 2, axis=3) / 4.0
        return dx, dcoeff, dbias
