"""
Local Response Normalization (Section 3.3, JOURNAL.md entry 2):
b^i_xy = a^i_xy / (k + alpha * sum_{j in window(i)} (a^j_xy)^2)^beta
normalizing across n=5 ADJACENT CHANNELS at the same spatial position,
not a spatial window.
"""
import numpy as np

K, N_ADJ, ALPHA, BETA = 2.0, 5, 1e-4, 0.75


class LRN:
    def forward(self, a):
        N, C, H, W = a.shape
        sq = a ** 2
        self.denom = np.empty_like(a)
        for i in range(C):
            lo = max(0, i - N_ADJ // 2)
            hi = min(C, i + N_ADJ // 2 + 1)
            self.denom[:, i] = K + ALPHA * sq[:, lo:hi].sum(axis=1)
        self.a = a
        self.pow_denom = self.denom ** BETA
        return a / self.pow_denom

    def backward(self, dout):
        N, C, H, W = self.a.shape
        # d(out_i)/d(a_j): direct term when i==j, cross term through the
        # shared denominator for every i whose window includes j.
        da = np.zeros_like(self.a)
        for i in range(C):
            lo = max(0, i - N_ADJ // 2)
            hi = min(C, i + N_ADJ // 2 + 1)
            # direct term: d(a_i / denom_i^beta)/d(a_i)
            da[:, i] += dout[:, i] / self.pow_denom[:, i]
            # cross term: every j in [lo,hi) contributes to denom_i via a_j^2
            cross_coef = dout[:, i] * self.a[:, i] * (-BETA) * self.denom[:, i] ** (-BETA - 1) * ALPHA * 2
            for j in range(lo, hi):
                da[:, j] += cross_coef * self.a[:, j]
        return da
