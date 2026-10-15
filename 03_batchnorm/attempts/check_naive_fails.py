"""Standalone check: prove the naive backward (attempts/naive_batchnorm_backward.py)
fails a numerical gradient check, before trusting the "fix" in bn_layer.py.
Not part of the real test suite -- this is a one-off verification artifact,
run once, output pasted into JOURNAL.md entry 4."""
import numpy as np
from naive_batchnorm_backward import naive_bn_forward, naive_bn_backward

EPS = 1e-5
rng = np.random.default_rng(0)
m, d = 8, 4
X = rng.normal(size=(m, d))
gamma = rng.normal(size=d) + 1.0
beta = rng.normal(size=d)
dy = rng.normal(size=(m, d))  # pretend upstream gradient

y, cache = naive_bn_forward(X, gamma, beta)
dx_analytic, _, _ = naive_bn_backward(dy, cache)


def loss(Xp):
    y, _ = naive_bn_forward(Xp, gamma, beta)
    return np.sum(y * dy)  # dL/dy = dy by construction


worst = 0.0
for i in range(m):
    for j in range(d):
        Xp, Xm = X.copy(), X.copy()
        Xp[i, j] += EPS
        Xm[i, j] -= EPS
        num = (loss(Xp) - loss(Xm)) / (2 * EPS)
        ana = dx_analytic[i, j]
        rel = abs(num - ana) / max(abs(num), abs(ana), 1e-8)
        worst = max(worst, rel)

print(f"naive backward: worst relative error over {m*d} entries = {worst:.4f}")
print("(anything >> 1e-4 confirms the naive dx is wrong, as expected)")
