"""
Full-network gradient check for BNMLP.backward(), both with and without BN,
on a tiny random classification problem. bn_layer.py already has its own
isolated gradcheck (test_gradcheck.py) -- this one exists because composing
a correct layer with a correct dense layer doesn't automatically guarantee
the composition (indexing, cache plumbing, which layer gets which gradient)
is wired correctly. Same reasoning as 02_dropout's separate layer-level and
network-level checks.
"""
import numpy as np
from mlp import BNMLP

EPS = 1e-5
REL_TOL = 1e-4
ABS_TOL = 1e-6


def check(use_bn, seed):
    rng = np.random.default_rng(seed)
    n, d_in, d_out = 12, 6, 3
    X = rng.normal(size=(n, d_in))
    labels = rng.integers(0, d_out, size=n)
    D = np.zeros((n, d_out))
    D[np.arange(n), labels] = 1.0

    net = BNMLP([d_in, 5, 4, d_out], use_bn=use_bn, seed=seed)
    activations, bn_caches = net.forward_train(X)
    grads_W, grads_b, grads_bn = net.backward(activations, bn_caches, D)

    def loss_at(Wl, l):
        saved = net.W[l].copy()
        net.W[l] = Wl
        activations2, _ = net.forward_train(X)
        loss = net.loss(activations2[-1], D)
        net.W[l] = saved
        return loss

    worst = 0.0
    checked = 0
    for l in range(net.n_layers):
        flat = list(np.ndindex(net.W[l].shape))
        sample = rng.choice(len(flat), size=min(5, len(flat)), replace=False)
        for k in sample:
            idx = flat[k]
            Wp, Wm = net.W[l].copy(), net.W[l].copy()
            Wp[idx] += EPS
            Wm[idx] -= EPS
            num = (loss_at(Wp, l) - loss_at(Wm, l)) / (2 * EPS)
            ana = grads_W[l][idx]
            rel = abs(num - ana) / max(abs(num), abs(ana), ABS_TOL)
            assert np.isclose(num, ana, rtol=REL_TOL, atol=ABS_TOL), (
                f"use_bn={use_bn} layer {l} W{idx}: analytic={ana:.8f} numeric={num:.8f} rel_err={rel:.2e}"
            )
            worst = max(worst, rel)
            checked += 1

    print(f"use_bn={use_bn}: {checked} W entries checked, worst relative error = {worst:.2e}")


if __name__ == "__main__":
    check(use_bn=False, seed=1)
    check(use_bn=True, seed=1)
    print("mlp gradcheck passed for both use_bn=False and use_bn=True")
