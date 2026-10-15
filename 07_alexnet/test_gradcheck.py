"""Numerical gradient checks for Conv2D (stride+padding), LRN, and
MaxPool, in isolation, before composing anything."""
import numpy as np
from conv2d import Conv2D
from lrn import LRN
from maxpool import MaxPool

EPS_FD = 1e-5
REL_TOL = 1e-3
ABS_TOL = 1e-4
rng = np.random.default_rng(0)


def rel_err(a, n):
    return abs(a - n) / max(1e-12, abs(a) + abs(n))


def check_array(x, dx, loss_fn, label, n_checks=10):
    idxs = rng.choice(x.size, size=min(n_checks, x.size), replace=False)
    worst = 0.0
    for idx in idxs:
        multi = np.unravel_index(idx, x.shape)
        orig = x[multi]
        x[multi] = orig + EPS_FD
        lp = loss_fn()
        x[multi] = orig - EPS_FD
        lm = loss_fn()
        x[multi] = orig
        numeric = (lp - lm) / (2 * EPS_FD)
        analytic = dx[multi]
        e = rel_err(analytic, numeric)
        worst = max(worst, e)
        assert e < REL_TOL or abs(analytic - numeric) < ABS_TOL, \
            f"{label}{multi}: analytic={analytic:.8f} numeric={numeric:.8f} rel_err={e:.2e}"
    return worst


def test_conv2d_stride_padding():
    x = rng.standard_normal((2, 3, 9, 9)) * 0.5
    conv = Conv2D(3, 4, 5, stride=2, padding=2, seed=1)

    out = conv.forward(x)
    dout = out.copy()
    dx, dW, db = conv.backward(dout)

    def f():
        return 0.5 * np.sum(conv.forward(x) ** 2)

    wW = check_array(conv.W, dW, f, "W")
    wb = check_array(conv.b, db, f, "b")
    wx = check_array(x, dx, f, "x")
    print(f"Conv2D (stride=2, padding=2) gradcheck passed: worst rel err dW={wW:.2e} db={wb:.2e} dx={wx:.2e}")


def test_lrn():
    x = rng.standard_normal((2, 6, 4, 4)) * 0.5 + 1.0  # keep away from 0
    layer = LRN()

    def f():
        return 0.5 * np.sum(layer.forward(x) ** 2)

    out = layer.forward(x)
    dx = layer.backward(out.copy())
    w = check_array(x, dx, f, "x", n_checks=12)
    print(f"LRN gradcheck passed: worst rel err dx={w:.2e}")


def test_maxpool():
    x = rng.standard_normal((2, 3, 7, 7))
    layer = MaxPool(z=3, s=2)

    def f():
        return 0.5 * np.sum(layer.forward(x) ** 2)

    out = layer.forward(x)
    dx = layer.backward(out.copy())
    w = check_array(x, dx, f, "x", n_checks=12)
    print(f"MaxPool (z=3,s=2 overlapping) gradcheck passed: worst rel err dx={w:.2e}")


if __name__ == "__main__":
    test_conv2d_stride_padding()
    test_lrn()
    test_maxpool()
