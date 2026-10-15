"""
Numerical gradient checks for Conv2D and SubsampleLayer, in isolation,
before composing them into the full network.
"""
import numpy as np
from conv2d import Conv2D
from subsample import SubsampleLayer

EPS_FD = 1e-5
REL_TOL = 1e-4
ABS_TOL = 1e-5
rng = np.random.default_rng(0)


def rel_err(a, n):
    return abs(a - n) / max(1e-12, abs(a) + abs(n))


def check_param(param, grad, forward_from_param, label, n_checks=12):
    flat_idx = rng.choice(param.size, size=min(n_checks, param.size), replace=False)
    worst = 0.0
    for idx in flat_idx:
        multi = np.unravel_index(idx, param.shape)
        orig = param[multi]
        param[multi] = orig + EPS_FD
        lp = forward_from_param()
        param[multi] = orig - EPS_FD
        lm = forward_from_param()
        param[multi] = orig
        numeric = (lp - lm) / (2 * EPS_FD)
        analytic = grad[multi]
        e = rel_err(analytic, numeric)
        worst = max(worst, e)
        assert e < REL_TOL or abs(analytic - numeric) < ABS_TOL, \
            f"{label}{multi}: analytic={analytic:.8f} numeric={numeric:.8f} rel_err={e:.2e}"
    return worst


def test_conv2d_full():
    x = rng.standard_normal((2, 3, 9, 9)) * 0.5
    conv = Conv2D(3, 4, 5, seed=1)

    def loss_fn(x_):
        out = conv.forward(x_)
        return 0.5 * np.sum(out ** 2)

    out = conv.forward(x)
    dout = out.copy()
    dx, dW, db = conv.backward(dout)

    def f_W():
        return 0.5 * np.sum(conv.forward(x) ** 2)

    def f_b():
        return 0.5 * np.sum(conv.forward(x) ** 2)

    wW = check_param(conv.W, dW, f_W, "W")
    wb = check_param(conv.b, db, f_b, "b")

    worst_x = 0.0
    idxs = rng.choice(x.size, size=10, replace=False)
    for idx in idxs:
        multi = np.unravel_index(idx, x.shape)
        orig = x[multi]
        x[multi] = orig + EPS_FD
        lp = loss_fn(x)
        x[multi] = orig - EPS_FD
        lm = loss_fn(x)
        x[multi] = orig
        numeric = (lp - lm) / (2 * EPS_FD)
        analytic = dx[multi]
        e = rel_err(analytic, numeric)
        worst_x = max(worst_x, e)
        assert e < REL_TOL or abs(analytic - numeric) < ABS_TOL, f"x{multi}: {analytic} vs {numeric}"

    print(f"Conv2D (full connectivity) gradcheck passed: worst rel err dW={wW:.2e} db={wb:.2e} dx={worst_x:.2e}")


def test_conv2d_sparse():
    connection_table = [[0, 1], [1, 2], [0, 2]]  # like C3's sparse pattern, scaled down
    x = rng.standard_normal((2, 3, 8, 8)) * 0.5
    conv = Conv2D(3, 3, 3, connection_table=connection_table, seed=2)

    assert conv.W[0, 2].sum() == 0.0 and conv.W[1, 0].sum() == 0.0 and conv.W[2, 1].sum() == 0.0, \
        "unconnected weight slices should be zero-initialized"

    out = conv.forward(x)
    dx, dW, db = conv.backward(out.copy())
    assert dW[0, 2].sum() == 0.0 and dW[1, 0].sum() == 0.0 and dW[2, 1].sum() == 0.0, \
        "unconnected weight slices should get zero gradient"

    def f_W():
        return 0.5 * np.sum(conv.forward(x) ** 2)

    w = check_param(conv.W, dW, f_W, "W_sparse", n_checks=8)
    print(f"Conv2D (sparse connectivity, C3-style) gradcheck passed: worst rel err dW={w:.2e}, zero-masked entries confirmed")


def test_subsample():
    x = rng.standard_normal((2, 4, 6, 6)) * 0.5
    layer = SubsampleLayer(4, seed=3)

    def loss_fn(x_):
        out = layer.forward(x_)
        return 0.5 * np.sum(out ** 2)

    out = layer.forward(x)
    dx, dcoeff, dbias = layer.backward(out.copy())

    def f_coeff():
        return 0.5 * np.sum(layer.forward(x) ** 2)

    def f_bias():
        return 0.5 * np.sum(layer.forward(x) ** 2)

    wc = check_param(layer.coeff, dcoeff, f_coeff, "coeff", n_checks=4)
    wb = check_param(layer.bias, dbias, f_bias, "bias", n_checks=4)

    worst_x = 0.0
    idxs = rng.choice(x.size, size=10, replace=False)
    for idx in idxs:
        multi = np.unravel_index(idx, x.shape)
        orig = x[multi]
        x[multi] = orig + EPS_FD
        lp = loss_fn(x)
        x[multi] = orig - EPS_FD
        lm = loss_fn(x)
        x[multi] = orig
        numeric = (lp - lm) / (2 * EPS_FD)
        analytic = dx[multi]
        e = rel_err(analytic, numeric)
        worst_x = max(worst_x, e)
        assert e < REL_TOL or abs(analytic - numeric) < ABS_TOL, f"x{multi}: {analytic} vs {numeric}"

    print(f"SubsampleLayer gradcheck passed: worst rel err dcoeff={wc:.2e} dbias={wb:.2e} dx={worst_x:.2e}")


if __name__ == "__main__":
    test_conv2d_full()
    test_conv2d_sparse()
    test_subsample()
