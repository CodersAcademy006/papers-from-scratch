"""Isolated gradchecks for conv2d.py and maxpool.py before composing
the full VGG-A network. dense.py/dropout.py are unchanged duplicates
already gradchecked in 07_alexnet, not re-checked here."""
import numpy as np
from conv2d import Conv2D
from maxpool import MaxPool

EPS_FD = 1e-5
REL_TOL = 1e-3
ABS_TOL = 1e-4
rng = np.random.default_rng(0)


def check(name, x, f, df_dx, n_checks=8):
    worst = 0.0
    flat_idx = rng.choice(x.size, size=min(n_checks, x.size), replace=False)
    for idx in flat_idx:
        multi = np.unravel_index(idx, x.shape)
        orig = x[multi]
        x[multi] = orig + EPS_FD
        lp = f()
        x[multi] = orig - EPS_FD
        lm = f()
        x[multi] = orig
        numeric = (lp - lm) / (2 * EPS_FD)
        analytic = df_dx[multi]
        e = abs(analytic - numeric) / max(1e-12, abs(analytic) + abs(numeric))
        worst = max(worst, e)
        assert e < REL_TOL or abs(analytic - numeric) < ABS_TOL, \
            f"{name}{multi}: analytic={analytic:.6f} numeric={numeric:.6f} rel_err={e:.2e}"
    return worst


if __name__ == "__main__":
    # Conv2D, stride=1 padding=1 (VGG's only configuration)
    x = rng.standard_normal((2, 4, 6, 6)) * 0.3
    y_target = rng.standard_normal((2, 8, 6, 6))
    conv = Conv2D(4, 8, 3, stride=1, padding=1, seed=1)

    def loss_conv():
        out = conv.forward(x)
        return 0.5 * np.sum((out - y_target) ** 2)

    out = conv.forward(x)
    dout = out - y_target
    dx, dW, db = conv.backward(dout)

    w_x = check("x", x, loss_conv, dx)
    w_W = check("W", conv.W, loss_conv, dW)
    w_b = check("b", conv.b, loss_conv, db)
    print(f"Conv2D (stride=1, padding=1, 3x3) gradcheck passed: worst rel err dx={w_x:.2e} dW={w_W:.2e} db={w_b:.2e}")

    # MaxPool, 2x2 stride 2 (non-overlapping)
    x2 = rng.standard_normal((2, 4, 8, 8))
    y_target2 = rng.standard_normal((2, 4, 4, 4))
    pool = MaxPool(2, 2)

    def loss_pool():
        out = pool.forward(x2)
        return 0.5 * np.sum((out - y_target2) ** 2)

    out2 = pool.forward(x2)
    dout2 = out2 - y_target2
    dx2 = pool.backward(dout2)

    w_px = check("x", x2, loss_pool, dx2)
    print(f"MaxPool (2x2, s=2, non-overlapping) gradcheck passed: worst rel err dx={w_px:.2e}")
