"""Numerical gradient checks for Conv2D (stride+padding), LRN, and
MaxPool, in isolation, before composing anything."""
import numpy as np
from pathlib import Path
from conv2d import Conv2D
from lrn import LRN
from maxpool import MaxPool

EPS_FD = 1e-5
REL_TOL = 1e-3
ABS_TOL = 1e-4
rng = np.random.default_rng(0)
CHECK_LOG = []  # (name, worst_rel_err) -- for figures/gradcheck_summary.png


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
    CHECK_LOG.append(("Conv2D.dW", wW))
    CHECK_LOG.append(("Conv2D.db", wb))
    CHECK_LOG.append(("Conv2D.dx", wx))


def test_lrn():
    x = rng.standard_normal((2, 6, 4, 4)) * 0.5 + 1.0  # keep away from 0
    layer = LRN()

    def f():
        return 0.5 * np.sum(layer.forward(x) ** 2)

    out = layer.forward(x)
    dx = layer.backward(out.copy())
    w = check_array(x, dx, f, "x", n_checks=12)
    print(f"LRN gradcheck passed: worst rel err dx={w:.2e}")
    CHECK_LOG.append(("LRN.dx", w))


def test_maxpool():
    x = rng.standard_normal((2, 3, 7, 7))
    layer = MaxPool(z=3, s=2)

    def f():
        return 0.5 * np.sum(layer.forward(x) ** 2)

    out = layer.forward(x)
    dx = layer.backward(out.copy())
    w = check_array(x, dx, f, "x", n_checks=12)
    print(f"MaxPool (z=3,s=2 overlapping) gradcheck passed: worst rel err dx={w:.2e}")
    CHECK_LOG.append(("MaxPool.dx", w))


def save_summary_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    STATUS_GOOD = "#0ca30c"
    fig, ax = plt.subplots(figsize=(7, max(2, 0.4 * len(CHECK_LOG))))
    names = [n for n, _ in CHECK_LOG]
    errs = [max(e, 1e-13) for _, e in CHECK_LOG]
    ax.barh(range(len(CHECK_LOG)), errs, color=STATUS_GOOD, height=0.6)
    ax.set_yticks(range(len(CHECK_LOG)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.axvline(REL_TOL, color="#555555", linestyle="--", linewidth=1, label=f"REL_TOL ({REL_TOL:.0e})")
    ax.set_xlabel("worst relative error vs. finite-difference (log scale)")
    ax.set_title("07_alexnet gradcheck -- worst relative error per layer/param")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "gradcheck_summary.png", dpi=150)
    plt.close(fig)
    print(f"saved {out_dir / 'gradcheck_summary.png'}")


if __name__ == "__main__":
    test_conv2d_stride_padding()
    test_lrn()
    test_maxpool()
    save_summary_figure()
