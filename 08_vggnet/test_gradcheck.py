"""Isolated gradchecks for conv2d.py and maxpool.py before composing
the full VGG-A network. dense.py/dropout.py are unchanged duplicates
already gradchecked in 07_alexnet, not re-checked here."""
import numpy as np
from pathlib import Path
from conv2d import Conv2D
from maxpool import MaxPool

EPS_FD = 1e-5
REL_TOL = 1e-3
ABS_TOL = 1e-4
rng = np.random.default_rng(0)
CHECK_LOG = []  # (name, worst_rel_err) -- for figures/gradcheck_summary.png


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
    CHECK_LOG.append((name, worst))
    return worst


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
    ax.set_title("08_vggnet gradcheck -- worst relative error per layer/param")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "gradcheck_summary.png", dpi=150)
    plt.close(fig)
    print(f"saved {out_dir / 'gradcheck_summary.png'}")


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

    w_x = check("Conv2D.dx", x, loss_conv, dx)
    w_W = check("Conv2D.dW", conv.W, loss_conv, dW)
    w_b = check("Conv2D.db", conv.b, loss_conv, db)
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

    w_px = check("MaxPool.dx", x2, loss_pool, dx2)
    print(f"MaxPool (2x2, s=2, non-overlapping) gradcheck passed: worst rel err dx={w_px:.2e}")
    save_summary_figure()
