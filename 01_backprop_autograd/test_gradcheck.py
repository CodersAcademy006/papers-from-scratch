"""
Numerical gradient check for the hand-derived backward() in backprop.py.

This is the one thing that actually proves eqs (4)-(7) were implemented
correctly: compare each analytic dE/dw (from backward()) against a
central finite-difference estimate of the same quantity, obtained by
perturbing that one weight and re-measuring E (eq 3) directly. This is
independent of whether training ever converges -- a network can have a
perfectly correct gradient and still get stuck in a local minimum (see
JOURNAL.md entry 4), so this check must pass regardless of that.
"""
import numpy as np
from pathlib import Path
from backprop import Network, make_symmetry_dataset

RNG_SEED = 42
EPS = 1e-5
REL_TOL = 1e-4
CHECK_LOG = []  # (name, worst_rel_err) -- for figures/gradcheck_summary.png


def numeric_grad(net, X, D, param_name, index):
    param = getattr(net, param_name)
    orig = param[index]

    param[index] = orig + EPS
    _, y2 = net.forward(X)
    e_plus = net.error(y2, D)

    param[index] = orig - EPS
    _, y2 = net.forward(X)
    e_minus = net.error(y2, D)

    param[index] = orig
    return (e_plus - e_minus) / (2 * EPS)


def check_param(net, X, D, y1, y2, analytic, param_name, n_samples=5, rng=None):
    param = getattr(net, param_name)
    flat_indices = list(np.ndindex(param.shape))
    sample = rng.choice(len(flat_indices), size=min(n_samples, len(flat_indices)), replace=False)
    max_rel_err = 0.0
    for k in sample:
        idx = flat_indices[k]
        num = numeric_grad(net, X, D, param_name, idx)
        ana = analytic[idx]
        rel_err = abs(num - ana) / max(abs(num), abs(ana), 1e-8)
        max_rel_err = max(max_rel_err, rel_err)
        assert rel_err < REL_TOL, (
            f"{param_name}{idx}: analytic={ana:.8f} numeric={num:.8f} "
            f"rel_err={rel_err:.2e} (tolerance {REL_TOL:.0e})"
        )
    CHECK_LOG.append((param_name, max_rel_err))
    return max_rel_err


def main():
    rng = np.random.default_rng(RNG_SEED)
    X, D = make_symmetry_dataset()
    X = X[:8]  # a small subset of cases is enough to check the gradient math
    D = D[:8]

    net = Network(n_in=X.shape[1], n_hidden=2, n_out=1, seed=RNG_SEED)
    y1, y2 = net.forward(X)
    dW1, db1, dW2, db2 = net.backward(X, y1, y2, D)

    worst = 0.0
    for name, analytic in [("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2)]:
        worst = max(worst, check_param(net, X, D, y1, y2, analytic, name, rng=rng))

    print(f"gradcheck passed, worst relative error across sampled params = {worst:.2e}")
    save_summary_figure()


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
    ax.set_title("01_backprop_autograd gradcheck -- worst relative error per parameter")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "gradcheck_summary.png", dpi=150)
    plt.close(fig)
    print(f"saved {out_dir / 'gradcheck_summary.png'}")


if __name__ == "__main__":
    main()
