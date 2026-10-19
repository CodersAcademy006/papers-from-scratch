"""Numerical gradient check for LayerNorm1d.backward(), same approach as
03_batchnorm/test_gradcheck.py (this is its axis-swapped cousin, entry 2)."""
import numpy as np
from pathlib import Path
from ln_layer import LayerNorm1d

EPS_FD = 1e-5
REL_TOL = 1e-4
ABS_TOL = 1e-6
CHECK_LOG = []  # (name, worst_rel_err) -- for figures/gradcheck_summary.png


def main():
    rng = np.random.default_rng(4)
    m, d = 6, 7
    X = rng.normal(size=(m, d))
    dy = rng.normal(size=(m, d))

    ln = LayerNorm1d(d)
    ln.gamma = rng.normal(size=d) + 1.0
    ln.beta = rng.normal(size=d)

    y, cache = ln.forward(X)
    dx_ana, dgamma_ana, dbeta_ana = ln.backward(dy, cache)

    def loss(Xp, gamma, beta):
        ln2 = LayerNorm1d(d)
        ln2.gamma, ln2.beta = gamma, beta
        y2, _ = ln2.forward(Xp)
        return np.sum(y2 * dy)

    worst = 0.0
    checked = 0
    dx_worst = dgamma_worst = dbeta_worst = 0.0

    for i in range(m):
        for j in range(d):
            Xp, Xm = X.copy(), X.copy()
            Xp[i, j] += EPS_FD
            Xm[i, j] -= EPS_FD
            num = (loss(Xp, ln.gamma, ln.beta) - loss(Xm, ln.gamma, ln.beta)) / (2 * EPS_FD)
            ana = dx_ana[i, j]
            rel = abs(num - ana) / max(abs(num), abs(ana), ABS_TOL)
            assert np.isclose(num, ana, rtol=REL_TOL, atol=ABS_TOL), (
                f"dx[{i},{j}]: analytic={ana:.8f} numeric={num:.8f} rel_err={rel:.2e}"
            )
            worst = max(worst, rel)
            dx_worst = max(dx_worst, rel)
            checked += 1
    CHECK_LOG.append(("dx", dx_worst))

    for k in range(d):
        gp, gm = ln.gamma.copy(), ln.gamma.copy()
        gp[k] += EPS_FD
        gm[k] -= EPS_FD
        num = (loss(X, gp, ln.beta) - loss(X, gm, ln.beta)) / (2 * EPS_FD)
        rel = abs(num - dgamma_ana[k]) / max(abs(num), abs(dgamma_ana[k]), ABS_TOL)
        assert np.isclose(num, dgamma_ana[k], rtol=REL_TOL, atol=ABS_TOL), (
            f"dgamma[{k}]: analytic={dgamma_ana[k]:.8f} numeric={num:.8f}"
        )
        worst = max(worst, rel)
        dgamma_worst = max(dgamma_worst, rel)
        checked += 1

        bp, bm = ln.beta.copy(), ln.beta.copy()
        bp[k] += EPS_FD
        bm[k] -= EPS_FD
        num = (loss(X, ln.gamma, bp) - loss(X, ln.gamma, bm)) / (2 * EPS_FD)
        rel = abs(num - dbeta_ana[k]) / max(abs(num), abs(dbeta_ana[k]), ABS_TOL)
        assert np.isclose(num, dbeta_ana[k], rtol=REL_TOL, atol=ABS_TOL), (
            f"dbeta[{k}]: analytic={dbeta_ana[k]:.8f} numeric={num:.8f}"
        )
        worst = max(worst, rel)
        dbeta_worst = max(dbeta_worst, rel)
        checked += 1
    CHECK_LOG.append(("dgamma", dgamma_worst))
    CHECK_LOG.append(("dbeta", dbeta_worst))

    print(f"gradcheck passed, {checked} entries (dx, dgamma, dbeta), worst relative error = {worst:.2e}")
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
    ax.set_title("05_layernorm gradcheck -- worst relative error per group")
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
