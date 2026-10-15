"""Numerical gradient check for LayerNorm1d.backward(), same approach as
03_batchnorm/test_gradcheck.py (this is its axis-swapped cousin, entry 2)."""
import numpy as np
from ln_layer import LayerNorm1d

EPS_FD = 1e-5
REL_TOL = 1e-4
ABS_TOL = 1e-6


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
            checked += 1

    for k in range(d):
        gp, gm = ln.gamma.copy(), ln.gamma.copy()
        gp[k] += EPS_FD
        gm[k] -= EPS_FD
        num = (loss(X, gp, ln.beta) - loss(X, gm, ln.beta)) / (2 * EPS_FD)
        assert np.isclose(num, dgamma_ana[k], rtol=REL_TOL, atol=ABS_TOL), (
            f"dgamma[{k}]: analytic={dgamma_ana[k]:.8f} numeric={num:.8f}"
        )
        checked += 1

        bp, bm = ln.beta.copy(), ln.beta.copy()
        bp[k] += EPS_FD
        bm[k] -= EPS_FD
        num = (loss(X, ln.gamma, bp) - loss(X, ln.gamma, bm)) / (2 * EPS_FD)
        assert np.isclose(num, dbeta_ana[k], rtol=REL_TOL, atol=ABS_TOL), (
            f"dbeta[{k}]: analytic={dbeta_ana[k]:.8f} numeric={num:.8f}"
        )
        checked += 1

    print(f"gradcheck passed, {checked} entries (dx, dgamma, dbeta), worst relative error = {worst:.2e}")


if __name__ == "__main__":
    main()
