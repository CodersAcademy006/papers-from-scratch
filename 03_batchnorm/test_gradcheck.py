"""
Numerical gradient check for BatchNorm1d.backward(), covering dx, dgamma,
dbeta. The dx check is the interesting one: it only passes if backward()
correctly accounts for x_i's influence on mu_B and sigma_B^2 (every x_i in
the batch affects the batch statistics used to normalize every other x_i).
attempts/check_naive_fails.py shows what happens if you don't -- worst
relative error 1.37, i.e. completely wrong, not a rounding issue.
"""
import numpy as np
from bn_layer import BatchNorm1d

EPS_FD = 1e-5
REL_TOL = 1e-4
ABS_TOL = 1e-6


def main():
    rng = np.random.default_rng(3)
    m, d = 8, 5
    X = rng.normal(size=(m, d))
    dy = rng.normal(size=(m, d))

    bn = BatchNorm1d(d)
    bn.gamma = rng.normal(size=d) + 1.0
    bn.beta = rng.normal(size=d)

    y, cache = bn.forward_train(X)
    dx_ana, dgamma_ana, dbeta_ana = bn.backward(dy, cache)

    def loss(Xp, gamma, beta):
        bn2 = BatchNorm1d(d)
        bn2.gamma, bn2.beta = gamma, beta
        y2, _ = bn2.forward_train(Xp)
        return np.sum(y2 * dy)

    worst = 0.0
    checked = 0

    # dx: perturb each input entry, batch statistics recomputed each time
    for i in range(m):
        for j in range(d):
            Xp, Xm = X.copy(), X.copy()
            Xp[i, j] += EPS_FD
            Xm[i, j] -= EPS_FD
            num = (loss(Xp, bn.gamma, bn.beta) - loss(Xm, bn.gamma, bn.beta)) / (2 * EPS_FD)
            ana = dx_ana[i, j]
            rel = abs(num - ana) / max(abs(num), abs(ana), ABS_TOL)
            assert np.isclose(num, ana, rtol=REL_TOL, atol=ABS_TOL), (
                f"dx[{i},{j}]: analytic={ana:.8f} numeric={num:.8f} rel_err={rel:.2e}"
            )
            worst = max(worst, rel)
            checked += 1

    # dgamma, dbeta: perturb the parameters themselves
    for k in range(d):
        gp, gm = bn.gamma.copy(), bn.gamma.copy()
        gp[k] += EPS_FD
        gm[k] -= EPS_FD
        num = (loss(X, gp, bn.beta) - loss(X, gm, bn.beta)) / (2 * EPS_FD)
        rel = abs(num - dgamma_ana[k]) / max(abs(num), abs(dgamma_ana[k]), ABS_TOL)
        assert np.isclose(num, dgamma_ana[k], rtol=REL_TOL, atol=ABS_TOL), (
            f"dgamma[{k}]: analytic={dgamma_ana[k]:.8f} numeric={num:.8f}"
        )
        worst = max(worst, rel)
        checked += 1

        bp, bm = bn.beta.copy(), bn.beta.copy()
        bp[k] += EPS_FD
        bm[k] -= EPS_FD
        num = (loss(X, bn.gamma, bp) - loss(X, bn.gamma, bm)) / (2 * EPS_FD)
        rel = abs(num - dbeta_ana[k]) / max(abs(num), abs(dbeta_ana[k]), ABS_TOL)
        assert np.isclose(num, dbeta_ana[k], rtol=REL_TOL, atol=ABS_TOL), (
            f"dbeta[{k}]: analytic={dbeta_ana[k]:.8f} numeric={num:.8f}"
        )
        worst = max(worst, rel)
        checked += 1

    print(f"gradcheck passed, {checked} entries (dx, dgamma, dbeta), worst relative error = {worst:.2e}")


if __name__ == "__main__":
    main()
