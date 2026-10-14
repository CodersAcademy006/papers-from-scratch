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
from backprop import Network, make_symmetry_dataset

RNG_SEED = 42
EPS = 1e-5
REL_TOL = 1e-4


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


if __name__ == "__main__":
    main()
