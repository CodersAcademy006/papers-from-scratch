"""
End-to-end gradcheck for the composed LeNet5 network: forward -> softmax
cross-entropy -> backward, checked against finite differences on a small
batch. Unlike the ReLU networks elsewhere in this repo, LeNet-5's squash
(tanh-based) is smooth everywhere, so no kink-adjacent filtering is
needed here.
"""
import numpy as np
from lenet5 import LeNet5

EPS_FD = 1e-4
REL_TOL = 1e-3
ABS_TOL = 1e-4
rng = np.random.default_rng(0)


def one_hot(y, n=10):
    out = np.zeros((len(y), n))
    out[np.arange(len(y)), y] = 1
    return out


def loss_only(net, x, y_onehot):
    logits = net.forward(x)
    loss, _, _ = net.softmax_ce(logits, y_onehot)
    return loss


def check_param(net, x, y_onehot, param, grad, label, n_checks=6):
    flat_idx = rng.choice(param.size, size=min(n_checks, param.size), replace=False)
    worst = 0.0
    for idx in flat_idx:
        multi = np.unravel_index(idx, param.shape)
        orig = param[multi]
        param[multi] = orig + EPS_FD
        lp = loss_only(net, x, y_onehot)
        param[multi] = orig - EPS_FD
        lm = loss_only(net, x, y_onehot)
        param[multi] = orig
        numeric = (lp - lm) / (2 * EPS_FD)
        analytic = grad[multi]
        e = abs(analytic - numeric) / max(1e-12, abs(analytic) + abs(numeric))
        worst = max(worst, e)
        assert e < REL_TOL or abs(analytic - numeric) < ABS_TOL, \
            f"{label}{multi}: analytic={analytic:.6f} numeric={numeric:.6f} rel_err={e:.2e}"
    return worst


if __name__ == "__main__":
    net = LeNet5(seed=0)
    x = rng.standard_normal((3, 1, 32, 32)) * 0.3
    y = rng.integers(0, 10, size=3)
    y_onehot = one_hot(y)

    logits = net.forward(x)
    loss, dlogits, _ = net.softmax_ce(logits, y_onehot)
    _, grads = net.backward(dlogits)

    param_map = {
        'C1.W': net.C1.W, 'C1.b': net.C1.b,
        'S2.coeff': net.S2.coeff, 'S2.bias': net.S2.bias,
        'C3.W': net.C3.W, 'C3.b': net.C3.b,
        'S4.coeff': net.S4.coeff, 'S4.bias': net.S4.bias,
        'C5.W': net.C5.W, 'C5.b': net.C5.b,
        'F6.W': net.F6.W, 'F6.b': net.F6.b,
        'out.W': net.out.W, 'out.b': net.out.b,
    }

    worst_overall = 0.0
    for name, param in param_map.items():
        w = check_param(net, x, y_onehot, param, grads[name], name)
        worst_overall = max(worst_overall, w)
        print(f"  {name}: worst rel err = {w:.2e}")

    print(f"\nfull LeNet5 gradcheck passed, worst relative error over all params = {worst_overall:.2e}")
