"""End-to-end gradcheck for the composed ResNet20_CIFAR. Unlike VGG's
gradcheck (08_vggnet/test_vgg_gradcheck.py, forced train=False to make
dropout deterministic), this one uses train=True directly -- BatchNorm2d's
forward_train output depends only on the current batch's own mean/var
(recomputed fresh from whatever x is passed, perturbed or not), never on
the running_mean/running_var it accumulates as a side effect, so there's
no dropout-style nondeterminism to work around."""
import numpy as np
from resnet_cifar import ResNet20_CIFAR

EPS_FD = 1e-8
REL_TOL = 1e-3
ABS_TOL = 1e-4
rng = np.random.default_rng(0)


def one_hot(y, n=10):
    out = np.zeros((len(y), n))
    out[np.arange(len(y)), y] = 1
    return out


def loss_only(net, x, y_onehot):
    logits = net.forward(x, train=True)
    loss, _, _ = net.softmax_ce(logits, y_onehot)
    return loss


def check_param(net, x, y_onehot, param, grad, label, n_checks=5):
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
    net = ResNet20_CIFAR(seed=0)
    x = rng.standard_normal((2, 3, 32, 32)) * 0.3
    y = rng.integers(0, 10, size=2)
    y_onehot = one_hot(y)

    logits = net.forward(x, train=True)
    loss, dlogits, _ = net.softmax_ce(logits, y_onehot)
    _, grads = net.backward(dlogits)

    params = net.params()
    worst_overall = 0.0
    for name, param in params.items():
        w = check_param(net, x, y_onehot, param, grads[name], name)
        worst_overall = max(worst_overall, w)
        print(f"  {name}: worst rel err = {w:.2e}")

    print(f"\nfull ResNet20_CIFAR gradcheck passed, worst relative error over all params = {worst_overall:.2e}")
