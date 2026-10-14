"""
Numerical gradient check for DropoutMLP.backward(), with the dropout
masks held FIXED across the +eps/-eps perturbations used for the finite
difference. This matters specifically because of dropout: if a fresh
random mask were drawn on every forward call, the loss would depend on
the mask draw as well as the weight being perturbed, and the finite
difference would be comparing two different stochastic functions, not
testing the gradient at all. Holding the mask fixed reduces this to
exactly the same kind of check as 01_backprop_autograd's gradcheck,
just through a masked subnetwork instead of the full one.
"""
import numpy as np
from mlp import DropoutMLP

RNG_SEED = 7
EPS = 1e-5
REL_TOL = 1e-4
ABS_TOL = 1e-6  # floor for near-zero gradients (e.g. a dropped unit's weight), where pure
                # relative error is numerically unstable even though both values are ~0


def numeric_grad(net, X, masks, D, layer, kind, index):
    param = net.W[layer] if kind == "W" else net.b[layer]
    orig = param[index]

    param[index] = orig + EPS
    activations = net.forward_with_masks(X, masks)
    e_plus = net.loss(activations[-1], D)

    param[index] = orig - EPS
    activations = net.forward_with_masks(X, masks)
    e_minus = net.loss(activations[-1], D)

    param[index] = orig
    return (e_plus - e_minus) / (2 * EPS)


def main():
    rng = np.random.default_rng(RNG_SEED)
    n_cases = 6
    X = rng.normal(size=(n_cases, 10))
    labels = rng.integers(0, 4, size=n_cases)
    D = np.zeros((n_cases, 4))
    D[np.arange(n_cases), labels] = 1.0

    net = DropoutMLP([10, 8, 6, 4], hidden_keep=0.5, input_keep=0.8,
                      maxnorm_sq=1e9, seed=RNG_SEED)  # maxnorm effectively off for this check

    mask_rng = np.random.default_rng(123)
    masks = net._masks(X, mask_rng)

    activations = net.forward_with_masks(X, masks)
    grads_W, grads_b = net.backward(activations, masks, D)

    worst = 0.0
    checked = 0
    for layer in range(net.n_layers):
        for kind, grads, param in [("W", grads_W[layer], net.W[layer]),
                                    ("b", grads_b[layer], net.b[layer])]:
            flat_indices = list(np.ndindex(param.shape))
            sample = rng.choice(len(flat_indices), size=min(4, len(flat_indices)), replace=False)
            for k in sample:
                idx = flat_indices[k]
                ana = grads[idx]
                num = numeric_grad(net, X, masks, D, layer, kind, idx)
                rel_err = abs(num - ana) / max(abs(num), abs(ana), 1e-8)
                worst = max(worst, rel_err)
                checked += 1
                ok = np.isclose(num, ana, rtol=REL_TOL, atol=ABS_TOL)
                assert ok, (
                    f"layer {layer} {kind}{idx}: analytic={ana:.10f} numeric={num:.10f} "
                    f"rel_err={rel_err:.2e}"
                )

    print(f"gradcheck passed, {checked} params sampled across {net.n_layers} layers, "
          f"worst relative error = {worst:.2e}")


if __name__ == "__main__":
    main()
