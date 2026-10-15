"""
Full-network gradient check for NormMLP.backward(), for all three
norm_mode values. The ReLU-mask bookkeeping in backward() looks a little
unusual on a read-through (see the comment in mlp_norm.py) -- this check
is the actual arbiter of whether it's correct, not a manual trace.
"""
import numpy as np
from mlp_norm import NormMLP

EPS = 1e-5
REL_TOL = 1e-4
ABS_TOL = 1e-6
KINK_MARGIN = 1e-3


def check(norm_mode, seed):
    rng = np.random.default_rng(seed)
    n, d_in, d_out = 12, 6, 3
    X = rng.normal(size=(n, d_in)) * 2.0  # *2.0: push pre-activations away from the ReLU kink (04_adam entry 6)
    labels = rng.integers(0, d_out, size=n)
    D = np.zeros((n, d_out))
    D[np.arange(n), labels] = 1.0

    net = NormMLP([d_in, 5, 4, d_out], norm_mode=norm_mode, weight_std=0.3, seed=seed)
    activations, caches = net.forward_train(X)
    grads_W, grads_b, grads_norm = net.backward(activations, caches, D)

    # exclude kink-adjacent output units from sampling, same discipline as 04_adam/test_gradcheck.py
    safe_units = []
    for l in range(net.n_layers):
        if l < net.n_layers - 1:
            z = activations[l] @ net.W[l].T
            min_abs = np.min(np.abs(z), axis=0)
            safe_units.append(set(np.where(min_abs > KINK_MARGIN)[0]))
        else:
            safe_units.append(set(range(net.sizes[l + 1])))

    def loss_at(Wl, l):
        saved = net.W[l].copy()
        net.W[l] = Wl
        a2, _ = net.forward_train(X)
        loss = net.loss(a2[-1], D)
        net.W[l] = saved
        return loss

    worst = 0.0
    checked = 0
    skipped = 0
    for l in range(net.n_layers):
        flat = list(np.ndindex(net.W[l].shape))
        candidates = rng.permutation(len(flat))
        taken = 0
        for k in candidates:
            idx = flat[k]
            unit = idx[0]
            if unit not in safe_units[l]:
                skipped += 1
                continue
            Wp, Wm = net.W[l].copy(), net.W[l].copy()
            Wp[idx] += EPS
            Wm[idx] -= EPS
            num = (loss_at(Wp, l) - loss_at(Wm, l)) / (2 * EPS)
            ana = grads_W[l][idx]
            rel = abs(num - ana) / max(abs(num), abs(ana), ABS_TOL)
            assert np.isclose(num, ana, rtol=REL_TOL, atol=ABS_TOL), (
                f"norm_mode={norm_mode} layer {l} W{idx}: analytic={ana:.8f} numeric={num:.8f} rel_err={rel:.2e}"
            )
            worst = max(worst, rel)
            checked += 1
            taken += 1
            if taken >= 5:
                break

    print(f"norm_mode={norm_mode}: {checked} W entries checked ({skipped} skipped as kink-adjacent), "
          f"worst relative error = {worst:.2e}")


if __name__ == "__main__":
    check("none", seed=1)
    check("batchnorm", seed=1)
    check("layernorm", seed=1)
    print("mlp gradcheck passed for all three norm_mode values")
