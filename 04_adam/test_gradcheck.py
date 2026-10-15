"""Numerical gradient check for ReLUMLP.backward(), same discipline as every
other paper in this repo: never trust a training curve before this passes.

First run hit a genuine ReLU-kink collision (see JOURNAL.md entry 6 for the
full diagnosis): with this network's small weight_std, example 8's entire
first hidden layer was EXACTLY zero (every unit inactive for that example),
which forces the second layer's pre-activation to be exactly 0 too,
regardless of the second layer's weights -- a structural zero, not a
coincidence of random seed, so re-seeding does not fix it. Perturbing a
parameter that touches a pre-activation sitting exactly at the kink makes
the finite-difference estimate see a different active/inactive ReLU state
than the analytic gradient's fixed one-sided convention (z>0), producing a
real discrepancy that is not a bug in backward(). Units whose pre-
activation is within a safety margin of zero for ANY example are now
excluded from the sample pool below, rather than papered over with a looser
tolerance."""
import numpy as np
from mlp_relu import ReLUMLP

EPS = 1e-5
REL_TOL = 1e-4
ABS_TOL = 1e-6
KINK_MARGIN = 1e-3  # exclude units whose pre-activation is this close to 0 for any example


def main():
    rng = np.random.default_rng(5)
    n, d_in, d_out = 10, 8, 4
    X = rng.normal(size=(n, d_in))
    labels = rng.integers(0, d_out, size=n)
    D = np.zeros((n, d_out))
    D[np.arange(n), labels] = 1.0

    net = ReLUMLP([d_in, 6, 5, d_out], l2=0.01, seed=5)
    activations, pre_act = net.forward(X)
    grads = net.backward(activations, pre_act, D)
    params = net.params()

    # per-layer set of output-unit indices that are safely away from the
    # ReLU kink for every example in this batch (output/softmax layer has
    # no kink, so it's never filtered)
    safe_units = []
    for l in range(net.n_layers):
        if l < net.n_layers - 1:
            min_abs_per_unit = np.min(np.abs(pre_act[l]), axis=0)
            safe_units.append(set(np.where(min_abs_per_unit > KINK_MARGIN)[0]))
        else:
            safe_units.append(set(range(pre_act[l].shape[1])))

    def loss_at(p_idx, value_at_index, idx):
        saved = params[p_idx][idx]
        params[p_idx][idx] = value_at_index
        a2, _ = net.forward(X)
        loss = net.loss(a2[-1], D)
        params[p_idx][idx] = saved
        return loss

    worst = 0.0
    checked = 0
    skipped = 0
    for p_idx, p in enumerate(params):
        l = p_idx if p_idx < net.n_layers else p_idx - net.n_layers  # W[0..n-1] then b[0..n-1]
        is_bias = p_idx >= net.n_layers
        flat = list(np.ndindex(p.shape))
        candidates = rng.permutation(len(flat))
        taken = 0
        for k in candidates:
            idx = flat[k]
            unit = idx[0] if is_bias else idx[0]  # row of W, or the single bias index -- both index the OUTPUT unit
            if unit not in safe_units[l]:
                skipped += 1
                continue
            orig = p[idx]
            num = (loss_at(p_idx, orig + EPS, idx) - loss_at(p_idx, orig - EPS, idx)) / (2 * EPS)
            ana = grads[p_idx][idx]
            rel = abs(num - ana) / max(abs(num), abs(ana), ABS_TOL)
            assert np.isclose(num, ana, rtol=REL_TOL, atol=ABS_TOL), (
                f"param {p_idx} idx {idx}: analytic={ana:.8f} numeric={num:.8f} rel_err={rel:.2e}"
            )
            worst = max(worst, rel)
            checked += 1
            taken += 1
            if taken >= 5:
                break

    print(f"gradcheck passed, {checked} params sampled ({skipped} skipped as kink-adjacent), "
          f"worst relative error = {worst:.2e}")


if __name__ == "__main__":
    main()
