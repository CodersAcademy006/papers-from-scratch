"""
Three checks against claims quoted in JOURNAL.md entry 2, none needing a
real dataset -- they're claims about the optimizer's update rule in the
abstract, not about any particular network's training curve.
"""
import numpy as np
from optimizer import Adam, AdaGrad


def check_converges_on_quadratic():
    """Sanity: minimize f(theta) = 0.5 * ||theta - target||^2 with Adam.
    Not a claim from the paper, just confirms the implementation isn't
    broken before trusting the two claims below."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=10)
    theta = [np.zeros(10)]
    opt = Adam(theta, alpha=0.05)
    for _ in range(2000):
        grad = [theta[0] - target]
        opt.step(theta, grad)
    err = np.max(np.abs(theta[0] - target))
    assert err < 1e-3, f"Adam failed to converge on a convex quadratic, max err={err}"
    print(f"quadratic convergence check passed, max abs error = {err:.2e}")


def check_rescaling_invariance():
    """Section 2.1: 'rescaling the gradients g with factor c will scale
    m_hat with a factor c and v_hat with a factor c^2, which cancel out' --
    i.e. the UPDATE (not the raw gradient) should be ~unchanged if every
    gradient in the sequence is scaled by the same constant c. Checked over
    a real sequence of steps, not just one, since m_t/v_t accumulate."""
    rng = np.random.default_rng(1)
    grads_seq = [rng.normal(size=5) for _ in range(50)]

    def run(c):
        theta = [np.zeros(5)]
        opt = Adam(theta, alpha=0.01)
        updates = []
        for g in grads_seq:
            before = theta[0].copy()
            opt.step(theta, [c * g])
            updates.append(theta[0] - before)
        return np.array(updates)

    updates_1x = run(1.0)
    updates_100x = run(100.0)
    # not exactly equal (eps breaks exact invariance, and m/v ratios differ
    # slightly at small t before bias correction fully kicks in) -- the
    # paper's claim is approximate ("the effective magnitude... are
    # APPROXIMATELY bounded"), so check closeness, not exact equality
    max_diff = np.max(np.abs(updates_1x - updates_100x))
    mean_update_size = np.mean(np.abs(updates_1x))
    assert max_diff < 0.05 * max(mean_update_size, 1e-6) * 10, (
        f"updates changed too much under 100x gradient rescaling: "
        f"max_diff={max_diff:.4f} vs typical update size={mean_update_size:.4f}"
    )
    print(f"rescaling invariance check passed: max update difference between "
          f"1x and 100x gradient scale = {max_diff:.4f} (typical update size {mean_update_size:.4f})")


def check_adagrad_correspondence():
    """Section 5: 'AdaGrad corresponds to a version of Adam with beta1=0,
    infinitesimal (1-beta2) and a replacement of alpha by an annealed
    version alpha_t = alpha * t^(-1/2).' Run both on the identical synthetic
    gradient sequence and compare the resulting parameter trajectories."""
    rng = np.random.default_rng(2)
    grads_seq = [rng.normal(size=8) for _ in range(200)]
    alpha = 0.1
    beta2 = 1 - 1e-8  # "infinitesimal (1-beta2)"

    theta_adagrad = [np.zeros(8)]
    adagrad = AdaGrad(theta_adagrad, alpha=alpha, eps=0.0)

    theta_adam = [np.zeros(8)]
    adam = Adam(theta_adam, alpha=alpha, beta1=0.0, beta2=beta2, eps=0.0)

    worst_rel = 0.0
    for t, g in enumerate(grads_seq, start=1):
        adagrad.step(theta_adagrad, [g.copy()])
        # Adam's alpha is constant in Algorithm 1; the correspondence
        # requires replacing it with alpha_t = alpha / sqrt(t) manually,
        # exactly as the paper states, before taking this step.
        adam.alpha = alpha / np.sqrt(t)
        adam.step(theta_adam, [g.copy()])

        diff = np.max(np.abs(theta_adagrad[0] - theta_adam[0]))
        scale = max(np.max(np.abs(theta_adagrad[0])), 1e-8)
        worst_rel = max(worst_rel, diff / scale)

    assert worst_rel < 0.05, f"AdaGrad/Adam-limit trajectories diverged, worst relative diff = {worst_rel:.4f}"
    print(f"AdaGrad correspondence check passed: worst relative trajectory "
          f"difference over {len(grads_seq)} steps = {worst_rel:.4f}")


if __name__ == "__main__":
    check_converges_on_quadratic()
    check_rescaling_invariance()
    check_adagrad_correspondence()
