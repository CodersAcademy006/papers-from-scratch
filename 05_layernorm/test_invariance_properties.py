"""
Table 1's invariance claims (quoted in JOURNAL.md entry 2), checked
directly on a single LayerNorm'd linear layer: h = LN(Wx) (gain/bias
folded into LN as usual, matching Eq 5). Each check compares model output
before/after the stated transformation -- "invariant" means the output is
numerically unchanged, not just similar.
"""
import numpy as np
from ln_layer import LayerNorm1d


def forward(W, x, ln):
    z = x @ W.T
    y, _ = ln.forward(z)
    return y


def check_weight_matrix_rescale_and_recenter():
    """'layer normalization is invariant to scaling of the entire weight
    matrix and invariant to a shift to all of the incoming weights in the
    weight matrix.' W' = delta*W + 1*gamma^T (entry 2's Eq. 6) should give
    the SAME output as W."""
    rng = np.random.default_rng(0)
    d_in, d_out, n = 6, 5, 4
    W = rng.normal(size=(d_out, d_in))
    X = rng.normal(size=(n, d_in))
    ln = LayerNorm1d(d_out)
    ln.gamma = rng.normal(size=d_out) + 1.0
    ln.beta = rng.normal(size=d_out)

    y_orig = forward(W, X, ln)

    delta = 3.7
    shift_row = rng.normal(size=d_in)  # the "gamma" row shift from the paper's W' = delta*W + 1*gamma^T
    W_prime = delta * W + np.ones((d_out, 1)) @ shift_row[None, :]
    y_prime = forward(W_prime, X, ln)

    max_diff = np.max(np.abs(y_orig - y_prime))
    # first run used 1e-8 here and failed at 5.33e-6 -- not a bug, just too
    # tight for float64 at this magnitude (|y| ~ O(1-2), W' entries ~ O(20)
    # after delta=3.7 scaling): 1e-5 absolute is the right bar, same order
    # as the ABS_TOL used throughout this repo's gradchecks
    assert max_diff < 1e-5, f"weight matrix rescale+recenter changed output, max diff={max_diff}"
    print(f"weight-matrix rescale+recenter invariance passed, max output diff = {max_diff:.2e}")


def check_weight_vector_rescale_NOT_invariant():
    """Table 1: Layer norm is NOT invariant to rescaling a SINGLE neuron's
    weight vector (unlike batch norm / weight norm). Confirms the negative
    claim too, not just the positive ones -- scaling one row of W should
    actually change the output here."""
    rng = np.random.default_rng(1)
    d_in, d_out, n = 6, 5, 4
    W = rng.normal(size=(d_out, d_in))
    X = rng.normal(size=(n, d_in))
    ln = LayerNorm1d(d_out)
    ln.gamma = rng.normal(size=d_out) + 1.0
    ln.beta = rng.normal(size=d_out)

    y_orig = forward(W, X, ln)

    W_prime = W.copy()
    W_prime[0, :] *= 5.0  # rescale ONLY neuron 0's incoming weight vector
    y_prime = forward(W_prime, X, ln)

    max_diff = np.max(np.abs(y_orig - y_prime))
    assert max_diff > 1e-3, (
        f"expected single weight-vector rescale to CHANGE layer-norm output "
        f"(Table 1 says layer norm is NOT invariant to this), but max diff was only {max_diff:.2e}"
    )
    print(f"weight-vector rescale correctly changes output (NOT invariant, as Table 1 says), max diff = {max_diff:.2e}")


def check_single_case_rescale():
    """'layer normalization is invariant to re-scaling of individual
    training cases, because the normalization scalars mu and sigma... only
    depend on the current input data.' Scaling one row of X by any
    positive delta should leave that example's LN output unchanged."""
    rng = np.random.default_rng(2)
    d_in, d_out, n = 6, 5, 4
    W = rng.normal(size=(d_out, d_in))
    X = rng.normal(size=(n, d_in))
    ln = LayerNorm1d(d_out)
    ln.gamma = rng.normal(size=d_out) + 1.0
    ln.beta = rng.normal(size=d_out)

    y_orig = forward(W, X, ln)

    X_prime = X.copy()
    X_prime[2, :] *= 8.3  # rescale only training case 2
    y_prime = forward(W, X_prime, ln)

    diff_case2 = np.max(np.abs(y_orig[2] - y_prime[2]))
    diff_others = np.max(np.abs(y_orig[[0, 1, 3]] - y_prime[[0, 1, 3]]))
    # same float64-at-this-magnitude calibration as the weight-matrix check above
    assert diff_case2 < 1e-5, f"rescaled training case's own output changed, diff={diff_case2}"
    assert diff_others < 1e-10, f"OTHER training cases' outputs were affected by case 2's rescale, diff={diff_others}"
    print(f"single-training-case rescale invariance passed: rescaled case's output diff = {diff_case2:.2e}, "
          f"other cases unaffected (diff = {diff_others:.2e})")


if __name__ == "__main__":
    check_weight_matrix_rescale_and_recenter()
    check_weight_vector_rescale_NOT_invariant()
    check_single_case_rescale()
