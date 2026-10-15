"""
FIRST ATTEMPT, KEPT DELIBERATELY BROKEN. Do not import this into the real
implementation. See JOURNAL.md entry 4 for the full story.

This is the "obvious" backward pass for Algorithm 1 if you only look at the
forward formula and differentiate x_hat = (x - mu_B) / sqrt(sigma_B^2 + eps)
as if mu_B and sigma_B^2 were just numbers, not functions of every x_i in
the batch (including this one). It's exactly the mistake the paper's own
Section 2 describes for a simpler case ("if a gradient descent step ignores
the dependence of E[x] on b... the model blows up when the normalization
parameters are computed outside the gradient descent step").

Treating mu_B, sigma_B^2 as constants w.r.t. x gives:

    dL/dx_i = dL/dx_hat_i * 1/sqrt(sigma_B^2 + eps)

i.e. just scale the upstream gradient by 1/std, same as if BN were a fixed
per-batch affine transform instead of a function OF the batch. This drops
the dL/dmu_B and dL/dsigma_B^2 terms entirely.
"""
import numpy as np


def naive_bn_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * std_inv
    y = gamma * x_hat + beta
    cache = (x, x_hat, mu, var, std_inv, gamma)
    return y, cache


def naive_bn_backward(dy, cache):
    x, x_hat, mu, var, std_inv, gamma = cache
    dgamma = np.sum(dy * x_hat, axis=0)
    dbeta = np.sum(dy, axis=0)
    dx_hat = dy * gamma
    # THE BUG: pretends mu_B/sigma_B^2 don't depend on x. See JOURNAL.md entry 4
    # for the gradcheck failure this produces and the fix.
    dx = dx_hat * std_inv
    return dx, dgamma, dbeta
