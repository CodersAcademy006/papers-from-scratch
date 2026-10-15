"""
Batch Normalization, Ioffe & Szegedy, arXiv:1502.03167, Algorithm 1 + the
backward derivation in Section 3 (quoted in full in JOURNAL.md entry 2).

Backward is implemented term-by-term exactly as the paper writes it, not the
algebraically-collapsed single-line form common in other from-scratch
write-ups -- deliberately, so each line here maps to one quoted equation.
This is the corrected version; see attempts/naive_batchnorm_backward.py and
JOURNAL.md entry 4 for the broken first attempt and why it's wrong.
"""
import numpy as np


class BatchNorm1d:
    def __init__(self, n_features, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        # Entry 2: paper says "average over" multiple mini-batches for
        # inference stats -- literal running (not exponential) average.
        self.running_mean = np.zeros(n_features)
        self.running_var = np.zeros(n_features)
        self.n_batches_seen = 0

    def forward_train(self, x):
        m = x.shape[0]
        mu_B = x.mean(axis=0)
        sigma_B_sq = x.var(axis=0)  # biased, i.e. divide by m, matching Algorithm 1
        std_inv = 1.0 / np.sqrt(sigma_B_sq + self.eps)
        x_hat = (x - mu_B) * std_inv
        y = self.gamma * x_hat + self.beta

        # unbiased variance estimate for the running average, per Algorithm 2:
        # "Var[x] = m/(m-1) * E_B[sigma_B^2]"
        unbiased_var = sigma_B_sq * m / (m - 1) if m > 1 else sigma_B_sq
        n = self.n_batches_seen
        self.running_mean = (self.running_mean * n + mu_B) / (n + 1)
        self.running_var = (self.running_var * n + unbiased_var) / (n + 1)
        self.n_batches_seen += 1

        cache = (x, x_hat, mu_B, sigma_B_sq, std_inv, m)
        return y, cache

    def forward_eval(self, x):
        x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta

    def backward(self, dy, cache):
        """Every line below is one quoted equation from JOURNAL.md entry 2,
        in the same order the paper presents them. Nothing simplified."""
        x, x_hat, mu_B, sigma_B_sq, std_inv, m = cache

        dgamma = np.sum(dy * x_hat, axis=0)                      # dL/dgamma
        dbeta = np.sum(dy, axis=0)                                # dL/dbeta

        dx_hat = dy * self.gamma                                  # dL/dx_hat_i

        dsigma_B_sq = np.sum(
            dx_hat * (x - mu_B) * (-0.5) * (sigma_B_sq + self.eps) ** (-1.5),
            axis=0,
        )                                                          # dL/dsigma_B^2

        dmu_B = (
            np.sum(dx_hat * (-std_inv), axis=0)
            + dsigma_B_sq * np.sum(-2.0 * (x - mu_B), axis=0) / m
        )                                                          # dL/dmu_B

        dx = (
            dx_hat * std_inv
            + dsigma_B_sq * 2.0 * (x - mu_B) / m
            + dmu_B / m
        )                                                          # dL/dx_i

        return dx, dgamma, dbeta
