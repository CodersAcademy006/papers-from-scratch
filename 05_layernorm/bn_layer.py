"""
Batch Normalization, Ioffe & Szegedy, arXiv:1502.03167, Algorithm 1 + the
backward derivation in Section 3. Duplicated from 03_batchnorm/bn_layer.py
(self-contained-folder convention, see 02_dropout/JOURNAL.md entry 2 for
the original reasoning) -- see that paper's JOURNAL.md for the full quoted
derivation this implements and its own EMA-vs-plain-average history
(05_layernorm's own entry 2 covers LayerNorm specifically, not this file).

Backward is implemented term-by-term exactly as the paper writes it, not the
algebraically-collapsed single-line form common in other from-scratch
write-ups -- deliberately, so each line maps to one of the paper's quoted
equations and can be checked against the paper directly.
"""
import numpy as np


class BatchNorm1d:
    def __init__(self, n_features, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        # Entry 7 (JOURNAL.md): this was a literal (arithmetic) running
        # average in the first version, per the paper's vague "average
        # over them" phrasing (quoted in entry 2) -- found empirically to
        # badly hurt test-time accuracy (weighs noisy pre-convergence batch
        # stats equally with the final ones), via a direct side-by-side
        # diagnostic (attempts/investigate_inference_stats.py). Switched to
        # an exponential moving average, the convention that became
        # standard for exactly this reason.
        self.running_mean = np.zeros(n_features)
        self.running_var = np.zeros(n_features)
        self.n_batches_seen = 0

    def _batch_normalize(self, x):
        """The pure Algorithm 1 math, no side effects. Shared by
        forward_train (which also updates running stats, below) and
        forward_probe (which doesn't -- see its docstring for why that
        distinction matters)."""
        m = x.shape[0]
        mu_B = x.mean(axis=0)
        sigma_B_sq = x.var(axis=0)  # biased, i.e. divide by m, matching Algorithm 1
        std_inv = 1.0 / np.sqrt(sigma_B_sq + self.eps)
        x_hat = (x - mu_B) * std_inv
        y = self.gamma * x_hat + self.beta
        cache = (x, x_hat, mu_B, sigma_B_sq, std_inv, m)
        return y, cache

    def forward_train(self, x):
        y, cache = self._batch_normalize(x)
        _, _, mu_B, sigma_B_sq, _, m = cache

        # unbiased variance estimate, per Algorithm 2: "Var[x] = m/(m-1) *
        # E_B[sigma_B^2]"; folded into an EXPONENTIAL moving average rather
        # than a plain one -- see entry 7, this is a corrected deviation
        # from a literal-average first attempt that measurably hurt
        # test-time accuracy.
        unbiased_var = sigma_B_sq * m / (m - 1) if m > 1 else sigma_B_sq
        if self.n_batches_seen == 0:
            self.running_mean = mu_B
            self.running_var = unbiased_var
        else:
            mom = self.momentum
            self.running_mean = mom * self.running_mean + (1 - mom) * mu_B
            self.running_var = mom * self.running_var + (1 - mom) * unbiased_var
        self.n_batches_seen += 1

        return y, cache

    def forward_probe(self, x):
        """Same batch-normalizing math as forward_train, but does NOT touch
        running_mean/running_var/n_batches_seen. Exists for instrumentation:
        run_experiment.py samples a fixed snapshot of examples periodically
        to chart how an activation's distribution evolves (reproducing
        Fig. 1b/c). If that used forward_train, every snapshot probe would
        silently feed into the inference-time running average alongside the
        real training mini-batches -- corrupting forward_eval for a reason
        that has nothing to do with training."""
        y, _ = self._batch_normalize(x)
        return y

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
