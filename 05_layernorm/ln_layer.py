"""
Layer Normalization, Ba/Kiros/Hinton, arXiv:1607.06450, Eq. (3)+(5)
(quoted in full in JOURNAL.md entry 2).

Structurally a close cousin of 03_batchnorm/bn_layer.py -- same affine
gain/bias restore pattern, same backward-pass shape -- with the single
change entry 2 identifies: normalize over the FEATURE axis (per example),
not the BATCH axis (per unit). That one change also means this class needs
no running statistics and no forward_train/forward_eval split: the paper's
own point is that LN computes "exactly the same computation at training
and test times," so there's only one forward() method.
"""
import numpy as np


class LayerNorm1d:
    def __init__(self, n_features, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)

    def forward(self, x):
        """x: (batch, n_features). mu/sigma computed per-example (axis=1),
        matching Eq. (3)'s per-training-case statistics -- the BatchNorm
        counterpart computes these per-unit (axis=0) instead."""
        mu = x.mean(axis=1, keepdims=True)
        sigma_sq = x.var(axis=1, keepdims=True)  # biased, i.e. divide by H, matching Eq. (3)
        std_inv = 1.0 / np.sqrt(sigma_sq + self.eps)
        x_hat = (x - mu) * std_inv
        y = self.gamma * x_hat + self.beta
        cache = (x_hat, std_inv, x.shape[1])
        return y, cache

    def backward(self, dy, cache):
        """Same chain-rule shape as 03_batchnorm's BatchNorm1d.backward(),
        with the sum/mean now taken over axis=1 (features) instead of
        axis=0 (batch) -- see JOURNAL.md entry 2 for why that's the whole
        difference between the two mechanisms."""
        x_hat, std_inv, H = cache

        dgamma = np.sum(dy * x_hat, axis=0)
        dbeta = np.sum(dy, axis=0)

        dx_hat = dy * self.gamma

        # same five-line derivation as BatchNorm1d.backward(), axis=1 here.
        # (x-mu) = x_hat/std_inv and (sigma_sq+eps)^(-3/2) = std_inv^3, so
        # (x-mu)*(sigma_sq+eps)^(-3/2) = x_hat*std_inv^2 -- NOT x_hat*std_inv
        # (first version here, which failed gradcheck: JOURNAL.md entry 4).
        dsigma_sq = np.sum(
            dx_hat * x_hat * (-0.5) * std_inv ** 2,
            axis=1, keepdims=True,
        )
        dmu = np.sum(dx_hat * (-std_inv), axis=1, keepdims=True) \
            + dsigma_sq * np.sum(-2.0 * x_hat / std_inv, axis=1, keepdims=True) / H
        dx = dx_hat * std_inv \
            + dsigma_sq * 2.0 * (x_hat / std_inv) / H \
            + dmu / H

        return dx, dgamma, dbeta
