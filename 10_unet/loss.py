"""
Pixel-wise softmax + weighted cross-entropy, Section 3 Eqn.1:
E = sum_x w(x) * log(p_l(x)(x)), softmax p_k(x) = exp(a_k(x)) / sum_k' exp(a_k'(x)).
The paper states E as an energy to be penalized when p deviates from 1 at
the true label; the actual SGD loss minimized is L = -E (standard
weighted cross-entropy, log(p) <= 0 so -E >= 0). Softmax computed with
the usual max-subtraction for numerical stability -- not in the paper,
every practical softmax implementation needs it, same as every prior
classification paper in this repo.

Normalization: the paper doesn't state whether E is summed or averaged
over pixels. This repo divides by the pixel count (standard practice,
keeps the gradient magnitude comparable across different output
resolutions/batch sizes) -- a documented choice, not a silent one.
"""
import numpy as np


class WeightedPixelSoftmaxCE:
    def forward(self, logits, labels, weights):
        # logits: (N, K, H, W), labels: (N, H, W) int in [0,K), weights: (N, H, W)
        N, K, H, W = logits.shape
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        self.probs, self.labels, self.weights = probs, labels, weights
        self.N, self.H, self.W = N, H, W
        n_idx, h_idx, w_idx = np.meshgrid(np.arange(N), np.arange(H), np.arange(W), indexing='ij')
        p_true = probs[n_idx, labels, h_idx, w_idx]
        eps = 1e-12
        loss = -(weights * np.log(p_true + eps)).sum() / (N * H * W)
        return loss

    def backward(self):
        N, H, W = self.N, self.H, self.W
        K = self.probs.shape[1]
        one_hot = np.zeros_like(self.probs)
        n_idx, h_idx, w_idx = np.meshgrid(np.arange(N), np.arange(H), np.arange(W), indexing='ij')
        one_hot[n_idx, self.labels, h_idx, w_idx] = 1.0
        dlogits = self.weights[:, None, :, :] * (self.probs - one_hot) / (N * H * W)
        return dlogits
