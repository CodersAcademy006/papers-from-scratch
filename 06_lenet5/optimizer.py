"""Adam, duplicated from 04_adam/optimizer.py (self-contained-folder
convention) -- used here as the training algorithm in place of the
paper's own Stochastic Diagonal Levenberg-Marquardt method, see
JOURNAL.md entry 3."""
import numpy as np


class Adam:
    def __init__(self, params, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(p) for k, p in params.items()}
        self.v = {k: np.zeros_like(p) for k, p in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)
