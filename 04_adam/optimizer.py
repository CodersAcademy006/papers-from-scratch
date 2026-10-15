"""
Adam (Algorithm 1) plus two baselines used for comparison in run_experiment.py:
momentum-SGD (classical, not the paper's subject but its Section 6.2
baseline) and AdaGrad (used both as a baseline and, in test_properties.py,
to check the correspondence claim in Section 5).

Each optimizer takes a flat list of parameter arrays and mutates them in
place given the matching list of gradient arrays -- decoupled from any
specific network so the same classes drive both the synthetic-function
sanity checks and the real MLP in mlp_relu.py.
"""
import numpy as np


class Adam:
    """Algorithm 1, verbatim. See JOURNAL.md entry 2 for the quoted
    pseudocode this mirrors line for line."""

    def __init__(self, params, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                 bias_correction=True):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.bias_correction = bias_correction  # off only for the entry-4 ablation
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            if self.bias_correction:
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            else:
                # entry 4: paper says dropping this "results in a version
                # of RMSProp with momentum" -- exactly this code path.
                m_hat = self.m[i]
                v_hat = self.v[i]
            p -= self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)


class MomentumSGD:
    """Classical momentum, the paper's own 6.2 baseline ("SGDNesterov" in
    their figures). Plain momentum here, not Nesterov's lookahead variant --
    documented simplification, see journal entry 5."""

    def __init__(self, params, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            self.v[i] = self.momentum * self.v[i] - self.lr * g
            p += self.v[i]


class AdaGrad:
    """Duchi et al 2011, as quoted in the paper's Section 5:
    theta_{t+1} = theta_t - alpha * g_t / sqrt(sum_{i=1}^t g_i^2)."""

    def __init__(self, params, alpha=0.01, eps=1e-8):
        self.alpha = alpha
        self.eps = eps
        self.sum_sq = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            self.sum_sq[i] += g * g
            p -= self.alpha * g / (np.sqrt(self.sum_sq[i]) + self.eps)
