"""
SGD with momentum and weight decay, Section 5's own update rule, used as
specified (not substituted with Adam this time, JOURNAL.md entry 3):

  v_{i+1} := 0.9 * v_i - 0.0005 * eps * w_i - eps * <dL/dw>|_{w_i}
  w_{i+1} := w_i + v_{i+1}

Weight decay is baked into the velocity term, scaled by the learning
rate, not a separate L2 penalty added to the loss.
"""
import numpy as np


class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0005):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = {k: np.zeros_like(p) for k, p in params.items()}

    def step(self, params, grads):
        for k in params:
            self.v[k] = (self.momentum * self.v[k]
                         - self.weight_decay * self.lr * params[k]
                         - self.lr * grads[k])
            params[k] += self.v[k]
