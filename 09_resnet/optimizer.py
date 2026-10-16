"""
SGD with momentum and weight decay, duplicated from 08_vggnet/optimizer.py
(self-contained-folder convention). Section 4.2's own CIFAR-10 values:
momentum 0.9, weight decay 0.0001 -- note this is 5e-4 in 07_alexnet/
08_vggnet (their ImageNet-recipe value); ResNet's CIFAR-10 recipe uses a
smaller 1e-4, passed in explicitly by run_experiment.py.
"""
import numpy as np


class SGDMomentum:
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1e-4):
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
