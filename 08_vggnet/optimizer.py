"""
SGD with momentum and weight decay, duplicated from 07_alexnet/optimizer.py
(self-contained-folder convention). VGG's Section 3.1 training recipe
("multinomial logistic regression... mini-batch gradient descent with
momentum... batch size 256, momentum 0.9... regularised by weight decay
(L2 penalty multiplier 5*10^-4)") is the same SGD+momentum+decay scheme
as AlexNet's, just with different hyperparameter values passed in.
"""
import numpy as np


class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=5e-4):
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
