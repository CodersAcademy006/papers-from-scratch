"""
Dropout, duplicated from 07_alexnet/dropout.py (self-contained-folder
convention). VGG's Section 3.1 just says "dropout regularisation for
the first two fully-connected layers (dropout ratio set to 0.5)" with
no further detail on the scaling convention, so this repo keeps the
same original "scale at test time" convention used throughout
(02_dropout, 07_alexnet) rather than inverted dropout.
"""
import numpy as np


class Dropout:
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob

    def forward(self, x, train, rng):
        self.train = train
        if train:
            self.mask = (rng.random(x.shape) < self.keep_prob).astype(x.dtype)
            return x * self.mask
        return x * self.keep_prob

    def backward(self, dout):
        return dout * self.mask if self.train else dout * self.keep_prob
