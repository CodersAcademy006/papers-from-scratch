"""
Dropout, duplicated in spirit from 02_dropout (self-contained-folder
convention) -- AlexNet's own description (Section 4.2) is literally the
original "scale at test time" convention 02_dropout implements, not the
inverted-dropout convention that later became standard:

  "The neurons which are dropped out... do not contribute to the forward
  pass and do not participate in back-propagation... At test time, we use
  all the neurons but multiply their outputs by 0.5."
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
        return x * self.keep_prob  # test-time scaling, Section 4.2

    def backward(self, dout):
        return dout * self.mask if self.train else dout * self.keep_prob
