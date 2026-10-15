"""
Max-pooling, duplicated from 07_alexnet/maxpool.py (self-contained-
folder convention). VGG uses non-overlapping 2x2/stride-2 windows
(Section 2.1), unlike AlexNet's overlapping 3x3/stride-2 -- same code,
different (z, s) at construction.
"""
import numpy as np


class MaxPool:
    def __init__(self, z=2, s=2):
        self.z = z
        self.s = s

    def forward(self, x):
        N, C, H, W = x.shape
        z, s = self.z, self.s
        Ho = (H - z) // s + 1
        Wo = (W - z) // s + 1
        self.x_shape = x.shape
        out = np.full((N, C, Ho, Wo), -np.inf)
        self.argmax = np.zeros((N, C, Ho, Wo, 2), dtype=np.int64)
        for i in range(z):
            for j in range(z):
                window = x[:, :, i:i + Ho * s:s, j:j + Wo * s:s]
                better = window > out
                out = np.where(better, window, out)
                self.argmax[..., 0] = np.where(better, i, self.argmax[..., 0])
                self.argmax[..., 1] = np.where(better, j, self.argmax[..., 1])
        self.out = out
        return out

    def backward(self, dout):
        N, C, Ho, Wo = dout.shape
        z, s = self.z, self.s
        dx = np.zeros(self.x_shape)
        for i in range(z):
            for j in range(z):
                mask = (self.argmax[..., 0] == i) & (self.argmax[..., 1] == j)
                contrib = dout * mask
                dx[:, :, i:i + Ho * s:s, j:j + Wo * s:s] += contrib
        return dx
