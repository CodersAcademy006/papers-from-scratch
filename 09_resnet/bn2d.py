"""
Spatial Batch Normalization for conv feature maps (N,C,H,W), normalizing
per-channel over the N,H,W axes. Same Ioffe & Szegedy math as
BatchNorm1d (bn_layer.py) -- just reshape (N,C,H,W) -> (N*H*W, C) so each
channel is a "feature" and every pixel across the batch is a "sample",
then delegate to BatchNorm1d's forward/backward unchanged.
"""
import numpy as np
from bn_layer import BatchNorm1d


class BatchNorm2d(BatchNorm1d):
    @staticmethod
    def _to_rows(x):
        N, C, H, W = x.shape
        return x.transpose(0, 2, 3, 1).reshape(-1, C), (N, H, W, C)

    @staticmethod
    def _to_nchw(rows, shape):
        N, H, W, C = shape
        return rows.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    def forward_train(self, x):
        rows, shape = self._to_rows(x)
        y, cache = super().forward_train(rows)
        return self._to_nchw(y, shape), cache

    def forward_eval(self, x):
        rows, shape = self._to_rows(x)
        y = super().forward_eval(rows)
        return self._to_nchw(y, shape)

    def backward(self, dy, cache):
        rows, shape = self._to_rows(dy)
        dx, dgamma, dbeta = super().backward(rows, cache)
        return self._to_nchw(dx, shape), dgamma, dbeta
